#!/usr/bin/env python3
"""Disposable checkpoint and functional-ablation preflight.

This is deliberately independent of the frozen 100K/1M flights.  It uses a
small BraidPrototypeLM, saves a complete model/optimizer/scheduler/RNG
checkpoint, verifies interrupted-resume equivalence, and exercises the
expert-intervention hooks needed by the later checkpoint-only experiment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from model_core_float64 import BraidPrototypeLM, ResearchMoETelemetry


VOCAB_SIZE = 259
PAD_ID = 0
BOS_ID = 1
RAW_BYTE_START = 3
NUM_EXPERTS = 4
NUM_LAYERS = 2
SEQ_LEN = 16
BATCH_SIZE = 6
TOTAL_STEPS = 6
SPLIT_STEP = 3
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-2
MAX_GRAD_NORM = 1.0

MODEL_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "n_layers": NUM_LAYERS,
    "d_model": 32,
    "n_heads": 4,
    "latent_dim_kv": 16,
    "latent_dim_q": 16,
    "pe_dim": 4,
    "num_experts": NUM_EXPERTS,
    "d_ff": 64,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def set_rng_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8).cpu() for item in state["cuda"]]
        )


def make_model(device: torch.device) -> BraidPrototypeLM:
    model = BraidPrototypeLM(**MODEL_CONFIG).to(device)
    model.train()
    return model


def make_optimizer(model: nn.Module) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 - 0.5 * min(step, TOTAL_STEPS) / TOTAL_STEPS,
    )
    return optimizer, scheduler


def make_batches() -> List[Tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(190616)
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for step in range(TOTAL_STEPS):
        x = torch.randint(
            RAW_BYTE_START,
            VOCAB_SIZE,
            (BATCH_SIZE, SEQ_LEN),
            generator=generator,
            dtype=torch.long,
        )
        x[:, 0] = BOS_ID
        x[step % BATCH_SIZE, -1] = PAD_ID
        y = torch.roll(x, shifts=-1, dims=1)
        y[:, -1] = PAD_ID
        batches.append((x, y))
    return batches


def forward_batch(
    model: BraidPrototypeLM,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    telemetry: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    x = x.to(device)
    y = y.to(device)
    seq_idx = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
    seq_idx = seq_idx.expand(x.size(0), -1)
    sequence_ids = torch.arange(x.size(0), device=device)
    if telemetry:
        ResearchMoETelemetry.reset_step_buffer()
    logits = model(
        x,
        seq_idx,
        current_branch="B",
        current_sequence_ids=sequence_ids,
        telemetry_class=ResearchMoETelemetry if telemetry else None,
    )
    ce_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=PAD_ID,
        reduction="sum",
    )
    valid_tokens = (y != PAD_ID).sum().clamp_min(1)
    auxiliary_loss = torch.stack(
        [layer.moe.last_aux_loss for layer in model.layers]
    ).mean()
    loss = (ce_sum + 0.01 * auxiliary_loss * valid_tokens) / valid_tokens
    summary = ResearchMoETelemetry.get_summary("B") if telemetry else {}
    if telemetry:
        ResearchMoETelemetry.reset_step_buffer()
    return logits, loss, summary


def train_one_step(
    model: BraidPrototypeLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss, _ = forward_batch(model, batch[0], batch[1], device, telemetry=False)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    scheduler.step()
    return float(loss.detach().cpu())


def run_steps(
    model: BraidPrototypeLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    start: int,
    end: int,
    device: torch.device,
) -> List[float]:
    return [
        train_one_step(model, optimizer, scheduler, batches[step], device)
        for step in range(start, end)
    ]


def tensor_tree_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensor_tree_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_tree_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tensor_tree_cpu(item) for item in value)
    return value


def max_state_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    maximum = 0.0
    assert left.keys() == right.keys()
    for key in left:
        a = left[key]
        b = right[key]
        if torch.is_tensor(a):
            maximum = max(maximum, float((a.detach().cpu() - b.detach().cpu()).abs().max()))
        elif isinstance(a, dict):
            maximum = max(maximum, max_state_difference(a, b))
        elif isinstance(a, (list, tuple)):
            for x, y in zip(a, b):
                if torch.is_tensor(x):
                    maximum = max(maximum, float((x.detach().cpu() - y.detach().cpu()).abs().max()))
    return maximum


def checkpoint_resume_test(
    device: torch.device,
    output_dir: Path,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    seed = 20261016
    set_rng_seed(seed)
    baseline = make_model(device)
    initial_state = copy.deepcopy(baseline.state_dict())
    baseline_optimizer, baseline_scheduler = make_optimizer(baseline)
    baseline_losses = run_steps(
        baseline, baseline_optimizer, baseline_scheduler, batches, 0, TOTAL_STEPS, device
    )

    interrupted = make_model(device)
    interrupted.load_state_dict(initial_state)
    interrupted_optimizer, interrupted_scheduler = make_optimizer(interrupted)
    resumed_losses = run_steps(
        interrupted,
        interrupted_optimizer,
        interrupted_scheduler,
        batches,
        0,
        SPLIT_STEP,
        device,
    )
    checkpoint = {
        "format": "crucible.checkpoint_preflight.v1",
        "step": SPLIT_STEP,
        "model_config": MODEL_CONFIG,
        "model_state_dict": tensor_tree_cpu(interrupted.state_dict()),
        "optimizer_state_dict": tensor_tree_cpu(interrupted_optimizer.state_dict()),
        "scheduler_state_dict": tensor_tree_cpu(interrupted_scheduler.state_dict()),
        "rng_state": tensor_tree_cpu(capture_rng_state()),
    }
    checkpoint_path = output_dir / "mini_resume_checkpoint.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    checkpoint_sha = sha256_file(checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    assert loaded["format"] == checkpoint["format"]
    assert loaded["step"] == SPLIT_STEP

    resumed = make_model(device)
    resumed.load_state_dict(loaded["model_state_dict"], strict=True)
    resumed_optimizer, resumed_scheduler = make_optimizer(resumed)
    resumed_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    resumed_scheduler.load_state_dict(loaded["scheduler_state_dict"])
    restore_rng_state(loaded["rng_state"])
    resumed_losses.extend(
        run_steps(
            resumed,
            resumed_optimizer,
            resumed_scheduler,
            batches,
            SPLIT_STEP,
            TOTAL_STEPS,
            device,
        )
    )

    model_delta = max_state_difference(baseline.state_dict(), resumed.state_dict())
    optimizer_delta = max_state_difference(
        baseline_optimizer.state_dict(), resumed_optimizer.state_dict()
    )
    scheduler_delta = max_state_difference(
        baseline_scheduler.state_dict(), resumed_scheduler.state_dict()
    )
    loss_delta = max(abs(a - b) for a, b in zip(baseline_losses, resumed_losses))
    tolerance = 2e-5 if device.type == "cuda" else 1e-7
    assert model_delta <= tolerance, (model_delta, tolerance)
    assert optimizer_delta <= tolerance, (optimizer_delta, tolerance)
    assert scheduler_delta <= tolerance, (scheduler_delta, tolerance)
    assert loss_delta <= tolerance, (loss_delta, tolerance)

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": SPLIT_STEP,
        "checkpoint_fields": sorted(checkpoint),
        "baseline_losses": baseline_losses,
        "resumed_losses": resumed_losses,
        "max_model_state_abs_delta": model_delta,
        "max_optimizer_state_abs_delta": optimizer_delta,
        "max_scheduler_state_abs_delta": scheduler_delta,
        "max_loss_abs_delta": loss_delta,
        "declared_resume_tolerance": tolerance,
        "status": "PASS",
        "model": resumed,
    }


def make_ablation_eval_batch() -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    categories = {
        "whitespace": [9, 10, 13, 32],
        "lowercase": list(range(97, 123)),
        "uppercase": list(range(65, 91)),
        "digit": list(range(48, 58)),
        "punctuation": [33, 34, 39, 44, 46, 58, 59, 63],
        "other": [0x80, 0x90, 0xA0, 0xB0],
    }
    names = list(categories)
    rows = []
    targets = []
    for name in names:
        raw = categories[name]
        sequence = torch.tensor(
            [RAW_BYTE_START + raw[index % len(raw)] for index in range(SEQ_LEN)],
            dtype=torch.long,
        )
        x = torch.empty(SEQ_LEN, dtype=torch.long)
        x[0] = BOS_ID
        x[1:] = sequence[:-1]
        rows.append(x)
        targets.append(sequence)
    return torch.stack(rows), torch.stack(targets), names


def nll_by_category(
    logits: torch.Tensor,
    targets: torch.Tensor,
    names: List[str],
) -> Dict[str, Any]:
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_ID,
        reduction="none",
    ).reshape_as(targets)
    result: Dict[str, Any] = {}
    for index, name in enumerate(names):
        values = per_token[index]
        result[name] = {
            "token_count": int(values.numel()),
            "nll": float(values.mean().detach().cpu()),
        }
    result["all"] = {
        "token_count": int(per_token.numel()),
        "nll": float(per_token.mean().detach().cpu()),
    }
    return result


def route_counts(
    model: BraidPrototypeLM,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> Dict[str, List[int]]:
    model.eval()
    with torch.no_grad():
        _, _, summary = forward_batch(model, x, y, device, telemetry=True)
    return {
        layer: list(summary[layer]["specialization"]["top2_token_count"])
        for layer in summary
    }


def evaluate_intervention(
    model: BraidPrototypeLM,
    x: torch.Tensor,
    y: torch.Tensor,
    names: List[str],
    device: torch.device,
    layer_index: int | None = None,
    expert_index: int | None = None,
    mode: str = "intact",
    donor_index: int | None = None,
) -> Dict[str, Any]:
    hooks = []
    if mode != "intact":
        assert layer_index is not None and expert_index is not None
        target = model.layers[layer_index].moe.experts[expert_index]
        if mode in {"zero", "matched_zero", "random_zero"}:
            def zero_hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(output)
            hooks.append(target.register_forward_hook(zero_hook))
        elif mode == "substitute":
            assert donor_index is not None and donor_index != expert_index
            donor = model.layers[layer_index].moe.experts[donor_index]
            def substitute_hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...], _output: torch.Tensor) -> torch.Tensor:
                return donor(inputs[0])
            hooks.append(target.register_forward_hook(substitute_hook))
        else:
            raise ValueError(mode)

    model.eval()
    with torch.no_grad():
        logits, _, _ = forward_batch(model, x, y, device, telemetry=False)
    for hook in hooks:
        hook.remove()
    return {
        "mode": mode,
        "layer": layer_index,
        "expert": expert_index,
        "donor": donor_index,
        "nll_by_category": nll_by_category(logits, y.to(device), names),
    }


def ablation_test(model: BraidPrototypeLM, device: torch.device) -> Dict[str, Any]:
    x, y, names = make_ablation_eval_batch()
    counts = route_counts(model, x, y, device)
    intact = evaluate_intervention(model, x, y, names, device)
    conditions = [intact]
    deterministic_random = random.Random(424242)
    for layer_index in range(NUM_LAYERS):
        layer_counts = counts[f"layer_{layer_index}_moe"]
        for expert_index in range(NUM_EXPERTS):
            other = [index for index in range(NUM_EXPERTS) if index != expert_index]
            random_control = deterministic_random.choice(other)
            matched_control = min(
                other,
                key=lambda index: (abs(layer_counts[index] - layer_counts[expert_index]), index),
            )
            substitution_donor = other[0]
            if substitution_donor == expert_index:
                substitution_donor = other[1]
            conditions.extend([
                evaluate_intervention(
                    model, x, y, names, device,
                    layer_index, expert_index, "zero",
                ),
                evaluate_intervention(
                    model, x, y, names, device,
                    layer_index, random_control, "random_zero",
                ),
                evaluate_intervention(
                    model, x, y, names, device,
                    layer_index, matched_control, "matched_zero",
                ),
                evaluate_intervention(
                    model, x, y, names, device,
                    layer_index, expert_index, "substitute", substitution_donor,
                ),
            ])
    baseline = intact["nll_by_category"]
    for condition in conditions:
        deltas = {}
        for category in [*names, "all"]:
            deltas[category] = condition["nll_by_category"][category]["nll"] - baseline[category]["nll"]
        condition["delta_nll_by_category"] = deltas
    assert len(conditions) == 1 + NUM_LAYERS * NUM_EXPERTS * 4
    return {
        "categories": names,
        "routing_top2_counts": counts,
        "conditions": conditions,
        "status": "PASS",
    }


def telemetry_mass_test(
    model: BraidPrototypeLM,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    max_relative_error = 0.0
    layer_results = {}
    with torch.no_grad():
        for x, y in batches[:2]:
            _, _, summary = forward_batch(model, x, y, device, telemetry=True)
            for layer, payload in summary.items():
                affinity = np.asarray(payload["specialization"]["_token_affinity"], dtype=np.float64)
                top1 = float(sum(payload["specialization"]["top1_token_count"]))
                relative_error = abs(float(affinity.sum()) - top1) / top1
                max_relative_error = max(max_relative_error, relative_error)
                layer_results.setdefault(layer, []).append({
                    "affinity_mass": float(affinity.sum()),
                    "top1_count": top1,
                    "relative_error": relative_error,
                    "shape": list(affinity.shape),
                    "finite": bool(np.isfinite(affinity).all()),
                    "nonnegative": bool((affinity >= 0).all()),
                })
    assert max_relative_error <= 1e-7, max_relative_error
    return {
        "max_relative_error": max_relative_error,
        "layers": layer_results,
        "status": "PASS",
    }


def checkpoint_budget() -> Dict[str, Any]:
    parameters = 43151104
    model_bytes = parameters * 4
    adam_moment_bytes = parameters * 4 * 2
    per_checkpoint = model_bytes + adam_moment_bytes
    return {
        "full_model_parameters": parameters,
        "assumed_parameter_dtype": "float32",
        "model_state_bytes_estimate": model_bytes,
        "adam_moment_bytes_estimate": adam_moment_bytes,
        "full_model_optimizer_checkpoint_bytes_estimate": per_checkpoint,
        "nine_checkpoints_four_cells_bytes_estimate": per_checkpoint * 36,
        "nine_checkpoints_four_cells_gib_estimate": per_checkpoint * 36 / (1024 ** 3),
        "model_only_nine_checkpoints_four_cells_gib_estimate": model_bytes * 36 / (1024 ** 3),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    batches = make_batches()
    resume = checkpoint_resume_test(device, output_dir, batches)
    model = resume.pop("model")
    telemetry = telemetry_mass_test(model, batches, device)
    ablation = ablation_test(model, device)
    result = {
        "protocol": "crucible.checkpoint_ablation_preflight.v1",
        "device": str(device),
        "torch_version": torch.__version__,
        "model_config": MODEL_CONFIG,
        "training": {
            "total_steps": TOTAL_STEPS,
            "checkpoint_split_step": SPLIT_STEP,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQ_LEN,
            "branch": "B",
        },
        "checkpoint_resume": resume,
        "float64_telemetry": telemetry,
        "ablation": ablation,
        "checkpoint_budget": checkpoint_budget(),
        "source_hashes": {
            "preflight_script_sha256": sha256_file(Path(__file__).resolve()),
            "float64_model_core_sha256": sha256_file(SCRIPT_DIR / "model_core_float64.py"),
        },
        "status": "PASS",
    }
    results_path = output_dir / "preflight_results.json"
    atomic_json(results_path, result)
    atomic_json(
        output_dir / "preflight_hashes.json",
        {
            "preflight_script_sha256": result["source_hashes"]["preflight_script_sha256"],
            "float64_model_core_sha256": result["source_hashes"]["float64_model_core_sha256"],
            "mini_resume_checkpoint.pt": result["checkpoint_resume"]["checkpoint_sha256"],
            "preflight_results.json": sha256_file(results_path),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
