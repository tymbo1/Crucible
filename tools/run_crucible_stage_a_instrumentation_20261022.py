#!/usr/bin/env python3
"""CPU-only Stage A instrumentation preflight for Crucible cognitive tasks.

This is a bounded learnability and instrumentation check, not a recursive
reasoning experiment.  It trains an E4 Crucible model and a dense active-
compute reference on the corrected v2 manifests, then checks checkpoint,
optimizer/RNG and expert-intervention reproducibility.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "experiments" / "crucible_cognitive_eval_v2_20261021" / "manifests"
E4_ROOT = Path("/home/timbushnell/crucible_sparse_capacity_gate_v1_20261017")
E4_MODEL_CORE = E4_ROOT / "model_core.py"
E4_ANCHOR_DIR = E4_ROOT / "reference_e4" / "crucible_manifest"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_stage_a_instrumentation_preflight_20261022"

sys.path.insert(0, str(E4_ROOT))
from model_core import (  # noqa: E402
    BraidPrototypeLM,
    MultiHeadLatentAttention,
    ResearchMoETelemetry,
    SwiGLU,
)


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
RAW_BYTE_START = 3
VOCAB_SIZE = 259
CONTEXT_LENGTH = 320
ANSWER_MARKER = b"\nAnswer:"
E4_EXPERTS = 4
E4_TOP_K = 2
E4_D_FF = 3447
DENSE_D_FF = 6894
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
LATENT_KV = 64
LATENT_Q = 64
PE_DIM = 32
LR = 3.0e-4
WEIGHT_DECAY = 0.1
BATCH_SIZE = 4
CLIP_NORM = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_state_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(prefix.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: str(item)):
                visit(f"{prefix}.{key}", value[key])
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(f"{prefix}[{index}]", item)
        else:
            digest.update(prefix.encode("utf-8"))
            digest.update(repr(value).encode("utf-8"))

    visit("root", state)
    return digest.hexdigest()


def index_fingerprint(indices: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(indices), separators=(",", ":")).encode("ascii")).hexdigest()


def batch_indices(order: Sequence[int], step: int) -> List[int]:
    start = ((step - 1) * BATCH_SIZE) % len(order)
    return [order[(start + offset) % len(order)] for offset in range(BATCH_SIZE)]


def diagnostic_steps(steps: int) -> set[int]:
    return {10, 20} if steps == 20 else {25, 50, 75, 100, 200, 300, 400, 500}


def checkpoint_steps(steps: int) -> set[int]:
    return {10, 20} if steps == 20 else {100, 500}


def state_fingerprints(model: nn.Module, optimizer: torch.optim.Optimizer, order: Sequence[int], completed_step: int) -> Dict[str, str]:
    return {
        "model_state_digest": canonical_state_digest(model.state_dict()),
        "optimizer_state_digest": canonical_state_digest(optimizer.state_dict()),
        "torch_rng_digest": canonical_state_digest({"torch_rng_state": torch.get_rng_state()}),
        "python_rng_digest": canonical_state_digest({"python_rng_state": random.getstate()}),
        "next_sample_fingerprint": index_fingerprint(batch_indices(order, completed_step + 1)),
    }


def serialize_bytes(value: bytes) -> List[int]:
    return [RAW_BYTE_START + byte for byte in value]


def load_records(split: str) -> List[Dict[str, Any]]:
    path = MANIFEST_DIR / f"{split}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for record in records:
        prompt = record["input"].encode("utf-8") + ANSWER_MARKER
        target = record["target"].encode("utf-8") + bytes([EOS_ID])
        full = [BOS_ID] + serialize_bytes(prompt) + serialize_bytes(target[:-1]) + [EOS_ID]
        if len(full) > CONTEXT_LENGTH:
            raise RuntimeError(f"task exceeds context: {record['record_id']} ({len(full)})")
        prefix_len = 1 + len(serialize_bytes(prompt))
        record["_full"] = full
        record["_prefix_len"] = prefix_len
    return records


def batchify(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    selected = [records[index] for index in indices]
    max_len = max(len(record["_full"]) - 1 for record in selected)
    x = torch.full((len(selected), max_len), PAD_ID, dtype=torch.long)
    y = torch.full((len(selected), max_len), PAD_ID, dtype=torch.long)
    target_mask = torch.zeros((len(selected), max_len), dtype=torch.bool)
    for row, record in enumerate(selected):
        full = torch.tensor(record["_full"], dtype=torch.long)
        input_tokens = full[:-1]
        labels = full[1:]
        length = input_tokens.numel()
        x[row, :length] = input_tokens
        y[row, :length] = labels
        start = int(record["_prefix_len"]) - 1
        target_mask[row, start:length] = True
    return {"x": x, "y": y, "target_mask": target_mask}


class DenseBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(D_MODEL)
        self.ffn_norm = nn.RMSNorm(D_MODEL)
        self.attn = MultiHeadLatentAttention(D_MODEL, N_HEADS, LATENT_KV, LATENT_Q, PE_DIM)
        self.ffn = SwiGLU(D_MODEL, DENSE_D_FF)

    def forward(self, x: torch.Tensor, seq_idx: torch.Tensor, current_sequence_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), seq_idx, current_sequence_ids=current_sequence_ids)
        return x + self.ffn(self.ffn_norm(x))


class DenseLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = VOCAB_SIZE
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.layers = nn.ModuleList([DenseBlock() for _ in range(N_LAYERS)])
        self.final_norm = nn.RMSNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, tokens: torch.Tensor, seq_idx: torch.Tensor, current_sequence_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)
        for layer in self.layers:
            x = layer(x, seq_idx, current_sequence_ids)
        return self.lm_head(self.final_norm(x))


def make_e4(seed: int) -> nn.Module:
    model = BraidPrototypeLM(VOCAB_SIZE, N_LAYERS, D_MODEL, N_HEADS, LATENT_KV, LATENT_Q, PE_DIM, E4_EXPERTS, E4_D_FF, top_k=E4_TOP_K)
    anchor = E4_ANCHOR_DIR / f"initial_seed_anchor_{seed}.pt"
    if not anchor.is_file():
        raise FileNotFoundError(anchor)
    model.load_state_dict(torch.load(anchor, map_location="cpu", weights_only=True), strict=True)
    return model


def make_dense(seed: int) -> nn.Module:
    torch.manual_seed(seed + 10_000)
    model = DenseLM()
    # Share the non-FFN initialization with the E4 anchor. This is a
    # learnability control, not a matched-total-parameter architecture claim.
    e4 = make_e4(seed)
    dense_state = model.state_dict()
    e4_state = e4.state_dict()
    for name in dense_state:
        if name.startswith("layers.") and ".ffn." in name:
            continue
        if name in e4_state and dense_state[name].shape == e4_state[name].shape:
            dense_state[name].copy_(e4_state[name])
    model.load_state_dict(dense_state, strict=True)
    return model


def optimizer_for(model: nn.Module) -> torch.optim.Optimizer:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if isinstance(module, (nn.RMSNorm, nn.LayerNorm, nn.Embedding)) or name.endswith("bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY, "lr": LR}, {"params": no_decay, "weight_decay": 0.0, "lr": LR}],
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def forward_model(model: nn.Module, batch: Mapping[str, torch.Tensor], telemetry: bool = False) -> torch.Tensor:
    x = batch["x"]
    seq_idx = torch.arange(x.size(1), dtype=torch.long).unsqueeze(0).expand(x.size(0), -1)
    sequence_ids = torch.arange(x.size(0), dtype=torch.long)
    if isinstance(model, BraidPrototypeLM):
        if telemetry:
            ResearchMoETelemetry.configure(VOCAB_SIZE)
            ResearchMoETelemetry.reset_step_buffer()
        return model(x, seq_idx, current_branch="B", current_sequence_ids=sequence_ids, telemetry_class=ResearchMoETelemetry if telemetry else None)
    return model(x, seq_idx, current_sequence_ids=sequence_ids)


def loss_and_accuracy(logits: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, int, int]:
    y = batch["y"]
    mask = batch["target_mask"] & (y != PAD_ID)
    token_losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view_as(y)
    loss = token_losses[mask].mean()
    predicted = logits.argmax(dim=-1)
    correct = int(((predicted == y) & mask).sum().item())
    total = int(mask.sum().item())
    return loss, correct, total


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, Any]:
    model.train()
    batch = batchify(records, indices)
    optimizer.zero_grad(set_to_none=True)
    logits = forward_model(model, batch)
    loss, correct, total = loss_and_accuracy(logits, batch)
    if isinstance(model, BraidPrototypeLM):
        aux = torch.stack([layer.moe.last_aux_loss for layer in model.layers]).mean()
        loss = loss + 0.1 * aux
    loss.backward()
    grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM).item())
    optimizer.step()
    return {"loss": float(loss.detach().item()), "task_tokens_correct": correct, "task_tokens": total, "grad_norm": grad_norm}


@torch.no_grad()
def evaluate(model: nn.Module, records: Sequence[Mapping[str, Any]], batch_size: int = BATCH_SIZE) -> Dict[str, Any]:
    model.eval()
    total_correct = 0
    total_tokens = 0
    exact = 0
    total = 0
    loss_sum = 0.0
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        batch = batchify(records, indices)
        logits = forward_model(model, batch)
        loss, correct, tokens = loss_and_accuracy(logits, batch)
        mask = batch["target_mask"] & (batch["y"] != PAD_ID)
        per_item = ((logits.argmax(dim=-1) == batch["y"]) | ~mask).all(dim=1)
        total_correct += correct
        total_tokens += tokens
        exact += int(per_item.sum().item())
        total += len(indices)
        loss_sum += float(loss.item()) * len(indices)
    return {
        "teacher_forced_token_accuracy": total_correct / max(1, total_tokens),
        "teacher_forced_exact_accuracy": exact / max(1, total),
        "mean_target_loss": loss_sum / max(1, total),
        "examples": total,
    }


@torch.no_grad()
def probe(model: nn.Module, records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, Any]:
    model.eval()
    batch = batchify(records, indices)
    logits = forward_model(model, batch, telemetry=isinstance(model, BraidPrototypeLM))
    digest = hashlib.sha256(logits.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
    result: Dict[str, Any] = {"logit_digest": digest, "parameter_digest": canonical_state_digest(model.state_dict())}
    if isinstance(model, BraidPrototypeLM):
        telemetry = ResearchMoETelemetry.get_summary("B")
        result["routing_digest"] = hashlib.sha256(json.dumps(telemetry, sort_keys=True).encode()).hexdigest()
        result["routing_layers"] = sorted(telemetry)
        result["routing_top1_counts"] = {layer: payload["top1_counts"] for layer, payload in telemetry.items()}
    return result


def intervention_probe(model: nn.Module, records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, Any]:
    baseline = probe(model, records, indices)
    if not isinstance(model, BraidPrototypeLM):
        return {"baseline": baseline, "intervention_supported": False}
    ablated = copy.deepcopy(model)
    def zero_expert(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(output)
    hook = ablated.layers[0].moe.experts[0].register_forward_hook(zero_expert)
    try:
        ablation = probe(ablated, records, indices)
    finally:
        hook.remove()
    permuted = copy.deepcopy(model)
    for layer in permuted.layers:
        if len(layer.moe.experts) >= 2:
            left = copy.deepcopy(layer.moe.experts[0].state_dict())
            right = copy.deepcopy(layer.moe.experts[1].state_dict())
            layer.moe.experts[0].load_state_dict(right)
            layer.moe.experts[1].load_state_dict(left)
    permutation = probe(permuted, records, indices)
    return {
        "baseline": baseline,
        "one_expert_zero_ablation": ablation,
        "expert_0_1_permutation": permutation,
        "intervention_supported": True,
        "original_parameter_digest_after_intervention": canonical_state_digest(model.state_dict()),
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, order: Sequence[int]) -> None:
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "sample_order_prefix": list(order),
    }, path)


def reload_probe(path: Path, model_factory, records: Sequence[Mapping[str, Any]], indices: Sequence[int], original: Mapping[str, Any]) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored = model_factory()
    restored.load_state_dict(checkpoint["model"], strict=True)
    result = probe(restored, records, indices)
    result["matches_original_probe"] = result["logit_digest"] == original["logit_digest"]
    result["checkpoint_step"] = int(checkpoint["step"])
    return result


def reload_intervention_probe(path: Path, model_factory, records: Sequence[Mapping[str, Any]], indices: Sequence[int], original: Mapping[str, Any]) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored = model_factory()
    restored.load_state_dict(checkpoint["model"], strict=True)
    result = intervention_probe(restored, records, indices)
    result["matches_original_intervention"] = json.dumps(result, sort_keys=True) == json.dumps(original, sort_keys=True)
    result["checkpoint_step"] = int(checkpoint["step"])
    return result


def resume_once(path: Path, model_factory, records: Sequence[Mapping[str, Any]], order: Sequence[int]) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = model_factory()
    optimizer = optimizer_for(model)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    completed_step = int(checkpoint["step"])
    indices = batch_indices(order, completed_step + 1)
    info = train_step(model, optimizer, records, indices)
    result = {
        "checkpoint_step": completed_step,
        "resumed_step": completed_step + 1,
        "indices": indices,
        "index_fingerprint": index_fingerprint(indices),
        "train_info": info,
        "state_fingerprints_after_step": state_fingerprints(model, optimizer, order, completed_step + 1),
    }
    return result


def resume_check(path: Path, model_factory, records: Sequence[Mapping[str, Any]], order: Sequence[int]) -> Dict[str, Any]:
    first = resume_once(path, model_factory, records, order)
    second = resume_once(path, model_factory, records, order)
    same_info = first["train_info"] == second["train_info"]
    same_fingerprints = first["state_fingerprints_after_step"] == second["state_fingerprints_after_step"]
    return {
        "checkpoint_step": first["checkpoint_step"],
        "resumed_step": first["resumed_step"],
        "expected_index_fingerprint": first["index_fingerprint"],
        "repeat_train_info_matches": same_info,
        "repeat_state_fingerprints_match": same_fingerprints,
        "first": first,
        "second": second,
        "pass": same_info and same_fingerprints,
    }


def preserving_rng(callable_obj):
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    try:
        return callable_obj()
    finally:
        torch.set_rng_state(torch_state)
        random.setstate(python_state)


def run_model(kind: str, seed: int, steps: int, output: Path, train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    factory = (lambda: make_e4(seed)) if kind == "e4" else (lambda: make_dense(seed))
    model = factory()
    optimizer = optimizer_for(model)
    model_dir = output / "checkpoints" / kind
    model_dir.mkdir(parents=True, exist_ok=True)
    probe_indices = list(range(min(8, len(validation))))
    order = list(range(len(train)))
    random.Random(seed + 100).shuffle(order)
    diagnostics = diagnostic_steps(steps)
    checkpoints = checkpoint_steps(steps)
    initial_probe = probe(model, validation, probe_indices)
    save_checkpoint(model_dir / "step_0000.pt", model, optimizer, 0, order)
    reload_initial = preserving_rng(lambda: reload_probe(model_dir / "step_0000.pt", factory, validation, probe_indices, initial_probe))
    reload_checks = {"initial_probe_reload_matches": reload_initial["matches_original_probe"]}
    resume_checks: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        indices = batch_indices(order, step)
        started = time.perf_counter()
        train_info = train_step(model, optimizer, train, indices)
        elapsed = time.perf_counter() - started
        row: Dict[str, Any] = {"kind": kind, "seed": seed, "step": step, "wall_seconds": elapsed, **train_info}
        if step in diagnostics:
            row["state_fingerprints"] = state_fingerprints(model, optimizer, order, step)
            row["validation"] = evaluate(model, validation)
        if step in checkpoints:
            checkpoint_path = model_dir / f"step_{step:04d}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, step, order[: min(len(order), step * BATCH_SIZE)])
            row["checkpoint"] = str(checkpoint_path.relative_to(output))
            row["checkpoint_model_digest"] = canonical_state_digest(model.state_dict())
            current_probe = probe(model, validation, probe_indices)
            current_intervention = intervention_probe(model, validation, probe_indices)
            row["reload_probe"] = preserving_rng(lambda: reload_probe(checkpoint_path, factory, validation, probe_indices, current_probe))
            row["intervention"] = current_intervention
            row["reload_intervention"] = preserving_rng(
                lambda: reload_intervention_probe(checkpoint_path, factory, validation, probe_indices, current_intervention)
            )
            reload_checks[f"step_{step}_probe_reload_matches"] = row["reload_probe"]["matches_original_probe"]
            reload_checks[f"step_{step}_intervention_reload_matches"] = row["reload_intervention"]["matches_original_intervention"]
            if step == 100 and steps >= 500:
                row["resume_check"] = preserving_rng(lambda: resume_check(checkpoint_path, factory, train, order))
                resume_checks[f"step_{step}_resume_pass"] = row["resume_check"]["pass"]
        records.append(row)
        print(json.dumps({"kind": kind, "seed": seed, "step": step, "loss": row["loss"], "wall_seconds": round(elapsed, 3)}, sort_keys=True), flush=True)
    final = evaluate(model, validation)
    return {
        "kind": kind,
        "seed": seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "steps": steps,
        "final_validation": final,
        "reload_checks": reload_checks,
        "resume_checks": resume_checks,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--steps", type=int, choices=(20, 500), default=20)
    parser.add_argument("--seed", type=int, choices=(42, 1337), default=42)
    args = parser.parse_args()
    if torch.cuda.is_available():
        # CPU-only is a contract for this stage.  Do not select CUDA even if
        # it exists on the host.
        device = "cpu"
    else:
        device = "cpu"
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train = load_records("train")
    validation = load_records("validation")
    test = load_records("test")
    max_len = max(len(record["_full"]) for record in train + validation + test)
    experiment_name = (
        "crucible_stage_a_instrumentation_preflight_20261022"
        if args.steps == 20
        else "crucible_stage_a_instrumentation_pilot_20261022"
    )
    config = {
        "experiment": experiment_name,
        "status": "cpu_only_preflight" if args.steps == 20 else "cpu_only_pilot",
        "device": device,
        "steps": args.steps,
        "seed": args.seed,
        "model_arms": ["e4", "dense_active_compute"],
        "context_length": CONTEXT_LENGTH,
        "max_encoded_sequence_length": max_len,
        "vocab_size": VOCAB_SIZE,
        "task_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "manifest_sha256": sha256_file(MANIFEST_DIR / "manifest.json"),
        "model_core_sha256": sha256_file(E4_MODEL_CORE),
        "e4_anchor_sha256": sha256_file(E4_ANCHOR_DIR / f"initial_seed_anchor_{args.seed}.pt"),
        "e4_parameter_geometry": {"num_experts": E4_EXPERTS, "top_k": E4_TOP_K, "d_ff": E4_D_FF},
        "dense_control_geometry": {"d_ff": DENSE_D_FF, "active_ffn_width": DENSE_D_FF},
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "diagnostic_steps": sorted(diagnostic_steps(args.steps)),
        "checkpoint_steps": [0, *sorted(checkpoint_steps(args.steps))],
        "task_accuracy": "teacher_forced_exact_accuracy; generation not claimed",
        "test_usage": "not used for checkpoint selection or preflight metrics",
    }
    write_json(output / "RUN_CONFIG.json", config)
    started = time.perf_counter()
    results = []
    for kind in ("e4", "dense_active_compute"):
        results.append(run_model(kind, args.seed, args.steps, output, train, validation, test))
    result = {
        "experiment": config["experiment"],
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - started,
        "test_evaluated": False,
        "results": results,
        "mechanical_pass": all(
            all(item["reload_checks"].values()) and all(item["resume_checks"].values())
            for item in results
        ),
        "interpretation": "learnability/instrumentation preflight only; not evidence of recursive cognition or architectural superiority",
    }
    write_json(output / "RESULTS.json", result)
    (output / "RUNS.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for result_row in results for row in result_row["records"]) + "\n", encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"status": result["status"], "output": str(output), "elapsed_seconds": result["elapsed_seconds"], "mechanical_pass": result["mechanical_pass"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
