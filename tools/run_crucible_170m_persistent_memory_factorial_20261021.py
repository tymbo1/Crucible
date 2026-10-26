#!/usr/bin/env python3
"""Run the local 170M Crucible persistent-memory factorial experiment.

The frozen E16 neural substrate is loaded from the existing context-128
package. This runner changes only the training procedure: retained top-k
output memory and a private loss-stability update-skip detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "crucible_170m_persistent_memory_factorial_20261021"
DEFAULT_OUTPUT = ROOT / "experiments" / EXPERIMENT
DATA_ROOT = Path("/home/timbushnell/crucible_context_gate_v1_20261018/context_128")
MANIFEST_PATH = DATA_ROOT / "crucible_manifest" / "experiment_config.json"
FROZEN_SHARED = Path("/home/timbushnell/crucible_context_gate_v1_20261018/shared")
MODEL_CORE_PATH = FROZEN_SHARED / "model_core.py"

SEEDS = (42, 1337)
ARMS = ("baseline", "memory_only", "skip_only", "full")
PHASES = 5
STEPS_PER_PHASE = 10_000
TOTAL_STEPS = PHASES * STEPS_PER_PHASE
MEMORY_TOP_K = 4
MEMORY_COEF = 0.10
MATURE_OBSERVATIONS = 8
STABILITY_DELTA = 0.02
PEAK_LR = 3.0e-4
WARMUP_STEPS = 1_000

if str(FROZEN_SHARED) not in sys.path:
    sys.path.insert(0, str(FROZEN_SHARED))
from model_core import BraidPrototypeLM  # noqa: E402


PAD_ID = 0
RAW_BYTE_START = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


class ImmutableBinaryDataset(Dataset):
    def __init__(self, path: Path, seq_len: int, expected_hash: str, expected_count: int):
        assert_hash(path, expected_hash)
        raw = path.read_bytes()
        stride = 2 * seq_len * 8
        if len(raw) % stride:
            raise RuntimeError(f"invalid binary geometry: {path}")
        count = len(raw) // stride
        if count != expected_count:
            raise RuntimeError(f"invalid binary count for {path}: {count} != {expected_count}")
        self.seq_len = seq_len
        self.total_samples = count
        self.raw_tensor = torch.frombuffer(bytearray(raw), dtype=torch.int64)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        offset = int(index) * 2 * self.seq_len
        return {
            "x": self.raw_tensor[offset : offset + self.seq_len].clone(),
            "y": self.raw_tensor[offset + self.seq_len : offset + 2 * self.seq_len].clone(),
        }


class LossHistory:
    """Private per-example history used only by skip-enabled arms."""

    def __init__(self, count: int):
        self.observations = torch.zeros(count, dtype=torch.int32)
        self.ema = torch.zeros(count, dtype=torch.float32)
        self.last = torch.zeros(count, dtype=torch.float32)
        self.skipped = 0

    def can_skip(self, indices: Sequence[int]) -> bool:
        ids = torch.tensor(indices, dtype=torch.long)
        if ids.numel() == 0:
            return False
        mature = self.observations.index_select(0, ids) >= MATURE_OBSERVATIONS
        stable = (self.last.index_select(0, ids) - self.ema.index_select(0, ids)).abs() <= STABILITY_DELTA
        return bool(torch.all(mature & stable).item())

    def observe(self, indices: Sequence[int], losses: torch.Tensor) -> None:
        ids = torch.tensor(indices, dtype=torch.long)
        values = losses.detach().to(dtype=torch.float32, device="cpu")
        old_count = self.observations.index_select(0, ids)
        old_ema = self.ema.index_select(0, ids)
        initialized = old_count == 0
        updated = torch.where(initialized, values, 0.90 * old_ema + 0.10 * values)
        self.ema.index_copy_(0, ids, updated)
        self.last.index_copy_(0, ids, values)
        self.observations.index_copy_(0, ids, old_count + 1)

    def summary(self) -> Dict[str, Any]:
        mature = int((self.observations >= MATURE_OBSERVATIONS).sum().item())
        return {
            "mature_examples": mature,
            "observed_examples": int((self.observations > 0).sum().item()),
            "skipped_updates": int(self.skipped),
        }


class OutputMemory:
    """Phase-frozen compact retained-output memory.

    Teacher arrays are never modified during a phase. New observations are
    written to next_* and become retrievable only at the next phase boundary.
    """

    def __init__(self, count: int, seq_len: int, top_k: int = MEMORY_TOP_K):
        self.count = count
        self.seq_len = seq_len
        self.top_k = top_k
        self.teacher_ids: Optional[torch.Tensor] = None
        self.teacher_probs: Optional[torch.Tensor] = None
        self.teacher_seen: Optional[torch.Tensor] = None
        self.next_ids = torch.zeros(count, seq_len, top_k, dtype=torch.int16)
        self.next_probs = torch.zeros(count, seq_len, top_k, dtype=torch.float16)
        self.next_seen = torch.zeros(count, dtype=torch.bool)
        self.lookup_count = 0
        self.teacher_hit_count = 0
        self.write_count = 0

    def begin_phase(self) -> None:
        self.next_ids.zero_()
        self.next_probs.zero_()
        self.next_seen.zero_()

    def teacher_batch(self, indices: Sequence[int]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.teacher_ids is None or self.teacher_probs is None or self.teacher_seen is None:
            return None
        ids = torch.tensor(indices, dtype=torch.long)
        seen = self.teacher_seen.index_select(0, ids)
        self.lookup_count += len(indices)
        self.teacher_hit_count += int(seen.sum().item())
        return (
            self.teacher_ids.index_select(0, ids),
            self.teacher_probs.index_select(0, ids),
            seen,
        )

    def write_batch(self, indices: Sequence[int], ids: torch.Tensor, probs: torch.Tensor) -> None:
        index_tensor = torch.tensor(indices, dtype=torch.long)
        self.next_ids.index_copy_(0, index_tensor, ids.detach().to("cpu", dtype=torch.int16))
        self.next_probs.index_copy_(0, index_tensor, probs.detach().to("cpu", dtype=torch.float16))
        self.next_seen.index_fill_(0, index_tensor, True)
        self.write_count += len(indices)

    def end_phase(self) -> None:
        self.teacher_ids = self.next_ids
        self.teacher_probs = self.next_probs
        self.teacher_seen = self.next_seen
        self.next_ids = torch.zeros(self.count, self.seq_len, self.top_k, dtype=torch.int16)
        self.next_probs = torch.zeros(self.count, self.seq_len, self.top_k, dtype=torch.float16)
        self.next_seen = torch.zeros(self.count, dtype=torch.bool)

    def save(self, path: Path) -> None:
        write_json(path.with_suffix(".meta.json"), {
            "count": self.count,
            "seq_len": self.seq_len,
            "top_k": self.top_k,
            "teacher_seen": int(self.teacher_seen.sum().item()) if self.teacher_seen is not None else 0,
            "lookup_count": self.lookup_count,
            "teacher_hit_count": self.teacher_hit_count,
            "write_count": self.write_count,
        })
        torch.save({
            "teacher_ids": self.teacher_ids,
            "teacher_probs": self.teacher_probs,
            "teacher_seen": self.teacher_seen,
        }, path)

    def summary(self) -> Dict[str, Any]:
        seen = int(self.teacher_seen.sum().item()) if self.teacher_seen is not None else 0
        return {
            "teacher_examples": seen,
            "lookup_count": int(self.lookup_count),
            "teacher_hit_count": int(self.teacher_hit_count),
            "write_count": int(self.write_count),
            "hit_rate": self.teacher_hit_count / max(1, self.lookup_count),
        }


def load_manifest() -> Dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["meta"].get("training_started"):
        raise RuntimeError("frozen manifest is already marked training-started")
    expected_model_hash = manifest["code_provenance"]["model_core_sha256"]
    actual_model_hash = sha256_file(MODEL_CORE_PATH)
    if actual_model_hash != expected_model_hash:
        raise RuntimeError(f"frozen model-core hash mismatch: {actual_model_hash} != {expected_model_hash}")
    return manifest


def build_inputs(manifest: Mapping[str, Any], seed: int) -> Tuple[ImmutableBinaryDataset, ImmutableBinaryDataset, List[Dict[str, Any]], Path]:
    hp = manifest["hyperparameters"]
    root = DATA_ROOT / "crucible_manifest"
    train_info = manifest["splits"]["train"]
    val_info = manifest["splits"]["validation"]
    train = ImmutableBinaryDataset(root / train_info["relative_path"], int(hp["context_window_length"]), train_info["binary_payload_sha256"], int(train_info["total_constructed_examples"]))
    validation = ImmutableBinaryDataset(root / val_info["relative_path"], int(hp["context_window_length"]), val_info["binary_payload_sha256"], int(val_info["total_constructed_examples"]))
    schedule_info = manifest["schedules"][str(seed)]
    schedule_path = root / schedule_info["relative_path"]
    assert_hash(schedule_path, schedule_info["binary_payload_sha256"])
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))["groups"]
    if len(schedule) != int(manifest["hyperparameters"]["num_steps"]):
        raise RuntimeError("frozen schedule length mismatch")
    return train, validation, schedule, root


def lr_value(step: int) -> float:
    if step < WARMUP_STEPS:
        return PEAK_LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS + 1) / (TOTAL_STEPS - WARMUP_STEPS)
    return PEAK_LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def optimizer_for(model: nn.Module, hp: Mapping[str, Any]) -> torch.optim.Optimizer:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    no_decay_types = (nn.RMSNorm, nn.LayerNorm, nn.Embedding)
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            (no_decay if isinstance(module, no_decay_types) or name.endswith("bias") else decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(hp["weight_decay"]), "lr": PEAK_LR},
            {"params": no_decay, "weight_decay": 0.0, "lr": PEAK_LR},
        ],
        betas=(float(hp["adamw_beta1"]), float(hp["adamw_beta2"])),
        eps=float(hp["adamw_eps"]),
    )


def model_for(manifest: Mapping[str, Any], seed: int, device: torch.device) -> nn.Module:
    hp = manifest["hyperparameters"]
    model = BraidPrototypeLM(
        int(hp["vocab_size"]), int(hp["n_layers"]), int(hp["d_model"]),
        int(hp["n_heads"]), int(hp["latent_dim_kv"]), int(hp["latent_dim_q"]),
        int(hp["pe_dim"]), int(hp["num_experts"]), int(hp["d_ff"]), top_k=int(hp["top_k"]),
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != int(hp["expected_parameter_geometry"]):
        raise RuntimeError("170M model parameter geometry mismatch")
    anchor_info = manifest["anchors"][str(seed)]
    anchor_path = DATA_ROOT / "crucible_manifest" / anchor_info["relative_path"]
    assert_hash(anchor_path, anchor_info["binary_payload_sha256"])
    model.load_state_dict(torch.load(anchor_path, map_location=device, weights_only=True), strict=True)
    return model


def output_capture(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    values, ids = torch.topk(logits.detach().float(), MEMORY_TOP_K, dim=-1)
    probs = F.softmax(values, dim=-1)
    return ids, probs


def per_example_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none").view(targets.shape)
    valid = targets != PAD_ID
    return (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def train_phase(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train: ImmutableBinaryDataset,
    schedule: Sequence[Mapping[str, Any]],
    phase: int,
    arm: str,
    device: torch.device,
    memory: Optional[OutputMemory],
    history: Optional[LossHistory],
    autocast_context: Any,
    steps_to_run: int = STEPS_PER_PHASE,
    phase_stride: int = STEPS_PER_PHASE,
) -> Dict[str, Any]:
    start_global = (phase - 1) * phase_stride
    applied = 0
    skipped = 0
    total_train_loss = 0.0
    total_train_tokens = 0
    memory_loss_total = 0.0
    memory_tokens = 0
    interval_start = time.perf_counter()
    model.train()
    if memory is not None:
        memory.begin_phase()

    for local_step in range(steps_to_run):
        global_step = start_global + local_step
        # Each continuation phase starts from the same deterministic corpus
        # schedule. This mirrors the sequential-run design and makes the
        # short 100-update validation exercise memory reuse rather than merely
        # traversing disjoint examples once.
        group = schedule[local_step % len(schedule)]
        indices = [int(index) for index in group["indices"]]
        skip_enabled = arm in {"skip_only", "full"}
        if skip_enabled and history is not None and history.can_skip(indices):
            skipped += 1
            history.skipped += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        total_step_tokens = int(group["nonpad_target_tokens"])
        step_losses: List[Tuple[List[int], torch.Tensor]] = []
        step_memory_loss = 0.0
        step_memory_tokens = 0
        microbatch_size = int(group["microbatch_size"])
        for start in range(0, len(indices), microbatch_size):
            batch_indices = indices[start : start + microbatch_size]
            x_batch = torch.stack([train[index]["x"] for index in batch_indices]).to(device)
            y_batch = torch.stack([train[index]["y"] for index in batch_indices]).to(device)
            seq_idx = torch.arange(x_batch.size(1), device=device).unsqueeze(0).expand(x_batch.size(0), -1)
            sequence_ids = torch.arange(x_batch.size(0), device=device) + start
            with autocast_context:
                logits = model(
                    x_batch, seq_idx, current_branch="B", current_sequence_ids=sequence_ids,
                    telemetry_class=None,
                )
                ce_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1), ignore_index=PAD_ID, reduction="sum")
                valid_count = int((y_batch != PAD_ID).sum().item())
                aux = torch.stack([layer.moe.last_aux_loss for layer in model.layers]).mean()
                loss_numerator = ce_sum + float(MANIFEST_HYPERPARAMS["branch_b_aux_coef"]) * aux * valid_count
                if memory is not None and phase > 1:
                    teacher = memory.teacher_batch(batch_indices)
                    if teacher is not None:
                        teacher_ids, teacher_probs, seen = teacher
                        if bool(seen.any().item()):
                            teacher_ids_device = teacher_ids.to(device=device, dtype=torch.long)
                            teacher_probs_device = teacher_probs.to(device=device, dtype=torch.float32)
                            token_log_probs = F.log_softmax(logits.float(), dim=-1)
                            retained_ce = -(teacher_probs_device * token_log_probs.gather(-1, teacher_ids_device)).sum(dim=-1)
                            token_mask = (y_batch != PAD_ID) & seen.to(device).view(-1, 1)
                            retained_sum = (retained_ce * token_mask).sum()
                            retained_tokens = int(token_mask.sum().item())
                            loss_numerator = loss_numerator + MEMORY_COEF * retained_sum
                            step_memory_loss += float(retained_sum.detach().item())
                            step_memory_tokens += retained_tokens
                loss = loss_numerator / max(1, total_step_tokens)
            loss.backward()
            example_losses = per_example_loss(logits, y_batch).detach().cpu()
            step_losses.append((batch_indices, example_losses))
            total_train_loss += float(ce_sum.detach().item())
            total_train_tokens += valid_count
            if memory is not None:
                ids, probs = output_capture(logits)
                memory.write_batch(batch_indices, ids, probs)

        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), float(MANIFEST_HYPERPARAMS["max_norm_clip"])).item())
        if math.isfinite(grad_norm):
            optimizer.step()
            applied += 1
        for batch_indices, losses in step_losses:
            if history is not None:
                history.observe(batch_indices, losses)
        memory_loss_total += step_memory_loss
        memory_tokens += step_memory_tokens

    elapsed = time.perf_counter() - interval_start
    return {
        "phase": phase,
        "arm": arm,
        "attempted_updates": steps_to_run,
        "applied_updates": applied,
        "skipped_updates": skipped,
        "mean_train_ce": total_train_loss / max(1, total_train_tokens),
        "mean_retained_output_ce": memory_loss_total / max(1, memory_tokens),
        "retained_output_tokens": memory_tokens,
        "wall_seconds": elapsed,
        "updates_per_second_attempted": steps_to_run / max(elapsed, 1e-9),
        "updates_per_second_applied": applied / max(elapsed, 1e-9),
    }


def evaluate(model: nn.Module, validation: ImmutableBinaryDataset, hp: Mapping[str, Any], device: torch.device, autocast_context: Any) -> Dict[str, Any]:
    model.eval()
    total_ce = 0.0
    total_raw_ce = 0.0
    total_tokens = 0
    total_raw = 0
    started = time.perf_counter()
    batch_size = int(hp["microbatch_size"])
    with torch.no_grad():
        for start in range(0, len(validation), batch_size):
            batch_indices = list(range(start, min(start + batch_size, len(validation))))
            x = torch.stack([validation[index]["x"] for index in batch_indices]).to(device)
            y = torch.stack([validation[index]["y"] for index in batch_indices]).to(device)
            seq_idx = torch.arange(x.size(1), device=device).unsqueeze(0).expand(x.size(0), -1)
            ids = torch.arange(x.size(0), device=device)
            with autocast_context:
                logits = model(x, seq_idx, current_branch="B", current_sequence_ids=ids, telemetry_class=None)
                token_losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(y.shape)
            valid = y != PAD_ID
            raw = y >= RAW_BYTE_START
            total_ce += float((token_losses * valid).sum().item())
            total_raw_ce += float((token_losses * raw).sum().item())
            total_tokens += int(valid.sum().item())
            total_raw += int(raw.sum().item())
    elapsed = time.perf_counter() - started
    raw_nll = total_raw_ce / max(1, total_raw)
    return {
        "validation_nll": total_ce / max(1, total_tokens),
        "validation_raw_byte_nll": raw_nll,
        "validation_bits_per_raw_byte": raw_nll / math.log(2.0),
        "validation_raw_byte_targets": total_raw,
        "validation_wall_seconds": elapsed,
    }


def clone_state_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def run_arm(
    seed: int,
    arm: str,
    manifest: Mapping[str, Any],
    output: Path,
    device: torch.device,
    max_steps: Optional[int],
    phase_count: int,
    steps_per_phase: int,
) -> List[Dict[str, Any]]:
    global MANIFEST_HYPERPARAMS
    MANIFEST_HYPERPARAMS = manifest["hyperparameters"]
    train, validation, schedule, _ = build_inputs(manifest, seed)
    model = model_for(manifest, seed, device)
    optimizer = optimizer_for(model, manifest["hyperparameters"])
    memory = OutputMemory(len(train), int(manifest["hyperparameters"]["context_window_length"])) if arm in {"memory_only", "full"} else None
    history = LossHistory(len(train)) if arm in {"skip_only", "full"} else None
    autocast_context = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    total_steps = TOTAL_STEPS if max_steps is None else int(max_steps)
    steps_limit = total_steps
    phase_records: List[Dict[str, Any]] = []
    for phase in range(1, phase_count + 1):
        phase_start = (phase - 1) * steps_per_phase
        if phase_start >= steps_limit:
            break
        phase_steps = min(steps_per_phase, steps_limit - phase_start)
        phase_info = train_phase(
            model, optimizer, train, schedule, phase, arm, device, memory,
            history, autocast_context, steps_to_run=phase_steps,
            phase_stride=steps_per_phase,
        )
        validation_info = evaluate(model, validation, manifest["hyperparameters"], device, autocast_context)
        record = {**phase_info, **validation_info, "seed": seed, "model_parameter_count": sum(p.numel() for p in model.parameters())}
        if history is not None:
            record["loss_history"] = history.summary()
        if memory is not None:
            memory.end_phase()
            record["output_memory"] = memory.summary()
            if max_steps is None:
                memory_path = output / "memory" / f"seed_{seed}_{arm}_phase_{phase:02d}.pt"
                memory.save(memory_path)
        if max_steps is None:
            model_path = output / "model_states" / f"seed_{seed}_{arm}_phase_{phase:02d}.pt"
            torch.save(clone_state_cpu(model), model_path)
            record["model_state_sha256"] = sha256_file(model_path)
            if memory is not None:
                record["memory_state_sha256"] = sha256_file(output / "memory" / f"seed_{seed}_{arm}_phase_{phase:02d}.pt")
        phase_records.append(record)
        print(json.dumps({"seed": seed, "arm": arm, "phase": phase, "validation_bits_per_raw_byte": record["validation_bits_per_raw_byte"], "applied": record["applied_updates"], "skipped": record["skipped_updates"], "wall_seconds": round(record["wall_seconds"], 2)}, sort_keys=True), flush=True)
    del model, optimizer, train, validation
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return phase_records


def summarize(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    final: Dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm]
        final_phase = max((int(row["phase"]) for row in rows), default=0)
        final[arm] = {
            "seeds": len({row["seed"] for row in rows}),
            "final_phase": final_phase,
            "validation_bits_per_raw_byte": {str(row["seed"]): row["validation_bits_per_raw_byte"] for row in rows if row["phase"] == final_phase},
            "validation_raw_byte_nll": {str(row["seed"]): row["validation_raw_byte_nll"] for row in rows if row["phase"] == final_phase},
            "applied_updates": {str(row["seed"]): row["applied_updates"] for row in rows if row["phase"] == final_phase},
            "skipped_updates": {str(row["seed"]): row["skipped_updates"] for row in rows if row["phase"] == final_phase},
            "wall_seconds": {str(row["seed"]): row["wall_seconds"] for row in rows if row["phase"] == final_phase},
        }
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-steps", type=int, default=None, help="preflight only; full scientific runs use 50,000")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    if any(seed not in SEEDS for seed in args.seeds):
        raise SystemExit(f"available frozen anchors are {SEEDS}")
    if args.max_steps is not None and args.max_steps not in {10, 100, 500, TOTAL_STEPS}:
        raise SystemExit("--max-steps is restricted to 10, 100, 500, or 50000")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch reports no CUDA device")
    if args.max_steps is None and device.type != "cuda":
        raise SystemExit("the 170M full run requires CUDA; refusing an accidental CPU launch")
    if args.max_steps == 10:
        phase_count, steps_per_phase = 1, 10
    elif args.max_steps == 100:
        phase_count, steps_per_phase = 5, 20
    elif args.max_steps == 500:
        # Ten phase boundaries provide ten observations for the same
        # deterministic schedule prefix.  The unchanged maturity threshold
        # of eight can therefore activate naturally in later phases.
        phase_count, steps_per_phase = 10, 50
    else:
        phase_count, steps_per_phase = PHASES, STEPS_PER_PHASE
    manifest = load_manifest()
    output = args.output.resolve()
    existing = [path for path in output.iterdir() if path.name != "PROTOCOL.md"] if output.exists() else []
    if existing:
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_states").mkdir()
    (output / "memory").mkdir()
    write_json(output / "RUN_CONFIG.json", {
        "experiment": EXPERIMENT,
        "device": str(device),
        "seeds": args.seeds,
        "arms": args.arms,
        "phases": phase_count,
        "steps_per_phase": steps_per_phase,
        "total_attempted_steps_per_arm_and_seed": TOTAL_STEPS if args.max_steps is None else args.max_steps,
        "model_parameter_count": int(manifest["hyperparameters"]["expected_parameter_geometry"]),
        "memory_top_k": MEMORY_TOP_K,
        "memory_coefficient": MEMORY_COEF,
        "mature_observations": MATURE_OBSERVATIONS,
        "stability_delta": STABILITY_DELTA,
        "data_manifest": str(MANIFEST_PATH),
        "data_manifest_sha256": sha256_file(MANIFEST_PATH),
        "model_core": str(MODEL_CORE_PATH),
        "model_core_sha256": sha256_file(MODEL_CORE_PATH),
        "preflight": args.max_steps is not None,
        "activation_test": args.max_steps == 500,
    })
    all_records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for seed in args.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        for arm in args.arms:
            all_records.extend(run_arm(seed, arm, manifest, output, device, args.max_steps, phase_count, steps_per_phase))
            # Persist the complete prefix after each arm so interruption leaves
            # inspectable records without pretending the run is complete.
            (output / "RUNS.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_records), encoding="utf-8")
    result = {
        "experiment": EXPERIMENT,
        "status": "completed" if args.max_steps is None else "short_run_completed",
        "elapsed_seconds": time.perf_counter() - started,
        "records": all_records,
        "final_summary": summarize(all_records),
    }
    write_json(output / "RESULTS.json", result)
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            try:
                key = str(path.relative_to(ROOT))
            except ValueError:
                key = str(path.relative_to(output))
            hashes[key] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"status": result["status"], "output": str(output), "elapsed_seconds": result["elapsed_seconds"], "records": len(all_records)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
