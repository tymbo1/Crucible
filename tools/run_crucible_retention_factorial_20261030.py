#!/usr/bin/env python3
"""Controlled four-arm retention continuation from the 120M step-15k checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
THROUGHPUT = ROOT / "experiments" / "crucible_heterogeneous_training_throughput_20261029"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(THROUGHPUT))
sys.path.insert(0, str(TOOLS))

from benchmark import make_model, parameter_groups, state_digest  # noqa: E402
from run_crucible_foundation_training_20261029 import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    PAD_ID,
    RAW_BYTE_START,
    NonFiniteTrainingError,
    batch_from_rows,
    encode_reasoning,
    evaluate_loss,
    finite_state_issue,
    foundation_model_config,
    load_packed,
    load_jsonl,
    reasoning_batch,
    save_checkpoint,
    sha256_file,
)


SCHEMA = "crucible-retention-factorial-v2-20261030"
ARMS = {
    "A0": {"head_weight": 0.0, "representation_weight": 0.0, "description": "ordinary continuation; matched replay compute, no retention gradient"},
    "A1": {"head_weight": 0.05, "representation_weight": 0.0, "description": "output-logit anchoring"},
    "A2": {"head_weight": 0.0, "representation_weight": 0.05, "description": "final-state representation anchoring"},
    "A3": {"head_weight": 0.05, "representation_weight": 0.05, "description": "output and representation anchoring"},
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def forward_with_final_hidden(model: torch.nn.Module, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_idx = torch.arange(tokens.size(1), device=tokens.device, dtype=torch.long).unsqueeze(0).expand(tokens.size(0), -1)
    hidden = model.token_emb(tokens)
    for layer in model.layers:
        hidden = layer(hidden, seq_idx)
    hidden = model.final_norm(hidden)
    return model.lm_head(hidden), hidden


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def anchor_losses(
    student_logits: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_hidden: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = targets != PAD_ID
    if not bool(mask.any().item()):
        zero = student_logits.sum() * 0.0
        return zero, zero
    temperature = 2.0
    student_log_probs = F.log_softmax(student_logits[mask] / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits[mask].float() / temperature, dim=-1).to(student_log_probs.dtype)
    head_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)
    student_norm = F.normalize(student_hidden[mask].float(), dim=-1)
    teacher_norm = F.normalize(teacher_hidden[mask].float(), dim=-1)
    representation_loss = F.mse_loss(student_norm, teacher_norm)
    return head_loss, representation_loss


@torch.no_grad()
def precompute_teacher_targets(
    teacher: torch.nn.Module,
    samples: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cache frozen-teacher outputs on CPU so retention training needs one model."""
    teacher.eval()
    logits_chunks: List[torch.Tensor] = []
    hidden_chunks: List[torch.Tensor] = []
    for start in range(0, len(samples), batch_size):
        indices = range(start, min(start + batch_size, len(samples)))
        x, _ = reasoning_batch(samples, indices)
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, hidden = forward_with_final_hidden(teacher, x)
        logits_chunks.append(logits.detach().to(dtype=torch.bfloat16, device="cpu"))
        hidden_chunks.append(hidden.detach().to(dtype=torch.bfloat16, device="cpu"))
    return torch.cat(logits_chunks, dim=0), torch.cat(hidden_chunks, dim=0)


def train_update(
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_batch: Tuple[torch.Tensor, torch.Tensor],
    replay_batch: Tuple[torch.Tensor, torch.Tensor],
    teacher_targets: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    step: int,
    source: str,
    head_weight: float,
    representation_weight: float,
    loss_scale: float,
    max_nonfinite_retries: int,
) -> Dict[str, Any]:
    current_scale = float(max(1.0, loss_scale))
    for attempt in range(max_nonfinite_retries + 1):
        optimizer.zero_grad(set_to_none=True)
        x_cpu, y_cpu = train_batch
        rx_cpu, ry_cpu = replay_batch
        teacher_logits_cpu, teacher_hidden_cpu = teacher_targets
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        rx = rx_cpu.to(device, non_blocking=True)
        ry = ry_cpu.to(device, non_blocking=True)
        teacher_logits = teacher_logits_cpu.to(device, non_blocking=True)
        teacher_hidden = teacher_hidden_cpu.to(device, non_blocking=True)
        valid_tokens = int((y != PAD_ID).sum().item())
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            train_logits, _ = forward_with_final_hidden(student, x)
            train_log_probs = F.log_softmax(train_logits.float(), dim=-1)
            train_loss = F.nll_loss(train_log_probs.reshape(-1, train_log_probs.size(-1)), y.reshape(-1), ignore_index=PAD_ID)
        if not all(finite_tensor(value) for value in (train_logits, train_loss)):
            raise NonFiniteTrainingError(step, source, "nonfinite_training_loss", attempt)
        (train_loss * current_scale).backward()
        del train_logits, train_log_probs, x, y

        # Run the replay forward/backward only after the ordinary training graph
        # has been released. This preserves additive gradients while avoiding two
        # simultaneous 120M-model activation graphs on the 8GB GPU.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            replay_logits, replay_hidden = forward_with_final_hidden(student, rx)
            head_loss, representation_loss = anchor_losses(replay_logits, replay_hidden, teacher_logits, teacher_hidden, ry)
            retention_loss = head_weight * head_loss + representation_weight * representation_loss
            total_loss = train_loss.detach() + retention_loss.detach()
        if not all(finite_tensor(value) for value in (replay_logits, head_loss, representation_loss, retention_loss, total_loss)):
            raise NonFiniteTrainingError(step, source, "nonfinite_retention_loss", attempt)
        (retention_loss * current_scale).backward()
        del replay_logits, replay_hidden, rx, ry, teacher_logits, teacher_hidden
        finite_grads = True
        for parameter in student.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(current_scale)
                if not finite_tensor(parameter.grad):
                    finite_grads = False
                    break
        if not finite_grads:
            if attempt < max_nonfinite_retries and current_scale > 1.0:
                current_scale = max(1.0, current_scale / 2.0)
                continue
            raise NonFiniteTrainingError(step, source, "nonfinite_gradients", attempt)
        gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        issue = finite_state_issue(student, optimizer)
        if issue is not None:
            raise NonFiniteTrainingError(step, source, issue, attempt)
        return {
            "loss": float(train_loss.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
            "head_anchor_loss": float(head_loss.detach().cpu().item()),
            "representation_anchor_loss": float(representation_loss.detach().cpu().item()),
            "valid_tokens": valid_tokens,
            "gradient_norm": float(gradient_norm.detach().cpu().item()),
            "finite_gradients": True,
            "loss_scale": current_scale,
            "loss_scale_retries": attempt,
        }
    raise NonFiniteTrainingError(step, source, "retry_exhausted", max_nonfinite_retries)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fineweb-root", type=Path, default=Path("/home/timbushnell/crucible_fineweb_foundation_100m_20261029"))
    parser.add_argument("--reasoning-root", type=Path, default=Path("/home/timbushnell/crucible_reasoning_pilot_20261029"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reasoning-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-rows", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--initial-loss-scale", type=float, default=128.0)
    parser.add_argument("--max-nonfinite-retries", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    arm = ARMS[args.arm]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)

    fineweb_root = args.fineweb_root.resolve()
    fineweb_manifest = json.loads((fineweb_root / "MANIFEST.json").read_text())
    context = int(fineweb_manifest["packing"]["context"])
    language_train = load_packed(fineweb_root / fineweb_manifest["packed_files"]["train"]["path"], fineweb_manifest["packed_files"]["train"]["shape"])
    language_validation = load_packed(fineweb_root / fineweb_manifest["packed_files"]["validation"]["path"], fineweb_manifest["packed_files"]["validation"]["shape"])
    reasoning_root = args.reasoning_root.resolve()
    reasoning_train_rows = load_jsonl(reasoning_root / "train.jsonl")
    reasoning_validation_rows = load_jsonl(reasoning_root / "validation.jsonl")
    reasoning_train = [sample for row in reasoning_train_rows if (sample := encode_reasoning(row, context)) is not None]
    reasoning_validation = [sample for row in reasoning_validation_rows if (sample := encode_reasoning(row, context)) is not None]
    if len(reasoning_validation) < args.batch_size:
        raise RuntimeError("insufficient validation replay samples")
    replay_ids = [row.get("record_id", index) for index, row in enumerate(reasoning_validation_rows) if encode_reasoning(row, context) is not None]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    teacher = make_model(foundation_model_config("120m", False), device, seed=args.seed)
    teacher.load_state_dict(checkpoint["model"])
    try:
        teacher_targets = precompute_teacher_targets(teacher, reasoning_validation, device, args.batch_size)
    except torch.cuda.OutOfMemoryError as exc:
        write_json(output / "RESOURCE_ABORT.json", {"schema_version": SCHEMA, "phase": "teacher_target_precompute", "resource": "cuda_oom", "error": str(exc), "last_known_good_checkpoint": str(args.checkpoint.resolve())})
        raise
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    student = make_model(foundation_model_config("120m", False), device, seed=args.seed)
    optimizer = torch.optim.AdamW(parameter_groups(student, 0.01), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    student.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_step = int(checkpoint["step"])
    training_tokens = int(checkpoint.get("training_tokens", 0))
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all([state.detach().cpu().to(dtype=torch.uint8) for state in checkpoint["cuda_rng_state_all"]])
    issue = finite_state_issue(student, optimizer)
    if issue is not None:
        raise RuntimeError(f"non-finite start checkpoint: {issue}")

    checkpoint_root = args.checkpoint_root.resolve()
    curve_path = output / "RETENTION_CURVE.jsonl"
    safety_log = output / "SAFETY_EVENTS.jsonl"
    loss_scale = float(checkpoint.get("loss_scale", args.initial_loss_scale))
    run_start_tokens = training_tokens
    started = time.perf_counter()
    updates: List[Dict[str, Any]] = []
    with curve_path.open("w", encoding="utf-8") as curve:
        for step in range(start_step, args.steps):
            is_reasoning = (step + 1) % max(1, args.reasoning_every) == 0
            if is_reasoning:
                indices = [((step // max(1, args.reasoning_every)) * args.batch_size + i) % len(reasoning_train) for i in range(args.batch_size)]
                train_batch = reasoning_batch(reasoning_train, indices)
                source = "reasoning"
            else:
                indices = [(step * args.batch_size + i) % language_train.size(0) for i in range(args.batch_size)]
                train_batch = batch_from_rows(language_train, indices, context)
                source = "fineweb_edu"
            replay_start = ((step - start_step) * args.batch_size) % len(reasoning_validation)
            replay_indices = [(replay_start + i) % len(reasoning_validation) for i in range(args.batch_size)]
            replay_batch = reasoning_batch(reasoning_validation, replay_indices)
            replay_teacher_targets = (
                teacher_targets[0][replay_indices],
                teacher_targets[1][replay_indices],
            )
            try:
                result = train_update(student, optimizer, train_batch, replay_batch, replay_teacher_targets, device, step + 1, source, arm["head_weight"], arm["representation_weight"], loss_scale, args.max_nonfinite_retries)
            except NonFiniteTrainingError as exc:
                write_json(output / "NONFINITE_ABORT.json", {"schema_version": SCHEMA, "step": exc.step, "source": exc.source, "reason": exc.reason, "retries": exc.retries, "last_known_good_checkpoint": f"step_{max(start_step, exc.step - 1):06d}"})
                raise
            except torch.cuda.OutOfMemoryError as exc:
                write_json(output / "RESOURCE_ABORT.json", {"schema_version": SCHEMA, "phase": "training_update", "step": step + 1, "source": source, "resource": "cuda_oom", "error": str(exc), "last_known_good_checkpoint": str(args.checkpoint.resolve())})
                raise
            loss_scale = float(result["loss_scale"])
            if result["loss_scale_retries"]:
                with safety_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step + 1, "loss_scale": loss_scale, "retries": result["loss_scale_retries"]}, sort_keys=True) + "\n")
            training_tokens += int(result["valid_tokens"])
            if (step + 1) % max(1, args.eval_every) == 0 or step == start_step:
                metrics = evaluate_loss(student, language_validation, reasoning_validation, device, args.batch_size, args.eval_rows)
                record = {"step": step + 1, "source": source, "training_tokens": training_tokens, **result, **metrics}
                updates.append(record)
                curve.write(json.dumps(record, sort_keys=True) + "\n")
                curve.flush()
            if (step + 1) % max(1, args.checkpoint_every) == 0 or step + 1 == args.steps:
                save_checkpoint(checkpoint_root / f"crucible_120m_retention_{args.arm}_step_{step + 1:06d}.pt", student, optimizer, step + 1, args.seed, training_tokens, loss_scale)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": SCHEMA,
        "classification": "exploratory_retention_factorial_arm",
        "arm": {"name": args.arm, **arm},
        "start_checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256_file(args.checkpoint), "step": start_step},
        "model": {"parameter_count": sum(parameter.numel() for parameter in student.parameters()), "state_sha256": state_digest(student)},
        "training": {"start_step": start_step, "steps_completed": args.steps, "steps_this_run": args.steps - start_step, "training_tokens_start": run_start_tokens, "training_tokens_this_run": training_tokens - run_start_tokens, "training_tokens_recorded": training_tokens, "elapsed_seconds": elapsed, "tokens_per_second": (training_tokens - run_start_tokens) / max(elapsed, 1e-9), "precision": "bf16", "micro_batch_size": args.batch_size, "effective_batch_size": args.batch_size, "reasoning_every": args.reasoning_every, "learning_rate": args.learning_rate, "loss_scale": loss_scale, "peak_vram_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, "peak_vram_reserved": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None},
        "data": {"fineweb_manifest_sha256": sha256_file(fineweb_root / "MANIFEST.json"), "fineweb_train_sha256": sha256_file(fineweb_root / fineweb_manifest["packed_files"]["train"]["path"]), "reasoning_train_sha256": sha256_file(reasoning_root / "train.jsonl"), "reasoning_validation_sha256": sha256_file(reasoning_root / "validation.jsonl"), "replay_records": len(replay_ids), "replay_record_ids_sha256": hashlib.sha256(json.dumps(replay_ids, sort_keys=True).encode()).hexdigest(), "teacher_targets": "precomputed_bfloat16_cpu_cache", "sealed_test_used_for_training": False},
        "validation": {"checkpoints": updates, "final": updates[-1] if updates else None},
        "safety": {"checkpoint_model_optimizer_finite_before_atomic_commit": True, "nonfinite_policy": "abort_before_optimizer_step", "loss_scale_recovery_events": sum(1 for row in updates if row.get("loss_scale_retries", 0) > 0)},
        "limitations": ["Replay uses reasoning validation data and is not the sealed test set.", "This exploratory run tests retention over a bounded continuation, not lifelong learning.", "The protected competency gate is evaluated after training on the sealed test set by the existing read-only evaluator."],
    }
    write_json(output / "RETENTION_TRAINING_REPORT.json", report)
    (output / "RETENTION_TRAINING_REPORT.md").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "HASHES.json"}
    write_json(output / "HASHES.json", {"schema_version": SCHEMA, "files": hashes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
