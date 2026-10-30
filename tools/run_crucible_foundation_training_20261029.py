#!/usr/bin/env python3
"""Run a resumable local Crucible foundation-training experiment.

This is a substantive local baseline, not the 500M run and not a Track-B
experiment.  It uses the existing dense Crucible latent-attention + SwiGLU
model, a packed FineWeb-Edu stream, and the five prepared reasoning sources.
The OP controller is evaluated separately after a competent checkpoint exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
THROUGHPUT = ROOT / "experiments" / "crucible_heterogeneous_training_throughput_20261029"
if str(THROUGHPUT) not in sys.path:
    sys.path.insert(0, str(THROUGHPUT))
from benchmark import LOSS_SCALE, loss_sum, make_model, model_config, parameter_groups, state_digest, unscale_grads  # noqa: E402


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
RAW_BYTE_START = 3
SCHEMA = "crucible-foundation-training-v1-20261029"


def foundation_model_config(name: str, activation_checkpointing: bool) -> Dict[str, int]:
    if name == "120m":
        return {
            "vocab_size": 259,
            "n_layers": 14,
            "d_model": 768,
            "n_heads": 12,
            "latent_dim_kv": 192,
            "latent_dim_q": 192,
            "pe_dim": 64,
            "d_ff": 3072,
            "activation_checkpointing": int(bool(activation_checkpointing)),
        }
    cfg = model_config(name)
    cfg["activation_checkpointing"] = int(bool(activation_checkpointing))
    return cfg


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class NonFiniteTrainingError(RuntimeError):
    """Raised when a training update cannot be safely committed."""

    def __init__(self, step: int, source: str, reason: str, retries: int = 0):
        super().__init__(f"non-finite training state at step {step} ({source}): {reason}")
        self.step = int(step)
        self.source = str(source)
        self.reason = str(reason)
        self.retries = int(retries)


def nonfinite_model_parameter(model: torch.nn.Module) -> Optional[str]:
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            return name
    return None


def nonfinite_optimizer_state(optimizer: torch.optim.Optimizer) -> Optional[str]:
    for parameter, state in optimizer.state.items():
        parameter_name = f"parameter:{id(parameter)}"
        for name, value in state.items():
            if not torch.is_tensor(value):
                continue
            if not (torch.is_floating_point(value) or torch.is_complex(value)):
                continue
            if not bool(torch.isfinite(value).all().item()):
                return f"{parameter_name}/{name}"
    return None


def finite_state_issue(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Optional[str]:
    parameter = nonfinite_model_parameter(model)
    if parameter is not None:
        return f"model:{parameter}"
    optimizer_state = nonfinite_optimizer_state(optimizer)
    if optimizer_state is not None:
        return f"optimizer:{optimizer_state}"
    return None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_math_answer(value: str) -> str:
    import re
    matches = re.findall(r"\\boxed\{([^{}]*)\}", str(value))
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    return lines[-1] if lines else str(value).strip()


def answer_for_record(record: Mapping[str, Any]) -> str:
    target = str(record.get("target", ""))
    if record.get("domain") == "math":
        target = extract_math_answer(target)
    raw = target.encode("utf-8")[:128]
    return raw.decode("utf-8", errors="ignore")


def prompt_for_record(record: Mapping[str, Any]) -> str:
    raw = f"DOMAIN={record.get('domain', 'unknown')}\nTASK={record.get('input', '')}\nANSWER=".encode("utf-8")[:192]
    return raw.decode("utf-8", errors="ignore")


def encode_reasoning(record: Mapping[str, Any], context: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    prompt = prompt_for_record(record)
    answer = answer_for_record(record) + "\n"
    prompt_bytes = prompt.encode("utf-8")
    answer_bytes = answer.encode("utf-8")
    combined = prompt_bytes + answer_bytes
    if len(combined) + 1 > context:
        answer_bytes = answer_bytes[: max(1, context - len(prompt_bytes) - 1)]
        answer_bytes = answer_bytes.decode("utf-8", errors="ignore").encode("utf-8")
        combined = prompt_bytes + answer_bytes
    if len(combined) + 1 > context or not answer_bytes:
        return None
    tokens = torch.tensor([RAW_BYTE_START + value for value in combined], dtype=torch.long)
    x = torch.zeros(context, dtype=torch.long)
    y = torch.zeros(context, dtype=torch.long)
    x[0] = BOS_ID
    x[1 : 1 + len(tokens)] = tokens
    y[: len(tokens)] = tokens
    y[len(tokens)] = EOS_ID
    y[: len(prompt_bytes)] = PAD_ID
    return x, y


def load_packed(path: Path, shape: Sequence[int]) -> torch.Tensor:
    rows, width = map(int, shape)
    array = np.memmap(path, mode="r", dtype=np.uint16, shape=(rows, width))
    tensor = torch.from_numpy(np.asarray(array).copy()).long()
    return tensor


def batch_from_rows(rows: torch.Tensor, indices: Sequence[int], context: int) -> Tuple[torch.Tensor, torch.Tensor]:
    selected = rows[torch.tensor(list(indices), dtype=torch.long)]
    return selected[:, :context].contiguous(), selected[:, 1 : context + 1].contiguous()


def reasoning_batch(samples: Sequence[Tuple[torch.Tensor, torch.Tensor]], indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.stack([samples[int(index)][0] for index in indices]),
        torch.stack([samples[int(index)][1] for index in indices]),
    )


def normalize_answer(value: str) -> str:
    import re
    value = str(value).lower().strip()
    value = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", value)
    return re.sub(r"\s+", "", value).strip(".,;:")


@torch.no_grad()
def generate(model: torch.nn.Module, prompt: str, device: torch.device, context: int, max_tokens: int = 96) -> str:
    model.eval()
    prefix = [BOS_ID] + [RAW_BYTE_START + value for value in prompt.encode("utf-8")]
    generated: List[int] = []
    for _ in range(max_tokens):
        if len(prefix) >= context:
            break
        tokens = torch.tensor([prefix], dtype=torch.long, device=device)
        logits = model(tokens)
        token = int(torch.argmax(logits[0, -1]).item())
        if token == EOS_ID or token < RAW_BYTE_START or token >= RAW_BYTE_START + 256:
            break
        generated.append(token - RAW_BYTE_START)
        prefix.append(token)
    model.train()
    return bytes(generated).decode("utf-8", errors="ignore").strip()


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, rows: torch.Tensor, samples: Sequence[Tuple[torch.Tensor, torch.Tensor]], device: torch.device, batch_size: int, max_rows: int) -> Dict[str, float]:
    model.eval()
    total_language = 0.0
    language_tokens = 0
    for start in range(0, min(max_rows, rows.size(0)), batch_size):
        x, y = batch_from_rows(rows, range(start, min(start + batch_size, min(max_rows, rows.size(0)))), rows.size(1) - 1)
        logits = model(x.to(device))
        ce_sum, valid = loss_sum(logits, y.to(device))
        total_language += float(ce_sum.cpu().item())
        language_tokens += valid
    total_reasoning = 0.0
    reasoning_tokens = 0
    for start in range(0, min(max_rows, len(samples)), batch_size):
        x, y = reasoning_batch(samples, range(start, min(start + batch_size, min(max_rows, len(samples)))))
        logits = model(x.to(device))
        ce_sum, valid = loss_sum(logits, y.to(device))
        total_reasoning += float(ce_sum.cpu().item())
        reasoning_tokens += valid
    model.train()
    return {
        "language_loss": total_language / max(1, language_tokens),
        "reasoning_loss": total_reasoning / max(1, reasoning_tokens),
        "language_tokens": language_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def train_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    step: int,
    source: str,
    loss_scale: float,
    max_nonfinite_retries: int,
    precision: str,
) -> Dict[str, Any]:
    """Run one update and never commit a non-finite optimizer step."""
    scale = float(max(1, len(batches)))
    current_loss_scale = float(max(1.0, loss_scale))
    for attempt in range(max(0, int(max_nonfinite_retries)) + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_valid = 0
        forward_failure: Optional[str] = None
        for x_cpu, y_cpu in batches:
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
            with torch.autocast(
                "cuda",
                dtype=autocast_dtype,
                enabled=device.type == "cuda" and precision != "fp32",
            ):
                logits = model(x)
                if not bool(torch.isfinite(logits).all().item()):
                    forward_failure = "nonfinite_logits"
                    break
                ce_sum, valid = loss_sum(logits, y)
                loss = ce_sum / float(max(1, valid))
            if not bool(torch.isfinite(loss).all().item()):
                forward_failure = "nonfinite_loss"
                break
            (loss * current_loss_scale / scale).backward()
            total_loss += float(loss.detach().cpu().item()) * valid
            total_valid += int(valid)
        if forward_failure is not None:
            raise NonFiniteTrainingError(step, source, forward_failure, attempt)

        finite = unscale_grads(model, current_loss_scale)
        if not finite:
            if attempt < max_nonfinite_retries and current_loss_scale > 1.0:
                current_loss_scale = max(1.0, current_loss_scale / 2.0)
                continue
            raise NonFiniteTrainingError(step, source, "nonfinite_gradients", attempt)

        try:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
        except RuntimeError as exc:
            if attempt < max_nonfinite_retries and current_loss_scale > 1.0:
                current_loss_scale = max(1.0, current_loss_scale / 2.0)
                continue
            raise NonFiniteTrainingError(step, source, f"gradient_clip:{exc}", attempt) from exc
        if not bool(torch.isfinite(gradient_norm).all().item()):
            if attempt < max_nonfinite_retries and current_loss_scale > 1.0:
                current_loss_scale = max(1.0, current_loss_scale / 2.0)
                continue
            raise NonFiniteTrainingError(step, source, "nonfinite_gradient_norm", attempt)

        optimizer.step()
        bad_parameter = nonfinite_model_parameter(model)
        if bad_parameter is not None:
            raise NonFiniteTrainingError(step, source, f"optimizer_step:model:{bad_parameter}", attempt)
        final_loss = total_loss / max(1, total_valid)
        if not math.isfinite(final_loss):
            raise NonFiniteTrainingError(step, source, "nonfinite_reduced_loss", attempt)
        return {
            "loss": final_loss,
            "valid_tokens": total_valid,
            "finite_gradients": True,
            "gradient_norm": float(gradient_norm.detach().cpu().item()),
            "loss_scale": current_loss_scale,
            "loss_scale_retries": attempt,
        }
    raise NonFiniteTrainingError(step, source, "retry_exhausted", max_nonfinite_retries)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    seed: int,
    training_tokens: int,
    loss_scale: float,
) -> None:
    issue = finite_state_issue(model, optimizer)
    if issue is not None:
        raise RuntimeError(f"refusing to save non-finite checkpoint at step {step}: {issue}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA,
        "step": step,
        "seed": seed,
        "training_tokens": training_tokens,
        "loss_scale": float(loss_scale),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb-root", type=Path, default=Path("/home/timbushnell/crucible_fineweb_foundation_20261029"))
    parser.add_argument("--reasoning-root", type=Path, default=Path("/home/timbushnell/crucible_reasoning_pilot_20261029"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/home/timbushnell/crucible_foundation_checkpoints_20261029"))
    parser.add_argument("--model-size", choices=("tiny", "small", "120m"), default="small")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--reasoning-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-rows", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--initial-loss-scale", type=float, default=LOSS_SCALE)
    parser.add_argument("--max-nonfinite-retries", type=int, default=4)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with host access or --device cpu")
    device = torch.device(args.device)
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by the active CUDA device")
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
    if context != args.context:
        raise ValueError(f"packed context {context} does not match requested context {args.context}")
    language_train = load_packed(fineweb_root / fineweb_manifest["packed_files"]["train"]["path"], fineweb_manifest["packed_files"]["train"]["shape"])
    language_validation = load_packed(fineweb_root / fineweb_manifest["packed_files"]["validation"]["path"], fineweb_manifest["packed_files"]["validation"]["shape"])
    reasoning_root = args.reasoning_root.resolve()
    reasoning_train_rows = load_jsonl(reasoning_root / "train.jsonl")
    reasoning_validation_rows = load_jsonl(reasoning_root / "validation.jsonl")
    reasoning_train = [sample for row in reasoning_train_rows if (sample := encode_reasoning(row, context)) is not None]
    reasoning_validation = [sample for row in reasoning_validation_rows if (sample := encode_reasoning(row, context)) is not None]
    if not reasoning_train or not reasoning_validation:
        raise RuntimeError("reasoning preparation produced no usable samples")

    cfg = foundation_model_config(args.model_size, args.activation_checkpointing)
    model = make_model(cfg, device, args.seed)
    optimizer = torch.optim.AdamW(parameter_groups(model, 0.01), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    start_step = 0
    resumed_training_tokens = 0
    resumed_loss_scale: Optional[float] = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        resumed_training_tokens = int(checkpoint.get("training_tokens", 0))
        if checkpoint.get("loss_scale") is not None:
            resumed_loss_scale = float(checkpoint["loss_scale"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
            cuda_states = [state.detach().cpu().to(dtype=torch.uint8) for state in checkpoint["cuda_rng_state_all"]]
            torch.cuda.set_rng_state_all(cuda_states)
    checkpoint_issue = finite_state_issue(model, optimizer)
    if checkpoint_issue is not None:
        raise RuntimeError(f"resume checkpoint is non-finite: {checkpoint_issue}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.checkpoint_root.resolve()
    curve_path = output / "TRAINING_CURVE.jsonl"
    mode = "a" if start_step else "w"
    started = time.perf_counter()
    training_tokens = resumed_training_tokens
    loss_scale = float(max(1.0, resumed_loss_scale if resumed_loss_scale is not None else args.initial_loss_scale))
    recovery_events: List[Dict[str, Any]] = []
    safety_log_path = output / "SAFETY_EVENTS.jsonl"
    if start_step and training_tokens == 0 and curve_path.exists():
        prior_records = [line for line in curve_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if prior_records:
            training_tokens = int(json.loads(prior_records[-1])["training_tokens"])
    updates = []
    run_start_training_tokens = training_tokens
    with curve_path.open(mode, encoding="utf-8") as curve:
        for step in range(start_step, args.steps):
            is_reasoning = (step + 1) % max(1, args.reasoning_every) == 0
            batches = []
            for micro in range(max(1, args.grad_accumulation)):
                if is_reasoning:
                    indices = [((step // max(1, args.reasoning_every)) * args.batch_size * max(1, args.grad_accumulation) + micro * args.batch_size + i) % len(reasoning_train) for i in range(args.batch_size)]
                    batches.append(reasoning_batch(reasoning_train, indices))
                    source = "reasoning"
                else:
                    indices = [(step * args.batch_size * max(1, args.grad_accumulation) + micro * args.batch_size + i) % language_train.size(0) for i in range(args.batch_size)]
                    batches.append(batch_from_rows(language_train, indices, context))
                    source = "fineweb_edu"
            try:
                result = train_update(
                    model,
                    optimizer,
                    batches,
                    device,
                    step + 1,
                    source,
                    loss_scale,
                    args.max_nonfinite_retries,
                    args.precision,
                )
            except NonFiniteTrainingError as exc:
                write_json(output / "NONFINITE_ABORT.json", {
                    "schema_version": SCHEMA,
                    "step": exc.step,
                    "source": exc.source,
                    "reason": exc.reason,
                    "retries": exc.retries,
                    "last_known_good_checkpoint": f"step_{max(start_step, exc.step - 1):06d}",
                })
                raise
            except torch.cuda.OutOfMemoryError as exc:
                write_json(output / "RESOURCE_ABORT.json", {
                    "schema_version": SCHEMA,
                    "step": step + 1,
                    "source": source,
                    "resource": "cuda_oom",
                    "message": str(exc),
                    "last_known_good_checkpoint": f"step_{max(start_step, step):06d}",
                    "micro_batch_size": args.batch_size,
                    "grad_accumulation": args.grad_accumulation,
                    "precision": args.precision,
                })
                raise
            loss_scale = float(result["loss_scale"])
            if int(result["loss_scale_retries"]) > 0:
                event = {
                    "step": step + 1,
                    "source": source,
                    "retries": int(result["loss_scale_retries"]),
                    "loss_scale": loss_scale,
                }
                recovery_events.append(event)
                with safety_log_path.open("a", encoding="utf-8") as safety_log:
                    safety_log.write(json.dumps(event, sort_keys=True) + "\n")
            training_tokens += int(result["valid_tokens"])
            record: Dict[str, Any] = {"step": step + 1, "source": source, **result, "training_tokens": training_tokens}
            if (step + 1) % max(1, args.eval_every) == 0 or step == start_step:
                metrics = evaluate_loss(model, language_validation, reasoning_validation, device, args.batch_size, args.eval_rows)
                record.update(metrics)
                updates.append(record)
                curve.write(json.dumps(record, sort_keys=True) + "\n")
                curve.flush()
            if (step + 1) % max(1, args.checkpoint_every) == 0 or step + 1 == args.steps:
                checkpoint_path = checkpoint_root / f"crucible_{args.model_size}_foundation_step_{step + 1:06d}.pt"
                save_checkpoint(checkpoint_path, model, optimizer, step + 1, args.seed, training_tokens, loss_scale)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    run_training_tokens = training_tokens - run_start_training_tokens
    run_steps = args.steps - start_step

    prompts = [
        "The goal of science is",
        "In mathematics, a prime number is",
        "A good explanation should",
        "Once upon a time",
    ]
    generations = [{"prompt": prompt, "completion": generate(model, prompt, device, context)} for prompt in prompts]
    report = {
        "schema_version": SCHEMA,
        "classification": "substantive_local_crucible_foundation_baseline",
        "scientific_status": "exploratory_not_500m_not_track_b",
        "model": {"family": "Crucible latent-attention + SwiGLU dense", "size": args.model_size, "config": cfg, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "final_state_sha256": state_digest(model)},
        "training": {"steps_completed": args.steps, "steps_this_run": run_steps, "resumed_from_step": start_step, "micro_batch_size": args.batch_size, "grad_accumulation": args.grad_accumulation, "effective_batch_size": args.batch_size * args.grad_accumulation, "context": context, "reasoning_every": args.reasoning_every, "seed": args.seed, "learning_rate": args.learning_rate, "precision": args.precision, "initial_loss_scale": args.initial_loss_scale, "starting_loss_scale": loss_scale if start_step else args.initial_loss_scale, "final_loss_scale": loss_scale, "loss_scale_recovery_events": len(recovery_events), "max_nonfinite_retries": args.max_nonfinite_retries, "checkpoint_validation": "model_and_optimizer_state_finite_before_atomic_commit", "nonfinite_policy": "abort_before_optimizer_step_after_retry_budget", "training_tokens_start": run_start_training_tokens, "training_tokens_this_run": run_training_tokens, "training_tokens_recorded": training_tokens, "elapsed_seconds": elapsed, "tokens_per_second": run_training_tokens / max(elapsed, 1e-9), "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "peak_vram_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, "peak_vram_reserved": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None},
        "data": {"fineweb_manifest_sha256": sha256_file(fineweb_root / "MANIFEST.json"), "fineweb_train_sha256": sha256_file(fineweb_root / fineweb_manifest["packed_files"]["train"]["path"]), "fineweb_validation_sha256": sha256_file(fineweb_root / fineweb_manifest["packed_files"]["validation"]["path"]), "reasoning_train_sha256": sha256_file(reasoning_root / "train.jsonl"), "reasoning_validation_sha256": sha256_file(reasoning_root / "validation.jsonl"), "raw_fineweb_committed_to_git": False, "test_partition_used_for_training": False},
        "validation": {"checkpoints": updates, "final": updates[-1] if updates else None},
        "generation_samples": generations,
        "op_status": "not run in foundation trainer; compare OP only after baseline competence is demonstrated",
        "checkpoints": {"external_root": "<local-checkpoint-root>", "last_step": args.steps},
        "limitations": [
            "This is an intermediate model, not the planned 500M model.",
            "The FineWeb-Edu source is one bounded shard and the reasoning sources are capped pilot subsets.",
            "The language source is now FineWeb-Edu; TinyStories is not used by this run.",
            "The exact OP loop is not credited here; it remains a separate external-controller evaluation.",
            "Cumulative learning, retention and transfer are not measured by this run.",
        ],
    }
    write_json(output / "FOUNDATION_TRAINING_REPORT.json", report)
    (output / "FOUNDATION_TRAINING_REPORT.md").write_text(
        "# Crucible substantive local foundation baseline\n\n"
        "Classification: `substantive_local_crucible_foundation_baseline`\n\n"
        f"Model: `{args.model_size}`, `{report['model']['parameter_count']:,}` parameters; steps: `{args.steps}`; context: `{context}`.\n\n"
        f"Training tokens this run: `{run_training_tokens:,}`; cumulative recorded: `{training_tokens:,}`; elapsed: `{elapsed:.2f}` s; throughput: `{report['training']['tokens_per_second']:.1f}` tokens/s.\n\n"
        f"Final validation record: `{json.dumps(report['validation']['final'], sort_keys=True)}`\n\n"
        "OP was not scored in this run because the foundation baseline must first demonstrate competence.\n",
        encoding="utf-8",
    )
    hashes = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "HASHES.json"}
    write_json(output / "HASHES.json", {"schema_version": SCHEMA, "files": hashes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
