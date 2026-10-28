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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def train_update(model: torch.nn.Module, optimizer: torch.optim.Optimizer, x_cpu: torch.Tensor, y_cpu: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    x = x_cpu.to(device, non_blocking=True)
    y = y_cpu.to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(x)
        ce_sum, valid = loss_sum(logits, y)
        loss = ce_sum / float(max(1, valid))
    (loss * LOSS_SCALE).backward()
    finite = unscale_grads(model, LOSS_SCALE)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(loss.detach().cpu().item()), "valid_tokens": int(valid), "finite_gradients": bool(finite)}


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, seed: int, training_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": SCHEMA,
        "step": step,
        "seed": seed,
        "training_tokens": training_tokens,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb-root", type=Path, default=Path("/home/timbushnell/crucible_fineweb_foundation_20261029"))
    parser.add_argument("--reasoning-root", type=Path, default=Path("/home/timbushnell/crucible_reasoning_pilot_20261029"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/home/timbushnell/crucible_foundation_checkpoints_20261029"))
    parser.add_argument("--model-size", choices=("tiny", "small"), default="small")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reasoning-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-rows", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with host access or --device cpu")
    device = torch.device(args.device)
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

    cfg = model_config(args.model_size)
    model = make_model(cfg, device, args.seed)
    optimizer = torch.optim.AdamW(parameter_groups(model, 0.01), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    start_step = 0
    resumed_training_tokens = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        resumed_training_tokens = int(checkpoint.get("training_tokens", 0))
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
            cuda_states = [state.detach().cpu().to(dtype=torch.uint8) for state in checkpoint["cuda_rng_state_all"]]
            torch.cuda.set_rng_state_all(cuda_states)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.checkpoint_root.resolve()
    curve_path = output / "TRAINING_CURVE.jsonl"
    mode = "a" if start_step else "w"
    started = time.perf_counter()
    training_tokens = resumed_training_tokens
    if start_step and training_tokens == 0 and curve_path.exists():
        prior_records = [line for line in curve_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if prior_records:
            training_tokens = int(json.loads(prior_records[-1])["training_tokens"])
    updates = []
    with curve_path.open(mode, encoding="utf-8") as curve:
        for step in range(start_step, args.steps):
            is_reasoning = (step + 1) % max(1, args.reasoning_every) == 0
            if is_reasoning:
                indices = [((step // max(1, args.reasoning_every)) * args.batch_size + i) % len(reasoning_train) for i in range(args.batch_size)]
                x_cpu, y_cpu = reasoning_batch(reasoning_train, indices)
                source = "reasoning"
            else:
                indices = [(step * args.batch_size + i) % language_train.size(0) for i in range(args.batch_size)]
                x_cpu, y_cpu = batch_from_rows(language_train, indices, context)
                source = "fineweb_edu"
            result = train_update(model, optimizer, x_cpu, y_cpu, device)
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
                save_checkpoint(checkpoint_path, model, optimizer, step + 1, args.seed, training_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

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
        "training": {"steps_completed": args.steps, "resumed_from_step": start_step, "batch_size": args.batch_size, "context": context, "reasoning_every": args.reasoning_every, "seed": args.seed, "training_tokens_recorded": training_tokens, "elapsed_seconds": elapsed, "tokens_per_second": training_tokens / max(elapsed, 1e-9), "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "peak_vram_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, "peak_vram_reserved": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None},
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
        f"Training tokens recorded: `{training_tokens:,}`; elapsed: `{elapsed:.2f}` s; throughput: `{report['training']['tokens_per_second']:.1f}` tokens/s.\n\n"
        f"Final validation record: `{json.dumps(report['validation']['final'], sort_keys=True)}`\n\n"
        "OP was not scored in this run because the foundation baseline must first demonstrate competence.\n",
        encoding="utf-8",
    )
    hashes = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "HASHES.json"}
    write_json(output / "HASHES.json", {"schema_version": SCHEMA, "files": hashes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
