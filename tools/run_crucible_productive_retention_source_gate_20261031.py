#!/usr/bin/env python3
"""Run the bounded neural source-competence phase for productive retention.

This script is intentionally narrower than the future C0--C3 experiment.  It
loads the verified step-15,000 Crucible checkpoint, trains only on the public
source-train split for 1,000 updates, evaluates only the public source-
validation split, and then stops.  It never reads target manifests or private
sealed answers.
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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
THROUGHPUT = ROOT / "experiments" / "crucible_heterogeneous_training_throughput_20261029"
sys.path.insert(0, str(THROUGHPUT))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark import make_model, parameter_groups, state_digest  # noqa: E402
from run_crucible_foundation_training_20261029 import foundation_model_config  # noqa: E402


SCHEMA = "crucible-productive-retention-source-gate-v1-20261031"
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
RAW_BYTE_START = 3
CONTEXT = 512
SOURCE_UPDATES = 1000
SOURCE_BATCH_SIZE = 8
SOURCE_GATE_MIN_CORRECT = 13
SOURCE_GATE_TRIALS = 16
SOURCE_GATE_THRESHOLD = 0.80
CHECKPOINT_UPDATES = (250, 500, 750, 1000)
DEFAULT_CHECKPOINT = Path("/home/timbushnell/crucible_120m_checkpoints_20261029/crucible_120m_foundation_step_015000.pt")
DEFAULT_MANIFEST = ROOT / "experiments" / "crucible_productive_retention_transfer_20261031" / "manifest"
DEFAULT_OUTPUT = ROOT / "experiments" / "crucible_productive_retention_transfer_20261031" / "source_gate"
DEFAULT_CHECKPOINT_ROOT = Path("/home/timbushnell/crucible_productive_retention_source_gate_checkpoints_20261031")


class SourceGateError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_model(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter.detach()).all().item()) for parameter in model.parameters())


def finite_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and (torch.is_floating_point(value) or torch.is_complex(value)):
                if not bool(torch.isfinite(value).all().item()):
                    return False
    return True


def format_prompt(row: Mapping[str, Any]) -> str:
    lines = ["BOOLEAN_FUNCTION\n", "DEMONSTRATIONS:\n"]
    for example in row["demonstrations"]:
        values = ",".join(str(int(value)) for value in example["input"])
        lines.append(f"INPUT={values};OUTPUT={int(example['output'])}\n")
    query = ",".join(str(int(value)) for value in row["query"])
    lines.append(f"QUERY={query};ANSWER=")
    return "".join(lines)


def encode_training_row(row: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if "query_output" not in row:
        raise SourceGateError(f"source training row has no public answer: {row['task_id']}")
    prompt_bytes = format_prompt(row).encode("utf-8")
    answer_bytes = (str(int(row["query_output"])) + "\n").encode("utf-8")
    combined = prompt_bytes + answer_bytes
    if len(combined) + 1 > CONTEXT:
        raise SourceGateError(f"source episode exceeds context: {row['task_id']}")
    x = torch.zeros(CONTEXT, dtype=torch.long)
    y = torch.zeros(CONTEXT, dtype=torch.long)
    tokens = torch.tensor([RAW_BYTE_START + value for value in combined], dtype=torch.long)
    x[0] = BOS_ID
    x[1 : 1 + len(tokens)] = tokens
    y[: len(tokens)] = tokens
    y[len(tokens)] = EOS_ID
    y[: len(prompt_bytes)] = PAD_ID
    return x, y, len(answer_bytes) + 1


def prompt_tensor(row: Mapping[str, Any]) -> torch.Tensor:
    prompt_bytes = format_prompt(row).encode("utf-8")
    if len(prompt_bytes) + 1 >= CONTEXT:
        raise SourceGateError(f"source validation prompt exceeds context: {row['task_id']}")
    tokens = torch.tensor([BOS_ID] + [RAW_BYTE_START + value for value in prompt_bytes], dtype=torch.long)
    return tokens


@torch.no_grad()
def predict_answer(model: torch.nn.Module, row: Mapping[str, Any], device: torch.device) -> Tuple[str, str]:
    model.eval()
    prefix = prompt_tensor(row).tolist()
    generated: List[int] = []
    for _ in range(8):
        if len(prefix) >= CONTEXT:
            break
        x = torch.tensor([prefix], dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        if not bool(torch.isfinite(logits).all().item()):
            raise SourceGateError(f"non-finite evaluation logits: {row['task_id']}")
        token = int(torch.argmax(logits[0, -1]).item())
        if token == EOS_ID:
            break
        if RAW_BYTE_START <= token < RAW_BYTE_START + 256:
            generated.append(token - RAW_BYTE_START)
        prefix.append(token)
    raw = bytes(generated).decode("utf-8", errors="ignore")
    prediction = next((char for char in raw if char in "01"), "")
    return prediction, raw


@torch.no_grad()
def evaluate_source(model: torch.nn.Module, rows: Sequence[Mapping[str, Any]], device: torch.device) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for row in rows:
        prediction, raw = predict_answer(model, row, device)
        expected = str(int(row["query_output"]))
        results.append({
            "task_id": row["task_id"],
            "prediction": prediction,
            "raw_generation": raw,
            "expected": expected,
            "correct": prediction == expected,
        })
    correct = sum(int(row["correct"]) for row in results)
    return {
        "correct": correct,
        "trials": len(results),
        "accuracy": correct / max(1, len(results)),
        "gate_pass": correct >= SOURCE_GATE_MIN_CORRECT,
        "rows": results,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    seed: int,
    update: int,
    training_examples: int,
    base_checkpoint_sha256: str,
    source_train_sha256: str,
    source_validation_sha256: str,
) -> None:
    if not finite_model(model) or not finite_optimizer(optimizer):
        raise SourceGateError(f"refusing to save non-finite source checkpoint at update {update}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA,
        "phase": "source_competence_only",
        "seed": seed,
        "update": update,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "training_examples": training_examples,
        "source_train_sha256": source_train_sha256,
        "source_validation_sha256": source_validation_sha256,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_seed(
    seed: int,
    base_checkpoint: Path,
    manifest_dir: Path,
    output_dir: Path,
    checkpoint_root: Path,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)

    manifest = read_json(manifest_dir / "MANIFEST.json")
    if manifest.get("scientific_authorization") is not False:
        raise SourceGateError("manifest authorization flag is not false")
    train_path = manifest_dir / f"seed_{seed}" / "source_train.jsonl"
    validation_path = manifest_dir / f"seed_{seed}" / "source_validation.jsonl"
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    if len(train_rows) != 32 or len(validation_rows) != 16:
        raise SourceGateError("source split counts do not match the frozen manifest")
    train_encoded = [encode_training_row(row) for row in train_rows]
    if any("query_output" in row and row["split"].endswith("sealed") for row in train_rows + validation_rows):
        raise SourceGateError("source gate encountered a sealed row")

    base_hash = sha256_file(base_checkpoint)
    checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != 15000:
        raise SourceGateError("source gate requires the step-15,000 checkpoint")
    cfg = foundation_model_config("120m", False)
    model = make_model(cfg, device, seed)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW(parameter_groups(model, 0.01), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    if not finite_model(model) or not finite_optimizer(optimizer):
        raise SourceGateError("base checkpoint is non-finite")

    rng = random.Random(seed * 1000003 + 271828)
    schedule: List[int] = []
    for _ in range(SOURCE_UPDATES):
        batch_indices = list(range(len(train_encoded)))
        rng.shuffle(batch_indices)
        schedule.extend(batch_indices[:SOURCE_BATCH_SIZE])

    seed_output = output_dir / f"seed_{seed}"
    seed_output.mkdir(parents=True, exist_ok=True)
    curve_path = seed_output / "SOURCE_GATE_CURVE.jsonl"
    validation_records: List[Dict[str, Any]] = []
    training_examples = 0
    started = time.perf_counter()
    schedule_position = 0
    with curve_path.open("w", encoding="utf-8") as curve:
        for update in range(1, SOURCE_UPDATES + 1):
            optimizer.zero_grad(set_to_none=True)
            indices = schedule[schedule_position : schedule_position + SOURCE_BATCH_SIZE]
            schedule_position += SOURCE_BATCH_SIZE
            x = torch.stack([train_encoded[index][0] for index in indices]).to(device, non_blocking=True)
            y = torch.stack([train_encoded[index][1] for index in indices]).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
                if not bool(torch.isfinite(logits).all().item()):
                    raise SourceGateError(f"non-finite logits at seed {seed}, update {update}")
                loss_values = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
                mask = y.reshape(-1) != PAD_ID
                loss = loss_values[mask].mean()
            if not bool(torch.isfinite(loss).item()):
                raise SourceGateError(f"non-finite loss at seed {seed}, update {update}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise SourceGateError(f"non-finite gradient norm at seed {seed}, update {update}")
            optimizer.step()
            if not finite_model(model) or not finite_optimizer(optimizer):
                raise SourceGateError(f"non-finite committed state at seed {seed}, update {update}")
            training_examples += len(indices)

            if update in CHECKPOINT_UPDATES:
                evaluation = evaluate_source(model, validation_rows, device)
                record = {
                    "seed": seed,
                    "update": update,
                    "training_examples": training_examples,
                    "loss": float(loss.detach().cpu().item()),
                    "gradient_norm": float(gradient_norm.detach().cpu().item()),
                    "source_validation_correct": evaluation["correct"],
                    "source_validation_accuracy": evaluation["accuracy"],
                    "source_gate_pass": evaluation["gate_pass"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                curve.write(json.dumps(record, sort_keys=True) + "\n")
                curve.flush()
                validation_records.append({"update": update, "evaluation": evaluation})
                save_checkpoint(
                    checkpoint_root / f"crucible_120m_source_gate_seed{seed}_update_{update:06d}.pt",
                    model,
                    optimizer,
                    seed,
                    update,
                    training_examples,
                    base_hash,
                    sha256_file(train_path),
                    sha256_file(validation_path),
                )

    final_eval = validation_records[-1]["evaluation"]
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": SCHEMA,
        "classification": "bounded_source_competence_phase_only",
        "scientific_authorization": False,
        "source_gate_authorized": True,
        "target_training_started": False,
        "sealed_target_evaluation_started": False,
        "seed": seed,
        "base_checkpoint": {
            "path": "<step-15000-checkpoint>",
            "sha256": base_hash,
            "step": int(checkpoint["step"]),
            "training_tokens": int(checkpoint.get("training_tokens", 0)),
        },
        "model": {"parameter_count": sum(parameter.numel() for parameter in model.parameters()), "state_digest": state_digest(model)},
        "source_training": {
            "updates": SOURCE_UPDATES,
            "batch_size": SOURCE_BATCH_SIZE,
            "examples": training_examples,
            "learning_rate": learning_rate,
            "precision": "bf16",
            "context": CONTEXT,
            "elapsed_seconds": elapsed,
            "examples_per_second": training_examples / max(elapsed, 1e-9),
            "peak_vram_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_vram_reserved": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        },
        "data": {
            "manifest_sha256": sha256_file(manifest_dir / "MANIFEST.json"),
            "source_train_sha256": sha256_file(train_path),
            "source_validation_sha256": sha256_file(validation_path),
            "target_data_loaded": False,
            "sealed_answer_files_loaded": False,
        },
        "source_gate": {
            "threshold": SOURCE_GATE_THRESHOLD,
            "minimum_correct": SOURCE_GATE_MIN_CORRECT,
            "trials": SOURCE_GATE_TRIALS,
            "final": final_eval,
            "trajectory": validation_records,
            "pass": bool(final_eval["correct"] >= SOURCE_GATE_MIN_CORRECT),
        },
        "safety": {
            "finite_model_and_optimizer_at_each_update": True,
            "checkpoint_integrity_checked_before_save": True,
            "hard_stop_after_update_1000": True,
        },
        "limitations": [
            "This phase tests source competence only; it does not test C0--C3 productive transfer.",
            "The Boolean prompt encoding and source fine-tuning settings are recorded for reproducibility and must remain fixed for interpretation.",
        ],
    }
    write_json(seed_output / "SOURCE_GATE_REPORT.json", report)
    (seed_output / "SOURCE_GATE_REPORT.md").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SourceGateError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise SourceGateError("BF16 is not supported by the active CUDA device")
    checkpoint = args.checkpoint.resolve()
    manifest_dir = args.manifest_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    if not checkpoint.is_file():
        raise SourceGateError(f"missing base checkpoint: {checkpoint}")
    reports = [
        train_seed(seed, checkpoint, manifest_dir, output_dir, checkpoint_root, args.learning_rate, device)
        for seed in (42, 1337)
    ]
    summary = {
        "schema_version": SCHEMA,
        "status": "completed",
        "classification": "bounded_source_competence_phase_only",
        "scientific_authorization": False,
        "source_gate_pass_both_seeds": all(report["source_gate"]["pass"] for report in reports),
        "seeds": reports,
        "target_training_started": False,
        "sealed_target_evaluation_started": False,
    }
    write_json(output_dir / "SOURCE_GATE_SUMMARY.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "source_gate_pass_both_seeds": summary["source_gate_pass_both_seeds"],
        "seed_results": [
            {"seed": report["seed"], "correct": report["source_gate"]["final"]["correct"], "trials": report["source_gate"]["final"]["trials"]}
            for report in reports
        ],
        "target_training_started": False,
        "sealed_target_evaluation_started": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
