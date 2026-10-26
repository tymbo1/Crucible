#!/usr/bin/env python3
"""CPU-only exposure-control continuation for 128 composition examples."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCALING_PATH = ROOT / "tools" / "run_crucible_composition_scaling_ladder_20261023.py"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_composition_exposure_control_20261023"
SEEDS = (42, 1337)
TRAINING_EXAMPLES = 128
VALIDATION_EXAMPLES = 64
MAX_STEPS = 2_000
CHECKPOINT_STEPS = (0, 500, 1_000, 1_500, 2_000)
HEARTBEAT_STEPS = 100
EOS_ID = 2
RAW_BYTE_START = 3

spec = importlib.util.spec_from_file_location("composition_scaling", SCALING_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import scaling runner: {SCALING_PATH}")
scaling = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scaling
spec.loader.exec_module(scaling)
ladder = scaling.ladder
stage_a = scaling.stage_a


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(dict(record), sort_keys=True) for record in records) + "\n", encoding="utf-8")


def answer_prediction_diagnostics(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    torch = stage_a.torch
    model.eval()
    with torch.no_grad():
        batch = stage_a.batchify(records, list(range(len(records))))
        logits = stage_a.forward_model(model, batch)
        mask = batch["target_mask"] & (batch["y"] != stage_a.PAD_ID)
        answer_mask = mask & (batch["y"] != EOS_ID)
        target_logits = logits[answer_mask, batch["y"][answer_mask]]
        probabilities = torch.softmax(logits[answer_mask], dim=-1)
        target_probabilities = probabilities.gather(1, batch["y"][answer_mask].unsqueeze(1)).squeeze(1)
        ranks = 1 + (logits[answer_mask] > target_logits.unsqueeze(1)).sum(dim=1)
        predicted = logits[answer_mask].argmax(dim=-1).tolist()
    digit_start = RAW_BYTE_START + ord("0")
    digit_end = RAW_BYTE_START + ord("9")
    predicted_digits = [token - digit_start for token in predicted if digit_start <= token <= digit_end]
    return {
        "answer_positions": int(answer_mask.sum().item()),
        "mean_target_probability": float(target_probabilities.mean().item()),
        "mean_target_rank": float(ranks.float().mean().item()),
        "predicted_digit_histogram": {str(digit): predicted_digits.count(digit) for digit in range(10)},
    }


def exact_by_record(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, bool]:
    torch = stage_a.torch
    model.eval()
    with torch.no_grad():
        batch = stage_a.batchify(records, list(range(len(records))))
        logits = stage_a.forward_model(model, batch)
        mask = batch["target_mask"] & (batch["y"] != stage_a.PAD_ID)
        correct = ((logits.argmax(dim=-1) == batch["y"]) | ~mask).all(dim=1).tolist()
    return {str(record["record_id"]): bool(value) for record, value in zip(records, correct)}


def grouped_evaluation(model: Any, prepared: Sequence[Mapping[str, Any]], originals: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aggregate = stage_a.evaluate(model, prepared)
    groups: Dict[str, List[int]] = {}
    for index, record in enumerate(originals):
        pair = "+".join(record["metadata"]["composition"])
        groups.setdefault(pair, []).append(index)
    by_pair = {}
    for pair, indices in sorted(groups.items()):
        subset = [prepared[index] for index in indices]
        by_pair[pair] = stage_a.evaluate(model, subset)
    return {"aggregate": aggregate, "by_operation_pair": by_pair}


def run_seed(seed: int, output: Path) -> Dict[str, Any]:
    torch = stage_a.torch
    full_train = [ladder.make_composition(seed, "train", index) for index in range(TRAINING_EXAMPLES)]
    validation = scaling.make_disjoint_validation(seed, full_train)
    if len(validation) != VALIDATION_EXAMPLES:
        raise RuntimeError(f"validation count mismatch for seed {seed}")
    if {record["input"] for record in full_train} & {record["input"] for record in validation}:
        raise RuntimeError(f"validation overlap for seed {seed}")
    prepared_train = ladder.prepare_records(full_train)
    prepared_validation = ladder.prepare_records(validation)
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    order = list(range(TRAINING_EXAMPLES))
    random.Random(seed + 102_023 + TRAINING_EXAMPLES).shuffle(order)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint_dir / "step_0000.pt", model, optimizer, 0, order)
    rows: List[Dict[str, Any]] = []
    previous_correct: Dict[str, bool] | None = None
    started = time.perf_counter()
    for step in range(MAX_STEPS + 1):
        train_info = None
        if step > 0:
            indices = stage_a.batch_indices(order, step)
            train_info = stage_a.train_step(model, optimizer, prepared_train, indices)
        if step in CHECKPOINT_STEPS:
            train_eval = grouped_evaluation(model, prepared_train, full_train)
            validation_eval = grouped_evaluation(model, prepared_validation, validation)
            current_correct = exact_by_record(model, prepared_train)
            forgotten = 0
            newly_correct = 0
            if previous_correct is not None:
                forgotten = sum(previous_correct[key] and not current_correct[key] for key in current_correct)
                newly_correct = sum(not previous_correct[key] and current_correct[key] for key in current_correct)
            previous_correct = current_correct
            row = {
                "step": step,
                "train_step": train_info,
                "train": train_eval,
                "validation": validation_eval,
                "train_prediction_diagnostics": answer_prediction_diagnostics(model, prepared_train),
                "validation_prediction_diagnostics": answer_prediction_diagnostics(model, prepared_validation),
                "train_forgetting_since_previous_checkpoint": forgotten,
                "train_newly_correct_since_previous_checkpoint": newly_correct,
            }
            rows.append(row)
            print(json.dumps({
                "seed": seed,
                "step": step,
                "train_exact": train_eval["aggregate"]["teacher_forced_exact_accuracy"],
                "validation_exact": validation_eval["aggregate"]["teacher_forced_exact_accuracy"],
                "forgotten": forgotten,
            }, sort_keys=True), flush=True)
            if step in (500, 1_000, 1_500, 2_000):
                stage_a.save_checkpoint(checkpoint_dir / f"step_{step:04d}.pt", model, optimizer, step, order)
        elif step % HEARTBEAT_STEPS == 0:
            print(json.dumps({"seed": seed, "step": step, "status": "training"}, sort_keys=True), flush=True)
    result = {
        "seed": seed,
        "training_examples": TRAINING_EXAMPLES,
        "validation_examples": VALIDATION_EXAMPLES,
        "max_steps": MAX_STEPS,
        "elapsed_seconds": time.perf_counter() - started,
        "operation_pair_counts": scaling.operation_pair_counts(full_train),
        "rows": rows,
    }
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_composition_exposure_control_20261023",
        "status": "cpu_only_dense_control",
        "seed": seed,
        "training_examples": TRAINING_EXAMPLES,
        "validation_examples": VALIDATION_EXAMPLES,
        "max_steps": MAX_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "test_evaluated": False,
        "heldout_evaluated": False,
        "validation_disjoint_from_training": True,
        "continuation": "single fresh initialization; same optimizer and data order through all checkpoints",
    })
    write_json(output / "RESULTS.json", result)
    write_jsonl(output / "train.jsonl", full_train)
    write_jsonl(output / "validation.jsonl", validation)
    (output / "RUNS.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_composition_exposure_control_20261023",
        "status": "cpu_only_dense_control",
        "seeds": list(SEEDS),
        "training_examples": TRAINING_EXAMPLES,
        "validation_examples": VALIDATION_EXAMPLES,
        "max_steps": MAX_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "test_evaluated": False,
        "heldout_evaluated": False,
        "validation_disjoint_from_training": True,
        "continuation": "single fresh initialization per seed; no restarts between exposure checkpoints",
    })
    results = []
    for seed in SEEDS:
        results.append(run_seed(seed, output / f"seed{seed}"))
    write_json(output / "SUMMARY.json", {
        "experiment": "crucible_composition_exposure_control_20261023",
        "status": "completed",
        "test_evaluated": False,
        "heldout_evaluated": False,
        "results": results,
    })
    print(json.dumps({"output": str(output), "seeds": len(results), "status": "completed"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
