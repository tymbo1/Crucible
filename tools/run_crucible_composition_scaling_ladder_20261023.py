#!/usr/bin/env python3
"""CPU-only composition training-set scaling ladder.

The neural architecture and optimizer remain fixed while the number of
training examples increases.  Each arm starts from a fresh dense control.
Only train and validation splits are evaluated; held-out and test splits are
deliberately sealed for this diagnostic.
"""

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
LADDER_PATH = ROOT / "tools" / "run_crucible_cognitive_learnability_ladder_20261022.py"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_composition_scaling_ladder_20261023"
SEEDS = (42, 1337)
SIZES = (8, 16, 32, 64, 128)
STEPS = 500
EVAL_STEPS = (0, 50, 100, 200, 300, 500)
VALIDATION_COUNT = 64
EOS_ID = 2
RAW_BYTE_START = 3

spec = importlib.util.spec_from_file_location("composition_ladder", LADDER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import ladder runner: {LADDER_PATH}")
ladder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ladder
spec.loader.exec_module(ladder)
stage_a = ladder.stage_a


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


def operation_pair_counts(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        operations = tuple(record["metadata"]["composition"])
        key = "+".join(operations)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def make_disjoint_validation(seed: int, full_train: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    train_inputs = {str(record["input"]) for record in full_train}
    validation: List[Dict[str, Any]] = []
    seen_inputs = set()
    pair_counts: Dict[str, int] = {}
    target_per_pair = VALIDATION_COUNT // 2
    candidate_index = 0
    while len(validation) < VALIDATION_COUNT:
        record = ladder.make_composition(seed, "validation", candidate_index)
        candidate_index += 1
        input_text = str(record["input"])
        pair = "+".join(record["metadata"]["composition"])
        if input_text in train_inputs or input_text in seen_inputs:
            continue
        if pair_counts.get(pair, 0) >= target_per_pair:
            continue
        validation.append(record)
        seen_inputs.add(input_text)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if candidate_index > 100_000:
            raise RuntimeError("could not construct disjoint balanced validation set")
    return validation


def fresh_records(seed: int, size: int, validation: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    train = [ladder.make_composition(seed, "train", index) for index in range(size)]
    if size % 2 or operation_pair_counts(train) != {"add+add": size // 2, "add+mul": size // 2}:
        raise RuntimeError(f"unexpected operation coverage for size {size}: {operation_pair_counts(train)}")
    return {"train": train, "validation": [dict(record) for record in validation]}


def run_arm(seed: int, size: int, output: Path, validation: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    torch = stage_a.torch
    records = fresh_records(seed, size, validation)
    prepared = {split: ladder.prepare_records(values) for split, values in records.items()}
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    order = list(range(size))
    random.Random(seed + 102_023 + size).shuffle(order)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint_dir / "step_0000.pt", model, optimizer, 0, order)
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(STEPS + 1):
        train_info = None
        if step > 0:
            indices = stage_a.batch_indices(order, step)
            train_info = stage_a.train_step(model, optimizer, prepared["train"], indices)
        if step in EVAL_STEPS:
            row = {
                "step": step,
                "train_step": train_info,
                "train": stage_a.evaluate(model, prepared["train"]),
                "validation": stage_a.evaluate(model, prepared["validation"]),
                "train_prediction_diagnostics": answer_prediction_diagnostics(model, prepared["train"]),
                "validation_prediction_diagnostics": answer_prediction_diagnostics(model, prepared["validation"]),
            }
            rows.append(row)
            print(json.dumps({
                "seed": seed,
                "size": size,
                "step": step,
                "train_exact": row["train"]["teacher_forced_exact_accuracy"],
                "validation_exact": row["validation"]["teacher_forced_exact_accuracy"],
            }, sort_keys=True), flush=True)
    stage_a.save_checkpoint(checkpoint_dir / f"step_{STEPS:04d}.pt", model, optimizer, STEPS, order)
    final = rows[-1]
    result = {
        "seed": seed,
        "training_examples": size,
        "validation_examples": VALIDATION_COUNT,
        "steps": STEPS,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "operation_pair_counts": operation_pair_counts(records["train"]),
        "final_train": final["train"],
        "final_validation": final["validation"],
        "final_train_prediction_diagnostics": final["train_prediction_diagnostics"],
        "final_validation_prediction_diagnostics": final["validation_prediction_diagnostics"],
        "records": rows,
    }
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_composition_scaling_ladder_20261023",
        "status": "cpu_only_dense_control",
        "seed": seed,
        "training_examples": size,
        "validation_examples": VALIDATION_COUNT,
        "steps": STEPS,
        "model_arm": "dense_active_compute_only",
        "test_evaluated": False,
        "heldout_evaluated": False,
        "validation_disjoint_from_full_128_train": True,
        "selection_rule": "validation recorded only; no checkpoint selection or test evaluation",
    })
    write_json(output / "RESULTS.json", result)
    write_jsonl(output / "train.jsonl", records["train"])
    write_jsonl(output / "validation.jsonl", records["validation"])
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
        "experiment": "crucible_composition_scaling_ladder_20261023",
        "status": "cpu_only_dense_control",
        "seeds": list(SEEDS),
        "training_sizes": list(SIZES),
        "steps_per_arm": STEPS,
        "validation_examples": VALIDATION_COUNT,
        "heldout_evaluated": False,
        "test_evaluated": False,
        "validation_disjoint_from_full_128_train": True,
    })
    summary: List[Dict[str, Any]] = []
    for seed in SEEDS:
        full_train = [ladder.make_composition(seed, "train", index) for index in range(max(SIZES))]
        validation = make_disjoint_validation(seed, full_train)
        validation_inputs = {str(record["input"]) for record in validation}
        if validation_inputs & {str(record["input"]) for record in full_train}:
            raise RuntimeError(f"validation overlap remains for seed {seed}")
        for size in SIZES:
            arm_output = output / f"seed{seed}_n{size}"
            arm_output.mkdir(parents=True, exist_ok=True)
            summary.append(run_arm(seed, size, arm_output, validation))
    write_json(output / "SUMMARY.json", {
        "experiment": "crucible_composition_scaling_ladder_20261023",
        "status": "completed",
        "test_evaluated": False,
        "heldout_evaluated": False,
        "arms": summary,
    })
    print(json.dumps({"arms": len(summary), "output": str(output), "status": "completed"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
