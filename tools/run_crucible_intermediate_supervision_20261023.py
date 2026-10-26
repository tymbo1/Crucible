#!/usr/bin/env python3
"""CPU-only final-answer versus intermediate-state supervision experiment."""

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
DIAGNOSTIC_PATH = ROOT / "tools" / "run_crucible_composition_diagnostic_20261022.py"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_intermediate_supervision_20261023"
SEEDS = (42, 1337)
TRAINING_EXAMPLES = 128
VALIDATION_EXAMPLES = 64
MAX_STEPS = 2_000
CHECKPOINT_STEPS = (0, 500, 1_000, 1_500, 2_000)
HEARTBEAT_STEPS = 100
EOS_ID = 2
PAD_ID = 0
RAW_BYTE_START = 3


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scaling = import_module("composition_scaling", SCALING_PATH)
diagnostic = import_module("composition_diagnostic", DIAGNOSTIC_PATH)
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


def make_records(seed: int) -> Dict[str, List[Dict[str, Any]]]:
    train = [ladder.make_composition(seed, "train", index) for index in range(TRAINING_EXAMPLES)]
    validation = scaling.make_disjoint_validation(seed, train)
    if {record["input"] for record in train} & {record["input"] for record in validation}:
        raise RuntimeError(f"validation overlap for seed {seed}")
    return {"train": train, "validation": validation}


def augment_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    oracle = diagnostic.independent_oracle(record)
    augmented = dict(record)
    augmented["input"] = str(record["input"]).replace(
        "Return one digit only.",
        "Return the intermediate digit, then the final digit, separated by comma.",
    )
    augmented["target"] = f"{oracle['intermediate']},{oracle['expected']}"
    metadata = dict(record.get("metadata", {}))
    metadata["supervision"] = "intermediate_and_final"
    metadata["intermediate"] = oracle["intermediate"]
    metadata["final"] = oracle["expected"]
    augmented["metadata"] = metadata
    return augmented


def target_positions(record: Mapping[str, Any], mode: str) -> Dict[str, int]:
    prefix_len = int(record["_prefix_len"])
    if mode == "final_only":
        return {"final": prefix_len - 1}
    if mode == "intermediate_and_final":
        return {"intermediate": prefix_len - 1, "final": prefix_len + 1}
    raise ValueError(mode)


def answer_prediction_diagnostics(model: Any, records: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    torch = stage_a.torch
    model.eval()
    with torch.no_grad():
        batch = stage_a.batchify(records, list(range(len(records))))
        logits = stage_a.forward_model(model, batch)
        positions = [target_positions(record, mode) for record in records]
        fields = ["final"] if mode == "final_only" else ["intermediate", "final"]
        probabilities: Dict[str, float] = {}
        ranks: Dict[str, float] = {}
        histograms: Dict[str, Dict[str, int]] = {}
        digit_start = RAW_BYTE_START + ord("0")
        digit_end = RAW_BYTE_START + ord("9")
        for field in fields:
            values = []
            field_ranks = []
            predicted_digits = []
            for row, position_map in enumerate(positions):
                position = position_map[field]
                target = int(batch["y"][row, position].item())
                row_logits = logits[row, position]
                values.append(float(torch.softmax(row_logits, dim=-1)[target].item()))
                field_ranks.append(int(1 + (row_logits > row_logits[target]).sum().item()))
                predicted = int(row_logits.argmax().item())
                if digit_start <= predicted <= digit_end:
                    predicted_digits.append(predicted - digit_start)
            probabilities[field] = sum(values) / len(values)
            ranks[field] = sum(field_ranks) / len(field_ranks)
            histograms[field] = {str(digit): predicted_digits.count(digit) for digit in range(10)}
    return {
        "answer_positions": len(records) if mode == "final_only" else 2 * len(records),
        "mean_target_probability": probabilities,
        "mean_target_rank": ranks,
        "predicted_digit_histogram": histograms,
    }


def exact_by_record(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, bool]:
    torch = stage_a.torch
    model.eval()
    with torch.no_grad():
        batch = stage_a.batchify(records, list(range(len(records))))
        logits = stage_a.forward_model(model, batch)
        mask = batch["target_mask"] & (batch["y"] != PAD_ID)
        correct = ((logits.argmax(dim=-1) == batch["y"]) | ~mask).all(dim=1).tolist()
    return {str(record["record_id"]): bool(value) for record, value in zip(records, correct)}


def task_metrics(model: Any, records: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    torch = stage_a.torch
    aggregate = stage_a.evaluate(model, records)
    with torch.no_grad():
        batch = stage_a.batchify(records, list(range(len(records))))
        logits = stage_a.forward_model(model, batch)
        positions = [target_positions(record, mode) for record in records]
        final_correct = []
        intermediate_correct = []
        both_correct = []
        for row, position_map in enumerate(positions):
            final_ok = int(logits[row, position_map["final"]].argmax().item()) == int(batch["y"][row, position_map["final"]].item())
            final_correct.append(final_ok)
            if mode == "intermediate_and_final":
                intermediate_ok = int(logits[row, position_map["intermediate"]].argmax().item()) == int(batch["y"][row, position_map["intermediate"]].item())
                intermediate_correct.append(intermediate_ok)
                both_correct.append(intermediate_ok and final_ok)
    result: Dict[str, Any] = {
        "sequence": aggregate,
        "final_token_accuracy": sum(final_correct) / len(final_correct),
        "answer_sequence_accuracy": sum(final_correct) / len(final_correct) if mode == "final_only" else sum(both_correct) / len(both_correct),
    }
    if mode == "intermediate_and_final":
        result["intermediate_token_accuracy"] = sum(intermediate_correct) / len(intermediate_correct)
        result["intermediate_and_final_accuracy"] = sum(both_correct) / len(both_correct)
    return result


def grouped_metrics(model: Any, records: Sequence[Mapping[str, Any]], prepared: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    groups: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        pair = "+".join(record["metadata"]["composition"])
        groups.setdefault(pair, []).append(index)
    result = {"aggregate": task_metrics(model, prepared, mode), "by_operation_pair": {}}
    for pair, indices in sorted(groups.items()):
        result["by_operation_pair"][pair] = task_metrics(model, [prepared[index] for index in indices], mode)
    return result


def run_arm(seed: int, mode: str, records: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> Dict[str, Any]:
    torch = stage_a.torch
    if mode == "final_only":
        arm_records = {split: [dict(record) for record in values] for split, values in records.items()}
    else:
        arm_records = {split: [augment_record(record) for record in values] for split, values in records.items()}
    prepared = {split: ladder.prepare_records(values) for split, values in arm_records.items()}
    target_positions_per_example = 2 if mode == "final_only" else 4
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
            train_info = stage_a.train_step(model, optimizer, prepared["train"], indices)
        if step in CHECKPOINT_STEPS:
            train_metrics = grouped_metrics(model, arm_records["train"], prepared["train"], mode)
            validation_metrics = grouped_metrics(model, arm_records["validation"], prepared["validation"], mode)
            current_correct = exact_by_record(model, prepared["train"])
            forgotten = 0
            newly_correct = 0
            if previous_correct is not None:
                forgotten = sum(previous_correct[key] and not current_correct[key] for key in current_correct)
                newly_correct = sum(not previous_correct[key] and current_correct[key] for key in current_correct)
            previous_correct = current_correct
            row = {
                "step": step,
                "train_step": train_info,
                "train": train_metrics,
                "validation": validation_metrics,
                "train_prediction_diagnostics": answer_prediction_diagnostics(model, prepared["train"], mode),
                "validation_prediction_diagnostics": answer_prediction_diagnostics(model, prepared["validation"], mode),
                "train_forgetting_since_previous_checkpoint": forgotten,
                "train_newly_correct_since_previous_checkpoint": newly_correct,
            }
            rows.append(row)
            print(json.dumps({
                "mode": mode,
                "seed": seed,
                "step": step,
                "train_sequence_exact": train_metrics["aggregate"]["sequence"]["teacher_forced_exact_accuracy"],
                "train_final_accuracy": train_metrics["aggregate"]["final_token_accuracy"],
                "train_intermediate_accuracy": train_metrics["aggregate"].get("intermediate_token_accuracy"),
                "validation_final_accuracy": validation_metrics["aggregate"]["final_token_accuracy"],
                "validation_intermediate_accuracy": validation_metrics["aggregate"].get("intermediate_token_accuracy"),
                "forgotten": forgotten,
            }, sort_keys=True), flush=True)
            if step in (500, 1_000, 1_500, 2_000):
                stage_a.save_checkpoint(checkpoint_dir / f"step_{step:04d}.pt", model, optimizer, step, order)
        elif step % HEARTBEAT_STEPS == 0:
            print(json.dumps({"mode": mode, "seed": seed, "step": step, "status": "training"}, sort_keys=True), flush=True)
    result = {
        "seed": seed,
        "mode": mode,
        "training_examples": TRAINING_EXAMPLES,
        "validation_examples": VALIDATION_EXAMPLES,
        "max_steps": MAX_STEPS,
        "target_positions_per_example": target_positions_per_example,
        "elapsed_seconds": time.perf_counter() - started,
        "operation_pair_counts": scaling.operation_pair_counts(records["train"]),
        "rows": rows,
    }
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_intermediate_supervision_20261023",
        "status": "cpu_only_dense_control",
        "seed": seed,
        "mode": mode,
        "training_examples": TRAINING_EXAMPLES,
        "validation_examples": VALIDATION_EXAMPLES,
        "max_steps": MAX_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "target_positions_per_example": target_positions_per_example,
        "test_evaluated": False,
        "heldout_evaluated": False,
        "validation_disjoint_from_training": True,
        "same_initialization_and_data_order_as_paired_arm": True,
    })
    write_json(output / "RESULTS.json", result)
    write_jsonl(output / "train.jsonl", arm_records["train"])
    write_jsonl(output / "validation.jsonl", arm_records["validation"])
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
        "experiment": "crucible_intermediate_supervision_20261023",
        "status": "cpu_only_dense_control",
        "seeds": list(SEEDS),
        "modes": ["final_only", "intermediate_and_final"],
        "training_examples": TRAINING_EXAMPLES,
        "validation_examples": VALIDATION_EXAMPLES,
        "max_steps": MAX_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "test_evaluated": False,
        "heldout_evaluated": False,
        "validation_disjoint_from_training": True,
        "paired_design": "same initialization, data order, architecture and optimizer per seed",
    })
    results = []
    for seed in SEEDS:
        records = make_records(seed)
        for mode in ("final_only", "intermediate_and_final"):
            results.append(run_arm(seed, mode, records, output / f"seed{seed}_{mode}"))
    write_json(output / "SUMMARY.json", {
        "experiment": "crucible_intermediate_supervision_20261023",
        "status": "completed",
        "test_evaluated": False,
        "heldout_evaluated": False,
        "results": results,
    })
    print(json.dumps({"output": str(output), "arms": len(results), "status": "completed"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
