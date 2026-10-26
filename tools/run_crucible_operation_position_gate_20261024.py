#!/usr/bin/env python3
"""CPU-only operation-position and isolated-second-step experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_PATH = ROOT / "tools" / "run_crucible_intermediate_supervision_20261023.py"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_operation_position_gate_20261024"
SEEDS = (42, 1337)
CONDITIONS = ("A_to_B", "B_only", "B_to_A")
MAX_STEPS = 2_000
CHECKPOINT_STEPS = (0, 500, 1_000, 1_500, 2_000)
RAW_BYTE_START = 3
COMMA = ","
OPERATIONS = ("add", "sub", "mul")
TRAIN_PAIRS = (
    ("add", "add"),
    ("add", "mul"),
    ("mul", "add"),
    ("mul", "mul"),
    ("sub", "add"),
    ("sub", "mul"),
    ("sub", "sub"),
)
HELDOUT_PAIRS = (("add", "sub"), ("mul", "sub"))
TRAIN_PER_PAIR = 16
VALIDATION_PER_PAIR = 8
HELDOUT_PER_PAIR = 32
B_ONLY_TRAIN_PER_OPERATION = 16
B_ONLY_VALIDATION_PER_OPERATION = 8
B_ONLY_HELDOUT_PER_OPERATION = 8


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


intermediate = import_module("intermediate_supervision_for_position_gate", INTERMEDIATE_PATH)
stage_a = intermediate.stage_a
ladder = intermediate.ladder


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


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def apply_mod(value: int, operation: str, operand: int) -> int:
    if operation == "add":
        return (value + operand) % 10
    if operation == "sub":
        return (value - operand) % 10
    if operation == "mul":
        return (value * operand) % 10
    raise ValueError(operation)


def token_for_byte(value: str) -> int:
    encoded = value.encode("utf-8")
    if len(encoded) != 1:
        raise ValueError(value)
    return RAW_BYTE_START + encoded[0]


def token_for_digit(value: int) -> int:
    return token_for_byte(str(value))


def token_label(token: int) -> str:
    if RAW_BYTE_START + ord("0") <= token <= RAW_BYTE_START + ord("9"):
        return str(token - RAW_BYTE_START - ord("0"))
    if token == token_for_byte(COMMA):
        return COMMA
    return f"token:{token}"


def record_id(seed: int, condition: str, split: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps({"seed": seed, "condition": condition, "split": split, **payload}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def make_composition_record(
    seed: int,
    condition: str,
    split: str,
    base_pair: Tuple[str, str],
    occurrence: int,
    used_inputs: set[str],
) -> Dict[str, Any]:
    actual_pair = base_pair if condition == "A_to_B" else (base_pair[1], base_pair[0])
    attempt = 0
    while True:
        rng = random.Random(stable_seed(seed, condition, split, base_pair, occurrence, attempt))
        start = rng.randrange(10)
        arg1 = rng.randrange(1, 5)
        arg2 = rng.randrange(1, 5)
        intermediate_value = apply_mod(start, actual_pair[0], arg1)
        final_value = apply_mod(intermediate_value, actual_pair[1], arg2)
        prompt = (
            "Task: position-gate modulo-ten arithmetic. Apply both operations "
            "left-to-right. "
            f"start={start}; step1={actual_pair[0]}({arg1}); step2={actual_pair[1]}({arg2}). "
            "Return the intermediate digit, then the final digit, separated by comma."
        )
        if prompt not in used_inputs:
            used_inputs.add(prompt)
            payload = {
                "start": start,
                "base_pair": list(base_pair),
                "actual_pair": list(actual_pair),
                "arg1": arg1,
                "arg2": arg2,
                "intermediate": intermediate_value,
                "final": final_value,
            }
            return {
                "schema_version": "crucible_operation_position_gate_v1",
                "record_id": record_id(seed, condition, split, payload),
                "stage": "operation_position_gate",
                "split": split,
                "condition": condition,
                "input": prompt,
                "target": f"{intermediate_value},{final_value}",
                "oracle": {"kind": "mod10_composition", "value": final_value, "intermediate": intermediate_value},
                "metadata": {
                    "condition": condition,
                    "base_pair": list(base_pair),
                    "composition": list(actual_pair),
                    "start": start,
                    "arg1": arg1,
                    "arg2": arg2,
                    "intermediate": intermediate_value,
                    "final": final_value,
                },
            }
        attempt += 1


def make_b_only_record(
    seed: int,
    split: str,
    operation: str,
    occurrence: int,
    used_inputs: set[str],
) -> Dict[str, Any]:
    attempt = 0
    while True:
        rng = random.Random(stable_seed(seed, "B_only", split, operation, occurrence, attempt))
        supplied_intermediate = rng.randrange(10)
        operand = rng.randrange(1, 5)
        final_value = apply_mod(supplied_intermediate, operation, operand)
        prompt = (
            "Task: position-gate modulo-ten arithmetic. The supplied intermediate "
            f"digit is {supplied_intermediate}. Apply operation {operation}({operand}) "
            "as the second operation. Return one digit only."
        )
        if prompt not in used_inputs:
            used_inputs.add(prompt)
            payload = {
                "operation": operation,
                "supplied_intermediate": supplied_intermediate,
                "operand": operand,
                "final": final_value,
            }
            return {
                "schema_version": "crucible_operation_position_gate_v1",
                "record_id": record_id(seed, "B_only", split, payload),
                "stage": "operation_position_gate",
                "split": split,
                "condition": "B_only",
                "input": prompt,
                "target": str(final_value),
                "oracle": {"kind": "mod10_single_operation", "value": final_value},
                "metadata": {
                    "condition": "B_only",
                    "operation": operation,
                    "composition": [operation],
                    "supplied_intermediate": supplied_intermediate,
                    "operand": operand,
                    "final": final_value,
                },
            }
        attempt += 1


def generate_records(seed: int, condition: str) -> Dict[str, List[Dict[str, Any]]]:
    used_inputs: set[str] = set()
    records: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "heldout": []}
    if condition in {"A_to_B", "B_to_A"}:
        for pair in TRAIN_PAIRS:
            for occurrence in range(TRAIN_PER_PAIR):
                records["train"].append(make_composition_record(seed, condition, "train", pair, occurrence, used_inputs))
            for occurrence in range(VALIDATION_PER_PAIR):
                records["validation"].append(make_composition_record(seed, condition, "validation", pair, occurrence, used_inputs))
        for pair in HELDOUT_PAIRS:
            for occurrence in range(HELDOUT_PER_PAIR):
                records["heldout"].append(make_composition_record(seed, condition, "heldout", pair, occurrence, used_inputs))
    elif condition == "B_only":
        for operation in OPERATIONS:
            for occurrence in range(B_ONLY_TRAIN_PER_OPERATION):
                records["train"].append(make_b_only_record(seed, "train", operation, occurrence, used_inputs))
            for occurrence in range(B_ONLY_VALIDATION_PER_OPERATION):
                records["validation"].append(make_b_only_record(seed, "validation", operation, occurrence, used_inputs))
            for occurrence in range(B_ONLY_HELDOUT_PER_OPERATION):
                records["heldout"].append(make_b_only_record(seed, "heldout", operation, occurrence, used_inputs))
    else:
        raise ValueError(condition)
    if any(len({str(record["input"]) for record in records[split]}) != len(records[split]) for split in records):
        raise RuntimeError(f"duplicate inputs within split for seed={seed} condition={condition}")
    all_inputs = [str(record["input"]) for values in records.values() for record in values]
    if len(set(all_inputs)) != len(all_inputs):
        raise RuntimeError(f"train/validation/heldout overlap for seed={seed} condition={condition}")
    return records


def oracle_audit(records: Mapping[str, Sequence[Mapping[str, Any]]], condition: str) -> None:
    for split, values in records.items():
        for record in values:
            metadata = record["metadata"]
            if condition == "B_only":
                expected = apply_mod(int(metadata["supplied_intermediate"]), str(metadata["operation"]), int(metadata["operand"]))
                if record["target"] != str(expected):
                    raise RuntimeError(f"B-only oracle mismatch: {record['record_id']}")
            else:
                ops = list(metadata["composition"])
                value = int(metadata["start"])
                value = apply_mod(value, ops[0], int(metadata["arg1"]))
                intermediate_value = value
                value = apply_mod(value, ops[1], int(metadata["arg2"]))
                if record["target"] != f"{intermediate_value},{value}":
                    raise RuntimeError(f"composition oracle mismatch: {record['record_id']}")


def prepare(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return ladder.prepare_records(records)


def next_token(model: Any, tokens: Sequence[int]) -> int:
    torch = stage_a.torch
    x = torch.tensor([list(tokens)], dtype=torch.long)
    logits = stage_a.forward_model(model, {"x": x})
    return int(logits[0, -1].argmax().item())


def prefix_tokens(record: Mapping[str, Any]) -> List[int]:
    return list(record["_full"][: int(record["_prefix_len"])])


def custom_metrics(model: Any, records: Sequence[Mapping[str, Any]], condition: str) -> Dict[str, Any]:
    torch = stage_a.torch
    teacher_sequence = stage_a.evaluate(model, records)
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for record in records:
            metadata = record["metadata"]
            prefix = prefix_tokens(record)
            if condition == "B_only":
                expected_final = int(metadata["final"])
                predicted_final = next_token(model, prefix)
                predicted_final_label = token_label(predicted_final)
                row = {
                    "record_id": record["record_id"],
                    "group": str(metadata["operation"]),
                    "expected_final": expected_final,
                    "teacher_forced_final": predicted_final_label,
                    "free_final": predicted_final_label,
                    "teacher_forced_final_correct": predicted_final_label == str(expected_final),
                    "free_final_correct": predicted_final_label == str(expected_final),
                    "free_end_to_end_correct": predicted_final_label == str(expected_final),
                }
            else:
                expected_intermediate = int(metadata["intermediate"])
                expected_final = int(metadata["final"])
                teacher_intermediate = next_token(model, prefix)
                teacher_final = next_token(model, prefix + [token_for_digit(expected_intermediate), token_for_byte(COMMA)])
                free_intermediate = next_token(model, prefix)
                free_delimiter = next_token(model, prefix + [free_intermediate])
                free_final = next_token(model, prefix + [free_intermediate, free_delimiter])
                intermediate_label = token_label(free_intermediate)
                delimiter_label = token_label(free_delimiter)
                free_final_label = token_label(free_final)
                row = {
                    "record_id": record["record_id"],
                    "group": "+".join(metadata["composition"]),
                    "expected_intermediate": expected_intermediate,
                    "expected_final": expected_final,
                    "teacher_forced_intermediate": token_label(teacher_intermediate),
                    "teacher_forced_final": token_label(teacher_final),
                    "free_intermediate": intermediate_label,
                    "free_delimiter": delimiter_label,
                    "free_final": free_final_label,
                    "teacher_forced_final_correct": token_label(teacher_final) == str(expected_final),
                    "free_intermediate_correct": intermediate_label == str(expected_intermediate),
                    "free_final_correct": free_final_label == str(expected_final),
                    "free_end_to_end_correct": (
                        intermediate_label == str(expected_intermediate)
                        and delimiter_label == COMMA
                        and free_final_label == str(expected_final)
                    ),
                }
            rows.append(row)

    def summarize(subset: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        n = len(subset)
        result: Dict[str, Any] = {
            "examples": n,
            "teacher_forced_sequence_exact": teacher_sequence["teacher_forced_exact_accuracy"] if subset is records else None,
            "teacher_forced_final_accuracy": sum(bool(row["teacher_forced_final_correct"]) for row in subset) / n if n else None,
            "free_final_accuracy": sum(bool(row["free_final_correct"]) for row in subset) / n if n else None,
            "free_end_to_end_accuracy": sum(bool(row["free_end_to_end_correct"]) for row in subset) / n if n else None,
        }
        if condition != "B_only":
            result["teacher_forced_intermediate_accuracy"] = sum(
                str(row["teacher_forced_intermediate"]) == str(row["expected_intermediate"]) for row in subset
            ) / n if n else None
            result["free_intermediate_accuracy"] = sum(bool(row["free_intermediate_correct"]) for row in subset) / n if n else None
            correct_intermediate = [row for row in subset if row["free_intermediate_correct"]]
            wrong_intermediate = [row for row in subset if not row["free_intermediate_correct"]]
            result["free_final_given_correct_intermediate"] = (
                sum(bool(row["free_final_correct"]) for row in correct_intermediate) / len(correct_intermediate)
                if correct_intermediate else None
            )
            result["free_final_given_wrong_intermediate"] = (
                sum(bool(row["free_final_correct"]) for row in wrong_intermediate) / len(wrong_intermediate)
                if wrong_intermediate else None
            )
        return result

    result = {"sequence": teacher_sequence, "overall": summarize(rows), "rows": rows, "by_group": {}}
    groups = sorted({str(row["group"]) for row in rows})
    for group in groups:
        result["by_group"][group] = summarize([row for row in rows if str(row["group"]) == group])
    return result


def run_arm(seed: int, condition: str, records: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> Dict[str, Any]:
    torch = stage_a.torch
    prepared = {split: prepare(values) for split, values in records.items()}
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    order = list(range(len(prepared["train"])))
    random.Random(stable_seed(seed, condition, "order")).shuffle(order)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint_dir / "step_0000.pt", model, optimizer, 0, order)
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(MAX_STEPS + 1):
        train_info = None
        if step > 0:
            indices = stage_a.batch_indices(order, step)
            train_info = stage_a.train_step(model, optimizer, prepared["train"], indices)
        if step in CHECKPOINT_STEPS:
            metrics = {
                split: custom_metrics(model, prepared[split], condition)
                for split in ("train", "validation", "heldout")
            }
            row = {"step": step, "train_step": train_info, "metrics": metrics}
            rows.append(row)
            print(json.dumps({
                "condition": condition,
                "seed": seed,
                "step": step,
                "train_free_end_to_end": metrics["train"]["overall"]["free_end_to_end_accuracy"],
                "validation_free_end_to_end": metrics["validation"]["overall"]["free_end_to_end_accuracy"],
                "heldout_free_end_to_end": metrics["heldout"]["overall"]["free_end_to_end_accuracy"],
                "validation_free_final": metrics["validation"]["overall"]["free_final_accuracy"],
            }, sort_keys=True), flush=True)
            if step in (500, 1_000, 1_500, 2_000):
                stage_a.save_checkpoint(checkpoint_dir / f"step_{step:04d}.pt", model, optimizer, step, order)
        elif step % 100 == 0:
            print(json.dumps({"condition": condition, "seed": seed, "step": step, "status": "training"}, sort_keys=True), flush=True)
    result = {
        "seed": seed,
        "condition": condition,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_examples": len(records["train"]),
        "validation_examples": len(records["validation"]),
        "heldout_examples": len(records["heldout"]),
        "max_steps": MAX_STEPS,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    write_json(output / "RESULTS.json", result)
    for split, values in records.items():
        write_jsonl(output / f"{split}.jsonl", values)
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_operation_position_gate_20261024",
        "seed": seed,
        "condition": condition,
        "status": "cpu_only_dense_control",
        "training_examples": len(records["train"]),
        "validation_examples": len(records["validation"]),
        "heldout_examples": len(records["heldout"]),
        "max_steps": MAX_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "test_evaluated": False,
        "original_benchmark_modified": False,
        "heldout_used_for_selection": False,
    })
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
        "experiment": "crucible_operation_position_gate_20261024",
        "status": "frozen_before_training",
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "train_pairs": [list(pair) for pair in TRAIN_PAIRS],
        "heldout_pairs": [list(pair) for pair in HELDOUT_PAIRS],
        "max_steps": MAX_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "original_benchmark_modified": False,
        "test_evaluated": False,
        "heldout_used_for_selection": False,
    })
    summary: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            records = generate_records(seed, condition)
            oracle_audit(records, condition)
            arm_output = output / f"seed{seed}_{condition}"
            arm_output.mkdir(parents=True, exist_ok=True)
            summary.append(run_arm(seed, condition, records, arm_output))
    write_json(output / "SUMMARY.json", {
        "experiment": "crucible_operation_position_gate_20261024",
        "status": "completed",
        "original_benchmark_modified": False,
        "test_evaluated": False,
        "heldout_used_for_selection": False,
        "results": summary,
    })
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"arms": len(summary), "output": str(output), "status": "completed"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
