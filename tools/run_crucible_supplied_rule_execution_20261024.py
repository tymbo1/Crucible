#!/usr/bin/env python3
"""Run the CPU-only supplied-rule execution gate."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "experiments" / "crucible_rule_induction_benchmark_20261024_v3"
BUILDER_PATH = ROOT / "tools" / "build_crucible_rule_induction_benchmark_20261024.py"
POSITION_PATH = ROOT / "tools" / "run_crucible_operation_position_gate_20261024.py"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_supplied_rule_execution_20261024"

SEEDS = (42, 1337)
STEPS = 2_000
CHECKPOINT_STEPS = (0, 500, 1_000, 1_500, 2_000)
EXECUTION_THRESHOLD = 0.90
TINY_THRESHOLD = 0.99
RAW_BYTE_START = 3


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = import_module("rule_builder_for_execution", BUILDER_PATH)
position = import_module("operation_position_for_execution", POSITION_PATH)
stage_a = position.stage_a
ladder = position.ladder


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(record), sort_keys=True) for record in records) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def formula_text(task: Mapping[str, Any]) -> str:
    family = str(task["rule_family"])
    params = [int(value) for value in task["rule_parameters"]]
    if family == "affine2":
        a, b, c = params
        return f"affine2:y=({a}*x+{b}*z+{c})mod10"
    if family == "max_shift":
        (c,) = params
        return f"max_shift:y=(max(x,z)+{c})mod10"
    if family == "absdiff_shift":
        (c,) = params
        return f"absdiff:y=(abs(x-z)+{c})mod10"
    if family == "threshold_select":
        threshold, low_offset, high_offset = params
        return f"threshold:t={threshold},low={low_offset},high={high_offset}"
    raise ValueError(family)


def execution_prompt(task: Mapping[str, Any]) -> str:
    query = task["query"]
    return f"Execute supplied rule {formula_text(task)}. Q:{query['x']},{query['z']}->? digit"


def model_record(task: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    return {
        "schema_version": "crucible_supplied_rule_execution_v1",
        "record_id": f"{task['task_id']}|{mode}",
        "task_id": task["task_id"],
        "rule_instance_id": task["rule_instance_id"],
        "family": task["rule_family"],
        "input": execution_prompt(task),
        "target": str(task["answer_by_demo_count"]["8"]),
        "mode": mode,
    }


def prepare(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return ladder.prepare_records(records)


def predict_digits(model: Any, records: Sequence[Mapping[str, Any]]) -> List[str]:
    if not records:
        return []
    torch = stage_a.torch
    model.eval()
    with torch.no_grad():
        batch = stage_a.batchify(records, list(range(len(records))))
        logits = stage_a.forward_model(model, batch)
        digit_start = RAW_BYTE_START + ord("0")
        digit_end = RAW_BYTE_START + ord("9")
        predictions: List[str] = []
        for row, record in enumerate(records):
            position = int(record["_prefix_len"]) - 1
            token = int(logits[row, position].argmax().item())
            predictions.append(str(token - digit_start) if digit_start <= token <= digit_end else f"token:{token}")
        return predictions


def evaluate(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prepared = list(records) if records and "_full" in records[0] else prepare(records)
    predictions = predict_digits(model, prepared)
    correct = [prediction == str(record["target"]) for prediction, record in zip(predictions, records)]
    by_family: Dict[str, List[bool]] = {}
    for record, is_correct in zip(records, correct):
        by_family.setdefault(str(record["family"]), []).append(is_correct)
    return {
        "examples": len(records),
        "correct": sum(correct),
        "exact_accuracy": sum(correct) / len(records) if records else None,
        "by_family": {
            family: {"examples": len(values), "exact_accuracy": sum(values) / len(values)}
            for family, values in sorted(by_family.items())
        },
        "predicted_digit_histogram": {
            str(digit): predictions.count(str(digit)) for digit in range(10)
        },
    }


def run_arm(seed: int, arm: str, train_tasks: Sequence[Mapping[str, Any]], validation_tasks: Sequence[Mapping[str, Any]], output: Path) -> Dict[str, Any]:
    if arm == "tiny":
        ordered_tasks = sorted(train_tasks, key=lambda task: str(task["task_id"]))[:8]
    else:
        ordered_tasks = list(train_tasks)
    train_records = [model_record(task, arm) for task in ordered_tasks]
    validation_records = [model_record(task, arm) for task in validation_tasks]
    prepared_train = prepare(train_records)
    torch = stage_a.torch
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    order = list(range(len(prepared_train)))
    random.Random(stable_seed(seed, arm, "training-order")).shuffle(order)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint_dir / "step_0000.pt", model, optimizer, 0, order)
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(STEPS + 1):
        train_info = None
        if step > 0:
            train_info = stage_a.train_step(model, optimizer, prepared_train, stage_a.batch_indices(order, step))
        if step in CHECKPOINT_STEPS:
            train_metrics = evaluate(model, prepared_train)
            validation_metrics = evaluate(model, validation_records)
            row = {"step": step, "train": train_metrics, "validation": validation_metrics, "train_step": train_info}
            rows.append(row)
            print(json.dumps({
                "arm": arm,
                "seed": seed,
                "step": step,
                "train_accuracy": train_metrics["exact_accuracy"],
                "validation_accuracy": validation_metrics["exact_accuracy"],
            }, sort_keys=True), flush=True)
        if step == STEPS:
            stage_a.save_checkpoint(checkpoint_dir / f"step_{STEPS:04d}.pt", model, optimizer, step, order)
    result = {
        "seed": seed,
        "arm": arm,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_examples": len(train_records),
        "validation_examples": len(validation_records),
        "steps": STEPS,
        "elapsed_seconds": time.perf_counter() - started,
        "final_train_accuracy": rows[-1]["train"]["exact_accuracy"],
        "final_validation_accuracy": rows[-1]["validation"]["exact_accuracy"],
        "tiny_overfit_pass": rows[-1]["train"]["exact_accuracy"] >= TINY_THRESHOLD if arm == "tiny" else None,
        "validation_pass": rows[-1]["validation"]["exact_accuracy"] >= EXECUTION_THRESHOLD if arm == "full" else None,
        "records": rows,
    }
    write_json(output / "RESULTS.json", result)
    write_jsonl(output / "train_records.jsonl", train_records)
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_supplied_rule_execution_20261024",
        "status": "cpu_only_dense_control",
        "seed": seed,
        "arm": arm,
        "steps": STEPS,
        "checkpoint_metrics": list(CHECKPOINT_STEPS),
        "validation_threshold": EXECUTION_THRESHOLD if arm == "full" else None,
        "gpu_used": False,
        "test_evaluated": False,
    })
    del model, optimizer
    gc.collect()
    return result


def evaluate_test(results: Sequence[Mapping[str, Any]], test_tasks: Sequence[Mapping[str, Any]], output: Path) -> Dict[str, Any]:
    torch = stage_a.torch
    evaluated: Dict[str, Any] = {}
    for result in results:
        seed = int(result["seed"])
        arm = str(result["arm"])
        checkpoint_path = output / f"seed{seed}_{arm}" / "checkpoints" / f"step_{STEPS:04d}.pt"
        model = stage_a.make_dense(seed)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        evaluated[f"seed{seed}_{arm}"] = evaluate(model, [model_record(task, arm) for task in test_tasks])
        del model
        gc.collect()
    write_json(output / "TEST_RESULTS.json", evaluated)
    return evaluated


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_tasks = load_jsonl(BENCHMARK_ROOT / "train.jsonl")
    validation_tasks = load_jsonl(BENCHMARK_ROOT / "validation.jsonl")
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_supplied_rule_execution_20261024",
        "status": "frozen_before_training",
        "benchmark": str(BENCHMARK_ROOT.relative_to(ROOT)),
        "seeds": list(SEEDS),
        "arms": ["tiny", "full"],
        "steps": STEPS,
        "validation_threshold": EXECUTION_THRESHOLD,
        "tiny_threshold": TINY_THRESHOLD,
        "test_evaluated": False,
        "gpu_used": False,
    })
    results: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ("tiny", "full"):
            arm_output = output / f"seed{seed}_{arm}"
            arm_output.mkdir(parents=True, exist_ok=True)
            results.append(run_arm(seed, arm, train_tasks, validation_tasks, arm_output))
    full_results = [result for result in results if result["arm"] == "full"]
    aggregate_validation = sum(result["final_validation_accuracy"] for result in full_results) / len(full_results)
    full_validation_pass = aggregate_validation >= EXECUTION_THRESHOLD
    test_evaluated = False
    if full_validation_pass:
        test_tasks = load_jsonl(BENCHMARK_ROOT / "test.jsonl")
        evaluate_test(results, test_tasks, output)
        test_evaluated = True
    summary = {
        "experiment": "crucible_supplied_rule_execution_20261024",
        "status": "completed",
        "test_evaluated": test_evaluated,
        "aggregate_validation_accuracy": aggregate_validation,
        "validation_threshold": EXECUTION_THRESHOLD,
        "validation_pass": full_validation_pass,
        "results": [
            {"seed": result["seed"], "arm": result["arm"], "final_train_accuracy": result["final_train_accuracy"], "final_validation_accuracy": result["final_validation_accuracy"], "elapsed_seconds": result["elapsed_seconds"]}
            for result in results
        ],
    }
    write_json(output / "SUMMARY.json", summary)
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"output": str(output), "status": "completed", "test_evaluated": test_evaluated, "validation_pass": full_validation_pass}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
