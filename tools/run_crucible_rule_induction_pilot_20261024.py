#!/usr/bin/env python3
"""Run the frozen CPU-only rule-induction pilot on benchmark v2."""

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
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_rule_induction_pilot_20261024_v3"

SEEDS = (42, 1337)
DEMO_COUNTS = (1, 2, 4, 8)
STEPS = 2_000
CHECKPOINT_STEPS = (0, 500, 1_000, 1_500, 2_000)
EVIDENCE_GAIN_THRESHOLD = 0.05
BASELINE_GAIN_THRESHOLD = 0.05
SHUFFLE_DROP_THRESHOLD = 0.05
RAW_BYTE_START = 3


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = import_module("rule_induction_builder_for_pilot", BUILDER_PATH)
position = import_module("operation_position_for_rule_pilot", POSITION_PATH)
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


def query_only_prompt(task: Mapping[str, Any]) -> str:
    return builder.prompt_for([], task["query"], 0)


def candidate_outputs(task: Mapping[str, Any], count: int, demonstrations: Sequence[Mapping[str, int]] | None = None) -> List[str]:
    rules = builder.unique_rules()
    demos = list(demonstrations if demonstrations is not None else task["demonstrations"])
    candidates = builder.consistent_candidates(rules, demos, count)
    query = task["query"]
    outputs = sorted({
        str(builder.apply_rule(rule, int(query["x"]), int(query["z"])))
        for rule in candidates
    })
    return outputs


def model_record(
    task: Mapping[str, Any],
    prompt: str,
    count: int,
    mode: str,
    candidate_output_values: Sequence[str],
) -> Dict[str, Any]:
    target = str(task["answer_by_demo_count"][str(count)])
    return {
        "schema_version": "crucible_rule_induction_pilot_v1",
        "record_id": f"{task['task_id']}|{mode}|{count}",
        "task_id": task["task_id"],
        "rule_instance_id": task["rule_instance_id"],
        "family": task["rule_family"],
        "demo_count": count,
        "mode": mode,
        "input": prompt,
        "target": target,
        "analysis_candidate_outputs": list(candidate_output_values),
        "analysis_query_identifiable": len(candidate_output_values) == 1,
    }


def valid_records(tasks: Sequence[Mapping[str, Any]], count: int, mode: str = "valid") -> List[Dict[str, Any]]:
    return [
        model_record(
            task,
            str(task["prompt_by_demo_count"][str(count)]),
            count,
            mode,
            candidate_outputs(task, count),
        )
        for task in tasks
    ]


def shuffled_records(tasks: Sequence[Mapping[str, Any]], count: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks):
        source = tasks[(index + 1) % len(tasks)]
        demonstrations = list(source["demonstrations"][:count])
        prompt = builder.prompt_for(demonstrations, task["query"], count)
        records.append(model_record(task, prompt, count, "shuffled", candidate_outputs(task, count, demonstrations)))
    return records


def query_only_records(tasks: Sequence[Mapping[str, Any]], count: int) -> List[Dict[str, Any]]:
    return [
        model_record(task, query_only_prompt(task), count, "query_only", candidate_outputs(task, count))
        for task in tasks
    ]


def mixed_training_records(tasks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(tasks, key=lambda task: str(task["task_id"]))
    rng = random.Random(stable_seed("mixed-training-counts", 20261024))
    rng.shuffle(ordered)
    records: List[Dict[str, Any]] = []
    for index, task in enumerate(ordered):
        count = DEMO_COUNTS[index % len(DEMO_COUNTS)]
        records.append(model_record(
            task,
            str(task["prompt_by_demo_count"][str(count)]),
            count,
            "mixed_training",
            candidate_outputs(task, count),
        ))
    return records


def prepare(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return ladder.prepare_records(records)


def predict_digits(model: Any, records: Sequence[Mapping[str, Any]]) -> List[str]:
    torch = stage_a.torch
    if not records:
        return []
    model.eval()
    with torch.no_grad():
        batch = stage_a.batchify(records, list(range(len(records))))
        logits = stage_a.forward_model(model, batch)
        predictions: List[str] = []
        digit_start = RAW_BYTE_START + ord("0")
        digit_end = RAW_BYTE_START + ord("9")
        for row, record in enumerate(records):
            position = int(record["_prefix_len"]) - 1
            token = int(logits[row, position].argmax().item())
            predictions.append(str(token - digit_start) if digit_start <= token <= digit_end else f"token:{token}")
        return predictions


def evaluate_records(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prepared = list(records) if records and "_full" in records[0] else prepare(records)
    predictions = predict_digits(model, prepared)
    raw_correct = [prediction == str(record["target"]) for prediction, record in zip(predictions, records)]
    identifiable = [index for index, record in enumerate(records) if bool(record["analysis_query_identifiable"])]
    candidate_consistent = [
        prediction in set(str(value) for value in record["analysis_candidate_outputs"])
        for prediction, record in zip(predictions, records)
    ]
    return {
        "examples": len(records),
        "raw_correct": sum(raw_correct),
        "raw_accuracy": sum(raw_correct) / len(records) if records else None,
        "identifiable_examples": len(identifiable),
        "identifiable_correct": sum(raw_correct[index] for index in identifiable),
        "identifiable_accuracy": (
            sum(raw_correct[index] for index in identifiable) / len(identifiable)
            if identifiable else None
        ),
        "candidate_consistent": sum(candidate_consistent),
        "candidate_consistent_accuracy": sum(candidate_consistent) / len(records) if records else None,
        "predicted_digit_histogram": {
            str(digit): predictions.count(str(digit)) for digit in range(10)
        },
    }


def evaluate_validation(model: Any, tasks: Sequence[Mapping[str, Any]], arm: str) -> Dict[str, Any]:
    valid: Dict[str, Any] = {}
    shuffled: Dict[str, Any] = {}
    query_only: Dict[str, Any] = {}
    for count in DEMO_COUNTS:
        if arm == "mixed":
            valid[str(count)] = evaluate_records(model, valid_records(tasks, count))
            shuffled[str(count)] = evaluate_records(model, shuffled_records(tasks, count))
        query_only[str(count)] = evaluate_records(model, query_only_records(tasks, count))
    return {"valid": valid, "shuffled": shuffled, "query_only_prompt": query_only}


def run_arm(seed: int, arm: str, train_tasks: Sequence[Mapping[str, Any]], validation_tasks: Sequence[Mapping[str, Any]], output: Path) -> Dict[str, Any]:
    torch = stage_a.torch
    train_records = mixed_training_records(train_tasks) if arm == "mixed" else query_only_records(train_tasks, 1)
    prepared_train = prepare(train_records)
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
            indices = stage_a.batch_indices(order, step)
            train_info = stage_a.train_step(model, optimizer, prepared_train, indices)
        if step in CHECKPOINT_STEPS:
            validation = evaluate_validation(model, validation_tasks, arm)
            train = evaluate_records(model, prepared_train)
            row = {"step": step, "train_step": train_info, "train": train, "validation": validation}
            rows.append(row)
            print(json.dumps({
                "arm": arm,
                "seed": seed,
                "step": step,
                "train_accuracy": train["raw_accuracy"],
                "valid_1": validation["valid"].get("1", {}).get("raw_accuracy"),
                "valid_4": validation["valid"].get("4", {}).get("raw_accuracy"),
                "valid_8": validation["valid"].get("8", {}).get("raw_accuracy"),
                "shuffle_8": validation["shuffled"].get("8", {}).get("raw_accuracy"),
            }, sort_keys=True), flush=True)
        if step == STEPS:
            stage_a.save_checkpoint(checkpoint_dir / f"step_{STEPS:04d}.pt", model, optimizer, step, order)
    result = {
        "seed": seed,
        "arm": arm,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_examples": len(train_records),
        "validation_tasks": len(validation_tasks),
        "steps": STEPS,
        "elapsed_seconds": time.perf_counter() - started,
        "records": rows,
    }
    write_json(output / "RESULTS.json", result)
    write_jsonl(output / "train_records.jsonl", train_records)
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_rule_induction_pilot_20261024",
        "status": "cpu_only_dense_control",
        "seed": seed,
        "arm": arm,
        "steps": STEPS,
        "checkpoint_metrics": list(CHECKPOINT_STEPS),
        "test_evaluated": False,
        "gpu_used": False,
        "heldout_used_for_selection": False,
    })
    del model, optimizer
    gc.collect()
    return result


def aggregate_gate(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    mixed = [result for result in results if result["arm"] == "mixed"]
    baseline = [result for result in results if result["arm"] == "query_only"]
    final_mixed = [result["records"][-1]["validation"] for result in mixed]
    final_baseline = [result["records"][-1]["validation"] for result in baseline]
    valid_1 = sum(item["valid"]["1"]["raw_accuracy"] for item in final_mixed) / len(final_mixed)
    valid_8 = sum(item["valid"]["8"]["raw_accuracy"] for item in final_mixed) / len(final_mixed)
    mixed_ident_8 = sum(item["valid"]["8"]["identifiable_accuracy"] for item in final_mixed) / len(final_mixed)
    baseline_ident_8 = sum(item["query_only_prompt"]["8"]["identifiable_accuracy"] for item in final_baseline) / len(final_baseline)
    valid_8_shuffle = sum(item["shuffled"]["8"]["raw_accuracy"] for item in final_mixed) / len(final_mixed)
    gate = {
        "evidence_response": valid_8 - valid_1 >= EVIDENCE_GAIN_THRESHOLD,
        "demonstration_advantage": mixed_ident_8 - baseline_ident_8 >= BASELINE_GAIN_THRESHOLD,
        "evidence_dependence": valid_8 - valid_8_shuffle >= SHUFFLE_DROP_THRESHOLD,
    }
    return {
        "aggregate_metrics": {
            "mixed_valid_1_raw_accuracy": valid_1,
            "mixed_valid_8_raw_accuracy": valid_8,
            "mixed_valid_8_identifiable_accuracy": mixed_ident_8,
            "query_only_valid_8_identifiable_accuracy": baseline_ident_8,
            "mixed_valid_8_shuffled_raw_accuracy": valid_8_shuffle,
        },
        "thresholds": {
            "evidence_gain": EVIDENCE_GAIN_THRESHOLD,
            "baseline_gain": BASELINE_GAIN_THRESHOLD,
            "shuffle_drop": SHUFFLE_DROP_THRESHOLD,
        },
        "gates": gate,
        "all_validation_gates_pass": all(gate.values()),
    }


def evaluate_test_if_authorized(results: Sequence[Mapping[str, Any]], test_tasks: Sequence[Mapping[str, Any]], output: Path) -> Dict[str, Any]:
    # Test evaluation is deliberately isolated in a separate fresh model replay.
    # The final checkpoints are loaded only after aggregate validation gates pass.
    torch = stage_a.torch
    test_results: Dict[str, Any] = {}
    for result in results:
        seed = int(result["seed"])
        arm = str(result["arm"])
        checkpoint = output / f"seed{seed}_{arm}" / "checkpoints" / f"step_{STEPS:04d}.pt"
        model = stage_a.make_dense(seed)
        optimizer = stage_a.optimizer_for(model)
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint_data["model"])
        if arm == "mixed":
            counts = {str(count): evaluate_records(model, valid_records(test_tasks, count)) for count in DEMO_COUNTS}
        else:
            counts = {str(count): evaluate_records(model, query_only_records(test_tasks, count)) for count in DEMO_COUNTS}
        test_results[f"seed{seed}_{arm}"] = counts
        del model, optimizer
        gc.collect()
    write_json(output / "TEST_RESULTS.json", test_results)
    return test_results


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
        "experiment": "crucible_rule_induction_pilot_20261024",
        "status": "frozen_before_training",
        "benchmark": str(BENCHMARK_ROOT.relative_to(ROOT)),
        "seeds": list(SEEDS),
        "arms": ["query_only", "mixed"],
        "steps": STEPS,
        "checkpoint_metrics": list(CHECKPOINT_STEPS),
        "demo_counts": list(DEMO_COUNTS),
        "validation_gates": {
            "evidence_gain": EVIDENCE_GAIN_THRESHOLD,
            "baseline_gain": BASELINE_GAIN_THRESHOLD,
            "shuffle_drop": SHUFFLE_DROP_THRESHOLD,
        },
        "test_evaluated": False,
        "gpu_used": False,
    })
    write_json(output / "DATA_MANIFEST.json", {
        "train_tasks": len(train_tasks),
        "validation_tasks": len(validation_tasks),
        "train_rule_instance_ids": sorted(task["rule_instance_id"] for task in train_tasks),
        "validation_rule_instance_ids": sorted(task["rule_instance_id"] for task in validation_tasks),
    })

    results: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ("query_only", "mixed"):
            arm_output = output / f"seed{seed}_{arm}"
            arm_output.mkdir(parents=True, exist_ok=True)
            results.append(run_arm(seed, arm, train_tasks, validation_tasks, arm_output))

    gate = aggregate_gate(results)
    test_evaluated = False
    if gate["all_validation_gates_pass"]:
        test_tasks = load_jsonl(BENCHMARK_ROOT / "test.jsonl")
        evaluate_test_if_authorized(results, test_tasks, output)
        test_evaluated = True
    summary = {
        "experiment": "crucible_rule_induction_pilot_20261024",
        "status": "completed",
        "test_evaluated": test_evaluated,
        "validation": gate,
        "results": [
            {"seed": result["seed"], "arm": result["arm"], "elapsed_seconds": result["elapsed_seconds"]}
            for result in results
        ],
    }
    write_json(output / "SUMMARY.json", summary)
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"output": str(output), "status": "completed", "test_evaluated": test_evaluated, "validation": gate}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
