#!/usr/bin/env python3
"""Analyze frozen free-running composition traces by error source.

This is a read-only analysis of the 2026-10-23 frozen-checkpoint diagnostic.
It does not train or re-evaluate a model.  The teacher-forced final-token
metric is reported as a conditional second-step probe: the correct
intermediate and delimiter are supplied, and only the final operation is
measured.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "experiments" / "crucible_free_running_composition_diagnostic_20261023"
VALIDATION_ROOT = ROOT / "experiments" / "crucible_intermediate_supervision_20261023"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_second_step_error_analysis_20261024"
SEEDS = (42, 1337)
CHECKPOINT_STEPS = (0, 500, 1_000, 1_500, 2_000)
PROMPT_RE = re.compile(
    r"Task: arithmetic is modulo ten\. Apply both operations left-to-right\. "
    r"start=(?P<start>[0-9]); step1=(?P<op1>add|sub|mul)\((?P<arg1>[1-4])\); "
    r"step2=(?P<op2>add|sub|mul)\((?P<arg2>[1-4])\)\."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def as_digit(value: Any) -> int | None:
    text = str(value)
    return int(text) if text.isdigit() and len(text) == 1 else None


def load_validation(seed: int) -> Dict[str, Dict[str, Any]]:
    path = VALIDATION_ROOT / f"seed{seed}_intermediate_and_final" / "validation.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result: Dict[str, Dict[str, Any]] = {}
    for record in records:
        match = PROMPT_RE.search(str(record["input"]))
        if match is None:
            raise RuntimeError(f"cannot parse validation prompt {record['record_id']}")
        result[str(record["record_id"])] = {
            "pair": "+".join(record["metadata"]["composition"]),
            "intermediate": int(record["metadata"]["intermediate"]),
            "final": int(record["metadata"]["final"]),
            "second_operation": match.group("op2"),
            "second_operand": int(match.group("arg2")),
        }
    if len(result) != 64:
        raise RuntimeError(f"expected 64 validation records for seed {seed}, found {len(result)}")
    return result


def load_traces(seed: int, step: int) -> List[Dict[str, Any]]:
    payload = json.loads((TRACE_ROOT / f"seed{seed}.json").read_text(encoding="utf-8"))
    for row in payload["rows"]:
        if int(row["step"]) == step:
            return list(row["metrics"]["traces"])
    raise RuntimeError(f"missing trace checkpoint seed={seed} step={step}")


def summarize(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    counters = {
        "examples": 0,
        "teacher_forced_second_step_correct": 0,
        "free_intermediate_correct": 0,
        "free_final_correct": 0,
        "free_end_to_end_correct": 0,
        "free_final_given_intermediate_correct": 0,
        "free_final_given_intermediate_wrong": 0,
        "intermediate_correct_examples": 0,
        "intermediate_wrong_examples": 0,
        "counterfactual_final_correct": 0,
        "counterfactual_no_change": 0,
    }
    for row in rows:
        counters["examples"] += 1
        counters["teacher_forced_second_step_correct"] += int(row["teacher_forced_final"] == str(row["expected_final"]))
        intermediate_ok = row["free_intermediate"] == str(row["expected_intermediate"])
        final_ok = row["free_final"] == str(row["expected_final"])
        end_to_end_ok = intermediate_ok and row["free_delimiter"] == "," and final_ok
        counters["free_intermediate_correct"] += int(intermediate_ok)
        counters["free_final_correct"] += int(final_ok)
        counters["free_end_to_end_correct"] += int(end_to_end_ok)
        if intermediate_ok:
            counters["intermediate_correct_examples"] += 1
            counters["free_final_given_intermediate_correct"] += int(final_ok)
        else:
            counters["intermediate_wrong_examples"] += 1
            counters["free_final_given_intermediate_wrong"] += int(final_ok)
        counters["counterfactual_final_correct"] += int(
            row["counterfactual_final"] == str(row["expected_counterfactual_final"])
        )
        counters["counterfactual_no_change"] += int(row["counterfactual_final"] == row["factual_final"])
    n = counters["examples"]
    correct_n = counters["intermediate_correct_examples"]
    wrong_n = counters["intermediate_wrong_examples"]
    return {
        **counters,
        "teacher_forced_second_step_accuracy": ratio(counters["teacher_forced_second_step_correct"], n),
        "free_intermediate_accuracy": ratio(counters["free_intermediate_correct"], n),
        "free_final_accuracy": ratio(counters["free_final_correct"], n),
        "free_end_to_end_accuracy": ratio(counters["free_end_to_end_correct"], n),
        "free_final_given_intermediate_correct_accuracy": ratio(
            counters["free_final_given_intermediate_correct"], correct_n
        ),
        "free_final_given_intermediate_wrong_accuracy": ratio(
            counters["free_final_given_intermediate_wrong"], wrong_n
        ),
        "counterfactual_final_accuracy": ratio(counters["counterfactual_final_correct"], n),
        "counterfactual_no_change_rate": ratio(counters["counterfactual_no_change"], n),
    }


def run_checkpoint(seed: int, step: int) -> Dict[str, Any]:
    validation = load_validation(seed)
    traces = load_traces(seed, step)
    by_record = {str(trace["record_id"]): trace for trace in traces}
    if set(by_record) != set(validation):
        raise RuntimeError(f"trace/validation record mismatch for seed={seed} step={step}")
    rows: List[Dict[str, Any]] = []
    for record_id, trace in by_record.items():
        expected = validation[record_id]
        rows.append({
            "record_id": record_id,
            "pair": expected["pair"],
            "second_operation": expected["second_operation"],
            "second_operand": expected["second_operand"],
            "expected_intermediate": expected["intermediate"],
            "expected_final": expected["final"],
            "expected_counterfactual_final": trace["counterfactual"]["expected_counterfactual_final"],
            "teacher_forced_final": trace["teacher_forced_final"],
            "free_intermediate": trace["free"]["intermediate"],
            "free_delimiter": trace["free"]["delimiter"],
            "free_final": trace["free"]["final"],
            "counterfactual_final": trace["counterfactual"]["counterfactual_final"],
            "factual_final": trace["counterfactual"]["factual_final"],
        })
    dimensions = {
        "overall": summarize(rows),
        "by_operation_pair": {},
        "by_second_operation": {},
        "by_expected_intermediate": {},
        "by_expected_final": {},
    }
    for key, field in (
        ("by_operation_pair", "pair"),
        ("by_second_operation", "second_operation"),
        ("by_expected_intermediate", "expected_intermediate"),
        ("by_expected_final", "expected_final"),
    ):
        values = sorted({row[field] for row in rows}, key=str)
        dimensions[key] = {str(value): summarize(row for row in rows if row[field] == value) for value in values}
    return {"seed": seed, "step": step, **dimensions}


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
        "experiment": "crucible_second_step_error_analysis_20261024",
        "status": "read_only_frozen_trace_analysis",
        "source_experiment": "crucible_free_running_composition_diagnostic_20261023",
        "seeds": list(SEEDS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "training_performed": False,
        "model_evaluation_performed": False,
        "heldout_evaluated": False,
        "test_evaluated": False,
        "second_step_probe": "teacher-forced correct intermediate and delimiter, measure final token",
    })
    results = []
    for seed in SEEDS:
        seed_results = [run_checkpoint(seed, step) for step in CHECKPOINT_STEPS]
        write_json(output / f"seed{seed}.json", {"seed": seed, "checkpoints": seed_results})
        results.extend(seed_results)
    write_json(output / "SUMMARY.json", {
        "experiment": "crucible_second_step_error_analysis_20261024",
        "status": "completed",
        "training_performed": False,
        "model_evaluation_performed": False,
        "heldout_evaluated": False,
        "test_evaluated": False,
        "results": results,
    })
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"output": str(output), "status": "completed", "records": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
