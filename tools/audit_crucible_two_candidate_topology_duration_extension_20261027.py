#!/usr/bin/env python3
"""Independently audit the completed topology training-duration extension."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_two_candidate_topology_duration_extension_20261027_v1"
SOURCE = ROOT / "experiments" / "crucible_two_candidate_topology_control_20261027_v1"
SUMMARY = OUTPUT / "SUMMARY.json"
HASHES = OUTPUT / "HASHES.json"
AUDIT_JSON = OUTPUT / "POST_RUN_AUDIT.json"
AUDIT_MD = OUTPUT / "POST_RUN_AUDIT.md"
THRESHOLD = 0.95
CHECKPOINT_STEPS = (1, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def metric_from_csv(path: Path) -> Dict[str, Any]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    groups: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["counterfactual_pair_id"]].append(row)
    complete = [values for values in groups.values() if len(values) == 2]
    value_correct = sum(row["value_correct"] == "True" for row in rows)
    slot_correct = sum(row["slot_correct"] == "True" for row in rows)
    paired_correct = sum(all(row["value_correct"] == "True" for row in values) for values in complete)
    flips = sum(values[0]["predicted_value"] != values[1]["predicted_value"] for values in complete)
    by_distance: Dict[str, Dict[str, Any]] = {}
    for distance in sorted({int(row["graph_distance"]) for row in rows}):
        subset = [row for row in rows if int(row["graph_distance"]) == distance]
        correct = sum(row["value_correct"] == "True" for row in subset)
        by_distance[str(distance)] = {"correct": correct, "rows": len(subset), "accuracy": correct / len(subset)}
    return {
        "rows": len(rows),
        "value_correct": value_correct,
        "value_accuracy": value_correct / len(rows),
        "slot_correct": slot_correct,
        "slot_accuracy": slot_correct / len(rows),
        "paired_groups": len(complete),
        "paired_correct": paired_correct,
        "paired_accuracy": paired_correct / len(complete),
        "candidate_order_flips": flips,
        "by_distance": by_distance,
    }


def assert_equal(actual: Any, expected: Any, label: str, failures: List[str]) -> None:
    if actual != expected:
        failures.append(f"{label}: expected {expected!r}, got {actual!r}")


def main() -> int:
    summary = load(SUMMARY)
    failures: List[str] = []
    checks: Dict[str, Any] = {}
    assert_equal(summary.get("status"), "completed", "summary.status", failures)
    assert_equal(summary.get("training_gate_passes"), 4, "summary.training_gate_passes", failures)
    assert_equal(summary.get("training_performed"), True, "summary.training_performed", failures)
    assert_equal(summary.get("sealed_test_accessed"), False, "summary.sealed_test_accessed", failures)
    assert_equal(tuple(summary.get("checkpoint_steps", ())), CHECKPOINT_STEPS, "summary.checkpoint_steps", failures)

    prior_hashes = load(SOURCE / "HASHES.json")
    prior_mismatches = [path for path, expected in prior_hashes.items() if not (ROOT / path).exists() or sha256(ROOT / path) != expected]
    checks["source_hashes_verified"] = not prior_mismatches
    if prior_mismatches:
        failures.append(f"source hash mismatches: {prior_mismatches}")

    result_checks: List[Dict[str, Any]] = []
    for result in summary["results"]:
        cell = OUTPUT / result["cell"]
        trace_path = cell / "TRAINING_TRACE.jsonl"
        train_csv = cell / "TRAIN_PREDICTIONS.csv"
        validation_csv = cell / "VALIDATION_PREDICTIONS.csv"
        result_json = cell / "RESULTS.json"
        trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert_equal(len(trace), 2000, f"{result['cell']}.trace_rows", failures)
        assert_equal([int(row["step"]) for row in trace], list(range(1, 2001)), f"{result['cell']}.trace_steps", failures)
        recorded = {int(row["step"]): row for row in trace if "training" in row}
        assert_equal(tuple(sorted(recorded)), CHECKPOINT_STEPS, f"{result['cell']}.recorded_checkpoints", failures)
        baseline = result["original_500_step"]
        step500 = recorded.get(500, {})
        assert_equal(step500.get("training"), baseline["train_metrics"], f"{result['cell']}.step500.training_reproduction", failures)
        assert_equal(step500.get("validation"), baseline["validation_metrics"], f"{result['cell']}.step500.validation_reproduction", failures)

        recomputed_train = metric_from_csv(train_csv)
        recomputed_validation = metric_from_csv(validation_csv)
        assert_equal(recomputed_train, result["train_metrics"], f"{result['cell']}.train_metrics_recomputed", failures)
        assert_equal(recomputed_validation, result["validation_metrics"], f"{result['cell']}.validation_metrics_recomputed", failures)
        assert_equal(load(result_json), result, f"{result['cell']}.RESULTS_consistency", failures)

        gate_steps = [step for step in CHECKPOINT_STEPS if recorded[step]["training"]["value_accuracy"] >= THRESHOLD]
        expected_first = min(gate_steps) if gate_steps else None
        assert_equal(expected_first, result["first_recorded_gate_step"], f"{result['cell']}.first_gate_step", failures)
        assert_equal(result["parameter_count"], 22106906, f"{result['cell']}.parameter_count", failures)
        result_checks.append({
            "cell": result["cell"],
            "trace_rows": len(trace),
            "step500_reproduced": step500.get("training") == baseline["train_metrics"] and step500.get("validation") == baseline["validation_metrics"],
            "train_csv_recomputed": recomputed_train == result["train_metrics"],
            "validation_csv_recomputed": recomputed_validation == result["validation_metrics"],
            "training_gate_pass": bool(result["training_gate_pass"]),
            "first_recorded_gate_step": result["first_recorded_gate_step"],
        })

    checks["arm_checks"] = result_checks
    checks["all_arms_passed"] = all(item["training_gate_pass"] for item in result_checks)
    checks["all_step500_reproduced"] = all(item["step500_reproduced"] for item in result_checks)
    checks["all_csv_metrics_recomputed"] = all(item["train_csv_recomputed"] and item["validation_csv_recomputed"] for item in result_checks)
    checks["all_hashes_verified_before_audit"] = all((ROOT / path).exists() and sha256(ROOT / path) == expected for path, expected in load(HASHES).items())
    checks["interpretation_boundary"] = "Duration extension establishes acquisition rescue/competence under the frozen recipe; it does not itself establish a topology or neural multi-hop mechanism."

    audit = {
        "schema_version": "crucible_topology_duration_extension_post_run_audit_v1",
        "experiment": summary["experiment"],
        "status": "completed",
        "verification_pass": not failures,
        "failures": failures,
        "checks": checks,
        "training_performed": True,
        "sealed_test_accessed": False,
        "source_checkpoints_modified": False,
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Post-run audit: topology training-duration extension",
        "",
        f"Verification pass: **{'yes' if not failures else 'no'}**.",
        "",
        "The audit verified the immutable source hashes, all 2,000 trace steps, every recorded checkpoint, exact reproduction of the committed 500-step train/validation metrics, CSV metric recomputation, parameter count and final training-gate outcomes.",
        "",
        "| Cell | 500-step reproduced | CSV metrics reproduced | Gate | First recorded gate |",
        "|---|---|---|---|---:|",
    ]
    for item in result_checks:
        report.append(f"| {item['cell']} | {'yes' if item['step500_reproduced'] else 'no'} | {'yes' if item['train_csv_recomputed'] and item['validation_csv_recomputed'] else 'no'} | {'PASS' if item['training_gate_pass'] else 'FAIL'} | {item['first_recorded_gate_step'] or 'not reached'} |")
    report.extend([
        "",
        "The duration extension is therefore valid as an acquisition/competence diagnostic. It does not, by itself, adjudicate whether graph topology, path length, effective resistance or a learned multi-hop mechanism caused the original transfer differences.",
    ])
    AUDIT_MD.write_text("\n".join(report) + "\n", encoding="utf-8")

    hash_paths = [AUDIT_JSON, AUDIT_MD]
    current_hashes = load(HASHES)
    for path in hash_paths:
        current_hashes[rel(path)] = sha256(path)
    HASHES.write_text(json.dumps(current_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        print(json.dumps({"status": "failed", "failures": failures}, indent=2))
        return 1
    print(json.dumps({"status": "completed", "verification_pass": True, "arms": len(result_checks), "training_gate_passes": sum(int(item["training_gate_pass"]) for item in result_checks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
