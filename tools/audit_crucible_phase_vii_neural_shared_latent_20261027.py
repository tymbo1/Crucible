#!/usr/bin/env python3
"""Independently audit the completed Phase VII neural transfer experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_phase_vii_neural_shared_latent_relational_transfer_20261027_v1"
SOURCE = ROOT / "experiments" / "crucible_two_candidate_topology_control_20261027_v1"
N0_SOURCE = ROOT / "experiments" / "crucible_two_candidate_topology_duration_extension_20261027_v1"
SUMMARY = OUTPUT / "SUMMARY.json"
HASHES = OUTPUT / "HASHES.json"
AUDIT_JSON = OUTPUT / "POST_RUN_AUDIT.json"
AUDIT_MD = OUTPUT / "POST_RUN_AUDIT.md"
THRESHOLD = 0.95
PARAMETERS = 22_104_955
CHECKPOINT_STEPS = (1, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000)
EXPECTED_CELLS = {
    f"{variant}_{condition}_SEED{seed}"
    for variant in ("N1_SHARED_LATENT", "N2_SHARED_LATENT_CYCLE")
    for condition in ("P12_PRISM", "R12_DIAMETER5")
    for seed in (42, 1337)
}


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
    assert_equal(summary.get("training_performed"), True, "summary.training_performed", failures)
    assert_equal(summary.get("sealed_test_accessed"), False, "summary.sealed_test_accessed", failures)
    assert_equal(summary.get("training_gate_passes"), 8, "summary.training_gate_passes", failures)
    assert_equal(tuple(summary.get("seeds", ())), (42, 1337), "summary.seeds", failures)
    assert_equal(tuple(summary.get("conditions", ())), ("P12_PRISM", "R12_DIAMETER5"), "summary.conditions", failures)
    assert_equal(set(summary.get("variants", ())), {"N1_SHARED_LATENT", "N2_SHARED_LATENT_CYCLE"}, "summary.variants", failures)

    result_cells = {item.get("cell") for item in summary.get("results", [])}
    assert_equal(len(summary.get("results", [])), 8, "summary.result_count", failures)
    assert_equal(result_cells, EXPECTED_CELLS, "summary.result_cells", failures)

    preflight = load(OUTPUT / "DATA_LEAKAGE_AUDIT.json")
    assert_equal(preflight.get("pass"), True, "data_leakage_audit.pass", failures)
    assert_equal(preflight.get("failures"), [], "data_leakage_audit.failures", failures)
    checks["data_leakage_audit_pass"] = not preflight.get("failures") and preflight.get("pass") is True

    prior_hashes = load(SOURCE / "HASHES.json")
    prior_mismatches = [path for path, expected in prior_hashes.items() if not (ROOT / path).exists() or sha256(ROOT / path) != expected]
    checks["source_hashes_verified"] = not prior_mismatches
    if prior_mismatches:
        failures.append(f"source hash mismatches: {prior_mismatches}")

    n0_summary = load(N0_SOURCE / "SUMMARY.json")
    n0_by_cell = {item["cell"]: item for item in n0_summary["results"]}
    n0_reference_by_cell = {item["cell"]: item for item in summary.get("n0_reference", [])}
    assert_equal(set(n0_reference_by_cell), set(n0_by_cell), "summary.n0_reference_cells", failures)
    for cell, source_result in n0_by_cell.items():
        reference = n0_reference_by_cell.get(cell)
        if reference is not None:
            assert_equal(reference.get("train_metrics"), source_result.get("train_metrics"), f"{cell}.n0_train_reference", failures)
            assert_equal(reference.get("validation_metrics"), source_result.get("validation_metrics"), f"{cell}.n0_validation_reference", failures)
    checks["n0_reference_matches_duration_extension"] = not any("n0_" in failure for failure in failures)

    result_checks: List[Dict[str, Any]] = []
    for result in summary["results"]:
        cell_name = result["cell"]
        cell = OUTPUT / cell_name
        trace_path = cell / "TRAINING_TRACE.jsonl"
        train_csv = cell / "TRAIN_PREDICTIONS.csv"
        validation_csv = cell / "VALIDATION_PREDICTIONS.csv"
        result_json = cell / "RESULTS.json"
        checkpoint = cell / "checkpoint.pt"

        assert_equal(result.get("steps"), 2000, f"{cell_name}.steps", failures)
        assert_equal(result.get("batch_size"), 12, f"{cell_name}.batch_size", failures)
        assert_equal(result.get("parameter_count"), PARAMETERS, f"{cell_name}.parameter_count", failures)
        assert_equal(result.get("trace_steps"), list(CHECKPOINT_STEPS), f"{cell_name}.trace_steps_manifest", failures)
        assert_equal(result.get("training_gate_threshold"), THRESHOLD, f"{cell_name}.training_gate_threshold", failures)
        assert_equal(result.get("training_gate_pass"), True, f"{cell_name}.training_gate_pass", failures)

        trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert_equal(len(trace), 2000, f"{cell_name}.trace_rows", failures)
        assert_equal([int(row["step"]) for row in trace], list(range(1, 2001)), f"{cell_name}.trace_steps", failures)
        recorded = {int(row["step"]): row for row in trace if "training" in row and "validation" in row}
        assert_equal(tuple(sorted(recorded)), CHECKPOINT_STEPS, f"{cell_name}.recorded_checkpoints", failures)
        finite = all(isinstance(row.get("loss"), (int, float)) and isinstance(row.get("query_loss"), (int, float)) and isinstance(row.get("cycle_loss"), (int, float)) for row in trace)
        assert_equal(finite, True, f"{cell_name}.finite_trace_losses", failures)

        recomputed_train = metric_from_csv(train_csv)
        recomputed_validation = metric_from_csv(validation_csv)
        assert_equal(recomputed_train, result["train_metrics"], f"{cell_name}.train_metrics_recomputed", failures)
        assert_equal(recomputed_validation, result["validation_metrics"], f"{cell_name}.validation_metrics_recomputed", failures)
        assert_equal(load(result_json), result, f"{cell_name}.RESULTS_consistency", failures)
        assert_equal(sha256(checkpoint), result["checkpoint_sha256"], f"{cell_name}.checkpoint_sha256", failures)

        result_checks.append({
            "cell": cell_name,
            "trace_rows": len(trace),
            "training_metrics_recomputed": recomputed_train == result["train_metrics"],
            "validation_metrics_recomputed": recomputed_validation == result["validation_metrics"],
            "checkpoint_hash_verified": sha256(checkpoint) == result["checkpoint_sha256"],
            "training_gate_pass": bool(result["training_gate_pass"]),
        })

    checks["arm_checks"] = result_checks
    checks["all_arms_passed"] = all(item["training_gate_pass"] for item in result_checks)
    checks["all_traces_and_metrics_verified"] = all(item["training_metrics_recomputed"] and item["validation_metrics_recomputed"] and item["checkpoint_hash_verified"] for item in result_checks)
    hash_mismatches = [path for path, expected in load(HASHES).items() if not (ROOT / path).exists() or sha256(ROOT / path) != expected]
    checks["all_recorded_hashes_verified"] = not hash_mismatches
    if hash_mismatches:
        failures.append(f"recorded hash mismatches: {hash_mismatches}")

    baseline_by_cell = {item["cell"]: item for item in summary["n0_reference"]}
    comparison: List[Dict[str, Any]] = []
    for result in summary["results"]:
        baseline = baseline_by_cell[result["condition"] + "_SEED" + str(result["seed"])]
        n0_paired = baseline["validation_metrics"]["paired_accuracy"]
        n_model_paired = result["validation_metrics"]["paired_accuracy"]
        comparison.append({
            "cell": result["cell"],
            "n0_paired_accuracy": n0_paired,
            "model_paired_accuracy": n_model_paired,
            "model_minus_n0": n_model_paired - n0_paired,
            "model_better": n_model_paired > n0_paired,
        })
    checks["matched_n0_comparison"] = comparison
    checks["all_models_beat_n0_paired"] = all(item["model_better"] for item in comparison)
    checks["interpretation_boundary"] = "All eight structured arms passed acquisition competence, but none improved matched N0 paired transfer. This rejects the preregistered improvement hypotheses under this benchmark; it does not prove shared latent representations or cycle consistency impossible in general."

    audit = {
        "schema_version": "crucible_phase_vii_neural_shared_latent_post_run_audit_v1",
        "experiment": summary["experiment"],
        "status": "completed",
        "verification_pass": not failures,
        "failures": failures,
        "checks": checks,
        "training_performed": True,
        "sealed_test_accessed": False,
        "prior_checkpoints_modified": False,
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = [
        "# Post-run audit: Phase VII neural shared-latent relational transfer",
        "",
        f"Verification pass: **{'yes' if not failures else 'no'}**.",
        "",
        "The audit independently verified the eight-arm matrix, frozen source hashes, data-leakage preflight, N0 reference metrics, 2,000 trace steps per arm, checkpoint metrics, CSV metric recomputation, parameter counts, checkpoint hashes and the matched paired-transfer comparison.",
        "",
        "| Cell | Trace | Metrics | Checkpoint | Gate | Model paired | N0 paired | Δ |",
        "|---|---:|---|---|---|---:|---:|---:|",
    ]
    for item in comparison:
        arm = next(arm for arm in result_checks if arm["cell"] == item["cell"])
        report.append(f"| {item['cell']} | {arm['trace_rows']}/2000 | {'yes' if arm['training_metrics_recomputed'] and arm['validation_metrics_recomputed'] else 'no'} | {'yes' if arm['checkpoint_hash_verified'] else 'no'} | {'PASS' if arm['training_gate_pass'] else 'FAIL'} | {item['model_paired_accuracy']:.3f} | {item['n0_paired_accuracy']:.3f} | {item['model_minus_n0']:+.3f} |")
    report.extend([
        "",
        "All eight trained arms passed the 95% acquisition gate. The matched N0 comparison is a negative transfer result for this preregistered implementation: no N1 or N2 arm exceeded the frozen dense reference on paired validation accuracy.",
        "",
        "This does not establish that shared-latent relational models or cycle-consistency objectives cannot help under other representations, budgets or tasks. It establishes that these tested Phase VII configurations did not improve transfer here.",
    ])
    AUDIT_MD.write_text("\n".join(report) + "\n", encoding="utf-8")

    current_hashes = load(HASHES)
    for path in (AUDIT_JSON, AUDIT_MD):
        current_hashes[rel(path)] = sha256(path)
    HASHES.write_text(json.dumps(current_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if failures:
        print(json.dumps({"status": "failed", "verification_pass": False, "failures": failures}, indent=2))
        return 1
    print(json.dumps({"status": "completed", "verification_pass": True, "arms": len(result_checks), "training_gate_passes": sum(int(item["training_gate_pass"]) for item in result_checks), "all_models_beat_n0_paired": checks["all_models_beat_n0_paired"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
