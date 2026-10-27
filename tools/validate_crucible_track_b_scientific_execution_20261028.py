#!/usr/bin/env python3
"""Validate the completed, approved Track B B0--B3 execution artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_track_b_scientific_execution_20261028"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_TRACK_B_NEURAL_PROPOSER_PROTOCOL_20261028.json"
AMENDMENT = ROOT / "docs" / "CRUCIBLE_TRACK_B_NEURAL_PROPOSER_PROTOCOL_FREEZE_AMENDMENT_20261028.json"
ARMS = {
    "B0_V0_DETERMINISTIC_RULE_FILTER",
    "B1_DETERMINISTIC_AST_SEARCH",
    "B2_LEARNED_FIXED_FAMILY_PROPOSER",
    "B3_LEARNED_COMPOSITIONAL_PROPOSER",
}


def load_json(name: str) -> Any:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def load_jsonl(name: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in (OUTPUT / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    errors: List[str] = []
    if not OUTPUT.is_dir():
        print(json.dumps({"status": "fail", "errors": [f"missing output: {OUTPUT}"]}, indent=2))
        return 1
    config = load_json("RUN_CONFIG.json")
    report = load_json("SCIENTIFIC_REPORT.json")
    comparison = load_json("PRIMARY_COMPARISON.json")
    rows = load_jsonl("TASK_RESULTS.jsonl")
    summaries = load_json("ARM_SUMMARIES.json")
    training = load_jsonl("TRAINING_SUMMARY.jsonl")
    hashes = load_json("HASHES.json")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))

    if config.get("status") != "completed" or not config.get("scientific_execution_started"):
        errors.append("run configuration is not a completed scientific execution")
    if not config.get("sealed_test_accessed") or config.get("sealed_test_access_phase") != "after_all_final_checkpoints":
        errors.append("sealed-test phase boundary was not recorded correctly")
    if protocol.get("run_approved") is not True or protocol.get("compute_budget", {}).get("run_approved") is not True:
        errors.append("original protocol approval flags are not true")
    if amendment.get("status") != "approved_for_execution" or amendment.get("scientific_execution_approved") is not True:
        errors.append("amendment approval flags are not true")
    if report.get("status") != "completed" or report.get("protocol_integrity") != "pass":
        errors.append("scientific report is not complete and protocol-clean")
    if report.get("no_cumulative_learning_library") is not True:
        errors.append("cumulative-learning-library experiment was not excluded")

    if len(rows) != 448:
        errors.append(f"expected 448 task rows, found {len(rows)}")
    if {row.get("arm") for row in rows} != ARMS:
        errors.append("task results do not contain exactly the four registered arms")
    if {row.get("seed") for row in rows} != {42, 1337}:
        errors.append("task results do not contain exactly the registered seeds")
    if sum(row.get("split") == "train" for row in rows) != 256 or sum(row.get("split") in {"validation", "test"} for row in rows) != 192:
        errors.append("task split row counts do not match 32/12/12 per seed and arm")
    if len(training) != 4:
        errors.append("expected four final learned-arm checkpoint summaries")
    for item in training:
        if item.get("updates") != 2000 or not item.get("final_update_fixed"):
            errors.append(f"non-final checkpoint rule in {item.get('arm')} seed {item.get('seed')}")
        checkpoint = ROOT / item["checkpoint"]
        if not checkpoint.is_file() or sha256(checkpoint) != item.get("checkpoint_sha256"):
            errors.append(f"checkpoint hash mismatch: {item.get('checkpoint')}")

    for item in summaries.get("per_arm_seed", []):
        if item.get("revision", {}).get("status") != "pass" or not item.get("interference", {}).get("pass"):
            errors.append(f"lifecycle revision/interference failure: {item.get('arm')} seed {item.get('seed')}")
    if comparison.get("candidate_budget_matched") is not True or comparison.get("verifier_call_cap_matched") is not True:
        errors.append("B1/B3 resource matching flags are not true")
    if comparison.get("b3_target_function_acquisition_count") != 0 or comparison.get("b1_target_function_acquisition_count") != 2:
        errors.append("primary B1/B3 acquisition result no longer matches the recorded run")
    if report.get("hypotheses", {}).get("H3_procedural_novelty", {}).get("successful_task_count") != 0:
        errors.append("H3 report is inconsistent with task results")
    gpu_seconds = float(report.get("resource_limit", {}).get("measured_gpu_seconds", 0.0))
    if gpu_seconds > 2 * 60 * 60:
        errors.append("measured GPU time exceeded the two-hour cap")

    for relative, expected in hashes.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"artifact hash mismatch: {relative}")

    result = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "checks": {
            "approved": protocol.get("run_approved") is True and amendment.get("scientific_execution_approved") is True,
            "completed": config.get("status") == "completed",
            "sealed_phase_after_checkpoints": config.get("sealed_test_access_phase") == "after_all_final_checkpoints",
            "rows": len(rows),
            "final_checkpoints": len(training) == 4 and all(item.get("updates") == 2000 for item in training),
            "lifecycle_revision_and_interference": not any(item.get("revision", {}).get("status") != "pass" or not item.get("interference", {}).get("pass") for item in summaries.get("per_arm_seed", [])),
            "artifact_hashes": not any("artifact hash mismatch" in error for error in errors),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
