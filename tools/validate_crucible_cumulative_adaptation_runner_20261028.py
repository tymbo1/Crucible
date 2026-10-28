#!/usr/bin/env python3
"""Independently validate the cumulative-adaptation runner readiness package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_cumulative_adaptation_runner_validation_20261028"
AMENDMENT = ROOT / "docs" / "CRUCIBLE_CUMULATIVE_ADAPTATION_PROTOCOL_INTEGRITY_AMENDMENT_20261028.json"
EXPECTED_FILES = {
    "BUDGET_VALIDATION.json",
    "FIXTURE_RESULTS.json",
    "HASHES.json",
    "LEAKAGE_VALIDATION.json",
    "README.md",
    "PROTOCOL_INTEGRITY.json",
    "REPLAY_COMPARISON.json",
    "RESOURCE_ACCOUNTING.json",
    "RUNNER_CONFIG.json",
    "RUNNER_READINESS_REPORT.json",
}


def load_json(name: str) -> Any:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    errors: List[str] = []
    if not OUTPUT.is_dir():
        print(json.dumps({"status": "fail", "errors": [f"missing output: {OUTPUT}"]}, indent=2))
        return 1
    actual = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    if actual != EXPECTED_FILES:
        errors.append(f"output file set mismatch: missing={sorted(EXPECTED_FILES - actual)} unexpected={sorted(actual - EXPECTED_FILES)}")

    report = load_json("RUNNER_READINESS_REPORT.json")
    fixtures = load_json("FIXTURE_RESULTS.json")
    integrity = load_json("PROTOCOL_INTEGRITY.json")
    config = load_json("RUNNER_CONFIG.json")
    budget = load_json("BUDGET_VALIDATION.json")
    leakage = load_json("LEAKAGE_VALIDATION.json")
    resources = load_json("RESOURCE_ACCOUNTING.json")
    replay = load_json("REPLAY_COMPARISON.json")
    hashes = load_json("HASHES.json")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))

    if report.get("implementation_readiness") != "pass" or report.get("classification") != "runner_validation_only":
        errors.append("implementation readiness did not pass or classification is wrong")
    for key in ("scientific_execution_started", "scientific_execution_approved", "target_evaluations_run", "sealed_target_answers_accessed", "gpu_allocated"):
        if report.get(key) is not False:
            errors.append(f"readiness report crosses prohibited boundary: {key}")
    if report.get("scientific_authorization") != "not_authorized":
        errors.append("scientific authorization is not explicitly withheld")
    if integrity.get("status") != "pass" or integrity.get("scientific_execution_approved") is not False:
        errors.append("protocol integrity artifact is not a clean unapproved pass")
    if any(config.get(key) for key in ("scientific_run_started", "training_or_eval_manifest_loaded", "validation_or_test_loaded", "sealed_test_accessed", "gpu_allocated")):
        errors.append("runner configuration reports scientific data or execution")
    if any(resources.get(key) for key in ("gpu_allocated", "scientific_execution_started", "sealed_target_answers_accessed")) or resources.get("gpu_seconds") != 0.0:
        errors.append("resource accounting reports prohibited execution")
    if not budget.get("mismatch_fixture_rejected") or not leakage.get("manifest_collision_fixture_rejected"):
        errors.append("negative integrity fixtures were not rejected")
    if not fixtures.get("positive", {}).get("four_arms") or not fixtures.get("positive", {}).get("lifecycle_pass"):
        errors.append("positive four-arm lifecycle fixture failed")
    if not fixtures.get("negative", {}).get("independent_verification_rejects"):
        errors.append("negative verification fixture failed")
    zero = fixtures.get("zero_promotion", {})
    if zero.get("activation_status") != "unactivated_inconclusive" or zero.get("promoted_operator_count") != 0 or not zero.get("threshold_unchanged"):
        errors.append("zero-promotion fixture does not preserve the registered interpretation")
    if not replay.get("deterministic") or not fixtures.get("replay", {}).get("deterministic"):
        errors.append("deterministic replay failed")
    if amendment.get("status") != "frozen_pending_execution_approval" or amendment.get("scientific_execution_approved") is not False:
        errors.append("protocol-integrity amendment is not frozen and unapproved")

    pool = ROOT / amendment["representative_grammar_pool"]["path"]
    if not pool.is_file() or sha256(pool) != amendment["representative_grammar_pool"]["sha256"]:
        errors.append("frozen representative grammar pool hash mismatch")
    if report.get("frozen_grammar_pool", {}).get("sha256") != amendment["representative_grammar_pool"]["sha256"]:
        errors.append("readiness report does not carry the amendment pool hash")

    expected_hashed = {str((OUTPUT / name).relative_to(ROOT)) for name in EXPECTED_FILES - {"HASHES.json"}}
    if set(hashes) != expected_hashed:
        errors.append("runner HASHES.json coverage mismatch")
    for relative, expected in hashes.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"runner artifact hash mismatch: {relative}")

    result: Dict[str, Any] = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "implementation_readiness": report.get("implementation_readiness"),
        "scientific_authorization": report.get("scientific_authorization"),
        "scientific_execution_started": False,
        "sealed_target_answers_accessed": False,
        "gpu_allocated": False,
        "checks": {
            "four_arms": bool(fixtures.get("positive", {}).get("four_arms")),
            "positive_lifecycle": bool(fixtures.get("positive", {}).get("lifecycle_pass")),
            "negative_verification": bool(fixtures.get("negative", {}).get("independent_verification_rejects")),
            "zero_promotion": zero.get("activation_status") == "unactivated_inconclusive",
            "leakage_rejection": bool(leakage.get("manifest_collision_fixture_rejected")),
            "budget_mismatch_rejection": bool(budget.get("mismatch_fixture_rejected")),
            "deterministic_replay": bool(replay.get("deterministic")),
            "pool_hash_verified": report.get("frozen_grammar_pool", {}).get("sha256") == amendment["representative_grammar_pool"]["sha256"],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
