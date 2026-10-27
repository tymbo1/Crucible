#!/usr/bin/env python3
"""Independently validate the Track B four-arm runner validation package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_track_b_scientific_runner_validation_20261028"
EXPECTED_FILES = {
    "ARM_VALIDATION.jsonl", "COUNTERS.jsonl", "HASHES.json", "LIFECYCLE_VALIDATION.jsonl",
    "PROTOCOL_AUDIT.json", "README.md", "REPLAY_COMPARISON.json", "RUNNER_CONFIG.json",
    "RUNNER_VALIDATION_REPORT.json", "B0_V0_DETERMINISTIC_RULE_FILTER.json",
    "B1_DETERMINISTIC_AST_SEARCH.json", "B2_LEARNED_FIXED_FAMILY_PROPOSER.json",
    "B3_LEARNED_COMPOSITIONAL_PROPOSER.json",
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
    actual = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    if actual != EXPECTED_FILES:
        errors.append(f"output file set mismatch: missing={sorted(EXPECTED_FILES - actual)} unexpected={sorted(actual - EXPECTED_FILES)}")

    report = load_json("RUNNER_VALIDATION_REPORT.json")
    audit = load_json("PROTOCOL_AUDIT.json")
    config = load_json("RUNNER_CONFIG.json")
    replay = load_json("REPLAY_COMPARISON.json")
    arms = load_jsonl("ARM_VALIDATION.jsonl")
    counters = load_jsonl("COUNTERS.jsonl")
    lifecycle = load_jsonl("LIFECYCLE_VALIDATION.jsonl")
    hashes = load_json("HASHES.json")

    if report.get("status") != "pass" or report.get("classification") != "runner_validation_only":
        errors.append("runner validation report is not passing or is misclassified")
    if report.get("scientific_execution_started") or report.get("scientific_execution_approved"):
        errors.append("runner validation claims scientific execution")
    if report.get("validation_or_test_loaded") or report.get("sealed_test_accessed"):
        errors.append("runner validation crossed data isolation boundary")
    if audit.get("status") != "incomplete_for_exact_scientific_execution":
        errors.append("protocol audit did not preserve unresolved execution fields")
    if audit.get("scientific_execution_permitted") or config.get("scientific_run_started"):
        errors.append("validation configuration permits scientific execution")
    if config.get("validation_or_test_loaded") or config.get("sealed_test_accessed") or config.get("training_or_eval_manifest_loaded"):
        errors.append("runner configuration loaded scientific data")
    if not replay.get("deterministic_replay"):
        errors.append("synthetic runner replay failed")

    expected_arms = [
        "B0_V0_DETERMINISTIC_RULE_FILTER",
        "B1_DETERMINISTIC_AST_SEARCH",
        "B2_LEARNED_FIXED_FAMILY_PROPOSER",
        "B3_LEARNED_COMPOSITIONAL_PROPOSER",
    ]
    if [row.get("arm") for row in arms] != expected_arms:
        errors.append("arm validation rows are incomplete or out of order")
    if [row.get("arm") for row in counters] != expected_arms:
        errors.append("counter rows are incomplete or out of order")
    if [row.get("arm") for row in lifecycle] != expected_arms:
        errors.append("lifecycle rows are incomplete or out of order")
    expected_sources = ["deterministic_enum", "deterministic_enum", "catalog_ranked", "neural_generated"]
    for row, source in zip(arms, expected_sources):
        if row.get("sources") != [source] or not row.get("grammar_valid") or row.get("proposal_count", 0) < 1:
            errors.append(f"arm proposal validation failed: {row.get('arm')}")
    for row, source in zip(counters, expected_sources):
        if row.get("candidate_source") != source:
            errors.append(f"counter source mismatch: {row.get('arm')}")
        if row.get("candidate_outputs", 0) < 1 or row.get("candidate_evaluations", 0) < 1:
            errors.append(f"proposal counters missing: {row.get('arm')}")
        if row.get("gpu_seconds") != 0.0:
            errors.append(f"synthetic runner used GPU time: {row.get('arm')}")
    for row in lifecycle:
        if row.get("status") != "validated" or not row.get("accepted") or not row.get("post_reset_correct"):
            errors.append(f"lifecycle validation failed: {row.get('arm')}")
        if not row.get("contradiction_invalidation") or not row.get("revision_verification"):
            errors.append(f"revision gate failed: {row.get('arm')}")
        if row.get("counters", {}).get("candidate_source") != expected_sources[expected_arms.index(row.get("arm"))]:
            errors.append(f"lifecycle counter source mismatch: {row.get('arm')}")

    expected_hashed = {str((OUTPUT / name).relative_to(ROOT)) for name in EXPECTED_FILES - {"HASHES.json", "RUNNER_VALIDATION_REPORT.json"}}
    if set(hashes) != expected_hashed:
        errors.append("runner HASHES.json coverage mismatch")
    for relative, expected in hashes.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"runner artifact hash mismatch: {relative}")

    result = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "checks": {
            "four_arms": len(arms) == 4,
            "deterministic_replay": bool(replay.get("deterministic_replay")),
            "lifecycle_rows": len(lifecycle),
            "all_lifecycle_validated": all(row.get("status") == "validated" for row in lifecycle),
            "scientific_execution_started": bool(report.get("scientific_execution_started")),
            "sealed_test_accessed": bool(report.get("sealed_test_accessed")),
            "protocol_exact_for_science": bool(report.get("protocol_exact_for_science")),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
