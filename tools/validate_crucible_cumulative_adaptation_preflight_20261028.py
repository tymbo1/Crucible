#!/usr/bin/env python3
"""Validate the CPU-only cumulative-adaptation preflight artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_cumulative_adaptation_preflight_20261028"


def load(name: str) -> Any:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    errors = []
    report = load("PREFLIGHT_REPORT.json")
    config = load("RUN_CONFIG.json")
    leakage = load("LEAKAGE_AUDIT.json")
    budget = load("BUDGET_MATCHING.json")
    promotion = load("PROMOTION_FEASIBILITY.json")
    statistics = load("STATISTICAL_FEASIBILITY.json")
    fixture = load("SYNTHETIC_FIXTURE_RESULTS.json")
    replay = load("REPLAY_COMPARISON.json")
    hashes = load("HASHES.json")

    if report.get("scientific_execution_approved") is not False:
        errors.append("scientific execution is not explicitly unapproved")
    if config.get("scientific_run_approved") is not False or config.get("gpu_allocation_approved") is not False:
        errors.append("run configuration contains an approval flag")
    if not leakage.get("passed") or leakage.get("target_answer_labels_written") or leakage.get("target_ast_written") or leakage.get("target_evaluations_run"):
        errors.append("leakage or target-access guard failed")
    if not budget.get("all_equal") or budget.get("target_task_seed_units_total") != 288:
        errors.append("budget matching or denominator is incorrect")
    if not promotion.get("allocation_feasible") or promotion.get("minimum_all_success_evidence", {}).get("trials") != 16:
        errors.append("promotion feasibility does not match the registered threshold")
    if statistics.get("paired_units") != 288 or statistics.get("primary_test") != "exact_two_sided_mcnemar":
        errors.append("statistical plan is not the registered paired design")
    if not fixture.get("schema_pass") or not fixture.get("promotion_decision") or not fixture.get("adaptive_score_exceeds_controls"):
        errors.append("synthetic update fixture failed")
    if not replay.get("identical"):
        errors.append("deterministic replay failed")
    for relative, expected in sorted(hashes.items()):
        path = ROOT / relative
        if not path.exists():
            errors.append(f"missing hashed artifact: {relative}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            errors.append(f"hash mismatch: {relative}")
    result: Dict[str, Any] = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "scientific_execution_approved": False,
        "target_answers_accessed": False,
        "gpu_allocated": False,
    }
    print(json.dumps(result, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
