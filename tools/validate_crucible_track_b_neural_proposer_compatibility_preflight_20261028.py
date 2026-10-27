#!/usr/bin/env python3
"""Validate the training-only B3 compatibility preflight artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from crucible_track_b_neural_proposer import GrammarConstraint, ast_hash


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_track_b_neural_proposer_compatibility_preflight_20261028"
EXPECTED_FILES = {
    "COMPATIBILITY_REPORT.json", "COUNTERS.jsonl", "HASHES.json", "LIFECYCLE_SMOKE.json",
    "LIFECYCLE_STORE.json", "PROPOSALS.jsonl", "README.md", "REPLAY_COMPARISON.json",
    "RUN_CONFIG.json", "TRAINING_TRACE.jsonl",
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

    report = load_json("COMPATIBILITY_REPORT.json")
    config = load_json("RUN_CONFIG.json")
    replay = load_json("REPLAY_COMPARISON.json")
    smoke = load_json("LIFECYCLE_SMOKE.json")
    counters = load_jsonl("COUNTERS.jsonl")
    proposals = load_jsonl("PROPOSALS.jsonl")
    trace = load_jsonl("TRAINING_TRACE.jsonl")
    hashes = load_json("HASHES.json")

    if report.get("status") != "pass":
        errors.append("compatibility report is not passing")
    if report.get("classification") != "compatibility_only_not_scientific_evidence":
        errors.append("compatibility classification is missing")
    if not report.get("training_only") or report.get("validation_or_test_scored") or report.get("sealed_test_accessed"):
        errors.append("report does not preserve training-only boundary")
    if report.get("scientific_checkpoint_selected") or report.get("scientific_execution_approved"):
        errors.append("report claims checkpoint selection or scientific approval")
    if config.get("device") != "cpu" or config.get("eval_oracle_loaded") or config.get("validation_or_test_scored") or config.get("sealed_test_accessed"):
        errors.append("run configuration violates CPU/training-only boundary")
    if not replay.get("deterministic_replay") or not replay.get("state_hash_equal") or not replay.get("proposal_outputs_equal"):
        errors.append("deterministic replay failed")
    required_smoke = {
        "schema_validation", "persistence_reload", "post_reset_execution_correct",
        "external_contradiction_invalidation", "successor_verification", "successor_active",
    }
    if any(not smoke.get(key) for key in required_smoke):
        errors.append("lifecycle interface smoke gate failed")

    if len(trace) != 50:
        errors.append(f"expected 50 training trace rows, got {len(trace)}")
    if len(counters) != 32:
        errors.append(f"expected 32 per-task counter rows, got {len(counters)}")
    if len(proposals) != 32 * 4:
        errors.append(f"expected 128 direct proposals, got {len(proposals)}")
    grammar_spec = json.loads((ROOT / "experiments/crucible_track_b_neural_proposer_preflight_20261028/GRAMMAR_SPEC.json").read_text(encoding="utf-8"))
    grammar = GrammarConstraint(grammar_spec["max_depth"], grammar_spec["max_nodes"])
    for proposal in proposals:
        if proposal.get("candidate_source") != "neural_generated":
            errors.append("proposal is not labelled neural_generated")
            break
        if not proposal.get("grammar_valid"):
            errors.append("proposal is marked grammar-invalid")
            break
        if any(key in proposal for key in ("target_ast", "target_ast_hash", "target_truth_table_hash")):
            errors.append("proposal contains target information")
            break
        try:
            canonical = grammar.validate(proposal["ast"])
        except Exception as exc:
            errors.append(f"proposal AST failed independent grammar validation: {exc}")
            break
        if ast_hash(canonical) != proposal.get("canonical_ast_hash"):
            errors.append("proposal canonical hash mismatch")
            break

    required_counter_fields = {
        "candidate_outputs", "unique_canonical_ast_outputs", "candidate_evaluations", "verifier_calls",
        "oracle_calls", "proposer_forward_passes", "decoder_tokens", "cpu_seconds", "gpu_seconds",
        "wall_clock_seconds", "peak_memory_bytes",
    }
    for row in counters:
        if not required_counter_fields.issubset(row):
            errors.append("per-task counters are incomplete")
            break
        if row["candidate_outputs"] != 4 or row["candidate_evaluations"] != 4 or row["verifier_calls"] != 4:
            errors.append(f"candidate/verifier count mismatch for {row.get('task_id')}")
            break
        if row["gpu_seconds"] != 0.0:
            errors.append("compatibility preflight used nonzero GPU time")
            break

    # Hashes cover all non-self-referential output artifacts, including the
    # persisted lifecycle smoke store.
    expected_hashed = {str((OUTPUT / name).relative_to(ROOT)) for name in EXPECTED_FILES - {"HASHES.json", "COMPATIBILITY_REPORT.json"}}
    if set(hashes) != expected_hashed:
        errors.append("compatibility HASHES.json coverage mismatch")
    for relative, expected in hashes.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"compatibility artifact hash mismatch: {relative}")

    result = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "checks": {
            "training_updates": len(trace),
            "training_tasks": len(counters),
            "direct_proposals": len(proposals),
            "all_proposals_grammar_valid": not any(not row.get("grammar_valid") for row in proposals),
            "deterministic_replay": bool(replay.get("deterministic_replay")),
            "lifecycle_smoke": all(bool(smoke.get(key)) for key in required_smoke),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
