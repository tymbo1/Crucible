#!/usr/bin/env python3
"""Independently audit the dated Track B protocol-freeze amendment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from crucible_track_b_neural_proposer import DirectASTProposer, GrammarConstraint, TOKEN_LIST
from run_crucible_track_b_scientific_experiment_20261028 import CatalogueScorer


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/CRUCIBLE_TRACK_B_NEURAL_PROPOSER_PROTOCOL_20261028.json"
AMENDMENT = ROOT / "docs/CRUCIBLE_TRACK_B_NEURAL_PROPOSER_PROTOCOL_FREEZE_AMENDMENT_20261028.json"
GRAMMAR = ROOT / "experiments/crucible_track_b_neural_proposer_preflight_20261028/GRAMMAR_SPEC.json"
OUTPUT = ROOT / "experiments/crucible_track_b_protocol_freeze_audit_20261028"
REQUIRED = {
    "search_budget.b1_candidate_budget", "search_budget.b1_verifier_call_cap", "b2_architecture",
    "b3_architecture", "optimizer_and_learning_rate", "contradiction_fixture",
    "interference_fixture", "sealed_test_source", "checkpoint_selection_rule",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    grammar_spec = json.loads(GRAMMAR.read_text(encoding="utf-8"))
    errors: List[str] = []
    notes: List[str] = []

    resolutions = amendment.get("resolutions", {})
    if set(resolutions) != REQUIRED:
        errors.append(f"resolution set mismatch: missing={sorted(REQUIRED - set(resolutions))} extra={sorted(set(resolutions) - REQUIRED)}")
    amendment_status = amendment.get("status")
    if amendment_status not in {"frozen_pending_execution_approval", "approved_for_execution"}:
        errors.append("amendment status is neither frozen_pending_execution_approval nor approved_for_execution")
    if amendment_status == "frozen_pending_execution_approval" and amendment.get("scientific_execution_approved"):
        errors.append("pending amendment unexpectedly authorizes execution")
    if amendment_status == "approved_for_execution" and not amendment.get("scientific_execution_approved"):
        errors.append("approved amendment does not authorize execution")
    invariants = amendment.get("preserved_protocol_invariants", {})
    if invariants.get("arms") != protocol.get("arms"):
        errors.append("arm list changed")
    if invariants.get("seeds") != protocol.get("compute_budget", {}).get("seeds"):
        errors.append("seed list changed")
    if invariants.get("full_updates_per_seed_max") != protocol.get("compute_budget", {}).get("full_updates_per_seed_max"):
        errors.append("update budget changed")
    if invariants.get("full_gpu_hours_total_max") != protocol.get("compute_budget", {}).get("full_gpu_hours_total_max"):
        errors.append("GPU-hour cap changed")
    if invariants.get("grammar") != protocol.get("grammar", {}).get("language"):
        errors.append("grammar changed")
    if not invariants.get("hypotheses_unchanged") or not invariants.get("success_criteria_unchanged"):
        errors.append("amendment does not preserve hypotheses/success criteria")
    if amendment_status == "frozen_pending_execution_approval" and protocol.get("run_approved"):
        errors.append("pending amendment has an approved original protocol")
    if amendment_status == "approved_for_execution" and not protocol.get("run_approved"):
        errors.append("approved amendment lacks an approved original protocol")

    search = resolutions.get("search_budget.b1_candidate_budget", {})
    verifier_budget = resolutions.get("search_budget.b1_verifier_call_cap", {})
    shared = amendment.get("common_resource_accounting", {})
    if search.get("value") != 4 or verifier_budget.get("value") != 4:
        errors.append("B1 budgets do not match the frozen four-candidate comparison")
    if shared.get("candidate_budget") != 4 or shared.get("verifier_call_cap") != 4 or shared.get("verifier_example_budget") != 16 or shared.get("oracle_call_cap") != 64:
        errors.append("shared resource accounting values are inconsistent")

    grammar = GrammarConstraint(grammar_spec["max_depth"], grammar_spec["max_nodes"])
    b2 = CatalogueScorer()
    b2_arch = resolutions["b2_architecture"]
    if b2.net[0].in_features != b2_arch["input_features"] or b2.net[0].out_features != b2_arch["hidden_layers"][0] or b2.net[2].out_features != b2_arch["output_classes"]:
        errors.append("B2 architecture does not match implementation")
    if list(b2_arch["output_order"]) != ["ADD_MOD10", "SUB_MOD10", "MUL_MOD10", "FIRST", "SECOND", "XOR_MOD10", "MAX", "MIN"]:
        errors.append("B2 output order is not the frozen v0 order")

    b3 = DirectASTProposer(grammar)
    b3_arch = resolutions["b3_architecture"]
    if b3.encoder[0].in_features != b3_arch["input_features"] or b3.encoder[0].out_features != b3_arch["hidden_layers"][0] or b3.encoder[2].out_features != b3_arch["hidden_layers"][1]:
        errors.append("B3 encoder architecture does not match implementation")
    if b3.decoder.out_features != b3_arch["max_output_tokens"] * b3_arch["vocabulary_size"]:
        errors.append("B3 decoder dimension does not match frozen grammar")
    if b3_arch["max_depth"] != grammar.max_depth or b3_arch["max_nodes"] != grammar.max_nodes or b3_arch["candidate_budget"] != 4 or b3_arch["candidate_source"] != "neural_generated":
        errors.append("B3 grammar or proposal budget does not match implementation")
    if b3_arch["vocabulary_size"] != len(TOKEN_LIST):
        errors.append("B3 vocabulary size does not match implementation")

    runner_source = (ROOT / "tools/run_crucible_track_b_scientific_experiment_20261028.py").read_text(encoding="utf-8")
    if "lr=0.01" not in runner_source or "lr=0.003" not in runner_source:
        errors.append("frozen optimizer rates are not present in the runner implementation")
    opt = resolutions["optimizer_and_learning_rate"]
    if opt["b2"]["learning_rate"] != 0.01 or opt["b3"]["learning_rate"] != 0.003 or opt["updates"] != 2000:
        errors.append("optimizer amendment values are inconsistent")

    contradiction = resolutions["contradiction_fixture"]
    if contradiction["version_1"]["ast"] != {"op": "var", "name": "a"} or contradiction["version_2"]["ast"] != {"op": "var", "name": "b"}:
        errors.append("contradiction fixture version ASTs are not FIRST then SECOND")
    if not contradiction["externally_triggered"] or contradiction["independent_detection_claim"]:
        errors.append("contradiction fixture improperly claims learned detection")
    if resolutions["interference_fixture"]["probe_pairs"] != [[0, 0], [0, 9], [1, 2], [2, 1], [3, 7], [7, 3], [8, 9], [9, 8]]:
        errors.append("interference probe set changed")

    sealed = resolutions["sealed_test_source"]
    sealed_path = ROOT / sealed["path"]
    if sealed["selector"] != "split == test" or sealed["access_phase"] != "after_final_fixed_checkpoint_only" or not sealed_path.is_file():
        errors.append("sealed-test source or phase is invalid")
    checkpoint = resolutions["checkpoint_selection_rule"]
    if checkpoint["final_update"] != 2000 or checkpoint["early_stopping"] or checkpoint["validation_based_selection"]:
        errors.append("checkpoint rule is not fixed-final-update")

    notes.append("The runner's scientific mode remains refusal-guarded until both approval flags are true.")
    notes.append("The amendment is a dated freeze and not a claim about historical preregistration.")
    notes.append("No sealed-test rows were loaded by this audit; only the sealed source path was checked.")
    audit = {
        "status": "pass" if not errors else "fail",
        "classification": "protocol_freeze_audit_only",
        "errors": errors,
        "notes": notes,
        "scientific_execution_approved": bool(amendment.get("scientific_execution_approved")),
        "implementation_bindings": {
            "b2_architecture": not any("B2 architecture" in error for error in errors),
            "b3_architecture": not any("B3" in error for error in errors),
            "optimizer_rates_present": "lr=0.01" in runner_source and "lr=0.003" in runner_source,
            "runner_scientific_guarded": "from crucible_track_b_scientific_execution_20261028 import run_scientific" in runner_source,
        },
        "source_hashes": {
            "original_protocol": sha256(PROTOCOL),
            "amendment": sha256(AMENDMENT),
            "runner": sha256(ROOT / "tools/run_crucible_track_b_scientific_experiment_20261028.py"),
            "grammar": sha256(GRAMMAR),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    allowed = {"AMENDMENT_AUDIT.json", "HASHES.json", "README.md"}
    existing = list(OUTPUT.iterdir())
    unexpected = [path.name for path in existing if path.name not in allowed or not path.is_file()]
    if unexpected:
        raise SystemExit(f"refusing to overwrite unexpected audit paths: {unexpected}")
    for path in existing:
        path.unlink()
    write_json(OUTPUT / "AMENDMENT_AUDIT.json", audit)
    (OUTPUT / "README.md").write_text(
        "# Track B protocol-freeze amendment audit\n\n"
        "Independent comparison of the dated freeze amendment against the original protocol, grammar, B2/B3 implementation shapes and runner execution guard. No training or sealed-test rows were accessed. The approval state is recorded separately in the machine-readable amendment.\n",
        encoding="utf-8",
    )
    hashes = {}
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    print(json.dumps({"status": audit["status"], "output": str(OUTPUT), "scientific_execution_approved": bool(amendment.get("scientific_execution_approved")), "errors": errors}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
