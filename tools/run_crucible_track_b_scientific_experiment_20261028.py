#!/usr/bin/env python3
"""Validation-first runner for the registered Track B B0--B3 experiment.

The default mode exercises every arm and the lifecycle harness on one
synthetic training-only fixture. Scientific mode is deliberately refused until
the protocol contains the missing machine-readable budgets/configuration and
explicitly sets ``run_approved``. This prevents runner validation from silently
becoming scientific execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from crucible_track_b_neural_proposer import (
    ASTProcedureStore,
    ASTVerifier,
    DirectASTProposer,
    GrammarConstraint,
    RuntimeCounters,
    ast_hash,
    canonical_ast,
    encode_demonstrations,
    evaluate_ast,
    target_token_ids,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "CRUCIBLE_TRACK_B_NEURAL_PROPOSER_PROTOCOL_20261028.json"
AMENDMENT = ROOT / "docs" / "CRUCIBLE_TRACK_B_NEURAL_PROPOSER_PROTOCOL_FREEZE_AMENDMENT_20261028.json"
GRAMMAR = ROOT / "experiments" / "crucible_track_b_neural_proposer_preflight_20261028" / "GRAMMAR_SPEC.json"
OUTPUT = ROOT / "experiments" / "crucible_track_b_scientific_runner_validation_20261028"
SEED = 42
VALIDATION_UPDATES = 20
VALIDATION_CANDIDATE_BUDGET = 64
VALIDATION_VERIFIER_BUDGET = 16


FIXED_RULES: Dict[str, Dict[str, Any]] = {
    "ADD_MOD10": {"op": "add_mod10", "args": [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]},
    "SUB_MOD10": {"op": "sub_mod10", "args": [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]},
    "MUL_MOD10": {"op": "mul_mod10", "args": [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]},
    "FIRST": {"op": "var", "name": "a"},
    "SECOND": {"op": "var", "name": "b"},
    "XOR_MOD10": {"op": "xor_bits_mod10", "args": [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]},
    "MAX": {"op": "max", "args": [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]},
    "MIN": {"op": "min", "args": [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def current_peak_memory_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


REQUIRED_AMENDMENT_RESOLUTIONS = (
    "search_budget.b1_candidate_budget",
    "search_budget.b1_verifier_call_cap",
    "b2_architecture",
    "b3_architecture",
    "optimizer_and_learning_rate",
    "contradiction_fixture",
    "interference_fixture",
    "sealed_test_source",
    "checkpoint_selection_rule",
)


def protocol_audit(protocol: Mapping[str, Any], amendment: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Check the original design plus the dated freeze amendment."""
    if amendment is None:
        missing = list(REQUIRED_AMENDMENT_RESOLUTIONS)
        amendment_status = "missing"
        amendment_approved = False
    else:
        resolutions = amendment.get("resolutions", {})
        missing = [key for key in REQUIRED_AMENDMENT_RESOLUTIONS if key not in resolutions]
        amendment_status = str(amendment.get("status", "missing"))
        amendment_approved = bool(amendment.get("scientific_execution_approved"))
        if amendment.get("preserved_protocol_invariants", {}).get("arms") != protocol.get("arms"):
            missing.append("preserved_protocol_invariants.arms")
        if amendment.get("preserved_protocol_invariants", {}).get("seeds") != protocol.get("compute_budget", {}).get("seeds"):
            missing.append("preserved_protocol_invariants.seeds")
    exact = not missing and amendment_status == "frozen_pending_execution_approval"
    return {
        "status": "pass" if exact else "incomplete_for_exact_scientific_execution",
        "protocol_status": protocol.get("status"),
        "run_approved": bool(protocol.get("run_approved")),
        "amendment_status": amendment_status,
        "scientific_execution_approved": amendment_approved,
        "missing_machine_readable_fields": missing,
        "scientific_execution_permitted": exact and bool(protocol.get("run_approved")) and amendment_approved,
        "note": "The dated amendment resolves configuration fields; separate execution approval remains required.",
    }


def synthetic_fixture() -> Dict[str, Any]:
    """One fully specified training-only task used to exercise all arm APIs."""
    demonstrations = [
        {"a": 0, "b": 1, "output": 0}, {"a": 2, "b": 7, "output": 2},
        {"a": 9, "b": 4, "output": 9}, {"a": 4, "b": 2, "output": 4},
        {"a": 7, "b": 0, "output": 7}, {"a": 3, "b": 8, "output": 3},
        {"a": 6, "b": 5, "output": 6}, {"a": 8, "b": 9, "output": 8},
    ]
    verifier_pairs = [(1, 9), (5, 2)]
    query_pairs = [(9, 3), (2, 8)]
    target = canonical_ast(FIXED_RULES["FIRST"])
    return {
        "task_id": "runner_validation_first",
        "split": "synthetic_training_fixture",
        "stratum": "fixed_family_integration",
        "demonstrations": demonstrations,
        "verifier_examples": [{"a": a, "b": b, "output": evaluate_ast(target, a, b)} for a, b in verifier_pairs],
        "query_examples": [{"a": a, "b": b, "output": evaluate_ast(target, a, b)} for a, b in query_pairs],
        "target_ast": target,
        "target_rule": "FIRST",
        "contradiction": {"a": 1, "b": 9, "observed_output": 7, "source": "synthetic_external_event"},
        "revision_ast": canonical_ast(FIXED_RULES["SECOND"]),
        "revision_examples": [{"a": 1, "b": 9, "output": 9}, {"a": 5, "b": 2, "output": 2}],
    }


def compatible(ast: Mapping[str, Any], examples: Sequence[Mapping[str, int]]) -> bool:
    return all(evaluate_ast(ast, int(row["a"]), int(row["b"])) == int(row["output"]) for row in examples)


def fixed_filter(task: Mapping[str, Any], grammar: GrammarConstraint) -> Tuple[List[Dict[str, Any]], RuntimeCounters]:
    counters = RuntimeCounters(candidate_source="deterministic_enum")
    proposals: List[Dict[str, Any]] = []
    for name, ast in FIXED_RULES.items():
        canonical = grammar.validate(ast)
        counters.candidate_outputs += 1
        counters.unique_canonical_ast_outputs += 1
        counters.candidate_evaluations += 1
        if compatible(canonical, task["demonstrations"]):
            proposals.append({"candidate_source": "deterministic_enum", "rule_name": name, "ast": canonical, "canonical_ast_hash": ast_hash(canonical), "confidence": 1.0})
    return proposals, counters


def bounded_ast_enumeration(grammar: GrammarConstraint, budget: int) -> List[Dict[str, Any]]:
    """Deterministic bounded enumeration; no target catalogue is supplied."""
    leaves: List[Dict[str, Any]] = [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]
    leaves.extend({"op": "const", "value": value} for value in range(10))
    output: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(ast: Mapping[str, Any]) -> bool:
        canonical = grammar.validate(ast)
        key = ast_hash(canonical)
        if key in seen:
            return False
        seen.add(key)
        output.append(canonical)
        return len(output) >= budget

    for leaf in leaves:
        if add(leaf):
            return output
    frontier = leaves
    for _depth in range(1, grammar.max_depth + 1):
        next_frontier: List[Dict[str, Any]] = []
        for operator in ("add_mod10", "sub_mod10", "mul_mod10", "xor_bits_mod10", "min", "max"):
            for left in frontier:
                for right in frontier:
                    candidate = {"op": operator, "args": [left, right]}
                    if grammar.max_nodes < 3:
                        continue
                    if add(candidate):
                        return output
                    next_frontier.append(canonical_ast(candidate))
        frontier = next_frontier
        if not frontier:
            break
    return output


def ast_search(task: Mapping[str, Any], grammar: GrammarConstraint, budget: int) -> Tuple[List[Dict[str, Any]], RuntimeCounters]:
    counters = RuntimeCounters(candidate_source="deterministic_enum")
    proposals: List[Dict[str, Any]] = []
    for ast in bounded_ast_enumeration(grammar, budget):
        counters.candidate_outputs += 1
        counters.unique_canonical_ast_outputs += 1
        counters.candidate_evaluations += 1
        if compatible(ast, task["demonstrations"]):
            proposals.append({"candidate_source": "deterministic_enum", "ast": ast, "canonical_ast_hash": ast_hash(ast), "confidence": 1.0})
            break
    return proposals, counters


class CatalogueScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8 * 30, 64), nn.Tanh(), nn.Linear(64, len(FIXED_RULES)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def train_catalogue_scorer(task: Mapping[str, Any], steps: int) -> CatalogueScorer:
    set_seed(SEED)
    model = CatalogueScorer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    inputs = encode_demonstrations(task["demonstrations"]).unsqueeze(0)
    target = torch.tensor([list(FIXED_RULES).index(task["target_rule"])], dtype=torch.long)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), target)
        loss.backward()
        optimizer.step()
    return model.eval()


def catalogue_propose(model: CatalogueScorer, task: Mapping[str, Any], grammar: GrammarConstraint, budget: int) -> Tuple[List[Dict[str, Any]], RuntimeCounters]:
    counters = RuntimeCounters(candidate_source="catalog_ranked")
    with torch.no_grad():
        logits = model(encode_demonstrations(task["demonstrations"]).unsqueeze(0))[0]
        probabilities = torch.softmax(logits, dim=0)
        order = torch.argsort(logits, descending=True).tolist()[:budget]
    proposals = []
    for index in order:
        name = list(FIXED_RULES)[int(index)]
        ast = grammar.validate(FIXED_RULES[name])
        counters.candidate_outputs += 1
        counters.unique_canonical_ast_outputs += 1
        counters.candidate_evaluations += 1
        proposals.append({"candidate_source": "catalog_ranked", "rule_name": name, "ast": ast, "canonical_ast_hash": ast_hash(ast), "confidence": float(probabilities[int(index)])})
    counters.proposer_forward_passes = 1
    return proposals, counters


def train_direct_proposer(task: Mapping[str, Any], grammar: GrammarConstraint, steps: int) -> DirectASTProposer:
    set_seed(SEED)
    model = DirectASTProposer(grammar)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    inputs = encode_demonstrations(task["demonstrations"]).unsqueeze(0)
    targets = target_token_ids(task["target_ast"], grammar).unsqueeze(0)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
    return model.eval()


def direct_propose(model: DirectASTProposer, task: Mapping[str, Any], grammar: GrammarConstraint, budget: int) -> Tuple[List[Dict[str, Any]], RuntimeCounters]:
    counters = RuntimeCounters(candidate_source="neural_generated")
    with torch.no_grad():
        logits = model(encode_demonstrations(task["demonstrations"]).unsqueeze(0))[0]
        generated = model.generate(logits, candidate_count=budget)
    proposals = []
    counters.proposer_forward_passes = 1
    for candidate in generated:
        counters.candidate_outputs += 1
        counters.unique_canonical_ast_outputs += 1
        counters.candidate_evaluations += 1
        counters.decoder_tokens += candidate["decoder_tokens"]
        proposals.append({
            "candidate_source": "neural_generated",
            "candidate_index": candidate["candidate_index"],
            "ast": candidate["ast"],
            "canonical_ast_hash": candidate["canonical_ast_hash"],
            "confidence": candidate["confidence"],
        })
    return proposals, counters


def lifecycle_run(
    arm: str,
    task: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
    grammar: GrammarConstraint,
    store_path: Path,
    proposal_counters: Mapping[str, Any],
) -> Dict[str, Any]:
    verifier = ASTVerifier(grammar)
    counters = RuntimeCounters(candidate_source=str(proposal_counters.get("candidate_source", "neural_generated")))
    for field in (
        "candidate_outputs", "unique_canonical_ast_outputs", "candidate_evaluations", "proposer_forward_passes",
        "decoder_tokens", "cpu_seconds", "gpu_seconds", "wall_clock_seconds", "peak_memory_bytes",
    ):
        if field in proposal_counters:
            setattr(counters, field, proposal_counters[field])
    accepted: Optional[Mapping[str, Any]] = None
    checks = 0
    for proposal in proposals:
        counters.verifier_calls += 1
        ok, used = verifier.verify(proposal["ast"], task["verifier_examples"])
        counters.oracle_calls += used
        checks += used
        if ok:
            accepted = proposal
            break
    if accepted is None:
        return {"arm": arm, "status": "abstained", "accepted": False, "verifier_checks": checks, "counters": counters.as_dict()}

    store = ASTProcedureStore(store_path)
    store.promote(arm, accepted["ast"], {"source": "runner_validation", "candidate_source": accepted["candidate_source"]})
    reloaded = ASTProcedureStore(store_path)
    query_outputs = [reloaded.execute(arm, row["a"], row["b"]) for row in task["query_examples"]]
    expected_outputs = [row["output"] for row in task["query_examples"]]
    post_reset_correct = query_outputs == expected_outputs
    contradiction_recorded = reloaded.invalidate(arm, task["contradiction"]["source"])
    revision_ok, revision_checks = verifier.verify(task["revision_ast"], task["revision_examples"])
    if revision_ok:
        reloaded.promote(arm, task["revision_ast"], {"source": "runner_validation_revision"})
    return {
        "arm": arm,
        "status": "validated" if post_reset_correct and contradiction_recorded and revision_ok else "failed_lifecycle_gate",
        "accepted": True,
        "accepted_hash": accepted["canonical_ast_hash"],
        "accepted_candidate_source": accepted["candidate_source"],
        "verifier_checks": checks,
        "post_reset_execution": query_outputs,
        "post_reset_expected": expected_outputs,
        "post_reset_correct": post_reset_correct,
        "contradiction_invalidation": contradiction_recorded,
        "revision_verification": revision_ok,
        "revision_checks": revision_checks,
        "store_path": str(store_path.relative_to(ROOT)),
        "counters": counters.as_dict(),
    }


def run_validation_once(grammar: GrammarConstraint, output_root: Path) -> Dict[str, Any]:
    task = synthetic_fixture()
    arms: List[Dict[str, Any]] = []

    b0, b0c = fixed_filter(task, grammar)
    arms.append({"arm": "B0_V0_DETERMINISTIC_RULE_FILTER", "proposals": b0, "counters": b0c.as_dict()})

    b1, b1c = ast_search(task, grammar, VALIDATION_CANDIDATE_BUDGET)
    arms.append({"arm": "B1_DETERMINISTIC_AST_SEARCH", "proposals": b1, "counters": b1c.as_dict()})

    b2_model = train_catalogue_scorer(task, VALIDATION_UPDATES)
    b2, b2c = catalogue_propose(b2_model, task, grammar, 4)
    arms.append({"arm": "B2_LEARNED_FIXED_FAMILY_PROPOSER", "proposals": b2, "counters": b2c.as_dict()})

    b3_model = train_direct_proposer(task, grammar, VALIDATION_UPDATES)
    b3, b3c = direct_propose(b3_model, task, grammar, 4)
    arms.append({"arm": "B3_LEARNED_COMPOSITIONAL_PROPOSER", "proposals": b3, "counters": b3c.as_dict()})

    lifecycle: List[Dict[str, Any]] = []
    for arm in arms:
        lifecycle.append(lifecycle_run(arm["arm"], task, arm["proposals"], grammar, output_root / f"{arm['arm']}.json", arm["counters"]))
    normalized = []
    for arm, life in zip(arms, lifecycle):
        normalized.append({
            "arm": arm["arm"],
            "proposal_hashes": [proposal["canonical_ast_hash"] for proposal in arm["proposals"]],
            "candidate_source": [proposal["candidate_source"] for proposal in arm["proposals"]],
            "lifecycle_status": life["status"],
            "post_reset_correct": life.get("post_reset_correct", False),
            "revision_verification": life.get("revision_verification", False),
        })
    return {"arms": arms, "lifecycle": lifecycle, "normalized": normalized, "task_id": task["task_id"]}


def safe_prepare_output() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    allowed = {
        "ARM_VALIDATION.jsonl", "COUNTERS.jsonl", "HASHES.json", "LIFECYCLE_VALIDATION.jsonl",
        "PROTOCOL_AUDIT.json", "README.md", "REPLAY_COMPARISON.json", "RUNNER_CONFIG.json",
        "RUNNER_VALIDATION_REPORT.json", "B0_V0_DETERMINISTIC_RULE_FILTER.json",
        "B1_DETERMINISTIC_AST_SEARCH.json", "B2_LEARNED_FIXED_FAMILY_PROPOSER.json",
        "B3_LEARNED_COMPOSITIONAL_PROPOSER.json",
    }
    existing = list(OUTPUT.iterdir())
    unexpected = [path.name for path in existing if path.name not in allowed or not path.is_file()]
    if unexpected:
        raise SystemExit(f"refusing to overwrite unexpected runner-validation paths: {unexpected}")
    for path in existing:
        path.unlink()


def run_validate() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8")) if AMENDMENT.exists() else None
    grammar_spec = json.loads(GRAMMAR.read_text(encoding="utf-8"))
    grammar = GrammarConstraint(grammar_spec["max_depth"], grammar_spec["max_nodes"])
    audit = protocol_audit(protocol, amendment)
    safe_prepare_output()

    first = run_validation_once(grammar, OUTPUT)
    second = run_validation_once(grammar, OUTPUT)
    replay_equal = first["normalized"] == second["normalized"]
    write_json(OUTPUT / "PROTOCOL_AUDIT.json", audit)
    write_json(OUTPUT / "REPLAY_COMPARISON.json", {"seed": SEED, "deterministic_replay": replay_equal, "normalized_equal": replay_equal})
    write_json(OUTPUT / "RUNNER_CONFIG.json", {
        "mode": "validate_only",
        "device": "cpu",
        "synthetic_fixture": True,
        "training_or_eval_manifest_loaded": False,
        "validation_or_test_loaded": False,
        "sealed_test_accessed": False,
        "scientific_run_started": False,
        "protocol_sha256": sha256(PROTOCOL),
        "amendment_sha256": sha256(AMENDMENT) if AMENDMENT.exists() else None,
        "amendment_status": audit["amendment_status"],
        "grammar_sha256": sha256(GRAMMAR),
        "arms": ["B0_V0_DETERMINISTIC_RULE_FILTER", "B1_DETERMINISTIC_AST_SEARCH", "B2_LEARNED_FIXED_FAMILY_PROPOSER", "B3_LEARNED_COMPOSITIONAL_PROPOSER"],
    })
    write_jsonl(OUTPUT / "ARM_VALIDATION.jsonl", [{"arm": arm["arm"], "proposal_count": len(arm["proposals"]), "sources": sorted({proposal["candidate_source"] for proposal in arm["proposals"]}), "grammar_valid": all(bool(grammar.validate(proposal["ast"])) for proposal in arm["proposals"])} for arm in first["arms"]])
    write_jsonl(OUTPUT / "COUNTERS.jsonl", [{"arm": arm["arm"], **arm["counters"]} for arm in first["arms"]])
    write_jsonl(OUTPUT / "LIFECYCLE_VALIDATION.jsonl", first["lifecycle"])
    for life in first["lifecycle"]:
        store_path = ROOT / life["store_path"]
        if store_path.exists():
            # Keep each persisted store as an auditable fixture artifact.
            pass
    all_lifecycle_pass = all(item["status"] == "validated" for item in first["lifecycle"])
    report = {
        "status": "pass" if replay_equal and all_lifecycle_pass else "fail",
        "classification": "runner_validation_only",
        "scientific_execution_started": False,
        "scientific_execution_approved": False,
        "protocol_exact_for_science": audit["status"] == "pass",
        "missing_protocol_fields": audit["missing_machine_readable_fields"],
        "four_arm_interfaces_exercised": len(first["arms"]) == 4,
        "direct_b3_exercised": any(proposal["candidate_source"] == "neural_generated" for proposal in first["arms"][3]["proposals"]),
        "deterministic_replay": replay_equal,
        "lifecycle_gates_passed_on_synthetic_fixture": all_lifecycle_pass,
        "validation_or_test_loaded": False,
        "sealed_test_accessed": False,
        "note": "A passing runner validation report does not authorize or constitute the registered B0-B3 scientific run.",
    }
    write_json(OUTPUT / "RUNNER_VALIDATION_REPORT.json", report)
    (OUTPUT / "README.md").write_text(
        "# Track B scientific runner validation\n\n"
        "This is a CPU-only synthetic-fixture validation of the B0, B1, B2 and B3 runner interfaces, lifecycle harness, deterministic replay and isolation guards. It does not load the training manifest, validation/test records or sealed oracle, and it cannot authorize scientific execution. The protocol audit records machine-readable fields that must be frozen before the registered experiment can run.\n",
        encoding="utf-8",
    )
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name not in {"HASHES.json", "RUNNER_VALIDATION_REPORT.json"}:
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    print(json.dumps({"status": report["status"], "output": str(OUTPUT), "scientific_execution_started": False, "protocol_audit": audit["status"]}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", help="run the synthetic runner validation (default)")
    parser.add_argument("--scientific", action="store_true", help="reserved for a separately approved run")
    args = parser.parse_args()
    if args.scientific:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        amendment = json.loads(AMENDMENT.read_text(encoding="utf-8")) if AMENDMENT.exists() else None
        audit = protocol_audit(protocol, amendment)
        raise SystemExit(json.dumps({"status": "refused", "reason": "scientific execution requires complete machine-readable protocol and explicit run_approved=true", "protocol_audit": audit}, sort_keys=True))
    return run_validate()


if __name__ == "__main__":
    raise SystemExit(main())
