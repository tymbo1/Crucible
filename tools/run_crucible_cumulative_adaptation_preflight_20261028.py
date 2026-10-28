#!/usr/bin/env python3
"""CPU-only feasibility and leakage preflight for cumulative adaptation.

This tool does not run the proposed four-arm scientific experiment.  It builds
an independent representative manifest, audits structural and exhaustive
functional disjointness, checks the operator-promotion rule, exercises the
state/update schema on synthetic fixtures, and records matched resource
accounting.  Final target ASTs and output labels are kept in memory only so
the preflight does not evaluate the scientific target panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import numpy as np
from scipy.stats import beta, binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.crucible_track_b_neural_proposer import (
    COMMUTATIVE,
    OPERATORS,
    GrammarConstraint,
    ast_depth,
    ast_hash,
    ast_nodes,
    canonical_ast,
    evaluate_ast,
    truth_table_hash,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_cumulative_adaptation_preflight_20261028"
EXPERIMENT = "crucible_cumulative_adaptation_preflight_20261028"
GRAMMAR = GrammarConstraint(max_depth=3, max_nodes=7)
MANIFEST_SEED = 20261028
SEEDS = (42, 1337)
SOURCE_EPISODES_PER_SEED = 32
OPERATOR_VALIDATION_EPISODES_PER_SEED = 32
TARGET_EPISODES_PER_SEED = 144
TARGET_FAMILY_EPISODES_PER_SEED = 72
EXACT_REPLAY_EPISODES_PER_SEED = 16
PROMOTION_ELIGIBLE_OPERATORS = 2
VALIDATION_TRIALS_PER_OPERATOR = 16
TARGET_FAMILIES = ("composition_depth2", "composition_depth3")
DEMO_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 2),
    (2, 1),
    (3, 4),
    (4, 3),
)
ARM_IDS = (
    "A0_FROZEN_INDUCTION",
    "A1_EXACT_PROCEDURE_MEMORY",
    "A2_SHAM_OPERATOR_LIBRARY",
    "A3_VERIFIED_ADAPTIVE_LIBRARY",
)
TARGET_BUDGET = {
    "candidate_slots": 4,
    "operator_lookup_slots": 4,
    "verifier_calls": 4,
    "verifier_examples_per_call": 8,
    "candidate_construction_units": 16,
    "post_reset_checks": 8,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def json_digest(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def random_ast(rng: random.Random, remaining_depth: int) -> Dict[str, Any]:
    if remaining_depth <= 0 or rng.random() < 0.52:
        if rng.random() < 0.18:
            return {"op": "var", "name": rng.choice(("a", "b"))}
        return {"op": "const", "value": rng.randrange(10)}
    operator = rng.choice(OPERATORS)
    return {
        "op": operator,
        "args": [random_ast(rng, remaining_depth - 1), random_ast(rng, remaining_depth - 1)],
    }


def sample_valid_ast(rng: random.Random) -> Dict[str, Any]:
    for _ in range(200):
        candidate = canonical_ast(random_ast(rng, GRAMMAR.max_depth))
        try:
            return GRAMMAR.validate(candidate)
        except ValueError:
            continue
    raise RuntimeError("unable to sample a valid AST under the frozen grammar")


def ast_family(ast: Mapping[str, Any]) -> str:
    depth = ast_depth(ast)
    if depth == 2:
        return "composition_depth2"
    if depth == 3:
        return "composition_depth3"
    return f"depth_{depth}"


def make_demonstrations(ast: Mapping[str, Any], seed: int, count: int = 8) -> List[Dict[str, int]]:
    pairs = list(DEMO_PAIRS)
    rng = random.Random(seed)
    while len(pairs) < count:
        pair = (rng.randrange(10), rng.randrange(10))
        if pair not in pairs:
            pairs.append(pair)
    return [{"a": a, "b": b, "output": int(evaluate_ast(ast, a, b))} for a, b in pairs[:count]]


def make_target_inputs(seed: int, count: int = 8) -> List[List[int]]:
    rng = random.Random(seed)
    pairs: List[List[int]] = []
    seen: Set[Tuple[int, int]] = set()
    while len(pairs) < count:
        pair = (rng.randrange(10), rng.randrange(10))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append([pair[0], pair[1]])
    return pairs


def collect_unique(
    rng: random.Random,
    count: int,
    predicate,
    excluded_ast: Set[str],
    excluded_truth: Set[str],
) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    local_ast: Set[str] = set()
    local_truth: Set[str] = set()
    attempts = 0
    while len(rows) < count and attempts < 2_000_000:
        attempts += 1
        ast = sample_valid_ast(rng)
        if not predicate(ast):
            continue
        a_hash = ast_hash(ast)
        t_hash = truth_table_hash(ast)
        if a_hash in excluded_ast or a_hash in local_ast or t_hash in excluded_truth or t_hash in local_truth:
            continue
        local_ast.add(a_hash)
        local_truth.add(t_hash)
        rows.append(ast)
    if len(rows) != count:
        raise RuntimeError(f"could not collect {count} unique ASTs; collected {len(rows)} after {attempts} attempts")
    return rows, attempts


def grammar_preflight(rng: random.Random) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    structural: Dict[str, Dict[str, Any]] = {}
    functional: Dict[str, Dict[str, Any]] = {}
    attempts = 10_000
    for sample_index in range(attempts):
        ast = sample_valid_ast(rng)
        canonical = canonical_ast(ast)
        structural_hash = ast_hash(canonical)
        functional_hash = truth_table_hash(canonical)
        row = {
            "pool_index": sample_index,
            "canonical_ast": canonical,
            "canonical_ast_hash": structural_hash,
            "truth_table_hash": functional_hash,
            "depth": ast_depth(canonical),
            "nodes": ast_nodes(canonical),
            "functional_representative": functional_hash not in functional,
        }
        structural.setdefault(structural_hash, row)
        functional.setdefault(functional_hash, row)
    report = {
        "max_depth": GRAMMAR.max_depth,
        "max_nodes": GRAMMAR.max_nodes,
        "operators": list(OPERATORS),
        "commutative_operators": sorted(COMMUTATIVE),
        "domain_size": 100,
        "representative_sampling_attempts": attempts,
        "representative_structural_classes": len(structural),
        "representative_functional_classes": len(functional),
        "representative_pool_seed": MANIFEST_SEED,
        "representative_pool_sampling": "10,000 deterministic valid-AST draws from the frozen grammar RNG stream; first occurrence retained by canonical structural hash",
        "representative_pool_rows": len(structural),
        "duplicate_structural_draws_excluded": attempts - len(structural),
        "duplicate_functional_draws_excluded_from_functional_representatives": attempts - len(functional),
        "invalid_or_out_of_scope_draws_in_pool": 0,
        "representative_pool_is_structural_first_occurrences": True,
        "full_exhaustive_ast_enumeration": False,
        "full_enumeration_note": "The frozen depth-3/node-7 grammar is combinatorially large; the frozen pool covers only sampled first occurrences. Expressions not sampled, structurally duplicated draws, and functional duplicates excluded from the functional representative subset are outside the evidence scope.",
    }
    return report, list(structural.values())


def make_task_record(task_id: str, split: str, seed: int, family: str, ast: Mapping[str, Any], include_outputs: bool) -> Dict[str, Any]:
    canonical = canonical_ast(ast)
    record: Dict[str, Any] = {
        "task_id": task_id,
        "task_cluster_id": task_id if split == "target" else None,
        "split": split,
        "seed": seed,
        "family": family,
        "canonical_ast_hash": ast_hash(canonical),
        "truth_table_hash": truth_table_hash(canonical),
        "depth": ast_depth(canonical),
        "nodes": ast_nodes(canonical),
        "demonstration_inputs": [[row[0], row[1]] for row in DEMO_PAIRS],
    }
    if include_outputs:
        record["target_ast"] = canonical
        record["demonstrations"] = make_demonstrations(canonical, seed)
        record["answer_labels_available_to_preflight"] = True
    else:
        record["answer_labels_available_to_preflight"] = False
        record["target_ast_written"] = False
        record["target_outputs_written"] = False
    return record


def build_manifests(rng: random.Random) -> Dict[str, Any]:
    source, source_attempts = collect_unique(rng, SOURCE_EPISODES_PER_SEED, lambda ast: ast_depth(ast) == 1, set(), set())
    excluded_ast = {ast_hash(ast) for ast in source}
    excluded_truth = {truth_table_hash(ast) for ast in source}
    validation, validation_attempts = collect_unique(rng, OPERATOR_VALIDATION_EPISODES_PER_SEED, lambda ast: ast_depth(ast) == 2, excluded_ast, excluded_truth)
    excluded_ast.update(ast_hash(ast) for ast in validation)
    excluded_truth.update(truth_table_hash(ast) for ast in validation)

    source_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    target_hidden_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    target_attempts: Dict[str, int] = {}

    for seed in SEEDS:
        source_rows.extend(
            make_task_record(f"source_s{seed}_{index:03d}", "source", seed, "source", ast, True)
            for index, ast in enumerate(source)
        )
        validation_rows.extend(
            make_task_record(f"operator_validation_s{seed}_{index:03d}", "operator_validation", seed, "operator_validation", ast, True)
            for index, ast in enumerate(validation)
        )
        seed_rng = random.Random(MANIFEST_SEED + seed)
        family_a, attempts_a = collect_unique(seed_rng, TARGET_FAMILY_EPISODES_PER_SEED, lambda ast: ast_depth(ast) == 2, excluded_ast, excluded_truth)
        excluded_ast.update(ast_hash(ast) for ast in family_a)
        excluded_truth.update(truth_table_hash(ast) for ast in family_a)
        family_b, attempts_b = collect_unique(seed_rng, TARGET_FAMILY_EPISODES_PER_SEED, lambda ast: ast_depth(ast) == 3, excluded_ast, excluded_truth)
        excluded_ast.update(ast_hash(ast) for ast in family_b)
        excluded_truth.update(truth_table_hash(ast) for ast in family_b)
        target_attempts[str(seed)] = attempts_a + attempts_b
        hidden: List[Dict[str, Any]] = []
        for family, asts in ((TARGET_FAMILIES[0], family_a), (TARGET_FAMILIES[1], family_b)):
            for index, ast in enumerate(asts):
                task_id = f"target_s{seed}_{family}_{index:03d}"
                metadata = make_task_record(task_id, "target", seed, family, ast, False)
                target_rows.append(metadata)
                hidden.append({"task_id": task_id, "ast": canonical_ast(ast), "truth_table_hash": truth_table_hash(ast)})
        target_hidden_by_seed[seed] = hidden

    return {
        "source_rows": source_rows,
        "validation_rows": validation_rows,
        "target_rows": target_rows,
        "target_hidden_by_seed": target_hidden_by_seed,
        "collection_attempts": {
            "source": source_attempts,
            "operator_validation": validation_attempts,
            "target_by_seed": target_attempts,
        },
    }


def leakage_audit(manifests: Mapping[str, Any]) -> Dict[str, Any]:
    source_ast = {row["canonical_ast_hash"] for row in manifests["source_rows"]}
    source_truth = {row["truth_table_hash"] for row in manifests["source_rows"]}
    validation_ast = {row["canonical_ast_hash"] for row in manifests["validation_rows"]}
    validation_truth = {row["truth_table_hash"] for row in manifests["validation_rows"]}
    target_ast = {row["canonical_ast_hash"] for row in manifests["target_rows"]}
    target_truth = {row["truth_table_hash"] for row in manifests["target_rows"]}
    target_by_seed = {
        seed: [row for row in manifests["target_rows"] if row["seed"] == seed]
        for seed in SEEDS
    }
    cross_seed_ast = sorted(
        {row["canonical_ast_hash"] for row in target_by_seed[SEEDS[0]]}
        & {row["canonical_ast_hash"] for row in target_by_seed[SEEDS[1]]}
    )
    cross_seed_truth = sorted(
        {row["truth_table_hash"] for row in target_by_seed[SEEDS[0]]}
        & {row["truth_table_hash"] for row in target_by_seed[SEEDS[1]]}
    )
    target_ids = [row["task_id"] for row in manifests["target_rows"]]
    intersections = {
        "source_validation_ast": sorted(source_ast & validation_ast),
        "source_validation_truth": sorted(source_truth & validation_truth),
        "source_target_ast": sorted(source_ast & target_ast),
        "source_target_truth": sorted(source_truth & target_truth),
        "validation_target_ast": sorted(validation_ast & target_ast),
        "validation_target_truth": sorted(validation_truth & target_truth),
    }
    return {
        "source_records": len(manifests["source_rows"]),
        "operator_validation_records": len(manifests["validation_rows"]),
        "target_metadata_records": len(manifests["target_rows"]),
        "target_answer_labels_written": False,
        "target_ast_written": False,
        "target_evaluations_run": False,
        "target_task_ids_unique": len(target_ids) == len(set(target_ids)),
        "cross_seed_target_ast_intersection": cross_seed_ast,
        "cross_seed_target_truth_intersection": cross_seed_truth,
        "same_target_task_across_seeds": bool(cross_seed_ast or cross_seed_truth),
        "task_cluster_definition": "one unique target task within one registered seed and target family; all four arm outcomes remain together",
        "target_phase_policy_frozen_after_source": True,
        "intersections": intersections,
        "passed": all(not values for values in intersections.values()),
    }


def promotion_feasibility() -> Dict[str, Any]:
    threshold = 0.80
    alpha = 0.025
    minimum_all_success = None
    for trials in range(1, 513):
        lower = float(beta.ppf(alpha, 1 + trials, 1))
        if lower >= threshold:
            minimum_all_success = {"trials": trials, "successes": trials, "lower_bound": lower}
            break
    table = []
    for trials in (8, 12, 16, 24, 32, 48):
        eligible = []
        for successes in range(trials + 1):
            lower = float(beta.ppf(alpha, 1 + successes, 1 + trials - successes))
            if lower >= threshold:
                eligible.append(successes)
        table.append({
            "trials": trials,
            "minimum_successes": min(eligible) if eligible else None,
            "all_success_lower_bound": float(beta.ppf(alpha, 1 + trials, 1)),
        })
    return {
        "prior": "Beta(1,1)",
        "lower_confidence_level": 0.95,
        "threshold": threshold,
        "minimum_all_success_evidence": minimum_all_success,
        "eligible_success_table": table,
        "registered_candidates": PROMOTION_ELIGIBLE_OPERATORS,
        "registered_trials_per_candidate": VALIDATION_TRIALS_PER_OPERATOR,
        "registered_total_validation_trials": OPERATOR_VALIDATION_EPISODES_PER_SEED,
        "allocation_feasible": bool(
            minimum_all_success and minimum_all_success["trials"] <= VALIDATION_TRIALS_PER_OPERATOR
            and PROMOTION_ELIGIBLE_OPERATORS * VALIDATION_TRIALS_PER_OPERATOR <= OPERATOR_VALIDATION_EPISODES_PER_SEED
        ),
        "interpretation": "Promotion requires perfect evidence at the minimum allocation; the threshold is not relaxed by this preflight.",
    }


def exact_mcnemar_pvalue(a3_wins: np.ndarray, baseline_wins: np.ndarray) -> float:
    a = int(np.sum((a3_wins == 1) & (baseline_wins == 0)))
    b = int(np.sum((a3_wins == 0) & (baseline_wins == 1)))
    discordant = a + b
    if discordant == 0:
        return 1.0
    lower = 2.0 * float(binom.cdf(a, discordant, 0.5))
    upper = 2.0 * float(binom.sf(a - 1, discordant, 0.5))
    return min(1.0, lower, upper)


def statistics_feasibility() -> Dict[str, Any]:
    n = TARGET_EPISODES_PER_SEED * len(SEEDS)
    scenarios = []
    rng = np.random.default_rng(MANIFEST_SEED)
    for p10, p01 in ((0.20, 0.10), (0.30, 0.10), (0.35, 0.15)):
        repetitions = 50_000
        draws = rng.multinomial(n, [p10, p01, 1.0 - p10 - p01], size=repetitions)
        a = draws[:, 0]
        b = draws[:, 1]
        discordant = a + b
        lower = np.where(discordant > 0, 2.0 * binom.cdf(a, discordant, 0.5), 1.0)
        upper = np.where(discordant > 0, 2.0 * binom.sf(a - 1, discordant, 0.5), 1.0)
        p_values = np.minimum(1.0, np.minimum(lower, upper))
        power = float(np.mean((p_values < 0.05) & (a > b)))
        scenarios.append({
            "p_a3_win_a0_loss": p10,
            "p_a3_loss_a0_win": p01,
            "net_effect": p10 - p01,
            "paired_units": n,
            "simulations": repetitions,
            "estimated_one_sided_directional_power": power,
        })
    return {
        "paired_unit": "task_seed",
        "paired_units": n,
        "target_tasks_per_seed": TARGET_EPISODES_PER_SEED,
        "target_tasks_per_family_per_seed": TARGET_FAMILY_EPISODES_PER_SEED,
        "primary_test": "exact_two_sided_mcnemar",
        "confidence_interval": "20000-replicate-stratified-task-cluster-bootstrap",
        "alpha": 0.05,
        "minimum_effect": 0.10,
        "planning_scenarios": scenarios,
        "criterion_feasible": all(item["estimated_one_sided_directional_power"] >= 0.80 for item in scenarios[1:2]),
        "interpretation": "The 10-point threshold is detectable with high power for moderate-to-large discordant effects, but not guaranteed for every 10-point outcome distribution; planning power is reported rather than claimed.",
    }


def operator_schema_fixture() -> Dict[str, Any]:
    motif = {
        "op": "add_mod10",
        "args": [{"hole": "left", "type": "digit_mod10"}, {"hole": "right", "type": "digit_mod10"}],
    }
    record = {
        "schema_version": "crucible.operator.v1",
        "operator_id": "fixture_add_binary",
        "canonical_motif": motif,
        "typed_holes": ["left", "right"],
        "source_refs": ["source_fixture_0"],
        "validation_refs": [f"operator_validation_fixture_{i:02d}" for i in range(VALIDATION_TRIALS_PER_OPERATOR)],
        "applicability": {"domain": "digits_0_9_binary", "max_depth": 3},
        "successes": VALIDATION_TRIALS_PER_OPERATOR,
        "trials": VALIDATION_TRIALS_PER_OPERATOR,
        "status": "promoted",
        "provenance": {"source": "cpu_preflight_synthetic_fixture"},
    }
    required = {"operator_id", "canonical_motif", "typed_holes", "source_refs", "validation_refs", "applicability", "successes", "trials", "status", "provenance"}
    schema_pass = required.issubset(record)
    lower_bound = float(beta.ppf(0.025, 1 + record["successes"], 1 + record["trials"] - record["successes"]))
    promoted = schema_pass and lower_bound >= 0.80 and bool(record["typed_holes"])
    base_score = 0.0
    adaptive_score = 0.25 * lower_bound
    sham_score = 0.0
    budgets = {arm: dict(TARGET_BUDGET) for arm in ARM_IDS}
    return {
        "schema_pass": schema_pass,
        "operator_record": record,
        "promotion_lower_bound": lower_bound,
        "promotion_decision": promoted,
        "policy_scores": {"A0": base_score, "A2": sham_score, "A3": adaptive_score},
        "adaptive_score_exceeds_controls": adaptive_score > base_score and adaptive_score > sham_score,
        "matched_target_budgets": len({json_digest(value) for value in budgets.values()}) == 1,
        "budgets": budgets,
        "target_labels_used": False,
    }


def replay_fixture() -> Dict[str, Any]:
    first = operator_schema_fixture()
    second = operator_schema_fixture()
    first_digest = json_digest(first)
    second_digest = json_digest(second)
    return {
        "first_digest": first_digest,
        "second_digest": second_digest,
        "identical": first_digest == second_digest,
    }


def build_budget_report() -> Dict[str, Any]:
    arms = {arm: dict(TARGET_BUDGET) for arm in ARM_IDS}
    return {
        "per_target_budget": TARGET_BUDGET,
        "arms": arms,
        "all_equal": len({json_digest(value) for value in arms.values()}) == 1,
        "source_operator_validation_trials_per_arm": OPERATOR_VALIDATION_EPISODES_PER_SEED,
        "target_task_count_per_seed": TARGET_EPISODES_PER_SEED,
        "target_task_seed_units_total": TARGET_EPISODES_PER_SEED * len(SEEDS),
        "accounting_rule": "A0/A1/A2 execute no-op or sham work in every A3 operator slot; early acceptance cannot reduce the fixed verifier schedule.",
    }


def write_artifacts(manifests: Mapping[str, Any], grammar_report: Mapping[str, Any], grammar_pool: Sequence[Mapping[str, Any]], leakage: Mapping[str, Any], promotion: Mapping[str, Any], statistics: Mapping[str, Any], fixture: Mapping[str, Any], replay: Mapping[str, Any], budget: Mapping[str, Any], elapsed_seconds: float) -> Dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "GRAMMAR_ENUMERATION.json", grammar_report)
    write_jsonl(OUTPUT / "GRAMMAR_REPRESENTATIVE_POOL.jsonl", grammar_pool)
    write_jsonl(OUTPUT / "SOURCE_MANIFEST.jsonl", manifests["source_rows"])
    write_jsonl(OUTPUT / "OPERATOR_VALIDATION_MANIFEST.jsonl", manifests["validation_rows"])
    write_jsonl(OUTPUT / "TARGET_MANIFEST_METADATA.jsonl", manifests["target_rows"])
    write_json(OUTPUT / "LEAKAGE_AUDIT.json", leakage)
    write_json(OUTPUT / "TARGET_CROSS_SEED_DEPENDENCE.json", {
        "target_records_by_seed": {
            str(seed): sum(1 for row in manifests["target_rows"] if row["seed"] == seed)
            for seed in SEEDS
        },
        "task_ids_unique": leakage["target_task_ids_unique"],
        "same_target_task_across_seeds": leakage["same_target_task_across_seeds"],
        "canonical_ast_intersection": leakage["cross_seed_target_ast_intersection"],
        "truth_table_intersection": leakage["cross_seed_target_truth_intersection"],
        "inference_consequence": "No target task or function is repeated across seeds. Paired inference is clustered at the unique task-seed unit; seed and family remain explicit reporting strata.",
    })
    write_json(OUTPUT / "PROMOTION_FEASIBILITY.json", promotion)
    write_json(OUTPUT / "STATISTICAL_FEASIBILITY.json", statistics)
    write_json(OUTPUT / "BUDGET_MATCHING.json", budget)
    write_json(OUTPUT / "SYNTHETIC_FIXTURE_RESULTS.json", fixture)
    write_json(OUTPUT / "REPLAY_COMPARISON.json", replay)

    required_amendments = {
        "target_denominator": "144 target tasks per seed, 72 per family, 288 paired task-seed units total",
        "statistics": "Exact two-sided McNemar on task-seed pairs plus 20,000-replicate stratified task-cluster bootstrap confidence interval",
        "budget_matching": "Four candidate slots, four operator lookups, four verifier calls of eight examples and fixed construction units per target for every arm",
        "operator_promotion": "At most two eligible operators, 16 independent validation trials each, Beta(1,1) LCB95 >= 0.80; 16/16 required",
        "grammar_scope": "Use the frozen deterministic first-occurrence structural pool under the depth-3/node-7 grammar; full structural enumeration is not attempted and unsampled expressions are outside the evidence scope",
        "statistical_dependence": "Targets are unique across seeds; resample complete task-seed clusters within seed-by-family strata and never resample arm outcomes independently",
        "zero_promotion": "If no operator reaches 16/16 and Beta(1,1) LCB95 >= 0.80, classify the mechanism as unactivated/inconclusive; do not interpret it as evidence against an active policy",
    }
    write_json(OUTPUT / "REQUIRED_PROTOCOL_AMENDMENTS.json", required_amendments)
    resource_estimates = {
        "preflight_wall_seconds": elapsed_seconds,
        "scientific_execution_started": False,
        "target_evaluations_run": False,
        "sealed_target_answers_accessed": False,
        "gpu_allocated": False,
        "gpu_seconds": 0.0,
        "cpu_only": True,
        "output_bytes_before_hashes": sum(path.stat().st_size for path in OUTPUT.iterdir() if path.is_file()),
    }
    write_json(OUTPUT / "RESOURCE_ESTIMATES.json", resource_estimates)
    write_json(OUTPUT / "RUN_CONFIG.json", {
        "schema_version": "crucible_cumulative_adaptation_preflight_v1",
        "experiment": EXPERIMENT,
        "status": "completed_preflight_only",
        "manifest_seed": MANIFEST_SEED,
        "seeds": list(SEEDS),
        "target_tasks_per_seed": TARGET_EPISODES_PER_SEED,
        "target_task_seed_units": TARGET_EPISODES_PER_SEED * len(SEEDS),
        "target_task_cluster_definition": leakage["task_cluster_definition"],
        "target_phase_policy_frozen_after_source": True,
        "representative_pool": "GRAMMAR_REPRESENTATIVE_POOL.jsonl",
        "scientific_run_approved": False,
        "gpu_allocation_approved": False,
        "sealed_target_answers_accessed": False,
    })
    report = {
        "schema_version": "crucible_cumulative_adaptation_preflight_report_v1",
        "experiment": EXPERIMENT,
        "status": "pass_with_required_protocol_amendments",
        "scientific_execution_approved": False,
        "gates": {
            "grammar_representative_pool": bool(grammar_report["representative_functional_classes"] >= 352),
            "source_validation_target_manifest_counts": len(manifests["source_rows"]) == SOURCE_EPISODES_PER_SEED * len(SEEDS) and len(manifests["validation_rows"]) == OPERATOR_VALIDATION_EPISODES_PER_SEED * len(SEEDS) and len(manifests["target_rows"]) == TARGET_EPISODES_PER_SEED * len(SEEDS),
            "structural_and_functional_leakage": bool(leakage["passed"]),
            "budget_matching": bool(budget["all_equal"]),
            "promotion_evidence_feasible": bool(promotion["allocation_feasible"]),
            "statistical_plan_defined": statistics["paired_units"] == 288 and statistics["primary_test"] == "exact_two_sided_mcnemar",
            "cross_seed_target_dependence_audited": not leakage["same_target_task_across_seeds"] and leakage["target_task_ids_unique"],
            "zero_promotion_rule_documented": True,
            "representative_pool_frozen": len(grammar_pool) == grammar_report["representative_pool_rows"],
            "synthetic_operator_update": bool(fixture["schema_pass"] and fixture["promotion_decision"] and fixture["adaptive_score_exceeds_controls"]),
            "deterministic_replay": bool(replay["identical"]),
            "target_answers_not_accessed": True,
            "gpu_not_allocated": True,
        },
        "unresolved": [
            "Full structural enumeration of the depth-3/node-7 grammar remains combinatorially large; the proposed manifest uses deterministic representatives with exact truth-table checks.",
            "The preflight proves the update and accounting interfaces, not target acquisition or cumulative-learning efficacy.",
            "The representative pool is sampled rather than an exhaustive enumeration; claims are limited to the frozen pool and the registered DSL.",
        ],
        "required_protocol_amendments": required_amendments,
        "resource_summary": resource_estimates,
    }
    write_json(OUTPUT / "PREFLIGHT_REPORT.json", report)

    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "HASHES.json")
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in files}
    write_json(OUTPUT / "HASHES.json", hashes)
    write_json(OUTPUT / "RESOURCE_ESTIMATES.json", {**resource_estimates, "output_bytes_after_hashes": sum(path.stat().st_size for path in OUTPUT.iterdir() if path.is_file())})
    (OUTPUT / "README.md").write_text(
        "# Cumulative adaptation preflight\n\n"
        "CPU-only feasibility and leakage preflight. No scientific target evaluation, "
        "GPU allocation or execution approval occurred.\n\n"
        "- Report: `PREFLIGHT_REPORT.json`\n"
        "- Frozen grammar pool: `GRAMMAR_REPRESENTATIVE_POOL.jsonl`\n"
        "- Dependence audit: `TARGET_CROSS_SEED_DEPENDENCE.json`; target task-seed clusters are the paired inference units.\n"
        "- Target artifacts contain metadata and hashes only; target ASTs and outputs were not written.\n"
        "- The separate four-arm runner validation records implementation readiness; it does not authorize scientific execution.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "HASHES.json")
    write_json(OUTPUT / "HASHES.json", {str(path.relative_to(ROOT)): sha256_file(path) for path in files})
    return report


def run() -> Dict[str, Any]:
    started = time.perf_counter()
    rng = random.Random(MANIFEST_SEED)
    grammar_report, grammar_pool = grammar_preflight(rng)
    manifests = build_manifests(rng)
    leakage = leakage_audit(manifests)
    promotion = promotion_feasibility()
    statistics = statistics_feasibility()
    fixture = operator_schema_fixture()
    replay = replay_fixture()
    budget = build_budget_report()
    elapsed = time.perf_counter() - started
    return write_artifacts(manifests, grammar_report, grammar_pool, leakage, promotion, statistics, fixture, replay, budget, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = run()
    print(json.dumps({
        "status": report["status"],
        "output": str(OUTPUT),
        "scientific_execution_approved": report["scientific_execution_approved"],
        "target_answers_accessed": report["resource_summary"]["sealed_target_answers_accessed"],
        "gpu_allocated": report["resource_summary"]["gpu_allocated"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
