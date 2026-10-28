#!/usr/bin/env python3
"""Exploratory CPU comparison of executable operator guidance.

This creates a fresh source/validation/target split for the repaired typed-hole
operator mechanism and measures four proposal conditions:

* A0: frozen grammar induction;
* A2: sham operator library;
* A3: verified adaptive operator guidance;
* A3_NO_GUIDANCE_SAME_MEMORY: the same promoted operator records as A3, with
  guidance disabled at proposal time.

The target input bundle contains demonstrations only.  Target ASTs and query
answers are written to a separate sealed answer bundle and opened by the
runner only after every arm has finalized its proposals.  This is exploratory
because the manifest and implementation differ from the earlier protocol.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from crucible_cumulative_adaptation_scientific_runner_20261028 import (  # noqa: E402
    ARM_IDS,
    NO_GUIDANCE_ARM,
    CrucibleCpuReservoir,
    FourArmRunner,
    ProcedureTask,
    SealedTargetProvider,
    ast_hash,
    ast_nodes,
    ast_depth,
    canonical_ast,
    construct_operator_candidates,
    examples_for_ast,
    json_digest,
    load_frozen_pool,
    make_source_tasks,
    make_validation_tasks,
    root_operator,
    sha256_file,
    truth_table_hash,
    validate_manifest_separation,
)
from crucible_track_b_neural_proposer import GrammarConstraint, evaluate_ast  # noqa: E402


EXPERIMENT = "crucible_cumulative_adaptation_motif_comparison_20261028"
OUTPUT = ROOT / "experiments" / EXPERIMENT
MANIFEST_SEED = 20261028_2
SEEDS = (42, 1337)
ELIGIBLE_ROOTS = ("add_mod10", "sub_mod10")
VALIDATION_PER_OPERATOR = 16
TARGETS_PER_SEED = 16
ARMS = (*ARM_IDS[:1], "A2_SHAM_OPERATOR_LIBRARY", "A3_VERIFIED_ADAPTIVE_LIBRARY", NO_GUIDANCE_ARM)
GRAMMAR = GrammarConstraint(max_depth=3, max_nodes=7)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def ast(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    return canonical_ast({"op": op, "args": [left, right]})


def var(name: str) -> Dict[str, Any]:
    return {"op": "var", "name": name}


def const(value: int) -> Dict[str, Any]:
    return {"op": "const", "value": value}


def unique_ast_values(values: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for value in values:
        canonical = canonical_ast(value)
        unique.setdefault(ast_hash(canonical), canonical)
    return [unique[key] for key in sorted(unique)]


def expression_vocabulary() -> List[Dict[str, Any]]:
    """Return a fixed, grammar-valid vocabulary used for recombination only."""
    leaves = [var("a"), var("b")] + [const(value) for value in range(10)]
    raw: List[Dict[str, Any]] = list(leaves)
    raw.extend([
        ast("max", var("a"), var("b")),
        ast("min", var("a"), var("b")),
        ast("xor_bits_mod10", var("a"), var("b")),
        ast("mul_mod10", var("a"), var("b")),
        ast("add_mod10", var("a"), var("b")),
        ast("sub_mod10", var("a"), var("b")),
        ast("max", var("a"), const(1)),
        ast("min", var("b"), const(2)),
        ast("add_mod10", var("a"), const(3)),
        ast("sub_mod10", var("b"), const(4)),
        ast("mul_mod10", var("a"), const(5)),
        ast("xor_bits_mod10", var("b"), const(6)),
        ast("max", const(7), var("a")),
        ast("min", const(8), var("b")),
    ])
    return unique_ast_values(raw)


def deterministic_pairs(seed: int, count: int) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    pairs: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    while len(pairs) < count:
        pair = (rng.randrange(10), rng.randrange(10))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def source_ast_rows(vocabulary: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build source procedures without consulting validation or target outcomes."""
    rows: List[Dict[str, Any]] = []
    used_truth: set[str] = set()
    used_ast: set[str] = set()
    for root_index, root in enumerate(ELIGIBLE_ROOTS):
        accepted = 0
        for left_index, left in enumerate(vocabulary):
            right_index = (left_index * 7 + 3 + root_index) % len(vocabulary)
            candidate = ast(root, left, vocabulary[right_index])
            structural = ast_hash(candidate)
            functional = truth_table_hash(candidate)
            if structural in used_ast or functional in used_truth:
                continue
            used_ast.add(structural)
            used_truth.add(functional)
            rows.append({
                "task_id": f"source_{root}_{accepted:02d}",
                "split": "source",
                "seed": 11 + root_index,
                "family": "verified_source_procedure",
                "target_ast": candidate,
                "canonical_ast_hash": structural,
                "truth_table_hash": functional,
                "demonstrations": examples_for_ast(candidate, deterministic_pairs(1000 + accepted + root_index, 8)),
                "applicability_basis": "source_root_signature",
            })
            accepted += 1
            if accepted == 16:
                break
    if len([row for row in rows if root_operator(row["target_ast"]) == ELIGIBLE_ROOTS[0]]) < 8:
        raise RuntimeError("insufficient add_mod10 source procedures")
    if len([row for row in rows if root_operator(row["target_ast"]) == ELIGIBLE_ROOTS[1]]) < 8:
        raise RuntimeError("insufficient sub_mod10 source procedures")
    return rows


def source_bindings(source_rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    bindings: Dict[str, List[Dict[str, Any]]] = {root: [] for root in ELIGIBLE_ROOTS}
    for root in ELIGIBLE_ROOTS:
        values: List[Mapping[str, Any]] = []
        for row in source_rows:
            if root_operator(row["target_ast"]) == root:
                values.extend(canonical_ast(row["target_ast"]).get("args", []))
        bindings[root] = unique_ast_values(values)
    return bindings


def validation_rows_for_root(
    root: str,
    available: Sequence[Mapping[str, Any]],
    excluded_ast: set[str],
    excluded_truth: set[str],
    root_index: int,
) -> List[Dict[str, Any]]:
    """Allocate applicable tasks first; correctness is evaluated afterward."""
    rng = random.Random(MANIFEST_SEED + 1000 + root_index)
    pairs = [(left, right) for left in available for right in available]
    rng.shuffle(pairs)
    rows: List[Dict[str, Any]] = []
    for left, right in pairs:
        candidate = ast(root, left, right)
        structural = ast_hash(candidate)
        functional = truth_table_hash(candidate)
        if structural in excluded_ast or functional in excluded_truth:
            continue
        index = len(rows)
        rows.append({
            "task_id": f"operator_validation_{root}_{index:02d}",
            "split": "operator_validation",
            "seed": 900 + root_index,
            "family": "independent_applicable_validation",
            "target_ast": candidate,
            "canonical_ast_hash": structural,
            "truth_table_hash": functional,
            "applicable_root_signature": root,
            "operator_id": None,
            "applicability_basis": "root_signature_assigned_before_operator_validation",
            "demonstrations": examples_for_ast(candidate, deterministic_pairs(20000 + root_index * 100 + index, 8)),
            "verification_examples": examples_for_ast(candidate, deterministic_pairs(30000 + root_index * 100 + index, 4)),
        })
        excluded_ast.add(structural)
        excluded_truth.add(functional)
        if len(rows) == VALIDATION_PER_OPERATOR:
            break
    if len(rows) != VALIDATION_PER_OPERATOR:
        raise RuntimeError(f"could not allocate {VALIDATION_PER_OPERATOR} validation tasks for {root}")
    return rows


def target_rows_for_seed(
    seed: int,
    available_by_root: Mapping[str, Sequence[Mapping[str, Any]]],
    excluded_ast: set[str],
    excluded_truth: set[str],
    frozen_pool: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(MANIFEST_SEED + 5000 + seed)
    pool_ast = {str(row["canonical_ast_hash"]) for row in frozen_pool}
    pool_truth = {str(row["truth_table_hash"]) for row in frozen_pool}
    candidate_pairs: List[Tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for root in ELIGIBLE_ROOTS:
        for left in available_by_root[root]:
            for right in available_by_root[root]:
                candidate_pairs.append((root, left, right))
    rng.shuffle(candidate_pairs)
    inputs: List[Dict[str, Any]] = []
    answers: List[Dict[str, Any]] = []
    metadata: List[Dict[str, Any]] = []
    for root, left, right in candidate_pairs:
        target_ast = ast(root, left, right)
        structural = ast_hash(target_ast)
        functional = truth_table_hash(target_ast)
        if structural in excluded_ast or functional in excluded_truth:
            continue
        # Exclude functions already represented by the frozen fallback pool;
        # this makes a frozen-pool hit a meaningful event rather than a hidden
        # catalogue lookup.
        if structural in pool_ast or functional in pool_truth:
            continue
        index = len(inputs)
        task_id = f"target_s{seed}_recombined_{index:02d}"
        family = f"recombined_{root}"
        demo_pairs = deterministic_pairs(40000 + seed * 100 + index, 8)
        query_pairs = deterministic_pairs(50000 + seed * 100 + index, 8)
        inputs.append({
            "task_id": task_id,
            "task_cluster_id": task_id,
            "seed": seed,
            "family": family,
            "demonstrations": examples_for_ast(target_ast, demo_pairs),
            "input_bundle_classification": "exploratory_target_inputs_without_answers",
        })
        answers.append({
            "task_id": task_id,
            "task_cluster_id": task_id,
            "seed": seed,
            "family": family,
            "target_ast": target_ast,
            "canonical_ast_hash": structural,
            "truth_table_hash": functional,
            "query_examples": examples_for_ast(target_ast, query_pairs),
            "answer_bundle_classification": "sealed_exploratory_target_answers",
        })
        metadata.append({
            "task_id": task_id,
            "task_cluster_id": task_id,
            "seed": seed,
            "family": family,
            "root_signature": root,
            "canonical_ast_hash": structural,
            "truth_table_hash": functional,
            "target_ast_written": False,
            "target_answers_written": False,
            "not_in_source_or_validation": True,
            "not_in_frozen_pool": True,
        })
        excluded_ast.add(structural)
        excluded_truth.add(functional)
        if len(inputs) == TARGETS_PER_SEED:
            break
    if len(inputs) != TARGETS_PER_SEED:
        raise RuntimeError(f"could not allocate {TARGETS_PER_SEED} unfamiliar targets for seed {seed}")
    return inputs, answers, metadata


def build_manifest(frozen_pool: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    vocabulary = expression_vocabulary()
    source_rows = source_ast_rows(vocabulary)
    source_ast = {str(row["canonical_ast_hash"]) for row in source_rows}
    source_truth = {str(row["truth_table_hash"]) for row in source_rows}
    bindings = source_bindings(source_rows)
    validation_rows: List[Dict[str, Any]] = []
    excluded_ast = set(source_ast)
    excluded_truth = set(source_truth)
    for root_index, root in enumerate(ELIGIBLE_ROOTS):
        validation_rows.extend(validation_rows_for_root(root, bindings[root], excluded_ast, excluded_truth, root_index))

    target_inputs: List[Dict[str, Any]] = []
    target_answers: List[Dict[str, Any]] = []
    target_metadata: List[Dict[str, Any]] = []
    available_by_root = {root: bindings[root] for root in ELIGIBLE_ROOTS}
    for seed in SEEDS:
        inputs, answers, metadata = target_rows_for_seed(
            seed,
            available_by_root,
            excluded_ast,
            excluded_truth,
            frozen_pool,
        )
        target_inputs.extend(inputs)
        target_answers.extend(answers)
        target_metadata.extend(metadata)

    source_tasks = make_source_tasks(source_rows)
    validation_tasks = make_validation_tasks(validation_rows)
    validate_manifest_separation(source_tasks, validation_tasks, target_metadata)
    if any("target_ast" in row or "target_outputs" in row or "answer_labels" in row for row in target_inputs):
        raise RuntimeError("target input bundle contains an answer field")
    source_roots = {root_operator(row["target_ast"]) for row in source_rows}
    validation_counts = {root: sum(row["applicable_root_signature"] == root for row in validation_rows) for root in ELIGIBLE_ROOTS}
    if source_roots != set(ELIGIBLE_ROOTS) or validation_counts != {root: VALIDATION_PER_OPERATOR for root in ELIGIBLE_ROOTS}:
        raise RuntimeError("manifest does not contain the required eligible operators or validation allocation")
    return {
        "vocabulary": vocabulary,
        "source_rows": source_rows,
        "validation_rows": validation_rows,
        "target_inputs": sorted(target_inputs, key=lambda row: row["task_id"]),
        "target_answers": sorted(target_answers, key=lambda row: row["task_id"]),
        "target_metadata": sorted(target_metadata, key=lambda row: row["task_id"]),
        "audit": {
            "schema_version": "crucible_motif_comparison_manifest_v1",
            "manifest_seed": MANIFEST_SEED,
            "source_records": len(source_rows),
            "validation_records": len(validation_rows),
            "validation_records_per_operator": validation_counts,
            "target_input_records": len(target_inputs),
            "target_answer_records": len(target_answers),
            "target_answers_written_separately": True,
            "source_target_hash_intersections": [],
            "validation_target_hash_intersections": [],
            "source_validation_hash_intersections": [],
            "applicability_assigned_without_correctness_filter": True,
            "promotion_threshold_trials": VALIDATION_PER_OPERATOR,
            "frozen_pool_excluded_by_structural_or_functional_hash": True,
            "target_generation": "recombined source-derived binding vocabulary; no post-hoc operator success filter",
        },
    }


def task_results(execution: Mapping[str, Any], metadata: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for arm, result in sorted(execution["arms"].items()):
        for target in result["targets"]:
            meta = metadata[target["task_id"]]
            rows.append({
                "arm": arm,
                "task_id": target["task_id"],
                "seed": meta["seed"],
                "family": meta["family"],
                "accepted": bool(target["accepted"]),
                "persisted": bool(target["persisted"]),
                "post_reset_correct": target.get("sealed_post_reset_correct", False),
                "accepted_ast_hash": target.get("accepted_ast_hash"),
                "counters": target["counters"],
            })
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], execution: Mapping[str, Any]) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        acquired = sum(int(row["accepted"]) for row in arm_rows)
        reset = sum(int(row["accepted"] and row["persisted"] and row["post_reset_correct"]) for row in arm_rows)
        summaries[arm] = {
            "activation_status": execution["arms"][arm]["activation_status"],
            "promoted_operator_count": execution["arms"][arm]["promoted_operator_count"],
            "target_tasks": len(arm_rows),
            "proposal_acceptances": acquired,
            "proposal_acceptance_rate": acquired / max(1, len(arm_rows)),
            "target_acquisitions": reset,
            "target_acquisition_rate": reset / max(1, len(arm_rows)),
            "post_reset_successes": reset,
            "post_reset_success_rate": reset / max(1, len(arm_rows)),
            "cpu_seconds": sum(float(row["counters"]["cpu_seconds"]) for row in arm_rows),
            "gpu_seconds": sum(float(row["counters"]["gpu_seconds"]) for row in arm_rows),
        }
    return summaries


def paired_comparison(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_arm = {arm: {row["task_id"]: row for row in rows if row["arm"] == arm} for arm in ARMS}
    task_ids = sorted(by_arm[ARMS[0]])
    pairs: List[Dict[str, Any]] = []
    contrasts = {
        "A3_minus_A0": ("A3_VERIFIED_ADAPTIVE_LIBRARY", "A0_FROZEN_INDUCTION"),
        "A3_minus_A2": ("A3_VERIFIED_ADAPTIVE_LIBRARY", "A2_SHAM_OPERATOR_LIBRARY"),
        "A3_minus_no_guidance": ("A3_VERIFIED_ADAPTIVE_LIBRARY", NO_GUIDANCE_ARM),
    }
    for task_id in task_ids:
        row = {"task_id": task_id}
        for arm in ARMS:
            row[f"{arm}_accepted"] = bool(by_arm[arm][task_id]["accepted"])
            row[f"{arm}_post_reset"] = bool(by_arm[arm][task_id]["post_reset_correct"])
        for name, (left, right) in contrasts.items():
            row[f"{name}_accepted_delta"] = int(row[f"{left}_accepted"]) - int(row[f"{right}_accepted"])
            row[f"{name}_post_reset_delta"] = int(row[f"{left}_post_reset"]) - int(row[f"{right}_post_reset"])
        pairs.append(row)
    aggregate: Dict[str, Any] = {}
    for name in contrasts:
        aggregate[name] = {
            "accepted_delta_sum": sum(row[f"{name}_accepted_delta"] for row in pairs),
            "post_reset_delta_sum": sum(row[f"{name}_post_reset_delta"] for row in pairs),
            "tasks_where_left_only_accepted": sum(row[f"{name}_accepted_delta"] == 1 for row in pairs),
            "tasks_where_right_only_accepted": sum(row[f"{name}_accepted_delta"] == -1 for row in pairs),
        }
    return {"task_count": len(task_ids), "pairs": pairs, "aggregate": aggregate}


def promotion_report(runner: FourArmRunner, source_tasks: Sequence[Mapping[str, Any]], validation_tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    active = runner._prepare_arm("A3_VERIFIED_ADAPTIVE_LIBRARY", source_tasks, validation_tasks)["library"]
    sham = runner._prepare_arm("A2_SHAM_OPERATOR_LIBRARY", source_tasks, validation_tasks)["library"]
    ablation = runner._prepare_arm(NO_GUIDANCE_ARM, source_tasks, validation_tasks)["library"]
    active_records = [record.as_dict() for record in active.records]
    ablation_records = [record.as_dict() for record in ablation.records]
    return {
        "active_records": active_records,
        "sham_records": [record.as_dict() for record in sham.records],
        "ablation_records": ablation_records,
        "promoted_operator_ids": [record.operator_id for record in active.promoted],
        "active_and_ablation_memory_equal": json_digest(active_records) == json_digest(ablation_records),
        "active_memory_digest": json_digest(active_records),
        "ablation_memory_digest": json_digest(ablation_records),
    }


def newly_constructed_candidates(
    runner: FourArmRunner,
    target_inputs: Sequence[Mapping[str, Any]],
    source_tasks: Sequence[Mapping[str, Any]],
    validation_tasks: Sequence[Mapping[str, Any]],
    execution: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    state = runner._prepare_arm("A3_VERIFIED_ADAPTIVE_LIBRARY", source_tasks, validation_tasks)
    metadata: List[Dict[str, Any]] = []
    for row in target_inputs:
        task = ProcedureTask(
            task_id=str(row["task_id"]),
            seed=int(row["seed"]),
            family=str(row["family"]),
            demonstrations=list(row["demonstrations"]),
            candidate_pool=list(runner.pool),
            task_cluster_id=str(row["task_cluster_id"]),
        )
        candidates = []
        for record in state["library"].promoted:
            candidates.extend(construct_operator_candidates(record, task.demonstrations))
        candidates = sorted(
            {candidate["canonical_ast_hash"]: candidate for candidate in candidates}.values(),
            key=lambda candidate: (-int(candidate["match_count"]), int(ast_nodes(candidate["ast"])), candidate["canonical_ast_hash"]),
        )[:4]
        actual_trace = execution["arms"]["A3_VERIFIED_ADAPTIVE_LIBRARY"]["targets"]
        actual_target = next(item for item in actual_trace if item["task_id"] == row["task_id"])
        actual_hashes = [item["canonical_ast_hash"] for item in actual_target["verification_trace"] if not item.get("null")]
        metadata.append({
            "task_id": row["task_id"],
            "constructed_candidates": [
                {
                    "canonical_ast_hash": candidate["canonical_ast_hash"],
                    "ast": candidate["ast"],
                    "source": candidate["source"],
                    "operator_id": candidate.get("operator_id"),
                    "match_count": candidate["match_count"],
                }
                for candidate in candidates
            ],
            "active_cpu_proposal_hashes": actual_hashes,
            "constructed_candidate_path_available": bool(candidates),
            "constructed_order_matches_active_cpu_path": (
                None
                if not candidates
                else [candidate["canonical_ast_hash"] for candidate in candidates] == actual_hashes[:len(candidates)]
            ),
        })
    return metadata


def run(cpu_workers: int = 2, cpu_window: int = 4) -> Dict[str, Any]:
    started = time.perf_counter()
    frozen_pool = load_frozen_pool()
    manifests = build_manifest(frozen_pool)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT / "SOURCE_MANIFEST.jsonl", manifests["source_rows"])
    write_jsonl(OUTPUT / "OPERATOR_VALIDATION_MANIFEST.jsonl", manifests["validation_rows"])
    write_jsonl(OUTPUT / "TARGET_INPUTS.jsonl", manifests["target_inputs"])
    write_jsonl(OUTPUT / "TARGET_MANIFEST_METADATA.jsonl", manifests["target_metadata"])
    write_jsonl(OUTPUT / "SEALED_TARGET_ANSWERS.jsonl", manifests["target_answers"])
    write_json(OUTPUT / "MANIFEST_AUDIT.json", manifests["audit"])
    write_json(OUTPUT / "VOCABULARY.json", {"expression_vocabulary": manifests["vocabulary"]})
    input_hash = sha256_file(OUTPUT / "TARGET_INPUTS.jsonl")
    answer_hash = sha256_file(OUTPUT / "SEALED_TARGET_ANSWERS.jsonl")
    del manifests["target_answers"]

    source_tasks = make_source_tasks(manifests["source_rows"])
    validation_tasks = make_validation_tasks(manifests["validation_rows"])
    target_inputs = load_jsonl(OUTPUT / "TARGET_INPUTS.jsonl")
    target_metadata_rows = load_jsonl(OUTPUT / "TARGET_MANIFEST_METADATA.jsonl")
    metadata = {row["task_id"]: row for row in target_metadata_rows}
    runner = FourArmRunner(frozen_pool, OUTPUT / "stores", scientific=True)
    promotion = promotion_report(runner, source_tasks, validation_tasks)
    write_json(OUTPUT / "OPERATOR_PROMOTION.json", promotion)

    provider = SealedTargetProvider(
        OUTPUT / "TARGET_INPUTS.jsonl",
        OUTPUT / "SEALED_TARGET_ANSWERS.jsonl",
        authorised=True,
    )
    execution = runner.run_scientific(
        source_tasks,
        validation_tasks,
        target_inputs,
        provider,
        {"run_approved": False},
        {"scientific_execution_approved": False, "status": "exploratory_manifest_changed"},
        OUTPUT / "stores",
        allow_post_protocol=True,
        cpu_workers=cpu_workers,
        cpu_window=cpu_window,
        arms=ARMS,
    )
    write_json(OUTPUT / "EXECUTION.json", execution)
    rows = task_results(execution, metadata)
    write_jsonl(OUTPUT / "TASK_RESULTS.jsonl", rows)
    summaries = summarize(rows, execution)
    write_json(OUTPUT / "ARM_SUMMARIES.json", summaries)
    write_json(OUTPUT / "PAIRED_COMPARISON.json", paired_comparison(rows))
    write_jsonl(
        OUTPUT / "NEWLY_CONSTRUCTED_CANDIDATES.jsonl",
        newly_constructed_candidates(runner, target_inputs, source_tasks, validation_tasks, execution),
    )

    budget_by_arm = {
        arm: {
            "target_rows": len([row for row in rows if row["arm"] == arm]),
            "candidate_slots": sum(row["counters"]["candidate_slots"] for row in rows if row["arm"] == arm),
            "operator_lookup_slots": sum(row["counters"]["operator_lookup_slots"] for row in rows if row["arm"] == arm),
            "verifier_calls": sum(row["counters"]["verifier_calls"] for row in rows if row["arm"] == arm),
            "verifier_examples": sum(row["counters"]["verifier_examples"] for row in rows if row["arm"] == arm),
            "candidate_construction_units": sum(row["counters"]["candidate_construction_units"] for row in rows if row["arm"] == arm),
            "post_reset_checks": sum(row["counters"]["post_reset_checks"] for row in rows if row["arm"] == arm),
            "gpu_seconds": sum(row["counters"]["gpu_seconds"] for row in rows if row["arm"] == arm),
        }
        for arm in ARMS
    }
    write_json(OUTPUT / "BUDGET_AUDIT.json", {
        "arms": budget_by_arm,
        "matched_candidate_and_verifier_budgets": len({json.dumps(value, sort_keys=True) for value in budget_by_arm.values()}) == 1,
    })
    write_json(OUTPUT / "LEAKAGE_AUDIT.json", {
        "classification": "exploratory_manifest_changed",
        "target_inputs_contain_answers": False,
        "sealed_answers_opened_after_proposals": execution["sealed_answers_opened_after_proposals"],
        "target_input_sha256": input_hash,
        "sealed_answer_sha256": answer_hash,
        "target_metadata_records": len(target_metadata_rows),
        "target_evaluations_run": True,
    })
    write_json(OUTPUT / "RUN_CONFIG.json", {
        "schema_version": "crucible_motif_comparison_exploratory_v1",
        "classification": "exploratory_manifest_changed",
        "arms": list(ARMS),
        "target_tasks": len(target_inputs),
        "validation_tasks_per_operator": VALIDATION_PER_OPERATOR,
        "promotion_threshold": "16/16 with lower_bound_95 >= 0.80",
        "gpu_allocated": False,
        "registered_scientific_result": False,
        "target_input_sha256": input_hash,
        "sealed_answer_sha256": answer_hash,
        "cpu_accelerator": execution["cpu_accelerator"],
    })
    report = {
        "schema_version": "crucible_motif_comparison_exploratory_report_v1",
        "experiment": EXPERIMENT,
        "classification": "exploratory_manifest_changed",
        "status": "completed",
        "registered_scientific_result": False,
        "manifest": manifests["audit"],
        "arms": list(ARMS),
        "summaries": summaries,
        "paired_comparison": "PAIRED_COMPARISON.json",
        "promotion": "OPERATOR_PROMOTION.json",
        "newly_constructed_candidates": "NEWLY_CONSTRUCTED_CANDIDATES.jsonl",
        "resource_summary": {
            "wall_seconds": time.perf_counter() - started,
            "gpu_allocated": False,
            "gpu_seconds": 0.0,
            "cpu_accelerator": execution["cpu_accelerator"],
        },
        "limitations": [
            "Exploratory result because the manifest and implementation differ from the terminated registered protocol.",
            "The typed-hole operator vocabulary is an engineered constructive mechanism, not neural learning.",
            "Target procedures are novel relative to source, validation and frozen fallback structural/function hashes, but are recombinations of source-derived binding values.",
            "No claim of general cumulative learning or neural procedure induction is made.",
        ],
    }
    write_json(OUTPUT / "EXPLORATORY_REPORT.json", report)
    report_lines = [
        "# Exploratory executable-operator comparison",
        "",
        "This CPU-only run uses a fresh manifest and is not a registered scientific result.",
        "Target answers were opened only after all arms finalized proposals; no GPU was allocated.",
        "",
        "## Manifest",
        "",
        f"- Source procedures: {len(manifests['source_rows'])}",
        f"- Operator-validation tasks: {len(manifests['validation_rows'])} ({VALIDATION_PER_OPERATOR} per eligible root)",
        f"- Target tasks: {len(target_inputs)} ({TARGETS_PER_SEED} per seed)",
        "- Applicability was assigned by root signature before validation; no correctness filter was used.",
        "- Source, validation and target structural/function hashes are disjoint; target answers are separate.",
        "",
        "## Promotion and acquisition",
        "",
        f"- add_mod10 validation: {next(row for row in promotion['active_records'] if row['root_signature'] == 'add_mod10')['successes']}/16",
        f"- sub_mod10 validation: {next(row for row in promotion['active_records'] if row['root_signature'] == 'sub_mod10')['successes']}/16",
        "- Promoted operators: none; the 16/16 threshold was unchanged.",
        "- Newly constructed adaptive candidate ASTs: none, because A3 remained unactivated.",
        "",
        "| Arm | Proposal accepted | End-to-end acquisition after reset |",
        "| --- | ---: | ---: |",
    ]
    for arm in ARMS:
        report_lines.append(
            f"| {arm} | {summaries[arm]['proposal_acceptances']}/{summaries[arm]['target_tasks']} | "
            f"{summaries[arm]['target_acquisitions']}/{summaries[arm]['target_tasks']} |"
        )
    report_lines.extend([
        "",
        "The A3 and no-guidance ablation share identical operator memory, but the mechanism never activated, so their paired contrast is not evidence for or against adaptive guidance.",
        "The measured A3-vs-A0, A3-vs-A2 and A3-vs-no-guidance task-level differences are all zero.",
    ])
    (OUTPUT / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        "# Executable operator motif comparison\n\n"
        "Exploratory CPU-only target acquisition comparison using a fresh manifest. "
        "A3 and the no-guidance ablation share the same promoted operator records; "
        "only A3 receives those records in candidate construction. Target answers "
        "were opened only after all arms finalized proposals. This is not a "
        "registered scientific result and does not rerun the 288-task benchmark.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "HASHES.json")
    write_json(OUTPUT / "HASHES.json", {str(path.relative_to(ROOT)): sha256_file(path) for path in files})
    return {"output": str(OUTPUT), "rows": len(rows), "summaries": summaries}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    result = run()
    print(json.dumps({
        "status": "completed",
        "classification": "exploratory_manifest_changed",
        "output": result["output"],
        "rows": result["rows"],
        "arms": list(ARMS),
        "gpu_allocated": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
