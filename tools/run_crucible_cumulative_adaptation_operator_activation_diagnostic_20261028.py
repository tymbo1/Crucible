#!/usr/bin/env python3
"""Diagnose operator activation without rerunning the cumulative benchmark.

This diagnostic reads the committed source/validation manifests and the saved
post-protocol target-input bundle.  It does not open target answers, execute a
target lifecycle, or run the 288-task benchmark.  It reports the current
root-signature validation rule, derives diagnostic AST motifs with typed
expression holes, checks the hypothetical candidate-order effect on four
saved target inputs, and exercises the existing policy hook on a tiny
synthetic activated operator.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from crucible_cumulative_adaptation_scientific_runner_20261028 import (  # noqa: E402
    ARM_IDS,
    FourArmRunner,
    IndependentVerifier,
    OperatorLibrary,
    ProcedureTask,
    ast_hash,
    ast_nodes,
    canonical_ast,
    candidate_pool_for_task,
    examples_for_ast,
    load_frozen_pool,
    make_source_tasks,
    make_validation_tasks,
    root_operator,
    sha256_file,
    truth_table_hash,
)


PREFLIGHT = ROOT / "experiments" / "crucible_cumulative_adaptation_preflight_20261028"
SAVED_RUN = ROOT / "experiments" / "crucible_cumulative_adaptation_exploratory_20261028_v3"
OUTPUT = ROOT / "experiments" / "crucible_cumulative_adaptation_operator_activation_diagnostic_20261028"
PROMOTION_TRIALS = 16
TARGET_ORDER_SAMPLE = 4


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def extract_diagnostic_motif(ast: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract a metadata-only root motif; hole nodes are not executable ASTs."""
    canonical = canonical_ast(ast)
    if canonical["op"] in {"var", "const"}:
        return {"canonical_motif": canonical, "typed_holes": []}
    holes = [
        {"id": f"arg{index}", "type": "expr_mod10", "path": ["args", index]}
        for index, _ in enumerate(canonical.get("args", []))
    ]
    motif = {
        "op": canonical["op"],
        "args": [
            {"op": "hole", "id": hole["id"], "type": hole["type"]}
            for hole in holes
        ],
    }
    return {"canonical_motif": motif, "typed_holes": holes}


def first_source_representative(source_tasks: Sequence[Mapping[str, Any]], signature: str) -> Mapping[str, Any]:
    for task in source_tasks:
        if root_operator(task["target_ast"]) == signature:
            return task
    raise RuntimeError(f"no source representative for operator {signature}")


def validation_trace_for_operator(
    signature: str,
    validation_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reproduce the current runner's exact shared-prefix validation rule."""
    trials = list(validation_rows[:PROMOTION_TRIALS])
    rows: List[Dict[str, Any]] = []
    for row in trials:
        observed = root_operator(row["target_ast"])
        matches = observed == signature
        rows.append(
            {
                "task_id": row["task_id"],
                "operator_signature": signature,
                "observed_root_signature": observed,
                "root_signature_match": matches,
                "failure_stage": None if matches else "applicability_matching",
                "candidate_construction": "not_run",
                "incorrect_output": "not_run",
                "verification": "not_run",
                "note": (
                    "Current implementation counts root-signature equality as a validation success; "
                    "it does not instantiate or independently verify an operator motif."
                ),
            }
        )
    successes = sum(int(row["root_signature_match"]) for row in rows)
    summary = {
        "trials": len(rows),
        "root_signature_successes": successes,
        "root_signature_failures": len(rows) - successes,
        "failure_counts": {
            "applicability_matching": len(rows) - successes,
            "candidate_construction": 0,
            "incorrect_output": 0,
            "verification": 0,
            "not_reached_after_applicability": successes,
        },
        "semantic_operator_validation_executed": False,
        "validation_selection_rule": "validation_rows[:16] shared by every eligible operator",
    }
    return rows, summary


def order_hashes(
    runner: FourArmRunner,
    library: OperatorLibrary,
    arm: str,
    row: Mapping[str, Any],
) -> List[str]:
    task = ProcedureTask(
        task_id=str(row["task_id"]),
        seed=int(row["seed"]),
        family=str(row["family"]),
        demonstrations=list(row["demonstrations"]),
        candidate_pool=runner.pool,
        task_cluster_id=str(row.get("task_cluster_id", row["task_id"])),
    )
    candidates = candidate_pool_for_task(
        runner.pool,
        task,
        library,
        arm,
        runner.pool_truth_tables,
    )
    return [str(candidate["canonical_ast_hash"]) for candidate in candidates]


def hypothetical_order_effect(
    runner: FourArmRunner,
    source_tasks: Sequence[Mapping[str, Any]],
    validation_tasks: Sequence[Mapping[str, Any]],
    target_input_rows: Sequence[Mapping[str, Any]],
    signature: str,
) -> Dict[str, Any]:
    cold = OperatorLibrary("A0_FROZEN_INDUCTION", source_tasks, validation_tasks)
    active = OperatorLibrary("A3_VERIFIED_ADAPTIVE_LIBRARY", source_tasks, validation_tasks)
    selected = [record for record in active.records if record.root_signature == signature]
    if not selected:
        raise RuntimeError(f"missing operator record for {signature}")
    selected[0].status = "promoted"
    selected[0].lower_bound_95 = 1.0
    comparisons = []
    for row in target_input_rows[:TARGET_ORDER_SAMPLE]:
        cold_order = order_hashes(runner, cold, "A0_FROZEN_INDUCTION", row)
        warm_order = order_hashes(runner, active, "A3_VERIFIED_ADAPTIVE_LIBRARY", row)
        comparisons.append(
            {
                "task_id": row["task_id"],
                "cold_order": cold_order,
                "activated_order": warm_order,
                "changed": cold_order != warm_order,
            }
        )
    return {
        "operator_signature": signature,
        "activation_forced_only_for_hypothetical": True,
        "sample_size": len(comparisons),
        "changed_order_count": sum(int(row["changed"]) for row in comparisons),
        "comparisons": comparisons,
    }


def synthetic_activation_demo(pool: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Show the existing root-signature policy hook changing proposal order."""
    candidates = [
        row for row in pool
        if root_operator(row["canonical_ast"]) not in {None, "add_mod10"}
    ]
    promoted_row = next(row for row in pool if root_operator(row["canonical_ast"]) == "add_mod10")
    distractor = next(
        row for row in candidates
        if ast_nodes(row["canonical_ast"]) == ast_nodes(promoted_row["canonical_ast"])
        and promoted_row["canonical_ast_hash"] < row["canonical_ast_hash"]
    )
    small_pool = [promoted_row, distractor]
    promoted_ast = canonical_ast(promoted_row["canonical_ast"])
    source_row = {
        "task_id": "synthetic_source_add",
        "seed": 1,
        "family": "synthetic_source",
        "target_ast": promoted_ast,
        "demonstrations": examples_for_ast(promoted_ast, [(0, 0), (1, 2), (3, 4), (7, 8)]),
        "canonical_ast_hash": ast_hash(promoted_ast),
        "truth_table_hash": truth_table_hash(promoted_ast),
    }
    validation_rows = [
        {
            "task_id": f"synthetic_validation_{index:02d}",
            "seed": 1,
            "family": "synthetic_validation",
            "target_ast": promoted_ast,
            "canonical_ast_hash": ast_hash(promoted_ast),
            "truth_table_hash": truth_table_hash(promoted_ast),
        }
        for index in range(PROMOTION_TRIALS)
    ]
    source_tasks = make_source_tasks([source_row])
    validation_tasks = make_validation_tasks(validation_rows)
    library = OperatorLibrary("A3_VERIFIED_ADAPTIVE_LIBRARY", source_tasks, validation_tasks)
    verifier = IndependentVerifier()
    source_verified, source_checks = verifier.verify(promoted_ast, source_row["demonstrations"])
    validation_verified = []
    for row in validation_rows:
        ok, checks = verifier.verify(promoted_ast, examples_for_ast(promoted_ast, [(2, 3), (4, 5), (8, 9), (1, 7)]))
        validation_verified.append({"task_id": row["task_id"], "verified": ok, "checks": checks})
    runner = FourArmRunner(small_pool, OUTPUT / "synthetic_store")
    synthetic_task = {
        "task_id": "synthetic_target_order",
        "seed": 1,
        "family": "synthetic_target",
        "task_cluster_id": "synthetic_target_order",
        "demonstrations": [],
    }
    cold = OperatorLibrary("A0_FROZEN_INDUCTION", source_tasks, validation_tasks)
    cold_order = order_hashes(runner, cold, "A0_FROZEN_INDUCTION", synthetic_task)
    active_order = order_hashes(runner, library, "A3_VERIFIED_ADAPTIVE_LIBRARY", synthetic_task)
    return {
        "source_verified": source_verified,
        "source_verifier_checks": source_checks,
        "validation_verified_count": sum(int(row["verified"]) for row in validation_verified),
        "validation_trials": len(validation_verified),
        "operator_record": library.records[0].as_dict(),
        "candidate_pool": [
            {"root_signature": root_operator(row["canonical_ast"]), "canonical_ast_hash": row["canonical_ast_hash"]}
            for row in small_pool
        ],
        "cold_order": cold_order,
        "activated_order": active_order,
        "order_changed": cold_order != active_order,
        "interpretation": "Synthetic root-signature policy hook activated and changed proposal order without target labels.",
    }


def run() -> Dict[str, Any]:
    source_path = PREFLIGHT / "SOURCE_MANIFEST.jsonl"
    validation_path = PREFLIGHT / "OPERATOR_VALIDATION_MANIFEST.jsonl"
    target_input_path = SAVED_RUN / "SEALED_TARGET_INPUTS.jsonl"
    source_rows = load_jsonl(source_path)
    validation_rows = load_jsonl(validation_path)
    target_input_rows = load_jsonl(target_input_path)
    source_tasks = make_source_tasks(source_rows)
    validation_tasks = make_validation_tasks(validation_rows)
    adaptive = OperatorLibrary("A3_VERIFIED_ADAPTIVE_LIBRARY", source_tasks, validation_tasks)
    eligible = adaptive.records[:2]
    pool = load_frozen_pool()
    runner = FourArmRunner(pool, OUTPUT / "stores")

    operators: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    order_effects: List[Dict[str, Any]] = []
    for record in eligible:
        representative = first_source_representative(source_tasks, record.root_signature)
        motif = extract_diagnostic_motif(representative["target_ast"])
        source_procedures = [
            {
                "task_id": task["task_id"],
                "canonical_ast_hash": task["canonical_ast_hash"],
                "target_ast": task["target_ast"],
            }
            for task in source_tasks
            if root_operator(task["target_ast"]) == record.root_signature
        ]
        trace, validation_summary = validation_trace_for_operator(record.root_signature, validation_rows)
        trace_rows.extend(trace)
        order_effect = hypothetical_order_effect(
            runner,
            source_tasks,
            validation_tasks,
            target_input_rows,
            record.root_signature,
        )
        order_effects.append(order_effect)
        operators.append(
            {
                "operator_id": record.operator_id,
                "root_signature": record.root_signature,
                "extracted_motif": motif["canonical_motif"],
                "typed_holes": motif["typed_holes"],
                "motif_is_diagnostic_metadata_only": True,
                "source_procedures": source_procedures,
                "current_record": record.as_dict(),
                "validation": validation_summary,
                "validation_trace_task_ids": [row["task_id"] for row in trace],
                "hypothetical_order_effect": order_effect,
            }
        )

    synthetic = synthetic_activation_demo(pool)
    report = {
        "schema_version": "crucible_operator_activation_diagnostic_v1",
        "experiment": "crucible_cumulative_adaptation_operator_activation_diagnostic_20261028",
        "classification": "bounded_cpu_diagnostic",
        "benchmark_repeated": False,
        "target_answers_accessed": False,
        "target_evaluations_run": False,
        "source_manifest": {"path": str(source_path), "sha256": sha256_file(source_path), "rows": len(source_rows)},
        "validation_manifest": {"path": str(validation_path), "sha256": sha256_file(validation_path), "rows": len(validation_rows)},
        "saved_target_input_sample": {
            "path": str(target_input_path),
            "sha256": sha256_file(target_input_path),
            "rows_loaded": len(target_input_rows),
            "rows_used_for_order_only": min(TARGET_ORDER_SAMPLE, len(target_input_rows)),
        },
        "eligible_operator_selection": {
            "rule": "same as current runner: first two distinct source root signatures",
            "operators": [record.root_signature for record in eligible],
        },
        "operators": operators,
        "synthetic_activation_demo": synthetic,
        "implementation_assessment": {
            "status": "narrow_policy_path_present_activation_unreachable_in_saved_run",
            "source_to_record": "Current code extracts only root_signature; no canonical motif or typed holes are persisted in OperatorRecord.",
            "record_to_validation": "Current code counts root-signature equality on one shared validation prefix; it does not construct or independently verify a motif candidate.",
            "record_to_target_policy": "When a record is manually/promoted, candidate_pool_for_task adds a +1 ranking bonus to matching candidate roots; the synthetic demo confirms order change.",
            "activation_failure": "Both eligible operators fail the shared-prefix 16-trial gate before candidate construction or verification is reached.",
            "conclusion": "A minimal ordering hook exists, but the intended verified motif-to-candidate path is not implemented and the saved manifest does not allocate operator-specific validation trials.",
        },
        "next_engineering_action": {
            "priority": 1,
            "action": "Create operator-specific validation allocations and replace root-signature-only records with canonical motifs plus typed-hole instantiation and independent verifier traces.",
            "required_tests_before_any_benchmark": [
                "16/16 activation for a synthetic operator with all later stages traced",
                "candidate construction rejection and incorrect-output rejection are separately observable",
                "activated operator changes ordering only through its instantiated motif/policy",
                "A0/A3 order replay is deterministic",
            ],
            "do_not": "Do not rerun the 288-task benchmark until these CPU fixtures pass and the validation manifest allocation is frozen.",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT / "VALIDATION_TRACE.jsonl", trace_rows)
    write_json(OUTPUT / "DIAGNOSTIC_REPORT.json", report)
    write_json(OUTPUT / "SYNTHETIC_ACTIVATION.json", synthetic)
    write_json(OUTPUT / "HASHES.json", {})
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "HASHES.json")
    write_json(OUTPUT / "HASHES.json", {str(path.relative_to(ROOT)): sha256_file(path) for path in files})
    (OUTPUT / "README.md").write_text(
        "# Operator activation diagnostic\n\n"
        "Bounded CPU diagnostic only. It did not rerun the 288-task benchmark, "
        "open target answers or execute target lifecycles. The report separates "
        "the existing narrow root-signature ordering hook from the missing "
        "motif/typed-hole construction path.\n",
        encoding="utf-8",
    )
    # Refresh hashes after writing the README.
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "HASHES.json")
    write_json(OUTPUT / "HASHES.json", {str(path.relative_to(ROOT)): sha256_file(path) for path in files})
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "status": "completed",
        "classification": result["classification"],
        "benchmark_repeated": result["benchmark_repeated"],
        "synthetic_order_changed": result["synthetic_activation_demo"]["order_changed"],
        "output": str(OUTPUT),
    }, sort_keys=True))
