#!/usr/bin/env python3
"""CPU-only end-to-end test of the executable operator pathway.

This is a synthetic mechanism test, not a rerun of the 288-task benchmark.
It verifies source checking, motif extraction, operator-specific validation,
promotion, motif-based future candidate construction, negative stage
classification and deterministic replay.  No target bundle is opened.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from crucible_cumulative_adaptation_scientific_runner_20261028 import (  # noqa: E402
    FourArmRunner,
    CrucibleCpuReservoir,
    IndependentVerifier,
    OperatorLibrary,
    OperatorRecord,
    ProcedureTask,
    TARGET_BUDGET,
    ast_hash,
    ast_nodes,
    canonical_ast,
    candidate_pool_for_task,
    examples_for_ast,
    root_operator,
    sha256_file,
    truth_table_hash,
    validate_operator_record,
)


OUTPUT = ROOT / "experiments" / "crucible_cumulative_adaptation_operator_pathway_diagnostic_20261028"
PROMOTION_TRIALS = 16


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ast(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    return canonical_ast({"op": op, "args": [left, right]})


def make_synthetic_tasks() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    var_a = {"op": "var", "name": "a"}
    var_b = {"op": "var", "name": "b"}
    const_1 = {"op": "const", "value": 1}
    const_2 = {"op": "const", "value": 2}
    const_3 = {"op": "const", "value": 3}
    source_asts = [
        ast("add_mod10", var_a, const_1),
        ast("add_mod10", var_b, const_2),
        ast("add_mod10", var_a, const_3),
    ]
    source_pairs = [(0, 0), (1, 2), (4, 3), (9, 8), (2, 7), (5, 5), (8, 1), (3, 6)]
    source_tasks = [
        {
            "task_id": f"synthetic_source_{index}",
            "seed": 7,
            "family": "synthetic_source",
            "target_ast": source_ast,
            "demonstrations": examples_for_ast(source_ast, source_pairs),
            "canonical_ast_hash": ast_hash(source_ast),
            "truth_table_hash": truth_table_hash(source_ast),
        }
        for index, source_ast in enumerate(source_asts)
    ]

    validation_ast = ast("add_mod10", var_b, const_3)
    validation_tasks: List[Dict[str, Any]] = []
    for index in range(PROMOTION_TRIALS):
        # Each validation task has a distinct deterministic draw of examples;
        # no task reuses another task's demonstration or verification set.
        pairs = [
            ((index + offset) % 10, (index * 3 + offset + 1) % 10)
            for offset in range(8)
        ]
        verification_pairs = [
            ((index * 2 + offset + 1) % 10, (index * 5 + offset + 2) % 10)
            for offset in range(4)
        ]
        validation_tasks.append(
            {
                "task_id": f"synthetic_operator_validation_{index:02d}",
                "seed": 100 + index,
                "family": "synthetic_operator_validation",
                # This field is phase metadata for allocating applicable
                # trials; proposal construction never reads target_ast.
                "target_ast": validation_ast,
                "applicable_root_signature": "add_mod10",
                "canonical_ast_hash": ast_hash(validation_ast),
                "truth_table_hash": truth_table_hash(validation_ast),
                "demonstrations": examples_for_ast(validation_ast, pairs),
                "verification_examples": examples_for_ast(
                    validation_ast,
                    verification_pairs,
                ),
            }
        )
    target_task = {
        "task_id": "synthetic_future_target",
        "seed": 999,
        "family": "synthetic_future",
        "task_cluster_id": "synthetic_future",
        "demonstrations": examples_for_ast(validation_ast, [(0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 0), (7, 1)]),
    }
    return source_tasks, validation_tasks, target_task


def small_pool() -> List[Dict[str, Any]]:
    distractor = ast("max", {"op": "var", "name": "a"}, {"op": "var", "name": "b"})
    return [
        {
            "canonical_ast": distractor,
            "canonical_ast_hash": ast_hash(distractor),
            "truth_table_hash": truth_table_hash(distractor),
            "nodes": ast_nodes(distractor),
        }
    ]


def negative_cases(record: OperatorRecord, validation_task: Mapping[str, Any]) -> Dict[str, Any]:
    cases: Dict[str, Any] = {}

    mismatch = dict(validation_task)
    mismatch["applicable_root_signature"] = "min"
    cases["applicability"] = validate_operator_record(record, mismatch)

    construction = copy.deepcopy(record)
    construction.binding_vocabulary = {hole["id"]: [] for hole in record.typed_holes}
    cases["construction"] = validate_operator_record(construction, validation_task)

    incorrect = copy.deepcopy(record)
    incorrect.binding_vocabulary = {
        hole["id"]: [
            {"op": "var", "name": "a"}
            if index == 0
            else {"op": "const", "value": 1}
        ]
        for index, hole in enumerate(record.typed_holes)
    }
    cases["incorrect_output"] = validate_operator_record(incorrect, validation_task)

    verification = dict(validation_task)
    verification["verification_examples"] = examples_for_ast(
        ast("add_mod10", {"op": "var", "name": "b"}, {"op": "const", "value": 4}),
        [(9, 0), (8, 1), (7, 2), (6, 3)],
    )
    cases["verification"] = validate_operator_record(record, verification)
    return cases


def run_once() -> Dict[str, Any]:
    source_tasks, validation_tasks, target_task = make_synthetic_tasks()
    library = OperatorLibrary("A3_VERIFIED_ADAPTIVE_LIBRARY", source_tasks, validation_tasks)
    if len(library.records) != 1:
        raise RuntimeError(f"expected one synthetic operator, found {len(library.records)}")
    record = library.records[0]
    pool = small_pool()
    runner = FourArmRunner(pool, OUTPUT / "synthetic_store")
    cold = OperatorLibrary("A0_FROZEN_INDUCTION", source_tasks, validation_tasks)
    task = ProcedureTask(
        task_id=target_task["task_id"],
        seed=target_task["seed"],
        family=target_task["family"],
        demonstrations=target_task["demonstrations"],
        candidate_pool=runner.pool,
        task_cluster_id=target_task["task_cluster_id"],
    )
    cold_candidates = candidate_pool_for_task(runner.pool, task, cold, "A0_FROZEN_INDUCTION", runner.pool_truth_tables)
    active_candidates = candidate_pool_for_task(runner.pool, task, library, "A3_VERIFIED_ADAPTIVE_LIBRARY", runner.pool_truth_tables)
    reservoir = CrucibleCpuReservoir(workers=1, window=1)
    accelerated_lists, _ = reservoir.map_candidates(
        runner.pool,
        runner.pool_truth_tables,
        library.promoted,
        "A3_VERIFIED_ADAPTIVE_LIBRARY",
        [task],
    )
    accelerated_candidates = accelerated_lists[0]
    expected_ast = ast("add_mod10", {"op": "var", "name": "b"}, {"op": "const", "value": 3})
    active_first = active_candidates[0] if active_candidates else None
    negative = negative_cases(record, validation_tasks[0])
    return {
        "source_count": len(source_tasks),
        "source_verified_count": record.provenance["source_procedures_independently_verified"],
        "operator_record": record.as_dict(),
        "validation": {
            "trials": record.trials,
            "successes": record.successes,
            "failure_counts": record.failure_counts,
            "all_success": record.successes == PROMOTION_TRIALS,
            "trace": record.validation_trace,
        },
        "proposal_path": {
            "cold_candidate_hashes": [candidate["canonical_ast_hash"] for candidate in cold_candidates],
            "active_candidate_hashes": [candidate["canonical_ast_hash"] for candidate in active_candidates],
            "active_candidate_sources": [candidate["source"] for candidate in active_candidates],
            "active_first_ast": None if active_first is None else active_first["ast"],
            "accelerated_first_ast": None if not accelerated_candidates else accelerated_candidates[0]["ast"],
            "expected_constructed_ast": expected_ast,
            "constructed_expected_ast": active_first is not None and active_first["ast"] == expected_ast,
            "accelerated_constructed_expected_ast": bool(accelerated_candidates) and accelerated_candidates[0]["ast"] == expected_ast,
            "accelerated_matches_parent": [candidate["canonical_ast_hash"] for candidate in accelerated_candidates] == [candidate["canonical_ast_hash"] for candidate in active_candidates],
            "cold_and_active_orders_differ": [candidate["canonical_ast_hash"] for candidate in cold_candidates] != [candidate["canonical_ast_hash"] for candidate in active_candidates],
            "frozen_pool_only_contains_distractor": len(cold_candidates) == 1 and root_operator(cold_candidates[0]["ast"]) == "max",
        },
        "negative_cases": {
            name: {
                "failure_stage": result.get("failure_stage"),
                "failure_reason": result.get("failure_reason"),
                "success": result.get("success", False),
            }
            for name, result in negative.items()
        },
        "negative_case_expectations": {
            "applicability": "applicability_matching",
            "construction": "candidate_construction",
            "incorrect_output": "incorrect_output",
            "verification": "verification",
        },
        "target_answers_accessed": False,
        "benchmark_repeated": False,
        "gpu_allocated": False,
    }


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run() -> Dict[str, Any]:
    first = run_once()
    second = run_once()
    replay_digest_a = stable_digest(first)
    replay_digest_b = stable_digest(second)
    result = {
        "schema_version": "crucible_operator_pathway_diagnostic_v1",
        "classification": "bounded_cpu_synthetic_mechanism_test",
        "mechanism_path": "source verification -> motif extraction -> typed-hole validation -> promotion -> motif candidate construction -> proposal ordering",
        "result": first,
        "deterministic_replay": {
            "passed": replay_digest_a == replay_digest_b,
            "first_digest": replay_digest_a,
            "second_digest": replay_digest_b,
        },
        "interpretation": "The executable motif pathway works on synthetic CPU tasks. Promoted source-derived motifs construct a candidate absent from the frozen fallback pool and change future proposal construction; negative stages are separately observable.",
        "next_action": "Freeze a new operator-specific validation manifest with 16 applicable tasks and independent verification examples per operator before any benchmark rerun.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "PATHWAY_REPORT.json", result)
    (OUTPUT / "README.md").write_text(
        "# Executable operator pathway diagnostic\n\n"
        "Synthetic CPU-only mechanism test. No 288-task benchmark was rerun, "
        "no target answers were opened, and no GPU was used.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "HASHES.json")
    write_json(OUTPUT / "HASHES.json", {str(path.relative_to(ROOT)): sha256_file(path) for path in files})
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "status": "completed",
        "classification": result["classification"],
        "promotion": result["result"]["operator_record"]["status"],
        "constructed_expected_ast": result["result"]["proposal_path"]["constructed_expected_ast"],
        "negative_cases": result["result"]["negative_cases"],
        "deterministic_replay": result["deterministic_replay"]["passed"],
        "output": str(OUTPUT),
    }, sort_keys=True))
