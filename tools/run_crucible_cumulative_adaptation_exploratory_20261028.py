#!/usr/bin/env python3
"""Run the terminated cumulative-adaptation design as an exploratory study.

This is deliberately not the registered scientific run.  The user terminated
the protocol before the target bundles existed, so this command reconstructs
the deterministic task set, writes explicit exploratory sealed bundles, and
uses the same four-arm runner after proposal finalisation.  Results are
post-protocol and cannot support the preregistered inference claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import run_crucible_cumulative_adaptation_preflight_20261028 as preflight  # noqa: E402
from crucible_cumulative_adaptation_scientific_runner_20261028 import (  # noqa: E402
    ARM_IDS,
    OUTPUT as RUNNER_VALIDATION_OUTPUT,
    FourArmRunner,
    SealedTargetProvider,
    load_frozen_pool,
    make_source_tasks,
    make_validation_tasks,
    sha256_file,
    truth_table_hash,
    ast_hash,
)


EXPERIMENT = "crucible_cumulative_adaptation_exploratory_20261028_v3"
OUTPUT = ROOT / "experiments" / EXPERIMENT
PREFLIGHT = ROOT / "experiments" / "crucible_cumulative_adaptation_preflight_20261028"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_query_seed(task_id: str) -> int:
    return int.from_bytes(hashlib.sha256(task_id.encode("utf-8")).digest()[:8], "big") & 0x7FFFFFFF


def reconstruct_frozen_manifest() -> Dict[str, Any]:
    rng = random.Random(preflight.MANIFEST_SEED)
    grammar_report, _ = preflight.grammar_preflight(rng)
    manifests = preflight.build_manifests(rng)
    metadata = load_jsonl(PREFLIGHT / "TARGET_MANIFEST_METADATA.jsonl")
    expected = {row["task_id"]: row for row in metadata}
    actual = {row["task_id"]: row for row in manifests["target_rows"]}
    if set(expected) != set(actual):
        raise RuntimeError("reconstructed target task IDs do not match frozen metadata")
    for task_id, row in expected.items():
        rebuilt = actual[task_id]
        for key in ("canonical_ast_hash", "truth_table_hash", "seed", "family"):
            if rebuilt[key] != row[key]:
                raise RuntimeError(f"reconstructed target mismatch for {task_id}: {key}")
    if grammar_report["representative_structural_classes"] != 2730:
        raise RuntimeError("frozen grammar draw stream no longer matches the committed preflight")
    return manifests


def build_exploratory_bundles(manifests: Mapping[str, Any]) -> Dict[str, Any]:
    target_metadata = {row["task_id"]: row for row in manifests["target_rows"]}
    input_rows: List[Dict[str, Any]] = []
    answer_rows: List[Dict[str, Any]] = []
    for seed in preflight.SEEDS:
        for hidden in manifests["target_hidden_by_seed"][seed]:
            task_id = hidden["task_id"]
            metadata = target_metadata[task_id]
            ast = hidden["ast"]
            demonstrations = preflight.make_demonstrations(ast, int(seed))
            query_pairs = preflight.make_target_inputs(task_query_seed(task_id))
            query_examples = [
                {"a": int(pair[0]), "b": int(pair[1]), "output": int(preflight.evaluate_ast(ast, int(pair[0]), int(pair[1])))}
                for pair in query_pairs
            ]
            input_rows.append({
                "task_id": task_id,
                "seed": int(seed),
                "family": metadata["family"],
                "task_cluster_id": metadata["task_cluster_id"],
                "demonstrations": demonstrations,
                "input_bundle_classification": "exploratory_post_protocol",
            })
            answer_rows.append({
                "task_id": task_id,
                "seed": int(seed),
                "family": metadata["family"],
                "canonical_ast_hash": ast_hash(ast),
                "truth_table_hash": truth_table_hash(ast),
                "target_ast": ast,
                "query_examples": query_examples,
                "answer_bundle_classification": "exploratory_post_protocol",
            })
    input_rows.sort(key=lambda row: row["task_id"])
    answer_rows.sort(key=lambda row: row["task_id"])
    return {"inputs": input_rows, "answers": answer_rows}


def flatten_task_results(execution: Mapping[str, Any], metadata: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for arm, result in sorted(execution["arms"].items()):
        for target in result["targets"]:
            meta = metadata[target["task_id"]]
            rows.append({
                "arm": arm,
                "task_id": target["task_id"],
                "seed": meta["seed"],
                "family": meta["family"],
                "task_cluster_id": meta["task_cluster_id"],
                "accepted": target["accepted"],
                "persisted": target["persisted"],
                "sealed_post_reset_correct": target.get("sealed_post_reset_correct", False),
                "accepted_ast_hash": target.get("accepted_ast_hash"),
                "counters": target["counters"],
            })
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], execution: Mapping[str, Any], source_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    source_truth = {row["truth_table_hash"] for row in source_rows}
    source_ast = {row["canonical_ast_hash"] for row in source_rows}
    for arm in ARM_IDS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        lifecycle = [row for row in arm_rows if row["accepted"] and row["persisted"] and row["sealed_post_reset_correct"]]
        source_candidate_hits = [row for row in arm_rows if row.get("accepted_ast_hash") in source_ast]
        by_seed_family: Dict[str, Dict[str, int]] = {}
        for row in arm_rows:
            key = f"seed_{row['seed']}::{row['family']}"
            bucket = by_seed_family.setdefault(key, {"tasks": 0, "lifecycle_successes": 0})
            bucket["tasks"] += 1
            bucket["lifecycle_successes"] += int(row["accepted"] and row["persisted"] and row["sealed_post_reset_correct"])
        summaries[arm] = {
            "activation_status": execution["arms"][arm]["activation_status"],
            "promoted_operator_count": execution["arms"][arm]["promoted_operator_count"],
            "target_tasks": len(arm_rows),
            "accepted": sum(int(row["accepted"]) for row in arm_rows),
            "lifecycle_successes": len(lifecycle),
            "lifecycle_success_rate": len(lifecycle) / max(1, len(arm_rows)),
            "accepted_candidate_hashes_overlapping_source": len(source_candidate_hits),
            "target_truth_table_retrieval_hits_not_counted": 0,
            "by_seed_family": by_seed_family,
            "all_gpu_seconds": sum(float(row["counters"]["gpu_seconds"]) for row in arm_rows),
            "cpu_seconds": sum(float(row["counters"]["cpu_seconds"]) for row in arm_rows),
        }
    a0 = summaries["A0_FROZEN_INDUCTION"]["lifecycle_success_rate"]
    a2 = summaries["A2_SHAM_OPERATOR_LIBRARY"]["lifecycle_success_rate"]
    a3 = summaries["A3_VERIFIED_ADAPTIVE_LIBRARY"]["lifecycle_success_rate"]
    return {
        "arms": summaries,
        "primary_contrast_a3_minus_max_a0_a2": a3 - max(a0, a2),
        "registered_success_criteria_applied": False,
        "interpretation": "Exploratory post-protocol descriptive result; no preregistered significance or cumulative-learning claim is made.",
    }


def run(cpu_workers: int | None = None, cpu_window: int | None = None) -> Dict[str, Any]:
    started = time.perf_counter()
    manifests = reconstruct_frozen_manifest()
    pool = load_frozen_pool()
    bundles = build_exploratory_bundles(manifests)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    input_path = OUTPUT / "SEALED_TARGET_INPUTS.jsonl"
    answer_path = OUTPUT / "SEALED_TARGET_ANSWERS.jsonl"
    write_jsonl(input_path, bundles["inputs"])
    write_jsonl(answer_path, bundles["answers"])
    answer_hash = sha256_file(answer_path)
    input_hash = sha256_file(input_path)
    del bundles["answers"]

    source_rows = load_jsonl(PREFLIGHT / "SOURCE_MANIFEST.jsonl")
    validation_rows = load_jsonl(PREFLIGHT / "OPERATOR_VALIDATION_MANIFEST.jsonl")
    source_tasks = make_source_tasks(source_rows)
    validation_tasks = make_validation_tasks(validation_rows)
    target_metadata_rows = load_jsonl(PREFLIGHT / "TARGET_MANIFEST_METADATA.jsonl")
    metadata = {row["task_id"]: row for row in target_metadata_rows}
    target_input_rows = load_jsonl(input_path)
    runner = FourArmRunner(pool=pool, root=OUTPUT / "stores")
    provider = SealedTargetProvider(input_path, answer_path, authorised=True)
    protocol_stub = {"run_approved": False}
    amendment_stub = {"scientific_execution_approved": False, "status": "terminated_by_user"}
    execution = runner.run_scientific(
        source_tasks,
        validation_tasks,
        target_input_rows,
        provider,
        protocol_stub,
        amendment_stub,
        OUTPUT / "stores",
        allow_post_protocol=True,
        cpu_workers=cpu_workers,
        cpu_window=cpu_window,
    )
    rows = flatten_task_results(execution, metadata)
    write_jsonl(OUTPUT / "TASK_RESULTS.jsonl", rows)
    summary = summarize(rows, execution, source_rows)
    write_json(OUTPUT / "ARM_SUMMARIES.json", summary)
    write_json(OUTPUT / "PROPOSAL_DIGEST.json", {
        "proposal_digest": execution["proposal_digest"],
        "sealed_answers_opened_after_proposals": execution["sealed_answers_opened_after_proposals"],
        "input_bundle_sha256": input_hash,
        "answer_bundle_sha256": answer_hash,
    })
    counters = {
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
        for arm in ARM_IDS
    }
    write_json(OUTPUT / "BUDGET_AUDIT.json", {"arms": counters, "all_equal": len({json.dumps(value, sort_keys=True) for value in counters.values()}) == 1})
    write_json(OUTPUT / "LEAKAGE_AUDIT.json", {
        "classification": "exploratory_post_protocol",
        "source_target_hashes_verified_against_metadata": True,
        "target_inputs_loaded_before_proposals": True,
        "sealed_answers_opened_after_proposal_digest": execution["sealed_answers_opened_after_proposals"],
        "target_answer_bundle_sha256": answer_hash,
        "accepted_candidate_hashes_overlapping_source": {
            arm: summary["accepted_candidate_hashes_overlapping_source"]
            for arm, summary in summary["arms"].items()
        },
    })
    write_json(OUTPUT / "RUN_CONFIG.json", {
        "schema_version": "crucible_cumulative_adaptation_exploratory_v1",
        "classification": "exploratory_post_protocol",
        "protocol_terminated_by_user": True,
        "scientific_execution_approved": False,
        "registered_success_criteria_applied": False,
        "target_evaluations_run": True,
        "sealed_target_answers_accessed_after_proposals": True,
        "gpu_allocated": False,
        "input_bundle_sha256": input_hash,
        "answer_bundle_sha256": answer_hash,
        "runner_validation_reference": str(RUNNER_VALIDATION_OUTPUT.relative_to(ROOT)),
        "cpu_accelerator": execution["cpu_accelerator"],
    })
    write_json(OUTPUT / "EXPLORATORY_REPORT.json", {
        "schema_version": "crucible_cumulative_adaptation_exploratory_report_v1",
        "experiment": EXPERIMENT,
        "classification": "exploratory_post_protocol",
        "status": "completed",
        "protocol_status": "terminated_by_user_before_execution",
        "registered_scientific_result": False,
        "target_tasks": len(target_input_rows),
        "seeds": list(preflight.SEEDS),
        "arms": list(ARM_IDS),
        "execution": execution,
        "summary": summary,
        "resource_summary": {
            "wall_seconds": time.perf_counter() - started,
            "gpu_allocated": False,
            "gpu_seconds": 0.0,
            "sealed_answers_opened_after_proposals": True,
            "cpu_accelerator": execution["cpu_accelerator"],
        },
        "limitations": [
            "The registered protocol was terminated before target bundles were frozen; this is not a preregistered result.",
            "Target query-pair generation was deterministic but selected for this exploratory execution after protocol termination.",
            "No registered McNemar/bootstrap success criteria are applied.",
        ],
    })
    (OUTPUT / "README.md").write_text(
        "# Cumulative adaptation exploratory execution\n\n"
        "This run was explicitly authorized after the registered protocol was "
        "terminated. It is a post-protocol exploratory execution, not a valid "
        "preregistered scientific result. The sealed answer bundle was opened "
        "only after all four arms finalized their proposals.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "HASHES.json")
    write_json(OUTPUT / "HASHES.json", {str(path.relative_to(ROOT)): sha256_file(path) for path in files})
    return {"summary": summary, "output": str(OUTPUT), "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-workers", type=int, default=None)
    parser.add_argument("--cpu-window", type=int, default=None)
    args = parser.parse_args()
    result = run(args.cpu_workers, args.cpu_window)
    print(json.dumps({"status": "completed", "classification": "exploratory_post_protocol", "output": result["output"], "rows": result["rows"], "gpu_allocated": False}, sort_keys=True))


if __name__ == "__main__":
    main()
