#!/usr/bin/env python3
"""Run the deterministic Crucible Learning Lifecycle v0 benchmark."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from crucible_learning_lifecycle_v0 import (
    DOMAIN,
    RULE_NAMES,
    Observation,
    PersistentStore,
    ProceduralLearner,
    RawExampleRetriever,
    apply_rule,
    compatible_rules,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_learning_lifecycle_v0_20251025"
SEEDS = (42, 1337, 2025)
TASKS_PER_SEED = len(RULE_NAMES)
SUPPORT_COUNT = 2
VERIFICATION_COUNT = 3
QUERY_COUNT = 12
PROBE_BUDGET = 8


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_id(seed: int, index: int) -> str:
    return f"task_{seed}_{index:02d}"


def make_tasks() -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for seed in SEEDS:
        rng = random.Random(seed)
        rules = list(RULE_NAMES)
        rng.shuffle(rules)
        for index, rule_name in enumerate(rules):
            pairs = list(DOMAIN)
            rng.shuffle(pairs)
            support_pairs = pairs[:SUPPORT_COUNT]
            verification_pairs = pairs[SUPPORT_COUNT : SUPPORT_COUNT + VERIFICATION_COUNT]
            query_pairs = pairs[SUPPORT_COUNT + VERIFICATION_COUNT : SUPPORT_COUNT + VERIFICATION_COUNT + QUERY_COUNT]
            probe_pool = pairs[SUPPORT_COUNT + VERIFICATION_COUNT + QUERY_COUNT :]
            tasks.append({
                "task_id": task_id(seed, index),
                "seed": seed,
                "index": index,
                "hidden_rule": rule_name,
                "alias": f"glyph_{seed}_{index:02d}",
                "support_pairs": [list(pair) for pair in support_pairs],
                "verification_pairs": [list(pair) for pair in verification_pairs],
                "query_pairs": [list(pair) for pair in query_pairs],
                "probe_pool": [list(pair) for pair in probe_pool],
            })
    return tasks


def oracle_for(rule_name: str):
    return lambda a, b: apply_rule(rule_name, a, b)


def make_demos(task: Mapping[str, Any]) -> List[Observation]:
    oracle = oracle_for(str(task["hidden_rule"]))
    return [
        Observation.make(
            int(a), int(b), int(oracle(int(a), int(b))), "demonstration",
            {"task_id": task["task_id"], "demonstration_index": index},
        )
        for index, (a, b) in enumerate(task["support_pairs"])
    ]


def query_metrics(learner: ProceduralLearner, task: Mapping[str, Any]) -> Dict[str, Any]:
    oracle = oracle_for(str(task["hidden_rule"]))
    rows = []
    for a, b in task["query_pairs"]:
        predicted = learner.execute(str(task["alias"]), int(a), int(b))
        expected = int(oracle(int(a), int(b)))
        rows.append({"a": a, "b": b, "expected": expected, "predicted": predicted, "correct": predicted == expected, "attempted": predicted is not None})
    attempted = sum(row["attempted"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    return {
        "queries": len(rows),
        "attempted": attempted,
        "correct": correct,
        "coverage": attempted / len(rows),
        "accuracy_all_queries": correct / len(rows),
        "accuracy_on_attempts": correct / attempted if attempted else None,
        "rows": rows,
    }


def run_full_lifecycle(tasks: Sequence[Mapping[str, Any]], store_path: Path) -> Dict[str, Any]:
    store = PersistentStore(store_path)
    learner = ProceduralLearner(store, probe_budget=PROBE_BUDGET)
    acquisitions = []
    for task in tasks:
        result = learner.learn(
            str(task["alias"]),
            make_demos(task),
            [tuple(pair) for pair in task["verification_pairs"]],
            oracle_for(str(task["hidden_rule"])),
            [tuple(pair) for pair in task["probe_pool"]],
            {"task_id": task["task_id"], "source": "lifecycle_v0_benchmark", "seed": task["seed"]},
        )
        acquisitions.append({"task_id": task["task_id"], "alias": task["alias"], "result": result})

    reloaded = ProceduralLearner(PersistentStore(store_path), probe_budget=PROBE_BUDGET)
    reloaded.reset_context()
    queries = {str(task["task_id"]): query_metrics(reloaded, task) for task in tasks}
    successful = [item for item in acquisitions if item["result"]["status"] == "validated"]
    retained_correct = sum(value["correct"] for value in queries.values())
    retained_attempts = sum(value["attempted"] for value in queries.values())
    total_queries = sum(value["queries"] for value in queries.values())
    return {
        "store_path": str(store_path),
        "task_count": len(tasks),
        "acquisitions": acquisitions,
        "queries_after_reset": queries,
        "acquisition_success": len(successful),
        "retained_procedure_count": len(PersistentStore(store_path).procedures),
        "retained_queries": total_queries,
        "retained_attempts": retained_attempts,
        "retained_correct": retained_correct,
        "retained_coverage": retained_attempts / total_queries,
        "retained_accuracy": retained_correct / total_queries,
        "retained_accuracy_on_attempts": retained_correct / retained_attempts if retained_attempts else None,
        "mean_examples_and_probes": sum(int(item["result"]["cost"]["demonstrations"]) + int(item["result"]["cost"]["probes"]) for item in successful) / len(successful),
        "mean_probes": sum(int(item["result"]["cost"]["probes"]) for item in successful) / len(successful),
    }


def run_no_persistence(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    acquisitions = []
    for task in tasks:
        learner = ProceduralLearner(None, probe_budget=PROBE_BUDGET)
        result = learner.learn(
            str(task["alias"]), make_demos(task), [tuple(pair) for pair in task["verification_pairs"]],
            oracle_for(str(task["hidden_rule"])), [tuple(pair) for pair in task["probe_pool"]], {"task_id": task["task_id"], "source": "no_persistence_control"},
        )
        acquisitions.append(result)
    after_reset = ProceduralLearner(None)
    queries = []
    for task in tasks:
        metrics = query_metrics(after_reset, task)
        queries.append(metrics)
    total = sum(item["queries"] for item in queries)
    attempted = sum(item["attempted"] for item in queries)
    correct = sum(item["correct"] for item in queries)
    return {
        "task_count": len(tasks),
        "ephemeral_acquisition_success": sum(result["status"] == "validated_ephemeral" for result in acquisitions),
        "retained_after_reset": False,
        "queries": total,
        "attempted": attempted,
        "correct": correct,
        "coverage": attempted / total,
        "accuracy": correct / total,
    }


def run_raw_retrieval(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    retrievers = {str(task["task_id"]): RawExampleRetriever(make_demos(task)) for task in tasks}
    rows = []
    for task in tasks:
        retriever = retrievers[str(task["task_id"])]
        oracle = oracle_for(str(task["hidden_rule"]))
        for a, b in task["query_pairs"]:
            predicted = retriever.execute(int(a), int(b))
            expected = int(oracle(int(a), int(b)))
            rows.append({"predicted": predicted, "expected": expected, "attempted": predicted is not None, "correct": predicted == expected})
    return {
        "queries": len(rows),
        "attempted": sum(row["attempted"] for row in rows),
        "correct": sum(row["correct"] for row in rows),
        "coverage": sum(row["attempted"] for row in rows) / len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "stored_examples_per_task": SUPPORT_COUNT,
    }


def run_negative_controls(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    insufficient = []
    contradictory = []
    for task in tasks:
        demo = make_demos(task)[0]
        store = PersistentStore()
        learner = ProceduralLearner(store, probe_budget=0)
        insufficient.append(learner.learn(
            str(task["alias"]), [demo], [], oracle_for(str(task["hidden_rule"])), [], {"task_id": task["task_id"], "source": "insufficient_control"},
        ))
        bad = Observation.make(demo.a, demo.b, (demo.output + 1) % 10, "incorrect_demo", {"task_id": task["task_id"]})
        contradictory_store = PersistentStore()
        contradictory_learner = ProceduralLearner(contradictory_store, probe_budget=0)
        contradictory.append(contradictory_learner.learn(
            str(task["alias"]), [demo, bad], [], oracle_for(str(task["hidden_rule"])), [], {"task_id": task["task_id"], "source": "contradictory_control"},
        ))
    return {
        "insufficient": {
            "tasks": len(insufficient),
            "uncertain": sum(result["status"] == "uncertain" for result in insufficient),
            "promoted": sum(result["status"] in {"validated", "validated_ephemeral"} for result in insufficient),
            "statuses": dict(Counter(result["status"] for result in insufficient)),
        },
        "contradictory": {
            "tasks": len(contradictory),
            "failed": sum(result["status"] == "failed" for result in contradictory),
            "promoted": sum(result["status"] in {"validated", "validated_ephemeral"} for result in contradictory),
            "statuses": dict(Counter(result["status"] for result in contradictory)),
        },
    }


def run_revision(full: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]], store_path: Path) -> Dict[str, Any]:
    task = tasks[0]
    store = PersistentStore(store_path)
    learner = ProceduralLearner(store, probe_budget=PROBE_BUDGET)
    before = store.active(str(task["alias"]))
    if before is None:
        return {"status": "blocked_missing_active_procedure"}
    contradiction = (int(task["query_pairs"][0][0]), int(task["query_pairs"][0][1]))
    observed = (int(oracle_for(str(task["hidden_rule"]))(*contradiction)) + 1) % 10
    invalidated = learner.invalidate(str(task["alias"]), contradiction[0], contradiction[1], observed, "deliberate_contradiction_control")
    after_invalidation = PersistentStore(store_path).active(str(task["alias"]))
    relearn = learner.learn(
        str(task["alias"]), make_demos(task), [tuple(pair) for pair in task["verification_pairs"]],
        oracle_for(str(task["hidden_rule"])), [tuple(pair) for pair in task["probe_pool"]], {"task_id": task["task_id"], "source": "revision_control"},
    )
    reloaded = PersistentStore(store_path)
    versions = reloaded.procedures.get(str(task["alias"]), [])
    return {
        "task_id": task["task_id"],
        "invalidated": invalidated,
        "active_after_invalidation": after_invalidation is not None,
        "relearn_status": relearn["status"],
        "version_count": len(versions),
        "version_statuses": [version["status"] for version in versions],
        "revision_history_preserved": bool(versions and versions[0].get("revision_history")),
    }


def build_report(results: Mapping[str, Any]) -> str:
    full = results["full_lifecycle"]
    no_persistence = results["controls"]["no_persistence"]
    raw = results["controls"]["raw_example_retrieval"]
    negative = results["controls"]["negative_controls"]
    revision = results["revision"]
    lines = [
        "# Crucible Learning Lifecycle v0 — Final Report",
        "",
        "Deterministic CPU prototype over a bounded binary digit-rule hypothesis space. No GPU, neural adaptation, CMS2, recursion, or sealed test data.",
        "",
        "## Demonstrated lifecycle",
        "",
        f"- Tasks: {full['task_count']} across seeds 42, 1337 and 2025.",
        f"- Procedures acquired and verified: {full['acquisition_success']}/{full['task_count']}.",
        f"- Mean demonstrations plus active probes: {full['mean_examples_and_probes']:.2f}; mean probes: {full['mean_probes']:.2f}.",
        f"- Post-reset retained execution: {full['retained_correct']}/{full['retained_queries']} correct, coverage {full['retained_coverage']:.1%}.",
        f"- Persistence store retained {full['retained_procedure_count']} versioned procedure entries.",
        "",
        "## Controls",
        "",
        "| Control | Result |",
        "|---|---|",
        f"| No persistence | {no_persistence['correct']}/{no_persistence['queries']} after reset; coverage {no_persistence['coverage']:.1%} |",
        f"| Raw-example retrieval | {raw['correct']}/{raw['queries']}; coverage {raw['coverage']:.1%} |",
        f"| Insufficient evidence | {negative['insufficient']['uncertain']}/{negative['insufficient']['tasks']} uncertain; promoted {negative['insufficient']['promoted']} |",
        f"| Contradictory demonstrations | failed {negative['contradictory']['failed']}/{negative['contradictory']['tasks']}; promoted {negative['contradictory']['promoted']} |",
        f"| Revision control | invalidated={revision.get('invalidated')}; active after invalidation={revision.get('active_after_invalidation')}; versions={revision.get('version_count')} |",
        "",
        "## Interpretation boundary",
        "",
        "This demonstrates an engineered, bounded learning lifecycle: hypothesis filtering, active evidence acquisition, independent verification, provenance-tagged persistence, context reset, deterministic execution, and contradiction-driven invalidation/revision. It does not demonstrate neural meta-learning, autonomous hypothesis-space discovery, or general intelligence.",
        "",
        "## Artifacts",
        "",
        "- [Design](../../docs/CRUCIBLE_LEARNING_LIFECYCLE_V0_DESIGN_20251025.md)",
        "- [Benchmark manifest](TASKS.json)",
        "- [Machine-readable results](RESULTS.json)",
        "- [Persistent store](persistent/full_store.json)",
        "- [Unit tests](../../tools/test_learning_lifecycle_v0.py)",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks()
    write_json(OUTPUT / "TASKS.json", {"tasks": tasks, "rule_names": list(RULE_NAMES), "domain_size": len(DOMAIN), "labels_hidden_from_learner": True})

    full_store = OUTPUT / "persistent" / "full_store.json"
    full = run_full_lifecycle(tasks, full_store)
    controls = {
        "no_persistence": run_no_persistence(tasks),
        "raw_example_retrieval": run_raw_retrieval(tasks),
        "negative_controls": run_negative_controls(tasks),
    }
    revision = run_revision(full, tasks, full_store)
    results = {
        "experiment": "crucible_learning_lifecycle_v0_20251025",
        "status": "completed",
        "hardware": "cpu_only",
        "sealed_test_accessed": False,
        "hypothesis_space": list(RULE_NAMES),
        "full_lifecycle": full,
        "controls": controls,
        "revision": revision,
        "unit_tests_required": True,
    }
    write_json(OUTPUT / "RESULTS.json", results)
    (OUTPUT / "REPORT.md").write_text(build_report(results), encoding="utf-8")
    hashes = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json"}:
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    write_json(OUTPUT / "FINALIZATION.json", {
        "status": "completed",
        "report_sha256": sha256(OUTPUT / "REPORT.md"),
        "results_sha256": sha256(OUTPUT / "RESULTS.json"),
        "store_sha256": sha256(full_store),
        "all_tasks_acquired": full["acquisition_success"] == full["task_count"],
        "post_reset_retained_accuracy": full["retained_accuracy"],
    })
    print(json.dumps({"status": "completed", "tasks": len(tasks), "acquired": full["acquisition_success"], "retained_accuracy": full["retained_accuracy"], "revision": revision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
