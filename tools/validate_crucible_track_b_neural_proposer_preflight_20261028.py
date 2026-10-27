#!/usr/bin/env python3
"""Independently validate the CPU-only Track B proposer preflight package.

This validator checks the frozen grammar, task partitioning, leakage boundaries,
evaluation denominator, and artifact hashes. It performs no training and does
not inspect any sealed test set.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments" / "crucible_track_b_neural_proposer_preflight_20261028"
SOURCE = ROOT / "tools" / "prepare_crucible_track_b_neural_proposer_preflight_20261028.py"

EXPECTED_FILES = {
    "GRAMMAR_SPEC.json",
    "TASK_MANIFEST.json",
    "PROPOSER_INPUTS.jsonl",
    "TRAIN_LABELS.jsonl",
    "EVAL_ORACLE.jsonl",
    "LEAKAGE_AUDIT.json",
    "INSTRUMENTATION_PLAN.json",
    "INSTRUMENTATION_PLAN.md",
    "README.md",
    "HASHES.json",
    "PREFLIGHT_RESULTS.json",
}
FORBIDDEN_PROPOSER_KEYS = {
    "target_ast",
    "target_ast_hash",
    "target_truth_table_hash",
    "truth_table",
    "answer_function",
}


def load_json(name: str) -> Any:
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def load_jsonl(name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate((PACKAGE / name).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{name}:{line_number} is not an object")
        rows.append(value)
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_keys(value: Any, keys: Iterable[str]) -> List[str]:
    wanted = set(keys)
    found: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in wanted:
                found.append(key)
            found.extend(find_keys(child, wanted))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_keys(child, wanted))
    return found


def eval_direct(op: str, a: int, b: int) -> int:
    if op == "add_mod10":
        return (a + b) % 10
    if op == "sub_mod10":
        return (a - b) % 10
    if op == "mul_mod10":
        return (a * b) % 10
    if op == "xor_bits_mod10":
        return (a ^ b) % 10
    if op == "min":
        return min(a, b)
    if op == "max":
        return max(a, b)
    raise ValueError(f"unknown operator {op}")


def pair_key(row: Mapping[str, Any]) -> Tuple[int, int]:
    return int(row["a"]), int(row["b"])


def pair_list_key(value: Sequence[Sequence[int]]) -> set[Tuple[int, int]]:
    return {(int(pair[0]), int(pair[1])) for pair in value}


def main() -> int:
    errors: List[str] = []
    if not PACKAGE.is_dir():
        print(json.dumps({"status": "fail", "errors": [f"missing package: {PACKAGE}"]}, indent=2))
        return 1

    actual_files = {path.name for path in PACKAGE.iterdir() if path.is_file()}
    missing = sorted(EXPECTED_FILES - actual_files)
    unexpected = sorted(actual_files - EXPECTED_FILES)
    if missing:
        errors.append(f"missing files: {missing}")
    if unexpected:
        errors.append(f"unexpected files: {unexpected}")
    if errors:
        print(json.dumps({"status": "fail", "errors": errors}, indent=2))
        return 1

    try:
        grammar = load_json("GRAMMAR_SPEC.json")
        manifest = load_json("TASK_MANIFEST.json")
        leakage = load_json("LEAKAGE_AUDIT.json")
        instrumentation = load_json("INSTRUMENTATION_PLAN.json")
        preflight = load_json("PREFLIGHT_RESULTS.json")
        hashes = load_json("HASHES.json")
        proposer_inputs = load_jsonl("PROPOSER_INPUTS.jsonl")
        train_labels = load_jsonl("TRAIN_LABELS.jsonl")
        eval_oracle = load_jsonl("EVAL_ORACLE.jsonl")
    except Exception as exc:  # pragma: no cover - failure path is user-facing
        errors.append(f"artifact parsing failed: {exc}")
        print(json.dumps({"status": "fail", "errors": errors}, indent=2))
        return 1

    # Frozen grammar and explicit XOR semantics.
    if grammar.get("schema_version") != "digit_expr_mod10_v1":
        errors.append("unexpected grammar schema")
    if grammar.get("xor_semantics") != "integer_bitwise_xor_then_mod10":
        errors.append("XOR semantics are not frozen as integer bitwise XOR then modulo 10")
    expected_ops = {"add_mod10", "sub_mod10", "mul_mod10", "xor_bits_mod10", "min", "max"}
    if set(grammar.get("operators", [])) != expected_ops:
        errors.append("operator set differs from the registered grammar")
    for op, vectors in grammar.get("test_vectors", {}).items():
        for vector in vectors:
            got = eval_direct(op, int(vector["a"]), int(vector["b"]))
            if got != int(vector["expected"]):
                errors.append(f"grammar vector failed: {op}({vector['a']},{vector['b']})={got}")

    tasks = manifest.get("tasks", [])
    task_by_id = {task.get("task_id"): task for task in tasks}
    proposer_by_id = {row.get("task_id"): row for row in proposer_inputs}
    oracle_by_id = {row.get("task_id"): row for row in eval_oracle}
    if len(task_by_id) != len(tasks):
        errors.append("task manifest contains duplicate task IDs")
    if set(task_by_id) != set(proposer_by_id):
        errors.append("proposer-input task IDs do not match task manifest")

    forbidden_found = sorted(set(find_keys(proposer_inputs, FORBIDDEN_PROPOSER_KEYS)))
    if forbidden_found:
        errors.append(f"proposer inputs contain forbidden target fields: {forbidden_found}")

    split_counts = {split: sum(task.get("split") == split for task in tasks) for split in ("train", "validation", "test")}
    reported = manifest.get("task_counts", {})
    if reported.get("all") != len(tasks) or any(reported.get(split) != count for split, count in split_counts.items()):
        errors.append(f"manifest task counts disagree with task rows: {split_counts}")

    train_tasks = [task for task in tasks if task.get("split") == "train"]
    eval_tasks = [task for task in tasks if task.get("split") in {"validation", "test"}]
    included_train = [task for task in train_tasks if task.get("training_included")]
    eval_novel = [task for task in eval_tasks if task.get("functionally_novel") and task.get("structurally_novel")]
    unresolvable = [task for task in eval_tasks if not task.get("identifiable_under_budget")]
    if len(included_train) != reported.get("train_included"):
        errors.append("included-training count mismatch")
    if len(eval_novel) != reported.get("held_out_novel"):
        errors.append("held-out novelty count mismatch")
    if len(unresolvable) != reported.get("eval_unresolvable"):
        errors.append("unresolvable evaluation count mismatch")
    if not unresolvable:
        errors.append("no retained unresolvable evaluation tasks")
    if any(task.get("split") == "train" and task.get("training_included") and not task.get("identifiable_under_budget") for task in tasks):
        errors.append("an unidentifiable training task is marked included")

    train_ast = {task.get("target_ast_hash") for task in train_tasks}
    eval_ast = {task.get("target_ast_hash") for task in eval_tasks}
    train_truth = {task.get("target_truth_table_hash") for task in train_tasks}
    eval_truth = {task.get("target_truth_table_hash") for task in eval_tasks}
    if train_ast & eval_ast:
        errors.append("train/evaluation AST hash overlap")
    if train_truth & eval_truth:
        errors.append("train/evaluation function hash overlap")
    if any(not task.get("structurally_novel") or not task.get("functionally_novel") for task in eval_tasks):
        errors.append("an evaluation task is not marked structurally and functionally novel")

    # Demonstrations, verifier pairs and query pairs must be disjoint per task.
    for task in tasks:
        demo = task.get("demonstration_pairs", [])
        verifier = task.get("verifier_pairs", [])
        queries = task.get("query_pairs", [])
        sets = [pair_list_key(rows) for rows in (demo, verifier, queries)]
        if any(len(rows) != len(pair_set) for rows, pair_set in zip((demo, verifier, queries), sets)):
            errors.append(f"duplicate pair within task {task.get('task_id')}")
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            errors.append(f"demonstration/verifier/query overlap in {task.get('task_id')}")
        if not task.get("identifiable_under_budget") and not verifier:
            errors.append(f"unresolvable task has no retained verifier example: {task.get('task_id')}")
        proposer = proposer_by_id.get(task.get("task_id"), {})
        proposer_demo = {pair_key(row) for row in proposer.get("demonstrations", [])}
        if proposer_demo != sets[0]:
            errors.append(f"proposer demonstrations do not match manifest: {task.get('task_id')}")
        if task.get("split") in {"validation", "test"}:
            oracle = oracle_by_id.get(task.get("task_id"), {})
            oracle_verifier = {pair_key(row) for row in oracle.get("verifier_examples", [])}
            oracle_queries = {pair_key(row) for row in oracle.get("query_examples", [])}
            if oracle_verifier != sets[1] or oracle_queries != sets[2]:
                errors.append(f"evaluation examples do not match manifest: {task.get('task_id')}")

    label_ids = {row.get("task_id") for row in train_labels}
    included_ids = {task.get("task_id") for task in included_train}
    if label_ids != included_ids:
        errors.append("training labels do not match included training tasks")
    if set(oracle_by_id) != {task.get("task_id") for task in eval_tasks}:
        errors.append("evaluation oracle rows do not match validation/test tasks")

    # Self-reported safety boundaries and instrumentation requirements.
    if leakage.get("status") != "pass":
        errors.append("generator leakage audit is not passing")
    if leakage.get("proposer_inputs_have_target_ast") or leakage.get("proposer_inputs_have_target_truth_hash"):
        errors.append("generator leakage audit reports target fields in proposer inputs")
    if leakage.get("sealed_test_accessed") or preflight.get("sealed_test_accessed"):
        errors.append("sealed test access is reported")
    if preflight.get("training_performed") or preflight.get("scientific_checkpoint_selected"):
        errors.append("preflight reports training or checkpoint selection")
    required_counters = {"candidate_evaluations", "verifier_calls", "proposer_forward_passes", "decoder_tokens", "cpu_seconds", "gpu_seconds", "wall_clock_seconds", "peak_memory_bytes"}
    if not required_counters.issubset(set(instrumentation.get("per_task_counters", []))):
        errors.append("instrumentation plan is missing required counters")

    # Hashes cover every final data artifact except the two self-referential metadata files.
    expected_hashed = {str((PACKAGE / name).relative_to(ROOT)) for name in EXPECTED_FILES - {"HASHES.json", "PREFLIGHT_RESULTS.json"}}
    if set(hashes) != expected_hashed:
        errors.append("HASHES.json does not cover exactly the non-self-referential artifacts")
    for relative, expected in hashes.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"artifact hash mismatch: {relative}")
    source_hash = sha256(SOURCE)
    if preflight.get("source_script_sha256") != source_hash:
        errors.append("preflight source-script hash does not match current generator")

    result = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "checks": {
            "task_count": len(tasks),
            "training_task_count": len(train_tasks),
            "held_out_novel_task_count": len(eval_novel),
            "unresolvable_eval_count": len(unresolvable),
            "train_eval_ast_overlap": len(train_ast & eval_ast),
            "train_eval_function_overlap": len(train_truth & eval_truth),
            "hashed_artifact_count": len(hashes),
            "training_performed": bool(preflight.get("training_performed")),
            "sealed_test_accessed": bool(preflight.get("sealed_test_accessed")),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
