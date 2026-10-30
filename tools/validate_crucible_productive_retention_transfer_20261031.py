#!/usr/bin/env python3
"""Independently validate the productive-retention manifest and sealed split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


EXPECTED_COUNTS = {
    "source_train": 32,
    "source_validation": 16,
    "target_dependent_train": 64,
    "target_dependent_validation": 32,
    "target_dependent_sealed": 64,
    "target_independent_train": 64,
    "target_independent_validation": 32,
    "target_independent_sealed": 64,
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(expr: Dict[str, Any], values: List[int]) -> int:
    op = expr["op"]
    if op == "var":
        return int(values[int(expr["index"])])
    children = [evaluate(child, values) for child in expr["children"]]
    if op == "not":
        return 1 - children[0]
    if op == "and":
        return int(children[0] & children[1])
    if op == "or":
        return int(children[0] | children[1])
    if op == "xor":
        return int(children[0] ^ children[1])
    if op == "nand":
        return int(1 - (children[0] & children[1]))
    if op == "nor":
        return int(1 - (children[0] | children[1]))
    if op == "xnor":
        return int(1 - (children[0] ^ children[1]))
    raise ValueError(f"unknown operator: {op}")


def depth(expr: Dict[str, Any]) -> int:
    if expr["op"] == "var":
        return 0
    return 1 + max(depth(child) for child in expr["children"])


def truth_table(expr: Dict[str, Any]) -> str:
    return "".join(
        str(evaluate(expr, [(mask >> bit) & 1 for bit in range(4)]))
        for mask in range(16)
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("experiments/crucible_productive_retention_transfer_20261031/manifest"),
    )
    parser.add_argument(
        "--sealed-dir",
        type=Path,
        default=Path("/home/timbushnell/crucible_productive_retention_transfer_sealed_20261031"),
    )
    args = parser.parse_args()
    manifest_dir = args.manifest_dir.resolve()
    sealed_dir = args.sealed_dir.resolve()
    failures: List[str] = []
    checks: Dict[str, bool] = {}

    manifest_path = manifest_dir / "MANIFEST.json"
    if not manifest_path.exists():
        failures.append(f"missing manifest: {manifest_path}")
        manifest: Dict[str, Any] = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    checks["manifest_exists"] = bool(manifest)
    checks["scientific_authorization_false"] = manifest.get("scientific_authorization") is False
    if not checks["scientific_authorization_false"]:
        failures.append("manifest scientific_authorization is not false")

    public_rows: List[Dict[str, Any]] = []
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for relative, expected_hash in manifest.get("public_files", {}).items():
        path = manifest_dir / relative
        if not path.exists():
            failures.append(f"missing public file: {relative}")
            continue
        if sha256_file(path) != expected_hash:
            failures.append(f"public file hash mismatch: {relative}")
        rows = read_jsonl(path)
        public_rows.extend(rows)
        split = path.name.removesuffix("_public.jsonl").removesuffix(".jsonl")
        rows_by_split[f"seed_{path.parent.name.split('_')[-1]}/{split}"] = rows

    checks["public_file_hashes"] = not any("public file hash mismatch" in item for item in failures)
    checks["public_rows_have_expected_schema"] = True
    task_ids = set()
    structural: Dict[str, set[str]] = {"source": set(), "target_dependent": set(), "target_independent": set()}
    functional: Dict[str, set[str]] = {"source": set(), "target_dependent": set(), "target_independent": set()}
    for row in public_rows:
        required = {"task_id", "family", "split", "canonical_ast_sha256", "truth_table_sha256", "demonstrations", "query"}
        if not required.issubset(row):
            checks["public_rows_have_expected_schema"] = False
            failures.append(f"schema missing in {row.get('task_id', '<unknown>')}")
            continue
        if row["task_id"] in task_ids:
            failures.append(f"duplicate task id: {row['task_id']}")
        task_ids.add(row["task_id"])
        if row["family"] not in structural:
            failures.append(f"unknown family: {row['family']}")
            continue
        structural[row["family"]].add(row["canonical_ast_sha256"])
        functional[row["family"]].add(row["truth_table_sha256"])
        if row["split"].endswith("sealed") and "query_output" in row:
            failures.append(f"sealed public answer exposed: {row['task_id']}")
        if len(row["demonstrations"]) != 8 or len(row["query"]) != 4:
            failures.append(f"episode shape mismatch: {row['task_id']}")

    checks["sealed_public_answers_hidden"] = not any("sealed public answer exposed" in item for item in failures)
    checks["task_ids_unique"] = len(task_ids) == len(public_rows)
    checks["episode_shapes"] = not any("episode shape mismatch" in item for item in failures)

    for seed in manifest.get("seeds", []):
        seed_key = f"seed_{seed}"
        for split, expected_count in EXPECTED_COUNTS.items():
            key = f"{seed_key}/{split}"
            actual = len(rows_by_split.get(key, []))
            if actual != expected_count:
                failures.append(f"count mismatch {key}: expected {expected_count}, got {actual}")
    checks["counts"] = not any(item.startswith("count mismatch") for item in failures)

    checks["source_target_structural_disjoint"] = not (
        structural["source"] & structural["target_dependent"]
        or structural["source"] & structural["target_independent"]
    )
    checks["source_target_functional_disjoint"] = not (
        functional["source"] & functional["target_dependent"]
        or functional["source"] & functional["target_independent"]
    )
    checks["cross_family_structural_disjoint"] = not (
        structural["target_dependent"] & structural["target_independent"]
    )
    if not checks["source_target_structural_disjoint"]:
        failures.append("source/target structural overlap")
    if not checks["source_target_functional_disjoint"]:
        failures.append("source/target functional overlap")
    if not checks["cross_family_structural_disjoint"]:
        failures.append("dependent/independent structural overlap")

    private_hashes = manifest.get("sealed_answer_files_outside_git", {})
    checks["sealed_files_present_and_hashed"] = True
    checks["sealed_rows_match_public"] = True
    checks["sealed_ast_depths"] = True
    sealed_functions: set[str] = set()
    for relative, expected_hash in private_hashes.items():
        path = sealed_dir / relative
        if not path.exists():
            checks["sealed_files_present_and_hashed"] = False
            failures.append(f"missing sealed file: {relative}")
            continue
        if sha256_file(path) != expected_hash:
            checks["sealed_files_present_and_hashed"] = False
            failures.append(f"sealed file hash mismatch: {relative}")
        for oracle in read_jsonl(path):
            if not {"ast", "truth_table", "query_output", "canonical_ast_sha256", "truth_table_sha256"}.issubset(oracle):
                checks["sealed_rows_match_public"] = False
                failures.append(f"sealed schema missing: {oracle.get('task_id', '<unknown>')}")
                continue
            if sha256_text(canonical(oracle["ast"])) != oracle["canonical_ast_sha256"]:
                checks["sealed_rows_match_public"] = False
                failures.append(f"AST hash mismatch: {oracle['task_id']}")
            if truth_table(oracle["ast"]) != oracle["truth_table"]:
                checks["sealed_rows_match_public"] = False
                failures.append(f"truth table mismatch: {oracle['task_id']}")
            if sha256_text(oracle["truth_table"]) != oracle["truth_table_sha256"]:
                checks["sealed_rows_match_public"] = False
                failures.append(f"truth table hash mismatch: {oracle['task_id']}")
            family = oracle.get("family")
            expression_depth = depth(oracle["ast"])
            if family == "source" and expression_depth > 2:
                checks["sealed_ast_depths"] = False
                failures.append(f"source AST exceeds depth 2: {oracle['task_id']}")
            if family in {"target_dependent", "target_independent"} and expression_depth != 3:
                checks["sealed_ast_depths"] = False
                failures.append(f"target AST is not exactly depth 3: {oracle['task_id']}")
            sealed_functions.add(oracle["truth_table_sha256"])

    public_training_functions = {
        row["truth_table_sha256"]
        for row in public_rows
        if not row.get("split", "").endswith("sealed")
    }
    checks["sealed_target_functions_do_not_overlap_public_training"] = not (
        sealed_functions & public_training_functions
    )
    if not checks["sealed_target_functions_do_not_overlap_public_training"]:
        failures.append("sealed target function overlaps a public function")

    report = {
        "schema_version": "crucible-productive-retention-transfer-independent-validation-v1-20261031",
        "status": "PASS" if not failures else "FAIL",
        "scientific_authorization": False,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "checks": checks,
        "counts": {key: len(value) for key, value in sorted(rows_by_split.items())},
        "failures": failures,
        "limitations": [
            "This validates task-family separation and sealed-answer handling; it does not establish source competence or productive transfer.",
            "The Boolean curriculum is a controlled diagnostic and does not establish transfer to natural-language reasoning.",
        ],
    }
    report_path = manifest_dir / "INDEPENDENT_VALIDATION_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
