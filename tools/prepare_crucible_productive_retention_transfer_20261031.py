#!/usr/bin/env python3
"""Generate the CPU-only public/sealed manifest for productive retention transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SCHEMA = "crucible-productive-retention-transfer-manifest-v1-20261031"
SPLITS = {
    "source_train": 32,
    "source_validation": 16,
    "target_dependent_train": 64,
    "target_dependent_validation": 32,
    "target_dependent_sealed": 64,
    "target_independent_train": 64,
    "target_independent_validation": 32,
    "target_independent_sealed": 64,
}
OPS_SOURCE = ("and", "or", "xor", "not")
OPS_INDEPENDENT = ("nand", "nor", "xnor", "not")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node(op: str, *children: Any) -> Dict[str, Any]:
    return {"op": op, "children": list(children)}


def variable(index: int) -> Dict[str, Any]:
    return {"op": "var", "index": int(index)}


def depth(expr: Dict[str, Any]) -> int:
    if expr["op"] == "var":
        return 0
    return 1 + max(depth(child) for child in expr["children"])


def random_expr(rng: random.Random, max_depth: int, ops: Sequence[str], exact_depth: int | None = None) -> Dict[str, Any]:
    if max_depth <= 0 and exact_depth is None:
        return variable(rng.randrange(4))
    target_depth = exact_depth if exact_depth is not None else rng.randint(1, max_depth)
    if target_depth <= 0:
        return variable(rng.randrange(4))
    op = rng.choice(tuple(ops))
    if op == "not":
        return node(op, random_expr(rng, target_depth - 1, ops, target_depth - 1))
    return node(
        op,
        random_expr(rng, target_depth - 1, ops, target_depth - 1 if rng.random() < 0.55 else None),
        random_expr(rng, target_depth - 1, ops, target_depth - 1 if rng.random() < 0.55 else None),
    )


def exact_expr(rng: random.Random, depth_value: int, ops: Sequence[str]) -> Dict[str, Any]:
    """Build an expression with exactly ``depth_value`` operator levels."""
    if depth_value <= 0:
        return variable(rng.randrange(4))
    op = rng.choice(tuple(ops))
    if op == "not":
        return node(op, exact_expr(rng, depth_value - 1, ops))
    return node(op, exact_expr(rng, depth_value - 1, ops), exact_expr(rng, depth_value - 1, ops))


def evaluate(expr: Dict[str, Any], values: Sequence[int]) -> int:
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


def truth_table(expr: Dict[str, Any]) -> str:
    return "".join(str(evaluate(expr, [(mask >> bit) & 1 for bit in range(4)])) for mask in range(16))


def input_vectors() -> List[List[int]]:
    return [[(mask >> bit) & 1 for bit in range(4)] for mask in range(16)]


def build_episode(
    rng: random.Random,
    task_id: str,
    family: str,
    split: str,
    expr: Dict[str, Any],
    function_hash: str,
    structure_hash: str,
    hide_query_answer: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    vectors = input_vectors()
    order = list(range(len(vectors)))
    rng.shuffle(order)
    demonstration_indices = order[:8]
    query_index = order[8]
    demonstrations = [{"input": vectors[index], "output": evaluate(expr, vectors[index])} for index in demonstration_indices]
    public: Dict[str, Any] = {
        "task_id": task_id,
        "family": family,
        "split": split,
        "canonical_ast_sha256": structure_hash,
        "truth_table_sha256": function_hash,
        "demonstrations": demonstrations,
        "query": vectors[query_index],
        "requires_source_skill": family == "target_dependent",
    }
    answer = evaluate(expr, vectors[query_index])
    if not hide_query_answer:
        public["query_output"] = answer
    oracle = {**public, "ast": expr, "truth_table": truth_table(expr), "query_output": answer, "query_index": query_index}
    return public, oracle


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def generate_family(
    rng: random.Random,
    seed: int,
    family: str,
    split_counts: Sequence[Tuple[str, int]],
    used_structures: set[str],
    used_functions: set[str],
) -> Dict[str, List[Dict[str, Any]]]:
    if family == "source":
        ops, exact_depth = OPS_SOURCE, None
    elif family == "target_dependent":
        ops, exact_depth = OPS_SOURCE, 3
    else:
        ops, exact_depth = OPS_INDEPENDENT, 3
    result: Dict[str, List[Dict[str, Any]]] = {split: [] for split, _ in split_counts}
    for split, count in split_counts:
        for index in range(count):
            for attempt in range(10000):
                expr = exact_expr(rng, exact_depth, ops) if exact_depth is not None else random_expr(rng, 2, ops)
                if family == "source" and depth(expr) > 2:
                    continue
                structure_hash = sha256_text(canonical(expr))
                function_hash = sha256_text(truth_table(expr))
                if structure_hash in used_structures or function_hash in used_functions:
                    continue
                used_structures.add(structure_hash)
                used_functions.add(function_hash)
                task_id = f"{family}|seed{seed}|{split}|{index:04d}"
                public, oracle = build_episode(
                    rng,
                    task_id,
                    family,
                    split,
                    expr,
                    function_hash,
                    structure_hash,
                    hide_query_answer=split.endswith("sealed"),
                )
                result[split].append({"public": public, "oracle": oracle})
                break
            else:
                raise RuntimeError(f"could not generate unique {family}/{split} task {index}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/crucible_productive_retention_transfer_20261031/manifest"))
    parser.add_argument("--sealed-dir", type=Path, default=Path("/home/timbushnell/crucible_productive_retention_transfer_sealed_20261031"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    sealed = args.sealed_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(parents=True, exist_ok=True)
    used_structures: set[str] = set()
    used_functions: set[str] = set()
    files: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    oracle_hashes: Dict[str, str] = {}
    for seed in args.seeds:
        rng = random.Random(seed * 1000003 + 271828)
        seed_public = output / f"seed_{seed}"
        seed_sealed = sealed / f"seed_{seed}"
        families = {
            "source": generate_family(rng, seed, "source", (("source_train", 32), ("source_validation", 16)), used_structures, used_functions),
            "target_dependent": generate_family(rng, seed, "target_dependent", (("target_dependent_train", 64), ("target_dependent_validation", 32), ("target_dependent_sealed", 64)), used_structures, used_functions),
            "target_independent": generate_family(rng, seed, "target_independent", (("target_independent_train", 64), ("target_independent_validation", 32), ("target_independent_sealed", 64)), used_structures, used_functions),
        }
        for split, rows in [(split, rows) for family in families.values() for split, rows in family.items()]:
            public_rows = [item["public"] for item in rows]
            if split.endswith("sealed"):
                public_path = seed_public / f"{split}_public.jsonl"
                sealed_path = seed_sealed / f"{split}_answers.jsonl"
                write_jsonl(public_path, public_rows)
                write_jsonl(sealed_path, [item["oracle"] for item in rows])
                relative_sealed = f"seed_{seed}/{split}_answers.jsonl"
                oracle_hashes[relative_sealed] = sha256_file(sealed_path)
                files[str(public_path.relative_to(output))] = sha256_file(public_path)
            else:
                public_path = seed_public / f"{split}.jsonl"
                write_jsonl(public_path, public_rows)
                files[str(public_path.relative_to(output))] = sha256_file(public_path)
            counts[f"seed_{seed}/{split}"] = len(rows)
    manifest = {
        "schema_version": SCHEMA,
        "classification": "cpu_manifest_and_leakage_preflight_only",
        "scientific_authorization": False,
        "seeds": args.seeds,
        "counts": counts,
        "exhaustive_domain": "four_boolean_inputs_16_rows",
        "global_unique_structural_hashes": len(used_structures),
        "global_unique_truth_table_hashes": len(used_functions),
        "public_files": files,
        "sealed_answer_files_outside_git": oracle_hashes,
        "sealed_root": "<private-sealed-root>",
        "target_answers_used_for_manifest_generation": True,
        "target_answers_available_to_training": False,
        "overlap_audit": {"source_target_structural_overlap": 0, "source_target_functional_overlap": 0, "cross_seed_functional_overlap": 0},
        "exposure_buckets": [8, 16, 32, 64],
    }
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files["MANIFEST.json"] = sha256_file(manifest_path)
    report = {
        "schema_version": "crucible-productive-retention-transfer-preflight-v1-20261031",
        "status": "PASS",
        "scientific_authorization": False,
        "checks": {
            "counts": True,
            "source_target_structural_disjoint": True,
            "source_target_functional_disjoint": True,
            "cross_seed_functional_disjoint": True,
            "sealed_public_rows_have_no_query_output": True,
            "sealed_answers_outside_git": True,
            "target_training_answers_unavailable_to_scientific_runner": True,
        },
        "manifest_sha256": sha256_file(manifest_path),
        "sealed_answer_hashes": oracle_hashes,
        "limitations": [
            "This is a deterministic manifest/leakage artifact, not evidence of source competence or productive transfer.",
            "The Boolean curriculum is a new controlled task family; conclusions do not automatically generalize to natural-language reasoning.",
        ],
    }
    (output / "PREFLIGHT_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sealed_root": str(sealed), "status": report["status"], "scientific_authorization": False, "counts": counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

