#!/usr/bin/env python3
"""Build and validate a CPU-only source-curriculum/evaluator repair.

This is deliberately separate from the failed productive-retention manifest and
source gate.  It creates no target procedures, performs no neural training and
does not read sealed answers.  The repair separates:

* query identifiability: every grammar-consistent function gives the same query;
* function identifiability: exactly one grammar function fits the demonstrations;
* explicit procedure execution: the procedure is supplied rather than induced.

Validation recomputes truth tables and compatible-function sets from the public
and private source fixtures, then exercises the one-step answer decoder with
synthetic logits.  The private source oracles are outside Git and are only for
CPU validation; they are not target answers.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "crucible_source_curriculum_repair_20261031"
DEFAULT_ORACLE = Path("/home/timbushnell/crucible_source_curriculum_repair_oracles_20261031")
SCHEMA = "crucible-source-curriculum-repair-v1-20261031"
SEEDS = (42, 1337)
OPS = ("and", "or", "xor", "not")
INPUT_COUNT = 4
DOMAIN_SIZE = 1 << INPUT_COUNT
RAW_BYTE_START = 3
ZERO_TOKEN = RAW_BYTE_START + ord("0")
ONE_TOKEN = RAW_BYTE_START + ord("1")

# The split sizes are intentionally larger than the failed 32/16 source gate,
# while remaining small enough for deterministic CPU validation.
SOURCE_COUNTS = {
    "source_train": 32,
    "source_validation": 32,
    "source_function_validation": 16,
    "source_test": 32,
}
EXECUTION_COUNTS = {
    "execution_train": 16,
    "execution_validation": 16,
    "execution_test": 16,
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node(op: str, *children: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": op, "children": list(children)}


def variable(index: int) -> Dict[str, Any]:
    return {"op": "var", "index": int(index)}


def depth(expr: Mapping[str, Any]) -> int:
    if expr["op"] == "var":
        return 0
    return 1 + max(depth(child) for child in expr["children"])


def evaluate(expr: Mapping[str, Any], values: Sequence[int]) -> int:
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
    raise ValueError(f"unknown operator {op}")


def input_vectors() -> List[List[int]]:
    return [[(mask >> bit) & 1 for bit in range(INPUT_COUNT)] for mask in range(DOMAIN_SIZE)]


VECTORS = input_vectors()


def truth_table(expr: Mapping[str, Any]) -> str:
    return "".join(str(evaluate(expr, values)) for values in VECTORS)


def enumerate_expression_pool(max_depth: int = 2) -> List[Dict[str, Any]]:
    """Return one canonical AST representative for every distinct truth table."""
    by_depth: Dict[int, List[Dict[str, Any]]] = {0: [variable(i) for i in range(INPUT_COUNT)]}
    all_exprs = list(by_depth[0])
    for current_depth in range(1, max_depth + 1):
        children = [expr for level in range(current_depth) for expr in by_depth[level]]
        generated: List[Dict[str, Any]] = []
        for child in children:
            generated.append(node("not", child))
        for left in children:
            for right in children:
                for op in ("and", "or", "xor"):
                    generated.append(node(op, left, right))
        by_depth[current_depth] = generated
        all_exprs.extend(generated)

    representatives: Dict[str, Dict[str, Any]] = {}
    for expr in all_exprs:
        table = truth_table(expr)
        encoded = canonical(expr)
        if table not in representatives or encoded < canonical(representatives[table]):
            representatives[table] = expr
    return [representatives[key] for key in sorted(representatives)]


def compatible_indices(functions: Sequence[Mapping[str, Any]], observations: Mapping[int, int]) -> List[int]:
    compatible: List[int] = []
    for index, function in enumerate(functions):
        table = truth_table(function)
        if all(int(table[row]) == int(value) for row, value in observations.items()):
            compatible.append(index)
    return compatible


def choose_query_evidence(
    functions: Sequence[Mapping[str, Any]],
    target_index: int,
    rng: random.Random,
    demo_count: int = 8,
) -> Dict[str, Any] | None:
    """Find eight demonstrations that uniquely determine the query answer."""
    target_table = truth_table(functions[target_index])
    rows = [row for row in range(DOMAIN_SIZE)]
    query_order = rows[:]
    rng.shuffle(query_order)
    for query_index in query_order:
        available = [row for row in rows if row != query_index]
        opposite = {
            index
            for index, function in enumerate(functions)
            if truth_table(function)[query_index] != target_table[query_index]
        }
        selected: List[int] = []
        remaining = set(opposite)
        while remaining:
            choices = []
            for row in available:
                if row in selected:
                    continue
                covered = sum(
                    1 for index in remaining if truth_table(functions[index])[row] != target_table[row]
                )
                choices.append((covered, -row, row))
            if not choices or max(item[0] for item in choices) == 0:
                break
            _, _, chosen = max(choices)
            selected.append(chosen)
            remaining = {
                index for index in remaining if truth_table(functions[index])[chosen] == target_table[chosen]
            }
        if remaining or len(selected) > demo_count:
            continue
        fillers = [row for row in available if row not in selected]
        rng.shuffle(fillers)
        selected.extend(fillers[: demo_count - len(selected)])
        rng.shuffle(selected)
        observations = {row: int(target_table[row]) for row in selected}
        compatible = compatible_indices(functions, observations)
        outputs = sorted({int(truth_table(functions[index])[query_index]) for index in compatible})
        if len(outputs) == 1 and len(selected) == demo_count:
            return {
                "query_index": query_index,
                "demonstration_indices": selected,
                "compatible_function_count": len(compatible),
                "compatible_query_outputs": outputs,
                "identifiability": "query",
            }
    return None


def choose_function_evidence(
    functions: Sequence[Mapping[str, Any]],
    target_index: int,
    rng: random.Random,
) -> Dict[str, Any] | None:
    """Find a held-out query whose other 15 rows identify one grammar function."""
    target_table = truth_table(functions[target_index])
    query_order = list(range(DOMAIN_SIZE))
    rng.shuffle(query_order)
    for query_index in query_order:
        observations = {row: int(target_table[row]) for row in range(DOMAIN_SIZE) if row != query_index}
        compatible = compatible_indices(functions, observations)
        outputs = sorted({int(truth_table(functions[index])[query_index]) for index in compatible})
        if len(compatible) == 1 and outputs == [int(target_table[query_index])]:
            return {
                "query_index": query_index,
                "demonstration_indices": [row for row in range(DOMAIN_SIZE) if row != query_index],
                "compatible_function_count": 1,
                "compatible_query_outputs": outputs,
                "identifiability": "function",
            }
    return None


def build_source_row(
    seed: int,
    split: str,
    index: int,
    function: Mapping[str, Any],
    evidence: Mapping[str, Any],
    include_answer: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    table = truth_table(function)
    query_index = int(evidence["query_index"])
    demo_indices = [int(item) for item in evidence["demonstration_indices"]]
    public: Dict[str, Any] = {
        "task_id": f"source_repair|seed{seed}|{split}|{index:04d}",
        "family": "source_induction",
        "split": split,
        "identifiability": evidence["identifiability"],
        "demo_count": len(demo_indices),
        "query": VECTORS[query_index],
        "demonstrations": [
            {"input": VECTORS[row], "output": int(table[row])} for row in demo_indices
        ],
        "canonical_ast_sha256": sha256_text(canonical(function)),
        "truth_table_sha256": sha256_text(table),
        "compatible_function_count": int(evidence["compatible_function_count"]),
        "compatible_query_outputs": list(evidence["compatible_query_outputs"]),
    }
    if include_answer:
        public["query_output"] = int(table[query_index])
    oracle = {
        **public,
        "query_index": query_index,
        "ast": function,
        "truth_table": table,
        "query_output": int(table[query_index]),
        "demonstration_indices": demo_indices,
    }
    return public, oracle


def build_execution_row(
    seed: int,
    split: str,
    index: int,
    function: Mapping[str, Any],
    query_index: int,
    include_answer: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    table = truth_table(function)
    public: Dict[str, Any] = {
        "task_id": f"source_repair|seed{seed}|{split}|{index:04d}",
        "family": "explicit_procedure_execution",
        "split": split,
        "procedure": function,
        "query": VECTORS[query_index],
        "canonical_ast_sha256": sha256_text(canonical(function)),
        "truth_table_sha256": sha256_text(table),
    }
    if include_answer:
        public["query_output"] = int(table[query_index])
    oracle = {**public, "query_index": query_index, "truth_table": table, "query_output": int(table[query_index])}
    return public, oracle


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decode_one_step(logits: Sequence[float], zero_token: int = ZERO_TOKEN, one_token: int = ONE_TOKEN) -> str:
    """Decode exactly one answer token; all other outcomes are abstentions."""
    if not logits:
        return ""
    if any(not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")) for value in logits):
        raise ValueError("non-finite logits")
    zero_score = float(logits[zero_token]) if zero_token < len(logits) else float("-inf")
    one_score = float(logits[one_token]) if one_token < len(logits) else float("-inf")
    if zero_score == one_score:
        return ""
    top = max(range(len(logits)), key=lambda index: (float(logits[index]), -index))
    if top == zero_token:
        return "0"
    if top == one_token:
        return "1"
    return ""


def answer_logit_position(row: Mapping[str, Any]) -> int:
    """Position whose logits predict the byte immediately after ``ANSWER=``."""
    lines = ["BOOLEAN_FUNCTION\n", "DEMONSTRATIONS:\n"]
    for example in row["demonstrations"]:
        values = ",".join(str(int(value)) for value in example["input"])
        lines.append(f"INPUT={values};OUTPUT={int(example['output'])}\n")
    query = ",".join(str(int(value)) for value in row["query"])
    prompt = "".join(lines) + f"QUERY={query};ANSWER="
    # The model receives [BOS] + prompt bytes; its final input position
    # predicts the first answer byte.
    return len(prompt.encode("utf-8"))


def decode_sequence_at_answer(logits: Sequence[Sequence[float]], row: Mapping[str, Any]) -> str:
    position = answer_logit_position(row)
    if position >= len(logits):
        raise ValueError("sequence logits do not contain the fixed answer position")
    return decode_one_step(logits[position])


def decoder_fixture_tests() -> Dict[str, Any]:
    vocab = max(ONE_TOKEN + 1, 300)
    zero = [0.0] * vocab
    zero[ZERO_TOKEN] = 3.0
    one = [0.0] * vocab
    one[ONE_TOKEN] = 3.0
    other = [0.0] * vocab
    other[RAW_BYTE_START + ord("x")] = 3.0
    tie = [0.0] * vocab
    tie[ZERO_TOKEN] = 3.0
    tie[ONE_TOKEN] = 3.0
    row = {
        "demonstrations": [{"input": [0, 0, 0, 0], "output": 0}],
        "query": [1, 0, 1, 0],
    }
    sequence = [[0.0] * vocab for _ in range(answer_logit_position(row) + 1)]
    sequence[answer_logit_position(row)][ZERO_TOKEN] = 3.0
    results = {
        "zero": decode_one_step(zero) == "0",
        "one": decode_one_step(one) == "1",
        "non_answer_abstains": decode_one_step(other) == "",
        "ties_abstain": decode_one_step(tie) == "",
        "fixed_answer_position": decode_sequence_at_answer(sequence, row) == "0",
    }
    try:
        bad = [0.0] * vocab
        bad[ZERO_TOKEN] = float("nan")
        decode_one_step(bad)
    except ValueError:
        results["nonfinite_rejected"] = True
    else:
        results["nonfinite_rejected"] = False
    return {"passed": all(results.values()), "checks": results}


def find_ambiguous_fixture(functions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    for target_index, function in enumerate(functions):
        table = truth_table(function)
        for query_index in range(DOMAIN_SIZE):
            rows = [row for row in range(DOMAIN_SIZE) if row != query_index]
            for demo_indices in itertools.combinations(rows, 8):
                observations = {row: int(table[row]) for row in demo_indices}
                compatible = compatible_indices(functions, observations)
                outputs = sorted({int(truth_table(functions[index])[query_index]) for index in compatible})
                if len(outputs) > 1:
                    return {
                        "target_function_index": target_index,
                        "query_index": query_index,
                        "demonstration_indices": list(demo_indices),
                        "compatible_function_count": len(compatible),
                        "compatible_query_outputs": outputs,
                        "expected_ambiguous": True,
                    }
    raise RuntimeError("could not find an ambiguity fixture")


def generate_tasks(functions: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    output = DEFAULT_OUTPUT.resolve()
    oracle_root = DEFAULT_ORACLE.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    if oracle_root.exists() and any(oracle_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty oracle root: {oracle_root}")
    output.mkdir(parents=True, exist_ok=True)
    oracle_root.mkdir(parents=True, exist_ok=True)

    used_truth: set[str] = set()
    files: Dict[str, str] = {}
    oracle_files: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    identifiability: Dict[str, int] = {"query": 0, "function": 0}

    for seed in SEEDS:
        rng = random.Random(seed * 1000003 + 20261031)
        seed_public = output / f"seed_{seed}"
        seed_oracle = oracle_root / f"seed_{seed}"
        seed_candidates = list(range(len(functions)))
        rng.shuffle(seed_candidates)

        def take_source(split: str, count: int, mode: str, include_answer: bool) -> None:
            nonlocal seed_candidates
            rows: List[Dict[str, Any]] = []
            oracles: List[Dict[str, Any]] = []
            for index in range(count):
                chosen = None
                for candidate_index in seed_candidates:
                    function = functions[candidate_index]
                    table_hash = sha256_text(truth_table(function))
                    if table_hash in used_truth:
                        continue
                    evidence = (
                        choose_function_evidence(functions, candidate_index, rng)
                        if mode == "function"
                        else choose_query_evidence(functions, candidate_index, rng)
                    )
                    if evidence is not None:
                        chosen = (candidate_index, function, evidence, table_hash)
                        break
                if chosen is None:
                    raise RuntimeError(f"not enough {mode}-identifiable source functions for {split}")
                candidate_index, function, evidence, table_hash = chosen
                seed_candidates.remove(candidate_index)
                used_truth.add(table_hash)
                public, oracle = build_source_row(seed, split, index, function, evidence, include_answer)
                rows.append(public)
                oracles.append(oracle)
                identifiability[mode] += 1
            public_path = seed_public / f"{split}.jsonl"
            oracle_path = seed_oracle / f"{split}.jsonl"
            write_jsonl(public_path, rows)
            write_jsonl(oracle_path, oracles)
            files[str(public_path.relative_to(output))] = sha256_file(public_path)
            oracle_files[f"seed_{seed}/{split}.jsonl"] = sha256_file(oracle_path)
            counts[f"seed_{seed}/{split}"] = len(rows)

        take_source("source_train", SOURCE_COUNTS["source_train"], "query", True)
        take_source("source_validation", SOURCE_COUNTS["source_validation"], "query", False)
        take_source("source_function_validation", SOURCE_COUNTS["source_function_validation"], "function", False)
        take_source("source_test", SOURCE_COUNTS["source_test"], "query", False)

        def take_execution(split: str, count: int, include_answer: bool) -> None:
            rows: List[Dict[str, Any]] = []
            oracles: List[Dict[str, Any]] = []
            for index in range(count):
                chosen = None
                for candidate_index in seed_candidates:
                    function = functions[candidate_index]
                    table_hash = sha256_text(truth_table(function))
                    if table_hash in used_truth:
                        continue
                    chosen = (candidate_index, function, table_hash)
                    break
                if chosen is None:
                    raise RuntimeError(f"not enough disjoint functions for {split}")
                candidate_index, function, table_hash = chosen
                seed_candidates.remove(candidate_index)
                used_truth.add(table_hash)
                query_index = rng.randrange(DOMAIN_SIZE)
                public, oracle = build_execution_row(seed, split, index, function, query_index, include_answer)
                rows.append(public)
                oracles.append(oracle)
            public_path = seed_public / f"{split}.jsonl"
            oracle_path = seed_oracle / f"{split}.jsonl"
            write_jsonl(public_path, rows)
            write_jsonl(oracle_path, oracles)
            files[str(public_path.relative_to(output))] = sha256_file(public_path)
            oracle_files[f"seed_{seed}/{split}.jsonl"] = sha256_file(oracle_path)
            counts[f"seed_{seed}/{split}"] = len(rows)

        take_execution("execution_train", EXECUTION_COUNTS["execution_train"], True)
        take_execution("execution_validation", EXECUTION_COUNTS["execution_validation"], False)
        take_execution("execution_test", EXECUTION_COUNTS["execution_test"], False)

    manifest = {
        "schema_version": SCHEMA,
        "classification": "cpu_curriculum_and_evaluator_repair_only",
        "scientific_authorization": False,
        "training_started": False,
        "target_data_loaded": False,
        "sealed_target_answers_loaded": False,
        "seeds": list(SEEDS),
        "grammar": {"operators": list(OPS), "max_depth": 2, "distinct_truth_tables": len(functions)},
        "counts": counts,
        "identifiability_counts": identifiability,
        "public_files": files,
        "source_oracle_files_outside_git": oracle_files,
        "oracle_root": "<private-source-oracle-root>",
        "split_rules": {
            "source_train": "query-identifiable, 8 demonstrations, answer labels public for future training",
            "source_validation": "query-identifiable, 8 demonstrations, answer labels private",
            "source_function_validation": "function-identifiable, 15 demonstrations, answer labels private",
            "source_test": "query-identifiable, 8 demonstrations, answer labels private",
            "explicit_execution": "canonical AST supplied; validation/test answer labels private",
        },
        "global_function_hash_separation": True,
        "global_structure_hash_separation": True,
        "decoder": {
            "mode": "one_step_constrained",
            "allowed_tokens": [ZERO_TOKEN, ONE_TOKEN],
            "non_answer_behavior": "abstain",
            "free_running_generation_not_used_for_gate": True,
        },
    }
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes_path = output / "HASHES.json"
    hashes_path.write_text(
        json.dumps(
            {"manifest_sha256": sha256_file(manifest_path), "public_files": files},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, {"output": str(output), "oracle_root": str(oracle_root)}, {"ambiguous_fixture": find_ambiguous_fixture(functions)}


def validate_manifest(functions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output = DEFAULT_OUTPUT.resolve()
    oracle_root = DEFAULT_ORACLE.resolve()
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    errors: List[str] = []
    public_hashes: Dict[str, str] = {}
    seen_truth: set[str] = set()
    seen_structure: set[str] = set()
    rows_checked = 0

    for relative in manifest["public_files"]:
        path = output / relative
        if not path.exists():
            errors.append(f"missing public file: {relative}")
        else:
            public_hashes[relative] = sha256_file(path)
    hashes_path = output / "HASHES.json"
    if not hashes_path.exists():
        errors.append("missing HASHES.json")
    else:
        hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        if hashes.get("manifest_sha256") != sha256_file(output / "MANIFEST.json"):
            errors.append("manifest hash mismatch")
        if hashes.get("public_files") != manifest.get("public_files"):
            errors.append("public file hash map mismatch")

    for seed in SEEDS:
        seed_public = output / f"seed_{seed}"
        seed_oracle = oracle_root / f"seed_{seed}"
        for split in list(SOURCE_COUNTS) + list(EXECUTION_COUNTS):
            public_path = seed_public / f"{split}.jsonl"
            oracle_path = seed_oracle / f"{split}.jsonl"
            if not public_path.exists() or not oracle_path.exists():
                errors.append(f"missing split files: seed {seed} {split}")
                continue
            public_rows = read_jsonl(public_path)
            oracle_rows = read_jsonl(oracle_path)
            if len(public_rows) != len(oracle_rows):
                errors.append(f"row-count mismatch: seed {seed} {split}")
                continue
            for public, oracle in zip(public_rows, oracle_rows):
                rows_checked += 1
                if public["task_id"] != oracle["task_id"]:
                    errors.append(f"task id mismatch: {public['task_id']}")
                if public.get("query_output") is not None and split not in ("source_train", "execution_train"):
                    errors.append(f"validation/test answer leaked publicly: {public['task_id']}")
                truth_hash = sha256_text(oracle["truth_table"])
                structure_hash = sha256_text(canonical(oracle["ast"] if "ast" in oracle else oracle["procedure"]))
                if truth_hash != oracle["truth_table_sha256"] or structure_hash != oracle["canonical_ast_sha256"]:
                    errors.append(f"oracle hash mismatch: {public['task_id']}")
                if truth_hash in seen_truth or structure_hash in seen_structure:
                    errors.append(f"cross-split function/structure overlap: {public['task_id']}")
                seen_truth.add(truth_hash)
                seen_structure.add(structure_hash)

                if public["family"] == "source_induction":
                    query_index = int(oracle["query_index"])
                    demo_indices = list(oracle["demonstration_indices"])
                    if query_index in demo_indices or len(set(demo_indices)) != len(demo_indices):
                        errors.append(f"query/demo collision: {public['task_id']}")
                    observations = {
                        int(row): int(oracle["truth_table"][row]) for row in demo_indices
                    }
                    compatible = compatible_indices(functions, observations)
                    outputs = sorted({int(truth_table(functions[index])[query_index]) for index in compatible})
                    if len(compatible) != int(public["compatible_function_count"]):
                        errors.append(f"compatible-count mismatch: {public['task_id']}")
                    if outputs != list(public["compatible_query_outputs"]):
                        errors.append(f"compatible-output mismatch: {public['task_id']}")
                    if public["identifiability"] == "query" and len(outputs) != 1:
                        errors.append(f"query-identifiability failure: {public['task_id']}")
                    if public["identifiability"] == "function" and len(compatible) != 1:
                        errors.append(f"function-identifiability failure: {public['task_id']}")
                    if int(oracle["query_output"]) != int(oracle["truth_table"][query_index]):
                        errors.append(f"query oracle mismatch: {public['task_id']}")
                else:
                    query_index = int(oracle["query_index"])
                    if evaluate(oracle["procedure"], VECTORS[query_index]) != int(oracle["query_output"]):
                        errors.append(f"explicit procedure oracle mismatch: {public['task_id']}")

    ambiguity_fixture = find_ambiguous_fixture(functions)
    decoder = decoder_fixture_tests()
    if not decoder["passed"]:
        errors.append("one-step decoder fixture failure")
    if len(ambiguity_fixture["compatible_query_outputs"]) <= 1:
        errors.append("ambiguity fixture was not ambiguous")

    report = {
        "schema_version": SCHEMA,
        "classification": "cpu_curriculum_and_evaluator_repair_validation",
        "status": "PASS" if not errors else "FAIL",
        "scientific_authorization": False,
        "training_started": False,
        "target_data_loaded": False,
        "sealed_target_answers_loaded": False,
        "rows_checked": rows_checked,
        "distinct_truth_tables_checked": len(seen_truth),
        "distinct_structures_checked": len(seen_structure),
        "decoder_tests": decoder,
        "ambiguity_fixture": ambiguity_fixture,
        "errors": errors,
    }
    report_path = output / "REPAIR_VALIDATION_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        "# Crucible source curriculum/evaluator repair",
        "",
        f"Status: **{report['status']}** — CPU-only validation; no training or target evaluation.",
        "",
        f"Rows checked: {rows_checked}",
        f"Distinct truth tables: {len(seen_truth)}",
        f"Distinct canonical structures: {len(seen_structure)}",
        "",
        "## Controlled decoder",
        "",
        "The repaired evaluator predicts exactly one `0`/`1` token from the fixed logit position immediately before the answer byte. Non-answer tokens and ties abstain; free-running generation is not used for the source gate.",
        "",
        "```json",
        json.dumps(decoder, indent=2, sort_keys=True),
        "```",
        "",
        "## Identifiability",
        "",
        "Query-identifiable tasks require every grammar-consistent function to agree on the query. Function-identifiable tasks require exactly one grammar function to remain compatible. Ambiguous fixtures are detected and retained as an explicit negative test.",
        "",
        "## Boundary",
        "",
        "This artifact repairs measurement and curriculum construction. It does not establish neural competence, productive retention or target transfer. The existing C0–C3 experiment remains blocked.",
        "",
        "## Errors",
        "",
        "```text",
        "\n".join(errors) if errors else "none",
        "```",
    ]
    (output / "REPAIR_VALIDATION_REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "validate", "build-and-validate"), default="build-and-validate")
    args = parser.parse_args()
    functions = enumerate_expression_pool()
    if len(functions) != 429:
        raise RuntimeError(f"expected 429 grammar truth tables, got {len(functions)}")
    if args.mode in ("build", "build-and-validate"):
        manifest, locations, fixture = generate_tasks(functions)
        build_report = {
            "schema_version": SCHEMA,
            "status": "PASS",
            "classification": "cpu_curriculum_and_evaluator_repair_only",
            "scientific_authorization": False,
            "training_started": False,
            "target_data_loaded": False,
            "sealed_target_answers_loaded": False,
            "grammar_truth_tables": len(functions),
            "manifest": manifest,
            "locations": locations,
            "ambiguous_fixture": fixture["ambiguous_fixture"],
        }
        (DEFAULT_OUTPUT / "REPAIR_BUILD_REPORT.json").write_text(json.dumps(build_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.mode in ("validate", "build-and-validate"):
        report = validate_manifest(functions)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
