#!/usr/bin/env python3
"""Generate and audit the CPU-only Track B neural-proposer preflight package.

This freezes a small grammar and task manifest. It does not train a proposer,
run a verifier, select a checkpoint, or access a sealed test set.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_track_b_neural_proposer_preflight_20261028"
SEED = 20261028
MAX_DEPTH = 3
MAX_NODES = 7
DEMONSTRATION_COUNT = 8
IDENTIFIABLE_VERIFIER_BUDGET = 16
UNRESOLVABLE_VERIFIER_BUDGET = 1
OPERATORS = ("add_mod10", "sub_mod10", "mul_mod10", "xor_bits_mod10", "min", "max")
COMMUTATIVE = {"add_mod10", "mul_mod10", "xor_bits_mod10", "min", "max"}
DOMAIN = tuple((a, b) for a in range(10) for b in range(10))

Expr = Tuple[Any, ...]


def canonical(expr: Expr) -> Expr:
    kind = expr[0]
    if kind in {"var", "const"}:
        return expr
    _, op, left, right = expr
    children = [canonical(left), canonical(right)]
    if op in COMMUTATIVE:
        children.sort(key=lambda item: json.dumps(to_json(item), sort_keys=True, separators=(",", ":")))
    return ("op", op, children[0], children[1])


def to_json(expr: Expr) -> Dict[str, Any]:
    if expr[0] == "var":
        return {"op": "var", "name": expr[1]}
    if expr[0] == "const":
        return {"op": "const", "value": int(expr[1])}
    return {"op": str(expr[1]), "args": [to_json(expr[2]), to_json(expr[3])]}


def expr_key(expr: Expr) -> str:
    return json.dumps(to_json(canonical(expr)), sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    material = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def ast_hash(expr: Expr) -> str:
    return digest(to_json(canonical(expr)))


def evaluate(expr: Expr, a: int, b: int) -> int:
    if expr[0] == "var":
        return a if expr[1] == "a" else b
    if expr[0] == "const":
        return int(expr[1])
    op = expr[1]
    left = evaluate(expr[2], a, b)
    right = evaluate(expr[3], a, b)
    if op == "add_mod10":
        return (left + right) % 10
    if op == "sub_mod10":
        return (left - right) % 10
    if op == "mul_mod10":
        return (left * right) % 10
    if op == "xor_bits_mod10":
        return (left ^ right) % 10
    if op == "min":
        return min(left, right)
    if op == "max":
        return max(left, right)
    raise ValueError(f"unknown operator: {op}")


def truth_table(expr: Expr) -> Tuple[int, ...]:
    return tuple(evaluate(expr, a, b) for a, b in DOMAIN)


def truth_hash(expr: Expr) -> str:
    return digest(list(truth_table(expr)))


def depth(expr: Expr) -> int:
    return 0 if expr[0] != "op" else 1 + max(depth(expr[2]), depth(expr[3]))


def nodes(expr: Expr) -> int:
    return 1 if expr[0] != "op" else 1 + nodes(expr[2]) + nodes(expr[3])


def variables(expr: Expr) -> set[str]:
    if expr[0] == "var":
        return {str(expr[1])}
    if expr[0] == "const":
        return set()
    return variables(expr[2]) | variables(expr[3])


def random_leaf(rng: random.Random) -> Expr:
    if rng.random() < 0.65:
        return ("var", rng.choice(("a", "b")))
    return ("const", rng.randrange(10))


def random_expr(rng: random.Random, remaining_depth: int) -> Expr:
    if remaining_depth <= 0 or rng.random() < 0.28:
        return random_leaf(rng)
    op = rng.choice(OPERATORS)
    left = random_expr(rng, remaining_depth - 1)
    right = random_expr(rng, remaining_depth - 1)
    return canonical(("op", op, left, right))


def sample_expressions(
    seed: int,
    count: int,
    min_depth: int,
    max_depth: int,
    excluded_truth_hashes: Iterable[str] = (),
) -> List[Expr]:
    rng = random.Random(seed)
    excluded = set(excluded_truth_hashes)
    structural: set[str] = set()
    functions: set[str] = set(excluded)
    result: List[Expr] = []
    attempts = 0
    while len(result) < count and attempts < 500_000:
        attempts += 1
        expr = canonical(random_expr(rng, max_depth))
        if not (min_depth <= depth(expr) <= max_depth):
            continue
        if nodes(expr) > MAX_NODES or variables(expr) != {"a", "b"}:
            continue
        ah = ast_hash(expr)
        th = truth_hash(expr)
        if ah in structural or th in functions:
            continue
        structural.add(ah)
        functions.add(th)
        result.append(expr)
    if len(result) != count:
        raise RuntimeError(f"could not sample {count} expressions after {attempts} attempts")
    return result


def fixed_rules() -> Dict[str, Expr]:
    a = ("var", "a")
    b = ("var", "b")
    return {
        "ADD_MOD10": canonical(("op", "add_mod10", a, b)),
        "SUB_MOD10": canonical(("op", "sub_mod10", a, b)),
        "MUL_MOD10": canonical(("op", "mul_mod10", a, b)),
        "FIRST": a,
        "SECOND": b,
        "XOR_MOD10": canonical(("op", "xor_bits_mod10", a, b)),
        "MAX": canonical(("op", "max", a, b)),
        "MIN": canonical(("op", "min", a, b)),
    }


def choose_discriminating_pairs(
    target: Expr,
    references: Sequence[Expr],
    budget: int,
    allowed_pairs: Sequence[Tuple[int, int]] | None = None,
) -> Tuple[List[Tuple[int, int]], bool]:
    target_table = truth_table(target)
    tables: Dict[str, Tuple[int, ...]] = {truth_hash(expr): truth_table(expr) for expr in references}
    tables[truth_hash(target)] = target_table
    candidates = set(tables)
    chosen: List[Tuple[int, int]] = []
    available = list(DOMAIN if allowed_pairs is None else allowed_pairs)
    for _ in range(budget):
        if len(candidates) <= 1:
            break
        best_pair = None
        best_remaining = None
        for pair in available:
            index = DOMAIN.index(pair)
            remaining = {th for th in candidates if tables[th][index] == target_table[index]}
            if best_remaining is None or len(remaining) < len(best_remaining):
                best_pair = pair
                best_remaining = remaining
        if best_pair is None or best_remaining is None:
            break
        chosen.append(best_pair)
        available.remove(best_pair)
        candidates = best_remaining
    return chosen, len(candidates) == 1


def choose_task_pairs(target: Expr, references: Sequence[Expr], seed: int, force_unresolvable: bool) -> Dict[str, Any]:
    rng = random.Random(seed)
    shuffled = list(DOMAIN)
    rng.shuffle(shuffled)
    demos = shuffled[:DEMONSTRATION_COUNT]
    remaining = shuffled[DEMONSTRATION_COUNT:]
    verifier, identifiable = choose_discriminating_pairs(
        target, references, IDENTIFIABLE_VERIFIER_BUDGET, allowed_pairs=remaining
    )
    if force_unresolvable or not identifiable:
        target_table = truth_table(target)
        reference_tables = {truth_hash(expr): truth_table(expr) for expr in references}
        reference_tables[truth_hash(target)] = target_table
        verifier = []
        for pair in remaining:
            index = DOMAIN.index(pair)
            surviving = {truth: table for truth, table in reference_tables.items() if table[index] == target_table[index]}
            if len(surviving) > 1:
                verifier = [pair]
                break
        identifiable = False
    verifier_set = set(verifier)
    queries = [pair for pair in remaining if pair not in verifier_set]
    return {"demonstrations": demos, "verifier_pairs": verifier, "query_pairs": queries, "identifiable_under_budget": identifiable}


def make_task(task_id: str, split: str, stratum: str, target: Expr, references: Sequence[Expr], seed: int, force_unresolvable: bool = False) -> Dict[str, Any]:
    pairs = choose_task_pairs(target, references, seed, force_unresolvable)
    table = truth_table(target)
    pair_index = {pair: index for index, pair in enumerate(DOMAIN)}
    demonstrations = [{"a": a, "b": b, "output": table[pair_index[(a, b)]]} for a, b in pairs["demonstrations"]]
    verifier = [{"a": a, "b": b, "output": table[pair_index[(a, b)]]} for a, b in pairs["verifier_pairs"]]
    queries = [{"a": a, "b": b, "output": table[pair_index[(a, b)]]} for a, b in pairs["query_pairs"]]
    return {
        "task_id": task_id,
        "split": split,
        "stratum": stratum,
        "demonstrations": demonstrations,
        "verifier_examples": verifier,
        "query_examples": queries,
        "target_ast": to_json(target),
        "target_ast_hash": ast_hash(target),
        "target_truth_table_hash": truth_hash(target),
        "identifiable_under_budget": pairs["identifiable_under_budget"],
        "structurally_novel": False,
        "functionally_novel": False,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generated_names = {
        "GRAMMAR_SPEC.json", "TASK_MANIFEST.json", "PROPOSER_INPUTS.jsonl", "TRAIN_LABELS.jsonl",
        "EVAL_ORACLE.jsonl", "LEAKAGE_AUDIT.json", "INSTRUMENTATION_PLAN.json", "INSTRUMENTATION_PLAN.md",
        "README.md", "HASHES.json", "PREFLIGHT_RESULTS.json",
    }
    existing = list(OUTPUT.iterdir())
    unexpected = [path.name for path in existing if path.name not in generated_names or not path.is_file()]
    if unexpected:
        raise SystemExit(f"refusing to overwrite unexpected preflight paths: {unexpected}")
    for path in existing:
        path.unlink()

    rules = fixed_rules()
    train_exprs = sample_expressions(SEED + 1, 24, 1, 2)
    train_truths = {truth_hash(expr) for expr in train_exprs}
    eval_exprs = sample_expressions(SEED + 2, 24, 3, 3, train_truths | {truth_hash(expr) for expr in rules.values()})
    val_exprs = eval_exprs[:12]
    test_exprs = eval_exprs[12:]
    fixed_refs = list(rules.values())
    compositional_refs = train_exprs + val_exprs + test_exprs

    tasks: List[Dict[str, Any]] = []
    for index, (name, expr) in enumerate(rules.items()):
        task = make_task(f"fixed_train_{index:02d}", "train", "fixed_family_integration", expr, fixed_refs, SEED + 100 + index)
        task["rule_name"] = name
        tasks.append(task)
    for index, expr in enumerate(train_exprs):
        tasks.append(make_task(f"compositional_train_{index:02d}", "train", "compositional_training", expr, compositional_refs, SEED + 200 + index))
    for index, expr in enumerate(val_exprs):
        tasks.append(make_task(f"compositional_validation_{index:02d}", "validation", "structural_and_functional_novelty", expr, compositional_refs, SEED + 300 + index, force_unresolvable=index >= 10))
    for index, expr in enumerate(test_exprs):
        tasks.append(make_task(f"compositional_test_{index:02d}", "test", "structural_and_functional_novelty", expr, compositional_refs, SEED + 400 + index, force_unresolvable=index >= 10))

    for task in tasks:
        if task["split"] in {"validation", "test"}:
            task["structurally_novel"] = task["target_ast_hash"] not in {x["target_ast_hash"] for x in tasks if x["split"] == "train"}
            task["functionally_novel"] = task["target_truth_table_hash"] not in {x["target_truth_table_hash"] for x in tasks if x["split"] == "train"}

    train_tasks = [task for task in tasks if task["split"] == "train"]
    included_train = [task for task in train_tasks if task["identifiable_under_budget"]]
    excluded_train = [task for task in train_tasks if not task["identifiable_under_budget"]]
    eval_tasks = [task for task in tasks if task["split"] in {"validation", "test"}]

    grammar_spec = {
        "schema_version": "digit_expr_mod10_v1",
        "max_depth": MAX_DEPTH,
        "max_nodes": MAX_NODES,
        "variables": ["a", "b"],
        "constants": list(range(10)),
        "operators": list(OPERATORS),
        "commutative_operators": sorted(COMMUTATIVE),
        "xor_semantics": "integer_bitwise_xor_then_mod10",
        "domain": [[a, b] for a, b in DOMAIN],
        "test_vectors": {
            "add_mod10": [{"a": 7, "b": 8, "expected": 5}],
            "sub_mod10": [{"a": 2, "b": 7, "expected": 5}],
            "mul_mod10": [{"a": 7, "b": 8, "expected": 6}],
            "xor_bits_mod10": [{"a": 7, "b": 8, "expected": 5}, {"a": 9, "b": 6, "expected": 5}],
            "min": [{"a": 7, "b": 8, "expected": 7}],
            "max": [{"a": 7, "b": 8, "expected": 8}],
        },
    }
    write_json(OUTPUT / "GRAMMAR_SPEC.json", grammar_spec)

    proposer_inputs = []
    labels = []
    oracle = []
    manifest_tasks = []
    for task in tasks:
        proposer_inputs.append({"task_id": task["task_id"], "split": task["split"], "stratum": task["stratum"], "demonstrations": task["demonstrations"]})
        manifest_tasks.append({
            "task_id": task["task_id"], "split": task["split"], "stratum": task["stratum"],
            "target_ast_hash": task["target_ast_hash"], "target_truth_table_hash": task["target_truth_table_hash"],
            "identifiable_under_budget": task["identifiable_under_budget"],
            "structurally_novel": task["structurally_novel"], "functionally_novel": task["functionally_novel"],
            "demonstration_count": len(task["demonstrations"]), "verifier_count": len(task["verifier_examples"]),
            "query_count": len(task["query_examples"]), "training_included": task in included_train,
            "demonstration_pairs": [[row["a"], row["b"]] for row in task["demonstrations"]],
            "verifier_pairs": [[row["a"], row["b"]] for row in task["verifier_examples"]],
            "query_pairs": [[row["a"], row["b"]] for row in task["query_examples"]],
        })
        if task in included_train:
            labels.append({"task_id": task["task_id"], "target_ast": task["target_ast"], "target_ast_hash": task["target_ast_hash"], "target_truth_table_hash": task["target_truth_table_hash"]})
        if task["split"] in {"validation", "test"}:
            oracle.append({
                "task_id": task["task_id"], "split": task["split"], "stratum": task["stratum"],
                "verifier_examples": task["verifier_examples"], "query_examples": task["query_examples"],
                "target_ast": task["target_ast"], "target_ast_hash": task["target_ast_hash"],
                "target_truth_table_hash": task["target_truth_table_hash"],
                "identifiable_under_budget": task["identifiable_under_budget"],
                "structurally_novel": task["structurally_novel"], "functionally_novel": task["functionally_novel"],
            })

    write_jsonl(OUTPUT / "PROPOSER_INPUTS.jsonl", proposer_inputs)
    write_jsonl(OUTPUT / "TRAIN_LABELS.jsonl", labels)
    write_jsonl(OUTPUT / "EVAL_ORACLE.jsonl", oracle)

    train_hashes = {task["target_truth_table_hash"] for task in train_tasks}
    eval_novel = [task for task in eval_tasks if task["functionally_novel"] and task["structurally_novel"]]
    leakage = {
        "status": "pass",
        "source_script": "tools/prepare_crucible_track_b_neural_proposer_preflight_20261028.py",
        "proposer_inputs_have_target_ast": any("target_ast" in row or "target_ast_hash" in row for row in proposer_inputs),
        "proposer_inputs_have_target_truth_hash": any("target_truth_table_hash" in row for row in proposer_inputs),
        "train_eval_function_hash_overlap": sorted(train_hashes & {task["target_truth_table_hash"] for task in eval_tasks}),
        "train_eval_ast_hash_overlap": sorted({task["target_ast_hash"] for task in train_tasks} & {task["target_ast_hash"] for task in eval_tasks}),
        "training_task_count": len(train_tasks),
        "training_included_count": len(included_train),
        "training_excluded_count": len(excluded_train),
        "training_excluded_proportion": len(excluded_train) / len(train_tasks),
        "validation_test_task_count": len(eval_tasks),
        "held_out_novel_task_count": len(eval_novel),
        "unresolvable_eval_count": sum(not task["identifiable_under_budget"] for task in eval_tasks),
        "persistent_state_labels_in_proposer_inputs": False,
        "sealed_test_accessed": False,
    }
    if leakage["proposer_inputs_have_target_ast"] or leakage["proposer_inputs_have_target_truth_hash"]:
        leakage["status"] = "fail"
    if leakage["train_eval_function_hash_overlap"] or leakage["train_eval_ast_hash_overlap"]:
        leakage["status"] = "fail"
    write_json(OUTPUT / "LEAKAGE_AUDIT.json", leakage)

    instrumentation = {
        "status": "plan_complete_implementation_pending",
        "per_task_counters": [
            "candidate_source", "candidate_outputs", "unique_canonical_ast_outputs", "candidate_evaluations",
            "verifier_calls", "oracle_calls", "proposer_forward_passes", "decoder_tokens",
            "cpu_seconds", "gpu_seconds", "wall_clock_seconds", "peak_memory_bytes",
        ],
        "lifecycle_events": ["proposal", "verification", "promotion", "reset", "execution", "contradiction", "invalidation", "relearning"],
        "aggregation": ["per_task", "per_split", "per_stratum", "per_seed", "arm_pooled"],
        "hardware": "record device model, dtype, batch size, process count and software revision",
        "checkpoint_rule": "preflight cannot select a scientific checkpoint or inspect held-out oracle outputs",
    }
    write_json(OUTPUT / "INSTRUMENTATION_PLAN.json", instrumentation)
    (OUTPUT / "INSTRUMENTATION_PLAN.md").write_text(
        "# Track B proposer instrumentation plan\n\n"
        "Status: plan complete; implementation pending.\n\n"
        "Every task must record candidate source/counts, verifier and oracle calls, proposer passes, decoder tokens, CPU/GPU/wall time and peak memory. Lifecycle events are recorded separately and aggregated per task, split, stratum, seed and arm. The 50-update compatibility preflight may not select a scientific checkpoint or consume held-out oracle outputs.\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "crucible_track_b_preflight_manifest_v1",
        "seed": SEED,
        "grammar": grammar_spec,
        "task_counts": {
            "all": len(tasks), "train": len(train_tasks), "train_included": len(included_train),
            "train_excluded": len(excluded_train), "validation": sum(task["split"] == "validation" for task in tasks),
            "test": sum(task["split"] == "test" for task in tasks), "held_out_novel": len(eval_novel),
            "eval_unresolvable": sum(not task["identifiable_under_budget"] for task in eval_tasks),
        },
        "tasks": manifest_tasks,
        "files": {
            "proposer_inputs": "PROPOSER_INPUTS.jsonl",
            "train_labels": "TRAIN_LABELS.jsonl",
            "eval_oracle": "EVAL_ORACLE.jsonl",
            "grammar": "GRAMMAR_SPEC.json",
            "leakage_audit": "LEAKAGE_AUDIT.json",
            "instrumentation": "INSTRUMENTATION_PLAN.json",
        },
    }
    write_json(OUTPUT / "TASK_MANIFEST.json", manifest)
    (OUTPUT / "README.md").write_text(
        "# Track B neural proposer preflight package\n\n"
        "CPU-generated, design/preflight artifacts only. No proposer was trained; no verifier run selected a checkpoint; no sealed test set was accessed. `PROPOSER_INPUTS.jsonl` contains no target AST or truth-table labels. `TRAIN_LABELS.jsonl` is for future training-task supervision only. `EVAL_ORACLE.jsonl` is evaluator-only and must not be supplied to proposer training.\n",
        encoding="utf-8",
    )

    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    write_json(OUTPUT / "PREFLIGHT_RESULTS.json", {
        "status": "completed_cpu_preparation",
        "training_performed": False,
        "scientific_checkpoint_selected": False,
        "sealed_test_accessed": False,
        "leakage_audit_status": leakage["status"],
        "task_counts": manifest["task_counts"],
        "source_script_sha256": sha256(ROOT / "tools" / "prepare_crucible_track_b_neural_proposer_preflight_20261028.py"),
        "artifacts_hashed": len(hashes),
    })
    print(json.dumps({"status": "completed_cpu_preparation", "output": str(OUTPUT), "leakage": leakage["status"], "tasks": len(tasks), "training_performed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
