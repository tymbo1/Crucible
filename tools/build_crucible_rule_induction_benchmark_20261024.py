#!/usr/bin/env python3
"""Build and audit the CPU-only Crucible rule-induction benchmark."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_rule_induction_benchmark_20261024_v3"
SCHEMA_VERSION = "crucible_rule_induction_v3"
CONTEXT_LENGTH = 320
ANSWER_MARKER = "\nAnswer:"
SPLIT_SEED = 20261024
DEMO_COUNTS = (1, 2, 4, 8)
INPUT_DOMAIN = tuple((x, z) for x in range(10) for z in range(10))
FAMILY_ORDER = ("affine2", "max_shift", "absdiff_shift", "threshold_select")

Rule = Tuple[str, Tuple[int, ...]]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(record), sort_keys=True) for record in records) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def apply_rule(rule: Rule, x: int, z: int) -> int:
    family, params = rule
    if family == "affine2":
        a, b, c = params
        return (a * x + b * z + c) % 10
    if family == "max_shift":
        (c,) = params
        return (max(x, z) + c) % 10
    if family == "absdiff_shift":
        (c,) = params
        return (abs(x - z) + c) % 10
    if family == "threshold_select":
        threshold, low_offset, high_offset = params
        if x >= threshold:
            return (x + high_offset) % 10
        return (z + low_offset) % 10
    raise ValueError(family)


def raw_rules() -> List[Rule]:
    rules: List[Rule] = []
    for a in range(4):
        for b in range(4):
            for c in range(10):
                if (a, b) != (0, 0):
                    rules.append(("affine2", (a, b, c)))
    for c in range(10):
        rules.append(("max_shift", (c,)))
        rules.append(("absdiff_shift", (c,)))
    for threshold in range(1, 10):
        for low_offset in range(4):
            for high_offset in range(4):
                rules.append(("threshold_select", (threshold, low_offset, high_offset)))
    return rules


def rule_signature(rule: Rule) -> Tuple[int, ...]:
    return tuple(apply_rule(rule, x, z) for x, z in INPUT_DOMAIN)


def rule_id(rule: Rule) -> str:
    family, params = rule
    return hashlib.sha256(canonical_json({"family": family, "params": params}).encode("utf-8")).hexdigest()[:20]


def unique_rules() -> List[Rule]:
    family_rank = {family: index for index, family in enumerate(FAMILY_ORDER)}
    seen: Dict[Tuple[int, ...], Rule] = {}
    for rule in sorted(raw_rules(), key=lambda item: (family_rank[item[0]], item[1])):
        seen.setdefault(rule_signature(rule), rule)
    return sorted(seen.values(), key=lambda item: rule_id(item))


def split_rules(rules: Sequence[Rule]) -> Dict[str, List[Rule]]:
    by_family: Dict[str, List[Rule]] = defaultdict(list)
    for rule in rules:
        by_family[rule[0]].append(rule)
    result: Dict[str, List[Rule]] = {"train": [], "validation": [], "test": []}
    for family in FAMILY_ORDER:
        family_rules = sorted(by_family[family], key=rule_id)
        random.Random(stable_seed(SPLIT_SEED, "family-split", family)).shuffle(family_rules)
        test_count = max(2, round(len(family_rules) * 0.10))
        validation_count = max(2, round(len(family_rules) * 0.20))
        train_count = len(family_rules) - test_count - validation_count
        if train_count < validation_count:
            raise RuntimeError(f"insufficient family training instances: {family}")
        result["train"].extend(family_rules[:train_count])
        result["validation"].extend(family_rules[train_count : train_count + validation_count])
        result["test"].extend(family_rules[train_count + validation_count :])
    for values in result.values():
        values.sort(key=rule_id)
    return result


def prompt_for(demonstrations: Sequence[Mapping[str, int]], query: Mapping[str, int], count: int) -> str:
    lines = ["Infer hidden mod10 rule."]
    for index, example in enumerate(demonstrations[:count], start=1):
        lines.append(f"E{index}:{example['x']},{example['z']}->{example['y']}")
    lines.append(f"Q:{query['x']},{query['z']}->? digit")
    return " ".join(lines)


def consistent_candidates(rules: Sequence[Rule], demonstrations: Sequence[Mapping[str, int]], count: int) -> List[Rule]:
    prefix = demonstrations[:count]
    return [
        rule
        for rule in rules
        if all(apply_rule(rule, int(example["x"]), int(example["z"])) == int(example["y"]) for example in prefix)
    ]


def make_task(rule: Rule, split: str, all_rules: Sequence[Rule]) -> Dict[str, Any]:
    identifier = rule_id(rule)
    rng = random.Random(stable_seed(SPLIT_SEED, "task", identifier))
    pairs = list(INPUT_DOMAIN)
    rng.shuffle(pairs)
    demonstration_pairs = pairs[:8]
    query_pair = pairs[8]
    demonstrations = [
        {"x": x, "z": z, "y": apply_rule(rule, x, z)}
        for x, z in demonstration_pairs
    ]
    query = {"x": query_pair[0], "z": query_pair[1]}
    identifiability: Dict[str, Any] = {}
    prompts: Dict[str, str] = {}
    answers: Dict[str, str] = {}
    for count in DEMO_COUNTS:
        candidates = consistent_candidates(all_rules, demonstrations, count)
        candidate_outputs = sorted({apply_rule(candidate, query_pair[0], query_pair[1]) for candidate in candidates})
        identifiability[str(count)] = {
            "consistent_rule_count": len(candidates),
            "query_output_count": len(candidate_outputs),
            "query_identifiable": len(candidate_outputs) == 1,
            "exact_rule_unique": len(candidates) == 1,
        }
        prompts[str(count)] = prompt_for(demonstrations, query, count)
        answers[str(count)] = str(apply_rule(rule, query_pair[0], query_pair[1]))
    family, params = rule
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": hashlib.sha256(f"{SCHEMA_VERSION}|{split}|{identifier}".encode("utf-8")).hexdigest()[:20],
        "rule_instance_id": identifier,
        "split": split,
        "rule_family": family,
        "rule_parameters": list(params),
        "demonstrations": demonstrations,
        "query": query,
        "prompt_by_demo_count": prompts,
        "answer_by_demo_count": answers,
        "identifiability_by_demo_count": identifiability,
        "input": prompts["8"],
        "target": answers["8"],
        "oracle": {"kind": "finite_rule_induction", "rule_instance_id": identifier},
    }


def audit_tasks(tasks: Sequence[Mapping[str, Any]], all_rules: Sequence[Rule]) -> Dict[str, Any]:
    oracle_errors: List[str] = []
    duplicate_pairs: List[str] = []
    for task in tasks:
        rule = next(rule for rule in all_rules if rule_id(rule) == task["rule_instance_id"])
        seen_pairs = set()
        for example in task["demonstrations"]:
            pair = (int(example["x"]), int(example["z"]))
            if pair in seen_pairs:
                duplicate_pairs.append(task["task_id"])
            seen_pairs.add(pair)
            expected = apply_rule(rule, pair[0], pair[1])
            if int(example["y"]) != expected:
                oracle_errors.append(task["task_id"])
        query_pair = (int(task["query"]["x"]), int(task["query"]["z"]))
        if query_pair in seen_pairs:
            duplicate_pairs.append(task["task_id"])
        expected_query = apply_rule(rule, query_pair[0], query_pair[1])
        for count in DEMO_COUNTS:
            if task["answer_by_demo_count"][str(count)] != str(expected_query):
                oracle_errors.append(task["task_id"])
            if not task["prompt_by_demo_count"][str(count)].endswith("digit"):
                oracle_errors.append(task["task_id"])
    by_split: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_split[str(task["split"])].append(task)
    overlaps: Dict[str, List[str]] = {}
    split_names = ("train", "validation", "test")
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            left_ids = {str(task["rule_instance_id"]) for task in by_split[left]}
            right_ids = {str(task["rule_instance_id"]) for task in by_split[right]}
            overlaps[f"{left}_{right}"] = sorted(left_ids & right_ids)
    identifiability: Dict[str, Dict[str, Any]] = {}
    for split, values in by_split.items():
        identifiability[split] = {}
        for count in DEMO_COUNTS:
            rows = [task["identifiability_by_demo_count"][str(count)] for task in values]
            identifiable = sum(bool(row["query_identifiable"]) for row in rows)
            unique = sum(bool(row["exact_rule_unique"]) for row in rows)
            identifiability[split][str(count)] = {
                "tasks": len(rows),
                "query_identifiable": identifiable,
                "query_identifiable_rate": identifiable / len(rows) if rows else None,
                "exact_rule_unique": unique,
                "exact_rule_unique_rate": unique / len(rows) if rows else None,
                "mean_consistent_rule_count": sum(int(row["consistent_rule_count"]) for row in rows) / len(rows) if rows else None,
            }
    family_counts = {
        split: dict(sorted(Counter(str(task["rule_family"]) for task in values).items()))
        for split, values in by_split.items()
    }
    prompt_lengths: Dict[str, int] = {}
    context_errors: List[str] = []
    for task in tasks:
        for count in DEMO_COUNTS:
            prompt = str(task["prompt_by_demo_count"][str(count)])
            full_length = 1 + len((prompt + ANSWER_MARKER).encode("utf-8")) + 1 + 1
            prompt_lengths[str(count)] = max(prompt_lengths.get(str(count), 0), full_length)
            if full_length > CONTEXT_LENGTH:
                context_errors.append(f"{task['task_id']}:{count}:{full_length}")
    return {
        "tasks": len(tasks),
        "oracle_errors": sorted(set(oracle_errors)),
        "duplicate_or_query_pair_errors": sorted(set(duplicate_pairs)),
        "rule_instance_overlaps": overlaps,
        "family_counts": family_counts,
        "max_serialized_context_length": prompt_lengths,
        "context_length_errors": sorted(context_errors),
        "identifiability": identifiability,
        "all_checks_pass": not oracle_errors and not duplicate_pairs and not any(overlaps.values()) and not context_errors,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rules = unique_rules()
    splits = split_rules(rules)
    tasks = [make_task(rule, split, rules) for split in ("train", "validation", "test") for rule in splits[split]]
    audit = audit_tasks(tasks, rules)
    if not audit["all_checks_pass"]:
        raise RuntimeError(json.dumps(audit, sort_keys=True))

    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_rule_induction_benchmark_20261024",
        "schema_version": SCHEMA_VERSION,
        "status": "completed_cpu_generation_and_audit",
        "split_seed": SPLIT_SEED,
        "demo_counts": list(DEMO_COUNTS),
        "rule_families": list(FAMILY_ORDER),
        "candidate_rules_after_behavioral_deduplication": len(rules),
        "context_length": CONTEXT_LENGTH,
        "model_training_performed": False,
        "gpu_used": False,
        "test_used_for_selection": False,
    })
    write_json(output / "RULE_LIBRARY.json", {
        "candidate_count": len(rules),
        "families": {
            family: sum(rule[0] == family for rule in rules)
            for family in FAMILY_ORDER
        },
        "behavioral_signature_domain": len(INPUT_DOMAIN),
    })
    write_json(output / "AUDIT.json", audit)
    write_json(output / "SUMMARY.json", {
        "experiment": "crucible_rule_induction_benchmark_20261024",
        "status": "completed",
        "model_training_performed": False,
        "split_counts": {split: len(splits[split]) for split in splits},
        "audit": audit,
    })
    for split in ("train", "validation", "test"):
        write_jsonl(output / f"{split}.jsonl", [task for task in tasks if task["split"] == split])

    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({
        "candidate_rules": len(rules),
        "tasks": len(tasks),
        "split_counts": {split: len(splits[split]) for split in splits},
        "identifiability": audit["identifiability"],
        "output": str(output),
        "status": "completed",
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
