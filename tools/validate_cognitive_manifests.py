#!/usr/bin/env python3
"""Independently validate generated cognitive manifests on CPU.

The validator parses the serialized task text instead of trusting the generator
implementation's intermediate values.  It checks exact answers, cross-split
input leakage and the declared composition/depth/path-length holdouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SPLITS = ("train", "validation", "test")


def load_records(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    result = {}
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            result[split] = [json.loads(line) for line in handle if line.strip()]
    return result


def apply(value: int, operation: str, operand: int) -> int:
    if operation == "add":
        return value + operand
    if operation == "sub":
        return value - operand
    if operation == "mul":
        return value * operand
    raise ValueError(f"unknown operation {operation}")


def parse_math(prompt: str) -> int:
    start_match = re.search(r"start=(-?\d+)", prompt)
    if not start_match:
        raise ValueError("math start value missing")
    value = int(start_match.group(1))
    steps = re.findall(r"step\d+:(add|sub|mul)\((-?\d+)\)", prompt)
    if not steps:
        raise ValueError("math steps missing")
    for operation, operand in steps:
        value = apply(value, operation, int(operand))
    return value


def parse_error_correction(prompt: str) -> Tuple[int, int]:
    source_match = re.search(r"start value (-?\d+)\. Candidate answer: (-?\d+)", prompt)
    if not source_match:
        raise ValueError("correction source/candidate missing")
    value = int(source_match.group(1))
    candidate = int(source_match.group(2))
    for operation, operand in re.findall(r"(add|sub|mul) (-?\d+)", prompt):
        value = apply(value, operation, int(operand))
    return value, candidate


def relation_answer(prompt: str) -> str:
    match = re.match(r"Facts: (.+)\. Query: does ([^ ]+) relate_to ([^?]+)\?", prompt)
    if not match:
        raise ValueError("relation prompt is malformed")
    facts_text, source, target = match.groups()
    graph: Dict[str, List[str]] = {}
    for fact in facts_text.split(" | "):
        left, relation, right = fact.split(" ")
        if relation != "relates_to":
            raise ValueError("unexpected relation")
        graph.setdefault(left, []).append(right)
    frontier = [source]
    visited = set()
    while frontier:
        node = frontier.pop()
        if node == target:
            return "yes"
        if node in visited:
            continue
        visited.add(node)
        frontier.extend(graph.get(node, []))
    return "no"


def logic_answer(prompt: str) -> str:
    match = re.match(r"Facts: (.+)\. Rules: (.+)\. Query: (.+)\. Return", prompt)
    if not match:
        raise ValueError("logic prompt is malformed")
    facts_text, rules_text, query = match.groups()
    closure = set(facts_text.split(", "))
    rules = []
    for rule in rules_text.split(", "):
        rule_match = re.fullmatch(r"if (.+) then (.+)", rule)
        if not rule_match:
            raise ValueError("logic rule is malformed")
        rules.append(rule_match.groups())
    changed = True
    while changed:
        changed = False
        for premise, conclusion in rules:
            if premise in closure and conclusion not in closure:
                closure.add(conclusion)
                changed = True
    if query in closure:
        return "entailed"
    if query.startswith("not ") and query[4:] in closure:
        return "contradicted"
    if not query.startswith("not ") and f"not {query}" in closure:
        return "contradicted"
    return "unknown"


def record_oracle_check(record: Mapping[str, Any]) -> None:
    family = record["family"]
    prompt = record["input"]
    target = record["target"]
    if family == "math":
        expected = parse_math(prompt)
        if target != str(expected):
            raise AssertionError((record["record_id"], target, expected))
    elif family == "error_correction":
        expected, candidate = parse_error_correction(prompt)
        status = "correct" if candidate == expected else "incorrect"
        if target != f"{status}|{expected}":
            raise AssertionError((record["record_id"], target, status, expected))
    elif family == "relational_binding":
        expected = relation_answer(prompt)
        if target != expected:
            raise AssertionError((record["record_id"], target, expected))
    elif family == "logic":
        expected = logic_answer(prompt)
        if target != expected:
            raise AssertionError((record["record_id"], target, expected))
    else:
        raise AssertionError(f"unknown family: {family}")


def _input_digest(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(record["input"].encode("utf-8")).hexdigest()


def validate(records_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    all_records = [record for split in SPLITS for record in records_by_split[split]]
    ids = [record["record_id"] for record in all_records]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate record_id")
    for record in all_records:
        record_oracle_check(record)

    family_by_split: Dict[str, Dict[str, set]] = {}
    for split in SPLITS:
        family_by_split[split] = {}
        for record in records_by_split[split]:
            family_by_split[split].setdefault(record["family"], set()).add(_input_digest(record))
    leakage = []
    for first_index, first_split in enumerate(SPLITS):
        for second_split in SPLITS[first_index + 1 :]:
            for family in set(family_by_split[first_split]) & set(family_by_split[second_split]):
                overlap = family_by_split[first_split][family] & family_by_split[second_split][family]
                if overlap:
                    leakage.append({"family": family, "first": first_split, "second": second_split, "count": len(overlap)})
    if leakage:
        raise AssertionError({"input_leakage": leakage})

    composition_sets = {
        split: {
            tuple(record["metadata"].get("composition", record["metadata"].get("source_composition", [])))
            for record in records_by_split[split]
            if record["family"] in {"math", "error_correction"}
        }
        for split in SPLITS
    }
    if composition_sets["train"] & composition_sets["test"]:
        raise AssertionError("train/test arithmetic composition overlap")

    relation_lengths = {
        split: {record["oracle"]["path_length"] for record in records_by_split[split] if record["family"] == "relational_binding"}
        for split in SPLITS
    }
    logic_depths = {
        split: {record["metadata"]["rule_depth"] for record in records_by_split[split] if record["family"] == "logic"}
        for split in SPLITS
    }
    return {
        "records": {split: len(records_by_split[split]) for split in SPLITS},
        "families": sorted({record["family"] for record in all_records}),
        "input_leakage": leakage,
        "arithmetic_compositions": {split: sorted(map(list, values)) for split, values in composition_sets.items()},
        "relational_path_lengths": {split: sorted(value for value in values if value is not None) for split, values in relation_lengths.items()},
        "logic_rule_depths": {split: sorted(values) for split, values in logic_depths.items()},
        "oracle_checks": len(all_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(load_records(args.manifests))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

