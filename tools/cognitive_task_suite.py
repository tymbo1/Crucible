#!/usr/bin/env python3
"""Deterministic CPU-only task generators and exact oracles for Crucible.

The suite is deliberately small and dependency-free.  It defines task records,
not model claims: a future training or evaluation runner may consume the
records, while the oracle fields remain the reference for validation.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "crucible_cognitive_task_suite_v1"
GENERATOR_VERSION = "2026-10-21.v1"
FAMILIES = ("math", "relational_binding", "logic", "error_correction")
SPLITS = ("train", "validation", "test")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rng(seed: int, family: str, split: str, index: int) -> random.Random:
    material = f"{GENERATOR_VERSION}|{seed}|{family}|{split}|{index}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _record_id(seed: int, family: str, split: str, index: int) -> str:
    return sha256_text(f"{SCHEMA_VERSION}|{seed}|{family}|{split}|{index}")[:20]


def _record(
    *,
    seed: int,
    family: str,
    split: str,
    index: int,
    prompt: str,
    target: str,
    oracle: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": _record_id(seed, family, split, index),
        "family": family,
        "split": split,
        "input": prompt,
        "target": target,
        "oracle": dict(oracle),
        "metadata": dict(metadata),
    }


MATH_COMPOSITIONS: Dict[str, Sequence[Tuple[str, ...]]] = {
    "train": (("add", "sub"), ("mul", "add"), ("sub", "mul")),
    "validation": (("add", "mul"), ("sub", "add", "mul")),
    "test": (("mul", "sub", "add"), ("add", "mul", "sub")),
}


def _apply_operation(value: int, operation: str, operand: int) -> int:
    if operation == "add":
        return value + operand
    if operation == "sub":
        return value - operand
    if operation == "mul":
        return value * operand
    raise ValueError(f"Unknown operation: {operation}")


def _math_case(rng: random.Random, split: str, index: int) -> Dict[str, Any]:
    compositions = MATH_COMPOSITIONS[split]
    operations = compositions[index % len(compositions)]
    value = rng.randint(-20, 20)
    start = value
    steps: List[Dict[str, int | str]] = []
    for operation in operations:
        operand = rng.randint(2, 9) if operation == "mul" else rng.randint(1, 12)
        value = _apply_operation(value, operation, operand)
        steps.append({"operation": operation, "operand": operand})
    return {"start": start, "steps": steps, "result": value, "operations": list(operations)}


def make_math_record(seed: int, split: str, index: int) -> Dict[str, Any]:
    rng = _rng(seed, "math", split, index)
    case = _math_case(rng, split, index)
    instructions = [f"start={case['start']}"]
    for number, step in enumerate(case["steps"], start=1):
        instructions.append(
            f"step{number}:{step['operation']}({step['operand']})"
        )
    prompt = (
        "Task: apply each arithmetic step from left to right. "
        "Return the final integer only. " + "; ".join(instructions)
    )
    return _record(
        seed=seed,
        family="math",
        split=split,
        index=index,
        prompt=prompt,
        target=str(case["result"]),
        oracle={"kind": "integer_exact", "value": case["result"]},
        metadata={
            "composition": case["operations"],
            "composition_length": len(case["operations"]),
            "held_out_composition": split != "train",
        },
    )


def make_relational_record(seed: int, split: str, index: int) -> Dict[str, Any]:
    rng = _rng(seed, "relational_binding", split, index)
    path_length = {"train": 2, "validation": 3, "test": 4}[split]
    entities = [f"E{index}_{position}" for position in range(path_length + 1)]
    facts = [
        f"{entities[position]} relates_to {entities[position + 1]}"
        for position in range(path_length)
    ]

    distractor_count = 2 + (index % 3)
    for distractor in range(distractor_count):
        left = f"D{index}_{distractor}_L"
        right = f"D{index}_{distractor}_R"
        facts.append(f"{left} relates_to {right}")
    rng.shuffle(facts)

    positive = index % 2 == 0
    if positive:
        query_left, query_right = entities[0], entities[-1]
        answer = "yes"
    else:
        query_left, query_right = entities[-1], entities[0]
        answer = "no"

    prompt = (
        "Facts: " + " | ".join(facts) +
        f". Query: does {query_left} relate_to {query_right}? "
        "Return yes or no only."
    )
    return _record(
        seed=seed,
        family="relational_binding",
        split=split,
        index=index,
        prompt=prompt,
        target=answer,
        oracle={
            "kind": "directed_transitive_relation",
            "answer": answer,
            "path_length": path_length if positive else None,
        },
        metadata={
            "distractor_count": distractor_count,
            "held_out_path_length": split != "train",
            "query_polarity": "positive" if positive else "reverse_negative",
        },
    )


def _logic_closure(facts: Iterable[str], rules: Iterable[Tuple[str, str]]) -> set[str]:
    closure = set(facts)
    changed = True
    while changed:
        changed = False
        for premise, conclusion in rules:
            if premise in closure and conclusion not in closure:
                closure.add(conclusion)
                changed = True
    return closure


def make_logic_record(seed: int, split: str, index: int) -> Dict[str, Any]:
    rng = _rng(seed, "logic", split, index)
    depth = {"train": 2, "validation": 3, "test": 4}[split]
    atoms = [f"P{index}_{position}" for position in range(depth + 1)]
    facts = [atoms[0]]
    rules = [(atoms[position], atoms[position + 1]) for position in range(depth)]
    distractors = [f"D{index}_{position}" for position in range(2)]
    facts.extend(distractors[:1])
    rules.append((distractors[0], distractors[1]))
    rng.shuffle(facts)
    rng.shuffle(rules)

    case = index % 3
    if case == 0:
        query = atoms[-1]
        answer = "entailed"
    elif case == 1:
        query = f"not {atoms[-1]}"
        answer = "contradicted"
    else:
        query = f"Q{index}"
        answer = "unknown"

    closure = _logic_closure(facts, rules)
    if answer == "entailed" and query not in closure:
        raise AssertionError("Generated entailed query is not in closure")
    if answer == "contradicted" and query.removeprefix("not ") not in closure:
        raise AssertionError("Generated contradicted query has no positive opposite")
    if answer == "unknown" and query in closure:
        raise AssertionError("Generated unknown query is in closure")

    fact_text = ", ".join(facts)
    rule_text = ", ".join(f"if {premise} then {conclusion}" for premise, conclusion in rules)
    prompt = (
        f"Facts: {fact_text}. Rules: {rule_text}. Query: {query}. "
        "Return entailed, contradicted or unknown only."
    )
    return _record(
        seed=seed,
        family="logic",
        split=split,
        index=index,
        prompt=prompt,
        target=answer,
        oracle={"kind": "forward_chain_three_way", "answer": answer, "closure": sorted(closure)},
        metadata={
            "rule_depth": depth,
            "held_out_rule_depth": split != "train",
            "query_case": case,
        },
    )


def make_error_correction_record(seed: int, split: str, index: int) -> Dict[str, Any]:
    rng = _rng(seed, "error_correction", split, index)
    case = _math_case(rng, split, index + 17)
    expected = int(case["result"])
    candidate_is_correct = index % 2 == 0
    candidate = expected if candidate_is_correct else expected + (1 if expected != 0 else 2)
    status = "correct" if candidate_is_correct else "incorrect"
    prompt = (
        "Question: apply the arithmetic steps "
        + ", ".join(
            f"{step['operation']} {step['operand']}" for step in case["steps"]
        )
        + f" to start value {case['start']}. Candidate answer: {candidate}. "
        "Return exactly status|correct_answer."
    )
    target = f"{status}|{expected}"
    return _record(
        seed=seed,
        family="error_correction",
        split=split,
        index=index,
        prompt=prompt,
        target=target,
        oracle={
            "kind": "candidate_verification",
            "candidate": candidate,
            "expected": expected,
            "status": status,
        },
        metadata={
            "source_composition": case["operations"],
            "candidate_is_correct": candidate_is_correct,
            "held_out_composition": split != "train",
        },
    )


GENERATORS = {
    "math": make_math_record,
    "relational_binding": make_relational_record,
    "logic": make_logic_record,
    "error_correction": make_error_correction_record,
}


def generate_records(seed: int, split: str, records_per_family: int) -> List[Dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if records_per_family < 1:
        raise ValueError("records_per_family must be positive")
    records: List[Dict[str, Any]] = []
    for family in FAMILIES:
        generator = GENERATORS[family]
        records.extend(
            generator(seed, split, index) for index in range(records_per_family)
        )
    return records

