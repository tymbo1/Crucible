#!/usr/bin/env python3
"""Frozen-checkpoint free-running and counterfactual composition diagnostic.

This diagnostic does not train or alter the intermediate-supervision runs.  It
evaluates their saved checkpoints on the same disjoint validation records in
three modes:

* teacher-forced reference;
* greedy free-running intermediate -> delimiter -> final generation;
* counterfactual final prediction after replacing the supplied intermediate
  with a different value and recomputing the expected second-step answer.

The final-token result for the teacher-forced and counterfactual conditions is
conditional on a supplied intermediate.  Free-running metrics are the
end-to-end execution diagnostic.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_PATH = ROOT / "tools" / "run_crucible_intermediate_supervision_20261023.py"
INPUT_ROOT = ROOT / "experiments" / "crucible_intermediate_supervision_20261023"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_free_running_composition_diagnostic_20261023"
SEEDS = (42, 1337)
CHECKPOINT_STEPS = (0, 500, 1_000, 1_500, 2_000)
RAW_BYTE_START = 3
COMMA = ","
ORIGINAL_PROMPT_SUFFIX = "Return one digit only."
AUGMENTED_PROMPT_SUFFIX = "Return the intermediate digit, then the final digit, separated by comma."
PROMPT_RE = re.compile(
    r"Task: arithmetic is modulo ten\. Apply both operations left-to-right\. "
    r"start=(?P<start>[0-9]); step1=(?P<op1>add|sub|mul)\((?P<arg1>[1-4])\); "
    r"step2=(?P<op2>add|sub|mul)\((?P<arg2>[1-4])\)\. Return one digit only\."
)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


intermediate = import_module("intermediate_supervision", INTERMEDIATE_PATH)
stage_a = intermediate.stage_a
ladder = intermediate.ladder


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_mod(value: int, operation: str, operand: int) -> int:
    if operation == "add":
        return (value + operand) % 10
    if operation == "sub":
        return (value - operand) % 10
    if operation == "mul":
        return (value * operand) % 10
    raise ValueError(operation)


def original_oracle(record: Mapping[str, Any]) -> Dict[str, Any]:
    original_input = str(record["input"]).replace(AUGMENTED_PROMPT_SUFFIX, ORIGINAL_PROMPT_SUFFIX)
    match = PROMPT_RE.fullmatch(original_input)
    if match is None:
        raise ValueError(f"cannot parse composition prompt: {record['record_id']}")
    start = int(match.group("start"))
    op1, arg1 = match.group("op1"), int(match.group("arg1"))
    op2, arg2 = match.group("op2"), int(match.group("arg2"))
    intermediate_value = apply_mod(start, op1, arg1)
    final_value = apply_mod(intermediate_value, op2, arg2)
    return {
        "start": start,
        "op1": op1,
        "arg1": arg1,
        "op2": op2,
        "arg2": arg2,
        "intermediate": intermediate_value,
        "final": final_value,
    }


def token_for_byte(value: str) -> int:
    encoded = value.encode("utf-8")
    if len(encoded) != 1:
        raise ValueError(value)
    return RAW_BYTE_START + encoded[0]


def token_for_digit(value: int) -> int:
    if not 0 <= value <= 9:
        raise ValueError(value)
    return token_for_byte(str(value))


def token_label(token: int) -> str:
    if RAW_BYTE_START + ord("0") <= token <= RAW_BYTE_START + ord("9"):
        return str(token - RAW_BYTE_START - ord("0"))
    if token == token_for_byte(COMMA):
        return COMMA
    return f"token:{token}"


def prompt_prefix(record: Mapping[str, Any]) -> List[int]:
    prefix_len = int(record["_prefix_len"])
    return list(record["_full"][:prefix_len])


def next_token(model: Any, tokens: Sequence[int]) -> int:
    torch = stage_a.torch
    x = torch.tensor([list(tokens)], dtype=torch.long)
    logits = stage_a.forward_model(model, {"x": x})
    return int(logits[0, -1].argmax().item())


def teacher_forced_final(model: Any, record: Mapping[str, Any], oracle: Mapping[str, Any]) -> int:
    prefix = prompt_prefix(record)
    return next_token(model, prefix + [token_for_digit(int(oracle["intermediate"])), token_for_byte(COMMA)])


def free_run(model: Any, record: Mapping[str, Any]) -> Dict[str, Any]:
    prefix = prompt_prefix(record)
    intermediate_token = next_token(model, prefix)
    delimiter_token = next_token(model, prefix + [intermediate_token])
    final_token = next_token(model, prefix + [intermediate_token, delimiter_token])
    return {
        "intermediate_token": intermediate_token,
        "delimiter_token": delimiter_token,
        "final_token": final_token,
        "intermediate": token_label(intermediate_token),
        "delimiter": token_label(delimiter_token),
        "final": token_label(final_token),
    }


def counterfactual(model: Any, record: Mapping[str, Any], oracle: Mapping[str, Any]) -> Dict[str, Any]:
    prefix = prompt_prefix(record)
    factual_intermediate = int(oracle["intermediate"])
    changed_intermediate = (factual_intermediate + 1) % 10
    factual_final_token = next_token(model, prefix + [token_for_digit(factual_intermediate), token_for_byte(COMMA)])
    counterfactual_final_token = next_token(model, prefix + [token_for_digit(changed_intermediate), token_for_byte(COMMA)])
    expected_counterfactual_final = apply_mod(changed_intermediate, str(oracle["op2"]), int(oracle["arg2"]))
    return {
        "factual_intermediate": factual_intermediate,
        "changed_intermediate": changed_intermediate,
        "factual_final_token": factual_final_token,
        "counterfactual_final_token": counterfactual_final_token,
        "factual_final": token_label(factual_final_token),
        "counterfactual_final": token_label(counterfactual_final_token),
        "expected_factual_final": int(oracle["final"]),
        "expected_counterfactual_final": expected_counterfactual_final,
    }


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_checkpoint(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    torch = stage_a.torch
    model.eval()
    oracle_rows = [original_oracle(record) for record in records]
    teacher_final_correct = 0
    free_intermediate_correct = 0
    free_delimiter_correct = 0
    free_final_correct = 0
    free_end_to_end_correct = 0
    free_final_given_intermediate_correct = 0
    free_final_when_intermediate_wrong_correct = 0
    free_intermediate_correct_count = 0
    free_intermediate_wrong_count = 0
    counterfactual_final_correct = 0
    counterfactual_no_change = 0
    factual_counterfactual_rows: List[Dict[str, Any]] = []
    by_pair: Dict[str, Dict[str, int]] = {}

    with torch.no_grad():
        for record, oracle in zip(records, oracle_rows):
            pair = "+".join(record["metadata"]["composition"])
            counters = by_pair.setdefault(pair, {
                "examples": 0,
                "teacher_forced_final_correct": 0,
                "free_intermediate_correct": 0,
                "free_delimiter_correct": 0,
                "free_final_correct": 0,
                "free_end_to_end_correct": 0,
                "counterfactual_final_correct": 0,
                "counterfactual_no_change": 0,
            })
            counters["examples"] += 1

            teacher_final_token = teacher_forced_final(model, record, oracle)
            teacher_ok = teacher_final_token == token_for_digit(int(oracle["final"]))
            teacher_final_correct += int(teacher_ok)
            counters["teacher_forced_final_correct"] += int(teacher_ok)

            generated = free_run(model, record)
            expected_intermediate_token = token_for_digit(int(oracle["intermediate"]))
            expected_final_token = token_for_digit(int(oracle["final"]))
            expected_delimiter_token = token_for_byte(COMMA)
            intermediate_ok = generated["intermediate_token"] == expected_intermediate_token
            delimiter_ok = generated["delimiter_token"] == expected_delimiter_token
            final_ok = generated["final_token"] == expected_final_token
            end_to_end_ok = intermediate_ok and delimiter_ok and final_ok
            free_intermediate_correct += int(intermediate_ok)
            free_delimiter_correct += int(delimiter_ok)
            free_final_correct += int(final_ok)
            free_end_to_end_correct += int(end_to_end_ok)
            counters["free_intermediate_correct"] += int(intermediate_ok)
            counters["free_delimiter_correct"] += int(delimiter_ok)
            counters["free_final_correct"] += int(final_ok)
            counters["free_end_to_end_correct"] += int(end_to_end_ok)
            if intermediate_ok:
                free_intermediate_correct_count += 1
                free_final_given_intermediate_correct += int(final_ok)
            else:
                free_intermediate_wrong_count += 1
                free_final_when_intermediate_wrong_correct += int(final_ok)

            cf = counterfactual(model, record, oracle)
            cf_ok = cf["counterfactual_final_token"] == token_for_digit(int(cf["expected_counterfactual_final"]))
            no_change = cf["counterfactual_final_token"] == cf["factual_final_token"]
            counterfactual_final_correct += int(cf_ok)
            counterfactual_no_change += int(no_change)
            counters["counterfactual_final_correct"] += int(cf_ok)
            counters["counterfactual_no_change"] += int(no_change)
            factual_counterfactual_rows.append({
                "record_id": record["record_id"],
                "pair": pair,
                "free": generated,
                "counterfactual": cf,
                "teacher_forced_final": token_label(teacher_final_token),
            })

    total = len(records)
    return {
        "examples": total,
        "teacher_forced_final_accuracy": rate(teacher_final_correct, total),
        "free_running_intermediate_accuracy": rate(free_intermediate_correct, total),
        "free_running_delimiter_accuracy": rate(free_delimiter_correct, total),
        "free_running_final_accuracy": rate(free_final_correct, total),
        "free_running_end_to_end_accuracy": rate(free_end_to_end_correct, total),
        "free_running_final_given_correct_intermediate": rate(
            free_final_given_intermediate_correct, free_intermediate_correct_count
        ),
        "free_running_final_when_intermediate_wrong": rate(
            free_final_when_intermediate_wrong_correct, free_intermediate_wrong_count
        ),
        "free_running_correct_intermediate_examples": free_intermediate_correct_count,
        "free_running_wrong_intermediate_examples": free_intermediate_wrong_count,
        "counterfactual_final_accuracy": rate(counterfactual_final_correct, total),
        "counterfactual_no_change_rate": rate(counterfactual_no_change, total),
        "by_operation_pair": {
            pair: {
                **counts,
                "teacher_forced_final_accuracy": rate(counts["teacher_forced_final_correct"], counts["examples"]),
                "free_running_intermediate_accuracy": rate(counts["free_intermediate_correct"], counts["examples"]),
                "free_running_delimiter_accuracy": rate(counts["free_delimiter_correct"], counts["examples"]),
                "free_running_final_accuracy": rate(counts["free_final_correct"], counts["examples"]),
                "free_running_end_to_end_accuracy": rate(counts["free_end_to_end_correct"], counts["examples"]),
                "counterfactual_final_accuracy": rate(counts["counterfactual_final_correct"], counts["examples"]),
                "counterfactual_no_change_rate": rate(counts["counterfactual_no_change"], counts["examples"]),
            }
            for pair, counts in sorted(by_pair.items())
        },
        "traces": factual_counterfactual_rows,
    }


def load_validation(seed: int) -> List[Dict[str, Any]]:
    path = INPUT_ROOT / f"seed{seed}_intermediate_and_final" / "validation.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    prepared = ladder.prepare_records(records)
    if len(prepared) != 64:
        raise RuntimeError(f"expected 64 validation records for seed {seed}, found {len(prepared)}")
    return prepared


def run_seed(seed: int, output: Path) -> Dict[str, Any]:
    torch = stage_a.torch
    validation = load_validation(seed)
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in CHECKPOINT_STEPS:
        checkpoint_path = INPUT_ROOT / f"seed{seed}_intermediate_and_final" / "checkpoints" / f"step_{step:04d}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = stage_a.make_dense(seed)
        model.load_state_dict(checkpoint["model"], strict=True)
        metrics = evaluate_checkpoint(model, validation)
        row = {"step": step, "checkpoint": str(checkpoint_path.relative_to(ROOT)), "metrics": metrics}
        rows.append(row)
        print(json.dumps({
            "seed": seed,
            "step": step,
            "teacher_forced_final_accuracy": metrics["teacher_forced_final_accuracy"],
            "free_running_intermediate_accuracy": metrics["free_running_intermediate_accuracy"],
            "free_running_final_accuracy": metrics["free_running_final_accuracy"],
            "free_running_end_to_end_accuracy": metrics["free_running_end_to_end_accuracy"],
            "counterfactual_final_accuracy": metrics["counterfactual_final_accuracy"],
            "counterfactual_no_change_rate": metrics["counterfactual_no_change_rate"],
        }, sort_keys=True), flush=True)
    result = {
        "seed": seed,
        "mode": "intermediate_and_final_frozen_checkpoints",
        "validation_examples": len(validation),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    write_json(output / f"seed{seed}.json", result)
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_free_running_composition_diagnostic_20261023",
        "status": "cpu_only_frozen_checkpoint_evaluation",
        "seeds": list(SEEDS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "validation_examples_per_seed": 64,
        "source_experiment": "crucible_intermediate_supervision_20261023",
        "training_performed": False,
        "test_evaluated": False,
        "heldout_evaluated": False,
        "counterfactual_change": "intermediate_plus_one_modulo_ten",
        "free_running_decode": "greedy_three_target_tokens",
    })
    results = [run_seed(seed, output) for seed in SEEDS]
    write_json(output / "SUMMARY.json", {
        "experiment": "crucible_free_running_composition_diagnostic_20261023",
        "status": "completed",
        "training_performed": False,
        "test_evaluated": False,
        "heldout_evaluated": False,
        "results": results,
    })
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"output": str(output), "seeds": len(results), "status": "completed"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
