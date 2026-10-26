#!/usr/bin/env python3
"""Run the CPU-only single-primitive transfer ladder."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
STAGE_A_PATH = ROOT / "tools" / "run_crucible_stage_a_instrumentation_20261022.py"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_single_primitive_transfer_ladder_20261024"

SEEDS = (42, 1337)
TIERS = ("single_fixed", "single_operand", "single_parameter", "depth_2", "depth_3")
STEPS = 500
CHECKPOINT_STEPS = (0, 100, 250, 500)
RAW_BYTE_START = 3
ANSWER_MARKER = b"\nAnswer:"


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_a = import_module("stage_a_for_single_primitive_ladder", STAGE_A_PATH)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(value), sort_keys=True) for value in values) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def apply_op(value: int, operation: str, operand: int) -> int:
    if operation == "add":
        return (value + operand) % 10
    if operation == "sub":
        return (value - operand) % 10
    if operation == "mul":
        return (value * operand) % 10
    raise ValueError(operation)


def evaluate_spec(tier: str, spec: Mapping[str, Any]) -> int:
    if tier == "single_fixed":
        return (int(spec["x"]) + 1) % 10
    if tier == "single_operand":
        return (int(spec["x"]) + int(spec["z"]) + 1) % 10
    if tier == "single_parameter":
        return (int(spec["x"]) + int(spec["z"]) + int(spec["c"])) % 10
    value = int(spec["x"])
    for operation, operand_key in spec["operations"]:
        value = apply_op(value, operation, int(spec[operand_key]))
    return value


def prompt_for(tier: str, spec: Mapping[str, Any]) -> str:
    if tier == "single_fixed":
        return f"Execute explicit rule y=(x+1)mod10. Input x={spec['x']}. Answer digit."
    if tier == "single_operand":
        return f"Execute explicit rule y=(x+z+1)mod10. Input x={spec['x']},z={spec['z']}. Answer digit."
    if tier == "single_parameter":
        return f"Execute explicit rule y=(x+z+c)mod10. Parameter c={spec['c']}. Input x={spec['x']},z={spec['z']}. Answer digit."
    operations = "; ".join(f"{name}={op}({spec[key]})" for op, key, name in spec["rendered_operations"])
    return f"Execute explicit rule y=op3(op2(op1(x))) with {operations}. Input x={spec['x']}. Answer digit." if tier == "depth_3" else f"Execute explicit rule y=op2(op1(x)) with {operations}. Input x={spec['x']}. Answer digit."


def record(tier: str, split: str, index: int, spec: Mapping[str, Any]) -> Dict[str, Any]:
    target = evaluate_spec(tier, spec)
    payload = {"tier": tier, "split": split, "index": index, "spec": dict(spec), "target": target}
    record_id = hashlib.sha256(json.dumps(payload, sort_keys=True, default=list).encode("utf-8")).hexdigest()[:20]
    return {
        "schema_version": "crucible_single_primitive_transfer_v1",
        "record_id": record_id,
        "tier": tier,
        "split": split,
        "group": str(spec.get("group", "transfer")),
        "input": prompt_for(tier, spec),
        "target": str(target),
        "oracle": {"kind": "deterministic_symbolic_executor", "value": target},
        "metadata": dict(spec),
    }


def generate_tier(tier: str) -> Dict[str, List[Dict[str, Any]]]:
    train_specs: List[Dict[str, Any]] = []
    validation_specs: List[Dict[str, Any]] = []

    if tier == "single_fixed":
        train_specs = [{"x": x, "group": "train_operand"} for x in range(8)]
        validation_specs = [{"x": x, "group": "unseen_operand"} for x in range(8, 10)]
    elif tier == "single_operand":
        heldout = {(x, z) for x in range(10) for z in range(10) if (x + 3 * z) % 11 == 0}
        train_specs = [{"x": x, "z": z, "group": "seen_pair"} for x in range(10) for z in range(10) if (x, z) not in heldout]
        validation_specs = [{"x": x, "z": z, "group": "unseen_pair"} for x, z in sorted(heldout)]
    elif tier == "single_parameter":
        train_specs = [{"x": x, "z": z, "c": c, "group": "seen_parameter"} for c in range(5) for x in range(5) for z in range(4)]
        validation_specs = [{"x": x, "z": z, "c": c, "group": "unseen_parameter"} for c in range(5, 10) for x in range(5) for z in range(4)]
    elif tier in {"depth_2", "depth_3"}:
        parameter_specs = (
            [(a, b, None) for a in range(1, 4) for b in range(1, 4)]
            if tier == "depth_2"
            else [(a, b, c) for a in range(1, 4) for b in range(1, 4) for c in range(1, 4)]
        )
        for parameter_index, (a, b, c) in enumerate(parameter_specs):
            base: Dict[str, Any] = {"a": a, "b": b, "c": c, "group": "seen_parameters"}
            if tier == "depth_2":
                base["operations"] = [("add", "a"), ("mul", "b")]
                base["rendered_operations"] = [("add", "a", "op1"), ("mul", "b", "op2")]
            else:
                base["operations"] = [("add", "a"), ("mul", "b"), ("sub", "c")]
                base["rendered_operations"] = [("add", "a", "op1"), ("mul", "b", "op2"), ("sub", "c", "op3")]
            for x in range(8):
                train_specs.append({**base, "x": x, "group": "seen_parameter"})
            for x in range(8, 10):
                validation_specs.append({**base, "x": x, "group": "unseen_operand"})
    else:
        raise ValueError(tier)

    train = [record(tier, "train", index, spec) for index, spec in enumerate(train_specs)]
    validation = [record(tier, "validation", index, spec) for index, spec in enumerate(validation_specs)]
    prompts = [str(item["input"]) for item in train + validation]
    if len(prompts) != len(set(prompts)):
        raise RuntimeError(f"duplicate prompts in {tier}")
    return {"train": train, "validation": validation}


def audit(records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> Dict[str, Any]:
    oracle_errors: List[str] = []
    overlaps: List[str] = []
    oracle_accuracy: Dict[str, float] = {}
    for tier, splits in records.items():
        train_prompts = {str(item["input"]) for item in splits["train"]}
        validation_prompts = {str(item["input"]) for item in splits["validation"]}
        overlaps.extend(sorted(train_prompts & validation_prompts))
        for split, values in splits.items():
            correct = 0
            for item in values:
                expected = evaluate_spec(tier, item["metadata"])
                correct += int(str(expected) == str(item["target"]) == str(item["oracle"]["value"]))
                if expected != int(item["target"]):
                    oracle_errors.append(str(item["record_id"]))
            oracle_accuracy[f"{tier}:{split}"] = correct / len(values) if values else None
    return {
        "oracle_errors": sorted(set(oracle_errors)),
        "train_validation_prompt_overlaps": overlaps,
        "oracle_accuracy": oracle_accuracy,
        "all_checks_pass": not oracle_errors and not overlaps and all(value == 1.0 for value in oracle_accuracy.values()),
    }


def prepare(values: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for original in values:
        item = dict(original)
        prompt = str(item["input"]).encode("utf-8") + ANSWER_MARKER
        target = str(item["target"]).encode("utf-8")
        full = [stage_a.BOS_ID] + stage_a.serialize_bytes(prompt) + stage_a.serialize_bytes(target) + [stage_a.EOS_ID]
        if len(full) > stage_a.CONTEXT_LENGTH:
            raise RuntimeError(f"record exceeds context: {item['record_id']} ({len(full)})")
        item["_full"] = full
        item["_prefix_len"] = 1 + len(stage_a.serialize_bytes(prompt))
        prepared.append(item)
    return prepared


def digit_metrics(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prepared = list(records)
    metrics = stage_a.evaluate(model, prepared)
    predictions: List[str] = []
    with stage_a.torch.no_grad():
        batch = stage_a.batchify(prepared, list(range(len(prepared))))
        logits = stage_a.forward_model(model, batch)
        digit_start = RAW_BYTE_START + ord("0")
        for row, item in enumerate(prepared):
            position = int(item["_prefix_len"]) - 1
            token = int(logits[row, position].argmax().item())
            predictions.append(str(token - digit_start) if digit_start <= token <= digit_start + 9 else f"token:{token}")
    by_group: Dict[str, List[bool]] = {}
    for item, prediction in zip(prepared, predictions):
        by_group.setdefault(str(item["group"]), []).append(prediction == str(item["target"]))
    return {
        **metrics,
        "digit_exact_accuracy": sum(prediction == str(item["target"]) for item, prediction in zip(prepared, predictions)) / len(prepared),
        "predicted_digit_histogram": {str(digit): predictions.count(str(digit)) for digit in range(10)},
        "by_group": {group: {"examples": len(values), "digit_exact_accuracy": sum(values) / len(values)} for group, values in sorted(by_group.items())},
    }


def run_tier(seed: int, tier: str, splits: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> Dict[str, Any]:
    torch = stage_a.torch
    prepared = {split: prepare(values) for split, values in splits.items()}
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    order = list(range(len(prepared["train"])))
    random.Random(stable_seed(seed, tier, "train-order")).shuffle(order)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint_dir / "step_0000.pt", model, optimizer, 0, order)
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(STEPS + 1):
        train_info = None
        if step > 0:
            train_info = stage_a.train_step(model, optimizer, prepared["train"], stage_a.batch_indices(order, step))
        if step in CHECKPOINT_STEPS:
            train_metrics = digit_metrics(model, prepared["train"])
            validation_metrics = digit_metrics(model, prepared["validation"])
            row = {"step": step, "train_step": train_info, "train": train_metrics, "validation": validation_metrics}
            rows.append(row)
            print(json.dumps({
                "tier": tier,
                "seed": seed,
                "step": step,
                "train_digit_accuracy": train_metrics["digit_exact_accuracy"],
                "validation_digit_accuracy": validation_metrics["digit_exact_accuracy"],
            }, sort_keys=True), flush=True)
            if step > 0:
                stage_a.save_checkpoint(checkpoint_dir / f"step_{step:04d}.pt", model, optimizer, step, order)
    result = {
        "tier": tier,
        "seed": seed,
        "steps": STEPS,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_examples": len(splits["train"]),
        "validation_examples": len(splits["validation"]),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "final_train": rows[-1]["train"],
        "final_validation": rows[-1]["validation"],
    }
    write_json(output / "RESULTS.json", result)
    write_json(output / "RUN_CONFIG.json", {"experiment": "crucible_single_primitive_transfer_ladder_20261024", "tier": tier, "seed": seed, "steps": STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "gpu_used": False, "test_evaluated": False})
    write_jsonl(output / "train.jsonl", splits["train"])
    write_jsonl(output / "validation.jsonl", splits["validation"])
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
    records = {tier: generate_tier(tier) for tier in TIERS}
    preflight = audit(records)
    if not preflight["all_checks_pass"]:
        raise SystemExit(f"preflight failed: {json.dumps(preflight, sort_keys=True)}")
    write_json(output / "RUN_CONFIG.json", {"experiment": "crucible_single_primitive_transfer_ladder_20261024", "status": "frozen_before_training", "tiers": list(TIERS), "seeds": list(SEEDS), "steps": STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "gpu_used": False, "test_evaluated": False, "positive_control": "deterministic_symbolic_executor"})
    write_json(output / "PREFLIGHT.json", preflight)
    results: List[Dict[str, Any]] = []
    for tier in TIERS:
        tier_dir = output / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            results.append(run_tier(seed, tier, records[tier], tier_dir / f"seed{seed}"))
            gc.collect()
    summary = {
        "experiment": "crucible_single_primitive_transfer_ladder_20261024",
        "status": "completed",
        "test_evaluated": False,
        "preflight_pass": preflight["all_checks_pass"],
        "symbolic_oracle_accuracy": preflight["oracle_accuracy"],
        "results": [{"tier": item["tier"], "seed": item["seed"], "train_accuracy": item["final_train"]["digit_exact_accuracy"], "validation_accuracy": item["final_validation"]["digit_exact_accuracy"], "elapsed_seconds": item["elapsed_seconds"]} for item in results],
    }
    write_json(output / "SUMMARY.json", summary)
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"output": str(output), "status": "completed", "preflight_pass": preflight["all_checks_pass"], "arms": len(results)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
