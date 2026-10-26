#!/usr/bin/env python3
"""Run the CPU-only canonical/factorized operand representation control."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
STAGE_A_PATH = ROOT / "tools" / "run_crucible_stage_a_instrumentation_20261022.py"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_operand_representation_20261024"

SEEDS = (42, 1337)
ARMS = ("canonical_numeric", "factorized", "opaque_association")
STEPS = 1_000
CHECKPOINT_STEPS = (0, 250, 500, 750, 1_000)
RAW_BYTE_START = 3
ANSWER_MARKER = b"\nAnswer:"
VALIDATION_VALUES = (3, 8, 14, 19, 25, 31, 36, 42, 47, 53, 58, 64, 69, 75, 80, 86, 91, 94, 97, 99)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_a = import_module("stage_a_for_operand_representation", STAGE_A_PATH)


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


def opaque_label(value: int) -> str:
    labels = [f"{left}{right}" for left in "abcdefghij" for right in "klmnopqrst"]
    order = list(range(len(labels)))
    random.Random(stable_seed("opaque-permutation")).shuffle(order)
    return labels[order[value]]


def target(value: int) -> int:
    return (value + 1) % 10


def prompt_for(arm: str, value: int) -> str:
    if arm == "canonical_numeric":
        return f"Execute explicit rule y=(x+1)mod10. Input x={value}. Answer digit."
    if arm == "factorized":
        return f"Execute explicit rule y=(x+1)mod10. operand_tens={value // 10}; operand_ones={value % 10}. Answer digit."
    if arm == "opaque_association":
        return f"Execute operation SUCC on opaque operand symbol={opaque_label(value)}. Answer digit."
    raise ValueError(arm)


def make_record(arm: str, split: str, index: int, value: int) -> Dict[str, Any]:
    return {
        "schema_version": "crucible_operand_representation_v1",
        "record_id": hashlib.sha256(f"operand-representation|{arm}|{split}|{index}|{value}".encode()).hexdigest()[:20],
        "arm": arm,
        "split": split,
        "input_value": value,
        "input": prompt_for(arm, value),
        "target": str(target(value)),
        "oracle": {"kind": "successor_mod10", "value": target(value)},
        "metadata": {
            "value": value,
            "tens": value // 10,
            "ones": value % 10,
            "opaque_symbol": opaque_label(value),
        },
    }


def generate_records() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    validation = set(VALIDATION_VALUES)
    training = [value for value in range(100) if value not in validation]
    if set(range(10)) - {value // 10 for value in training}:
        raise RuntimeError("training does not cover every tens digit")
    if set(range(10)) - {value % 10 for value in training}:
        raise RuntimeError("training does not cover every ones digit")
    records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for arm in ARMS:
        records[arm] = {
            "train": [make_record(arm, "train", index, value) for index, value in enumerate(training)],
            "validation": [make_record(arm, "validation", index, value) for index, value in enumerate(VALIDATION_VALUES)],
        }
    return records


def preflight(records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> Dict[str, Any]:
    errors: List[str] = []
    overlaps: List[str] = []
    oracle_accuracy: Dict[str, float] = {}
    for arm, splits in records.items():
        train_prompts = {str(item["input"]) for item in splits["train"]}
        valid_prompts = {str(item["input"]) for item in splits["validation"]}
        overlaps.extend(sorted(train_prompts & valid_prompts))
        for split, values in splits.items():
            correct = 0
            for item in values:
                expected = target(int(item["input_value"]))
                if expected != int(item["target"]) or expected != int(item["oracle"]["value"]):
                    errors.append(str(item["record_id"]))
                else:
                    correct += 1
            oracle_accuracy[f"{arm}:{split}"] = correct / len(values)
    return {
        "oracle_errors": sorted(set(errors)),
        "train_validation_prompt_overlaps": overlaps,
        "oracle_accuracy": oracle_accuracy,
        "all_checks_pass": not errors and not overlaps and all(value == 1.0 for value in oracle_accuracy.values()),
        "training_values": [value for value in range(100) if value not in set(VALIDATION_VALUES)],
        "validation_values": list(VALIDATION_VALUES),
    }


def prepare(values: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for original in values:
        item = dict(original)
        prompt = str(item["input"]).encode("utf-8") + ANSWER_MARKER
        target_bytes = str(item["target"]).encode("utf-8")
        full = [stage_a.BOS_ID] + stage_a.serialize_bytes(prompt) + stage_a.serialize_bytes(target_bytes) + [stage_a.EOS_ID]
        if len(full) > stage_a.CONTEXT_LENGTH:
            raise RuntimeError(f"record exceeds context: {item['record_id']} ({len(full)})")
        item["_full"] = full
        item["_prefix_len"] = 1 + len(stage_a.serialize_bytes(prompt))
        prepared.append(item)
    return prepared


def metrics(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prepared = list(records)
    aggregate = stage_a.evaluate(model, prepared)
    predictions: List[str] = []
    with stage_a.torch.no_grad():
        batch = stage_a.batchify(prepared, list(range(len(prepared))))
        logits = stage_a.forward_model(model, batch)
        start = RAW_BYTE_START + ord("0")
        for row, item in enumerate(prepared):
            position = int(item["_prefix_len"]) - 1
            token = int(logits[row, position].argmax().item())
            predictions.append(str(token - start) if start <= token <= start + 9 else f"token:{token}")
    return {
        **aggregate,
        "digit_exact_accuracy": sum(prediction == str(item["target"]) for item, prediction in zip(prepared, predictions)) / len(prepared),
        "predicted_digit_histogram": {str(digit): predictions.count(str(digit)) for digit in range(10)},
    }


def run_arm(seed: int, arm: str, splits: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> Dict[str, Any]:
    torch = stage_a.torch
    prepared = {split: prepare(values) for split, values in splits.items()}
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    order = list(range(len(prepared["train"])))
    random.Random(stable_seed(seed, arm, "order")).shuffle(order)
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
            train_metrics = metrics(model, prepared["train"])
            validation_metrics = metrics(model, prepared["validation"])
            rows.append({"step": step, "train_step": train_info, "train": train_metrics, "validation": validation_metrics})
            print(json.dumps({"arm": arm, "seed": seed, "step": step, "train_accuracy": train_metrics["digit_exact_accuracy"], "validation_accuracy": validation_metrics["digit_exact_accuracy"]}, sort_keys=True), flush=True)
            if step > 0:
                stage_a.save_checkpoint(checkpoint_dir / f"step_{step:04d}.pt", model, optimizer, step, order)
    result = {
        "arm": arm,
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
    write_json(output / "RUN_CONFIG.json", {"experiment": "crucible_operand_representation_20261024", "arm": arm, "seed": seed, "steps": STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "gpu_used": False, "test_evaluated": False})
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
    records = generate_records()
    audit = preflight(records)
    if not audit["all_checks_pass"]:
        raise SystemExit(f"preflight failed: {json.dumps(audit, sort_keys=True)}")
    write_json(output / "RUN_CONFIG.json", {"experiment": "crucible_operand_representation_20261024", "status": "frozen_before_training", "arms": list(ARMS), "seeds": list(SEEDS), "steps": STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "gpu_used": False, "test_evaluated": False, "positive_control": "deterministic_successor_mod10"})
    write_json(output / "PREFLIGHT.json", audit)
    results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            results.append(run_arm(seed, arm, records[arm], output / arm / f"seed{seed}"))
            gc.collect()
    summary = {
        "experiment": "crucible_operand_representation_20261024",
        "status": "completed",
        "test_evaluated": False,
        "preflight_pass": audit["all_checks_pass"],
        "results": [{"arm": item["arm"], "seed": item["seed"], "train_accuracy": item["final_train"]["digit_exact_accuracy"], "validation_accuracy": item["final_validation"]["digit_exact_accuracy"], "elapsed_seconds": item["elapsed_seconds"]} for item in results],
    }
    write_json(output / "SUMMARY.json", summary)
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"output": str(output), "status": "completed", "preflight_pass": audit["all_checks_pass"], "arms": len(results)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
