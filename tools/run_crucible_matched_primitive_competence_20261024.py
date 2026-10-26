#!/usr/bin/env python3
"""Run the frozen CPU-only matched primitive-competence gate."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
POSITION_PATH = ROOT / "tools" / "run_crucible_operation_position_gate_20261024.py"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_matched_primitive_competence_20261024"

SEEDS = (42, 1337)
SPLIT_SEED = 20261024
OPERATIONS = ("add", "sub", "mul")
OPERANDS = (1, 2, 3, 4)
INTERMEDIATES = tuple(range(10))
TRAIN_CONTEXTS = 72
INTERPOLATION_CONTEXTS = 28
GENERALIZATION_CONTEXTS = 20
TINY_CONTEXTS = 8
STEPS = 2_000
CHECKPOINT_STEPS = (0, 500, 1_000, 1_500, 2_000)
INTERPOLATION_THRESHOLD = 0.90
ANSWER_MARKER = "Return the final digit only."


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


position = import_module("operation_position_gate_for_matched_primitive", POSITION_PATH)
stage_a = position.stage_a


Context = Tuple[str, int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(record), sort_keys=True) for record in records) + "\n", encoding="utf-8")


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def context_key(context: Context) -> str:
    operation, supplied_intermediate, operand = context
    return f"{operation}:{supplied_intermediate}:{operand}"


def all_contexts() -> List[Context]:
    return [
        (operation, supplied_intermediate, operand)
        for operation in OPERATIONS
        for supplied_intermediate in INTERMEDIATES
        for operand in OPERANDS
    ]


def generalization_contexts() -> List[Context]:
    withheld_pairs = (("mul", 4), ("sub", 1))
    return [
        (operation, supplied_intermediate, operand)
        for operation, operand in withheld_pairs
        for supplied_intermediate in INTERMEDIATES
    ]


def build_context_splits() -> Dict[str, List[Context]]:
    generalization = generalization_contexts()
    generalization_set = set(generalization)
    candidates = [context for context in all_contexts() if context not in generalization_set]
    candidates_by_key = {context: context for context in candidates}
    selected: List[Context] = []

    def add(context: Context) -> None:
        if context in candidates_by_key and context not in selected:
            selected.append(context)

    # Cover every available operation/operand pair first.
    for operation in OPERATIONS:
        for operand in OPERANDS:
            pair_candidates = [context for context in candidates if context[0] == operation and context[2] == operand]
            if pair_candidates:
                add(sorted(pair_candidates)[0])

    # Cover every supplied intermediate value explicitly.
    for supplied_intermediate in INTERMEDIATES:
        value_candidates = [context for context in candidates if context[1] == supplied_intermediate]
        add(sorted(value_candidates)[0])

    remaining = [context for context in candidates if context not in selected]
    random.Random(stable_seed(SPLIT_SEED, "fill")).shuffle(remaining)
    selected.extend(remaining[: TRAIN_CONTEXTS - len(selected)])
    selected = sorted(selected)
    if len(selected) != TRAIN_CONTEXTS:
        raise RuntimeError(f"unexpected training context count: {len(selected)}")

    training_set = set(selected)
    interpolation = sorted([context for context in candidates if context not in training_set])
    if len(interpolation) != INTERPOLATION_CONTEXTS:
        raise RuntimeError(f"unexpected interpolation context count: {len(interpolation)}")
    if len(generalization) != GENERALIZATION_CONTEXTS:
        raise RuntimeError(f"unexpected generalization context count: {len(generalization)}")

    if {context[0] for context in selected} != set(OPERATIONS):
        raise RuntimeError("training does not cover every operation")
    if {context[1] for context in selected} != set(INTERMEDIATES):
        raise RuntimeError("training does not cover every supplied intermediate")
    if {context[2] for context in selected} != set(OPERANDS):
        raise RuntimeError("training does not cover every operand")

    return {
        "train": selected,
        "interpolation": interpolation,
        "generalization": sorted(generalization),
    }


def choose_tiny_contexts(training: Sequence[Context]) -> List[Context]:
    selected: List[Context] = []
    for operation in OPERATIONS:
        for context in training:
            if context[0] == operation and context not in selected:
                selected.append(context)
                break
    for operand in OPERANDS:
        for context in training:
            if context[2] == operand and context not in selected:
                selected.append(context)
                break
    for context in training:
        if len(selected) >= TINY_CONTEXTS:
            break
        if context not in selected:
            selected.append(context)
    if len(selected) != TINY_CONTEXTS:
        raise RuntimeError("could not construct tiny overfit set")
    return sorted(selected)


def make_record(split: str, context: Context) -> Dict[str, Any]:
    operation, supplied_intermediate, operand = context
    final_value = position.apply_mod(supplied_intermediate, operation, operand)
    prompt = (
        "Task: position-gate modulo-ten arithmetic. Apply the supplied intermediate digit "
        f"{supplied_intermediate} as the current state. step2={operation}({operand}). "
        f"{ANSWER_MARKER}"
    )
    payload = {
        "split": split,
        "operation": operation,
        "supplied_intermediate": supplied_intermediate,
        "operand": operand,
        "final": final_value,
    }
    record_id = hashlib.sha256(
        json.dumps({"schema": "matched_primitive_v1", **payload}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return {
        "schema_version": "crucible_matched_primitive_competence_v1",
        "record_id": record_id,
        "stage": "matched_primitive_competence",
        "split": split,
        "condition": "B_only",
        "input": prompt,
        "target": str(final_value),
        "oracle": {"kind": "mod10_single_operation", "value": final_value},
        "metadata": {
            "condition": "B_only",
            "operation": operation,
            "composition": [operation],
            "supplied_intermediate": supplied_intermediate,
            "operand": operand,
            "final": final_value,
            "context_key": context_key(context),
        },
    }


def build_records() -> Dict[str, Any]:
    contexts = build_context_splits()
    tiny_train_contexts = choose_tiny_contexts(contexts["train"])
    tiny_probe_contexts = contexts["interpolation"][:TINY_CONTEXTS]
    main = {
        split: [make_record(split, context) for context in values]
        for split, values in contexts.items()
    }
    tiny = {
        "train": [make_record("tiny_train", context) for context in tiny_train_contexts],
        "probe": [make_record("tiny_probe", context) for context in tiny_probe_contexts],
    }
    return {"main": main, "tiny": tiny, "contexts": contexts, "tiny_train_contexts": tiny_train_contexts}


def oracle_audit(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    seen_ids: set[str] = set()
    seen_inputs: set[str] = set()
    for values in groups.values():
        for record in values:
            metadata = record["metadata"]
            expected = position.apply_mod(
                int(metadata["supplied_intermediate"]),
                str(metadata["operation"]),
                int(metadata["operand"]),
            )
            if record["target"] != str(expected) or int(metadata["final"]) != expected:
                raise RuntimeError(f"oracle mismatch: {record['record_id']}")
            if record["record_id"] in seen_ids or record["input"] in seen_inputs:
                raise RuntimeError(f"duplicate record/input: {record['record_id']}")
            seen_ids.add(record["record_id"])
            seen_inputs.add(record["input"])


def prepare(values: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return position.prepare(values)


def checkpoint_metrics(model: Any, prepared: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    return {
        split: position.custom_metrics(model, values, "B_only")
        for split, values in prepared.items()
    }


def overall_accuracy(metrics: Mapping[str, Any], split: str) -> float:
    return float(metrics[split]["overall"]["free_end_to_end_accuracy"])


def run_arm(seed: int, arm: str, records: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> Dict[str, Any]:
    torch = stage_a.torch
    prepared = {split: prepare(values) for split, values in records.items()}
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    order = list(range(len(prepared["train"])))
    random.Random(stable_seed(seed, arm, "training-order")).shuffle(order)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint_dir / "step_0000.pt", model, optimizer, 0, order)
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(STEPS + 1):
        train_info = None
        if step > 0:
            indices = stage_a.batch_indices(order, step)
            train_info = stage_a.train_step(model, optimizer, prepared["train"], indices)
        if step in CHECKPOINT_STEPS:
            metrics = checkpoint_metrics(model, prepared)
            row = {"step": step, "train_step": train_info, "metrics": metrics}
            rows.append(row)
            values = {
                "arm": arm,
                "seed": seed,
                "step": step,
                "train_exact": overall_accuracy(metrics, "train"),
            }
            if arm == "full":
                values.update({
                    "interpolation_exact": overall_accuracy(metrics, "interpolation"),
                    "generalization_exact": overall_accuracy(metrics, "generalization"),
                })
            else:
                values["probe_exact"] = overall_accuracy(metrics, "probe")
            print(json.dumps(values, sort_keys=True), flush=True)
        if step == STEPS:
            stage_a.save_checkpoint(checkpoint_dir / f"step_{STEPS:04d}.pt", model, optimizer, step, order)

    final_metrics = rows[-1]["metrics"]
    result = {
        "seed": seed,
        "arm": arm,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_examples": len(records["train"]),
        "evaluation_examples": {split: len(values) for split, values in records.items()},
        "steps": STEPS,
        "interpolation_threshold": INTERPOLATION_THRESHOLD if arm == "full" else None,
        "final_train_exact": overall_accuracy(final_metrics, "train"),
        "final_interpolation_exact": overall_accuracy(final_metrics, "interpolation") if arm == "full" else None,
        "final_generalization_exact": overall_accuracy(final_metrics, "generalization") if arm == "full" else None,
        "final_probe_exact": overall_accuracy(final_metrics, "probe") if arm == "tiny" else None,
        "tiny_overfit_pass": overall_accuracy(final_metrics, "train") >= 0.99 if arm == "tiny" else None,
        "interpolation_pass": overall_accuracy(final_metrics, "interpolation") >= INTERPOLATION_THRESHOLD if arm == "full" else None,
        "elapsed_seconds": time.perf_counter() - started,
        "records": rows,
    }
    write_json(output / "RESULTS.json", result)
    for split, values in records.items():
        write_jsonl(output / f"{split}.jsonl", values)
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_matched_primitive_competence_20261024",
        "status": "cpu_only_dense_control",
        "seed": seed,
        "arm": arm,
        "steps": STEPS,
        "checkpoint_metrics": list(CHECKPOINT_STEPS),
        "checkpoint_files": ["checkpoints/step_0000.pt", f"checkpoints/step_{STEPS:04d}.pt"],
        "interpolation_threshold": INTERPOLATION_THRESHOLD if arm == "full" else None,
        "heldout_used_for_selection": False,
        "gpu_used": False,
        "original_benchmark_modified": False,
    })
    del model, optimizer
    gc.collect()
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

    bundle = build_records()
    oracle_audit(bundle["main"])
    oracle_audit(bundle["tiny"])
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_matched_primitive_competence_20261024",
        "status": "frozen_before_training",
        "seeds": list(SEEDS),
        "split_seed": SPLIT_SEED,
        "steps": STEPS,
        "checkpoint_metrics": list(CHECKPOINT_STEPS),
        "train_contexts": TRAIN_CONTEXTS,
        "interpolation_contexts": INTERPOLATION_CONTEXTS,
        "generalization_contexts": GENERALIZATION_CONTEXTS,
        "tiny_contexts": TINY_CONTEXTS,
        "interpolation_threshold": INTERPOLATION_THRESHOLD,
        "generalization_withheld_operation_operand_pairs": [["mul", 4], ["sub", 1]],
        "gpu_used": False,
        "original_benchmark_modified": False,
    })
    write_json(output / "SPLIT_MANIFEST.json", {
        "split_seed": SPLIT_SEED,
        "contexts": {
            split: [context_key(context) for context in values]
            for split, values in bundle["contexts"].items()
        },
        "tiny_train": [context_key(context) for context in bundle["tiny_train_contexts"]],
        "tiny_probe": [context_key(context) for context in bundle["contexts"]["interpolation"][:TINY_CONTEXTS]],
    })

    results: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm, records in (("tiny", bundle["tiny"]), ("full", bundle["main"])):
            arm_output = output / f"seed{seed}_{arm}"
            arm_output.mkdir(parents=True, exist_ok=True)
            results.append(run_arm(seed, arm, records, arm_output))

    summary = {
        "experiment": "crucible_matched_primitive_competence_20261024",
        "status": "completed",
        "interpolation_threshold": INTERPOLATION_THRESHOLD,
        "results": results,
    }
    write_json(output / "SUMMARY.json", summary)
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"arms": len(results), "output": str(output), "status": "completed"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
