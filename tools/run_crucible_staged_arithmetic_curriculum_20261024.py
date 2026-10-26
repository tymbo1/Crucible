#!/usr/bin/env python3
"""Run the CPU-only staged arithmetic curriculum against direct training."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
STAGE_A_PATH = ROOT / "tools" / "run_crucible_stage_a_instrumentation_20261022.py"
CARRY_PATH = ROOT / "tools" / "run_crucible_carry_sensitive_arithmetic_20261024.py"
SUCCESSOR_CHECKPOINT = ROOT / "experiments" / "crucible_operand_representation_20261024" / "factorized" / "seed42" / "checkpoints" / "step_1000.pt"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_staged_arithmetic_curriculum_20261024"

SEEDS = (42, 1337)
ARM = "factorized_digits"
STAGES = ("stage_1_no_carry", "stage_2_ones_carry", "stage_3_cascading_carry")
STAGE_REGIMES = {
    "stage_1_no_carry": ("no_carry",),
    "stage_2_ones_carry": ("no_carry", "ones_carry"),
    "stage_3_cascading_carry": ("no_carry", "ones_carry", "cascading_carry"),
}
STAGE_STEPS = 1_000
DIRECT_STEPS = 3_000
CHECKPOINT_STEPS = (0, 1_000, 2_000, 3_000)
TRAIN_THRESHOLD = 0.95
VALIDATION_THRESHOLD = 0.90
ANSWER_MARKER = b"\nAnswer:"


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_a = import_module("stage_a_for_curriculum", STAGE_A_PATH)
carry = import_module("carry_for_curriculum", CARRY_PATH)


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


def build_records() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    pair_splits = carry.choose_splits()
    records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for stage_name, regimes in STAGE_REGIMES.items():
        train_pairs = [pair for current_regime in regimes for pair in pair_splits[current_regime]["train"]]
        validation_pairs = [pair for current_regime in regimes for pair in pair_splits[current_regime]["validation"]]
        records[stage_name] = {
            "train": [carry.make_record(ARM, stage_name, index, a, b) for index, (a, b) in enumerate(train_pairs)],
            "validation": [carry.make_record(ARM, stage_name, index, a, b) for index, (a, b) in enumerate(validation_pairs)],
        }
    return records


def audit(records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> Dict[str, Any]:
    errors: List[str] = []
    overlaps: List[str] = []
    stage_summary: Dict[str, Any] = {}
    for stage_name, splits in records.items():
        train_pairs = {(int(item["metadata"]["a"]), int(item["metadata"]["b"])) for item in splits["train"]}
        validation_pairs = {(int(item["metadata"]["a"]), int(item["metadata"]["b"])) for item in splits["validation"]}
        overlaps.extend([f"{stage_name}:{pair}" for pair in sorted(train_pairs & validation_pairs)])
        regime_counts: Dict[str, int] = defaultdict(int)
        for split, values in splits.items():
            for item in values:
                a = int(item["metadata"]["a"])
                b = int(item["metadata"]["b"])
                expected = a + b
                if expected != int(item["target"]) or expected != int(item["oracle"]["value"]):
                    errors.append(str(item["record_id"]))
                regime_counts[f"{split}:{item['metadata']['regime']}"] += 1
        stage_summary[stage_name] = {
            "train_examples": len(splits["train"]),
            "validation_examples": len(splits["validation"]),
            "regime_counts": dict(sorted(regime_counts.items())),
        }
    return {
        "oracle_errors": sorted(set(errors)),
        "train_validation_pair_overlaps": overlaps,
        "stage_summary": stage_summary,
        "all_checks_pass": not errors and not overlaps,
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


def metrics(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prepared = list(records)
    aggregate = stage_a.evaluate(model, prepared)
    by_regime: Dict[str, List[int]] = defaultdict(list)
    by_length: Dict[str, List[int]] = defaultdict(list)
    for index, item in enumerate(prepared):
        by_regime[str(item["metadata"]["regime"])].append(index)
        by_length[str(item["metadata"]["output_length"])].append(index)
    def summarize(indices: Sequence[int]) -> Dict[str, Any]:
        values = [prepared[index] for index in indices]
        return {"examples": len(values), **stage_a.evaluate(model, values)}
    return {
        "aggregate": aggregate,
        "by_regime": {key: summarize(indices) for key, indices in sorted(by_regime.items())},
        "by_output_length": {key: summarize(indices) for key, indices in sorted(by_length.items())},
    }


def gate_status(train_metrics: Mapping[str, Any], validation_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    train_accuracy = float(train_metrics["aggregate"]["teacher_forced_exact_accuracy"])
    validation_accuracy = float(validation_metrics["aggregate"]["teacher_forced_exact_accuracy"])
    return {
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "train_threshold": TRAIN_THRESHOLD,
        "validation_threshold": VALIDATION_THRESHOLD,
        "train_pass": train_accuracy >= TRAIN_THRESHOLD,
        "validation_pass": validation_accuracy >= VALIDATION_THRESHOLD,
        "promotion_pass": train_accuracy >= TRAIN_THRESHOLD and validation_accuracy >= VALIDATION_THRESHOLD,
    }


def run_curriculum(seed: int, records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], output: Path) -> Dict[str, Any]:
    torch = stage_a.torch
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    stage_results: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for stage_index, stage_name in enumerate(STAGES, start=1):
        prepared = {split: prepare(values) for split, values in records[stage_name].items()}
        order = list(range(len(prepared["train"])))
        random.Random(stable_seed(seed, "curriculum", stage_name, "order")).shuffle(order)
        stage_started = time.perf_counter()
        for local_step in range(1, STAGE_STEPS + 1):
            stage_a.train_step(model, optimizer, prepared["train"], stage_a.batch_indices(order, local_step))
        train_metrics = metrics(model, prepared["train"])
        validation_metrics = metrics(model, prepared["validation"])
        gate = gate_status(train_metrics, validation_metrics)
        global_step = stage_index * STAGE_STEPS
        checkpoint = output / "checkpoints" / f"step_{global_step:04d}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        stage_a.save_checkpoint(checkpoint, model, optimizer, global_step, order)
        stage_results.append({
            "stage": stage_name,
            "global_step": global_step,
            "elapsed_seconds": time.perf_counter() - stage_started,
            "gate": gate,
            "train": train_metrics,
            "validation": validation_metrics,
            "checkpoint": str(checkpoint.relative_to(output)),
        })
        print(json.dumps({"arm": "curriculum", "seed": seed, "stage": stage_name, "step": global_step, "train_exact": gate["train_accuracy"], "validation_exact": gate["validation_accuracy"], "promotion_pass": gate["promotion_pass"]}, sort_keys=True), flush=True)
    result = {"arm": "curriculum", "seed": seed, "steps": len(STAGES) * STAGE_STEPS, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "elapsed_seconds": time.perf_counter() - started, "stages": stage_results}
    write_json(output / "RESULTS.json", result)
    write_json(output / "RUN_CONFIG.json", {"experiment": "crucible_staged_arithmetic_curriculum_20261024", "arm": "curriculum", "seed": seed, "stage_steps": STAGE_STEPS, "total_steps": len(STAGES) * STAGE_STEPS, "stage_gates": {"train": TRAIN_THRESHOLD, "validation": VALIDATION_THRESHOLD}, "gpu_used": False, "test_evaluated": False})
    return result


def run_direct(seed: int, records: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> Dict[str, Any]:
    torch = stage_a.torch
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    prepared = {split: prepare(values) for split, values in records.items()}
    order = list(range(len(prepared["train"])))
    random.Random(stable_seed(seed, "direct", "order")).shuffle(order)
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, DIRECT_STEPS + 1):
        train_info = stage_a.train_step(model, optimizer, prepared["train"], stage_a.batch_indices(order, step))
        if step in CHECKPOINT_STEPS[1:]:
            train_metrics = metrics(model, prepared["train"])
            validation_metrics = metrics(model, prepared["validation"])
            rows.append({"step": step, "train_step": train_info, "train": train_metrics, "validation": validation_metrics})
            print(json.dumps({"arm": "direct", "seed": seed, "step": step, "train_exact": train_metrics["aggregate"]["teacher_forced_exact_accuracy"], "validation_exact": validation_metrics["aggregate"]["teacher_forced_exact_accuracy"]}, sort_keys=True), flush=True)
            checkpoint = output / "checkpoints" / f"step_{step:04d}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            stage_a.save_checkpoint(checkpoint, model, optimizer, step, order)
    result = {"arm": "direct", "seed": seed, "steps": DIRECT_STEPS, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "elapsed_seconds": time.perf_counter() - started, "rows": rows, "final": rows[-1]}
    write_json(output / "RESULTS.json", result)
    write_json(output / "RUN_CONFIG.json", {"experiment": "crucible_staged_arithmetic_curriculum_20261024", "arm": "direct", "seed": seed, "total_steps": DIRECT_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "gpu_used": False, "test_evaluated": False})
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
    records = build_records()
    audit_result = audit(records)
    successor_reference = {
        "path": str(SUCCESSOR_CHECKPOINT.relative_to(ROOT)),
        "exists": SUCCESSOR_CHECKPOINT.is_file(),
        "sha256": sha256_file(SUCCESSOR_CHECKPOINT) if SUCCESSOR_CHECKPOINT.is_file() else None,
    }
    if not audit_result["all_checks_pass"] or not successor_reference["exists"]:
        raise SystemExit(f"preflight failed: {json.dumps({'audit': audit_result, 'successor_reference': successor_reference}, sort_keys=True)}")
    write_json(output / "RUN_CONFIG.json", {"experiment": "crucible_staged_arithmetic_curriculum_20261024", "status": "frozen_before_training", "arms": ["curriculum", "direct"], "seeds": list(SEEDS), "stage_steps": STAGE_STEPS, "direct_steps": DIRECT_STEPS, "stage_gates": {"train": TRAIN_THRESHOLD, "validation": VALIDATION_THRESHOLD}, "gpu_used": False, "test_evaluated": False, "successor_reference": successor_reference})
    write_json(output / "PREFLIGHT.json", {"audit": audit_result, "successor_reference": successor_reference})
    for stage_name, splits in records.items():
        stage_dir = output / "data" / stage_name
        write_jsonl(stage_dir / "train.jsonl", splits["train"])
        write_jsonl(stage_dir / "validation.jsonl", splits["validation"])
    results: List[Dict[str, Any]] = []
    for seed in SEEDS:
        results.append(run_curriculum(seed, records, output / "curriculum" / f"seed{seed}"))
        gc.collect()
        results.append(run_direct(seed, records[STAGES[-1]], output / "direct" / f"seed{seed}"))
        gc.collect()
    write_json(output / "SUMMARY.json", {"experiment": "crucible_staged_arithmetic_curriculum_20261024", "status": "completed", "test_evaluated": False, "preflight_pass": True, "successor_reference": successor_reference, "results": [{"arm": item["arm"], "seed": item["seed"], "steps": item["steps"], "elapsed_seconds": item["elapsed_seconds"]} for item in results]})
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"output": str(output), "status": "completed", "preflight_pass": True, "arms": len(results)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
