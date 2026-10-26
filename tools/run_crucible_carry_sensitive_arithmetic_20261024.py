#!/usr/bin/env python3
"""Run the CPU-only carry-sensitive two-digit addition gate."""

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
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_carry_sensitive_arithmetic_20261024"

SEEDS = (42, 1337)
ARMS = ("canonical_numeric", "factorized_digits")
STEPS = 1_000
CHECKPOINT_STEPS = (0, 250, 500, 750, 1_000)
ANSWER_MARKER = b"\nAnswer:"
TRAIN_PER_REGIME = 64
VALIDATION_PER_REGIME = 24


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_a = import_module("stage_a_for_carry_gate", STAGE_A_PATH)


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


def regime(a: int, b: int) -> str:
    ones = (a % 10) + (b % 10)
    tens = (a // 10) + (b // 10) + (1 if ones >= 10 else 0)
    if ones < 10 and tens < 10:
        return "no_carry"
    if ones >= 10 and tens < 10:
        return "ones_carry"
    if ones >= 10 and tens >= 10:
        return "cascading_carry"
    if ones < 10 and tens >= 10:
        return "tens_carry_only"
    raise AssertionError((a, b, ones, tens))


def output_length(a: int, b: int) -> int:
    return len(str(a + b))


def make_record(arm: str, split: str, index: int, a: int, b: int) -> Dict[str, Any]:
    current_regime = regime(a, b)
    if arm == "canonical_numeric":
        prompt = f"Add two decimal integers. a={a}; b={b}. Return digits only."
    elif arm == "factorized_digits":
        prompt = (
            "Add two decimal integers. "
            f"a_tens={a // 10}; a_ones={a % 10}; "
            f"b_tens={b // 10}; b_ones={b % 10}. Return digits only."
        )
    else:
        raise ValueError(arm)
    answer = str(a + b)
    return {
        "schema_version": "crucible_carry_sensitive_arithmetic_v1",
        "record_id": hashlib.sha256(f"carry|{arm}|{split}|{index}|{a}|{b}".encode()).hexdigest()[:20],
        "arm": arm,
        "split": split,
        "input": prompt,
        "target": answer,
        "oracle": {"kind": "decimal_addition", "value": int(answer)},
        "metadata": {
            "a": a,
            "b": b,
            "regime": current_regime,
            "output_length": len(answer),
            "a_tens": a // 10,
            "a_ones": a % 10,
            "b_tens": b // 10,
            "b_ones": b % 10,
        },
    }


def all_pairs_by_regime() -> Dict[str, List[Tuple[int, int]]]:
    values: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for a in range(10, 100):
        for b in range(10, 100):
            current_regime = regime(a, b)
            if current_regime in {"no_carry", "ones_carry", "cascading_carry"}:
                values[current_regime].append((a, b))
    return {key: sorted(value) for key, value in values.items()}


def choose_splits() -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
    candidates = all_pairs_by_regime()
    result: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
    for current_regime, pairs in candidates.items():
        shuffled = list(pairs)
        random.Random(stable_seed("carry-split", current_regime)).shuffle(shuffled)
        train = shuffled[:TRAIN_PER_REGIME]
        validation = shuffled[TRAIN_PER_REGIME : TRAIN_PER_REGIME + VALIDATION_PER_REGIME]
        if len(validation) != VALIDATION_PER_REGIME:
            raise RuntimeError(f"not enough pairs for {current_regime}")
        result[current_regime] = {"train": sorted(train), "validation": sorted(validation)}
    return result


def build_records() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    splits = choose_splits()
    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for arm in ARMS:
        train_pairs = [pair for current_regime in sorted(splits) for pair in splits[current_regime]["train"]]
        validation_pairs = [pair for current_regime in sorted(splits) for pair in splits[current_regime]["validation"]]
        result[arm] = {
            "train": [make_record(arm, "train", index, a, b) for index, (a, b) in enumerate(train_pairs)],
            "validation": [make_record(arm, "validation", index, a, b) for index, (a, b) in enumerate(validation_pairs)],
        }
    return result


def preflight(records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> Dict[str, Any]:
    errors: List[str] = []
    overlaps: List[str] = []
    coverage: Dict[str, List[int]] = {}
    regime_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    oracle_accuracy: Dict[str, float] = {}
    for arm, splits in records.items():
        train_pairs = {(int(item["metadata"]["a"]), int(item["metadata"]["b"])) for item in splits["train"]}
        validation_pairs = {(int(item["metadata"]["a"]), int(item["metadata"]["b"])) for item in splits["validation"]}
        overlaps.extend([str(pair) for pair in sorted(train_pairs & validation_pairs)])
        regime_counts[arm] = {}
        for split, values in splits.items():
            correct = 0
            counts: Dict[str, int] = defaultdict(int)
            for item in values:
                a = int(item["metadata"]["a"])
                b = int(item["metadata"]["b"])
                expected = a + b
                if int(item["target"]) != expected or int(item["oracle"]["value"]) != expected:
                    errors.append(str(item["record_id"]))
                else:
                    correct += 1
                counts[str(item["metadata"]["regime"])] += 1
            regime_counts[arm][split] = dict(sorted(counts.items()))
            oracle_accuracy[f"{arm}:{split}"] = correct / len(values)
        for position in ("a_tens", "a_ones", "b_tens", "b_ones"):
            coverage[f"{arm}:train:{position}"] = sorted({int(item["metadata"][position]) for item in splits["train"]})
    required_regimes = {"no_carry", "ones_carry", "cascading_carry"}
    all_regimes_present = all(set(regime_counts[arm]["train"]) == required_regimes and set(regime_counts[arm]["validation"]) == required_regimes for arm in records)
    all_digits_present = all(
        set(values) == (set(range(1, 10)) if position in {"a_tens", "b_tens"} else set(range(10)))
        for key, values in coverage.items()
        if ":train:" in key
        for position in [key.rsplit(":", 1)[-1]]
    )
    return {
        "oracle_errors": sorted(set(errors)),
        "train_validation_pair_overlaps": overlaps,
        "oracle_accuracy": oracle_accuracy,
        "regime_counts": regime_counts,
        "feature_coverage": coverage,
        "all_regimes_present": all_regimes_present,
        "all_digits_present_in_training": all_digits_present,
        "all_checks_pass": not errors and not overlaps and all(value == 1.0 for value in oracle_accuracy.values()) and all_regimes_present and all_digits_present,
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


def exact_metrics(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prepared = list(records)
    aggregate = stage_a.evaluate(model, prepared)
    by_regime: Dict[str, List[int]] = defaultdict(list)
    by_length: Dict[str, List[int]] = defaultdict(list)
    for index, item in enumerate(prepared):
        by_regime[str(item["metadata"]["regime"])].append(index)
        by_length[str(item["metadata"]["output_length"])].append(index)
    def summarize(indices: Sequence[int]) -> Dict[str, Any]:
        values = [prepared[index] for index in indices]
        result = stage_a.evaluate(model, values)
        return {"examples": len(values), **result}
    return {
        "aggregate": aggregate,
        "by_regime": {key: summarize(indices) for key, indices in sorted(by_regime.items())},
        "by_output_length": {key: summarize(indices) for key, indices in sorted(by_length.items())},
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
            train_metrics = exact_metrics(model, prepared["train"])
            validation_metrics = exact_metrics(model, prepared["validation"])
            rows.append({"step": step, "train_step": train_info, "train": train_metrics, "validation": validation_metrics})
            print(json.dumps({"arm": arm, "seed": seed, "step": step, "train_exact": train_metrics["aggregate"]["teacher_forced_exact_accuracy"], "validation_exact": validation_metrics["aggregate"]["teacher_forced_exact_accuracy"], "validation_by_regime": {key: value["teacher_forced_exact_accuracy"] for key, value in validation_metrics["by_regime"].items()}}, sort_keys=True), flush=True)
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
    write_json(output / "RUN_CONFIG.json", {"experiment": "crucible_carry_sensitive_arithmetic_20261024", "arm": arm, "seed": seed, "steps": STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "gpu_used": False, "test_evaluated": False})
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
    records = build_records()
    audit = preflight(records)
    if not audit["all_checks_pass"]:
        raise SystemExit(f"preflight failed: {json.dumps(audit, sort_keys=True)}")
    write_json(output / "RUN_CONFIG.json", {"experiment": "crucible_carry_sensitive_arithmetic_20261024", "status": "frozen_before_training", "arms": list(ARMS), "seeds": list(SEEDS), "steps": STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "gpu_used": False, "test_evaluated": False, "positive_control": "deterministic_decimal_addition"})
    write_json(output / "PREFLIGHT.json", audit)
    results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            results.append(run_arm(seed, arm, records[arm], output / arm / f"seed{seed}"))
            gc.collect()
    summary = {
        "experiment": "crucible_carry_sensitive_arithmetic_20261024",
        "status": "completed",
        "test_evaluated": False,
        "preflight_pass": audit["all_checks_pass"],
        "results": [{"arm": item["arm"], "seed": item["seed"], "train_exact": item["final_train"]["aggregate"]["teacher_forced_exact_accuracy"], "validation_exact": item["final_validation"]["aggregate"]["teacher_forced_exact_accuracy"], "validation_by_regime": {key: value["teacher_forced_exact_accuracy"] for key, value in item["final_validation"]["by_regime"].items()}, "elapsed_seconds": item["elapsed_seconds"]} for item in results],
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
