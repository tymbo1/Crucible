#!/usr/bin/env python3
"""Run the CPU-only single-digit to two-digit addition competence ladder."""

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
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_digitwise_addition_ladder_20261025"

SEEDS = (42, 1337)
INTERFACES = ("canonical", "factorized")
STAGES = ("single_digit_no_carry", "two_digit_no_carry")
STAGE_STEPS = 500
TRAIN_THRESHOLD = 0.95
VALIDATION_THRESHOLD = 0.90
ANSWER_MARKER = b"\nAnswer:"


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


stage_a = import_module("stage_a_for_digitwise_ladder", STAGE_A_PATH)


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


def make_record(interface: str, stage: str, split: str, index: int, a: int, b: int) -> Dict[str, Any]:
    if interface == "canonical":
        prompt = f"Add integers with no carry. a={a}; b={b}. Return digits only."
    elif interface == "factorized":
        if stage == "single_digit_no_carry":
            prompt = f"Add integers with no carry. a_digit={a}; b_digit={b}. Return digits only."
        else:
            prompt = f"Add integers with no carry. a_tens={a // 10}; a_ones={a % 10}; b_tens={b // 10}; b_ones={b % 10}. Return digits only."
    else:
        raise ValueError(interface)
    answer = str(a + b)
    return {
        "schema_version": "crucible_digitwise_addition_ladder_v1",
        "record_id": hashlib.sha256(f"digitwise|{interface}|{stage}|{split}|{index}|{a}|{b}".encode()).hexdigest()[:20],
        "interface": interface,
        "stage": stage,
        "split": split,
        "input": prompt,
        "target": answer,
        "oracle": {"kind": "decimal_addition", "value": int(answer)},
        "metadata": {
            "a": a,
            "b": b,
            "answer": int(answer),
            "output_length": len(answer),
            "a_tens": a // 10,
            "a_ones": a % 10,
            "b_tens": b // 10,
            "b_ones": b % 10,
            "single_sum": a + b,
        },
    }


def choose_single_digit_pairs() -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    candidates = [(a, b) for a in range(10) for b in range(10) if a + b < 10]
    shuffled = list(candidates)
    random.Random(stable_seed("digitwise", "single")).shuffle(shuffled)
    selected: List[Tuple[int, int]] = []
    required = [("a", value) for value in range(10)] + [("b", value) for value in range(10)] + [("sum", value) for value in range(10)]
    for feature, value in required:
        for pair in shuffled:
            a, b = pair
            observed = a if feature == "a" else b if feature == "b" else a + b
            if observed == value and pair not in selected:
                selected.append(pair)
                break
    remaining = [pair for pair in shuffled if pair not in selected]
    selected.extend(remaining[: max(0, 40 - len(selected))])
    train = sorted(selected[:40])
    validation = sorted([pair for pair in candidates if pair not in set(train)])
    if len(validation) > 15:
        validation = validation[:15]
    return train, validation


def choose_two_digit_pairs() -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    candidates = [(a, b) for a in range(10, 100) for b in range(10, 100) if (a % 10) + (b % 10) < 10 and (a // 10) + (b // 10) < 10]
    shuffled = list(candidates)
    random.Random(stable_seed("digitwise", "two")).shuffle(shuffled)
    selected: List[Tuple[int, int]] = []
    required = []
    for field in ("a_tens", "a_ones", "b_tens", "b_ones"):
        legal = range(1, 10) if field in {"a_tens", "b_tens"} else range(10)
        required.extend((field, value) for value in legal)
    required.extend(("ones_sum", value) for value in range(10))
    required.extend(("tens_sum", value) for value in range(2, 10))
    for field, value in required:
        for pair in shuffled:
            a, b = pair
            features = {
                "a_tens": a // 10,
                "a_ones": a % 10,
                "b_tens": b // 10,
                "b_ones": b % 10,
                "ones_sum": (a % 10) + (b % 10),
                "tens_sum": (a // 10) + (b // 10),
            }
            if features[field] == value and pair not in selected:
                selected.append(pair)
                break
    remaining = [pair for pair in shuffled if pair not in selected]
    selected.extend(remaining[: max(0, 64 - len(selected))])
    train = sorted(selected[:64])
    validation = sorted([pair for pair in candidates if pair not in set(train)])[:24]
    return train, validation


def build_records() -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    single_train, single_validation = choose_single_digit_pairs()
    two_train, two_validation = choose_two_digit_pairs()
    pairs = {
        "single_digit_no_carry": (single_train, single_validation),
        "two_digit_no_carry": (two_train, two_validation),
    }
    result: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    for interface in INTERFACES:
        result[interface] = {}
        for stage in STAGES:
            train_pairs, validation_pairs = pairs[stage]
            result[interface][stage] = {
                "train": [make_record(interface, stage, "train", i, a, b) for i, (a, b) in enumerate(train_pairs)],
                "validation": [make_record(interface, stage, "validation", i, a, b) for i, (a, b) in enumerate(validation_pairs)],
            }
    return result


def audit(records: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]]) -> Dict[str, Any]:
    errors: List[str] = []
    overlaps: List[str] = []
    summaries: Dict[str, Any] = {}
    coverage: Dict[str, Any] = {}
    for interface, stages in records.items():
        summaries[interface] = {}
        for stage, splits in stages.items():
            train_pairs = {(int(item["metadata"]["a"]), int(item["metadata"]["b"])) for item in splits["train"]}
            valid_pairs = {(int(item["metadata"]["a"]), int(item["metadata"]["b"])) for item in splits["validation"]}
            overlaps.extend([f"{interface}:{stage}:{pair}" for pair in sorted(train_pairs & valid_pairs)])
            for split, values in splits.items():
                for item in values:
                    a = int(item["metadata"]["a"])
                    b = int(item["metadata"]["b"])
                    if a + b != int(item["target"]) or a + b != int(item["oracle"]["value"]):
                        errors.append(str(item["record_id"]))
            summaries[interface][stage] = {"train_examples": len(splits["train"]), "validation_examples": len(splits["validation"]), "train_output_lengths": sorted({int(item["metadata"]["output_length"]) for item in splits["train"]}), "validation_output_lengths": sorted({int(item["metadata"]["output_length"]) for item in splits["validation"]})}
            if stage == "single_digit_no_carry":
                train = splits["train"]
                coverage[f"{interface}:{stage}"] = {
                    "a": sorted({int(item["metadata"]["a"]) for item in train}),
                    "b": sorted({int(item["metadata"]["b"]) for item in train}),
                    "sum": sorted({int(item["metadata"]["single_sum"]) for item in train}),
                }
            else:
                train = splits["train"]
                coverage[f"{interface}:{stage}"] = {
                    "a_tens": sorted({int(item["metadata"]["a_tens"]) for item in train}),
                    "a_ones": sorted({int(item["metadata"]["a_ones"]) for item in train}),
                    "b_tens": sorted({int(item["metadata"]["b_tens"]) for item in train}),
                    "b_ones": sorted({int(item["metadata"]["b_ones"]) for item in train}),
                    "ones_sum": sorted({int(item["metadata"]["a_ones"]) + int(item["metadata"]["b_ones"]) for item in train}),
                    "tens_sum": sorted({int(item["metadata"]["a_tens"]) + int(item["metadata"]["b_tens"]) for item in train}),
                }
    coverage_pass = True
    for key, values in coverage.items():
        if key.endswith("single_digit_no_carry"):
            coverage_pass = coverage_pass and values["a"] == list(range(10)) and values["b"] == list(range(10)) and values["sum"] == list(range(10))
        else:
            coverage_pass = coverage_pass and values["a_tens"] == list(range(1, 9)) and values["b_tens"] == list(range(1, 9)) and values["a_ones"] == list(range(10)) and values["b_ones"] == list(range(10)) and values["ones_sum"] == list(range(10)) and values["tens_sum"] == list(range(2, 10))
    return {"oracle_errors": sorted(set(errors)), "train_validation_pair_overlaps": overlaps, "summaries": summaries, "coverage": coverage, "coverage_pass": coverage_pass, "all_checks_pass": not errors and not overlaps and coverage_pass}


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
    by_length: Dict[str, List[int]] = defaultdict(list)
    by_sum: Dict[str, List[int]] = defaultdict(list)
    for index, item in enumerate(prepared):
        by_length[str(item["metadata"]["output_length"])].append(index)
        by_sum[str(item["metadata"]["single_sum"])].append(index)
    def summarize(indices: Sequence[int]) -> Dict[str, Any]:
        values = [prepared[index] for index in indices]
        return {"examples": len(values), **stage_a.evaluate(model, values)}
    return {"aggregate": aggregate, "by_output_length": {key: summarize(value) for key, value in sorted(by_length.items())}, "by_sum": {key: summarize(value) for key, value in sorted(by_sum.items())}}


def train_model(seed: int, interface: str, stage: str, records: Mapping[str, Sequence[Mapping[str, Any]]], output: Path, steps: int, initial_checkpoint: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    torch = stage_a.torch
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["model"], strict=True)
        optimizer.load_state_dict(initial_checkpoint["optimizer"])
    prepared = {split: prepare(values) for split, values in records.items()}
    order = list(range(len(prepared["train"])))
    random.Random(stable_seed(seed, interface, stage, "order")).shuffle(order)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        stage_a.train_step(model, optimizer, prepared["train"], stage_a.batch_indices(order, step))
    train_metrics = metrics(model, prepared["train"])
    validation_metrics = metrics(model, prepared["validation"])
    checkpoint = output / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint, model, optimizer, steps, order)
    result = {
        "seed": seed,
        "interface": interface,
        "stage": stage,
        "steps": steps,
        "elapsed_seconds": time.perf_counter() - started,
        "train": train_metrics,
        "validation": validation_metrics,
        "train_exact": train_metrics["aggregate"]["teacher_forced_exact_accuracy"],
        "validation_exact": validation_metrics["aggregate"]["teacher_forced_exact_accuracy"],
        "fit_pass": train_metrics["aggregate"]["teacher_forced_exact_accuracy"] >= TRAIN_THRESHOLD,
        "advancement_pass": train_metrics["aggregate"]["teacher_forced_exact_accuracy"] >= TRAIN_THRESHOLD and validation_metrics["aggregate"]["teacher_forced_exact_accuracy"] >= VALIDATION_THRESHOLD,
        "checkpoint": str(checkpoint),
    }
    write_json(output / "RESULTS.json", result)
    write_json(output / "RUN_CONFIG.json", {"experiment": "crucible_digitwise_addition_ladder_20261025", "interface": interface, "stage": stage, "seed": seed, "steps": steps, "gpu_used": False, "test_evaluated": False})
    write_jsonl(output / "train.jsonl", records["train"])
    write_jsonl(output / "validation.jsonl", records["validation"])
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
    records = build_records()
    audit_result = audit(records)
    if not audit_result["all_checks_pass"]:
        raise SystemExit(f"preflight failed: {json.dumps(audit_result, sort_keys=True)}")
    write_json(output / "RUN_CONFIG.json", {"experiment": "crucible_digitwise_addition_ladder_20261025", "status": "frozen_before_training", "interfaces": list(INTERFACES), "stages": list(STAGES), "seeds": list(SEEDS), "stage_steps": STAGE_STEPS, "thresholds": {"train": TRAIN_THRESHOLD, "validation": VALIDATION_THRESHOLD}, "gpu_used": False, "test_evaluated": False})
    write_json(output / "PREFLIGHT.json", audit_result)
    for interface in INTERFACES:
        for stage in STAGES:
            stage_dir = output / "data" / interface / stage
            write_jsonl(stage_dir / "train.jsonl", records[interface][stage]["train"])
            write_jsonl(stage_dir / "validation.jsonl", records[interface][stage]["validation"])
    results: List[Dict[str, Any]] = []
    stage1_results: List[Dict[str, Any]] = []
    for interface in INTERFACES:
        for seed in SEEDS:
            direct_dir = output / "direct" / interface / f"seed{seed}" / STAGES[0]
            curriculum_dir = output / "curriculum" / interface / f"seed{seed}" / STAGES[0]
            direct_result = train_model(seed, interface, STAGES[0], records[interface][STAGES[0]], direct_dir, STAGE_STEPS)
            curriculum_result = train_model(seed, interface, STAGES[0], records[interface][STAGES[0]], curriculum_dir, STAGE_STEPS)
            direct_result["arm"] = "direct"
            curriculum_result["arm"] = "curriculum"
            results.extend([direct_result, curriculum_result])
            stage1_results.extend([direct_result, curriculum_result])
    stage1_pass = all(item["advancement_pass"] for item in stage1_results)
    stage2_ran = False
    if stage1_pass:
        stage2_ran = True
        for interface in INTERFACES:
            for seed in SEEDS:
                curriculum_checkpoint_path = output / "curriculum" / interface / f"seed{seed}" / STAGES[0] / "checkpoint.pt"
                checkpoint = stage_a.torch.load(curriculum_checkpoint_path, map_location="cpu", weights_only=False)
                curriculum_dir = output / "curriculum" / interface / f"seed{seed}" / STAGES[1]
                curriculum_result = train_model(seed, interface, STAGES[1], records[interface][STAGES[1]], curriculum_dir, STAGE_STEPS, checkpoint)
                curriculum_result["arm"] = "curriculum"
                curriculum_result["total_steps"] = 2 * STAGE_STEPS
                direct_dir = output / "direct" / interface / f"seed{seed}" / STAGES[1]
                direct_result = train_model(seed, interface, STAGES[1], records[interface][STAGES[1]], direct_dir, 2 * STAGE_STEPS)
                direct_result["arm"] = "direct"
                direct_result["total_steps"] = 2 * STAGE_STEPS
                results.extend([curriculum_result, direct_result])
    summary = {
        "experiment": "crucible_digitwise_addition_ladder_20261025",
        "status": "completed",
        "preflight_pass": True,
        "stage1_advancement_pass": stage1_pass,
        "stage2_ran": stage2_ran,
        "test_evaluated": False,
        "results": [{"arm": item["arm"], "interface": item["interface"], "stage": item["stage"], "seed": item["seed"], "steps": item["steps"], "train_exact": item["train_exact"], "validation_exact": item["validation_exact"], "fit_pass": item["fit_pass"], "advancement_pass": item["advancement_pass"], "elapsed_seconds": item["elapsed_seconds"]} for item in results],
    }
    write_json(output / "SUMMARY.json", summary)
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"output": str(output), "status": "completed", "preflight_pass": True, "stage1_advancement_pass": stage1_pass, "stage2_ran": stage2_ran, "arms": len(results)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
