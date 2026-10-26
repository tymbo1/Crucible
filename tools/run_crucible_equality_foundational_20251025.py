#!/usr/bin/env python3
"""Run the bounded equality foundational-learning experiment on CPU."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_equality_foundational_20251025_v2"
SOURCE = ROOT / "tools" / "run_crucible_digitwise_diagonal_ablation_20251025.py"
INTERFACES = ("canonical", "factorized")
SEEDS = (2025, 2026)
ARMS = ("A0_original", "A1_equality_pretrained", "A2_explicit_equality")
ADDITION_STAGE = "single_digit_no_carry"
PRETRAIN_STAGE = "equality_pretraining"
ADDITION_STEPS = 500
PRETRAIN_STEPS = 500


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


base = import_module("equality_foundational_diagonal_runner", SOURCE)
ladder = base.ladder
stage_a = base.stage_a
torch = base.torch
PAIR_SET = set(base.PAIR_SET)
ADDITION_VALIDATION = set(base.VALIDATION_PAIRS["diagonal"])
ADDITION_BACKGROUND = tuple(sorted(ADDITION_VALIDATION - base.DIAGONAL_SPECIAL))
ADDITION_VALIDATION = tuple(sorted(set(ADDITION_BACKGROUND) | base.DIAGONAL_SPECIAL))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def addition_records(interface: str, arm: str) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    validation = set(ADDITION_VALIDATION)
    for split, selected in (("train", sorted(PAIR_SET - validation)), ("validation", sorted(validation))):
        values: List[Dict[str, Any]] = []
        for index, (a, b) in enumerate(selected):
            record = ladder.make_record(interface, ADDITION_STAGE, split, index, a, b)
            record["condition"] = arm
            record["partition"] = "diagonal"
            record["record_id"] = hashlib.sha256(f"equality_foundational|{arm}|{interface}|{split}|{index}|{a}|{b}".encode()).hexdigest()[:20]
            if arm == "A2_explicit_equality":
                record["input"] = record["input"].replace("Return digits only.", f"equal={int(a == b)}. Return digits only.")
                record["metadata"]["equality_feature"] = int(a == b)
            values.append(record)
        result[split] = values
    return result


def equality_split() -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    equal = [(digit, digit) for digit in range(10)]
    unequal = [(a, b) for a in range(10) for b in range(10) if a != b]
    rng = random.Random(20251025)
    rng.shuffle(equal)
    rng.shuffle(unequal)
    return sorted(equal[:8] + unequal[:72]), sorted(equal[8:] + unequal[72:])


def equality_records(interface: str) -> Dict[str, List[Dict[str, Any]]]:
    train_pairs, validation_pairs = equality_split()
    result: Dict[str, List[Dict[str, Any]]] = {}
    for split, selected in (("train", train_pairs), ("validation", validation_pairs)):
        values: List[Dict[str, Any]] = []
        for index, (a, b) in enumerate(selected):
            if interface == "canonical":
                prompt = f"Judge digit equality. a={a}; b={b}. Return 1 if equal and 0 otherwise."
            else:
                prompt = f"Judge digit equality. a_digit={a}; b_digit={b}. Return 1 if equal and 0 otherwise."
            target = str(int(a == b))
            values.append({
                "schema_version": "crucible_equality_pretraining_v1",
                "record_id": hashlib.sha256(f"equality_pretraining|{interface}|{split}|{index}|{a}|{b}".encode()).hexdigest()[:20],
                "interface": interface,
                "stage": PRETRAIN_STAGE,
                "split": split,
                "input": prompt,
                "target": target,
                "oracle": {"kind": "digit_equality", "value": int(a == b)},
                "metadata": {
                    "a": a,
                    "b": b,
                    "answer": int(a == b),
                    "equality": int(a == b),
                    "output_length": 1,
                    "a_tens": 0,
                    "a_ones": a,
                    "b_tens": 0,
                    "b_ones": b,
                    "single_sum": int(a == b),
                },
            })
        result[split] = values
    return result


def audit_addition(records: Mapping[str, Sequence[Mapping[str, Any]]], arm: str) -> List[str]:
    errors: List[str] = []
    train = {(int(record["metadata"]["a"]), int(record["metadata"]["b"])) for record in records["train"]}
    validation = {(int(record["metadata"]["a"]), int(record["metadata"]["b"])) for record in records["validation"]}
    if len(train) != 40 or len(validation) != 15 or train & validation or train | validation != PAIR_SET:
        errors.append(f"addition_split:{arm}")
    if {(2, 2), (3, 3)} - validation:
        errors.append(f"special_diagonal:{arm}")
    for split in ("train", "validation"):
        for record in records[split]:
            a, b = int(record["metadata"]["a"]), int(record["metadata"]["b"])
            if str(record["target"]) != str(a + b) or int(record["oracle"]["value"]) != a + b:
                errors.append(f"addition_label:{arm}:{record['record_id']}")
            if arm == "A2_explicit_equality" and f"equal={int(a == b)}" not in record["input"]:
                errors.append(f"missing_feature:{record['record_id']}")
            if arm != "A2_explicit_equality" and "equal=" in record["input"]:
                errors.append(f"unexpected_feature:{record['record_id']}")
    return errors


def audit_equality(records: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[str]:
    errors: List[str] = []
    train = {record["record_id"] for record in records["train"]}
    validation = {record["record_id"] for record in records["validation"]}
    if len(train) != 80 or len(validation) != 20 or train & validation:
        errors.append("equality_split")
    for split in ("train", "validation"):
        for record in records[split]:
            a, b = int(record["metadata"]["a"]), int(record["metadata"]["b"])
            if str(record["target"]) != str(int(a == b)) or int(record["oracle"]["value"]) != int(a == b):
                errors.append(f"equality_label:{record['record_id']}")
    return errors


def train_model(seed: int, interface: str, stage: str, records: Mapping[str, Sequence[Mapping[str, Any]]], output: Path, steps: int, initial_model: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    stage_a.torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    if initial_model is not None:
        model.load_state_dict(initial_model, strict=True)
    prepared = {split: ladder.prepare(values) for split, values in records.items()}
    order = list(range(len(prepared["train"])))
    random.Random(ladder.stable_seed(seed, interface, stage, "order")).shuffle(order)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        stage_a.train_step(model, optimizer, prepared["train"], stage_a.batch_indices(order, step))
    train_metrics = ladder.metrics(model, prepared["train"])
    validation_metrics = ladder.metrics(model, prepared["validation"])
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
        "fit_pass": train_metrics["aggregate"]["teacher_forced_exact_accuracy"] >= ladder.TRAIN_THRESHOLD,
        "checkpoint": str(checkpoint),
    }
    write_json(output / "RESULTS.json", result)
    write_json(output / "RUN_CONFIG.json", {"experiment": "crucible_equality_foundational_20251025", "interface": interface, "stage": stage, "seed": seed, "steps": steps, "gpu_used": False, "test_evaluated": False})
    for split, values in records.items():
        (output / f"{split}.jsonl").write_text("\n".join(json.dumps(dict(value), sort_keys=True) for value in values) + "\n", encoding="utf-8")
    del model, optimizer
    gc.collect()
    return result


def stat(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    output = sum(bool(row["output_token_correct"]) for row in rows)
    eos = sum(bool(row["eos_correct"]) for row in rows)
    full = sum(bool(row["full_sequence_correct"]) for row in rows)
    return {"observations": n, "output_token_correct": output, "output_token_accuracy": output / n if n else None, "eos_correct": eos, "eos_accuracy": eos / n if n else None, "full_sequence_correct": full, "full_sequence_accuracy": full / n if n else None, "output_format_errors": sum(bool(row["output_format_error"]) for row in rows)}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base.OUTPUT = OUTPUT
    base.ANCHOR_DIR = OUTPUT / "generated_reference_e4" / "crucible_manifest"
    anchor_hashes = base.prepare_fresh_anchors()

    addition_by_arm: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    equality_by_interface: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    errors: List[str] = []
    for arm in ARMS:
        addition_by_arm[arm] = {interface: addition_records(interface, arm) for interface in INTERFACES}
        errors.extend(audit_addition(addition_by_arm[arm]["canonical"], arm))
    for interface in INTERFACES:
        equality_by_interface[interface] = equality_records(interface)
        errors.extend(audit_equality(equality_by_interface[interface]))
    preflight = {
        "experiment": "crucible_equality_foundational_20251025",
        "status": "frozen_before_training",
        "arms": list(ARMS),
        "interfaces": list(INTERFACES),
        "seeds": list(SEEDS),
        "addition_validation_pairs": [list(pair) for pair in sorted(ADDITION_VALIDATION)],
        "addition_special_diagonal_pairs": [[2, 2], [3, 3]],
        "equality_split": {interface: {split: [[int(record["metadata"]["a"]), int(record["metadata"]["b"])] for record in values] for split, values in records.items()} for interface, records in equality_by_interface.items()},
        "equality_steps": PRETRAIN_STEPS,
        "addition_steps": ADDITION_STEPS,
        "anchor_sha256": anchor_hashes,
        "architecture_changed": False,
        "sealed_test_accessed": False,
        "errors": errors,
        "pass": not errors,
    }
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    if errors:
        raise SystemExit(f"preflight failed: {json.dumps(preflight, sort_keys=True)}")

    for arm in ARMS:
        for interface in INTERFACES:
            base_path = OUTPUT / "data" / arm / interface
            for split, values in addition_by_arm[arm][interface].items():
                base_path.mkdir(parents=True, exist_ok=True)
                (base_path / f"addition_{split}.jsonl").write_text("\n".join(json.dumps(dict(value), sort_keys=True) for value in values) + "\n", encoding="utf-8")
    for interface, records in equality_by_interface.items():
        base_path = OUTPUT / "data" / "equality_pretraining" / interface
        base_path.mkdir(parents=True, exist_ok=True)
        for split, values in records.items():
            (base_path / f"{split}.jsonl").write_text("\n".join(json.dumps(dict(value), sort_keys=True) for value in values) + "\n", encoding="utf-8")
    write_json(OUTPUT / "INITIAL_ANCHORS.json", {"directory": rel(base.ANCHOR_DIR), "seeds": list(SEEDS), "sha256": anchor_hashes, "generated_without_training": True})
    print(json.dumps({"status": "preflight_passed", "arms_expected": len(ARMS) * len(INTERFACES) * len(SEEDS)}, sort_keys=True), flush=True)

    arms_summary: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for interface in INTERFACES:
            for seed in SEEDS:
                aid = f"{arm}_{interface}_seed{seed}"
                addition_records_for_arm = addition_by_arm[arm][interface]
                arm_output = OUTPUT / "arm_outputs" / arm / interface / f"seed{seed}" / "addition"
                pretraining_result = None
                pretrained_model_state = None
                if arm == "A1_equality_pretrained":
                    pre_output = OUTPUT / "arm_outputs" / arm / interface / f"seed{seed}" / "pretraining"
                    pretraining_result = train_model(seed, interface, PRETRAIN_STAGE, equality_by_interface[interface], pre_output, PRETRAIN_STEPS)
                    pre_state = torch.load(pre_output / "checkpoint.pt", map_location="cpu", weights_only=False)
                    pretrained_model_state = pre_state["model"]
                    del pre_state
                result = train_model(seed, interface, ADDITION_STAGE, addition_records_for_arm, arm_output, ADDITION_STEPS, pretrained_model_state)
                checkpoint = arm_output / "checkpoint.pt"
                result.update({"arm_id": aid, "arm": arm, "condition": arm, "interface": interface, "seed": seed, "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint), "fit_pass": bool(result["train_exact"] >= ladder.TRAIN_THRESHOLD), "pretraining": None if pretraining_result is None else {"steps": PRETRAIN_STEPS, "train_exact": pretraining_result["train_exact"], "validation_exact": pretraining_result["validation_exact"], "checkpoint": rel(OUTPUT / "arm_outputs" / arm / interface / f"seed{seed}" / "pretraining" / "checkpoint.pt"), "checkpoint_sha256": sha256(OUTPUT / "arm_outputs" / arm / interface / f"seed{seed}" / "pretraining" / "checkpoint.pt")}})
                write_json(arm_output / "RESULTS.json", result)
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = stage_a.make_dense(seed)
                model.load_state_dict(state["model"], strict=True)
                prediction_arm = {"arm_id": aid, "partition": "diagonal", "interface": interface, "seed": seed, "checkpoint": rel(checkpoint), "checkpoint_sha256": result["checkpoint_sha256"]}
                rows = base.base.predict_rows(model, addition_records_for_arm["train"] + addition_records_for_arm["validation"], prediction_arm)
                for row in rows:
                    row["condition"] = arm
                split_rows = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation")}
                summary = {"arm_id": aid, "arm": arm, "condition": arm, "interface": interface, "seed": seed, "fit_pass": result["fit_pass"], "train": stat(split_rows["train"]), "validation": stat(split_rows["validation"]), "pretraining": result["pretraining"], "checkpoint": result["checkpoint"], "checkpoint_sha256": result["checkpoint_sha256"]}
                write_json(arm_output / "FORENSIC.json", {"summary": summary, "rows": rows})
                arms_summary.append(summary)
                all_rows.extend(rows)
                del model, state
                gc.collect()
                print(json.dumps({"event": "complete", "arm_id": aid, "train": result["train_exact"], "validation": result["validation_exact"]}, sort_keys=True), flush=True)

    write_csv(OUTPUT / "PREDICTIONS.csv", all_rows)
    summary = {"experiment": "crucible_equality_foundational_20251025", "status": "completed", "training_performed": True, "sealed_test_accessed": False, "architecture_changed": False, "arms": arms_summary, "prediction_rows": len(all_rows), "pretraining_steps": PRETRAIN_STEPS, "addition_steps": ADDITION_STEPS}
    write_json(OUTPUT / "SUMMARY.json", summary)
    (OUTPUT / "REPORT.md").write_text("# Crucible Equality Foundational Experiment\n\nCompleted CPU-only run. The final forensic report is `REPORT_FINAL.md`.\n", encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "HASHES.json" and path.suffix != ".pt":
            hashes[rel(path)] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    print(json.dumps({"status": "completed", "arms": len(arms_summary), "prediction_rows": len(all_rows), "output": rel(OUTPUT)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
