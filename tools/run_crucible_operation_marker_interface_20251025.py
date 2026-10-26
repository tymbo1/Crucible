#!/usr/bin/env python3
"""Run the encoding-only A0/A1 operation-conditioning comparison."""

from __future__ import annotations

import csv
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
OUTPUT = ROOT / "experiments" / "crucible_operation_marker_interface_20251025_v3"
S3_ROOT = ROOT / "experiments" / "crucible_role_selection_competence_20251025_v5"
ROLE_SOURCE = ROOT / "tools" / "run_crucible_role_selection_competence_20251025.py"
INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
STEPS = 500
TRAIN_THRESHOLD = 0.95
VALIDATION_THRESHOLD = 0.90
CURVE_STEPS = (1, 10, 25, 50, 100, 200, 300, 400, 500)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


role = import_module("operation_marker_role_runner", ROLE_SOURCE)
ladder = role.ladder
stage_a = role.stage_a
torch = stage_a.torch
TRAIN_SET = set(role.TRAIN_SET)
VALIDATION_SET = set(role.VALIDATION_SET)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(value), sort_keys=True) for value in values) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def s3_arm_dir(interface: str, seed: int) -> Path:
    return S3_ROOT / "arm_outputs" / "S3_JOINT_OPERATION_CONDITIONED" / interface / f"seed{seed}"


def stable_record_id(interface: str, split: str, index: int, a: int, b: int, operation: str) -> str:
    return hashlib.sha256(f"operation_marker|{interface}|{split}|{index}|{a}|{b}|{operation}".encode()).hexdigest()[:20]


def original_prompt(interface: str, operation: str, a: int, b: int) -> str:
    if interface == "canonical":
        return f"Select an operand. operation={operation}; a={a}; b={b}. Return digit only."
    return f"Select an operand. operation={operation}; a_digit={a}; b_digit={b}. Return digit only."


def marker_for(operation: str) -> str:
    return "F" if operation == "SELECT_FIRST" else "S"


def marker_field(operation: str) -> str:
    original_length = len(operation.encode("utf-8"))
    return marker_for(operation) + ("_" * (original_length - 1))


def marker_prompt(interface: str, operation: str, a: int, b: int) -> str:
    field = marker_field(operation)
    if interface == "canonical":
        return f"Select an operand. operation={field}; a={a}; b={b}. Return digit only."
    return f"Select an operand. operation={field}; a_digit={a}; b_digit={b}. Return digit only."


def make_records(interface: str, split: str, pairs: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    index = 0
    for a, b in pairs:
        for operation in ("SELECT_FIRST", "SELECT_SECOND"):
            target = a if operation == "SELECT_FIRST" else b
            rows.append({
                "schema_version": "crucible_operation_marker_interface_v1",
                "record_id": stable_record_id(interface, split, index, a, b, operation),
                "interface": interface,
                "stage": "role_selection_competence",
                "split": split,
                "condition": "A1_SINGLE_TOKEN_MARKER",
                "input": marker_prompt(interface, operation, a, b),
                "target": str(target),
                "oracle": {"kind": "role_selection", "operation": operation, "value": target},
                "metadata": {
                    "a": a,
                    "b": b,
                    "answer": target,
                    "operation": operation,
                    "marker": marker_for(operation),
                    "marker_field": marker_field(operation),
                    "pair_kind": "equal" if a == b else "unequal",
                    "reversed_pair": [b, a],
                    "output_length": 1,
                },
            })
            index += 1
    return rows


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit_records(interface: str, marker_records: Mapping[str, Sequence[Mapping[str, Any]]], original_records: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    errors: List[str] = []
    for split in ("train", "validation"):
        markers = marker_records[split]
        originals = original_records[split]
        if len(markers) != len(originals):
            errors.append(f"count:{split}")
        original_by_key = {(int(row["metadata"]["a"]), int(row["metadata"]["b"]), row["metadata"]["operation"]): row for row in originals}
        marker_by_key = {(int(row["metadata"]["a"]), int(row["metadata"]["b"]), row["metadata"]["operation"]): row for row in markers}
        if set(original_by_key) != set(marker_by_key):
            errors.append(f"keys:{split}")
        for key, marker in marker_by_key.items():
            original = original_by_key[key]
            if int(marker["target"]) != int(original["target"]):
                errors.append(f"label:{split}:{marker['record_id']}")
            original_input = str(original["input"])
            marker_input = str(marker["input"])
            if len(original_input.encode("utf-8")) != len(marker_input.encode("utf-8")):
                errors.append(f"prompt_length:{split}:{marker['record_id']}")
            for anchor in ("a=", "a_digit=", "b=", "b_digit=", "Return digit only."):
                if anchor in original_input and original_input.index(anchor) != marker_input.index(anchor):
                    errors.append(f"position:{anchor}:{split}:{marker['record_id']}")
            operation = key[2]
            field = marker_field(operation)
            if f"operation={field};" not in marker_input or field[0] != marker["metadata"]["marker"]:
                errors.append(f"marker:{split}:{marker['record_id']}")
            if len(field) != len(operation.encode("utf-8")):
                errors.append(f"marker_length:{split}:{marker['record_id']}")
    train_pairs = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in marker_records["train"]}
    validation_pairs = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in marker_records["validation"]}
    if train_pairs != TRAIN_SET or validation_pairs != VALIDATION_SET or train_pairs & validation_pairs:
        errors.append("split")
    return {
        "interface": interface,
        "train_rows": len(marker_records["train"]),
        "validation_rows": len(marker_records["validation"]),
        "errors": errors,
        "pass": not errors,
    }


def prepare(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return ladder.prepare(records)


def load_model_checkpoint(path: Path) -> Any:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = stage_a.DenseLM()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def decode_output(token_id: int) -> Tuple[str, bool]:
    value = int(token_id) - int(stage_a.RAW_BYTE_START)
    if 48 <= value <= 57:
        return chr(value), False
    return f"<token:{token_id}>", True


def evaluate_records(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return stage_a.evaluate(model, prepare(records))


def stat(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row["output_token_correct"]) for row in rows)
    full_correct = sum(bool(row["full_sequence_correct"]) for row in rows)
    return {
        "examples": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "full_sequence_correct": full_correct,
        "full_sequence_accuracy": full_correct / n if n else None,
        "format_errors": sum(bool(row["output_format_error"]) for row in rows),
        "eos_correct": sum(bool(row["eos_correct"]) for row in rows),
    }


def predict_rows(model: Any, records: Sequence[Mapping[str, Any]], arm: Mapping[str, Any]) -> List[Dict[str, Any]]:
    prepared = prepare(records)
    rows: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prepared), int(stage_a.BATCH_SIZE)):
            indices = list(range(start, min(start + int(stage_a.BATCH_SIZE), len(prepared))))
            batch = stage_a.batchify(prepared, indices)
            logits = stage_a.forward_model(model, batch)
            for local, index in enumerate(indices):
                item = prepared[index]
                output_position = int(item["_prefix_len"]) - 1
                eos_position = output_position + 1
                prediction_id = int(logits[local, output_position].argmax().item())
                prediction, format_error = decode_output(prediction_id)
                target_id = int(batch["y"][local, output_position].item())
                predicted_eos_id = int(logits[local, eos_position].argmax().item())
                target_eos_id = int(batch["y"][local, eos_position].item())
                rows.append({
                    **arm,
                    "record_id": item["record_id"],
                    "split": item["split"],
                    "a": int(item["metadata"]["a"]),
                    "b": int(item["metadata"]["b"]),
                    "operation": item["metadata"]["operation"],
                    "marker": item["metadata"]["marker"],
                    "expected_output": str(item["target"]),
                    "predicted_output": prediction,
                    "predicted_token_id": prediction_id,
                    "output_format_error": format_error,
                    "eos_correct": predicted_eos_id == target_eos_id,
                    "output_token_correct": prediction_id == target_id,
                    "full_sequence_correct": prediction_id == target_id and predicted_eos_id == target_eos_id,
                })
    return rows


def arm_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_operation = {}
    for operation in ("SELECT_FIRST", "SELECT_SECOND"):
        by_operation[operation] = stat([row for row in rows if row["operation"] == operation])
    by_pair_kind = {}
    for kind in ("equal", "unequal"):
        by_pair_kind[kind] = stat([row for row in rows if (row["a"] == row["b"]) == (kind == "equal")])
    return {"aggregate": stat(rows), "by_operation": by_operation, "by_pair_kind": by_pair_kind}


def paired_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[int, int], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[(int(row["a"]), int(row["b"]))][str(row["operation"])] = row
    result: Dict[str, Any] = {}
    for kind, selected in (("all", groups), ("unequal", {key: value for key, value in groups.items() if key[0] != key[1]})):
        group_values = list(selected.values())
        complete = [value for value in group_values if "SELECT_FIRST" in value and "SELECT_SECOND" in value]
        both = sum(bool(value["SELECT_FIRST"]["output_token_correct"] and value["SELECT_SECOND"]["output_token_correct"]) for value in complete)
        switched = sum(value["SELECT_FIRST"]["predicted_output"] != value["SELECT_SECOND"]["predicted_output"] for value in complete)
        correct_and_switched = sum(bool(value["SELECT_FIRST"]["output_token_correct"] and value["SELECT_SECOND"]["output_token_correct"] and value["SELECT_FIRST"]["predicted_output"] != value["SELECT_SECOND"]["predicted_output"]) for value in complete)
        result[kind] = {
            "pair_groups": len(complete),
            "both_operations_correct": both,
            "both_operations_correct_rate": both / len(complete) if complete else None,
            "predictions_switched": switched,
            "prediction_switch_rate": switched / len(complete) if complete else None,
            "correct_and_switched": correct_and_switched,
            "correct_and_switched_rate": correct_and_switched / len(complete) if complete else None,
            "pairs": [
                {
                    "pair": list(key),
                    "first_prediction": value["SELECT_FIRST"]["predicted_output"],
                    "second_prediction": value["SELECT_SECOND"]["predicted_output"],
                    "first_correct": bool(value["SELECT_FIRST"]["output_token_correct"]),
                    "second_correct": bool(value["SELECT_SECOND"]["output_token_correct"]),
                }
                for key, value in sorted(selected.items()) if "SELECT_FIRST" in value and "SELECT_SECOND" in value
            ],
        }
    return result


def train_a1(interface: str, seed: int, records: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    initial_digest = stage_a.canonical_state_digest(model.state_dict())
    optimizer = stage_a.optimizer_for(model)
    prepared = {split: prepare(values) for split, values in records.items()}
    order = list(range(len(prepared["train"])))
    random.Random(ladder.stable_seed(seed, "S3_JOINT_OPERATION_CONDITIONED", interface, "order")).shuffle(order)
    curve_points: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        result = stage_a.train_step(model, optimizer, prepared["train"], stage_a.batch_indices(order, step))
        if step in CURVE_STEPS:
            curve_points.append({
                "step": step,
                "last_step_loss": float(result["loss"]),
                "train": evaluate_records(model, records["train"]),
                "validation": evaluate_records(model, records["validation"]),
            })
    train_eval = evaluate_records(model, records["train"])
    validation_eval = evaluate_records(model, records["validation"])
    checkpoint = output / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint, model, optimizer, STEPS, order)
    result = {
        "arm_id": f"A1_SINGLE_TOKEN_MARKER_{interface}_seed{seed}",
        "condition": "A1_SINGLE_TOKEN_MARKER",
        "interface": interface,
        "seed": seed,
        "steps": STEPS,
        "elapsed_seconds": time.perf_counter() - started,
        "initial_dense_state_digest": initial_digest,
        "train": train_eval,
        "validation": validation_eval,
        "train_exact": train_eval["teacher_forced_exact_accuracy"],
        "validation_exact": validation_eval["teacher_forced_exact_accuracy"],
        "fit_pass": train_eval["teacher_forced_exact_accuracy"] >= TRAIN_THRESHOLD,
        "validation_pass": validation_eval["teacher_forced_exact_accuracy"] >= VALIDATION_THRESHOLD,
        "checkpoint": rel(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
    }
    result["gate_pass"] = bool(result["fit_pass"] and result["validation_pass"])
    write_json(output / "RESULTS.json", result)
    write_json(output / "TRAINING_CURVE.json", {"curve_steps": list(CURVE_STEPS), "points": curve_points})
    write_jsonl(output / "train.jsonl", records["train"])
    write_jsonl(output / "validation.jsonl", records["validation"])
    del model, optimizer
    gc.collect()
    return result


def load_a0_rows() -> List[Dict[str, Any]]:
    path = S3_ROOT / "PREDICTIONS.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    return [row for row in rows if row["condition"] == "S3_JOINT_OPERATION_CONDITIONED"]


def compare_a0_a1(a0_rows: Sequence[Mapping[str, Any]], a1_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def key(row: Mapping[str, Any]) -> Tuple[str, int, str, str, int, int]:
        return (str(row["interface"]), int(row["seed"]), str(row["split"]), str(row["operation"]), int(row["a"]), int(row["b"]))
    a0 = {key(row): row for row in a0_rows}
    a1 = {key(row): row for row in a1_rows}
    rows: List[Dict[str, Any]] = []
    for match_key in sorted(set(a0) & set(a1)):
        old = a0[match_key]
        new = a1[match_key]
        rows.append({
            "interface": match_key[0],
            "seed": match_key[1],
            "split": match_key[2],
            "operation": match_key[3],
            "a": match_key[4],
            "b": match_key[5],
            "a0_prediction": old["predicted_output"],
            "a1_prediction": new["predicted_output"],
            "a0_correct": old["output_token_correct"] == "True",
            "a1_correct": bool(new["output_token_correct"]) if isinstance(new["output_token_correct"], bool) else new["output_token_correct"] == "True",
            "prediction_changed": old["predicted_output"] != new["predicted_output"],
        })
    return {"matched_rows": len(rows), "rows": rows}


def summarize_comparison(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for interface in INTERFACES:
        for seed in SEEDS:
            selected = [row for row in rows if row["interface"] == interface and row["seed"] == seed]
            result[f"{interface}_seed{seed}"] = {
                "rows": len(selected),
                "a0_correct": sum(bool(row["a0_correct"]) for row in selected),
                "a1_correct": sum(bool(row["a1_correct"]) for row in selected),
                "prediction_changes": sum(bool(row["prediction_changed"]) for row in selected),
            }
    return result


def build_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Crucible Single-Token Operation-Marker Experiment — Final Report",
        "",
        "CPU-only A0/A1 encoding comparison. A0 is the frozen S3 baseline; A1 is retrained with length-preserving one-byte operation markers. No sealed-test access.",
        "",
        "## Results",
        "",
        "| Interface | Seed | A0 validation | A1 validation | A1 unequal validation | A1 gate | A1 paired both correct | A1 correct-and-switched |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for arm in summary["a1_arms"]:
        paired = arm["paired"]["validation"]["unequal"]
        a0 = summary["a0_by_arm"][f"{arm['interface']}_seed{arm['seed']}"]
        lines.append(f"| {arm['interface']} | {arm['seed']} | {a0['validation_accuracy']:.2%} | {arm['validation_accuracy']:.2%} | {arm['unequal_validation_accuracy']:.2%} | {'PASS' if arm['gate_pass'] else 'FAIL'} | {paired['both_operations_correct']}/{paired['pair_groups']} | {paired['correct_and_switched']}/{paired['pair_groups']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The primary result is paired conditional correctness on the same unequal operand pair, not aggregate accuracy alone. A1 changes only the operation encoding while preserving the corresponding prompt length and operand/answer positions. The neutral padding remains a possible lexical cue, so this is an encoding intervention rather than a complete tokenizer ablation.",
        "",
        "The original S3 baseline remains preserved in its source experiment. This experiment does not run the positional A2/A3 matrix; that is the follow-up if A1 does not resolve the conditional-execution failure.",
        "",
        "## Artifacts",
        "",
        "- [Protocol](../../docs/CRUCIBLE_OPERATION_MARKER_INTERFACE_PROTOCOL_20251025.md)",
        "- [Summary](SUMMARY.json)",
        "- [A1 predictions](PREDICTIONS.csv)",
        "- [A0/A1 comparison](COMPARISON.csv)",
        "- [Machine-readable metrics](RESULTS.json)",
        "- [Finalization](FINALIZATION.json)",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    a0_rows = load_a0_rows()
    if len(a0_rows) != 400:
        raise SystemExit(f"unexpected A0 row count: {len(a0_rows)}")
    protocol = {
        "experiment": "crucible_operation_marker_interface_20251025_v1",
        "status": "frozen_before_training",
        "interfaces": list(INTERFACES),
        "seeds": list(SEEDS),
        "steps": STEPS,
        "condition": "A1_SINGLE_TOKEN_MARKER",
        "marker_encoding": {"SELECT_FIRST": marker_field("SELECT_FIRST"), "SELECT_SECOND": marker_field("SELECT_SECOND")},
        "sealed_test_accessed": False,
        "architecture_changed": False,
        "a0_source": rel(S3_ROOT),
    }
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", protocol)
    all_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    audits: Dict[str, Any] = {}
    for interface in INTERFACES:
        original = {
            "train": load_jsonl(s3_arm_dir(interface, 42) / "train.jsonl"),
            "validation": load_jsonl(s3_arm_dir(interface, 42) / "validation.jsonl"),
        }
        all_records[interface] = {
            "train": make_records(interface, "train", sorted(TRAIN_SET)),
            "validation": make_records(interface, "validation", sorted(VALIDATION_SET)),
        }
        audits[interface] = audit_records(interface, all_records[interface], original)
        for split in ("train", "validation"):
            write_jsonl(OUTPUT / "data" / interface / f"{split}.jsonl", all_records[interface][split])
    preflight = {"audits": audits, "pass": all(audit["pass"] for audit in audits.values()), "training_performed": False, "sealed_test_accessed": False}
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    if not preflight["pass"]:
        raise SystemExit(f"preflight failed: {json.dumps(preflight, sort_keys=True)}")
    print(json.dumps({"event": "preflight_passed", "output": rel(OUTPUT)}, sort_keys=True), flush=True)

    a1_results: List[Dict[str, Any]] = []
    a1_rows: List[Dict[str, Any]] = []
    for interface in INTERFACES:
        for seed in SEEDS:
            arm_output = OUTPUT / "arm_outputs" / interface / f"seed{seed}"
            print(json.dumps({"event": "start", "interface": interface, "seed": seed}, sort_keys=True), flush=True)
            result = train_a1(interface, seed, all_records[interface], arm_output)
            model = load_model_checkpoint(arm_output / "checkpoint.pt")
            rows = predict_rows(model, all_records[interface]["train"] + all_records[interface]["validation"], {"arm_id": result["arm_id"], "interface": interface, "seed": seed, "checkpoint": rel(arm_output / "checkpoint.pt"), "checkpoint_sha256": result["checkpoint_sha256"]})
            metrics = arm_metrics(rows)
            train_rows = [row for row in rows if row["split"] == "train"]
            validation_rows = [row for row in rows if row["split"] == "validation"]
            train_metrics = arm_metrics(train_rows)
            validation_metrics = arm_metrics(validation_rows)
            paired = {"train": paired_metrics(train_rows), "validation": paired_metrics(validation_rows)}
            result.update({
                "output_token_train_accuracy": train_metrics["aggregate"]["accuracy"],
                "output_token_validation_accuracy": validation_metrics["aggregate"]["accuracy"],
                "validation_accuracy": validation_metrics["aggregate"]["accuracy"],
                "train_accuracy": train_metrics["aggregate"]["accuracy"],
                "unequal_validation_accuracy": validation_metrics["by_pair_kind"]["unequal"]["accuracy"],
                "prediction_train_metrics": train_metrics["aggregate"],
                "prediction_validation_metrics": validation_metrics["aggregate"],
                "by_operation": validation_metrics["by_operation"],
                "by_pair_kind": validation_metrics["by_pair_kind"],
                "paired": paired,
                "gate_pass": bool(result["train_exact"] >= TRAIN_THRESHOLD and result["validation_exact"] >= VALIDATION_THRESHOLD and validation_metrics["by_pair_kind"]["unequal"]["full_sequence_accuracy"] >= VALIDATION_THRESHOLD),
            })
            write_json(arm_output / "RESULTS.json", result)
            write_json(arm_output / "FORENSIC.json", {"metrics": metrics, "paired": paired, "prediction_rows": len(rows)})
            a1_results.append(result)
            a1_rows.extend(rows)
            del model
            gc.collect()
            print(json.dumps({"event": "complete", "interface": interface, "seed": seed, "validation": result["validation_exact"], "paired_unequal": paired["unequal"]["both_operations_correct_rate"]}, sort_keys=True), flush=True)

    comparison = compare_a0_a1(a0_rows, a1_rows)
    write_csv(OUTPUT / "PREDICTIONS.csv", a1_rows)
    write_csv(OUTPUT / "COMPARISON.csv", comparison["rows"])
    a0_by_arm: Dict[str, Any] = {}
    for interface in INTERFACES:
        for seed in SEEDS:
            selected = [row for row in a0_rows if row["interface"] == interface and int(row["seed"]) == seed and row["split"] == "validation"]
            a0_by_arm[f"{interface}_seed{seed}"] = {"validation_correct": sum(row["output_token_correct"] == "True" for row in selected), "validation_rows": len(selected), "validation_accuracy": sum(row["output_token_correct"] == "True" for row in selected) / len(selected)}
    summary = {
        "experiment": "crucible_operation_marker_interface_20251025_v1",
        "status": "completed",
        "a0_by_arm": a0_by_arm,
        "a1_arms": a1_results,
        "a0_a1_comparison": {"matched_rows": comparison["matched_rows"], "by_arm": summarize_comparison(comparison["rows"])},
        "prediction_rows": len(a1_rows),
        "sealed_test_accessed": False,
        "architecture_changed": False,
        "preflight": preflight,
    }
    write_json(OUTPUT / "RESULTS.json", {"a0": a0_by_arm, "a1": a1_results, "comparison": summary["a0_a1_comparison"]})
    write_json(OUTPUT / "SUMMARY.json", summary)
    preflight["training_performed"] = True
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    (OUTPUT / "REPORT_FINAL.md").write_text(build_report(summary), encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt":
            hashes[rel(path)] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = {
        "status": "completed",
        "verification_pass": bool(preflight["pass"] and len(a1_results) == len(INTERFACES) * len(SEEDS) and all((OUTPUT / result["checkpoint"]).is_file() for result in a1_results)),
        "preflight_pass": preflight["pass"],
        "sealed_test_accessed": False,
        "architecture_changed": False,
        "a1_checkpoint_hashes": {result["arm_id"]: result["checkpoint_sha256"] for result in a1_results},
        "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"),
        "comparison_sha256": sha256(OUTPUT / "COMPARISON.csv"),
    }
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "completed", "output": rel(OUTPUT), "arms": len(a1_results), "prediction_rows": len(a1_rows), "comparison_rows": comparison["matched_rows"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
