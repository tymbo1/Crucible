#!/usr/bin/env python3
"""Run the A0-A3 operation encoding/position matrix."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_operation_position_matrix_20251025_v1"
S3_ROOT = ROOT / "experiments" / "crucible_role_selection_competence_20251025_v5"
A1_ROOT = ROOT / "experiments" / "crucible_operation_marker_interface_20251025_v3"
ROLE_SOURCE = ROOT / "tools" / "run_crucible_role_selection_competence_20251025.py"
FORENSICS_SOURCE = ROOT / "tools" / "run_crucible_operation_conditioning_forensics_20251025.py"
INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
NEW_CONDITIONS = ("A2_WORD_FIXED", "A3_MARKER_FIXED")
ALL_CONDITIONS = ("A0_WORD_ORIGINAL", "A1_MARKER_ORIGINAL", "A2_WORD_FIXED", "A3_MARKER_FIXED")
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


role = import_module("operation_position_role_runner", ROLE_SOURCE)
forensics = import_module("operation_position_forensics", FORENSICS_SOURCE)
ladder = role.ladder
stage_a = role.stage_a
torch = stage_a.torch


def prepare_with_positions(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Locate the actual operation field for word and marker interfaces."""
    prepared = ladder.prepare(records)
    for item in prepared:
        raw_input = str(item["input"]).encode("utf-8")
        field_start = raw_input.find(b"operation=")
        if field_start < 0:
            raise RuntimeError(f"operation field not found: {item['record_id']}")
        value_start = field_start + len(b"operation=")
        value_end = raw_input.find(b";", value_start)
        if value_end <= value_start:
            raise RuntimeError(f"operation field terminator not found: {item['record_id']}")
        item["_operation_start"] = 1 + value_start
        item["_operation_last"] = 1 + value_end - 1
        item["_output_position"] = int(item["_prefix_len"]) - 1
        if not (0 <= item["_operation_start"] <= item["_operation_last"] < len(item["_full"])):
            raise RuntimeError(f"invalid operation position: {item['record_id']}")
        if not (0 <= item["_output_position"] < len(item["_full"])):
            raise RuntimeError(f"invalid output position: {item['record_id']}")
    return prepared


# The shared forensic functions are retained; only their position locator is
# specialized because A1/A3 use markers instead of literal operation words.
forensics.prepare_with_positions = prepare_with_positions
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


def a1_arm_dir(interface: str, seed: int) -> Path:
    return A1_ROOT / "arm_outputs" / interface / f"seed{seed}"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def marker_field(operation: str) -> str:
    return ("F" if operation == "SELECT_FIRST" else "S") + ("_" * (len(operation.encode("utf-8")) - 1))


def fixed_prompt(interface: str, operation: str, a: int, b: int, marker: bool) -> str:
    field = marker_field(operation) if marker else operation
    if interface == "canonical":
        return f"Select an operand. a={a}; b={b}. Return digit only. operation={field};"
    return f"Select an operand. a_digit={a}; b_digit={b}. Return digit only. operation={field};"


def make_records(condition: str, interface: str, split: str, pairs: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    marker = condition == "A3_MARKER_FIXED"
    rows: List[Dict[str, Any]] = []
    index = 0
    for a, b in pairs:
        for operation in ("SELECT_FIRST", "SELECT_SECOND"):
            target = a if operation == "SELECT_FIRST" else b
            rows.append({
                "schema_version": "crucible_operation_position_matrix_v1",
                "record_id": hashlib.sha256(f"operation_position|{condition}|{interface}|{split}|{index}|{a}|{b}|{operation}".encode()).hexdigest()[:20],
                "interface": interface,
                "stage": "role_selection_competence",
                "split": split,
                "condition": condition,
                "input": fixed_prompt(interface, operation, a, b, marker),
                "target": str(target),
                "oracle": {"kind": "role_selection", "operation": operation, "value": target},
                "metadata": {
                    "a": a,
                    "b": b,
                    "answer": target,
                    "operation": operation,
                    "marker": ("F" if operation == "SELECT_FIRST" else "S") if marker else None,
                    "position": "fixed_before_answer",
                    "pair_kind": "equal" if a == b else "unequal",
                    "output_length": 1,
                },
            })
            index += 1
    return rows


def audit_fixed(condition: str, interface: str, records: Mapping[str, Sequence[Mapping[str, Any]]], original: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    errors: List[str] = []
    for split in ("train", "validation"):
        current = records[split]
        source = original[split]
        source_by_key = {(int(row["metadata"]["a"]), int(row["metadata"]["b"]), row["metadata"]["operation"]): row for row in source}
        current_by_key = {(int(row["metadata"]["a"]), int(row["metadata"]["b"]), row["metadata"]["operation"]): row for row in current}
        if set(source_by_key) != set(current_by_key):
            errors.append(f"keys:{split}")
        for key, row in current_by_key.items():
            source_row = source_by_key[key]
            if len(str(row["input"]).encode()) != len(str(source_row["input"]).encode()):
                errors.append(f"length:{split}:{row['record_id']}")
            if int(row["target"]) != int(source_row["target"]):
                errors.append(f"label:{split}:{row['record_id']}")
            if "Return digit only." not in str(row["input"]) or "operation=" not in str(row["input"]):
                errors.append(f"format:{split}:{row['record_id']}")
            if str(row["input"]).rfind("operation=") < str(row["input"]).find("Return digit only."):
                errors.append(f"not_fixed:{split}:{row['record_id']}")
    train_pairs = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in records["train"]}
    validation_pairs = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in records["validation"]}
    if train_pairs != TRAIN_SET or validation_pairs != VALIDATION_SET or train_pairs & validation_pairs:
        errors.append("split")
    return {"condition": condition, "interface": interface, "train_rows": len(records["train"]), "validation_rows": len(records["validation"]), "errors": errors, "pass": not errors}


def load_dense(path: Path) -> Any:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = stage_a.DenseLM()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def decode(token_id: int) -> Tuple[str, bool]:
    value = int(token_id) - int(stage_a.RAW_BYTE_START)
    if 48 <= value <= 57:
        return chr(value), False
    return f"<token:{token_id}>", True


def predict_rows(model: Any, records: Sequence[Mapping[str, Any]], arm: Mapping[str, Any]) -> List[Dict[str, Any]]:
    prepared = ladder.prepare(records)
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(prepared), int(stage_a.BATCH_SIZE)):
            indices = list(range(start, min(start + int(stage_a.BATCH_SIZE), len(prepared))))
            batch = stage_a.batchify(prepared, indices)
            logits = stage_a.forward_model(model, batch)
            for local, index in enumerate(indices):
                item = prepared[index]
                output_position = int(item["_prefix_len"]) - 1
                eos_position = output_position + 1
                predicted_id = int(logits[local, output_position].argmax().item())
                target_id = int(batch["y"][local, output_position].item())
                predicted_eos = int(logits[local, eos_position].argmax().item())
                target_eos = int(batch["y"][local, eos_position].item())
                prediction, format_error = decode(predicted_id)
                rows.append({
                    **arm,
                    "record_id": item["record_id"],
                    "split": item["split"],
                    "a": int(item["metadata"]["a"]),
                    "b": int(item["metadata"]["b"]),
                    "operation": item["metadata"]["operation"],
                    "expected_output": str(item["target"]),
                    "predicted_output": prediction,
                    "output_token_correct": predicted_id == target_id,
                    "eos_correct": predicted_eos == target_eos,
                    "full_sequence_correct": predicted_id == target_id and predicted_eos == target_eos,
                    "output_format_error": format_error,
                })
    return rows


def normalize_csv_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        for key in ("output_token_correct", "eos_correct", "full_sequence_correct", "output_format_error"):
            if key in row and isinstance(row[key], str):
                row[key] = row[key] == "True"
        for key in ("a", "b", "seed"):
            if key in row:
                row[key] = int(row[key])
        normalized.append(row)
    return normalized


def train_new(condition: str, interface: str, seed: int, records: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    initial_digest = stage_a.canonical_state_digest(model.state_dict())
    optimizer = stage_a.optimizer_for(model)
    prepared = {split: ladder.prepare(values) for split, values in records.items()}
    order = list(range(len(prepared["train"])))
    random.Random(ladder.stable_seed(seed, condition, interface, "order")).shuffle(order)
    curves: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        stage_a.train_step(model, optimizer, prepared["train"], stage_a.batch_indices(order, step))
        if step in CURVE_STEPS:
            curves.append({"step": step, "train": stage_a.evaluate(model, prepared["train"]), "validation": stage_a.evaluate(model, prepared["validation"])})
    train_eval = stage_a.evaluate(model, prepared["train"])
    validation_eval = stage_a.evaluate(model, prepared["validation"])
    checkpoint = output / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint, model, optimizer, STEPS, order)
    result = {
        "arm_id": f"{condition}_{interface}_seed{seed}",
        "condition": condition,
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
    write_json(output / "RESULTS.json", result)
    write_json(output / "TRAINING_CURVE.json", {"curve_steps": list(CURVE_STEPS), "points": curves})
    write_jsonl(output / "train.jsonl", records["train"])
    write_jsonl(output / "validation.jsonl", records["validation"])
    del model, optimizer
    gc.collect()
    return result


def condition_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    def stat(values: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        n = len(values)
        correct = sum(bool(row["output_token_correct"]) for row in values)
        full = sum(bool(row["full_sequence_correct"]) for row in values)
        return {"rows": n, "correct": correct, "accuracy": correct / n if n else None, "full_sequence_correct": full, "full_sequence_accuracy": full / n if n else None}
    by_operation = {operation: stat([row for row in validation if row["operation"] == operation]) for operation in ("SELECT_FIRST", "SELECT_SECOND")}
    unequal = [row for row in validation if row["a"] != row["b"]]
    pairs: Dict[Tuple[int, int], Dict[str, Mapping[str, Any]]] = {}
    for row in validation:
        pairs.setdefault((row["a"], row["b"]), {})[row["operation"]] = row
    unequal_pairs = {key: value for key, value in pairs.items() if key[0] != key[1]}
    complete = [value for value in unequal_pairs.values() if len(value) == 2]
    both = sum(value["SELECT_FIRST"]["output_token_correct"] and value["SELECT_SECOND"]["output_token_correct"] for value in complete)
    switched = sum(value["SELECT_FIRST"]["predicted_output"] != value["SELECT_SECOND"]["predicted_output"] for value in complete)
    correct_switched = sum(value["SELECT_FIRST"]["output_token_correct"] and value["SELECT_SECOND"]["output_token_correct"] and value["SELECT_FIRST"]["predicted_output"] != value["SELECT_SECOND"]["predicted_output"] for value in complete)
    return {
        "train": stat(train),
        "validation": stat(validation),
        "validation_unequal": stat(unequal),
        "validation_by_operation": by_operation,
        "validation_paired_unequal": {"pair_groups": len(complete), "both_correct": both, "both_correct_rate": both / len(complete) if complete else None, "predictions_switched": switched, "correct_and_switched": correct_switched, "correct_and_switched_rate": correct_switched / len(complete) if complete else None},
    }


def build_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Crucible Operation-Position Matrix — Final Report",
        "",
        "CPU-only A0–A3 comparison. A0/A1 are verified existing artifacts; A2/A3 are newly trained. No sealed-test access.",
        "",
        "## Results",
        "",
        "| Condition | Interface | Seed | Train exact | Validation exact | Unequal validation | Paired both correct | Correct-and-switched | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in summary["arms"]:
        pair = result["metrics"]["validation_paired_unequal"]
        lines.append(f"| {result['condition']} | {result['interface']} | {result['seed']} | {result['metrics']['train']['accuracy']:.2%} | {result['metrics']['validation']['accuracy']:.2%} | {result['metrics']['validation_unequal']['accuracy']:.2%} | {pair['both_correct']}/{pair['pair_groups']} | {pair['correct_and_switched']}/{pair['pair_groups']} | {'PASS' if result['gate_pass'] else 'FAIL'} |")
    lines += [
        "",
        "## Answer-position probes",
        "",
        "The operation-identity probe is fitted on training states and evaluated on disjoint validation states. Probe decodability is not treated as proof of causal use.",
        "",
        "| Condition | Interface | Seed | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Final norm |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in summary["answer_position_probes"].items():
        parts = key.rsplit("_", 2)
        condition = parts[0]
        interface = parts[1]
        seed = parts[2].replace("seed", "")
        row = [condition, interface, seed]
        for state in ("layer_0", "layer_1", "layer_2", "layer_3", "final_norm"):
            row.append(f"{values[state]['ridge']['validation_accuracy']:.2%}")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Training paired correctness is shown separately from held-out paired correctness. A held-out failure is not interpreted as a generalization result when the training gate fails.",
        "",
        "A2/A3 position changes affect distance, absolute positions and prompt token distributions together. Any improvement would therefore establish position sensitivity of this interface, not a unique attention mechanism.",
        "",
        "## Artifacts",
        "",
        "- [Protocol](../../docs/CRUCIBLE_OPERATION_POSITION_MATRIX_PROTOCOL_20251025.md)",
        "- [Summary](SUMMARY.json)",
        "- [Predictions](PREDICTIONS.csv)",
        "- [Answer-position probes](ANSWER_POSITION_PROBES.json)",
        "- [Output sensitivity](OUTPUT_SENSITIVITY.json)",
        "- [Finalization](FINALIZATION.json)",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_condition_rows(condition: str, interface: str, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if condition == "A0_WORD_ORIGINAL":
        directory = s3_arm_dir(interface, seed)
    else:
        directory = a1_arm_dir(interface, seed)
    source_rows = normalize_csv_rows(list(csv.DictReader((S3_ROOT / "PREDICTIONS.csv" if condition == "A0_WORD_ORIGINAL" else A1_ROOT / "PREDICTIONS.csv").open(newline="", encoding="utf-8"))))
    selected = [row for row in source_rows if row["interface"] == interface and row["seed"] == seed]
    records = {"train": load_jsonl(directory / "train.jsonl"), "validation": load_jsonl(directory / "validation.jsonl")}
    checkpoint = directory / "checkpoint.pt"
    model = load_dense(checkpoint)
    return selected, {"model": model, "records": records, "checkpoint": checkpoint, "source_rows": selected}


def compare_source_predictions(source_rows: Sequence[Mapping[str, Any]], reproduced_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def key(row: Mapping[str, Any]) -> Tuple[str, str, int, int]:
        return (str(row["split"]), str(row["operation"]), int(row["a"]), int(row["b"]))

    source = {key(row): row for row in source_rows}
    reproduced = {key(row): row for row in reproduced_rows}
    errors: List[str] = []
    if set(source) != set(reproduced):
        errors.append("record_key_set")
    for match_key in sorted(set(source) & set(reproduced)):
        left = source[match_key]
        right = reproduced[match_key]
        for field in ("predicted_output", "output_token_correct", "eos_correct", "full_sequence_correct", "output_format_error"):
            left_value = left.get(field)
            if isinstance(left_value, str) and left_value in {"True", "False"}:
                left_value = left_value == "True"
            if left_value != right.get(field):
                errors.append(f"{match_key}:{field}")
    return {"rows_source": len(source_rows), "rows_reproduced": len(reproduced_rows), "errors": errors, "pass": not errors}


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {"experiment": "crucible_operation_position_matrix_20251025_v1", "status": "frozen_before_training", "conditions": list(ALL_CONDITIONS), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "steps": STEPS, "sealed_test_accessed": False, "architecture_changed": False}
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", protocol)

    all_condition_rows: List[Dict[str, Any]] = []
    arms: List[Dict[str, Any]] = []
    answer_probes: Dict[str, Any] = {}
    output_sensitivity: Dict[str, Any] = {}
    audits: Dict[str, Any] = {}
    source_reproduction: Dict[str, Any] = {}
    (OUTPUT / "hidden_states").mkdir(parents=True, exist_ok=True)

    for condition in ALL_CONDITIONS:
        for interface in INTERFACES:
            for seed in SEEDS:
                key = f"{condition}_{interface}_seed{seed}"
                if condition in ("A0_WORD_ORIGINAL", "A1_MARKER_ORIGINAL"):
                    rows, source = source_condition_rows(condition, interface, seed)
                    model = source["model"]
                    records = source["records"]
                    checkpoint = source["checkpoint"]
                    result_path = checkpoint.parent / "RESULTS.json"
                    recorded = load_json(result_path)
                    source_rows = [{**row, "condition": condition, "arm_id": key} for row in rows]
                    source_result = {"arm_id": key, "condition": condition, "interface": interface, "seed": seed, "train_exact": recorded.get("train_exact"), "validation_exact": recorded.get("validation_exact"), "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint), "source_checkpoint_hash": recorded.get("checkpoint_sha256")}
                    reproduced_rows = predict_rows(model, records["train"] + records["validation"], {"arm_id": key, "condition": condition, "interface": interface, "seed": seed, "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint)})
                    reproduction = compare_source_predictions(source_rows, reproduced_rows)
                    source_reproduction[key] = reproduction
                    reproduced_metrics = condition_metrics(reproduced_rows)
                    source_result["reproduced_train_accuracy"] = reproduced_metrics["train"]["accuracy"]
                    source_result["reproduced_validation_accuracy"] = reproduced_metrics["validation"]["accuracy"]
                    source_result["reported_metric_match"] = bool(
                        math.isclose(float(source_result["train_exact"]), float(source_result["reproduced_train_accuracy"]), rel_tol=1e-7, abs_tol=1e-7)
                        and math.isclose(float(source_result["validation_exact"]), float(source_result["reproduced_validation_accuracy"]), rel_tol=1e-7, abs_tol=1e-7)
                    )
                    rows = reproduced_rows
                else:
                    original = {"train": load_jsonl(s3_arm_dir(interface, seed) / "train.jsonl"), "validation": load_jsonl(s3_arm_dir(interface, seed) / "validation.jsonl")}
                    records = {"train": make_records(condition, interface, "train", sorted(TRAIN_SET)), "validation": make_records(condition, interface, "validation", sorted(VALIDATION_SET))}
                    audit = audit_fixed(condition, interface, records, original)
                    audits[key] = audit
                    if not audit["pass"]:
                        raise SystemExit(f"preflight failed: {json.dumps(audit, sort_keys=True)}")
                    for split in ("train", "validation"):
                        write_jsonl(OUTPUT / "data" / condition / interface / f"{split}.jsonl", records[split])
                    arm_dir = OUTPUT / "arm_outputs" / condition / interface / f"seed{seed}"
                    source_result = train_new(condition, interface, seed, records, arm_dir)
                    model = load_dense(arm_dir / "checkpoint.pt")
                    checkpoint = arm_dir / "checkpoint.pt"
                    rows = predict_rows(model, records["train"] + records["validation"], {"arm_id": key, "condition": condition, "interface": interface, "seed": seed, "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint)})
                # Recompute the authoritative prediction metrics from rows.
                metrics = condition_metrics(rows)
                source_result.update({"metrics": metrics, "gate_pass": bool(source_result["train_exact"] is not None and source_result["validation_exact"] is not None and source_result["train_exact"] >= TRAIN_THRESHOLD and source_result["validation_exact"] >= VALIDATION_THRESHOLD and metrics["validation_unequal"]["full_sequence_accuracy"] >= VALIDATION_THRESHOLD), "checkpoint_hash_match": source_result.get("source_checkpoint_hash", source_result["checkpoint_sha256"]) == source_result["checkpoint_sha256"]})
                arms.append(source_result)
                all_condition_rows.extend(rows)
                bundle = forensics.capture_records(model, records["train"] + records["validation"])
                probes = forensics.build_hidden_probe_results(bundle, records["train"] + records["validation"])
                answer_probes[key] = {state: probes[state]["output_position"] for state in ("layer_0", "layer_1", "layer_2", "layer_3", "final_norm")}
                output_sensitivity[key] = forensics.output_sensitivity(bundle, records["train"] + records["validation"])
                torch.save({"arm_id": key, "record_ids": bundle["record_ids"], "states": bundle["states"], "output_logits": bundle["output_logits"]}, OUTPUT / "hidden_states" / f"{key}.pt")
                del model
                gc.collect()
                print(json.dumps({"event": "complete", "arm_id": key, "validation": metrics["validation"]["accuracy"], "paired": metrics["validation_paired_unequal"]["both_correct"]}, sort_keys=True), flush=True)

    # Source-control checks: A0/A1 exact source rows and hashes are retained in
    # the summary; the newly trained arms have independent checkpoint hashes.
    write_csv(OUTPUT / "PREDICTIONS.csv", all_condition_rows)
    write_json(OUTPUT / "ANSWER_POSITION_PROBES.json", answer_probes)
    write_json(OUTPUT / "OUTPUT_SENSITIVITY.json", output_sensitivity)
    write_json(OUTPUT / "AUDITS.json", audits)
    verification_errors = [key for key, audit in audits.items() if not audit["pass"]]
    verification_errors += [key for key, result in source_reproduction.items() if not result["pass"]]
    verification_errors += [arm["arm_id"] for arm in arms if not arm.get("checkpoint_hash_match", True) or not arm.get("reported_metric_match", True)]
    summary = {"experiment": "crucible_operation_position_matrix_20251025_v1", "status": "completed", "arms": arms, "answer_position_probes": answer_probes, "output_sensitivity": output_sensitivity, "audits": audits, "source_reproduction": source_reproduction, "verification_errors": verification_errors, "prediction_rows": len(all_condition_rows), "sealed_test_accessed": False, "architecture_changed": False}
    write_json(OUTPUT / "SUMMARY.json", summary)
    (OUTPUT / "REPORT_FINAL.md").write_text(build_report(summary), encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt":
            hashes[rel(path)] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = {"status": "completed", "verification_pass": not verification_errors, "verification_errors": verification_errors, "new_training_arms": 4, "source_control_arms": 8, "sealed_test_accessed": False, "architecture_changed": False, "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"), "prediction_sha256": sha256(OUTPUT / "PREDICTIONS.csv")}
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "completed", "output": rel(OUTPUT), "arms": len(arms), "prediction_rows": len(all_condition_rows)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
