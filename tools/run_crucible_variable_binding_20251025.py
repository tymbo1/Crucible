#!/usr/bin/env python3
"""Run the CPU-only variable-binding and role-conditioned execution gate."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_variable_binding_20251025_v1"
SOURCE = ROOT / "tools" / "run_crucible_digitwise_diagonal_ablation_20251025.py"
INTERFACES = ("canonical", "factorized")
SEEDS = (2025, 2026)
ARMS = (
    "B0_original",
    "B1_role_binding_pretrained",
    "B2_role_explicit",
    "B3_binding_operation_conditioned",
)
ADDITION_STAGE = "single_digit_no_carry"
ROLE_STAGE = "role_binding_pretraining"
ADDITION_STEPS = 500
ROLE_STEPS = 500
TRAIN_THRESHOLD = 0.95
ROLE_VALIDATION_PAIRS = (
    (0, 9), (1, 8), (2, 7), (3, 6), (4, 0),
    (5, 1), (6, 2), (7, 3), (8, 4), (9, 5),
)
ROLE_TRAIN_PAIRS = (
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
    (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
    (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2),
    (3, 4), (4, 3), (4, 5), (5, 4), (5, 6), (6, 5),
    (6, 7), (7, 6), (7, 8), (8, 7), (8, 9), (9, 8),
    (0, 2), (2, 0), (1, 3), (3, 1), (2, 4), (4, 2),
    (3, 5), (5, 3), (4, 6), (6, 4), (5, 7), (7, 5),
)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


base = import_module("variable_binding_diagonal_runner", SOURCE)
ladder = base.ladder
stage_a = base.stage_a
torch = base.torch
balanced = base.base
PAIR_SET = set(base.PAIR_SET)
ADDITION_VALIDATION = set(base.VALIDATION_PAIRS["diagonal"])
ROLE_TRAIN_SET = set(ROLE_TRAIN_PAIRS)
ROLE_VALIDATION_SET = set(ROLE_VALIDATION_PAIRS)


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


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(dict(value), sort_keys=True) for value in values) + "\n",
        encoding="utf-8",
    )


def record_id(prefix: str, *parts: Any) -> str:
    material = "|".join([prefix, *[str(part) for part in parts]])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def addition_prompt(interface: str, arm: str, a: int, b: int) -> str:
    if arm in {"B0_original", "B1_role_binding_pretrained"}:
        if interface == "canonical":
            return f"Add integers with no carry. a={a}; b={b}. Return digits only."
        return f"Add integers with no carry. a_digit={a}; b_digit={b}. Return digits only."
    if arm == "B2_role_explicit":
        if interface == "canonical":
            return f"Add integers with no carry. first_operand={a}; second_operand={b}. Return digits only."
        return f"Add integers with no carry. first_operand_digit={a}; second_operand_digit={b}. Return digits only."
    if arm == "B3_binding_operation_conditioned":
        if interface == "canonical":
            return f"operation=ADD_NO_CARRY; first_operand={a}; second_operand={b}. Return digits only."
        return f"operation=ADD_NO_CARRY; first_operand_digit={a}; second_operand_digit={b}. Return digits only."
    raise ValueError(arm)


def addition_records(interface: str, arm: str) -> Dict[str, List[Dict[str, Any]]]:
    validation = set(ADDITION_VALIDATION)
    result: Dict[str, List[Dict[str, Any]]] = {}
    for split, selected in (("train", sorted(PAIR_SET - validation)), ("validation", sorted(validation))):
        values: List[Dict[str, Any]] = []
        for index, (a, b) in enumerate(selected):
            answer = a + b
            values.append({
                "schema_version": "crucible_variable_binding_addition_v1",
                "record_id": record_id("variable_binding_addition", arm, interface, split, index, a, b),
                "interface": interface,
                "stage": ADDITION_STAGE,
                "split": split,
                "condition": arm,
                "partition": "diagonal",
                "input": addition_prompt(interface, arm, a, b),
                "target": str(answer),
                "oracle": {"kind": "decimal_addition", "value": answer},
                "metadata": {
                    "a": a,
                    "b": b,
                    "answer": answer,
                    "output_length": 1,
                    "a_tens": 0,
                    "a_ones": a,
                    "b_tens": 0,
                    "b_ones": b,
                    "single_sum": answer,
                    "special_diagonal": [a, b] in ([2, 2], [3, 3]),
                    "operation": "ADD_NO_CARRY",
                    "first_role": "first_operand",
                    "second_role": "second_operand",
                },
            })
        result[split] = values
    return result


def role_prompt(interface: str, operation: str, a: int, b: int) -> str:
    if interface == "canonical":
        return f"Bind operand roles. operation={operation}; a={a}; b={b}. Return selected digit only."
    return f"Bind operand roles. operation={operation}; a_digit={a}; b_digit={b}. Return selected digit only."


def role_records(interface: str) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for split, selected in (("train", sorted(ROLE_TRAIN_SET)), ("validation", sorted(ROLE_VALIDATION_SET))):
        values: List[Dict[str, Any]] = []
        index = 0
        for a, b in selected:
            for operation, answer in (("SELECT_FIRST", a), ("SELECT_SECOND", b)):
                values.append({
                    "schema_version": "crucible_variable_binding_role_v1",
                    "record_id": record_id("variable_binding_role", interface, split, index, a, b, operation),
                    "interface": interface,
                    "stage": ROLE_STAGE,
                    "split": split,
                    "input": role_prompt(interface, operation, a, b),
                    "target": str(answer),
                    "oracle": {"kind": "role_selection", "operation": operation, "value": answer},
                    "metadata": {
                        "a": a,
                        "b": b,
                        "answer": answer,
                        "output_length": 1,
                        "operation": operation,
                        "selected_role": "first_operand" if operation == "SELECT_FIRST" else "second_operand",
                        "single_sum": a + b,
                    },
                })
                index += 1
        result[split] = values
    return result


def audit_addition(records: Mapping[str, Sequence[Mapping[str, Any]]], arm: str) -> List[str]:
    errors: List[str] = []
    train = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in records["train"]}
    validation = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in records["validation"]}
    if len(train) != 40 or len(validation) != 15 or train & validation or train | validation != PAIR_SET:
        errors.append(f"addition_split:{arm}")
    if not {(2, 2), (3, 3)} <= validation:
        errors.append(f"special_diagonal:{arm}")
    for split in ("train", "validation"):
        for row in records[split]:
            a, b = int(row["metadata"]["a"]), int(row["metadata"]["b"])
            if row.get("condition") != arm or row.get("partition") != "diagonal":
                errors.append(f"addition_metadata:{arm}:{row['record_id']}")
            if str(row["target"]) != str(a + b) or int(row["oracle"]["value"]) != a + b:
                errors.append(f"addition_label:{arm}:{row['record_id']}")
            if row["metadata"].get("operation") != "ADD_NO_CARRY":
                errors.append(f"addition_operation:{arm}:{row['record_id']}")
    return errors


def audit_roles(records: Mapping[str, Sequence[Mapping[str, Any]]], interface: str) -> List[str]:
    errors: List[str] = []
    train_pairs = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in records["train"]}
    validation_pairs = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in records["validation"]}
    if train_pairs != ROLE_TRAIN_SET or validation_pairs != ROLE_VALIDATION_SET or train_pairs & validation_pairs:
        errors.append(f"role_pair_split:{interface}")
    if len(records["train"]) != 80 or len(records["validation"]) != 20:
        errors.append(f"role_counts:{interface}")
    for split in ("train", "validation"):
        for row in records[split]:
            a, b = int(row["metadata"]["a"]), int(row["metadata"]["b"])
            operation = row["metadata"].get("operation")
            expected = a if operation == "SELECT_FIRST" else b if operation == "SELECT_SECOND" else None
            if operation not in {"SELECT_FIRST", "SELECT_SECOND"} or expected is None:
                errors.append(f"role_operation:{interface}:{row['record_id']}")
            elif int(row["target"]) != expected or int(row["oracle"]["value"]) != expected:
                errors.append(f"role_label:{interface}:{row['record_id']}")
            if row.get("interface") != interface:
                errors.append(f"role_interface:{interface}:{row['record_id']}")
    for role in ("a", "b"):
        train_values = {int(row["metadata"][role]) for row in records["train"]}
        validation_values = {int(row["metadata"][role]) for row in records["validation"]}
        if train_values != set(range(10)) or validation_values != set(range(10)):
            errors.append(f"role_coverage:{interface}:{role}")
    return errors


def evaluate_records(records: Sequence[Mapping[str, Any]], model: Any, grouping: str) -> Dict[str, Any]:
    prepared = ladder.prepare(records)
    aggregate = stage_a.evaluate(model, prepared)
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    if grouping == "role":
        for row in records:
            groups[str(row["metadata"]["operation"])].append(row)
    else:
        for row in records:
            groups[str(row["metadata"]["single_sum"])].append(row)
    grouped = {}
    for key, values in sorted(groups.items()):
        grouped[key] = {
            "examples": len(values),
            **stage_a.evaluate(model, ladder.prepare(values)),
        }
    return {"aggregate": aggregate, "by_group": grouped}


def train_model(
    seed: int,
    interface: str,
    stage: str,
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    output: Path,
    steps: int,
    initial_model: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    stage_a.torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    optimizer_reset = initial_model is not None
    if initial_model is not None:
        model.load_state_dict(initial_model, strict=True)
    prepared = {split: ladder.prepare(values) for split, values in records.items()}
    order = list(range(len(prepared["train"])))
    random.Random(ladder.stable_seed(seed, interface, stage, "order")).shuffle(order)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        stage_a.train_step(model, optimizer, prepared["train"], stage_a.batch_indices(order, step))
    grouping = "role" if stage == ROLE_STAGE else "addition"
    train_metrics = evaluate_records(records["train"], model, grouping)
    validation_metrics = evaluate_records(records["validation"], model, grouping)
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
        "optimizer_reset_from_pretraining": optimizer_reset,
        "checkpoint": str(checkpoint),
    }
    write_json(output / "RESULTS.json", result)
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_variable_binding_20251025",
        "interface": interface,
        "stage": stage,
        "seed": seed,
        "steps": steps,
        "gpu_used": False,
        "test_evaluated": False,
        "optimizer_reset_from_pretraining": optimizer_reset,
    })
    for split, values in records.items():
        write_jsonl(output / f"{split}.jsonl", values)
    del model, optimizer
    gc.collect()
    return result


def decode_output(token_id: int) -> Tuple[str, bool]:
    value = token_id - int(stage_a.RAW_BYTE_START)
    if 48 <= value <= 57:
        return chr(value), False
    return f"<token:{token_id}>", True


def predict_rows(model: Any, records: Sequence[Mapping[str, Any]], arm: Mapping[str, Any], task: str) -> List[Dict[str, Any]]:
    prepared = ladder.prepare(records)
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
                predicted_id = int(logits[local, output_position].argmax().item())
                predicted_output, format_error = decode_output(predicted_id)
                predicted_eos_id = int(logits[local, eos_position].argmax().item())
                target_id = int(batch["y"][local, output_position].item())
                target_eos_id = int(batch["y"][local, eos_position].item())
                token_correct = predicted_id == target_id
                eos_correct = predicted_eos_id == target_eos_id
                if int(batch["target_mask"][local].sum().item()) != 2:
                    raise RuntimeError(f"unexpected target mask for {item['record_id']}")
                row: Dict[str, Any] = {
                    **{key: arm[key] for key in ("arm_id", "interface", "seed", "checkpoint", "checkpoint_sha256")},
                    "task": task,
                    "record_id": item["record_id"],
                    "split": item["split"],
                    "a": int(item["metadata"]["a"]),
                    "b": int(item["metadata"]["b"]),
                    "operation": str(item["metadata"].get("operation", "ADD_NO_CARRY")),
                    "target_output": str(item["target"]),
                    "predicted_output": predicted_output,
                    "target_token_id": target_id,
                    "predicted_token_id": predicted_id,
                    "target_eos_token_id": target_eos_id,
                    "predicted_eos_token_id": predicted_eos_id,
                    "output_format_error": format_error,
                    "output_token_correct": token_correct,
                    "eos_correct": eos_correct,
                    "full_sequence_correct": token_correct and eos_correct,
                }
                if task == "addition":
                    row.update({
                        "expected_sum": int(item["metadata"]["a"]) + int(item["metadata"]["b"]),
                        "absolute_difference": abs(int(item["metadata"]["a"]) - int(item["metadata"]["b"])),
                        "signed_difference": int(item["metadata"]["a"]) - int(item["metadata"]["b"]),
                        "special_diagonal": bool(int(item["metadata"]["a"]) == int(item["metadata"]["b"])),
                    })
                else:
                    row.update({
                        "selected_role": str(item["metadata"]["selected_role"]),
                        "expected_selected_value": int(item["target"]),
                    })
                rows.append(row)
    return rows


def stat(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    output = sum(bool(row["output_token_correct"]) for row in rows)
    eos = sum(bool(row["eos_correct"]) for row in rows)
    full = sum(bool(row["full_sequence_correct"]) for row in rows)
    return {
        "observations": n,
        "output_token_correct": output,
        "output_token_accuracy": output / n if n else None,
        "eos_correct": eos,
        "eos_accuracy": eos / n if n else None,
        "full_sequence_correct": full,
        "full_sequence_accuracy": full / n if n else None,
        "output_format_errors": sum(bool(row["output_format_error"]) for row in rows),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_analysis(summary: Mapping[str, Any], addition_rows: Sequence[Mapping[str, Any]], role_rows: Sequence[Mapping[str, Any]], preflight: Mapping[str, Any]) -> Dict[str, Any]:
    arms = summary["arms"]
    per_arm: Dict[str, Any] = {}
    pair_summary: Dict[str, Any] = {}
    for arm in arms:
        aid = arm["arm_id"]
        rows = [row for row in addition_rows if row["arm_id"] == aid]
        train_rows = [row for row in rows if row["split"] == "train"]
        val_rows = [row for row in rows if row["split"] == "validation"]
        diagonal = [row for row in val_rows if int(row["a"]) == int(row["b"])]
        unequal = [row for row in val_rows if int(row["a"]) != int(row["b"])]
        per_pair: Dict[str, Any] = {}
        for row in val_rows:
            key = f"({row['a']},{row['b']})"
            per_pair[key] = {
                "correct": bool(row["output_token_correct"]),
                "predicted_output": row["predicted_output"],
                "expected_output": row["target_output"],
            }
        pair_summary[aid] = per_pair
        per_arm[aid] = {
            "arm": arm["arm"],
            "interface": arm["interface"],
            "seed": arm["seed"],
            "fit_pass": arm["fit_pass"],
            "train": stat(train_rows),
            "validation": stat(val_rows),
            "validation_diagonal": stat(diagonal),
            "validation_unequal": stat(unequal),
        }

    role_by_arm: Dict[str, Any] = {}
    for arm in arms:
        if arm["arm"] not in {"B1_role_binding_pretrained", "B3_binding_operation_conditioned"}:
            continue
        aid = arm["arm_id"]
        rows = [row for row in role_rows if row["arm_id"] == aid]
        role_by_arm[aid] = {
            "train": stat([row for row in rows if row["split"] == "train"]),
            "validation": stat([row for row in rows if row["split"] == "validation"]),
            "operations": {
                operation: stat([row for row in rows if row["operation"] == operation])
                for operation in ("SELECT_FIRST", "SELECT_SECOND")
            },
        }

    errors: List[str] = []
    if len(arms) != len(ARMS) * len(INTERFACES) * len(SEEDS):
        errors.append("arm_count")
    if len(addition_rows) != len(arms) * 55:
        errors.append("addition_prediction_count")
    expected_role_arms = sum(arm["arm"] in {"B1_role_binding_pretrained", "B3_binding_operation_conditioned"} for arm in arms)
    if len(role_rows) != expected_role_arms * 100:
        errors.append("role_prediction_count")
    if any(bool(row["output_format_error"]) for row in addition_rows + list(role_rows)):
        errors.append("output_format_errors")
    if any(not Path(ROOT / arm["checkpoint"]).is_file() for arm in arms):
        errors.append("missing_addition_checkpoint")
    if any(not Path(ROOT / arm["checkpoint"]).is_file() for arm in arms if arm.get("checkpoint")):
        errors.append("checkpoint_path")
    for arm in arms:
        if arm["arm"] in {"B1_role_binding_pretrained", "B3_binding_operation_conditioned"} and not arm.get("pretraining", {}).get("checkpoint"):
            errors.append(f"missing_pretraining:{arm['arm_id']}")
    fit_pass = sum(bool(arm["fit_pass"]) for arm in arms)
    role_fit_pass = sum(
        bool(arm.get("pretraining", {}).get("fit_pass"))
        and bool(arm.get("pretraining", {}).get("validation_fit_pass"))
        for arm in arms
        if arm["arm"] in {"B1_role_binding_pretrained", "B3_binding_operation_conditioned"}
    )
    return {
        "experiment": "crucible_variable_binding_20251025",
        "status": "completed",
        "total_arms": len(arms),
        "fit_pass_arms": fit_pass,
        "role_pretraining_arms_passing_both_gates": role_fit_pass,
        "addition_rows": len(addition_rows),
        "role_rows": len(role_rows),
        "per_arm": per_arm,
        "role_pretraining": role_by_arm,
        "pair_summary": pair_summary,
        "verification": {
            "pass": not errors and bool(preflight.get("pass")),
            "errors": errors,
            "preflight_pass": bool(preflight.get("pass")),
            "sealed_test_accessed": False,
            "architecture_changed": False,
        },
    }


def build_report(analysis: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    rows = [
        "# Crucible Variable Binding — Final Analysis",
        "",
        f"CPU-only, {analysis['total_arms']} arms, frozen diagonal addition split, 500 addition updates per arm. No sealed-test access.",
        "",
        f"Addition fit gate: {analysis['fit_pass_arms']}/{analysis['total_arms']} arms passed. Role-pretraining train/validation gate: {analysis['role_pretraining_arms_passing_both_gates']}/8 pretrained arms passed. Verification: {'PASS' if analysis['verification']['pass'] else 'FAIL'}.",
        "",
        "## Addition results",
        "",
        "| Arm | Interface | Seed | Train | Validation | Diagonal | Unequal |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm in summary["arms"]:
        metrics = analysis["per_arm"][arm["arm_id"]]
        rows.append(
            f"| {arm['arm']} | {arm['interface']} | {arm['seed']} | "
            f"{metrics['train']['output_token_accuracy']:.1%} ({metrics['train']['output_token_correct']}/{metrics['train']['observations']}) | "
            f"{metrics['validation']['output_token_accuracy']:.1%} ({metrics['validation']['output_token_correct']}/{metrics['validation']['observations']}) | "
            f"{metrics['validation_diagonal']['output_token_correct']}/{metrics['validation_diagonal']['observations']} | "
            f"{metrics['validation_unequal']['output_token_correct']}/{metrics['validation_unequal']['observations']} |"
        )
    rows += [
        "",
        "## Role-binding pretraining",
        "",
        "The auxiliary task required selecting the first or second operand on disjoint ordered-pair examples. It contained no addition targets. Pretrained arms reset the addition optimizer before fine-tuning.",
        "",
        "| Arm | Interface | Seed | Role train | Role validation | Addition optimizer reset |",
        "|---|---|---:|---:|---:|---|",
    ]
    for arm in summary["arms"]:
        if arm["arm"] not in {"B1_role_binding_pretrained", "B3_binding_operation_conditioned"}:
            continue
        aid = arm["arm_id"]
        role = analysis["role_pretraining"][aid]
        rows.append(
            f"| {arm['arm']} | {arm['interface']} | {arm['seed']} | "
            f"{role['train']['output_token_accuracy']:.1%} ({role['train']['output_token_correct']}/{role['train']['observations']}) | "
            f"{role['validation']['output_token_accuracy']:.1%} ({role['validation']['output_token_correct']}/{role['validation']['observations']}) | yes |"
        )
    rows += [
        "",
        "## Interpretation boundary",
        "",
        "Training competence is not held-out transfer. A pretrained arm that fails its role-binding gate is not evidence that a learned binding skill cannot transfer. A role-explicit or operation-conditioned gain would be a representation result, not evidence that the model independently discovered the relation.",
        "",
        "## Artifacts",
        "",
        "- [Protocol](../../docs/CRUCIBLE_VARIABLE_BINDING_PROTOCOL_20251025.md)",
        "- [Analysis](ANALYSIS_FINAL.json)",
        "- [Addition predictions](PREDICTIONS.csv)",
        "- [Role-binding predictions](ROLE_BINDING_PREDICTIONS.csv)",
        "- [Summary](SUMMARY.json)",
        "- [Preflight](PREFLIGHT.json)",
    ]
    return "\n".join(rows) + "\n"


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base.OUTPUT = OUTPUT
    base.ANCHOR_DIR = OUTPUT / "generated_reference_e4" / "crucible_manifest"

    addition_by_arm: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {
        arm: {interface: addition_records(interface, arm) for interface in INTERFACES}
        for arm in ARMS
    }
    role_by_interface = {interface: role_records(interface) for interface in INTERFACES}
    errors: List[str] = []
    for arm in ARMS:
        for interface in INTERFACES:
            errors.extend(audit_addition(addition_by_arm[arm][interface], arm))
    for interface in INTERFACES:
        errors.extend(audit_roles(role_by_interface[interface], interface))

    preflight = {
        "experiment": "crucible_variable_binding_20251025",
        "status": "frozen_before_training",
        "arms": list(ARMS),
        "interfaces": list(INTERFACES),
        "seeds": list(SEEDS),
        "addition_validation_pairs": [list(pair) for pair in sorted(ADDITION_VALIDATION)],
        "addition_special_diagonal_pairs": [[2, 2], [3, 3]],
        "role_train_pairs": [list(pair) for pair in sorted(ROLE_TRAIN_SET)],
        "role_validation_pairs": [list(pair) for pair in sorted(ROLE_VALIDATION_SET)],
        "role_train_record_count_per_interface": 80,
        "role_validation_record_count_per_interface": 20,
        "role_steps": ROLE_STEPS,
        "addition_steps": ADDITION_STEPS,
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
            addition_path = OUTPUT / "data" / arm / interface
            write_jsonl(addition_path / "addition_train.jsonl", addition_by_arm[arm][interface]["train"])
            write_jsonl(addition_path / "addition_validation.jsonl", addition_by_arm[arm][interface]["validation"])
            if arm in {"B1_role_binding_pretrained", "B3_binding_operation_conditioned"}:
                role_path = OUTPUT / "data" / "role_binding" / interface
                write_jsonl(role_path / "train.jsonl", role_by_interface[interface]["train"])
                write_jsonl(role_path / "validation.jsonl", role_by_interface[interface]["validation"])

    anchor_hashes = base.prepare_fresh_anchors()
    write_json(OUTPUT / "INITIAL_ANCHORS.json", {
        "directory": rel(base.ANCHOR_DIR),
        "seeds": list(SEEDS),
        "sha256": anchor_hashes,
        "generated_without_training": True,
    })
    preflight["training_performed"] = False
    preflight["initial_anchor_sha256"] = anchor_hashes
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    print(json.dumps({"status": "preflight_passed", "arms_expected": len(ARMS) * len(INTERFACES) * len(SEEDS)}, sort_keys=True), flush=True)

    arms_summary: List[Dict[str, Any]] = []
    addition_rows: List[Dict[str, Any]] = []
    role_rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for interface in INTERFACES:
            for seed in SEEDS:
                aid = f"{arm}_{interface}_seed{seed}"
                arm_root = OUTPUT / "arm_outputs" / arm / interface / f"seed{seed}"
                pretrained_result = None
                pretrained_state = None
                if arm in {"B1_role_binding_pretrained", "B3_binding_operation_conditioned"}:
                    pre_output = arm_root / "role_binding_pretraining"
                    print(json.dumps({"event": "start_pretraining", "arm_id": aid}, sort_keys=True), flush=True)
                    pretrained_result = train_model(seed, interface, ROLE_STAGE, role_by_interface[interface], pre_output, ROLE_STEPS)
                    pre_checkpoint = pre_output / "checkpoint.pt"
                    state = torch.load(pre_checkpoint, map_location="cpu", weights_only=False)
                    pretrained_state = state["model"]
                    del state
                addition_output = arm_root / "addition"
                print(json.dumps({"event": "start_addition", "arm_id": aid}, sort_keys=True), flush=True)
                result = train_model(
                    seed,
                    interface,
                    ADDITION_STAGE,
                    addition_by_arm[arm][interface],
                    addition_output,
                    ADDITION_STEPS,
                    pretrained_state,
                )
                checkpoint = addition_output / "checkpoint.pt"
                result.update({
                    "arm_id": aid,
                    "arm": arm,
                    "interface": interface,
                    "seed": seed,
                    "checkpoint": rel(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "fit_pass": bool(result["train_exact"] >= TRAIN_THRESHOLD),
                    "pretraining": None if pretrained_result is None else {
                        "steps": ROLE_STEPS,
                        "train_exact": pretrained_result["train_exact"],
                        "validation_exact": pretrained_result["validation_exact"],
                        "fit_pass": pretrained_result["train_exact"] >= TRAIN_THRESHOLD,
                        "validation_fit_pass": pretrained_result["validation_exact"] >= TRAIN_THRESHOLD,
                        "checkpoint": rel(arm_root / "role_binding_pretraining" / "checkpoint.pt"),
                        "checkpoint_sha256": sha256(arm_root / "role_binding_pretraining" / "checkpoint.pt"),
                        "elapsed_seconds": pretrained_result["elapsed_seconds"],
                    },
                })
                write_json(addition_output / "RESULTS.json", result)
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = stage_a.make_dense(seed)
                model.load_state_dict(state["model"], strict=True)
                prediction_arm = {
                    "arm_id": aid,
                    "interface": interface,
                    "seed": seed,
                    "checkpoint": rel(checkpoint),
                    "checkpoint_sha256": result["checkpoint_sha256"],
                }
                add_records = addition_by_arm[arm][interface]["train"] + addition_by_arm[arm][interface]["validation"]
                rows = predict_rows(model, add_records, prediction_arm, "addition")
                addition_rows.extend(rows)
                if pretrained_result is not None:
                    pre_checkpoint = arm_root / "role_binding_pretraining" / "checkpoint.pt"
                    pre_state = torch.load(pre_checkpoint, map_location="cpu", weights_only=False)
                    pre_model = stage_a.make_dense(seed)
                    pre_model.load_state_dict(pre_state["model"], strict=True)
                    role_arm = dict(prediction_arm)
                    role_arm["checkpoint"] = rel(pre_checkpoint)
                    role_arm["checkpoint_sha256"] = sha256(pre_checkpoint)
                    role_rows.extend(predict_rows(pre_model, role_by_interface[interface]["train"] + role_by_interface[interface]["validation"], role_arm, "role_binding"))
                    write_json(arm_root / "role_binding_pretraining" / "FORENSIC.json", {
                        "results": pretrained_result,
                        "rows": [row for row in role_rows if row["arm_id"] == aid],
                    })
                    del pre_model, pre_state
                write_json(addition_output / "FORENSIC.json", {
                    "results": result,
                    "rows": [row for row in addition_rows if row["arm_id"] == aid],
                })
                summary_arm = {
                    "arm_id": aid,
                    "arm": arm,
                    "interface": interface,
                    "seed": seed,
                    "fit_pass": result["fit_pass"],
                    "train": stat([row for row in rows if row["split"] == "train"]),
                    "validation": stat([row for row in rows if row["split"] == "validation"]),
                    "checkpoint": result["checkpoint"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "pretraining": result["pretraining"],
                }
                arms_summary.append(summary_arm)
                del model, state, pretrained_state
                gc.collect()
                print(json.dumps({"event": "complete", "arm_id": aid, "train": result["train_exact"], "validation": result["validation_exact"]}, sort_keys=True), flush=True)

    write_csv(OUTPUT / "PREDICTIONS.csv", addition_rows)
    write_csv(OUTPUT / "ROLE_BINDING_PREDICTIONS.csv", role_rows)
    summary = {
        "experiment": "crucible_variable_binding_20251025",
        "status": "completed",
        "training_performed": True,
        "sealed_test_accessed": False,
        "architecture_changed": False,
        "arms": arms_summary,
        "addition_prediction_rows": len(addition_rows),
        "role_prediction_rows": len(role_rows),
        "role_steps": ROLE_STEPS,
        "addition_steps": ADDITION_STEPS,
    }
    write_json(OUTPUT / "SUMMARY.json", summary)
    analysis = build_analysis(summary, addition_rows, role_rows, preflight)
    write_json(OUTPUT / "PAIR_SUMMARY.json", analysis["pair_summary"])
    write_json(OUTPUT / "ANALYSIS_FINAL.json", analysis)
    (OUTPUT / "REPORT_FINAL.md").write_text(build_report(analysis, summary), encoding="utf-8")
    preflight["training_performed"] = True
    preflight["status"] = "completed"
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt":
            hashes[rel(path)] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    write_json(OUTPUT / "FINALIZATION.json", {
        "status": "completed",
        "verification_pass": analysis["verification"]["pass"],
        "analysis_sha256": sha256(OUTPUT / "ANALYSIS_FINAL.json"),
        "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"),
        "addition_prediction_sha256": sha256(OUTPUT / "PREDICTIONS.csv"),
        "role_prediction_sha256": sha256(OUTPUT / "ROLE_BINDING_PREDICTIONS.csv"),
    })
    print(json.dumps({"status": "completed", "arms": len(arms_summary), "addition_rows": len(addition_rows), "role_rows": len(role_rows), "output": rel(OUTPUT)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
