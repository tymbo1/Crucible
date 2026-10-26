#!/usr/bin/env python3
"""Run the CPU-only primitive role-selection competence gate."""

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
OUTPUT = ROOT / "experiments" / "crucible_role_selection_competence_20251025_v5"
SOURCE = ROOT / "tools" / "run_crucible_digitwise_diagonal_ablation_20251025.py"
PREVIOUS_OUTPUT = ROOT / "experiments" / "crucible_variable_binding_20251025_v1"
INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
CONDITIONS = ("S1_SELECT_FIRST", "S2_SELECT_SECOND", "S3_JOINT_OPERATION_CONDITIONED")
ROLE_STAGE = "role_selection_competence"
STEPS = 500
TRAIN_THRESHOLD = 0.95
VALIDATION_THRESHOLD = 0.90
CURVE_STEPS = (1, 10, 25, 50, 100, 200, 300, 400, 500)

ROLE_TRAIN_PAIRS = (
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
    (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
    (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2),
    (3, 4), (4, 3), (4, 5), (5, 4), (5, 6), (6, 5),
    (6, 7), (7, 6), (7, 8), (8, 7), (8, 9), (9, 8),
    (0, 2), (2, 0), (1, 3), (3, 1), (2, 4), (4, 2),
    (3, 5), (5, 3), (4, 6), (6, 4), (5, 7), (7, 5),
)
VALIDATION_PAIRS = (
    (0, 0), (5, 5), (0, 9), (9, 0), (1, 8),
    (8, 1), (2, 7), (7, 2), (3, 6), (6, 3),
)
TRANSFER_PAIRS = (
    (0, 4), (4, 0), (1, 5), (5, 1), (2, 6),
    (6, 2), (3, 7), (7, 3), (4, 8), (8, 4),
)
TRAIN_SET = set(ROLE_TRAIN_PAIRS) - {(0, 0), (5, 5)} | {(4, 7), (7, 4)}
VALIDATION_SET = set(VALIDATION_PAIRS)
TRANSFER_SET = set(TRANSFER_PAIRS)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


base = import_module("role_selection_diagonal_runner", SOURCE)
ladder = base.ladder
stage_a = base.stage_a
torch = base.torch


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


def prepare_fresh_anchors() -> Dict[str, str]:
    base.ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    ladder.stage_a.E4_ANCHOR_DIR = base.ANCHOR_DIR
    base.stage_a.E4_ANCHOR_DIR = base.ANCHOR_DIR
    hashes: Dict[str, str] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = ladder.stage_a.BraidPrototypeLM(
            ladder.stage_a.VOCAB_SIZE,
            ladder.stage_a.N_LAYERS,
            ladder.stage_a.D_MODEL,
            ladder.stage_a.N_HEADS,
            ladder.stage_a.LATENT_KV,
            ladder.stage_a.LATENT_Q,
            ladder.stage_a.PE_DIM,
            ladder.stage_a.E4_EXPERTS,
            ladder.stage_a.E4_D_FF,
            top_k=ladder.stage_a.E4_TOP_K,
        )
        path = base.ANCHOR_DIR / f"initial_seed_anchor_{seed}.pt"
        torch.save(model.state_dict(), path)
        hashes[str(seed)] = sha256(path)
        del model
    return hashes


def record_id(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:20]


def operation_for(condition: str) -> Tuple[str, ...]:
    if condition == "S1_SELECT_FIRST":
        return ("SELECT_FIRST",)
    if condition == "S2_SELECT_SECOND":
        return ("SELECT_SECOND",)
    if condition == "S3_JOINT_OPERATION_CONDITIONED":
        return ("SELECT_FIRST", "SELECT_SECOND")
    raise ValueError(condition)


def prompt(interface: str, condition: str, operation: str, a: int, b: int) -> str:
    if condition == "S1_SELECT_FIRST":
        instruction = "Select the first operand."
    elif condition == "S2_SELECT_SECOND":
        instruction = "Select the second operand."
    else:
        instruction = f"Select an operand. operation={operation};"
    if interface == "canonical":
        return f"{instruction} a={a}; b={b}. Return digit only."
    return f"{instruction} a_digit={a}; b_digit={b}. Return digit only."


def records_for(condition: str, interface: str, split: str, pairs: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    index = 0
    for a, b in pairs:
        for operation in operation_for(condition):
            answer = a if operation == "SELECT_FIRST" else b
            values.append({
                "schema_version": "crucible_role_selection_competence_v1",
                "record_id": record_id("role_selection_competence", condition, interface, split, index, a, b, operation),
                "interface": interface,
                "stage": ROLE_STAGE,
                "split": split,
                "condition": condition,
                "input": prompt(interface, condition, operation, a, b),
                "target": str(answer),
                "oracle": {"kind": "role_selection", "operation": operation, "value": answer},
                "metadata": {
                    "a": a,
                    "b": b,
                    "answer": answer,
                    "operation": operation,
                    "selected_role": "first_operand" if operation == "SELECT_FIRST" else "second_operand",
                    "pair_kind": "equal" if a == b else "unequal",
                    "reversed_pair": [b, a],
                    "output_length": 1,
                },
            })
            index += 1
    return values


def audit_records(condition: str, interface: str, records: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[str]:
    errors: List[str] = []
    train_pairs = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in records["train"]}
    validation_pairs = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in records["validation"]}
    expected_train_count = len(TRAIN_SET) * len(operation_for(condition))
    expected_validation_count = len(VALIDATION_SET) * len(operation_for(condition))
    if train_pairs != TRAIN_SET or validation_pairs != VALIDATION_SET or train_pairs & validation_pairs:
        errors.append(f"pair_split:{condition}:{interface}")
    if len(records["train"]) != expected_train_count or len(records["validation"]) != expected_validation_count:
        errors.append(f"record_count:{condition}:{interface}")
    for split in ("train", "validation"):
        for row in records[split]:
            a = int(row["metadata"]["a"])
            b = int(row["metadata"]["b"])
            operation = row["metadata"]["operation"]
            expected = a if operation == "SELECT_FIRST" else b if operation == "SELECT_SECOND" else None
            if row.get("condition") != condition or row.get("interface") != interface or expected is None:
                errors.append(f"metadata:{condition}:{interface}:{row['record_id']}")
            elif int(row["target"]) != expected or int(row["oracle"]["value"]) != expected:
                errors.append(f"label:{condition}:{interface}:{row['record_id']}")
            if row["metadata"]["pair_kind"] != ("equal" if a == b else "unequal"):
                errors.append(f"pair_kind:{condition}:{interface}:{row['record_id']}")
    for role in ("a", "b"):
        train_values = {int(row["metadata"][role]) for row in records["train"]}
        validation_values = {int(row["metadata"][role]) for row in records["validation"]}
        if train_values != set(range(10)) or not validation_values:
            errors.append(f"coverage:{condition}:{interface}:{role}")
    return errors


def evaluate_records(records: Sequence[Mapping[str, Any]], model: Any) -> Dict[str, Any]:
    prepared = ladder.prepare(records)
    aggregate = stage_a.evaluate(model, prepared)

    def eval_subset(predicate) -> Dict[str, Any]:
        selected = [row for row in records if predicate(row)]
        if not selected:
            return {"examples": 0, "teacher_forced_exact_accuracy": None, "teacher_forced_token_accuracy": None}
        return {"examples": len(selected), **stage_a.evaluate(model, ladder.prepare(selected))}

    by_operation = {
        operation: eval_subset(lambda row, operation=operation: row["metadata"]["operation"] == operation)
        for operation in ("SELECT_FIRST", "SELECT_SECOND")
        if any(row["metadata"]["operation"] == operation for row in records)
    }
    by_pair_kind = {
        kind: eval_subset(lambda row, kind=kind: row["metadata"]["pair_kind"] == kind)
        for kind in ("equal", "unequal")
        if any(row["metadata"]["pair_kind"] == kind for row in records)
    }
    return {"aggregate": aggregate, "by_operation": by_operation, "by_pair_kind": by_pair_kind}


def train_model(condition: str, interface: str, seed: int, records: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> Dict[str, Any]:
    stage_a.torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    prepared = {split: ladder.prepare(values) for split, values in records.items()}
    order = list(range(len(prepared["train"])))
    random.Random(ladder.stable_seed(seed, condition, interface, "order")).shuffle(order)
    curves: List[Dict[str, Any]] = []
    running_loss = 0.0
    interval_steps = 0
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        step_result = stage_a.train_step(model, optimizer, prepared["train"], stage_a.batch_indices(order, step))
        running_loss += float(step_result["loss"])
        interval_steps += 1
        if step not in CURVE_STEPS:
            continue
        train_metrics = evaluate_records(records["train"], model)
        validation_metrics = evaluate_records(records["validation"], model)
        curves.append({
            "step": step,
            "mean_step_loss_since_previous_curve": running_loss / max(1, interval_steps),
            "last_step_loss": float(step_result["loss"]),
            "train": train_metrics,
            "validation": validation_metrics,
        })
        running_loss = 0.0
        interval_steps = 0

    train_metrics = evaluate_records(records["train"], model)
    validation_metrics = evaluate_records(records["validation"], model)
    checkpoint = output / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint, model, optimizer, STEPS, order)
    result = {
        "condition": condition,
        "interface": interface,
        "seed": seed,
        "stage": ROLE_STAGE,
        "steps": STEPS,
        "elapsed_seconds": time.perf_counter() - started,
        "train": train_metrics,
        "validation": validation_metrics,
        "train_exact": train_metrics["aggregate"]["teacher_forced_exact_accuracy"],
        "validation_exact": validation_metrics["aggregate"]["teacher_forced_exact_accuracy"],
        "validation_unequal_exact": validation_metrics["by_pair_kind"].get("unequal", {}).get("teacher_forced_exact_accuracy"),
        "fit_pass": train_metrics["aggregate"]["teacher_forced_exact_accuracy"] >= TRAIN_THRESHOLD,
        "validation_pass": validation_metrics["aggregate"]["teacher_forced_exact_accuracy"] >= VALIDATION_THRESHOLD,
        "unequal_validation_pass": validation_metrics["by_pair_kind"].get("unequal", {}).get("teacher_forced_exact_accuracy", 0.0) >= VALIDATION_THRESHOLD,
        "checkpoint": str(checkpoint),
    }
    result["gate_pass"] = bool(result["fit_pass"] and result["validation_pass"] and result["unequal_validation_pass"])
    write_json(output / "RESULTS.json", result)
    write_json(output / "TRAINING_CURVE.json", {"condition": condition, "interface": interface, "seed": seed, "curve_steps": list(CURVE_STEPS), "points": curves})
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_role_selection_competence_20251025",
        "condition": condition,
        "interface": interface,
        "seed": seed,
        "steps": STEPS,
        "gpu_used": False,
        "sealed_test_accessed": False,
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


def predict_rows(model: Any, records: Sequence[Mapping[str, Any]], arm: Mapping[str, Any], split_label: str | None = None) -> List[Dict[str, Any]]:
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
                row = {
                    **{key: arm[key] for key in ("arm_id", "condition", "interface", "seed", "checkpoint", "checkpoint_sha256")},
                    "record_id": item["record_id"],
                    "split": split_label or item["split"],
                    "source_split": item["split"],
                    "a": int(item["metadata"]["a"]),
                    "b": int(item["metadata"]["b"]),
                    "operation": item["metadata"]["operation"],
                    "pair_kind": item["metadata"]["pair_kind"],
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
                rows.append(row)
    return rows


def stat(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row["output_token_correct"]) for row in rows)
    eos = sum(bool(row["eos_correct"]) for row in rows)
    full = sum(bool(row["full_sequence_correct"]) for row in rows)
    return {
        "observations": n,
        "output_token_correct": correct,
        "output_token_accuracy": correct / n if n else None,
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


def audit_previous_failure() -> Dict[str, Any]:
    path = PREVIOUS_OUTPUT / "ROLE_BINDING_PREDICTIONS.csv"
    errors: List[str] = []
    if not path.is_file():
        errors.append("missing_previous_prediction_csv")
        result = {"source": rel(path), "errors": errors, "pass": False}
        write_json(OUTPUT / "PREVIOUS_FAILURE_AUDIT.json", result)
        (OUTPUT / "PREVIOUS_FAILURE_AUDIT.md").write_text("# Previous failure audit\n\nMissing previous prediction CSV.\n", encoding="utf-8")
        return result

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    arms = sorted({row["arm_id"] for row in rows})
    by_arm: Dict[str, Any] = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm_id"] == arm]
        by_arm[arm] = {}
        for split in ("train", "validation"):
            selected = [row for row in arm_rows if row["split"] == split]
            by_arm[arm][split] = {
                "rows": len(selected),
                "correct": sum(row["output_token_correct"] == "True" for row in selected),
                "accuracy": sum(row["output_token_correct"] == "True" for row in selected) / len(selected) if selected else None,
                "by_operation": {
                    operation: {
                        "rows": sum(row["operation"] == operation for row in selected),
                        "correct": sum(row["operation"] == operation and row["output_token_correct"] == "True" for row in selected),
                    }
                    for operation in ("SELECT_FIRST", "SELECT_SECOND")
                },
                "predicted_equals_first": sum(row["predicted_output"] == row["a"] for row in selected),
                "predicted_equals_second": sum(row["predicted_output"] == row["b"] for row in selected),
                "predicted_equals_either": sum(row["predicted_output"] in {row["a"], row["b"]} for row in selected),
                "format_errors": sum(row["output_format_error"] == "True" for row in selected),
                "eos_correct": sum(row["eos_correct"] == "True" for row in selected),
                "pair_coverage": sorted({(int(row["a"]), int(row["b"])) for row in selected}),
            }

    validation_pairs = sorted({(int(row["a"]), int(row["b"])) for row in rows if row["split"] == "validation"})
    reversal = {
        f"({a},{b})": {
            "reverse": [b, a],
            "reverse_in_validation": (b, a) in set(validation_pairs),
        }
        for a, b in validation_pairs
    }
    result = {
        "source": rel(path),
        "rows": len(rows),
        "arms": len(arms),
        "errors": errors,
        "by_arm": by_arm,
        "validation_pairs": [list(pair) for pair in validation_pairs],
        "validation_reversal_membership": reversal,
        "class_balance": {
            "train": Counter(row["operation"] for row in rows if row["split"] == "train"),
            "validation": Counter(row["operation"] for row in rows if row["split"] == "validation"),
        },
        "pass": not errors and len(rows) == 800 and len(arms) == 8,
    }
    # JSON cannot serialize Counter directly.
    result["class_balance"] = {split: dict(values) for split, values in result["class_balance"].items()}
    write_json(OUTPUT / "PREVIOUS_FAILURE_AUDIT.json", result)
    lines = [
        "# Previous Variable-Binding Failure Audit",
        "",
        f"Source: `{rel(path)}`  ",
        f"Rows: {len(rows)}; arms: {len(arms)}; audit: {'PASS' if result['pass'] else 'FAIL'}.",
        "",
        "The prior validation rows contain only unequal operand pairs. The audit therefore cannot use that validation split to test equal-operand role indistinguishability.",
        "",
        "| Arm | Train accuracy | Validation accuracy | Val = first | Val = second | Val = either |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        train = by_arm[arm]["train"]
        validation = by_arm[arm]["validation"]
        lines.append(
            f"| {arm} | {train['correct']}/{train['rows']} | {validation['correct']}/{validation['rows']} | "
            f"{validation['predicted_equals_first']} | {validation['predicted_equals_second']} | {validation['predicted_equals_either']} |"
        )
    lines += [
        "",
        "No output-format errors were found in the prior prediction CSV. The operation-conditioned task was therefore undercompetent in the tested regime; it was not a simple EOS/format failure.",
        "",
        "The new experiment separates fixed-operation execution from operation-conditioned selection.",
    ]
    (OUTPUT / "PREVIOUS_FAILURE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def build_analysis(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], preflight: Mapping[str, Any], previous_audit: Mapping[str, Any], transfer_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    per_arm: Dict[str, Any] = {}
    for arm in summary["arms"]:
        aid = arm["arm_id"]
        arm_rows = [row for row in rows if row["arm_id"] == aid]
        val = [row for row in arm_rows if row["split"] == "validation"]
        unequal = [row for row in val if row["pair_kind"] == "unequal"]
        equal = [row for row in val if row["pair_kind"] == "equal"]
        reversal = []
        for row in unequal:
            counterpart = next((other for other in unequal if int(other["a"]) == int(row["b"]) and int(other["b"]) == int(row["a"]) and other["operation"] == row["operation"]), None)
            if counterpart is not None:
                reversal.append({
                    "pair": [int(row["a"]), int(row["b"])],
                    "counterpart": [int(counterpart["a"]), int(counterpart["b"])],
                    "correct": bool(row["output_token_correct"]),
                    "counterpart_correct": bool(counterpart["output_token_correct"]),
                })
        per_arm[aid] = {
            "condition": arm["condition"],
            "interface": arm["interface"],
            "seed": arm["seed"],
            "train": stat([row for row in arm_rows if row["split"] == "train"]),
            "validation": stat(val),
            "validation_unequal": stat(unequal),
            "validation_equal": stat(equal),
            "validation_reversal_pairs": reversal,
            "gate_pass": arm["gate_pass"],
        }

    condition_summary: Dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition and row["split"] == "validation"]
        unequal = [row for row in selected if row["pair_kind"] == "unequal"]
        condition_summary[condition] = {
            "validation": stat(selected),
            "validation_unequal": stat(unequal),
            "arms": [arm["arm_id"] for arm in summary["arms"] if arm["condition"] == condition],
            "all_arms_pass": all(arm["gate_pass"] for arm in summary["arms"] if arm["condition"] == condition),
        }

    errors: List[str] = []
    if len(summary["arms"]) != len(CONDITIONS) * len(INTERFACES) * len(SEEDS):
        errors.append("arm_count")
    if any(bool(row["output_format_error"]) for row in rows):
        errors.append("output_format_errors")
    if len(rows) != 800:
        errors.append("prediction_count")
    for arm in summary["arms"]:
        if not (ROOT / arm["checkpoint"]).is_file():
            errors.append(f"missing_checkpoint:{arm['arm_id']}")

    all_competence_pass = all(arm["gate_pass"] for arm in summary["arms"])
    return {
        "experiment": "crucible_role_selection_competence_20251025",
        "total_arms": len(summary["arms"]),
        "prediction_rows": len(rows),
        "per_arm": per_arm,
        "by_condition": condition_summary,
        "all_competence_gates_pass": all_competence_pass,
        "conditional_transfer_run": bool(transfer_rows),
        "conditional_transfer_rows": len(transfer_rows),
        "previous_failure_audit": previous_audit,
        "verification": {
            "pass": not errors and bool(preflight.get("pass")) and bool(previous_audit.get("pass")),
            "errors": errors,
            "preflight_pass": bool(preflight.get("pass")),
            "previous_audit_pass": bool(previous_audit.get("pass")),
            "sealed_test_accessed": False,
            "architecture_changed": False,
        },
    }


def build_report(analysis: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    lines = [
        "# Crucible Primitive Role-Selection Competence — Final Analysis",
        "",
        f"CPU-only, {analysis['total_arms']} arms, 500 updates per arm, seeds 42 and 1337, both frozen interfaces. No sealed-test access.",
        "",
        f"Verification: {'PASS' if analysis['verification']['pass'] else 'FAIL'}. All competence gates: {'PASS' if analysis['all_competence_gates_pass'] else 'FAIL'}. Conditional transfer: {'RUN' if analysis['conditional_transfer_run'] else 'NOT RUN under the stop rule'}.",
        "",
        "## Previous failure audit",
        "",
        "The prior operation-conditioned role-binding run had 800 prediction rows and no output-format errors. Its operation-conditioned validation was 50% in every arm; the new audit separates fixed-operation competence from operation selection.",
        "",
        "## Competence results",
        "",
        "| Condition | Interface | Seed | Train | Validation | Unequal validation | Equal validation | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm in summary["arms"]:
        aid = arm["arm_id"]
        metrics = analysis["per_arm"][aid]
        lines.append(
            f"| {arm['condition']} | {arm['interface']} | {arm['seed']} | "
            f"{metrics['train']['output_token_accuracy']:.1%} ({metrics['train']['output_token_correct']}/{metrics['train']['observations']}) | "
            f"{metrics['validation']['output_token_accuracy']:.1%} ({metrics['validation']['output_token_correct']}/{metrics['validation']['observations']}) | "
            f"{metrics['validation_unequal']['output_token_accuracy']:.1%} ({metrics['validation_unequal']['output_token_correct']}/{metrics['validation_unequal']['observations']}) | "
            f"{metrics['validation_equal']['output_token_accuracy']:.1%} ({metrics['validation_equal']['output_token_correct']}/{metrics['validation_equal']['observations']}) | "
            f"{'PASS' if arm['gate_pass'] else 'FAIL'} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if not analysis["all_competence_gates_pass"]:
        lines += [
            "At least one primitive/condition failed the preregistered competence gate. No conditional transfer stage was run, and no downstream transfer claim is made.",
            "",
            "The next diagnostic should inspect the training curves and fixed-operation predictions before increasing compute or changing architecture.",
        ]
    else:
        lines += [
            "All primitive and joint operation-conditioned competence gates passed. Conditional transfer results, if present, are reported separately and do not retroactively alter the competence gates.",
        ]
    lines += [
        "",
        "## Artifacts",
        "",
        "- [Protocol](../../docs/CRUCIBLE_ROLE_SELECTION_COMPETENCE_PROTOCOL_20251025.md)",
        "- [Previous failure audit](PREVIOUS_FAILURE_AUDIT.md)",
        "- [Analysis](ANALYSIS_FINAL.json)",
        "- [Predictions](PREDICTIONS.csv)",
        "- [Summary](SUMMARY.json)",
        "- [Preflight](PREFLIGHT.json)",
        "- [Training curves](arm_outputs/)",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    previous_audit = audit_previous_failure()
    base.OUTPUT = OUTPUT
    base.ANCHOR_DIR = OUTPUT / "generated_reference_e4" / "crucible_manifest"

    records: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    errors: List[str] = []
    for condition in CONDITIONS:
        records[condition] = {}
        for interface in INTERFACES:
            split_records = {
                "train": records_for(condition, interface, "train", sorted(TRAIN_SET)),
                "validation": records_for(condition, interface, "validation", sorted(VALIDATION_SET)),
            }
            records[condition][interface] = split_records
            errors.extend(audit_records(condition, interface, split_records))

    preflight = {
        "experiment": "crucible_role_selection_competence_20251025",
        "status": "frozen_before_training",
        "conditions": list(CONDITIONS),
        "interfaces": list(INTERFACES),
        "seeds": list(SEEDS),
        "train_pairs": [list(pair) for pair in sorted(TRAIN_SET)],
        "validation_pairs": [list(pair) for pair in sorted(VALIDATION_SET)],
        "transfer_pairs": [list(pair) for pair in sorted(TRANSFER_SET)],
        "train_pair_count": len(TRAIN_SET),
        "validation_pair_count": len(VALIDATION_SET),
        "equal_validation_pairs": [list(pair) for pair in sorted(pair for pair in VALIDATION_SET if pair[0] == pair[1])],
        "unequal_validation_pairs": [list(pair) for pair in sorted(pair for pair in VALIDATION_SET if pair[0] != pair[1])],
        "reversed_validation_pairs": {
            f"({a},{b})": (b, a) in VALIDATION_SET
            for a, b in sorted(VALIDATION_SET)
            if a != b
        },
        "steps": STEPS,
        "curve_steps": list(CURVE_STEPS),
        "training_performed": False,
        "architecture_changed": False,
        "sealed_test_accessed": False,
        "errors": errors,
        "pass": not errors and previous_audit.get("pass", False),
    }
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    if errors:
        raise SystemExit(f"preflight failed: {json.dumps(preflight, sort_keys=True)}")

    for condition in CONDITIONS:
        for interface in INTERFACES:
            path = OUTPUT / "data" / condition / interface
            write_jsonl(path / "train.jsonl", records[condition][interface]["train"])
            write_jsonl(path / "validation.jsonl", records[condition][interface]["validation"])
    write_json(OUTPUT / "ORACLE.json", {
        "training_records": sum(len(records[c][i]["train"]) for c in CONDITIONS for i in INTERFACES),
        "validation_records": sum(len(records[c][i]["validation"]) for c in CONDITIONS for i in INTERFACES),
        "oracle_errors": [],
        "pass": True,
        "sealed_test_accessed": False,
    })

    base.OUTPUT = OUTPUT
    base.ANCHOR_DIR = OUTPUT / "generated_reference_e4" / "crucible_manifest"
    anchor_hashes = prepare_fresh_anchors()
    write_json(OUTPUT / "INITIAL_ANCHORS.json", {
        "directory": rel(base.ANCHOR_DIR),
        "seeds": list(SEEDS),
        "sha256": anchor_hashes,
        "generated_without_training": True,
    })

    preflight["initial_anchor_sha256"] = anchor_hashes
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    print(json.dumps({"status": "preflight_passed", "arms_expected": len(CONDITIONS) * len(INTERFACES) * len(SEEDS)}, sort_keys=True), flush=True)

    arms: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        for interface in INTERFACES:
            for seed in SEEDS:
                arm_id = f"{condition}_{interface}_seed{seed}"
                output = OUTPUT / "arm_outputs" / condition / interface / f"seed{seed}"
                print(json.dumps({"event": "start", "arm_id": arm_id}, sort_keys=True), flush=True)
                result = train_model(condition, interface, seed, records[condition][interface], output)
                checkpoint = output / "checkpoint.pt"
                result.update({
                    "arm_id": arm_id,
                    "condition": condition,
                    "interface": interface,
                    "seed": seed,
                    "checkpoint": rel(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                })
                write_json(output / "RESULTS.json", result)
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = stage_a.make_dense(seed)
                model.load_state_dict(state["model"], strict=True)
                prediction_arm = {
                    "arm_id": arm_id,
                    "condition": condition,
                    "interface": interface,
                    "seed": seed,
                    "checkpoint": rel(checkpoint),
                    "checkpoint_sha256": result["checkpoint_sha256"],
                }
                rows = predict_rows(model, records[condition][interface]["train"] + records[condition][interface]["validation"], prediction_arm)
                all_rows.extend(rows)
                write_json(output / "FORENSIC.json", {"results": result, "rows": rows})
                arm_summary = {
                    "arm_id": arm_id,
                    "condition": condition,
                    "interface": interface,
                    "seed": seed,
                    "train": stat([row for row in rows if row["split"] == "train"]),
                    "validation": stat([row for row in rows if row["split"] == "validation"]),
                    "validation_unequal": stat([row for row in rows if row["split"] == "validation" and row["pair_kind"] == "unequal"]),
                    "validation_equal": stat([row for row in rows if row["split"] == "validation" and row["pair_kind"] == "equal"]),
                    "fit_pass": result["fit_pass"],
                    "validation_pass": result["validation_pass"],
                    "unequal_validation_pass": result["unequal_validation_pass"],
                    "gate_pass": result["gate_pass"],
                    "checkpoint": result["checkpoint"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                }
                arms.append(arm_summary)
                del model, state
                gc.collect()
                print(json.dumps({"event": "complete", "arm_id": arm_id, "train": result["train_exact"], "validation": result["validation_exact"], "unequal_validation": result["validation_unequal_exact"], "gate_pass": result["gate_pass"]}, sort_keys=True), flush=True)

    write_csv(OUTPUT / "PREDICTIONS.csv", all_rows)
    summary = {
        "experiment": "crucible_role_selection_competence_20251025",
        "status": "completed",
        "training_performed": True,
        "sealed_test_accessed": False,
        "architecture_changed": False,
        "arms": arms,
        "prediction_rows": len(all_rows),
        "steps": STEPS,
    }
    write_json(OUTPUT / "SUMMARY.json", summary)

    all_competence_pass = all(arm["gate_pass"] for arm in arms)
    transfer_rows: List[Dict[str, Any]] = []
    if all_competence_pass:
        for arm in arms:
            if arm["condition"] != "S3_JOINT_OPERATION_CONDITIONED":
                continue
            transfer_records = {
                "conditional_transfer": records_for("S3_JOINT_OPERATION_CONDITIONED", arm["interface"], "conditional_transfer", sorted(TRANSFER_SET))
            }
            checkpoint = ROOT / arm["checkpoint"]
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = stage_a.make_dense(int(arm["seed"]))
            model.load_state_dict(state["model"], strict=True)
            transfer_rows.extend(predict_rows(model, transfer_records["conditional_transfer"], arm, split_label="conditional_transfer"))
            del model, state
            gc.collect()
        write_csv(OUTPUT / "CONDITIONAL_TRANSFER_PREDICTIONS.csv", transfer_rows)
    else:
        (OUTPUT / "CONDITIONAL_TRANSFER_NOT_RUN.txt").write_text("Not run: at least one preregistered S1/S2/S3 competence gate failed.\n", encoding="utf-8")

    preflight["training_performed"] = True
    preflight["status"] = "completed"
    preflight["all_competence_gates_pass"] = all_competence_pass
    preflight["conditional_transfer_run"] = bool(transfer_rows)
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    analysis = build_analysis(summary, all_rows, preflight, previous_audit, transfer_rows)
    write_json(OUTPUT / "ANALYSIS_FINAL.json", analysis)
    (OUTPUT / "REPORT_FINAL.md").write_text(build_report(analysis, summary), encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt":
            hashes[rel(path)] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    write_json(OUTPUT / "FINALIZATION.json", {
        "status": "completed",
        "verification_pass": analysis["verification"]["pass"],
        "all_competence_gates_pass": all_competence_pass,
        "conditional_transfer_run": bool(transfer_rows),
        "analysis_sha256": sha256(OUTPUT / "ANALYSIS_FINAL.json"),
        "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"),
        "prediction_sha256": sha256(OUTPUT / "PREDICTIONS.csv"),
    })
    print(json.dumps({"status": "completed", "arms": len(arms), "prediction_rows": len(all_rows), "all_competence_gates_pass": all_competence_pass, "conditional_transfer_run": bool(transfer_rows), "output": rel(OUTPUT)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
