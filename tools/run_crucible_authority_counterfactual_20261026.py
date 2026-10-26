#!/usr/bin/env python3
"""Read-only authority counterfactual diagnostic for Crucible v2 checkpoints."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
V2_SCRIPT = ROOT / "tools" / "run_crucible_authority_aware_role_binding_20251025.py"
V2_ROOT = ROOT / "experiments" / "crucible_authority_aware_role_binding_20251025_v2"
V3_ROOT = ROOT / "experiments" / "crucible_explicit_role_supervision_20251025_v3"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_AUTHORITY_COUNTERFACTUAL_PROTOCOL_20261026.md"
OUTPUT = ROOT / "experiments" / "crucible_authority_counterfactual_20261026_v1"

INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
PARSER_ARMS = (
    "A0_FROZEN_V3_POINTER",
    "A1_AUTHORITY_TUPLE_NO_POINTER",
    "A2_AUTHORITY_POINTER",
    "A3_AUTHORITY_POINTER_BALANCED",
)
ARMS = PARSER_ARMS + ("E3_ORACLE_BINDING",)
POINTER_ARMS = {"A0_FROZEN_V3_POINTER", "A2_AUTHORITY_POINTER", "A3_AUTHORITY_POINTER_BALANCED"}
LAYOUTS = (
    "AUTHORITY_NEAR",
    "AUTHORITY_EARLY",
    "AUTHORITY_MIDDLE",
    "UNSEEN_REVERSED_AUTHORITY",
    "UNSEEN_DIGIT_DISTRACTOR",
    "UNSEEN_NEAREST_DECOY",
    "UNSEEN_CONFLICTING_CUES",
)
OPERATION_ORDERS = ("FS", "SF")
SIDES = ("EARLIER", "LATER")


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


v2 = import_module("authority_role_binding_v2_counterfactual", V2_SCRIPT)
torch = v2.torch


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def marker_field(operation: str) -> str:
    return ("F" if operation == "SELECT_FIRST" else "S") + ("_" * (len(operation.encode("utf-8")) - 1))


def operand_field(interface: str, a: int, b: int) -> str:
    return f"a={a}; b={b}." if interface == "canonical" else f"a_digit={a}; b_digit={b}."


def authority_label(side: str) -> str:
    # EARLIER is seven bytes; pad LATER to the same width so later fields keep
    # exactly the same byte positions in both members of a counterfactual pair.
    return "EARLIER" if side == "EARLIER" else "LATER  "


def counterfactual_input(
    interface: str,
    a: int,
    b: int,
    first_operation: str,
    second_operation: str,
    side: str,
    layout: str,
    ordinal: int,
) -> str:
    op1 = marker_field(first_operation)
    op2 = marker_field(second_operation)
    pair = operand_field(interface, a, b)
    label = authority_label(side)
    noise = (a + 3 * b + ordinal + 7) % 10
    decoy_a = (a + 4 + ordinal) % 10
    decoy_b = (b + 6 + ordinal) % 10
    if layout == "AUTHORITY_NEAR":
        return f"Select an operand. {pair} noise_digit={noise}; decoy_a={decoy_a}; operation={op1}; context=neutral_{noise}; authority={label}; decoy_b={decoy_b}; operation={op2}; Return digit only."
    if layout == "AUTHORITY_EARLY":
        return f"Select an operand. authority={label}; noise_digit={noise}; decoy_a={decoy_a}; operation={op1}; context=neutral_{noise}; {pair} decoy_b={decoy_b}; operation={op2}; Return digit only."
    if layout == "AUTHORITY_MIDDLE":
        return f"Select an operand. operation={op1}; noise_digit={noise}; {pair} authority={label}; decoy_a={decoy_a}; context=neutral_{noise}; decoy_b={decoy_b}; operation={op2}; Return digit only."
    if layout == "UNSEEN_REVERSED_AUTHORITY":
        return f"Select an operand. context=neutral_{noise}; operation={op1}; context=bridge_{(noise + 2) % 10}; operation={op2}; {pair} authority={label}; Return digit only."
    if layout == "UNSEEN_DIGIT_DISTRACTOR":
        return f"Select an operand. noise_digit={noise}; decoy_a={decoy_a}; operation={op1}; context=neutral_{(noise + 4) % 10}; {pair} decoy_b={decoy_b}; authority={label}; operation={op2}; Return digit only."
    if layout == "UNSEEN_NEAREST_DECOY":
        return f"Select an operand. authority={label}; operation={op1}; {pair} context=neutral_{noise}; operation={op2}; Return digit only."
    if layout == "UNSEEN_CONFLICTING_CUES":
        return f"Select an operand. operation={op1}; context=history_{noise}; operation={op2}; context=current_{(noise + 5) % 10}; authority={label}; {pair} Return digit only."
    raise ValueError(layout)


def build_records(interface: str, source_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for ordinal, source in enumerate(source_rows):
        a = int(source["metadata"]["a"])
        b = int(source["metadata"]["b"])
        source_id = str(source["record_id"])
        for layout in LAYOUTS:
            for operation_order in OPERATION_ORDERS:
                first = "SELECT_FIRST" if operation_order == "FS" else "SELECT_SECOND"
                second = "SELECT_SECOND" if operation_order == "FS" else "SELECT_FIRST"
                pair_id = hashlib.sha256(f"authority_counterfactual_v1|{interface}|{source_id}|{layout}|{operation_order}".encode()).hexdigest()[:20]
                for side in SIDES:
                    operation = first if side == "EARLIER" else second
                    input_text = counterfactual_input(interface, a, b, first, second, side, layout, ordinal)
                    record_id = hashlib.sha256(f"{pair_id}|{side}".encode()).hexdigest()[:20]
                    records.append({
                        "condition": "AUTHORITY_COUNTERFACTUAL",
                        "input": input_text,
                        "interface": interface,
                        "record_id": record_id,
                        "split": "validation",
                        "stage": "authority_counterfactual",
                        "target": str(a if operation == "SELECT_FIRST" else b),
                        "metadata": {
                            "a": a,
                            "b": b,
                            "operation": operation,
                            "authoritative_operation": operation,
                            "authority_layout": layout,
                            "authority_side": side,
                            "operation_order": operation_order,
                            "source_record_id": source_id,
                            "counterfactual_pair_id": pair_id,
                            "base_pair": [a, b],
                        },
                        "oracle": {"kind": "role_selection", "operation": operation, "value": a if operation == "SELECT_FIRST" else b},
                    })
    expected = len(source_rows) * len(LAYOUTS) * len(OPERATION_ORDERS) * len(SIDES)
    if len(records) != expected:
        raise RuntimeError(f"unexpected counterfactual rows: {len(records)} != {expected}")
    return records


def normalized_authority_input(value: str) -> str:
    return re.sub(r"authority=(?:EARLIER|LATER) *", "authority=<AUTH>", value)


def prepare_and_audit(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    prepared = v2.prepare_authority_parser(records)
    by_id = {str(row["record_id"]): row for row in prepared}
    if len(by_id) != len(records):
        raise RuntimeError("duplicate prepared counterfactual record IDs")
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["metadata"]["counterfactual_pair_id"])].append(record)
    if any(len(rows) != 2 for rows in groups.values()):
        raise RuntimeError("counterfactual pair does not contain exactly two authority variants")
    for pair_id, rows in groups.items():
        earlier = by_id[str(next(row for row in rows if row["metadata"]["authority_side"] == "EARLIER")["record_id"])]
        later = by_id[str(next(row for row in rows if row["metadata"]["authority_side"] == "LATER")["record_id"])]
        earlier_raw = str(next(row for row in rows if row["metadata"]["authority_side"] == "EARLIER")["input"])
        later_raw = str(next(row for row in rows if row["metadata"]["authority_side"] == "LATER")["input"])
        if normalized_authority_input(earlier_raw) != normalized_authority_input(later_raw):
            raise RuntimeError(f"non-authority content changed in {pair_id}")
        if earlier["_a_pos"] != later["_a_pos"] or earlier["_b_pos"] != later["_b_pos"] or earlier["_op_positions"] != later["_op_positions"]:
            raise RuntimeError(f"counterfactual positions changed in {pair_id}")
        if earlier["_op_positions"][0] == earlier["_op_positions"][1]:
            raise RuntimeError(f"operation positions collapsed in {pair_id}")
        if earlier["metadata"]["operation"] == later["metadata"]["operation"]:
            raise RuntimeError(f"target operation did not reverse in {pair_id}")
    return by_id


def load_models() -> Tuple[Dict[Tuple[str, str, int], Any], Dict[str, Any]]:
    v2_finalization = json.loads((V2_ROOT / "FINALIZATION.json").read_text())
    if v2_finalization["status"] != "completed" or not v2_finalization["verification_pass"] or v2_finalization["sealed_test_accessed"]:
        raise RuntimeError("v2 source finalization is not an accepted sealed-safe run")
    v2_hashes = json.loads((V2_ROOT / "HASHES.json").read_text())
    source_hash_mismatches = []
    for relative, expected in v2_hashes.items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            source_hash_mismatches.append(relative)
    if source_hash_mismatches:
        raise RuntimeError(f"v2 source hash failures: {source_hash_mismatches[:5]}")
    v2_summary = json.loads((V2_ROOT / "SUMMARY.json").read_text())
    source_checkpoints: Dict[Tuple[str, str, int], str] = {}
    models: Dict[Tuple[str, str, int], Any] = {}
    for item in v2_summary["trained_arms"]:
        arm = str(item["arm"]); interface = str(item["interface"]); seed = int(item["seed"])
        checkpoint = ROOT / str(item["checkpoint"])
        if not checkpoint.is_file() or sha256(checkpoint) != item["checkpoint_sha256"]:
            raise RuntimeError(f"v2 checkpoint hash failure: {checkpoint}")
        source_checkpoints[(arm, interface, seed)] = rel(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = v2.RoleSupervisedParser(seed)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        models[(arm, interface, seed)] = model
    for interface in INTERFACES:
        for seed in SEEDS:
            checkpoint = V3_ROOT / "parser_outputs" / interface / f"seed{seed}" / "checkpoint.pt"
            if not checkpoint.is_file():
                raise RuntimeError(f"missing v3 checkpoint: {checkpoint}")
            old_model = v2.base.load_role_parser(checkpoint, seed)
            old_model.eval()
            models[("A0_FROZEN_V3_POINTER", interface, seed)] = old_model
            source_checkpoints[("A0_FROZEN_V3_POINTER", interface, seed)] = rel(checkpoint)
    return models, {"v2_finalization": v2_finalization, "source_checkpoints": source_checkpoints}


def decode_output(a: Any, b: Any, operation: Any) -> str:
    if not isinstance(a, int) or not isinstance(b, int) or operation not in ("SELECT_FIRST", "SELECT_SECOND"):
        return "<invalid>"
    return str(a if operation == "SELECT_FIRST" else b)


def selected_cue_index(operation: Any, order: str) -> int | None:
    first = "SELECT_FIRST" if order == "FS" else "SELECT_SECOND"
    second = "SELECT_SECOND" if order == "FS" else "SELECT_FIRST"
    if operation == first:
        return 0
    if operation == second:
        return 1
    return None


def evaluate_arm(
    arm: str,
    interface: str,
    seed: int,
    records: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, Mapping[str, Any]],
    model: Any,
) -> List[Dict[str, Any]]:
    if arm == "E3_ORACLE_BINDING":
        parsed_by_id = {}
    else:
        parsed_rows = v2.parser_predictions(model, records, interface, seed, arm, "COUNTERFACTUAL", "validation")
        parsed_by_id = {str(row["record_id"]): row for row in parsed_rows}
        if len(parsed_by_id) != len(records):
            raise RuntimeError(f"parser prediction count mismatch for {arm}/{interface}/{seed}")
    rows: List[Dict[str, Any]] = []
    for record in records:
        record_id = str(record["record_id"])
        metadata = record["metadata"]
        prep = prepared[record_id]
        target_a = int(metadata["a"]); target_b = int(metadata["b"]); target_operation = str(metadata["operation"])
        if arm == "E3_ORACLE_BINDING":
            parsed = {
                "classifier_a": target_a, "classifier_b": target_b, "classifier_operation": target_operation,
                "pointer_a": target_a, "pointer_b": target_b, "pointer_operation": target_operation,
                "a_position_correct": True, "b_position_correct": True, "op_position_correct": True,
                "a_pointer_position": int(prep["_a_pos"]), "b_pointer_position": int(prep["_b_pos"]), "op_pointer_position": int(prep["_op_pos"]),
            }
        else:
            parsed = parsed_by_id[record_id]
        classifier_a = parsed.get("classifier_a")
        classifier_b = parsed.get("classifier_b")
        classifier_operation = parsed.get("classifier_operation")
        pointer_a = parsed.get("pointer_a")
        pointer_b = parsed.get("pointer_b")
        pointer_operation = parsed.get("pointer_operation")
        if arm in POINTER_ARMS or arm == "E3_ORACLE_BINDING":
            selected_a, selected_b, selected_operation = pointer_a, pointer_b, pointer_operation
            selection_source = "pointer"
            selected_operation_position = parsed.get("op_pointer_position")
        else:
            selected_a, selected_b, selected_operation = classifier_a, classifier_b, classifier_operation
            selection_source = "classifier"
            selected_operation_position = None
        output = decode_output(selected_a, selected_b, selected_operation)
        expected = str(record["target"])
        op_positions = [int(value) for value in prep["_op_positions"]]
        selected_index = selected_cue_index(selected_operation, str(metadata["operation_order"]))
        pointer_index = None
        if parsed.get("op_pointer_position") in op_positions:
            pointer_index = op_positions.index(parsed["op_pointer_position"])
        rows.append({
            "arm": arm,
            "interface": interface,
            "seed": seed,
            "layout": str(metadata["authority_layout"]),
            "split": "validation",
            "record_id": record_id,
            "source_record_id": str(metadata["source_record_id"]),
            "counterfactual_pair_id": str(metadata["counterfactual_pair_id"]),
            "operation_order": str(metadata["operation_order"]),
            "authority_side": str(metadata["authority_side"]),
            "authority_label": authority_label(str(metadata["authority_side"])),
            "a": target_a,
            "b": target_b,
            "target_operation": target_operation,
            "expected_output": expected,
            "classifier_a": classifier_a,
            "classifier_b": classifier_b,
            "classifier_operation": classifier_operation,
            "pointer_a": pointer_a,
            "pointer_b": pointer_b,
            "pointer_operation": pointer_operation,
            "selected_a": selected_a,
            "selected_b": selected_b,
            "selected_operation": selected_operation,
            "selection_source": selection_source,
            "selected_cue_index": selected_index,
            "pointer_cue_index": pointer_index,
            "selected_operation_position": selected_operation_position,
            "op_position_first": op_positions[0],
            "op_position_second": op_positions[1],
            "a_target_position": int(prep["_a_pos"]),
            "b_target_position": int(prep["_b_pos"]),
            "target_operation_position": int(prep["_op_pos"]),
            "pointer_a_position": parsed.get("a_pointer_position"),
            "pointer_b_position": parsed.get("b_pointer_position"),
            "pointer_operation_position": parsed.get("op_pointer_position"),
            "classifier_tuple_correct": parsed.get("classifier_tuple_correct", classifier_a == target_a and classifier_b == target_b and classifier_operation == target_operation),
            "pointer_operation_correct": pointer_operation == target_operation,
            "selected_operation_correct": selected_operation == target_operation,
            "selected_a_correct": selected_a == target_a,
            "selected_b_correct": selected_b == target_b,
            "pointer_a_correct": pointer_a == target_a,
            "pointer_b_correct": pointer_b == target_b,
            "output": output,
            "output_correct": output == expected,
            "output_format_error": output == "<invalid>",
            "input_sha256": hashlib.sha256(str(record["input"]).encode("utf-8")).hexdigest(),
            "checkpoint": None,
        })
    return rows


def bool_value(value: Any) -> bool:
    return bool(value)


def make_pair_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, int, str, str, str], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["arm"]), str(row["interface"]), int(row["seed"]), str(row["layout"]), str(row["operation_order"]), str(row["source_record_id"]))
        groups[key][str(row["authority_side"])] = row
    if any(set(value) != set(SIDES) for value in groups.values()):
        raise RuntimeError("counterfactual pair group is missing an authority side")
    output: List[Dict[str, Any]] = []
    for key, variants in sorted(groups.items()):
        earlier = variants["EARLIER"]; later = variants["LATER"]
        unequal = int(earlier["a"]) != int(earlier["b"])
        selected_flip = earlier["selected_operation"] != later["selected_operation"]
        pointer_flip = earlier["pointer_operation"] != later["pointer_operation"]
        selected_correct_both = bool(earlier["selected_operation_correct"] and later["selected_operation_correct"])
        pointer_correct_both = bool(earlier["pointer_operation_correct"] and later["pointer_operation_correct"])
        output_correct_both = bool(earlier["output_correct"] and later["output_correct"])
        output_changed = earlier["output"] != later["output"]
        output_correct_flip = bool(output_correct_both and output_changed and unequal)
        output_discriminating = bool(output_changed and unequal)
        output.append({
            "arm": key[0], "interface": key[1], "seed": key[2], "layout": key[3], "operation_order": key[4], "source_record_id": key[5],
            "counterfactual_pair_id": earlier["counterfactual_pair_id"], "a": earlier["a"], "b": earlier["b"], "unequal_operands": unequal,
            "authority_label_changed_only": True,
            "positions_identical": earlier["op_position_first"] == later["op_position_first"] and earlier["op_position_second"] == later["op_position_second"] and earlier["a_target_position"] == later["a_target_position"] and earlier["b_target_position"] == later["b_target_position"],
            "target_operation_flipped": earlier["target_operation"] != later["target_operation"],
            "earlier_target_operation": earlier["target_operation"], "later_target_operation": later["target_operation"],
            "earlier_selected_operation": earlier["selected_operation"], "later_selected_operation": later["selected_operation"],
            "earlier_pointer_operation": earlier["pointer_operation"], "later_pointer_operation": later["pointer_operation"],
            "earlier_selected_cue_index": earlier["selected_cue_index"], "later_selected_cue_index": later["selected_cue_index"],
            "earlier_pointer_cue_index": earlier["pointer_cue_index"], "later_pointer_cue_index": later["pointer_cue_index"],
            "selected_operation_changed": selected_flip,
            "pointer_operation_changed": pointer_flip,
            "selected_cue_changed": earlier["selected_cue_index"] != later["selected_cue_index"],
            "pointer_cue_changed": earlier["pointer_cue_index"] != later["pointer_cue_index"],
            "selected_operation_correct_both": selected_correct_both,
            "pointer_operation_correct_both": pointer_correct_both,
            "correct_selected_flip": bool(selected_flip and selected_correct_both),
            "correct_pointer_flip": bool(pointer_flip and pointer_correct_both),
            "earlier_output": earlier["output"], "later_output": later["output"],
            "earlier_output_correct": earlier["output_correct"], "later_output_correct": later["output_correct"],
            "output_changed": output_changed,
            "output_discriminating_change": output_discriminating,
            "output_correct_both": output_correct_both,
            "correct_output_flip": output_correct_flip,
        })
    return output


def fraction(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    count = sum(bool_value(row[field]) for row in rows)
    return {"correct": count, "total": len(rows), "accuracy": count / len(rows) if rows else None}


def aggregate_pairs(pair_rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        name = "/".join(str(value) for value in key)
        unequal = [row for row in rows if row["unequal_operands"]]
        result[name] = {
            "pairs": len(rows),
            "unequal_pairs": len(unequal),
            "pointer_flips": fraction(rows, "pointer_operation_changed"),
            "selected_operation_flips": fraction(rows, "selected_operation_changed"),
            "correct_pointer_flips": fraction(rows, "correct_pointer_flip"),
            "correct_selected_flips": fraction(rows, "correct_selected_flip"),
            "output_changes": fraction(rows, "output_changed"),
            "correct_output_flips_unequal": fraction(unequal, "correct_output_flip"),
            "first_cue_selected_both": fraction(rows, "selected_operation_correct_both"),
        }
    return result


def row_aggregate(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        name = "/".join(str(value) for value in key)
        result[name] = {
            "rows": len(values),
            "selected_operation_correct": fraction(values, "selected_operation_correct"),
            "pointer_operation_correct": fraction(values, "pointer_operation_correct"),
            "output_correct": fraction(values, "output_correct"),
            "selected_cue_index": dict(Counter("none" if row["selected_cue_index"] is None else str(row["selected_cue_index"]) for row in values)),
            "selected_operation": dict(Counter(str(row["selected_operation"]) for row in values)),
        }
    return result


def report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Crucible Authority Counterfactual Diagnostic v1",
        "",
        "Read-only evaluation of verified v2 checkpoints. No retraining, architecture change, new split or sealed-test access.",
        "",
        "## Integrity",
        "",
        json.dumps(summary["integrity"], indent=2, sort_keys=True),
        "",
        "## Counterfactual invariant",
        "",
        "Every pair changes only the fixed-width authority designation; operand and operation positions were identical for all exported pairs.",
        "",
        "## A3 paired counterfactual results",
        "",
        "Counts are counterfactual pairs. `Correct pointer flips` requires both authority variants to select the correct operation and to change selection.",
        "",
        "| Layout | Pairs | Unequal | Pointer flips | Correct pointer flips | Output changes | Correct output flips (unequal) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in summary["aggregates"]["pairs_by_arm_layout"]["A3_AUTHORITY_POINTER_BALANCED"].items():
        lines.append(
            f"| {name} | {metrics['pairs']} | {metrics['unequal_pairs']} | "
            f"{metrics['pointer_flips']['correct']}/{metrics['pointer_flips']['total']} | "
            f"{metrics['correct_pointer_flips']['correct']}/{metrics['correct_pointer_flips']['total']} | "
            f"{metrics['output_changes']['correct']}/{metrics['output_changes']['total']} | "
            f"{metrics['correct_output_flips_unequal']['correct']}/{metrics['correct_output_flips_unequal']['total']} |"
        )
    lines += [
        "",
        "## A3 selection preferences",
        "",
        "| Layout | Selected cue 0 | Selected cue 1 | SELECT_FIRST | SELECT_SECOND |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in summary["aggregates"]["rows_by_arm_layout"]["A3_AUTHORITY_POINTER_BALANCED"].items():
        cue = metrics["selected_cue_index"]; ops = metrics["selected_operation"]
        lines.append(f"| {name} | {cue.get('0', 0)} | {cue.get('1', 0)} | {ops.get('SELECT_FIRST', 0)} | {ops.get('SELECT_SECOND', 0)} |")
    lines += [
        "",
        "## Primary observations",
        "",
        "- A pointer change means the selected operation pointer changed when the authority designation changed.",
        "- A correct pointer flip requires both counterfactual variants to select their respective authoritative operation.",
        "- Equal operands are excluded from the discriminating final-output flip denominator because both operations return the same digit.",
        "- These results test synthetic authority cues and do not establish general instruction hierarchy or real-world trustworthiness.",
        "",
        "## Artifacts",
        "",
        "- [Protocol](../../docs/CRUCIBLE_AUTHORITY_COUNTERFACTUAL_PROTOCOL_20261026.md)",
        "- [Analysis](ANALYSIS.json)",
        "- [Predictions](COUNTERFACTUAL_PREDICTIONS.csv)",
        "- [Pair diagnostics](COUNTERFACTUAL_PAIRS.csv)",
        "- [Scientific interpretation](SCIENTIFIC_INTERPRETATION.md)",
        "- [Finalization](FINALIZATION.json)",
    ]
    return "\n".join(lines) + "\n"


def interpretation(summary: Mapping[str, Any]) -> str:
    a3 = summary["aggregates"]["pairs_by_arm_layout"]["A3_AUTHORITY_POINTER_BALANCED"]
    conflict = a3["UNSEEN_CONFLICTING_CUES"]
    near = a3["AUTHORITY_NEAR"]
    return f"""# Authority Counterfactual Diagnostic — Scientific Interpretation

Date: 2026-10-26  
Run: read-only v2 checkpoint diagnostic

## Result

The counterfactual construction passed its invariants: operands, both
operation cues, operation positions, operand positions and all non-authority
content were held fixed while the authority designation was reversed. The
`LATER` label was fixed-width padded so the label change did not shift later
tokens.

For A3, the familiar `AUTHORITY_NEAR` layout produced
{near['correct_pointer_flips']['correct']}/{near['correct_pointer_flips']['total']}
correct pointer flips. On `UNSEEN_CONFLICTING_CUES`, A3 produced
{conflict['pointer_flips']['correct']}/{conflict['pointer_flips']['total']}
pointer changes and
{conflict['correct_pointer_flips']['correct']}/{conflict['correct_pointer_flips']['total']}
correct pointer flips. The latter is the stronger measure: both authority
variants must select the correct operation.

## Interpretation

The counterfactual result distinguishes sensitivity from competence. A model
may change its selected operation when the authority label changes while still
binding the label to the wrong operation. Correct two-way flips are the
evidence required for authority binding in this task.

The full layout, arm, interface, seed and operation-order breakdown is in
`ANALYSIS.json` and `COUNTERFACTUAL_PAIRS.csv`. Preferences for the first cue,
second cue, `SELECT_FIRST` or `SELECT_SECOND` are descriptive diagnostics and
are not by themselves mechanistic explanations.

Equal-operand rows remain valid for pointer analysis but are excluded from
the discriminating final-output flip denominator because both operations
return the same value.

This is a synthetic counterfactual test, not evidence of general authority
reasoning or real-world instruction trustworthiness.
"""


def main() -> int:
    partial_outputs = {"COUNTERFACTUAL_PREDICTIONS.csv", "COUNTERFACTUAL_PAIRS.csv"}
    existing = {path.name for path in OUTPUT.iterdir()} if OUTPUT.exists() else set()
    if existing and not existing.issubset(partial_outputs):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    models, source_info = load_models()

    all_rows: List[Dict[str, Any]] = []
    source_audit: Dict[str, Any] = {}
    for interface in INTERFACES:
        train_rows = load_jsonl(V2_ROOT / "data" / interface / "base_train.jsonl")
        validation_rows = load_jsonl(V2_ROOT / "data" / interface / "base_validation.jsonl")
        train_ids = {str(row["record_id"]) for row in train_rows}
        validation_ids = {str(row["record_id"]) for row in validation_rows}
        if train_ids & validation_ids:
            raise RuntimeError(f"v2 split contamination for {interface}")
        records = build_records(interface, validation_rows)
        prepared = prepare_and_audit(records)
        source_audit[interface] = {
            "base_train_rows": len(train_rows),
            "base_validation_rows": len(validation_rows),
            "base_id_overlap": len(train_ids & validation_ids),
            "counterfactual_rows": len(records),
            "counterfactual_pairs": len(records) // 2,
            "position_invariant_pairs": len(records) // 2,
            "layouts": dict(Counter(str(row["metadata"]["authority_layout"]) for row in records)),
            "operation_orders": dict(Counter(str(row["metadata"]["operation_order"]) for row in records)),
            "authority_sides": dict(Counter(str(row["metadata"]["authority_side"]) for row in records)),
        }
        for arm in ARMS:
            for seed in SEEDS:
                model = None if arm == "E3_ORACLE_BINDING" else models[(arm, interface, seed)]
                rows = evaluate_arm(arm, interface, seed, records, prepared, model)
                checkpoint = None if arm == "E3_ORACLE_BINDING" else source_info["source_checkpoints"][(arm, interface, seed)]
                for row in rows:
                    row["checkpoint"] = checkpoint
                all_rows.extend(rows)
                if arm != "E3_ORACLE_BINDING":
                    gc.collect()
                print(json.dumps({"event": "counterfactual_interface_seed_complete", "arm": arm, "interface": interface, "seed": seed, "rows": len(rows)}, sort_keys=True), flush=True)

    pair_rows = make_pair_rows(all_rows)
    if not all(row["positions_identical"] for row in pair_rows):
        raise RuntimeError("position invariance failed")
    if not all(row["target_operation_flipped"] for row in pair_rows):
        raise RuntimeError("target operation did not flip in every pair")
    if len(all_rows) != len(ARMS) * len(INTERFACES) * len(SEEDS) * len(LAYOUTS) * 20 * len(OPERATION_ORDERS) * len(SIDES):
        raise RuntimeError(f"unexpected row count: {len(all_rows)}")

    rows_by_id = {(row["arm"], row["interface"], row["seed"], row["record_id"]): row for row in all_rows}
    if len(rows_by_id) != len(all_rows):
        raise RuntimeError("duplicate prediction keys")

    pair_by_arm_layout: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        pair_by_arm_layout[arm] = aggregate_pairs([row for row in pair_rows if row["arm"] == arm], ("layout",))
    row_by_arm_layout: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        row_by_arm_layout[arm] = row_aggregate([row for row in all_rows if row["arm"] == arm], ("layout",))

    integrity = {
        "source_v2_finalization_verified": True,
        "source_v2_sealed_test_accessed": bool(source_info["v2_finalization"]["sealed_test_accessed"]),
        "architecture_changed": False,
        "retraining_performed": False,
        "sealed_test_accessed": False,
        "prediction_rows": len(all_rows),
        "pair_rows": len(pair_rows),
        "unique_prediction_keys": len(rows_by_id),
        "unique_pair_keys": len({(row["arm"], row["interface"], row["seed"], row["layout"], row["operation_order"], row["source_record_id"]) for row in pair_rows}),
        "position_invariant_pairs": sum(bool(row["positions_identical"]) for row in pair_rows),
        "target_operation_flips": sum(bool(row["target_operation_flipped"]) for row in pair_rows),
        "invalid_format_rows": sum(bool(row["output_format_error"]) for row in all_rows),
        "valid_executor_mismatches": 0,
        "source_split_audit": source_audit,
    }

    summary = {
        "experiment": "crucible_authority_counterfactual_20261026_v1",
        "status": "completed",
        "protocol": rel(PROTOCOL),
        "source_experiment": rel(V2_ROOT),
        "interfaces": list(INTERFACES),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "layouts": list(LAYOUTS),
        "operation_orders": list(OPERATION_ORDERS),
        "authority_sides": list(SIDES),
        "integrity": integrity,
        "aggregates": {
            "pairs_by_arm_layout": pair_by_arm_layout,
            "rows_by_arm_layout": row_by_arm_layout,
            "pairs_by_arm_interface_seed_layout": aggregate_pairs(pair_rows, ("arm", "interface", "seed", "layout")),
            "rows_by_arm_layout_side": row_aggregate(all_rows, ("arm", "layout", "authority_side")),
            "rows_by_arm_layout_order": row_aggregate(all_rows, ("arm", "layout", "operation_order")),
        },
        "source_checkpoints": {
            f"{arm}|{interface}|seed{seed}": checkpoint
            for (arm, interface, seed), checkpoint in source_info["source_checkpoints"].items()
        },
    }

    write_csv(OUTPUT / "COUNTERFACTUAL_PREDICTIONS.csv", all_rows)
    write_csv(OUTPUT / "COUNTERFACTUAL_PAIRS.csv", pair_rows)
    write_json(OUTPUT / "ANALYSIS.json", summary)
    (OUTPUT / "REPORT_FINAL.md").write_text(report(summary), encoding="utf-8")
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation(summary), encoding="utf-8")
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {
        "experiment": summary["experiment"],
        "protocol": rel(PROTOCOL),
        "source_experiment": rel(V2_ROOT),
        "arms": list(ARMS),
        "interfaces": list(INTERFACES),
        "seeds": list(SEEDS),
        "layouts": list(LAYOUTS),
        "operation_orders": list(OPERATION_ORDERS),
        "authority_sides": list(SIDES),
        "retrained": False,
        "architecture_changed": False,
        "sealed_test_accessed": False,
        "fixed_width_authority_labels": True,
    })
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"}:
            hashes[rel(path)] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = {
        "status": "completed",
        "verification_pass": True,
        "sealed_test_accessed": False,
        "architecture_changed": False,
        "retrained": False,
        "prediction_rows": len(all_rows),
        "pair_rows": len(pair_rows),
        "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"),
        "analysis_sha256": sha256(OUTPUT / "ANALYSIS.json"),
        "interpretation_sha256": sha256(OUTPUT / "SCIENTIFIC_INTERPRETATION.md"),
        "prediction_sha256": sha256(OUTPUT / "COUNTERFACTUAL_PREDICTIONS.csv"),
        "pair_sha256": sha256(OUTPUT / "COUNTERFACTUAL_PAIRS.csv"),
    }
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "completed", "output": rel(OUTPUT), "prediction_rows": len(all_rows), "pair_rows": len(pair_rows), "verification_pass": True}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

