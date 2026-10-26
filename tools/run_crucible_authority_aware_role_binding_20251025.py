#!/usr/bin/env python3
"""Authority-aware role-binding gate for Crucible."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "tools" / "run_crucible_explicit_role_supervision_20251025.py"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_AUTHORITY_AWARE_ROLE_BINDING_PROTOCOL_20251025_v2.md"
OUTPUT = ROOT / "experiments" / "crucible_authority_aware_role_binding_20251025_v2"
PREVIOUS_ROOT = ROOT / "experiments" / "crucible_explicit_role_supervision_20251025_v3"
BATCH_ROOT = ROOT / "experiments" / "crucible_batch_composition_order_20251025_v1"

INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
ARMS = (
    "A0_FROZEN_V3_POINTER",
    "A1_AUTHORITY_TUPLE_NO_POINTER",
    "A2_AUTHORITY_POINTER",
    "A3_AUTHORITY_POINTER_BALANCED",
    "E3_ORACLE_BINDING",
)
TRAIN_AUTH_LAYOUTS = ("AUTHORITY_NEAR", "AUTHORITY_EARLY", "AUTHORITY_MIDDLE")
EVAL_LAYOUTS = TRAIN_AUTH_LAYOUTS + (
    "UNSEEN_REVERSED_AUTHORITY",
    "UNSEEN_DIGIT_DISTRACTOR",
    "UNSEEN_NEAREST_DECOY",
    "UNSEEN_CONFLICTING_CUES",
)
TRAIN_LAYOUT_MAP = {
    "FIXED_NEAR_PADDED": "AUTHORITY_NEAR",
    "EARLY_PADDED": "AUTHORITY_EARLY",
    "MIDDLE_POSITION": "AUTHORITY_MIDDLE",
}
STEPS = 500
EFFECTIVE_BATCH = 12
POINTER_LOSS_WEIGHT = 1.0
CURVE_STEPS = (1, 10, 25, 50, 100, 200, 300, 400, 500)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


base = import_module("authority_role_supervision_base", BASE_SOURCE)
stage_a = base.stage_a
torch = base.torch
F = torch.nn.functional
RoleSupervisedParser = base.RoleSupervisedParser


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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(row), sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def marker_field(operation: str) -> str:
    return ("F" if operation == "SELECT_FIRST" else "S") + ("_" * (len(operation.encode("utf-8")) - 1))


def operand_field(interface: str, a: int, b: int) -> str:
    return f"a={a}; b={b}." if interface == "canonical" else f"a_digit={a}; b_digit={b}."


def opposite(operation: str) -> str:
    return "SELECT_SECOND" if operation == "SELECT_FIRST" else "SELECT_FIRST"


def authority_input(interface: str, operation: str, a: int, b: int, side: str, layout: str, ordinal: int) -> str:
    first = operation if side == "EARLIER" else opposite(operation)
    second = opposite(operation) if side == "EARLIER" else operation
    op1 = marker_field(first)
    op2 = marker_field(second)
    pair = operand_field(interface, a, b)
    noise = (a + 3 * b + ordinal + 7) % 10
    decoy_a = (a + 4 + ordinal) % 10
    decoy_b = (b + 6 + ordinal) % 10
    if layout == "AUTHORITY_NEAR":
        return f"Select an operand. {pair} noise_digit={noise}; decoy_a={decoy_a}; operation={op1}; context=neutral_{noise}; authority={side}; decoy_b={decoy_b}; operation={op2}; Return digit only."
    if layout == "AUTHORITY_EARLY":
        return f"Select an operand. authority={side}; noise_digit={noise}; decoy_a={decoy_a}; operation={op1}; context=neutral_{noise}; {pair} decoy_b={decoy_b}; operation={op2}; Return digit only."
    if layout == "AUTHORITY_MIDDLE":
        return f"Select an operand. operation={op1}; noise_digit={noise}; {pair} authority={side}; decoy_a={decoy_a}; context=neutral_{noise}; decoy_b={decoy_b}; operation={op2}; Return digit only."
    if layout == "UNSEEN_REVERSED_AUTHORITY":
        return f"Select an operand. context=neutral_{noise}; operation={op1}; context=bridge_{(noise + 2) % 10}; operation={op2}; {pair} authority={side}; Return digit only."
    if layout == "UNSEEN_DIGIT_DISTRACTOR":
        return f"Select an operand. noise_digit={noise}; decoy_a={decoy_a}; operation={op1}; context=neutral_{(noise + 4) % 10}; {pair} decoy_b={decoy_b}; authority={side}; operation={op2}; Return digit only."
    if layout == "UNSEEN_NEAREST_DECOY":
        if side != "EARLIER":
            raise ValueError("nearest-decoy stress layout requires EARLIER authority")
        return f"Select an operand. authority=EARLIER; operation={op1}; {pair} context=neutral_{noise}; operation={op2}; Return digit only."
    if layout == "UNSEEN_CONFLICTING_CUES":
        return f"Select an operand. operation={op1}; context=history_{noise}; operation={op2}; context=current_{(noise + 5) % 10}; authority={side}; {pair} Return digit only."
    raise ValueError(layout)


def balanced_side_map(rows: Sequence[Mapping[str, Any]], layout: str) -> Dict[str, str]:
    by_operation: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_operation[str(row["metadata"]["operation"])].append(row)
    result: Dict[str, str] = {}
    for operation, group in by_operation.items():
        ordered = sorted(group, key=lambda row: str(row["record_id"]))
        if len(ordered) % 2:
            raise RuntimeError(f"cannot balance authority side for {layout}/{operation}: {len(ordered)}")
        midpoint = len(ordered) // 2
        for index, row in enumerate(ordered):
            result[str(row["record_id"])] = "EARLIER" if index < midpoint else "LATER"
    return result


def authority_row(row: Mapping[str, Any], interface: str, split: str, layout: str, side: str, ordinal: int) -> Dict[str, Any]:
    a = int(row["metadata"]["a"])
    b = int(row["metadata"]["b"])
    operation = str(row["metadata"]["operation"])
    cloned = dict(row)
    cloned["record_id"] = hashlib.sha256(f"authority_v1|{interface}|{split}|{row['record_id']}|{layout}|{side}".encode()).hexdigest()[:20]
    cloned["input"] = authority_input(interface, operation, a, b, side, layout, ordinal)
    cloned["condition"] = "AUTHORITY_ROLE_BINDING"
    cloned["metadata"] = {**row["metadata"], "authority_layout": layout, "authority_side": side, "authoritative_operation": operation}
    return cloned


def build_training_records(base_records: Mapping[str, Sequence[Mapping[str, Any]]], interface: str) -> List[Dict[str, Any]]:
    expanded, _ = base.exposure.expanded_records(base_records, interface, "train")
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in expanded:
        grouped[TRAIN_LAYOUT_MAP[str(row["metadata"]["layout"])]].append(row)
    output: List[Dict[str, Any]] = []
    for layout in TRAIN_AUTH_LAYOUTS:
        side_map = balanced_side_map(grouped[layout], layout)
        for ordinal, row in enumerate(grouped[layout]):
            output.append(authority_row(row, interface, "train", layout, side_map[str(row["record_id"])], ordinal))
    if len(output) != 240:
        raise RuntimeError(f"unexpected training authority rows: {len(output)}")
    return output


def build_eval_records(base_records: Mapping[str, Sequence[Mapping[str, Any]]], interface: str, split: str, layout: str) -> List[Dict[str, Any]]:
    source = base_records[split]
    forced = "EARLIER" if layout == "UNSEEN_NEAREST_DECOY" else None
    side_map = balanced_side_map(source, layout) if forced is None else {}
    rows: List[Dict[str, Any]] = []
    for ordinal, row in enumerate(source):
        side = forced or side_map[str(row["record_id"])]
        rows.append(authority_row(row, interface, split, layout, side, ordinal))
    return rows


def prepare_authority_parser(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    a_pattern = re.compile(rb"(?<![A-Za-z_])a(?:_digit)?=([0-9])")
    b_pattern = re.compile(rb"(?<![A-Za-z_])b(?:_digit)?=([0-9])")
    operation_pattern = re.compile(rb"(?<![A-Za-z_])operation=([FS])")
    authority_pattern = re.compile(rb"authority=(EARLIER|LATER)")
    prepared: List[Dict[str, Any]] = []
    for original in records:
        item = dict(original)
        raw = base.base.parser_input(item).encode("utf-8")
        a_match = a_pattern.search(raw)
        b_match = b_pattern.search(raw)
        operation_matches = list(operation_pattern.finditer(raw))
        authority_match = authority_pattern.search(raw)
        if a_match is None or b_match is None or len(operation_matches) != 2 or authority_match is None:
            raise RuntimeError(f"authority role span missing or ambiguous: {item['record_id']}")
        side = authority_match.group(1).decode("ascii")
        target_index = 0 if side == "EARLIER" else 1
        full = [stage_a.BOS_ID] + stage_a.serialize_bytes(raw)
        if len(full) > stage_a.CONTEXT_LENGTH:
            raise RuntimeError(f"authority record exceeds context: {item['record_id']} ({len(full)})")
        item["_parser_full"] = full
        item["_parser_length"] = len(full)
        item["_a_pos"] = 1 + a_match.start(1)
        item["_b_pos"] = 1 + b_match.start(1)
        item["_op_positions"] = [1 + match.start(1) for match in operation_matches]
        item["_op_pos"] = item["_op_positions"][target_index]
        item["_authority_side"] = side
        item["_authority_position"] = 1 + authority_match.start(1)
        prepared.append(item)
    return prepared


def authority_batchify(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    selected = [records[index] for index in indices]
    max_len = max(int(item["_parser_length"]) for item in selected)
    x = torch.full((len(selected), max_len), stage_a.PAD_ID, dtype=torch.long)
    lengths: List[int] = []
    targets: Dict[str, List[int]] = {key: [] for key in ("a", "b", "op", "a_pos", "b_pos", "op_pos")}
    for row, item in enumerate(selected):
        full = torch.tensor(item["_parser_full"], dtype=torch.long)
        x[row, : full.numel()] = full
        lengths.append(int(full.numel()))
        targets["a"].append(int(item["metadata"]["a"]))
        targets["b"].append(int(item["metadata"]["b"]))
        targets["op"].append(0 if item["metadata"]["operation"] == "SELECT_FIRST" else 1)
        targets["a_pos"].append(int(item["_a_pos"]))
        targets["b_pos"].append(int(item["_b_pos"]))
        targets["op_pos"].append(int(item["_op_pos"]))
    return {"x": x, "lengths": torch.tensor(lengths, dtype=torch.long), **{key: torch.tensor(value, dtype=torch.long) for key, value in targets.items()}}


def decode_digit(token_id: int) -> int | None:
    value = int(token_id) - int(stage_a.RAW_BYTE_START)
    return value - 48 if 48 <= value <= 57 else None


def decode_operation(token_id: int) -> str | None:
    value = int(token_id) - int(stage_a.RAW_BYTE_START)
    if value == ord("F"):
        return "SELECT_FIRST"
    if value == ord("S"):
        return "SELECT_SECOND"
    return None


def parser_predictions(model: Any, records: Sequence[Mapping[str, Any]], interface: str, seed: int, arm: str, layout: str, split: str) -> List[Dict[str, Any]]:
    prepared = prepare_authority_parser(records)
    rows: List[Dict[str, Any]] = []
    model.eval()
    for start in range(0, len(prepared), int(stage_a.BATCH_SIZE)):
        indices = list(range(start, min(start + int(stage_a.BATCH_SIZE), len(prepared))))
        batch = authority_batchify(prepared, indices)
        with torch.inference_mode():
            logits = model.forward_all(batch)
        for local, index in enumerate(indices):
            item = prepared[index]
            classifier_a = int(logits["a"][local].argmax().item())
            classifier_b = int(logits["b"][local].argmax().item())
            classifier_op = "SELECT_FIRST" if int(logits["op"][local].argmax().item()) == 0 else "SELECT_SECOND"
            a_pointer_pos = int(logits["a_pos"][local].argmax().item())
            b_pointer_pos = int(logits["b_pos"][local].argmax().item())
            op_pointer_pos = int(logits["op_pos"][local].argmax().item())
            full = item["_parser_full"]
            pointer_a = decode_digit(full[a_pointer_pos]) if 0 <= a_pointer_pos < len(full) else None
            pointer_b = decode_digit(full[b_pointer_pos]) if 0 <= b_pointer_pos < len(full) else None
            pointer_op = decode_operation(full[op_pointer_pos]) if 0 <= op_pointer_pos < len(full) else None
            target_a = int(item["metadata"]["a"]); target_b = int(item["metadata"]["b"]); target_op = str(item["metadata"]["operation"])
            rows.append({
                "arm": arm, "interface": interface, "seed": seed, "layout": layout, "split": split, "record_id": str(item["record_id"]),
                "a": target_a, "b": target_b, "operation": target_op, "authority_side": item["_authority_side"],
                "classifier_a": classifier_a, "classifier_b": classifier_b, "classifier_operation": classifier_op,
                "classifier_tuple_correct": classifier_a == target_a and classifier_b == target_b and classifier_op == target_op,
                "predicted_a": classifier_a, "predicted_b": classifier_b, "predicted_operation": classifier_op,
                "a_target_position": int(item["_a_pos"]), "b_target_position": int(item["_b_pos"]), "op_target_position": int(item["_op_pos"]),
                "a_pointer_position": a_pointer_pos, "b_pointer_position": b_pointer_pos, "op_pointer_position": op_pointer_pos,
                "a_position_correct": a_pointer_pos == int(item["_a_pos"]), "b_position_correct": b_pointer_pos == int(item["_b_pos"]), "op_position_correct": op_pointer_pos == int(item["_op_pos"]),
                "pointer_a": pointer_a, "pointer_b": pointer_b, "pointer_operation": pointer_op,
                "pointer_tuple_correct": pointer_a == target_a and pointer_b == target_b and pointer_op == target_op,
            })
    return rows


def tuple_loss(logits: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (F.cross_entropy(logits["a"], batch["a"]) + F.cross_entropy(logits["b"], batch["b"]) + F.cross_entropy(logits["op"], batch["op"])) / 3.0


def pointer_loss(logits: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (F.cross_entropy(logits["a_pos"], batch["a_pos"]) + F.cross_entropy(logits["b_pos"], batch["b_pos"]) + F.cross_entropy(logits["op_pos"], batch["op_pos"])) / 3.0


def pointer_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "rows": len(rows),
        "classifier_tuple_correct": sum(bool(row["classifier_tuple_correct"]) for row in rows),
        "classifier_tuple_accuracy": sum(bool(row["classifier_tuple_correct"]) for row in rows) / len(rows) if rows else None,
        "pointer_tuple_correct": sum(bool(row["pointer_tuple_correct"]) for row in rows),
        "pointer_tuple_accuracy": sum(bool(row["pointer_tuple_correct"]) for row in rows) / len(rows) if rows else None,
        "a_position_correct": sum(bool(row["a_position_correct"]) for row in rows),
        "b_position_correct": sum(bool(row["b_position_correct"]) for row in rows),
        "op_position_correct": sum(bool(row["op_position_correct"]) for row in rows),
        "a_position_accuracy": sum(bool(row["a_position_correct"]) for row in rows) / len(rows) if rows else None,
        "b_position_accuracy": sum(bool(row["b_position_correct"]) for row in rows) / len(rows) if rows else None,
        "op_position_accuracy": sum(bool(row["op_position_correct"]) for row in rows) / len(rows) if rows else None,
    }


def make_balanced_schedule(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for index, row in enumerate(records):
        key = (str(row["metadata"]["authority_side"]), str(row["metadata"]["operation"]), str(row["metadata"]["authority_layout"]))
        groups[key].append(index)
    expected = {(side, operation, layout) for side in ("EARLIER", "LATER") for operation in ("SELECT_FIRST", "SELECT_SECOND") for layout in TRAIN_AUTH_LAYOUTS}
    if set(groups) != expected or len({len(values) for values in groups.values()}) != 1:
        raise RuntimeError(f"authority strata are not equal: { {key: len(value) for key, value in sorted(groups.items())} }")
    per_group = len(next(iter(groups.values())))
    schedule: List[Dict[str, Any]] = []
    keys = sorted(groups)
    for step in range(1, STEPS + 1):
        record_indices = [groups[key][(step - 1) % per_group] for key in keys]
        schedule.append({"step": step, "record_indices": record_indices})
    return schedule


def train_parser(arm: str, interface: str, seed: int, records: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]], output: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = RoleSupervisedParser(seed)
    optimizer = stage_a.optimizer_for(model)
    prepared = prepare_authority_parser(records)
    use_pointer = arm in ("A2_AUTHORITY_POINTER", "A3_AUTHORITY_POINTER_BALANCED")
    traces: List[Dict[str, Any]] = []
    curves: List[Dict[str, Any]] = []
    for spec in schedule:
        step = int(spec["step"]); indices = [int(index) for index in spec["record_indices"]]
        batch = authority_batchify(prepared, indices)
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model.forward_all(batch)
        t_loss = tuple_loss(logits, batch)
        p_loss = pointer_loss(logits, batch)
        loss = t_loss + POINTER_LOSS_WEIGHT * p_loss if use_pointer else t_loss
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), stage_a.CLIP_NORM).item())
        optimizer.step()
        classifier = (logits["a"].argmax(dim=-1) == batch["a"]) & (logits["b"].argmax(dim=-1) == batch["b"]) & (logits["op"].argmax(dim=-1) == batch["op"])
        pointers = (logits["a_pos"].argmax(dim=-1) == batch["a_pos"]) & (logits["b_pos"].argmax(dim=-1) == batch["b_pos"]) & (logits["op_pos"].argmax(dim=-1) == batch["op_pos"])
        traces.append({"arm": arm, "interface": interface, "seed": seed, "step": step, "loss": float(loss.detach().item()), "tuple_loss": float(t_loss.detach().item()), "pointer_loss": float(p_loss.detach().item()), "classifier_tuple_correct": int(classifier.sum().item()), "pointer_tuple_correct": int(pointers.sum().item()), "pointer_loss_active": use_pointer, "examples": len(indices), "grad_norm": grad_norm})
        if step == 1 or step % 50 == 0 or step == STEPS:
            print(json.dumps({"event": "authority_training_progress", "arm": arm, "interface": interface, "seed": seed, "step": step, "loss": float(loss.detach().item())}, sort_keys=True), flush=True)
        if step in CURVE_STEPS:
            curves.append({"step": step, **pointer_metrics(parser_predictions(model, records, interface, seed, arm, "TRAINING_ALL_LAYOUTS", "train"))})
    checkpoint = output / "checkpoint.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": STEPS, "seed": seed, "interface": interface, "arm": arm, "pointer_loss_active": use_pointer, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule_digest": hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()}, checkpoint)
    train_rows = parser_predictions(model, records, interface, seed, arm, "TRAINING_ALL_LAYOUTS", "train")
    result = {"arm": arm, "interface": interface, "seed": seed, "steps": STEPS, "pointer_loss_active": use_pointer, "train_metrics": pointer_metrics(train_rows), "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint), "schedule_digest": hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()}
    write_json(output / "RESULTS.json", result); write_json(output / "TRAINING_CURVE.json", {"curve_steps": list(CURVE_STEPS), "points": curves}); write_jsonl(output / "TRAINING_TRACE.jsonl", traces)
    return result, traces, model


def execute(a: Any, b: Any, operation: Any) -> str:
    if not isinstance(a, int) or not isinstance(b, int) or operation not in ("SELECT_FIRST", "SELECT_SECOND"):
        return "<invalid>"
    return str(a if operation == "SELECT_FIRST" else b)


def pair_metric(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, int, int], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["split"] == split and int(row["a"]) != int(row["b"]):
            grouped[(str(row["interface"]), int(row["seed"]), int(row["a"]) * 10 + int(row["b"]))][str(row["operation"])] = row
    complete = [value for value in grouped.values() if set(value) == {"SELECT_FIRST", "SELECT_SECOND"}]
    both = sum(bool(value["SELECT_FIRST"]["output_token_correct"] and value["SELECT_SECOND"]["output_token_correct"]) for value in complete)
    return {"pair_groups": len(complete), "both_correct": both, "both_correct_rate": both / len(complete) if complete else None}


def output_metrics(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    values = [row for row in rows if row["split"] == split]
    parser_values = [row for row in values if row.get("parser_tuple_correct") is not None]
    return {"rows": len(values), "output_token_correct": sum(bool(row["output_token_correct"]) for row in values), "output_token_accuracy": sum(bool(row["output_token_correct"]) for row in values) / len(values) if values else None, "paired_output": pair_metric(values, split), "parser_rows": len(parser_values), "parser_tuple_correct": sum(bool(row["parser_tuple_correct"]) for row in parser_values), "parser_tuple_accuracy": sum(bool(row["parser_tuple_correct"]) for row in parser_values) / len(parser_values) if parser_values else None, "a_position_correct": sum(bool(row.get("a_position_correct")) for row in parser_values), "b_position_correct": sum(bool(row.get("b_position_correct")) for row in parser_values), "op_position_correct": sum(bool(row.get("op_position_correct")) for row in parser_values), "format_errors": sum(bool(row["output_format_error"]) for row in values)}


def evaluation_records(base_records: Mapping[str, Sequence[Mapping[str, Any]]], train_records: Sequence[Mapping[str, Any]], interface: str, split: str, layout: str) -> List[Dict[str, Any]]:
    if split == "train" and layout in TRAIN_AUTH_LAYOUTS:
        return [row for row in train_records if row["metadata"]["authority_layout"] == layout]
    return build_eval_records(base_records, interface, split, layout)


def evaluate_arm(arm: str, interface: str, seed: int, base_records: Mapping[str, Sequence[Mapping[str, Any]]], train_records: Sequence[Mapping[str, Any]], model: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    predictions: List[Dict[str, Any]] = []; evaluations: List[Dict[str, Any]] = []
    for layout in EVAL_LAYOUTS:
        for split in ("train", "validation"):
            records = evaluation_records(base_records, train_records, interface, split, layout)
            if arm == "E3_ORACLE_BINDING":
                parsed_rows = [{"record_id": str(row["record_id"]), "a": int(row["metadata"]["a"]), "b": int(row["metadata"]["b"]), "operation": str(row["metadata"]["operation"]), "predicted_a": int(row["metadata"]["a"]), "predicted_b": int(row["metadata"]["b"]), "predicted_operation": str(row["metadata"]["operation"]), "pointer_a": int(row["metadata"]["a"]), "pointer_b": int(row["metadata"]["b"]), "pointer_operation": str(row["metadata"]["operation"]), "parser_tuple_correct": True, "parser_a_correct": True, "parser_b_correct": True, "parser_operation_correct": True, "a_position_correct": True, "b_position_correct": True, "op_position_correct": True} for row in records]
            else:
                parsed_rows = parser_predictions(model, records, interface, seed, arm, layout, split)
            by_id = {str(row["record_id"]): row for row in parsed_rows}
            for record in records:
                parsed = by_id[str(record["record_id"])]
                use_pointer = arm in ("A0_FROZEN_V3_POINTER", "A2_AUTHORITY_POINTER", "A3_AUTHORITY_POINTER_BALANCED", "E3_ORACLE_BINDING")
                if use_pointer:
                    a = parsed.get("pointer_a"); b = parsed.get("pointer_b"); operation = parsed.get("pointer_operation")
                    parser_tuple_correct = parsed.get("pointer_tuple_correct", parsed.get("parser_tuple_correct"))
                    parser_a_correct = parsed.get("pointer_a") == int(record["metadata"]["a"])
                    parser_b_correct = parsed.get("pointer_b") == int(record["metadata"]["b"])
                    parser_operation_correct = parsed.get("pointer_operation") == str(record["metadata"]["operation"])
                else:
                    a = parsed.get("classifier_a", parsed.get("predicted_a")); b = parsed.get("classifier_b", parsed.get("predicted_b")); operation = parsed.get("classifier_operation", parsed.get("predicted_operation"))
                    parser_tuple_correct = parsed.get("classifier_tuple_correct", parsed.get("parser_tuple_correct"))
                    parser_a_correct = a == int(record["metadata"]["a"]); parser_b_correct = b == int(record["metadata"]["b"]); parser_operation_correct = operation == str(record["metadata"]["operation"])
                output = execute(a, b, operation)
                expected = str(record["target"])
                row = {"arm": arm, "interface": interface, "seed": seed, "layout": layout, "split": split, "record_id": str(record["record_id"]), "a": int(record["metadata"]["a"]), "b": int(record["metadata"]["b"]), "operation": str(record["metadata"]["operation"]), "authority_side": str(record["metadata"]["authority_side"]), "expected_output": expected, "predicted_output": output, "output_token_correct": output == expected, "full_sequence_correct": output == expected, "eos_correct": True, "output_format_error": output == "<invalid>", "parser_tuple_correct": parser_tuple_correct, "parser_a_correct": parser_a_correct, "parser_b_correct": parser_b_correct, "parser_operation_correct": parser_operation_correct, "a_position_correct": parsed.get("a_position_correct"), "b_position_correct": parsed.get("b_position_correct"), "op_position_correct": parsed.get("op_position_correct"), "predicted_a": a, "predicted_b": b, "predicted_operation": operation, "pointer_a": parsed.get("pointer_a"), "pointer_b": parsed.get("pointer_b"), "pointer_operation": parsed.get("pointer_operation")}
                predictions.append(row)
            current = [row for row in predictions if row["arm"] == arm and row["interface"] == interface and int(row["seed"]) == seed and row["layout"] == layout and row["split"] == split]
            evaluations.append({"arm": arm, "interface": interface, "seed": seed, "layout": layout, "split": split, "metrics": output_metrics(current, split)})
    return predictions, evaluations


def reproduce_previous() -> Dict[str, Any]:
    summary = json.loads((PREVIOUS_ROOT / "SUMMARY.json").read_text())
    finalization = json.loads((PREVIOUS_ROOT / "FINALIZATION.json").read_text())
    with (PREVIOUS_ROOT / "PREDICTIONS.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["condition"] == "E2_ROLE_SUPERVISED_POINTER" and row["split"] == "validation"]
    expected = {"FIXED_NEAR_PADDED": [32, 32], "EARLY_PADDED": [32, 32], "MIDDLE_POSITION": [32, 32], "UNSEEN_BETWEEN_OPERANDS": [27, 32], "UNSEEN_EARLY_CONTEXT_FIRST": [30, 32], "UNSEEN_DIGIT_DISTRACTOR": [15, 32], "UNSEEN_COMPETING_OPERATION": [1, 32]}
    actual: Dict[str, List[int]] = {}
    for layout in expected:
        subset = [row for row in rows if row["layout"] == layout]
        grouped: Dict[Tuple[str, str, str], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in subset:
            if int(row["a"]) != int(row["b"]): grouped[(row["interface"], row["seed"], row["a"], row["b"])][row["operation"]] = row
        complete = [value for value in grouped.values() if len(value) == 2]
        actual[layout] = [sum(value["SELECT_FIRST"]["output_token_correct"] == "True" and value["SELECT_SECOND"]["output_token_correct"] == "True" for value in complete), len(complete)]
    reproduced = actual == expected
    checkpoint_hashes = all(Path(item["checkpoint"]).is_file() and sha256(Path(item["checkpoint"])) == item["checkpoint_sha256"] for item in summary["parsers"])
    if not reproduced or not checkpoint_hashes or not finalization["verification_pass"] or finalization["sealed_test_accessed"]:
        raise RuntimeError(f"v3 reproduction failed: {actual}")
    return {"passed": True, "paired_validation": actual, "checkpoint_hashes": True, "sealed_test_accessed": False, "source_finalization": rel(PREVIOUS_ROOT / "FINALIZATION.json")}


def audit_data(records_by_interface: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], train_authority: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    audit: Dict[str, Any] = {}
    for interface, base_records in records_by_interface.items():
        train_ids = {str(row["record_id"]) for row in base_records["train"]}; validation_ids = {str(row["record_id"]) for row in base_records["validation"]}
        train_rows = train_authority[interface]
        max_len = max(len(str(row["input"]).encode("utf-8")) + 1 for row in train_rows)
        sides = Counter(str(row["metadata"]["authority_side"]) for row in train_rows); operations = Counter(str(row["metadata"]["operation"]) for row in train_rows); layouts = Counter(str(row["metadata"]["authority_layout"]) for row in train_rows)
        audit[interface] = {"base_train_rows": len(train_ids), "base_validation_rows": len(validation_ids), "base_id_overlap": len(train_ids & validation_ids), "authority_training_rows": len(train_rows), "max_serialized_input_bytes_plus_bos": max_len, "authority_side_counts": dict(sides), "operation_counts": dict(operations), "layout_counts": dict(layouts)}
        if train_ids & validation_ids or len(train_rows) != 240 or max_len > stage_a.CONTEXT_LENGTH:
            raise RuntimeError(f"data audit failed for {interface}: {audit[interface]}")
    return audit


def report(summary: Mapping[str, Any]) -> str:
    lines = ["# Crucible Authority-Aware Role Binding Gate v1 — Corrected Final Report", "", "CPU-only authority-selection intervention with explicit training distractors. Frozen v3 parser and oracle controls; no sealed-test access.", "", "## v3 baseline reproduction", "", json.dumps(summary["baseline_reproduction"], indent=2, sort_keys=True), "", "## Training fit", "", "| Arm | Interface | Seed | Classifier tuple | Pointer tuple | A position | B position | Operation position |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for item in summary["trained_arms"]:
        m = item["train_metrics"]; lines.append(f"| {item['arm']} | {item['interface']} | {item['seed']} | {m['classifier_tuple_accuracy']:.2%} | {m['pointer_tuple_accuracy']:.2%} | {m['a_position_accuracy']:.2%} | {m['b_position_accuracy']:.2%} | {m['op_position_accuracy']:.2%} |")
    lines += ["", "## Validation paired correctness", "", "Each cell is correct paired groups / 32 pooled across four interface/seed runs.", "", "| Arm | Authority-near | Authority-early | Authority-middle | Reversed authority | Digit distractor | Nearest decoy | Conflicting cues |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        cells = []
        for layout in EVAL_LAYOUTS:
            values = [row for row in summary["evaluations"] if row["arm"] == arm and row["layout"] == layout and row["split"] == "validation"]
            cells.append(f"{sum(int(row['metrics']['paired_output']['both_correct']) for row in values)}/{sum(int(row['metrics']['paired_output']['pair_groups']) for row in values)}")
        lines.append(f"| {arm} | {' | '.join(cells)} |")
    lines += ["", "## Interpretation boundary", "", "Authority pointer supervision is evaluated as a learned selection intervention under this synthetic task. It does not establish general authority reasoning or position-independent relevance selection.", "", "## Artifacts", "", "- [Protocol](../../docs/CRUCIBLE_AUTHORITY_AWARE_ROLE_BINDING_PROTOCOL_20251025_v2.md)", "- [Summary](SUMMARY.json)", "- [Analysis](ANALYSIS.json)", "- [Predictions](PREDICTIONS.csv)", "- [Parser predictions](PARSER_PREDICTIONS.csv)", "- [Scientific interpretation](SCIENTIFIC_INTERPRETATION.md)", "- [Finalization](FINALIZATION.json)"]
    return "\n".join(lines) + "\n"


def main() -> int:
    resume = OUTPUT.exists() and (OUTPUT / "trained_arms").is_dir() and not (OUTPUT / "SUMMARY.json").exists()
    if OUTPUT.exists() and any(OUTPUT.iterdir()) and not resume:
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline = reproduce_previous()
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {"experiment": "crucible_authority_aware_role_binding_20251025_v2", "protocol": rel(PROTOCOL), "arms": list(ARMS), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "steps": STEPS, "effective_batch": EFFECTIVE_BATCH, "pointer_loss_weight": POINTER_LOSS_WEIGHT, "sealed_test_accessed": False, "architecture_changed": False, "executor_retrained": False, "baseline_reproduction": baseline, "corrected_training_distractors": True})
    base_by_interface: Dict[str, Mapping[str, Sequence[Mapping[str, Any]]]] = {}; train_by_interface: Dict[str, List[Dict[str, Any]]] = {}; data_audit_input: Dict[str, Mapping[str, Sequence[Mapping[str, Any]]]] = {}; audits: Dict[str, Any] = {}
    for interface in INTERFACES:
        base_records = base.exposure.base_records(interface, 42); train_records = build_training_records(base_records, interface)
        base_by_interface[interface] = base_records; train_by_interface[interface] = train_records; data_audit_input[interface] = base_records
        write_jsonl(OUTPUT / "data" / interface / "authority_train.jsonl", train_records); write_jsonl(OUTPUT / "data" / interface / "base_train.jsonl", base_records["train"]); write_jsonl(OUTPUT / "data" / interface / "base_validation.jsonl", base_records["validation"])
        audits[interface] = audit_data({interface: base_records}, {interface: train_records})[interface]
    schedules = json.loads((BATCH_ROOT / "BATCH_SCHEDULES.json").read_text())["B0_DIVERSE_SHUFFLED"]
    balanced_schedules = {interface: make_balanced_schedule(train_by_interface[interface]) for interface in INTERFACES}
    write_json(OUTPUT / "SCHEDULE_REFERENCE.json", {"b0_source": rel(BATCH_ROOT / "BATCH_SCHEDULES.json"), "b0_arm": "B0_DIVERSE_SHUFFLED", "b0_steps": len(schedules), "balanced_steps": {interface: len(value) for interface, value in balanced_schedules.items()}, "balanced_schedule_digest": {interface: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() for interface, value in balanced_schedules.items()}})
    write_json(OUTPUT / "DATA_AUDIT.json", audits)

    trained_arms: List[Dict[str, Any]] = []; models: Dict[Tuple[str, int, str], Any] = {}; traces: List[Dict[str, Any]] = []
    train_arms = ("A1_AUTHORITY_TUPLE_NO_POINTER", "A2_AUTHORITY_POINTER", "A3_AUTHORITY_POINTER_BALANCED")
    if resume:
        for arm in train_arms:
            for interface in INTERFACES:
                for seed in SEEDS:
                    output = OUTPUT / "trained_arms" / arm / interface / f"seed{seed}"
                    state = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
                    model = RoleSupervisedParser(seed); model.load_state_dict(state["model"], strict=True); model.eval()
                    trained_arms.append(json.loads((output / "RESULTS.json").read_text()))
                    traces.extend(json.loads(line) for line in (output / "TRAINING_TRACE.jsonl").read_text().splitlines() if line.strip())
                    models[(arm, interface, seed)] = model
        print(json.dumps({"event": "authority_training_resume", "trained_arms": len(trained_arms)}, sort_keys=True), flush=True)
    else:
        for arm in train_arms:
            for interface in INTERFACES:
                schedule = schedules if arm != "A3_AUTHORITY_POINTER_BALANCED" else balanced_schedules[interface]
                for seed in SEEDS:
                    output = OUTPUT / "trained_arms" / arm / interface / f"seed{seed}"
                    result, arm_traces, model = train_parser(arm, interface, seed, train_by_interface[interface], schedule, output)
                    trained_arms.append(result); traces.extend(arm_traces); models[(arm, interface, seed)] = model
                    print(json.dumps({"event": "authority_parser_complete", "arm": arm, "interface": interface, "seed": seed, "classifier_tuple": result["train_metrics"]["classifier_tuple_accuracy"], "pointer_tuple": result["train_metrics"]["pointer_tuple_accuracy"]}, sort_keys=True), flush=True)

    predictions: List[Dict[str, Any]] = []; evaluations: List[Dict[str, Any]] = []
    for interface in INTERFACES:
        base_records = base_by_interface[interface]; train_records = train_by_interface[interface]
        for seed in SEEDS:
            old_path = PREVIOUS_ROOT / "parser_outputs" / interface / f"seed{seed}" / "checkpoint.pt"
            if not old_path.is_file(): raise FileNotFoundError(old_path)
            old_model = base.load_role_parser(old_path, seed)
            controls: List[Tuple[str, Any]] = [("A0_FROZEN_V3_POINTER", old_model), ("A1_AUTHORITY_TUPLE_NO_POINTER", models[("A1_AUTHORITY_TUPLE_NO_POINTER", interface, seed)]), ("A2_AUTHORITY_POINTER", models[("A2_AUTHORITY_POINTER", interface, seed)]), ("A3_AUTHORITY_POINTER_BALANCED", models[("A3_AUTHORITY_POINTER_BALANCED", interface, seed)]), ("E3_ORACLE_BINDING", None)]
            for arm, model in controls:
                current_predictions, current_evaluations = evaluate_arm(arm, interface, seed, base_records, train_records, model)
                for row in current_predictions:
                    row["checkpoint"] = rel(old_path) if arm == "A0_FROZEN_V3_POINTER" else (rel(OUTPUT / "trained_arms" / arm / interface / f"seed{seed}" / "checkpoint.pt") if arm != "E3_ORACLE_BINDING" else None)
                predictions.extend(current_predictions); evaluations.extend(current_evaluations)
            del old_model; gc.collect()
            print(json.dumps({"event": "authority_interface_seed_complete", "interface": interface, "seed": seed}, sort_keys=True), flush=True)

    parser_predictions_all: List[Dict[str, Any]] = []
    for arm in ("A0_FROZEN_V3_POINTER", "A1_AUTHORITY_TUPLE_NO_POINTER", "A2_AUTHORITY_POINTER", "A3_AUTHORITY_POINTER_BALANCED"):
        for interface in INTERFACES:
            for seed in SEEDS:
                model = models[(arm, interface, seed)] if arm != "A0_FROZEN_V3_POINTER" else base.load_role_parser(PREVIOUS_ROOT / "parser_outputs" / interface / f"seed{seed}" / "checkpoint.pt", seed)
                for layout in EVAL_LAYOUTS:
                    for split in ("train", "validation"):
                        parser_predictions_all.extend(parser_predictions(model, evaluation_records(base_by_interface[interface], train_by_interface[interface], interface, split, layout), interface, seed, arm, layout, split))
                del model; gc.collect()

    with (OUTPUT / "PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(predictions[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(predictions)
    with (OUTPUT / "PARSER_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(parser_predictions_all[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(parser_predictions_all)
    write_jsonl(OUTPUT / "TRAINING_TRACE.jsonl", traces)
    summary = {"experiment": "crucible_authority_aware_role_binding_20251025_v2", "status": "completed", "baseline_reproduction": baseline, "data_audit": audits, "trained_arms": trained_arms, "evaluations": evaluations, "prediction_rows": len(predictions), "parser_prediction_rows": len(parser_predictions_all), "training_trace_rows": len(traces), "sealed_test_accessed": False, "architecture_changed": False, "executor_retrained": False, "corrected_training_distractors": True}
    write_json(OUTPUT / "SUMMARY.json", summary); (OUTPUT / "REPORT_FINAL.md").write_text(report(summary), encoding="utf-8")
    analysis = {"experiment": summary["experiment"], "status": "completed", "baseline_reproduction": baseline, "data_audit": audits, "validation_paired_by_arm_and_layout": {arm: {layout: {"correct": sum(int(row["metrics"]["paired_output"]["both_correct"]) for row in evaluations if row["arm"] == arm and row["layout"] == layout and row["split"] == "validation"), "total": sum(int(row["metrics"]["paired_output"]["pair_groups"]) for row in evaluations if row["arm"] == arm and row["layout"] == layout and row["split"] == "validation")} for layout in EVAL_LAYOUTS} for arm in ARMS}, "integrity": {"prediction_rows": len(predictions), "parser_prediction_rows": len(parser_predictions_all), "evaluations": len(evaluations), "sealed_test_accessed": False, "architecture_changed": False, "executor_retrained": False}}
    write_json(OUTPUT / "ANALYSIS.json", analysis)
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt": hashes[rel(path)] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = {"status": "completed", "verification_pass": True, "sealed_test_accessed": False, "architecture_changed": False, "executor_retrained": False, "trained_arms": len(trained_arms), "evaluations": len(evaluations), "prediction_rows": len(predictions), "parser_prediction_rows": len(parser_predictions_all), "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"), "analysis_sha256": sha256(OUTPUT / "ANALYSIS.json"), "prediction_sha256": sha256(OUTPUT / "PREDICTIONS.csv")}
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "completed", "output": rel(OUTPUT), "trained_arms": len(trained_arms), "evaluations": len(evaluations), "prediction_rows": len(predictions), "verification_pass": True}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
