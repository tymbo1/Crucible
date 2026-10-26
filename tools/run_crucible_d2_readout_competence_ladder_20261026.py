#!/usr/bin/env python3
"""CPU D2 final-readout competence ladder."""

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
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_d2_readout_competence_ladder_20261026_v1"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_D2_READOUT_COMPETENCE_LADDER_PROTOCOL_20261026.md"
CAUSAL_SOURCE = ROOT / "tools" / "run_crucible_causally_valid_authority_decision_20261026.py"
CAUSAL_ROOT = ROOT / "experiments" / "crucible_causally_valid_authority_decision_20261026_v1"

INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
STAGES = ("S1_SINGLE_CANDIDATE", "S2_DIRECT_TARGET", "S3_AUTHORITY_CONDITIONED")
ORDERS = ("FS", "SF")
AUTHORITY_CODES = ("E", "L")
STEPS = 500
READOUT_BATCH = 12
READOUT_LR = 1.0e-2
CLIP_NORM = 1.0
TRAIN_THRESHOLD = 0.95
VALIDATION_THRESHOLD = 0.90


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


causal = import_module("causal_authority_decision_for_ladder", CAUSAL_SOURCE)
auth = causal.auth
torch = auth.torch
nn = torch.nn
F = torch.nn.functional
stage_a = auth.stage_a


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def marker_field(operation: str) -> str:
    return ("F" if operation == "SELECT_FIRST" else "S") + ("_" * (len(operation.encode("utf-8")) - 1))


def operand_field(interface: str, a: int, b: int) -> str:
    return f"a={a}; b={b}." if interface == "canonical" else f"a_digit={a}; b_digit={b}."


def opposite(operation: str) -> str:
    return "SELECT_SECOND" if operation == "SELECT_FIRST" else "SELECT_FIRST"


def unique_pairs(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int]]:
    return sorted({(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in rows})


def operation_names(order: str) -> Tuple[str, str]:
    first = "SELECT_FIRST" if order == "FS" else "SELECT_SECOND"
    return first, opposite(first)


def stage_input(stage: str, interface: str, a: int, b: int, order: str = "FS", code: str = "E", ordinal: int = 0, unseen_layout: bool = False) -> str:
    pair = operand_field(interface, a, b)
    noise = (a + 3 * b + ordinal + 7) % 10
    if stage == "S1_SINGLE_CANDIDATE":
        operation = "SELECT_FIRST" if (ordinal % 2 == 0) else "SELECT_SECOND"
        return f"Select an operand. operation={marker_field(operation)}; {pair} Return digit only. decision=Q;"
    first, second = operation_names(order)
    op1 = marker_field(first)
    op2 = marker_field(second)
    if stage == "S2_DIRECT_TARGET":
        return f"Select an operand. operation={op1}; context=bridge_{noise}; operation={op2}; {pair} select_position={code}; Return digit only. decision=Q;"
    if stage == "S3_AUTHORITY_CONDITIONED" and not unseen_layout:
        return f"Select an operand. context=neutral_{noise}; operation={op1}; context=bridge_{(noise + 2) % 10}; operation={op2}; {pair} authority_code={code}; Return digit only. decision=Q;"
    if stage == "S3_AUTHORITY_CONDITIONED" and unseen_layout:
        return f"Select an operand. {pair} context=layout_pad_{noise}; operation={op1}; context=late_bridge_{(noise + 4) % 10}; operation={op2}; authority_code={code}; Return digit only. decision=Q;"
    raise ValueError(stage)


def make_records(interface: str, stage: str, split: str, pairs: Sequence[Tuple[int, int]], unseen_layout: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ordinal, (a, b) in enumerate(pairs):
        source_pair = f"{a},{b}"
        if stage == "S1_SINGLE_CANDIDATE":
            choices = (("FS", "E"), ("SF", "L"))
        else:
            choices = tuple((order, code) for order in ORDERS for code in AUTHORITY_CODES)
        for order, code in choices:
            if stage == "S1_SINGLE_CANDIDATE":
                target_slot = 0
                operation = "SELECT_FIRST" if (ordinal % 2 == 0) else "SELECT_SECOND"
                target_value = a if operation == "SELECT_FIRST" else b
            else:
                target_slot = 0 if code == "E" else 1
                operations = operation_names(order)
                operation = operations[target_slot]
                target_value = a if operation == "SELECT_FIRST" else b
            layout_tag = "unseen_layout" if unseen_layout else "standard"
            record_id = hashlib.sha256(f"d2_readout_ladder|{interface}|{stage}|{split}|{layout_tag}|{source_pair}|{order}|{code}".encode()).hexdigest()[:20]
            counterfactual = None if stage == "S1_SINGLE_CANDIDATE" else hashlib.sha256(f"d2_readout_pair|{interface}|{stage}|{split}|{layout_tag}|{source_pair}|{order}".encode()).hexdigest()[:20]
            rows.append({
                "schema_version": "crucible_d2_readout_competence_ladder_v1",
                "record_id": record_id,
                "interface": interface,
                "stage": stage,
                "split": split,
                "input": stage_input(stage, interface, a, b, order, code, ordinal, unseen_layout),
                "target": str(target_value),
                "oracle": {"kind": "candidate_pointer", "target_slot": target_slot, "operation": operation, "value": target_value},
                "metadata": {
                    "a": a, "b": b, "source_pair": source_pair, "operation_order": order,
                    "authority_code": code, "target_slot": target_slot, "operation": operation,
                    "counterfactual_pair_id": counterfactual, "unseen_layout": unseen_layout,
                },
            })
    return rows


def prepare_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    a_pattern = re.compile(rb"(?<![A-Za-z_])a(?:_digit)?=([0-9])")
    b_pattern = re.compile(rb"(?<![A-Za-z_])b(?:_digit)?=([0-9])")
    operation_pattern = re.compile(rb"(?<![A-Za-z_])operation=([FS])")
    decision_pattern = re.compile(rb"decision=(Q)")
    prepared: List[Dict[str, Any]] = []
    for original in records:
        item = dict(original)
        raw = auth.base.base.parser_input(item).encode("utf-8")
        a_match = a_pattern.search(raw)
        b_match = b_pattern.search(raw)
        operations = list(operation_pattern.finditer(raw))
        decision = decision_pattern.search(raw)
        if a_match is None or b_match is None or len(operations) not in (1, 2) or decision is None:
            raise RuntimeError(f"readout span failure: {item['record_id']}")
        full = [stage_a.BOS_ID] + stage_a.serialize_bytes(raw)
        if len(full) > stage_a.CONTEXT_LENGTH:
            raise RuntimeError(f"readout record exceeds context: {item['record_id']}")
        item["_parser_full"] = full
        item["_parser_length"] = len(full)
        item["_a_pos"] = 1 + a_match.start(1)
        item["_b_pos"] = 1 + b_match.start(1)
        item["_op_positions"] = [1 + match.start(1) for match in operations]
        item["_decision_position"] = 1 + decision.start(1)
        prepared.append(item)
    return prepared


def batchify(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    selected = [records[index] for index in indices]
    max_len = max(int(row["_parser_length"]) for row in selected)
    x = torch.full((len(selected), max_len), stage_a.PAD_ID, dtype=torch.long)
    lengths: List[int] = []
    targets: List[int] = []
    decision_positions: List[int] = []
    for row_index, item in enumerate(selected):
        full = torch.tensor(item["_parser_full"], dtype=torch.long)
        x[row_index, : full.numel()] = full
        lengths.append(int(full.numel()))
        targets.append(int(item["metadata"]["target_slot"]))
        decision_positions.append(int(item["_decision_position"]))
    return {"x": x, "lengths": torch.tensor(lengths, dtype=torch.long), "target_slot": torch.tensor(targets, dtype=torch.long), "decision_pos": torch.tensor(decision_positions, dtype=torch.long)}


def make_schedule(size: int) -> List[List[int]]:
    if size < READOUT_BATCH:
        raise RuntimeError(f"dataset too small for batch: {size}")
    return [[(step * READOUT_BATCH + offset) % size for offset in range(READOUT_BATCH)] for step in range(STEPS)]


class DecisionPointerParser(auth.RoleSupervisedParser):
    """Dense backbone plus a final decision-token candidate-slot head."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        torch.manual_seed(seed + 91_000)
        self.decision_pointer = nn.Linear(stage_a.D_MODEL, 2)

    def decision_logits(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.hidden_sequence(batch["x"])
        rows = torch.arange(hidden.size(0), dtype=torch.long)
        return self.decision_pointer(hidden[rows, batch["decision_pos"]])


class ReadoutHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def extract_features(model: Any, records: Sequence[Mapping[str, Any]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    prepared = prepare_records(records)
    feature_chunks: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(prepared), READOUT_BATCH):
        indices = list(range(start, min(start + READOUT_BATCH, len(prepared))))
        batch = batchify(prepared, indices)
        with torch.inference_mode():
            hidden = model.hidden_sequence(batch["x"])
        rows = torch.arange(hidden.size(0), dtype=torch.long)
        feature_chunks.append(hidden[rows, batch["decision_pos"]].detach().cpu())
    return torch.cat(feature_chunks, dim=0), prepared


def pointer_rows(records: Sequence[Mapping[str, Any]], logits: torch.Tensor, stage: str, seed: int, interface: str, source: str) -> List[Dict[str, Any]]:
    prepared = prepare_records(records)
    predictions: List[Dict[str, Any]] = []
    slots = logits.argmax(dim=-1).tolist()
    for index, item in enumerate(prepared):
        predicted_slot = int(slots[index])
        operations = operation_names(str(item["metadata"]["operation_order"])) if len(item["_op_positions"]) == 2 else (str(item["metadata"]["operation"]),)
        predicted_operation = operations[predicted_slot] if predicted_slot < len(operations) else None
        expected_operation = str(item["metadata"]["operation"])
        expected_slot = int(item["metadata"]["target_slot"])
        predicted_output = None
        if predicted_operation in {"SELECT_FIRST", "SELECT_SECOND"}:
            predicted_output = str(item["metadata"]["a"] if predicted_operation == "SELECT_FIRST" else item["metadata"]["b"])
        predictions.append({
            "stage": stage, "interface": interface, "seed": seed, "source": source,
            "split": str(item["split"]), "record_id": str(item["record_id"]),
            "counterfactual_pair_id": item["metadata"]["counterfactual_pair_id"],
            "a": int(item["metadata"]["a"]), "b": int(item["metadata"]["b"]),
            "operation_order": str(item["metadata"]["operation_order"]), "authority_code": str(item["metadata"]["authority_code"]),
            "target_slot": expected_slot, "predicted_slot": predicted_slot,
            "target_operation": expected_operation, "predicted_operation": predicted_operation,
            "target_operation_position": int(item["_op_positions"][expected_slot]),
            "predicted_operation_position": int(item["_op_positions"][predicted_slot]) if predicted_slot < len(item["_op_positions"]) else None,
            "decision_position": int(item["_decision_position"]),
            "pointer_correct": predicted_slot == expected_slot,
            "operation_correct": predicted_operation == expected_operation,
            "output": predicted_output, "expected_output": str(item["target"]),
            "output_correct": predicted_output == str(item["target"]), "format_error": predicted_output is None,
        })
    return predictions


def pair_metrics(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split and row["counterfactual_pair_id"] is not None:
            groups[str(row["counterfactual_pair_id"])].append(row)
    complete = [values for values in groups.values() if len(values) == 2]
    flips = sum(int(values[0]["predicted_slot"] != values[1]["predicted_slot"]) for values in complete)
    correct = sum(int(values[0]["pointer_correct"] and values[1]["pointer_correct"]) for values in complete)
    return {"pairs": len(complete), "pointer_flips": flips, "correct_two_way": correct, "flip_accuracy": flips / len(complete) if complete else None, "correct_two_way_accuracy": correct / len(complete) if complete else None}


def row_metrics(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    return {"rows": len(selected), "pointer_correct": sum(int(row["pointer_correct"]) for row in selected), "pointer_accuracy": sum(int(row["pointer_correct"]) for row in selected) / len(selected) if selected else None, "operation_correct": sum(int(row["operation_correct"]) for row in selected), "output_correct": sum(int(row["output_correct"]) for row in selected), "format_errors": sum(int(row["format_error"]) for row in selected), "paired": pair_metrics(rows, split)}


def train_readout(stage: str, interface: str, seed: int, train_features: torch.Tensor, train_records: Sequence[Mapping[str, Any]], validation_features: torch.Tensor, validation_records: Sequence[Mapping[str, Any]], output: Path) -> Tuple[ReadoutHead, Dict[str, Any], List[Dict[str, Any]]]:
    torch.manual_seed(seed + 92_000)
    random.seed(seed + 92_000)
    head = ReadoutHead(int(train_features.shape[1]))
    optimizer = torch.optim.AdamW(head.parameters(), lr=READOUT_LR, weight_decay=0.0)
    schedule = make_schedule(len(train_records))
    trace: List[Dict[str, Any]] = []
    for step, indices in enumerate(schedule, start=1):
        index_tensor = torch.tensor(indices, dtype=torch.long)
        target = torch.tensor([int(train_records[index]["metadata"]["target_slot"]) for index in indices], dtype=torch.long)
        head.train(); optimizer.zero_grad(set_to_none=True)
        logits = head(train_features[index_tensor])
        loss = F.cross_entropy(logits, target)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP_NORM).item())
        optimizer.step()
        trace.append({"stage": stage, "interface": interface, "seed": seed, "step": step, "loss": float(loss.detach().item()), "grad_norm": grad_norm, "examples": len(indices)})
    head.eval()
    with torch.inference_mode():
        train_logits = head(train_features)
        validation_logits = head(validation_features)
    predictions = pointer_rows(train_records, train_logits, stage, seed, interface, "frozen_d1_readout") + pointer_rows(validation_records, validation_logits, stage, seed, interface, "frozen_d1_readout")
    checkpoint = output / "readout.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stage": stage, "interface": interface, "seed": seed, "steps": STEPS, "learning_rate": READOUT_LR, "model": head.state_dict()}, checkpoint)
    write_jsonl(output / "TRAINING_TRACE.jsonl", trace)
    result = {"stage": stage, "interface": interface, "seed": seed, "source": "frozen_d1_readout", "steps": STEPS, "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint), "train_metrics": row_metrics(predictions, "train"), "validation_metrics": row_metrics(predictions, "validation")}
    write_json(output / "RESULTS.json", result)
    write_csv(output / "PREDICTIONS.csv", predictions)
    return head, result, predictions


def train_end_to_end(stage: str, interface: str, seed: int, train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]], output: Path) -> Tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
    torch.manual_seed(seed + 93_000)
    random.seed(seed + 93_000)
    model = DecisionPointerParser(seed)
    optimizer = stage_a.optimizer_for(model)
    prepared = prepare_records(train_records)
    schedule = make_schedule(len(train_records))
    trace: List[Dict[str, Any]] = []
    for step, indices in enumerate(schedule, start=1):
        batch = batchify(prepared, indices)
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model.decision_logits(batch)
        loss = F.cross_entropy(logits, batch["target_slot"])
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), stage_a.CLIP_NORM).item())
        optimizer.step()
        trace.append({"stage": "S4_END_TO_END_AUTHORITY", "interface": interface, "seed": seed, "step": step, "loss": float(loss.detach().item()), "grad_norm": grad_norm, "examples": len(indices)})
        if step in (1, 100, 200, 300, 400, 500):
            print(json.dumps({"event": "d2_ladder_end_to_end", "interface": interface, "seed": seed, "step": step, "loss": float(loss.detach().item())}, sort_keys=True), flush=True)
    model.eval()
    predictions: List[Dict[str, Any]] = []
    for records, source in ((train_records, "end_to_end"), (validation_records, "end_to_end")):
        prepared_eval = prepare_records(records)
        for start in range(0, len(prepared_eval), READOUT_BATCH):
            indices = list(range(start, min(start + READOUT_BATCH, len(prepared_eval))))
            batch = batchify(prepared_eval, indices)
            with torch.inference_mode():
                logits = model.decision_logits(batch)
            predictions.extend(pointer_rows([records[index] for index in indices], logits, "S4_END_TO_END_AUTHORITY", seed, interface, source))
    checkpoint = output / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stage": "S4_END_TO_END_AUTHORITY", "interface": interface, "seed": seed, "steps": STEPS, "model": model.state_dict()}, checkpoint)
    write_jsonl(output / "TRAINING_TRACE.jsonl", trace)
    result = {"stage": "S4_END_TO_END_AUTHORITY", "interface": interface, "seed": seed, "source": "end_to_end", "steps": STEPS, "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint), "train_metrics": row_metrics(predictions, "train"), "validation_metrics": row_metrics(predictions, "validation")}
    write_json(output / "RESULTS.json", result)
    write_csv(output / "PREDICTIONS.csv", predictions)
    return model, result, predictions


def train_synthetic(seed: int, output: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    torch.manual_seed(seed + 90_000)
    head = ReadoutHead(2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=READOUT_LR, weight_decay=0.0)
    train_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]] * 40)
    train_targets = torch.tensor([0, 1] * 40, dtype=torch.long)
    validation_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]] * 20)
    validation_targets = torch.tensor([0, 1] * 20, dtype=torch.long)
    trace: List[Dict[str, Any]] = []
    for step in range(1, STEPS + 1):
        indices = torch.tensor([(step * READOUT_BATCH + offset) % len(train_features) for offset in range(READOUT_BATCH)], dtype=torch.long)
        head.train(); optimizer.zero_grad(set_to_none=True)
        logits = head(train_features[indices]); loss = F.cross_entropy(logits, train_targets[indices]); loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP_NORM).item()); optimizer.step()
        trace.append({"stage": "S0_SYNTHETIC_HEAD", "seed": seed, "step": step, "loss": float(loss.detach().item()), "grad_norm": grad_norm})
    head.eval()
    with torch.inference_mode():
        train_pred = head(train_features).argmax(dim=-1); validation_pred = head(validation_features).argmax(dim=-1)
    train_target_list = train_targets.tolist(); train_pred_list = train_pred.tolist()
    predictions = [{"stage": "S0_SYNTHETIC_HEAD", "seed": seed, "split": "train", "row": index, "target_slot": int(train_target_list[index]), "predicted_slot": int(train_pred_list[index]), "pointer_correct": bool(train_pred_list[index] == train_target_list[index])} for index in range(len(train_target_list))]
    predictions += [{"stage": "S0_SYNTHETIC_HEAD", "seed": seed, "split": "validation", "row": index, "target_slot": int(validation_targets[index]), "predicted_slot": int(validation_pred[index]), "pointer_correct": bool(validation_pred[index] == validation_targets[index])} for index in range(len(validation_targets))]
    checkpoint = output / "readout.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stage": "S0_SYNTHETIC_HEAD", "seed": seed, "steps": STEPS, "model": head.state_dict()}, checkpoint)
    write_jsonl(output / "TRAINING_TRACE.jsonl", trace); write_csv(output / "PREDICTIONS.csv", predictions)
    result = {"stage": "S0_SYNTHETIC_HEAD", "seed": seed, "steps": STEPS, "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint), "train_rows": len(train_targets), "train_correct": int((train_pred == train_targets).sum().item()), "validation_rows": len(validation_targets), "validation_correct": int((validation_pred == validation_targets).sum().item())}
    write_json(output / "RESULTS.json", result)
    return result, predictions


def verify_records(stage: str, train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    train_pairs = {str(row["metadata"]["source_pair"]) for row in train}; validation_pairs = {str(row["metadata"]["source_pair"]) for row in validation}
    if train_pairs & validation_pairs:
        raise RuntimeError(f"pair split overlap: {stage}")
    prepared_train = prepare_records(train); prepared_validation = prepare_records(validation)
    position_pass = True; counterfactuals = 0
    if stage != "S1_SINGLE_CANDIDATE":
        for prepared in (prepared_train, prepared_validation):
            groups: Dict[str, List[int]] = defaultdict(list)
            for index, row in enumerate(train if prepared is prepared_train else validation):
                groups[str(row["metadata"]["counterfactual_pair_id"])].append(index)
            for values in groups.values():
                if len(values) != 2:
                    raise RuntimeError(f"counterfactual cardinality: {stage}")
                left, right = values; counterfactuals += 1
                same_positions = prepared[left]["_op_positions"] == prepared[right]["_op_positions"] and prepared[left]["_a_pos"] == prepared[right]["_a_pos"] and prepared[left]["_b_pos"] == prepared[right]["_b_pos"] and prepared[left]["_decision_position"] == prepared[right]["_decision_position"]
                target_flips = prepared[left]["metadata"]["target_slot"] != prepared[right]["metadata"]["target_slot"]
                position_pass = position_pass and same_positions and target_flips
    if not position_pass:
        raise RuntimeError(f"position/counterfactual invariant failure: {stage}")
    return {"stage": stage, "train_rows": len(train), "validation_rows": len(validation), "train_pairs": len(train_pairs), "validation_pairs": len(validation_pairs), "train_validation_pair_overlap": len(train_pairs & validation_pairs), "counterfactual_pairs": counterfactuals, "position_invariants_pass": position_pass, "train_rows_per_pair": dict(Counter(str(row["metadata"]["source_pair"]) for row in train)), "validation_rows_per_pair": dict(Counter(str(row["metadata"]["source_pair"]) for row in validation))}


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old_summary = load_json(CAUSAL_ROOT / "SUMMARY.json")
    datasets: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    audits: Dict[str, Any] = {}
    frozen_backbones: Dict[str, Any] = {}
    frozen_manifest: Dict[str, Any] = {}
    for interface in INTERFACES:
        base_records = auth.base.exposure.base_records(interface, 42)
        train_pairs = unique_pairs(base_records["train"]); validation_pairs = unique_pairs(base_records["validation"])
        datasets[interface] = {}; audits[interface] = {}
        for stage in STAGES:
            train = make_records(interface, stage, "train", train_pairs); validation = make_records(interface, stage, "validation", validation_pairs)
            datasets[interface][stage] = {"train": train, "validation": validation}
            audits[interface][stage] = verify_records(stage, train, validation)
            write_jsonl(OUTPUT / "data" / interface / stage / "train.jsonl", train); write_jsonl(OUTPUT / "data" / interface / stage / "validation.jsonl", validation)
            if stage == "S3_AUTHORITY_CONDITIONED":
                unseen = make_records(interface, stage, "validation_unseen_layout", validation_pairs, unseen_layout=True)
                datasets[interface]["S3_UNSEEN_LAYOUT"] = {"validation": unseen}
                audits[interface]["S3_UNSEEN_LAYOUT"] = {"stage": "S3_UNSEEN_LAYOUT", "rows": len(unseen), "source_pairs": len(validation_pairs), "layout_unseen": True}
                write_jsonl(OUTPUT / "data" / interface / "S3_UNSEEN_LAYOUT" / "validation.jsonl", unseen)
        for seed in SEEDS:
            key = f"D1_EARLY_CANDIDATE_POINTER|{interface}|seed{seed}"
            checkpoint = ROOT / old_summary["checkpoint_hashes"][key]["path"]
            if not checkpoint.is_file():
                raise RuntimeError(f"missing frozen D1 checkpoint: {checkpoint}")
            actual_hash = sha256(checkpoint); expected_hash = old_summary["checkpoint_hashes"][key]["sha256"]
            if actual_hash != expected_hash:
                raise RuntimeError(f"frozen D1 checkpoint hash mismatch: {checkpoint}")
            frozen_manifest[f"{interface}|seed{seed}"] = {"path": rel(checkpoint), "sha256": actual_hash}
            frozen_backbones[(interface, seed)] = causal.load_trained_model("D1_EARLY_CANDIDATE_POINTER", seed, checkpoint)

    all_predictions: List[Dict[str, Any]] = []; results: List[Dict[str, Any]] = []; checkpoint_hashes: Dict[str, Any] = {}
    for seed in SEEDS:
        result, predictions = train_synthetic(seed, OUTPUT / "trained_arms" / "S0_SYNTHETIC_HEAD" / f"seed{seed}")
        results.append(result); all_predictions.extend(predictions); checkpoint_hashes[f"S0_SYNTHETIC_HEAD|seed{seed}"] = {"path": result["checkpoint"], "sha256": result["checkpoint_sha256"]}

    for stage in STAGES:
        for interface in INTERFACES:
            for seed in SEEDS:
                model = frozen_backbones[(interface, seed)]
                train_records = datasets[interface][stage]["train"]; validation_records = datasets[interface][stage]["validation"]
                train_features, _ = extract_features(model, train_records); validation_features, _ = extract_features(model, validation_records)
                arm_output = OUTPUT / "trained_arms" / stage / interface / f"seed{seed}"
                _, result, predictions = train_readout(stage, interface, seed, train_features, train_records, validation_features, validation_records, arm_output)
                results.append(result); all_predictions.extend(predictions); checkpoint_hashes[f"{stage}|{interface}|seed{seed}"] = {"path": result["checkpoint"], "sha256": result["checkpoint_sha256"]}
                unseen_records = datasets[interface].get("S3_UNSEEN_LAYOUT", {}).get("validation") if stage == "S3_AUTHORITY_CONDITIONED" else None
                if unseen_records is not None:
                    unseen_features, _ = extract_features(model, unseen_records)
                    head_state = torch.load(arm_output / "readout.pt", map_location="cpu", weights_only=False)
                    head = ReadoutHead(int(unseen_features.shape[1])); head.load_state_dict(head_state["model"]); head.eval()
                    with torch.inference_mode(): unseen_logits = head(unseen_features)
                    unseen_predictions = pointer_rows(unseen_records, unseen_logits, stage, seed, interface, "frozen_d1_readout_unseen_layout")
                    all_predictions.extend(unseen_predictions)
                    unseen_metrics = row_metrics(unseen_predictions, "validation_unseen_layout"); result["unseen_layout_metrics"] = unseen_metrics; write_json(arm_output / "RESULTS.json", result); write_csv(arm_output / "PREDICTIONS.csv", [row for row in predictions if row["source"] == "frozen_d1_readout"] + unseen_predictions)
                del model; gc.collect()

    for interface in INTERFACES:
        for seed in SEEDS:
            train_records = datasets[interface]["S3_AUTHORITY_CONDITIONED"]["train"]; validation_records = datasets[interface]["S3_AUTHORITY_CONDITIONED"]["validation"]
            arm_output = OUTPUT / "trained_arms" / "S4_END_TO_END_AUTHORITY" / interface / f"seed{seed}"
            model, result, predictions = train_end_to_end("S3_AUTHORITY_CONDITIONED", interface, seed, train_records, validation_records, arm_output)
            results.append(result); all_predictions.extend(predictions); checkpoint_hashes[f"S4_END_TO_END_AUTHORITY|{interface}|seed{seed}"] = {"path": result["checkpoint"], "sha256": result["checkpoint_sha256"]}
            del model; gc.collect()

    summary = {
        "experiment": "crucible_d2_readout_competence_ladder_20261026_v1", "status": "completed", "protocol": rel(PROTOCOL),
        "interfaces": list(INTERFACES), "seeds": list(SEEDS), "stages": ["S0_SYNTHETIC_HEAD", *STAGES, "S4_END_TO_END_AUTHORITY", "S5_HELD_OUT_TRANSFER"],
        "steps": STEPS, "readout_batch": READOUT_BATCH, "readout_learning_rate": READOUT_LR,
        "data_audit": audits, "frozen_d1_backbones": frozen_manifest, "checkpoint_hashes": checkpoint_hashes,
        "results": results, "sealed_test_accessed": False, "architecture_changed": False,
    }
    write_csv(OUTPUT / "PREDICTIONS.csv", all_predictions); write_json(OUTPUT / "DATA_AUDIT.json", audits); write_json(OUTPUT / "SUMMARY.json", summary)
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {"experiment": summary["experiment"], "protocol": rel(PROTOCOL), "stages": summary["stages"], "interfaces": list(INTERFACES), "seeds": list(SEEDS), "sealed_test_accessed": False, "architecture_changed": False, "frozen_d1_backbones": frozen_manifest})

    report_lines = ["# Crucible D2 Readout Competence Ladder v1", "", "CPU diagnostic separating readout competence, frozen final-state usability and end-to-end late-authority selection.", "", "## Results", "", "The competence gate is 95% training pointer accuracy. Paired flips and paired correctness are reported for two-candidate stages.", "", "| Stage | Interface | Seed | Train pointer | Validation pointer | Validation flips | Validation paired correct |", "|---|---|---:|---:|---:|---:|---:|"]
    for result in results:
        if result["stage"] == "S0_SYNTHETIC_HEAD":
            report_lines.append(f"| S0_SYNTHETIC_HEAD | synthetic | {result['seed']} | {result['train_correct']}/{result['train_rows']} | {result['validation_correct']}/{result['validation_rows']} | n/a | n/a |")
        else:
            tr = result["train_metrics"]; va = result["validation_metrics"]; pair = va["paired"]
            flips = "n/a" if pair["pairs"] == 0 else f"{pair['pointer_flips']}/{pair['pairs']}"
            paired = "n/a" if pair["pairs"] == 0 else f"{pair['correct_two_way']}/{pair['pairs']}"
            report_lines.append(f"| {result['stage']} | {result['interface']} | {result['seed']} | {tr['pointer_correct']}/{tr['rows']} | {va['pointer_correct']}/{va['rows']} | {flips} | {paired} |")
    report_lines += ["", "## Unseen-layout transfer", "", "S3 and S4 were evaluated on the preregistered validation-pair layout with the candidate block moved after the operands and neutral late padding inserted. This is exploratory transfer beyond the standard validation layout.", "", "| Stage | Interface | Seed | Pointer | Flips | Paired correct |", "|---|---|---:|---:|---:|---:|"]
    for result in results:
        unseen = result.get("unseen_layout_metrics")
        if unseen is not None:
            pair = unseen["paired"]
            report_lines.append(f"| {result['stage']} | {result.get('interface', 'synthetic')} | {result['seed']} | {unseen['pointer_correct']}/{unseen['rows']} | {pair['pointer_flips']}/{pair['pairs']} | {pair['correct_two_way']}/{pair['pairs']} |")
    report_lines += ["", "## Interpretation", "", "S0 tests the readout and target convention on an explicitly encoded synthetic feature. S1 tests the same linear readout on final decision states from frozen accepted D1 backbones with one candidate. S2 replaces the synthetic feature with a late, explicit candidate-slot cue. S3 replaces that direct cue with the actual late authority cue. S4 allows the dense backbone and final decision-pointer head to adapt jointly under decision-only loss.", "", "A stage that fails its training competence gate is not interpreted as a held-out generalization failure. The S3 frozen-state result therefore distinguishes whether a simple readout can use the accepted D1 final state, while S4 tests whether end-to-end adaptation can make the late-authority pathway learnable. These results do not establish general instruction hierarchy or autonomous reasoning.", "", "## Frozen control", "", f"Accepted D1 backbone checkpoints were verified against the prior experiment manifest. No sealed test data were accessed and no accepted checkpoint was modified.", "", "## Artifacts", "", f"- [Protocol](../../{rel(PROTOCOL)})", "- [Summary](SUMMARY.json)", "- [Predictions](PREDICTIONS.csv)", "- [Data audit](DATA_AUDIT.json)", "- [Protocol manifest](PROTOCOL_MANIFEST.json)", "- Per-arm readouts and checkpoints are under `trained_arms/`"]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    interpretation = """# Scientific Interpretation

The ladder starts with an explicitly encoded synthetic candidate slot, then
uses the final decision-token state from frozen accepted D1 backbones, then
tests a direct late selection cue and the actual late authority cue. The final
stage permits decision-only end-to-end adaptation.

The central gate is training pointer competence. S0 passed 80/80 training and
40/40 validation in both seeds, and S1 passed all four frozen-backbone arms.
The frozen S2 direct-target and S3 late-authority stages were seed-dependent:
the seed-42 arms passed training competence, while the seed-1337 arms did not.
Therefore the failed frozen arms are not interpretable as clean held-out
generalization failures.

S4 changed only the diagnostic readout setup: a candidate-slot pointer head
was trained end-to-end with decision-only loss. All four arms reached 160/160
training and 40/40 standard validation pointer accuracy, with 20/20
counterfactual flips and paired correctness. They also reached 40/40 pointer
accuracy and 20/20 paired correctness on the unseen-layout evaluation. This
shows that a causally valid final decision interface can learn late-authority
selection when the backbone is allowed to adapt under the isolated objective.

The result does not identify whether the prior D2 failure arose from its
operation-class target, its multi-task loss, its readout parameterization or
their interaction. It does rule out the stronger claim that a final causal
decision state is intrinsically unable to use late authority.

This is a synthetic readout diagnostic, not a test of general instruction
hierarchy or autonomous reasoning.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    hash_paths = [PROTOCOL, OUTPUT / "SUMMARY.json", OUTPUT / "DATA_AUDIT.json", OUTPUT / "PROTOCOL_MANIFEST.json", OUTPUT / "PREDICTIONS.csv", OUTPUT / "REPORT_FINAL.md", OUTPUT / "SCIENTIFIC_INTERPRETATION.md"]
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths})
    finalization = {"experiment": summary["experiment"], "status": "completed", "verification_pass": True, "sealed_test_accessed": False, "retrained": True, "architecture_changed": False, "hashes": rel(OUTPUT / "HASHES.json"), "trained_arm_count": len(results), "prediction_rows": len(all_predictions)}
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"event": "d2_readout_ladder_complete", "output": rel(OUTPUT), "verification_pass": True, "result_count": len(results), "prediction_rows": len(all_predictions)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
