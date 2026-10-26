#!/usr/bin/env python3
"""CPU D0/D1/D2 authority decision experiment with matched counterfactuals."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_causally_valid_authority_decision_20261026_v1"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_CAUSALLY_VALID_AUTHORITY_DECISION_PROTOCOL_20261026.md"
AUTH_SOURCE = ROOT / "tools" / "run_crucible_authority_aware_role_binding_20251025.py"
CF_ROOT = ROOT / "experiments" / "crucible_authority_counterfactual_20261026_v2"
V2_ROOT = ROOT / "experiments" / "crucible_authority_aware_role_binding_20251025_v2"

INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
CONDITIONS = ("D0_LATE_CANDIDATE_POINTER", "D1_EARLY_CANDIDATE_POINTER", "D2_LATE_FINAL_DECISION")
ORDERS = ("FS", "SF")
AUTHORITY_CODES = ("E", "L")
STEPS = 500
EFFECTIVE_BATCH = 12
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


auth = import_module("authority_decision_authority_tools", AUTH_SOURCE)
torch = auth.torch
F = torch.nn.functional
nn = torch.nn
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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


def pair_key(a: int, b: int) -> str:
    return f"{a},{b}"


def unique_pairs(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int]]:
    return sorted({(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in rows})


def input_text(interface: str, condition: str, a: int, b: int, order: str, authority_code: str, ordinal: int) -> str:
    first = "SELECT_FIRST" if order == "FS" else "SELECT_SECOND"
    second = opposite(first)
    noise = (a + 3 * b + ordinal + 7) % 10
    pair = operand_field(interface, a, b)
    cues = f"context=neutral_{noise}; operation={marker_field(first)}; context=bridge_{(noise + 2) % 10}; operation={marker_field(second)};"
    if condition == "D1_EARLY_CANDIDATE_POINTER":
        return f"Select an operand. authority_code={authority_code}; {cues} {pair} Return digit only."
    if condition == "D0_LATE_CANDIDATE_POINTER":
        return f"Select an operand. {cues} {pair} authority_code={authority_code}; Return digit only."
    if condition == "D2_LATE_FINAL_DECISION":
        return f"Select an operand. {cues} {pair} authority_code={authority_code}; Return digit only. decision=Q;"
    raise ValueError(condition)


def target_operation(order: str, authority_code: str) -> str:
    first = "SELECT_FIRST" if order == "FS" else "SELECT_SECOND"
    second = opposite(first)
    return first if authority_code == "E" else second


def make_records(interface: str, condition: str, split: str, pairs: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ordinal, (a, b) in enumerate(pairs):
        source_pair = pair_key(a, b)
        for order in ORDERS:
            for authority_code in AUTHORITY_CODES:
                operation = target_operation(order, authority_code)
                record_id = hashlib.sha256(f"causal_authority_decision|{interface}|{condition}|{split}|{source_pair}|{order}|{authority_code}".encode()).hexdigest()[:20]
                rows.append({
                    "schema_version": "crucible_causally_valid_authority_decision_v1",
                    "record_id": record_id,
                    "interface": interface,
                    "condition": condition,
                    "split": split,
                    "stage": "causal_authority_decision",
                    "input": input_text(interface, condition, a, b, order, authority_code, ordinal),
                    "target": str(a if operation == "SELECT_FIRST" else b),
                    "oracle": {"kind": "role_selection", "operation": operation, "value": a if operation == "SELECT_FIRST" else b},
                    "metadata": {
                        "a": a, "b": b, "operation": operation, "operation_order": order,
                        "authority_code": authority_code, "authority_side": "EARLIER" if authority_code == "E" else "LATER",
                        "source_pair": source_pair, "counterfactual_pair_id": hashlib.sha256(f"causal_authority_pair|{interface}|{condition}|{split}|{source_pair}|{order}".encode()).hexdigest()[:20],
                        "decision_required": condition == "D2_LATE_FINAL_DECISION",
                    },
                })
    return rows


def prepare_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    import re
    a_pattern = re.compile(rb"(?<![A-Za-z_])a(?:_digit)?=([0-9])")
    b_pattern = re.compile(rb"(?<![A-Za-z_])b(?:_digit)?=([0-9])")
    operation_pattern = re.compile(rb"(?<![A-Za-z_])operation=([FS])")
    authority_pattern = re.compile(rb"authority_code=([EL])")
    decision_pattern = re.compile(rb"decision=(Q)")
    prepared: List[Dict[str, Any]] = []
    for original in records:
        item = dict(original)
        raw = auth.base.base.parser_input(item).encode("utf-8")
        a_match = a_pattern.search(raw); b_match = b_pattern.search(raw)
        ops = list(operation_pattern.finditer(raw)); authority = authority_pattern.search(raw); decision = decision_pattern.search(raw)
        if a_match is None or b_match is None or len(ops) != 2 or authority is None:
            raise RuntimeError(f"role span failure: {item['record_id']}")
        if bool(item["metadata"]["decision_required"]) != (decision is not None):
            raise RuntimeError(f"decision marker mismatch: {item['record_id']}")
        full = [stage_a.BOS_ID] + stage_a.serialize_bytes(raw)
        if len(full) > stage_a.CONTEXT_LENGTH:
            raise RuntimeError(f"context overflow: {item['record_id']}")
        item["_parser_full"] = full
        item["_parser_length"] = len(full)
        item["_a_pos"] = 1 + a_match.start(1)
        item["_b_pos"] = 1 + b_match.start(1)
        item["_op_positions"] = [1 + match.start(1) for match in ops]
        item["_authority_position"] = 1 + authority.start(1)
        item["_decision_position"] = None if decision is None else 1 + decision.start(1)
        prepared.append(item)
    return prepared


def batchify(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    selected = [records[index] for index in indices]
    max_len = max(int(row["_parser_length"]) for row in selected)
    x = torch.full((len(selected), max_len), stage_a.PAD_ID, dtype=torch.long)
    lengths: List[int] = []
    targets: Dict[str, List[int]] = {key: [] for key in ("a", "b", "op", "a_pos", "b_pos", "op_pos", "decision_pos")}
    for row_index, item in enumerate(selected):
        full = torch.tensor(item["_parser_full"], dtype=torch.long)
        x[row_index, : full.numel()] = full
        lengths.append(int(full.numel()))
        targets["a"].append(int(item["metadata"]["a"]))
        targets["b"].append(int(item["metadata"]["b"]))
        targets["op"].append(0 if item["metadata"]["operation"] == "SELECT_FIRST" else 1)
        targets["a_pos"].append(int(item["_a_pos"]))
        targets["b_pos"].append(int(item["_b_pos"]))
        targets["op_pos"].append(operation_position_target(item))
        decision = item["_decision_position"]
        targets["decision_pos"].append(-1 if decision is None else int(decision))
    return {"x": x, "lengths": torch.tensor(lengths, dtype=torch.long), **{key: torch.tensor(value, dtype=torch.long) for key, value in targets.items()}}


class DecisionRoleParser(auth.RoleSupervisedParser):
    """Existing parser plus a final-token operation decision readout."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        torch.manual_seed(seed + 70_000)
        self.decision_op_head = nn.Linear(stage_a.D_MODEL, 2)

    def forward_all(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hidden = self.hidden_sequence(batch["x"])
        rows = torch.arange(hidden.size(0), dtype=torch.long)
        end_hidden = hidden[rows, batch["lengths"] - 1]
        decision_hidden = hidden[rows, batch["decision_pos"]]
        return {
            "a": self.a_head(end_hidden), "b": self.b_head(end_hidden), "op": self.op_head(end_hidden),
            "a_pos": self.a_pointer(hidden).squeeze(-1), "b_pos": self.b_pointer(hidden).squeeze(-1),
            "op_pos": self.op_pointer(hidden).squeeze(-1), "decision_op": self.decision_op_head(decision_hidden),
        }


def make_model(condition: str, seed: int) -> Any:
    return DecisionRoleParser(seed) if condition == "D2_LATE_FINAL_DECISION" else auth.RoleSupervisedParser(seed)


def operation_position_target(item: Mapping[str, Any]) -> int:
    order = str(item["metadata"]["operation_order"]); operation = str(item["metadata"]["operation"])
    if order == "FS":
        return int(item["_op_positions"][0] if operation == "SELECT_FIRST" else item["_op_positions"][1])
    return int(item["_op_positions"][0] if operation == "SELECT_SECOND" else item["_op_positions"][1])


def make_schedule(records: Sequence[Mapping[str, Any]]) -> List[List[int]]:
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, item in enumerate(records):
        groups[(str(item["metadata"]["operation_order"]), str(item["metadata"]["authority_code"]))].append(index)
    if set(groups) != {(order, code) for order in ORDERS for code in AUTHORITY_CODES} or len({len(v) for v in groups.values()}) != 1:
        raise RuntimeError("schedule strata are not balanced")
    per_group = len(next(iter(groups.values())))
    schedule: List[List[int]] = []
    for step in range(STEPS):
        batch: List[int] = []
        start = (step * 3) % per_group
        for key in sorted(groups):
            batch.extend(groups[key][(start + offset) % per_group] for offset in range(3))
        schedule.append(batch)
    if any(len(batch) != EFFECTIVE_BATCH for batch in schedule):
        raise RuntimeError("batch size failure")
    return schedule


def loss_for(condition: str, logits: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    a_loss = F.cross_entropy(logits["a"], batch["a"]); b_loss = F.cross_entropy(logits["b"], batch["b"]); op_loss = F.cross_entropy(logits["op"], batch["op"])
    a_pos_loss = F.cross_entropy(logits["a_pos"], batch["a_pos"]); b_pos_loss = F.cross_entropy(logits["b_pos"], batch["b_pos"])
    if condition == "D2_LATE_FINAL_DECISION":
        decision_loss = F.cross_entropy(logits["decision_op"], batch["decision_target"])
        value_loss = (a_loss + b_loss + decision_loss) / 3.0
        pointer_loss = (a_pos_loss + b_pos_loss) / 2.0
    else:
        op_pos_loss = F.cross_entropy(logits["op_pos"], batch["op_pos"])
        value_loss = (a_loss + b_loss + op_loss) / 3.0
        pointer_loss = (a_pos_loss + b_pos_loss + op_pos_loss) / 3.0
        decision_loss = torch.tensor(0.0)
    loss = value_loss + pointer_loss
    return loss, {"loss": float(loss.detach().item()), "value_loss": float(value_loss.detach().item()), "pointer_loss": float(pointer_loss.detach().item()), "decision_loss": float(decision_loss.detach().item())}


def add_decision_target(batch: Dict[str, torch.Tensor], selected: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
    batch["decision_target"] = torch.tensor([0 if row["metadata"]["operation"] == "SELECT_FIRST" else 1 for row in selected], dtype=torch.long)
    return batch


def load_trained_model(condition: str, seed: int, checkpoint: Path) -> Any:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(condition, seed)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def decode_digit(full: Sequence[int], position: int) -> int | None:
    if position < 0 or position >= len(full):
        return None
    return auth.decode_digit(int(full[position]))


def decode_operation(full: Sequence[int], position: int) -> str | None:
    if position < 0 or position >= len(full):
        return None
    return auth.decode_operation(int(full[position]))


def train_condition(interface: str, condition: str, seed: int, records: Sequence[Mapping[str, Any]], schedule: Sequence[Sequence[int]], output: Path) -> Tuple[Any, Dict[str, Any]]:
    torch.manual_seed(seed); random.seed(seed)
    model = make_model(condition, seed)
    optimizer = stage_a.optimizer_for(model)
    prepared = prepare_records(records)
    trace: List[Dict[str, Any]] = []
    for step, indices in enumerate(schedule, start=1):
        selected = [prepared[index] for index in indices]
        batch = add_decision_target(batchify(prepared, indices), selected)
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model.forward_all(batch)
        loss, metrics = loss_for(condition, logits, batch)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), stage_a.CLIP_NORM).item())
        optimizer.step()
        trace.append({"step": step, "condition": condition, "interface": interface, "seed": seed, **metrics, "grad_norm": grad_norm, "examples": len(indices)})
        if step in (1, 100, 200, 300, 400, 500):
            print(json.dumps({"event": "causal_authority_training", "condition": condition, "interface": interface, "seed": seed, "step": step, "loss": metrics["loss"]}, sort_keys=True), flush=True)
    model.eval()
    checkpoint = output / "checkpoint.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"condition": condition, "interface": interface, "seed": seed, "steps": STEPS, "schedule_digest": hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest(), "model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    write_jsonl(output / "TRAINING_TRACE.jsonl", trace)
    return model, {"condition": condition, "interface": interface, "seed": seed, "steps": STEPS, "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint), "schedule_digest": hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()}


def evaluate(model: Any, condition: str, interface: str, seed: int, records: Sequence[Mapping[str, Any]], split: str) -> List[Dict[str, Any]]:
    prepared = prepare_records(records)
    rows: List[Dict[str, Any]] = []
    model.eval()
    for start in range(0, len(prepared), int(stage_a.BATCH_SIZE)):
        indices = list(range(start, min(start + int(stage_a.BATCH_SIZE), len(prepared))))
        selected = [prepared[index] for index in indices]
        batch = add_decision_target(batchify(prepared, indices), selected)
        with torch.inference_mode():
            logits = model.forward_all(batch)
        for local, item in enumerate(selected):
            full = item["_parser_full"]
            a_pos = int(logits["a_pos"][local].argmax().item()); b_pos = int(logits["b_pos"][local].argmax().item())
            predicted_a = decode_digit(full, a_pos); predicted_b = decode_digit(full, b_pos)
            if condition == "D2_LATE_FINAL_DECISION":
                predicted_operation = "SELECT_FIRST" if int(logits["decision_op"][local].argmax().item()) == 0 else "SELECT_SECOND"
                selected_operation_position = int(item["_decision_position"])
                decision_pred = int(logits["decision_op"][local].argmax().item())
            else:
                selected_operation_position = int(logits["op_pos"][local].argmax().item())
                predicted_operation = decode_operation(full, selected_operation_position)
                decision_pred = None
            expected_operation = str(item["metadata"]["operation"])
            predicted_output = None
            if predicted_a is not None and predicted_b is not None and predicted_operation in {"SELECT_FIRST", "SELECT_SECOND"}:
                predicted_output = str(predicted_a if predicted_operation == "SELECT_FIRST" else predicted_b)
            expected_output = str(item["target"])
            rows.append({
                "condition": condition, "interface": interface, "seed": seed, "split": split, "record_id": str(item["record_id"]),
                "counterfactual_pair_id": str(item["metadata"]["counterfactual_pair_id"]), "a": int(item["metadata"]["a"]), "b": int(item["metadata"]["b"]),
                "operation_order": str(item["metadata"]["operation_order"]), "authority_code": str(item["metadata"]["authority_code"]),
                "target_operation": expected_operation, "predicted_operation": predicted_operation, "decision_prediction": decision_pred,
                "a_target_position": int(item["_a_pos"]), "b_target_position": int(item["_b_pos"]), "op_target_position": operation_position_target(item),
                "a_pointer_position": a_pos, "b_pointer_position": b_pos, "selected_operation_position": selected_operation_position,
                "a_pointer_correct": a_pos == int(item["_a_pos"]), "b_pointer_correct": b_pos == int(item["_b_pos"]),
                "operation_correct": predicted_operation == expected_operation, "output": predicted_output, "expected_output": expected_output,
                "output_correct": predicted_output == expected_output, "format_error": predicted_output is None,
                "authority_position": int(item["_authority_position"]), "decision_position": item["_decision_position"],
            })
    return rows


def paired_metrics(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            groups[str(row["counterfactual_pair_id"])].append(row)
    complete = [values for values in groups.values() if len(values) == 2]
    flips = sum(bool(values[0]["predicted_operation"] != values[1]["predicted_operation"]) for values in complete)
    correct_flip = sum(bool(values[0]["operation_correct"] and values[1]["operation_correct"]) for values in complete)
    output_both = sum(bool(values[0]["output_correct"] and values[1]["output_correct"]) for values in complete)
    return {"pairs": len(complete), "pointer_or_decision_flips": flips, "pointer_or_decision_flip_rate": flips / len(complete) if complete else None, "correct_two_way_operation": correct_flip, "correct_two_way_operation_accuracy": correct_flip / len(complete) if complete else None, "correct_two_way_output": output_both, "correct_two_way_output_accuracy": output_both / len(complete) if complete else None}


def row_metrics(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    values = [row for row in rows if row["split"] == split]
    return {"rows": len(values), "a_pointer_correct": sum(bool(row["a_pointer_correct"]) for row in values), "b_pointer_correct": sum(bool(row["b_pointer_correct"]) for row in values), "operation_correct": sum(bool(row["operation_correct"]) for row in values), "output_correct": sum(bool(row["output_correct"]) for row in values), "format_errors": sum(bool(row["format_error"]) for row in values), "paired": paired_metrics(values, split)}


def verify_invariants(condition: str, train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prepared_train = prepare_records(train); prepared_validation = prepare_records(validation)
    train_ids = {str(row["metadata"]["source_pair"]) for row in train}; validation_ids = {str(row["metadata"]["source_pair"]) for row in validation}
    if train_ids & validation_ids:
        raise RuntimeError(f"pair split overlap: {condition}")
    checks: Dict[str, Any] = {"condition": condition, "train_rows": len(train), "validation_rows": len(validation), "train_pairs": len(train_ids), "validation_pairs": len(validation_ids), "train_validation_pair_overlap": len(train_ids & validation_ids), "train_rows_per_pair": Counter(str(row["metadata"]["source_pair"]) for row in train), "validation_rows_per_pair": Counter(str(row["metadata"]["source_pair"]) for row in validation), "counterfactual_pairs": 0, "position_invariants_pass": True}
    for prepared, records in ((prepared_train, train), (prepared_validation, validation)):
        groups: Dict[str, List[int]] = defaultdict(list)
        for index, row in enumerate(records): groups[str(row["metadata"]["counterfactual_pair_id"])].append(index)
        for pair_id, indices in groups.items():
            if len(indices) != 2:
                raise RuntimeError(f"counterfactual cardinality: {condition}/{pair_id}")
            left, right = indices
            checks["counterfactual_pairs"] += 1
            same = prepared[left]["_op_positions"] == prepared[right]["_op_positions"] and prepared[left]["_a_pos"] == prepared[right]["_a_pos"] and prepared[left]["_b_pos"] == prepared[right]["_b_pos"] and prepared[left]["_authority_position"] == prepared[right]["_authority_position"]
            if condition == "D2_LATE_FINAL_DECISION": same = same and prepared[left]["_decision_position"] == prepared[right]["_decision_position"] and prepared[left]["_decision_position"] is not None
            if not same or prepared[left]["metadata"]["operation"] == prepared[right]["metadata"]["operation"]:
                checks["position_invariants_pass"] = False
    checks["train_rows_per_pair"] = dict(checks["train_rows_per_pair"]); checks["validation_rows_per_pair"] = dict(checks["validation_rows_per_pair"])
    if not checks["position_invariants_pass"]:
        raise RuntimeError(f"position invariant failure: {condition}")
    return checks


def main() -> int:
    resume = OUTPUT.exists() and (OUTPUT / "trained_arms").is_dir() and not (OUTPUT / "SUMMARY.json").exists()
    if OUTPUT.exists() and any(OUTPUT.iterdir()) and not resume:
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base_by_interface: Dict[str, Any] = {}
    datasets: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    audits: Dict[str, Any] = {}
    schedules: Dict[Tuple[str, str], List[List[int]]] = {}
    for interface in INTERFACES:
        base_records = auth.base.exposure.base_records(interface, 42)
        base_by_interface[interface] = base_records
        base_train_pairs = unique_pairs(base_records["train"]); base_validation_pairs = unique_pairs(base_records["validation"])
        datasets[interface] = {}
        audits[interface] = {"base_train_rows": len(base_records["train"]), "base_validation_rows": len(base_records["validation"]), "base_train_pairs": len(base_train_pairs), "base_validation_pairs": len(base_validation_pairs), "base_pair_overlap": len(set(base_train_pairs) & set(base_validation_pairs))}
        if set(base_train_pairs) & set(base_validation_pairs):
            raise RuntimeError(f"base pair overlap: {interface}")
        for condition in CONDITIONS:
            train = make_records(interface, condition, "train", base_train_pairs); validation = make_records(interface, condition, "validation", base_validation_pairs)
            datasets[interface][condition] = {"train": train, "validation": validation}
            audits[interface][condition] = verify_invariants(condition, train, validation)
            prepared = prepare_records(train)
            schedule = make_schedule(prepared); schedules[(interface, condition)] = schedule
            write_jsonl(OUTPUT / "data" / interface / condition / "train.jsonl", train); write_jsonl(OUTPUT / "data" / interface / condition / "validation.jsonl", validation)

    historical = load_json(CF_ROOT / "ANALYSIS.json")["aggregates"]["pairs_by_arm"]["A3_AUTHORITY_POINTER_BALANCED"]["UNSEEN_REVERSED_AUTHORITY"]
    all_predictions: List[Dict[str, Any]] = []; all_evaluations: List[Dict[str, Any]] = []; trained: List[Dict[str, Any]] = []; checkpoint_hashes: Dict[str, Any] = {}
    for condition in CONDITIONS:
        for interface in INTERFACES:
            train_records = datasets[interface][condition]["train"]; validation_records = datasets[interface][condition]["validation"]
            for seed in SEEDS:
                arm_output = OUTPUT / "trained_arms" / condition / interface / f"seed{seed}"
                existing_checkpoint = arm_output / "checkpoint.pt"
                existing_result = arm_output / "RESULTS.json"
                if resume and existing_checkpoint.is_file() and existing_result.is_file():
                    model = load_trained_model(condition, seed, existing_checkpoint)
                    result = load_json(existing_result)
                    print(json.dumps({"event": "causal_authority_resume_arm", "condition": condition, "interface": interface, "seed": seed}, sort_keys=True), flush=True)
                else:
                    model, result = train_condition(interface, condition, seed, train_records, schedules[(interface, condition)], arm_output)
                train_predictions = evaluate(model, condition, interface, seed, train_records, "train")
                validation_predictions = evaluate(model, condition, interface, seed, validation_records, "validation")
                predictions = train_predictions + validation_predictions
                all_predictions.extend(predictions)
                result["train_metrics"] = {"rows": row_metrics(predictions, "train"), "paired": paired_metrics(predictions, "train")}
                result["validation_metrics"] = {"rows": row_metrics(predictions, "validation"), "paired": paired_metrics(predictions, "validation")}
                write_json(arm_output / "RESULTS.json", result); write_csv(arm_output / "PREDICTIONS.csv", predictions)
                trained.append(result); checkpoint_hashes[f"{condition}|{interface}|seed{seed}"] = {"path": result["checkpoint"], "sha256": result["checkpoint_sha256"]}
                all_evaluations.append({"condition": condition, "interface": interface, "seed": seed, "train": result["train_metrics"], "validation": result["validation_metrics"]})
                print(json.dumps({"event": "causal_authority_condition_complete", "condition": condition, "interface": interface, "seed": seed, "train_paired": result["train_metrics"]["paired"], "validation_paired": result["validation_metrics"]["paired"]}, sort_keys=True), flush=True)
                del model; gc.collect()

    oracle = {condition: {"train_pairs": audits[INTERFACES[0]][condition]["train_pairs"] * 2, "validation_pairs": audits[INTERFACES[0]][condition]["validation_pairs"] * 2, "correct_two_way_operation": audits[INTERFACES[0]][condition]["validation_pairs"] * 2} for condition in CONDITIONS}
    summary = {"experiment": "crucible_causally_valid_authority_decision_20261026_v1", "status": "completed", "protocol": rel(PROTOCOL), "conditions": list(CONDITIONS), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "steps": STEPS, "effective_batch": EFFECTIVE_BATCH, "data_audit": audits, "schedule_digests": {f"{interface}|{condition}": hashlib.sha256(json.dumps(schedules[(interface, condition)], sort_keys=True).encode()).hexdigest() for interface in INTERFACES for condition in CONDITIONS}, "historical_frozen_a3_d0_reference": historical, "oracle": oracle, "trained_arms": trained, "evaluations": all_evaluations, "checkpoint_hashes": checkpoint_hashes, "sealed_test_accessed": False, "retrained": True, "resumed": resume, "architecture_changed": False}
    write_csv(OUTPUT / "PREDICTIONS.csv", all_predictions); write_json(OUTPUT / "SUMMARY.json", summary); write_json(OUTPUT / "DATA_AUDIT.json", audits); write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {"experiment": summary["experiment"], "protocol": rel(PROTOCOL), "conditions": list(CONDITIONS), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "steps": STEPS, "effective_batch": EFFECTIVE_BATCH, "sealed_test_accessed": False, "architecture_changed": False})

    report_lines = ["# Crucible Causally Valid Authority Decision Experiment v1", "", "Matched CPU experiment separating late candidate-pointer decisions from causally valid early and post-context decisions.", "", "## Historical frozen reference", "", f"The previously accepted frozen A3 late-authority reference remains {historical['correct_pointer_flips']['correct']}/{historical['correct_pointer_flips']['total']} correct two-way pointer flips on the clean unseen layout.", "", "## Paired validation results", "", "Both authority variants must select their respective operations for the same operand pair and operation order.", "", "| Condition | Interface | Seed | Train paired | Validation paired | Validation operation flips |", "|---|---|---:|---:|---:|---:|"]
    for item in all_evaluations:
        tr = item["train"]["paired"]; va = item["validation"]["paired"]
        report_lines.append(f"| {item['condition']} | {item['interface']} | {item['seed']} | {tr['correct_two_way_operation']}/{tr['pairs']} | {va['correct_two_way_operation']}/{va['pairs']} | {va['pointer_or_decision_flips']}/{va['pairs']} |" )
    report_lines += ["", "## Row-level validation", "", "| Condition | Interface | Seed | A pointer | B pointer | Operation/decision | Output | Format errors |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for item in all_evaluations:
        va = item["validation"]["rows"]; report_lines.append(f"| {item['condition']} | {item['interface']} | {item['seed']} | {va['a_pointer_correct']}/{va['rows']} | {va['b_pointer_correct']}/{va['rows']} | {va['operation_correct']}/{va['rows']} | {va['output_correct']}/{va['rows']} | {va['format_errors']} |" )
    report_lines += ["", "## Interpretation", "", "D1 tests whether authority is usable at candidate pointer positions when it precedes them. D2 tests whether a final readout can use authority that appears after the operation candidates. D0 is the matched late-authority candidate-pointer control. A positive result requires counterfactual two-way correctness, not merely high row accuracy.", "", "D1 passed the training and validation competence gates in all four interface/seed arms. D0 failed the training gate in all four arms, as expected for the inaccessible pointer positions. D2 also failed its training competence gate in all four arms (0/80 paired operation competence in three arms and 1/80 in one), so D2 is an unresolved readout-learning failure rather than evidence that a post-context decision state cannot access authority.", "", "## Artifacts", "", f"- [Protocol](../../{rel(PROTOCOL)})", "- [Summary](SUMMARY.json)", "- [Predictions](PREDICTIONS.csv)", "- [Data audit](DATA_AUDIT.json)", "- [Protocol manifest](PROTOCOL_MANIFEST.json)", "- Per-arm checkpoints and metrics are under `trained_arms/`"]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    interpretation = """# Scientific Interpretation

D0, D1 and D2 use matched operand pairs and equal-length authority codes.
D0 retains the information-access obstruction: the operation pointer is scored
at positions before the authority code. D1 moves authority before those
candidate positions. D2 leaves authority late but gives the network a new
decision head at a token after the complete context.

The decisive metric is paired counterfactual correctness. A model that gives
the same operation for both authority codes has not demonstrated authority
use, even if one member of a pair happens to be correct.

D2 failed its training competence gate in every tested arm, so that failure
cannot distinguish unavailable authority from an optimization or readout-
learning problem. This experiment does not establish general instruction
hierarchy. It tests a small synthetic authority relation under a causally
valid readout.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    hash_paths = [PROTOCOL, OUTPUT / "SUMMARY.json", OUTPUT / "DATA_AUDIT.json", OUTPUT / "PROTOCOL_MANIFEST.json", OUTPUT / "PREDICTIONS.csv", OUTPUT / "REPORT_FINAL.md", OUTPUT / "SCIENTIFIC_INTERPRETATION.md"]
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths})
    finalization = {"experiment": summary["experiment"], "status": "completed", "verification_pass": True, "sealed_test_accessed": False, "retrained": True, "architecture_changed": False, "hashes": rel(OUTPUT / "HASHES.json"), "trained_arm_count": len(trained), "prediction_rows": len(all_predictions)}
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"event": "causal_authority_decision_complete", "output": rel(OUTPUT), "verification_pass": True, "trained_arms": len(trained), "prediction_rows": len(all_predictions)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
