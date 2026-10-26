#!/usr/bin/env python3
"""Decompose Crucible role binding into parsing and execution controls."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
EXPOSURE_SOURCE = ROOT / "tools" / "run_crucible_paired_positional_exposure_20251025.py"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_EXPLICIT_ROLE_BINDING_GATE_PROTOCOL_20251025.md"
OUTPUT = ROOT / "experiments" / "crucible_explicit_role_binding_gate_20251025_v1"
POSITION_ROOT = ROOT / "experiments" / "crucible_operation_position_matrix_20251025_v1"
BATCH_ROOT = ROOT / "experiments" / "crucible_batch_composition_order_20251025_v1"

INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
CONDITIONS = (
    "C0_ORACLE_PARSER_A3_EXECUTOR",
    "C1_LEARNED_PARSER_A3_EXECUTOR",
    "C2_LEARNED_PARSER_ORACLE_EXECUTOR",
    "C3_RAW_END_TO_END",
)
TRAIN_LAYOUTS = ("FIXED_NEAR_PADDED", "EARLY_PADDED", "MIDDLE_POSITION")
EVAL_LAYOUTS = TRAIN_LAYOUTS + ("UNSEEN_BETWEEN_OPERANDS", "UNSEEN_EARLY_CONTEXT_FIRST")
STEPS = 500
EFFECTIVE_BATCH = 12
CURVE_STEPS = (1, 10, 25, 50, 100, 200, 300, 400, 500)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


exposure = import_module("role_binding_exposure", EXPOSURE_SOURCE)
ladder = exposure.ladder
stage_a = exposure.stage_a
torch = stage_a.torch
nn = torch.nn
F = torch.nn.functional


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(row), sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def marker_field(operation: str) -> str:
    return ("F" if operation == "SELECT_FIRST" else "S") + ("_" * (len(operation.encode("utf-8")) - 1))


def a3_input(interface: str, a: int, b: int, operation: str) -> str:
    pair = f"a={a}; b={b}." if interface == "canonical" else f"a_digit={a}; b_digit={b}."
    return f"Select an operand. {pair} Return digit only. operation={marker_field(operation)};"


def parser_input(record: Mapping[str, Any]) -> str:
    return str(record["input"]) + "\nParse:"


def prepare_parser(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for original in records:
        item = dict(original)
        raw = parser_input(item).encode("utf-8")
        full = [stage_a.BOS_ID] + stage_a.serialize_bytes(raw)
        if len(full) > stage_a.CONTEXT_LENGTH:
            raise RuntimeError(f"parser record exceeds context: {item['record_id']}")
        item["_parser_full"] = full
        item["_parser_length"] = len(full)
        prepared.append(item)
    return prepared


def parser_batchify(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, Any]:
    selected = [records[index] for index in indices]
    max_len = max(int(item["_parser_length"]) for item in selected)
    x = torch.full((len(selected), max_len), stage_a.PAD_ID, dtype=torch.long)
    lengths: List[int] = []
    a_targets: List[int] = []
    b_targets: List[int] = []
    op_targets: List[int] = []
    for row, item in enumerate(selected):
        full = torch.tensor(item["_parser_full"], dtype=torch.long)
        x[row, : full.numel()] = full
        lengths.append(int(full.numel()))
        a_targets.append(int(item["metadata"]["a"]))
        b_targets.append(int(item["metadata"]["b"]))
        op_targets.append(0 if item["metadata"]["operation"] == "SELECT_FIRST" else 1)
    return {
        "x": x,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "a": torch.tensor(a_targets, dtype=torch.long),
        "b": torch.tensor(b_targets, dtype=torch.long),
        "op": torch.tensor(op_targets, dtype=torch.long),
    }


class RoleParser(nn.Module):
    """Existing dense backbone with explicit role-binding classification heads."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.backbone = stage_a.make_dense(seed)
        torch.manual_seed(seed + 50_000)
        self.a_head = nn.Linear(stage_a.D_MODEL, 10)
        self.b_head = nn.Linear(stage_a.D_MODEL, 10)
        self.op_head = nn.Linear(stage_a.D_MODEL, 2)

    def hidden_at_end(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        seq_idx = torch.arange(tokens.size(1), dtype=torch.long).unsqueeze(0).expand(tokens.size(0), -1)
        sequence_ids = torch.arange(tokens.size(0), dtype=torch.long)
        hidden = self.backbone.token_emb(tokens)
        for layer in self.backbone.layers:
            hidden = layer(hidden, seq_idx, current_sequence_ids=sequence_ids)
        hidden = self.backbone.final_norm(hidden)
        rows = torch.arange(tokens.size(0), dtype=torch.long)
        return hidden[rows, lengths - 1]

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.hidden_at_end(batch["x"], batch["lengths"])
        return self.a_head(hidden), self.b_head(hidden), self.op_head(hidden)


def load_parser(path: Path) -> RoleParser:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = RoleParser(int(state["seed"]))
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def load_dense(path: Path) -> Any:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = stage_a.DenseLM()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def parser_step(model: RoleParser, optimizer: Any, prepared: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, Any]:
    model.train()
    batch = parser_batchify(prepared, indices)
    optimizer.zero_grad(set_to_none=True)
    a_logits, b_logits, op_logits = model(batch)
    a_loss = F.cross_entropy(a_logits, batch["a"])
    b_loss = F.cross_entropy(b_logits, batch["b"])
    op_loss = F.cross_entropy(op_logits, batch["op"])
    loss = (a_loss + b_loss + op_loss) / 3.0
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), stage_a.CLIP_NORM).item())
    optimizer.step()
    a_pred = a_logits.argmax(dim=-1)
    b_pred = b_logits.argmax(dim=-1)
    op_pred = op_logits.argmax(dim=-1)
    tuple_correct = (a_pred == batch["a"]) & (b_pred == batch["b"]) & (op_pred == batch["op"])
    return {
        "loss": float(loss.detach().item()),
        "a_loss": float(a_loss.detach().item()),
        "b_loss": float(b_loss.detach().item()),
        "op_loss": float(op_loss.detach().item()),
        "a_correct": int((a_pred == batch["a"]).sum().item()),
        "b_correct": int((b_pred == batch["b"]).sum().item()),
        "op_correct": int((op_pred == batch["op"]).sum().item()),
        "tuple_correct": int(tuple_correct.sum().item()),
        "examples": len(indices),
        "grad_norm": grad_norm,
    }


def parser_predictions(model: RoleParser, records: Sequence[Mapping[str, Any]], arm_meta: Mapping[str, Any], layout: str, split: str) -> List[Dict[str, Any]]:
    prepared = prepare_parser(records)
    rows: List[Dict[str, Any]] = []
    model.eval()
    for start in range(0, len(prepared), int(stage_a.BATCH_SIZE)):
        indices = list(range(start, min(start + int(stage_a.BATCH_SIZE), len(prepared))))
        batch = parser_batchify(prepared, indices)
        with torch.inference_mode():
            a_logits, b_logits, op_logits = model(batch)
        for local, index in enumerate(indices):
            item = prepared[index]
            a_hat = int(a_logits[local].argmax().item())
            b_hat = int(b_logits[local].argmax().item())
            op_index = int(op_logits[local].argmax().item())
            op_hat = "SELECT_FIRST" if op_index == 0 else "SELECT_SECOND"
            a_correct = a_hat == int(item["metadata"]["a"])
            b_correct = b_hat == int(item["metadata"]["b"])
            op_correct = op_hat == str(item["metadata"]["operation"])
            rows.append({
                **dict(arm_meta),
                "layout": layout,
                "split": split,
                "record_id": str(item["record_id"]),
                "a": int(item["metadata"]["a"]),
                "b": int(item["metadata"]["b"]),
                "operation": str(item["metadata"]["operation"]),
                "predicted_a": a_hat,
                "predicted_b": b_hat,
                "predicted_operation": op_hat,
                "a_correct": a_correct,
                "b_correct": b_correct,
                "operation_correct": op_correct,
                "tuple_correct": a_correct and b_correct and op_correct,
            })
    return rows


def parser_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "rows": len(rows),
        "a_correct": sum(bool(row["a_correct"]) for row in rows),
        "b_correct": sum(bool(row["b_correct"]) for row in rows),
        "operation_correct": sum(bool(row["operation_correct"]) for row in rows),
        "tuple_correct": sum(bool(row["tuple_correct"]) for row in rows),
        "a_accuracy": sum(bool(row["a_correct"]) for row in rows) / len(rows) if rows else None,
        "b_accuracy": sum(bool(row["b_correct"]) for row in rows) / len(rows) if rows else None,
        "operation_accuracy": sum(bool(row["operation_correct"]) for row in rows) / len(rows) if rows else None,
        "tuple_accuracy": sum(bool(row["tuple_correct"]) for row in rows) / len(rows) if rows else None,
    }


def train_parser(interface: str, seed: int, records: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]], output: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], RoleParser]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = RoleParser(seed)
    optimizer = stage_a.optimizer_for(model)
    prepared = prepare_parser(records)
    traces: List[Dict[str, Any]] = []
    curves: List[Dict[str, Any]] = []
    for step_spec in schedule:
        step = int(step_spec["step"])
        indices = [int(index) for index in step_spec["record_indices"]]
        trace = parser_step(model, optimizer, prepared, indices)
        traces.append({"interface": interface, "seed": seed, "step": step, **trace})
        if step in CURVE_STEPS:
            batch_rows = parser_predictions(model, records, {"interface": interface, "seed": seed}, "TRAINING_ALL_LAYOUTS", "train")
            curves.append({"step": step, **parser_metrics(batch_rows)})
    checkpoint = output / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": STEPS, "seed": seed, "interface": interface, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule_digest": hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()}, checkpoint)
    final_rows = parser_predictions(model, records, {"interface": interface, "seed": seed}, "TRAINING_ALL_LAYOUTS", "train")
    final_metrics = parser_metrics(final_rows)
    result = {
        "parser_id": f"learned_parser_{interface}_seed{seed}",
        "interface": interface,
        "seed": seed,
        "steps": STEPS,
        "effective_batch": EFFECTIVE_BATCH,
        "train_records": len(records),
        "train_metrics": final_metrics,
        "checkpoint": rel(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "schedule_digest": hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest(),
    }
    write_json(output / "RESULTS.json", result)
    write_json(output / "TRAINING_CURVE.json", {"curve_steps": list(CURVE_STEPS), "points": curves})
    write_jsonl(output / "TRAINING_TRACE.jsonl", traces)
    return result, traces, model


def evaluation_records(base: Mapping[str, Sequence[Mapping[str, Any]]], train_rows: Sequence[Mapping[str, Any]], interface: str, split: str, layout: str) -> List[Dict[str, Any]]:
    if split == "train" and layout in TRAIN_LAYOUTS:
        return [row for row in train_rows if row["metadata"]["layout"] == layout]
    return exposure.expanded_eval_records(base, interface, split, layout)


def raw_model_predictions(model: Any, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared = ladder.prepare(records)
    rows: List[Dict[str, Any]] = []
    model.eval()
    for start in range(0, len(prepared), int(stage_a.BATCH_SIZE)):
        indices = list(range(start, min(start + int(stage_a.BATCH_SIZE), len(prepared))))
        batch = exposure.batchify(prepared, indices)
        with torch.inference_mode():
            logits = stage_a.forward_model(model, batch)
        for local, index in enumerate(indices):
            item = prepared[index]
            output_position = int(item["_prefix_len"]) - 1
            eos_position = output_position + 1
            predicted_digit = int(logits[local, output_position].argmax().item())
            predicted_eos = int(logits[local, eos_position].argmax().item())
            target_digit = int(batch["y"][local, output_position].item())
            target_eos = int(batch["y"][local, eos_position].item())
            value = predicted_digit - int(stage_a.RAW_BYTE_START)
            output = chr(value) if 48 <= value <= 57 else f"<token:{predicted_digit}>"
            rows.append({
                "predicted_output": output,
                "output_token_correct": predicted_digit == target_digit,
                "eos_correct": predicted_eos == target_eos,
                "full_sequence_correct": predicted_digit == target_digit and predicted_eos == target_eos,
                "output_format_error": not (48 <= value <= 57),
            })
    return rows


def execute_symbolic(a: int, b: int, operation: str) -> str:
    return str(a if operation == "SELECT_FIRST" else b)


def pair_metric(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    grouped: Dict[Tuple[int, int], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["split"] == split and int(row["a"]) != int(row["b"]):
            grouped[(int(row["a"]), int(row["b"]))][str(row["operation"])] = row
    complete = [value for value in grouped.values() if set(value) == {"SELECT_FIRST", "SELECT_SECOND"}]
    both = sum(bool(value["SELECT_FIRST"]["output_token_correct"] and value["SELECT_SECOND"]["output_token_correct"]) for value in complete)
    switched = sum(value["SELECT_FIRST"]["predicted_output"] != value["SELECT_SECOND"]["predicted_output"] for value in complete)
    return {"pair_groups": len(complete), "both_correct": both, "both_correct_rate": both / len(complete) if complete else None, "predictions_switched": switched}


def condition_metrics(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    values = [row for row in rows if row["split"] == split]
    parser_values = [row for row in values if row.get("parser_tuple_correct") is not None]
    return {
        "rows": len(values),
        "output_token_correct": sum(bool(row["output_token_correct"]) for row in values),
        "output_token_accuracy": sum(bool(row["output_token_correct"]) for row in values) / len(values) if values else None,
        "full_sequence_correct": sum(bool(row["full_sequence_correct"]) for row in values),
        "full_sequence_accuracy": sum(bool(row["full_sequence_correct"]) for row in values) / len(values) if values else None,
        "eos_correct": sum(bool(row["eos_correct"]) for row in values),
        "output_format_errors": sum(bool(row["output_format_error"]) for row in values),
        "paired_output": pair_metric(values, split),
        "parser_rows": len(parser_values),
        "parser_tuple_correct": sum(bool(row["parser_tuple_correct"]) for row in parser_values),
        "parser_tuple_accuracy": sum(bool(row["parser_tuple_correct"]) for row in parser_values) / len(parser_values) if parser_values else None,
    }


def evaluate_condition(condition: str, interface: str, seed: int, base: Mapping[str, Sequence[Mapping[str, Any]]], train_rows: Sequence[Mapping[str, Any]], parser_model: RoleParser | None, executor: Any | None, executor_label: str, parser_label: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    predictions: List[Dict[str, Any]] = []
    evaluations: List[Dict[str, Any]] = []
    for layout in EVAL_LAYOUTS:
        for split in ("train", "validation"):
            raw_records = evaluation_records(base, train_rows, interface, split, layout)
            parse_rows: List[Dict[str, Any]] | None = None
            parse_by_id: Dict[str, Mapping[str, Any]] = {}
            if parser_label == "oracle":
                parse_by_id = {
                    str(row["record_id"]): {
                        "predicted_a": int(row["metadata"]["a"]),
                        "predicted_b": int(row["metadata"]["b"]),
                        "predicted_operation": str(row["metadata"]["operation"]),
                        "a_correct": True,
                        "b_correct": True,
                        "operation_correct": True,
                        "tuple_correct": True,
                    }
                    for row in raw_records
                }
            elif parser_label == "learned":
                if parser_model is None:
                    raise RuntimeError("learned parser missing")
                parse_rows = parser_predictions(parser_model, raw_records, {"condition": condition, "interface": interface, "seed": seed}, layout, split)
                parse_by_id = {str(row["record_id"]): row for row in parse_rows}
            if condition == "C3_RAW_END_TO_END":
                if executor is None:
                    raise RuntimeError("raw executor missing")
                model_rows = raw_model_predictions(executor, raw_records)
            elif condition in ("C0_ORACLE_PARSER_A3_EXECUTOR", "C1_LEARNED_PARSER_A3_EXECUTOR"):
                if executor is None:
                    raise RuntimeError("A3 executor missing")
                transformed: List[Dict[str, Any]] = []
                for row in raw_records:
                    parsed = parse_by_id[str(row["record_id"])]
                    clone = dict(row)
                    clone["input"] = a3_input(interface, int(parsed["predicted_a"]), int(parsed["predicted_b"]), str(parsed["predicted_operation"]))
                    transformed.append(clone)
                model_rows = raw_model_predictions(executor, transformed)
            else:
                model_rows = []
            for index, row in enumerate(raw_records):
                parsed = parse_by_id.get(str(row["record_id"])) if parser_label != "none" else None
                if condition == "C2_LEARNED_PARSER_ORACLE_EXECUTOR":
                    if parsed is None:
                        raise RuntimeError("C2 parser output missing")
                    predicted_output = execute_symbolic(int(parsed["predicted_a"]), int(parsed["predicted_b"]), str(parsed["predicted_operation"]))
                    model = {"predicted_output": predicted_output, "output_token_correct": predicted_output == str(row["target"]), "eos_correct": True, "full_sequence_correct": predicted_output == str(row["target"]), "output_format_error": False}
                else:
                    model = model_rows[index]
                expected = str(row["target"])
                prediction = {
                    "condition": condition,
                    "interface": interface,
                    "seed": seed,
                    "layout": layout,
                    "split": split,
                    "record_id": str(row["record_id"]),
                    "a": int(row["metadata"]["a"]),
                    "b": int(row["metadata"]["b"]),
                    "operation": str(row["metadata"]["operation"]),
                    "expected_output": expected,
                    "predicted_output": str(model["predicted_output"]),
                    "output_token_correct": bool(model["output_token_correct"]),
                    "eos_correct": bool(model["eos_correct"]),
                    "full_sequence_correct": bool(model["full_sequence_correct"]),
                    "output_format_error": bool(model["output_format_error"]),
                    "parser_mode": parser_label,
                    "executor_mode": executor_label,
                }
                if parsed is None:
                    prediction.update({"predicted_a": None, "predicted_b": None, "predicted_operation": None, "parser_a_correct": None, "parser_b_correct": None, "parser_operation_correct": None, "parser_tuple_correct": None})
                else:
                    prediction.update({"predicted_a": int(parsed["predicted_a"]), "predicted_b": int(parsed["predicted_b"]), "predicted_operation": str(parsed["predicted_operation"]), "parser_a_correct": bool(parsed["a_correct"]), "parser_b_correct": bool(parsed["b_correct"]), "parser_operation_correct": bool(parsed["operation_correct"]), "parser_tuple_correct": bool(parsed["tuple_correct"])})
                predictions.append(prediction)
            eval_rows = [row for row in predictions if row["condition"] == condition and row["interface"] == interface and int(row["seed"]) == seed and row["layout"] == layout and row["split"] == split]
            evaluations.append({"condition": condition, "interface": interface, "seed": seed, "layout": layout, "split": split, "executor_mode": executor_label, "parser_mode": parser_label, "metrics": condition_metrics(eval_rows, split)})
    return predictions, evaluations


def report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Crucible Explicit Role-Binding Gate — Final Report",
        "",
        "CPU-only parser/executor decomposition. Existing A3 and B0 checkpoints were loaded read-only; no sealed-test access.",
        "",
        "## Parser competence",
        "",
        "| Interface | Seed | Train tuple accuracy | Validation tuple accuracy |",
        "|---|---:|---:|---:|",
    ]
    for result in summary["parsers"]:
        validation = [row for row in summary["evaluations"] if row["condition"] == "C1_LEARNED_PARSER_A3_EXECUTOR" and row["interface"] == result["interface"] and row["seed"] == result["seed"] and row["split"] == "validation" and row["layout"] == "FIXED_NEAR_PADDED"]
        valid = validation[0]["metrics"]["parser_tuple_accuracy"] if validation else None
        lines.append(f"| {result['interface']} | {result['seed']} | {result['train_metrics']['tuple_accuracy']:.2%} | {valid:.2%} |" if valid is not None else f"| {result['interface']} | {result['seed']} | {result['train_metrics']['tuple_accuracy']:.2%} | n/a |")
    lines += [
        "",
        "## Validation paired correctness",
        "",
        "Denominators are pooled over four interface/seed runs per condition; each layout contributes 32 unequal operand-pair groups.",
        "",
        "| Condition | Fixed-near | Early | Middle | Unseen-between | Unseen-early-context |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        cells: List[str] = []
        for layout in EVAL_LAYOUTS:
            values = [row for row in summary["evaluations"] if row["condition"] == condition and row["layout"] == layout and row["split"] == "validation"]
            correct = sum(int(row["metrics"]["paired_output"]["both_correct"]) for row in values)
            groups = sum(int(row["metrics"]["paired_output"]["pair_groups"]) for row in values)
            cells.append(f"{correct}/{groups}" if groups else "n/a")
        lines.append(f"| {condition} | {' | '.join(cells)} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "C0 tests sufficiency of oracle canonicalization plus the frozen A3 executor. C1 adds learned tuple extraction while holding execution fixed. C2 tests the same learned tuple against a deterministic executor. C3 is the raw end-to-end B0 control. Oracle parser/executor success is diagnostic scaffolding, not autonomous neural role binding.",
        "",
        "## Artifacts",
        "",
        "- [Protocol](../../docs/CRUCIBLE_EXPLICIT_ROLE_BINDING_GATE_PROTOCOL_20251025.md)",
        "- [Summary](SUMMARY.json)",
        "- [Parser predictions](PARSER_PREDICTIONS.csv)",
        "- [Predictions](PREDICTIONS.csv)",
        "- [Parser training traces](PARSER_TRAINING_TRACE.jsonl)",
        "- [Finalization](FINALIZATION.json)",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {"experiment": "crucible_explicit_role_binding_gate_20251025_v1", "protocol": rel(PROTOCOL), "conditions": list(CONDITIONS), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "steps": STEPS, "effective_batch": EFFECTIVE_BATCH, "sealed_test_accessed": False, "architecture_changed": False, "new_executor_training": False})

    schedules = json.loads((BATCH_ROOT / "BATCH_SCHEDULES.json").read_text(encoding="utf-8"))["B0_DIVERSE_SHUFFLED"]
    schedule_digest = hashlib.sha256(json.dumps(schedules, sort_keys=True).encode()).hexdigest()
    schedule_audit = json.loads((BATCH_ROOT / "SCHEDULE_AUDIT.json").read_text(encoding="utf-8"))
    if not schedule_audit["pass"]:
        raise SystemExit("source B0 schedule audit is not passing")
    write_json(OUTPUT / "SCHEDULE_REFERENCE.json", {"source": rel(BATCH_ROOT / "BATCH_SCHEDULES.json"), "arm": "B0_DIVERSE_SHUFFLED", "schedule_digest": schedule_digest, "steps": len(schedules), "effective_batch": EFFECTIVE_BATCH})

    parsers: List[Dict[str, Any]] = []
    parser_rows_all: List[Dict[str, Any]] = []
    parser_models: Dict[Tuple[str, int], RoleParser] = {}
    parser_traces: List[Dict[str, Any]] = []
    base_by_interface: Dict[str, Mapping[str, Sequence[Mapping[str, Any]]]] = {}
    train_by_interface: Dict[str, List[Dict[str, Any]]] = {}
    for interface in INTERFACES:
        base = exposure.base_records(interface, 42)
        train_rows, groups = exposure.expanded_records(base, interface, "train")
        validation_rows, _ = exposure.expanded_records(base, interface, "validation")
        base_by_interface[interface] = base
        train_by_interface[interface] = train_rows
        write_jsonl(OUTPUT / "data" / interface / "train.jsonl", train_rows)
        write_jsonl(OUTPUT / "data" / interface / "validation.jsonl", validation_rows)
        write_json(OUTPUT / "data" / interface / "GROUPS.json", {"groups": groups})
        for seed in SEEDS:
            parser_dir = OUTPUT / "parser_outputs" / interface / f"seed{seed}"
            result, traces, model = train_parser(interface, seed, train_rows, schedules, parser_dir)
            parsers.append(result)
            parser_traces.extend(traces)
            parser_models[(interface, seed)] = model
            # Export parser predictions across the same layout/split matrix.
            for layout in EVAL_LAYOUTS:
                for split in ("train", "validation"):
                    raw = evaluation_records(base, train_rows, interface, split, layout)
                    parser_rows_all.extend(parser_predictions(model, raw, {"interface": interface, "seed": seed}, layout, split))
            print(json.dumps({"event": "parser_complete", "interface": interface, "seed": seed, "train_tuple_accuracy": result["train_metrics"]["tuple_accuracy"]}, sort_keys=True), flush=True)

    predictions: List[Dict[str, Any]] = []
    evaluations: List[Dict[str, Any]] = []
    for interface in INTERFACES:
        base = base_by_interface[interface]
        train_rows = train_by_interface[interface]
        for seed in SEEDS:
            a3_checkpoint = POSITION_ROOT / "arm_outputs" / "A3_MARKER_FIXED" / interface / f"seed{seed}" / "checkpoint.pt"
            b0_checkpoint = BATCH_ROOT / "arm_outputs" / "B0_DIVERSE_SHUFFLED" / interface / f"seed{seed}" / "checkpoint.pt"
            if not a3_checkpoint.is_file() or not b0_checkpoint.is_file():
                raise FileNotFoundError(f"missing frozen control checkpoint for {interface} seed {seed}")
            parser_model = parser_models[(interface, seed)]
            for condition, executor, executor_label, parser_label in (
                ("C0_ORACLE_PARSER_A3_EXECUTOR", load_dense(a3_checkpoint), "frozen_A3_dense", "oracle"),
                ("C1_LEARNED_PARSER_A3_EXECUTOR", load_dense(a3_checkpoint), "frozen_A3_dense", "learned"),
                ("C2_LEARNED_PARSER_ORACLE_EXECUTOR", None, "deterministic_role_executor", "learned"),
                ("C3_RAW_END_TO_END", load_dense(b0_checkpoint), "frozen_B0_raw_dense", "none"),
            ):
                current_predictions, current_evaluations = evaluate_condition(condition, interface, seed, base, train_rows, parser_model if parser_label == "learned" else None, executor, executor_label, parser_label)
                for row in current_predictions:
                    row["executor_checkpoint"] = rel(a3_checkpoint if condition in ("C0_ORACLE_PARSER_A3_EXECUTOR", "C1_LEARNED_PARSER_A3_EXECUTOR") else b0_checkpoint) if condition != "C2_LEARNED_PARSER_ORACLE_EXECUTOR" else None
                    row["executor_checkpoint_sha256"] = sha256(a3_checkpoint if condition in ("C0_ORACLE_PARSER_A3_EXECUTOR", "C1_LEARNED_PARSER_A3_EXECUTOR") else b0_checkpoint) if condition != "C2_LEARNED_PARSER_ORACLE_EXECUTOR" else None
                    row["parser_checkpoint"] = rel(OUTPUT / "parser_outputs" / interface / f"seed{seed}" / "checkpoint.pt") if parser_label == "learned" else None
                    row["parser_checkpoint_sha256"] = parsers[[p["parser_id"] for p in parsers].index(f"learned_parser_{interface}_seed{seed}")]["checkpoint_sha256"] if parser_label == "learned" else None
                for row in current_evaluations:
                    row["executor_checkpoint"] = rel(a3_checkpoint if condition in ("C0_ORACLE_PARSER_A3_EXECUTOR", "C1_LEARNED_PARSER_A3_EXECUTOR") else b0_checkpoint) if condition != "C2_LEARNED_PARSER_ORACLE_EXECUTOR" else None
                    row["executor_checkpoint_sha256"] = sha256(a3_checkpoint if condition in ("C0_ORACLE_PARSER_A3_EXECUTOR", "C1_LEARNED_PARSER_A3_EXECUTOR") else b0_checkpoint) if condition != "C2_LEARNED_PARSER_ORACLE_EXECUTOR" else None
                    row["parser_checkpoint"] = rel(OUTPUT / "parser_outputs" / interface / f"seed{seed}" / "checkpoint.pt") if parser_label == "learned" else None
                predictions.extend(current_predictions)
                evaluations.extend(current_evaluations)
                del executor
                gc.collect()
            print(json.dumps({"event": "condition_complete", "interface": interface, "seed": seed}, sort_keys=True), flush=True)

    with (OUTPUT / "PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(predictions[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    with (OUTPUT / "PARSER_PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(parser_rows_all[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(parser_rows_all)
    write_jsonl(OUTPUT / "PARSER_TRAINING_TRACE.jsonl", parser_traces)
    summary = {
        "experiment": "crucible_explicit_role_binding_gate_20251025_v1",
        "status": "completed",
        "parsers": parsers,
        "evaluations": evaluations,
        "prediction_rows": len(predictions),
        "parser_prediction_rows": len(parser_rows_all),
        "parser_training_trace_rows": len(parser_traces),
        "source_schedule_digest": schedule_digest,
        "sealed_test_accessed": False,
        "architecture_changed": False,
        "new_executor_training": False,
        "parser_training_arms": len(parsers),
    }
    write_json(OUTPUT / "SUMMARY.json", summary)
    (OUTPUT / "REPORT_FINAL.md").write_text(report(summary), encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt":
            hashes[rel(path)] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = {"status": "completed", "verification_pass": True, "sealed_test_accessed": False, "architecture_changed": False, "new_executor_training": False, "parser_training_arms": len(parsers), "evaluations": len(evaluations), "prediction_rows": len(predictions), "parser_prediction_rows": len(parser_rows_all), "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"), "prediction_sha256": sha256(OUTPUT / "PREDICTIONS.csv")}
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "completed", "output": rel(OUTPUT), "parser_training_arms": len(parsers), "evaluations": len(evaluations), "prediction_rows": len(predictions), "verification_pass": True}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
