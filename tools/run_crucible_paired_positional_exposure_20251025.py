#!/usr/bin/env python3
"""Compute-matched paired-positional exposure and consistency experiment."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_paired_positional_exposure_20251025_v1"
POSITION_ROOT = ROOT / "experiments" / "crucible_operation_position_matrix_20251025_v1"
ROLE_SOURCE = ROOT / "tools" / "run_crucible_role_selection_competence_20251025.py"
ADJACENCY_SOURCE = ROOT / "tools" / "run_crucible_relevance_adjacency_gate_20251025.py"
INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
ARMS = ("A0_SHUFFLED_EXPOSURE", "A1_GROUPED_EXPOSURE", "A2_GROUPED_CONSISTENCY")
TRAIN_LAYOUTS = ("FIXED_NEAR_PADDED", "EARLY_PADDED", "MIDDLE_POSITION")
EVAL_LAYOUTS = TRAIN_LAYOUTS + ("UNSEEN_BETWEEN_OPERANDS", "UNSEEN_EARLY_CONTEXT_FIRST")
STEPS = 500
EFFECTIVE_BATCH = 12
GROUPS_PER_BATCH = 2
CONSISTENCY_WEIGHT = 0.1
CURVE_STEPS = (1, 10, 25, 50, 100, 200, 300, 400, 500)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


role = import_module("paired_exposure_role_runner", ROLE_SOURCE)
adjacency = import_module("paired_exposure_adjacency_runner", ADJACENCY_SOURCE)
ladder = role.ladder
stage_a = role.stage_a
torch = stage_a.torch


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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def base_records(interface: str, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    directory = POSITION_ROOT / "data" / "A3_MARKER_FIXED" / interface
    return {"train": load_jsonl(directory / "train.jsonl"), "validation": load_jsonl(directory / "validation.jsonl")}


def marker_field(operation: str) -> str:
    return ("F" if operation == "SELECT_FIRST" else "S") + ("_" * (len(operation.encode("utf-8")) - 1))


def operands(interface: str, a: int, b: int) -> str:
    return f"a={a}; b={b}." if interface == "canonical" else f"a_digit={a}; b_digit={b}."


def layout_input(interface: str, operation: str, a: int, b: int, layout: str) -> str:
    field = marker_field(operation)
    pair = operands(interface, a, b)
    if layout == "FIXED_NEAR_PADDED":
        return f"Select an operand. context=neutral_17; {pair} Return digit only. operation={field};"
    if layout == "EARLY_PADDED":
        return f"Select an operand. operation={field}; context=neutral_17; {pair} Return digit only."
    if layout == "MIDDLE_POSITION":
        return f"Select an operand. {pair} context=neutral_17; operation={field}; Return digit only."
    if layout == "UNSEEN_BETWEEN_OPERANDS":
        if interface == "canonical":
            pair_split = f"a={a}; operation={field}; context=neutral_17; b={b}."
        else:
            pair_split = f"a_digit={a}; operation={field}; context=neutral_17; b_digit={b}."
        return f"Select an operand. {pair_split} Return digit only."
    if layout == "UNSEEN_EARLY_CONTEXT_FIRST":
        return f"Select an operand. context=neutral_17; operation={field}; {pair} Return digit only."
    raise ValueError(layout)


def pair_order(records: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int]]:
    return sorted({(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in records})


def expanded_records(base: Mapping[str, Sequence[Mapping[str, Any]]], interface: str, split: str) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
    source = base[split]
    by_key = {(int(row["metadata"]["a"]), int(row["metadata"]["b"]), str(row["metadata"]["operation"])): row for row in source}
    records: List[Dict[str, Any]] = []
    groups: List[List[int]] = []
    for a, b in pair_order(source):
        pair_group: List[int] = []
        for operation in ("SELECT_FIRST", "SELECT_SECOND"):
            source_row = by_key[(a, b, operation)]
            for layout in TRAIN_LAYOUTS:
                cloned = dict(source_row)
                cloned["record_id"] = hashlib.sha256(f"paired_exposure|{interface}|{split}|{a}|{b}|{operation}|{layout}".encode()).hexdigest()[:20]
                cloned["input"] = layout_input(interface, operation, a, b, layout)
                cloned["condition"] = "PAIRED_POSITIONAL_EXPOSURE"
                cloned["metadata"] = {**source_row["metadata"], "layout": layout, "base_pair": [a, b]}
                pair_group.append(len(records))
                records.append(cloned)
        groups.append(pair_group)
    return records, groups


def expanded_eval_records(base: Mapping[str, Sequence[Mapping[str, Any]]], interface: str, split: str, layout: str) -> List[Dict[str, Any]]:
    source = base[split]
    rows: List[Dict[str, Any]] = []
    for index, source_row in enumerate(source):
        a = int(source_row["metadata"]["a"])
        b = int(source_row["metadata"]["b"])
        operation = str(source_row["metadata"]["operation"])
        cloned = dict(source_row)
        cloned["record_id"] = hashlib.sha256(f"paired_eval|{interface}|{split}|{a}|{b}|{operation}|{layout}".encode()).hexdigest()[:20]
        cloned["input"] = layout_input(interface, operation, a, b, layout)
        cloned["condition"] = "PAIRED_EVALUATION"
        cloned["metadata"] = {**source_row["metadata"], "layout": layout}
        rows.append(cloned)
    return rows


def batchify(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    selected = [records[index] for index in indices]
    max_len = max(len(row["_full"]) - 1 for row in selected)
    x = torch.full((len(selected), max_len), stage_a.PAD_ID, dtype=torch.long)
    y = torch.full((len(selected), max_len), stage_a.PAD_ID, dtype=torch.long)
    target_mask = torch.zeros((len(selected), max_len), dtype=torch.bool)
    for row_index, record in enumerate(selected):
        full = torch.tensor(record["_full"], dtype=torch.long)
        x[row_index, : full.numel() - 1] = full[:-1]
        y[row_index, : full.numel() - 1] = full[1:]
        start = int(record["_prefix_len"]) - 1
        target_mask[row_index, start : full.numel() - 1] = True
    return {"x": x, "y": y, "target_mask": target_mask}


def selection_indices(arm: str, step: int, record_order: Sequence[int], groups: Sequence[Sequence[int]]) -> Tuple[List[int], List[Tuple[int, int, int]]]:
    if arm == "A0_SHUFFLED_EXPOSURE":
        start = ((step - 1) * EFFECTIVE_BATCH) % len(record_order)
        indices = [record_order[(start + offset) % len(record_order)] for offset in range(EFFECTIVE_BATCH)]
        return indices, []
    group_start = ((step - 1) * GROUPS_PER_BATCH) % len(groups)
    selected_groups = [groups[(group_start + offset) % len(groups)] for offset in range(GROUPS_PER_BATCH)]
    indices = [index for group in selected_groups for index in group]
    triplets: List[Tuple[int, int, int]] = []
    for group_offset in range(GROUPS_PER_BATCH):
        base = group_offset * 6
        triplets.extend([(base, base + 1, base + 2), (base + 3, base + 4, base + 5)])
    return indices, triplets


def consistency_loss(logits: torch.Tensor, prepared: Sequence[Mapping[str, Any]], triplets: Sequence[Tuple[int, int, int]]) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for left, middle, right in triplets:
        positions = [int(prepared[index]["_prefix_len"]) - 1 for index in (left, middle, right)]
        stacked = torch.stack([logits[index, position] for index, position in zip((left, middle, right), positions)])
        probabilities = torch.softmax(stacked.float(), dim=-1)
        mean_probability = probabilities.mean(dim=0, keepdim=True)
        log_mean = torch.log(mean_probability.clamp_min(1.0e-8))
        forward = (probabilities * (torch.log(probabilities.clamp_min(1.0e-8)) - log_mean)).sum(dim=-1).mean()
        reverse = (mean_probability * (log_mean - torch.log(probabilities.clamp_min(1.0e-8)))).sum(dim=-1).mean()
        values.append(0.5 * (forward + reverse))
    return torch.stack(values).mean() if values else logits.new_zeros(())


def train_step(model: Any, optimizer: Any, prepared: Sequence[Mapping[str, Any]], indices: Sequence[int], triplets: Sequence[Tuple[int, int, int]], use_consistency: bool) -> Dict[str, Any]:
    model.train()
    batch = batchify(prepared, indices)
    optimizer.zero_grad(set_to_none=True)
    logits = stage_a.forward_model(model, batch)
    task_loss, correct, total = stage_a.loss_and_accuracy(logits, batch)
    agreement = consistency_loss(logits, [prepared[index] for index in indices], triplets) if use_consistency else logits.new_zeros(())
    loss = task_loss + CONSISTENCY_WEIGHT * agreement
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), stage_a.CLIP_NORM).item())
    optimizer.step()
    return {"loss": float(loss.detach().item()), "task_loss": float(task_loss.detach().item()), "consistency_loss": float(agreement.detach().item()), "correct": correct, "total": total, "grad_norm": grad_norm}


def train_model(arm: str, interface: str, seed: int, records: Sequence[Mapping[str, Any]], groups: Sequence[Sequence[int]], output: Path) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    prepared = ladder.prepare(records)
    record_order = list(range(len(prepared)))
    random.Random(ladder.stable_seed(seed, "paired_exposure_shuffled_order", interface)).shuffle(record_order)
    curves: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        indices, triplets = selection_indices(arm, step, record_order, groups)
        train_step(model, optimizer, prepared, indices, triplets, arm == "A2_GROUPED_CONSISTENCY")
        if step in CURVE_STEPS:
            curves.append({"step": step, "task": stage_a.evaluate(model, prepared)})
    final_eval = stage_a.evaluate(model, prepared)
    checkpoint = output / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint, model, optimizer, STEPS, record_order)
    result = {"arm_id": f"{arm}_{interface}_seed{seed}", "arm": arm, "interface": interface, "seed": seed, "steps": STEPS, "effective_batch": EFFECTIVE_BATCH, "expanded_records": len(records), "groups": len(groups), "consistency_weight": CONSISTENCY_WEIGHT, "elapsed_seconds": time.perf_counter() - started, "train_expanded": final_eval, "train_expanded_exact": final_eval["teacher_forced_exact_accuracy"], "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint), "training_layout_counts": {layout: sum(1 for row in records if row["metadata"]["layout"] == layout) for layout in TRAIN_LAYOUTS}}
    write_json(output / "RESULTS.json", result)
    write_json(output / "TRAINING_CURVE.json", {"curve_steps": list(CURVE_STEPS), "points": curves})
    del model, optimizer
    gc.collect()
    return result


def load_model(path: Path) -> Any:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = stage_a.DenseLM()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def report(summary: Mapping[str, Any]) -> str:
    lines = ["# Crucible Paired-Positional Exposure Gate — Final Report", "", "Compute-matched CPU comparison. No sealed-test access.", "", "## Training arms", "", "| Arm | Interface | Seed | Expanded train exact | Fixed-near paired | Early paired | Middle paired |", "|---|---|---:|---:|---:|---:|---:|"]
    for result in summary["arms"]:
        values = [x for x in summary["evaluations"] if x["arm_id"] == result["arm_id"]]
        pairs = {x["layout"]: x["metrics"]["by_split"]["train"]["paired_unequal"]["both_correct_rate"] for x in values}
        lines.append(f"| {result['arm']} | {result['interface']} | {result['seed']} | {result['train_expanded_exact']:.2%} | {pairs['FIXED_NEAR_PADDED']:.2%} | {pairs['EARLY_PADDED']:.2%} | {pairs['MIDDLE_POSITION']:.2%} |")
    lines += ["", "## Held-out operand-pair evaluation", "", "| Layout | A0 shuffled | A1 grouped | A2 grouped + consistency | A0 paired | A1 paired | A2 paired |", "|---|---:|---:|---:|---:|---:|---:|"]
    for layout in EVAL_LAYOUTS:
        cells = []
        for arm in ARMS:
            values = [x for x in summary["evaluations"] if x["arm"] == arm and x["layout"] == layout]
            accuracy = sum(x["metrics"]["by_split"]["validation"]["accuracy"] for x in values) / len(values)
            paired = sum(x["metrics"]["by_split"]["validation"]["paired_unequal"]["both_correct_rate"] for x in values) / len(values)
            cells.extend([f"{accuracy:.2%}", f"{paired:.2%}"])
        lines.append(f"| {layout} | {cells[0]} | {cells[2]} | {cells[4]} | {cells[1]} | {cells[3]} | {cells[5]} |")
    lines += ["", "## Interpretation boundary", "", "A positive A1 result would support aligned exposure as a training intervention. A2 beating A1 under equal exposure would support the consistency objective specifically. Neither result would establish general relational reasoning beyond the tested layouts.", "", "Attention summaries are descriptive shadow Q/K probabilities while native SDPA remains on the output path.", "", "## Artifacts", "", "- [Protocol](../../docs/CRUCIBLE_PAIRED_POSITIONAL_EXPOSURE_PROTOCOL_20251025.md)", "- [Summary](SUMMARY.json)", "- [Predictions](PREDICTIONS.csv)", "- [Attention summaries](ATTENTION_SUMMARIES.json)", "- [Finalization](FINALIZATION.json)"]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {"experiment": "crucible_paired_positional_exposure_20251025_v1", "status": "frozen_before_training", "arms": list(ARMS), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "train_layouts": list(TRAIN_LAYOUTS), "eval_layouts": list(EVAL_LAYOUTS), "steps": STEPS, "effective_batch": EFFECTIVE_BATCH, "consistency_weight": CONSISTENCY_WEIGHT, "sealed_test_accessed": False, "architecture_changed": False})
    arms: List[Dict[str, Any]] = []
    evaluations: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    attention_summaries: Dict[str, Any] = {}
    for arm in ARMS:
        for interface in INTERFACES:
            for seed in SEEDS:
                base = base_records(interface, seed)
                train_rows, train_groups = expanded_records(base, interface, "train")
                validation_rows, _ = expanded_records(base, interface, "validation")
                arm_dir = OUTPUT / "arm_outputs" / arm / interface / f"seed{seed}"
                write_jsonl(OUTPUT / "data" / arm / interface / f"seed{seed}" / "train.jsonl", train_rows)
                write_jsonl(OUTPUT / "data" / arm / interface / f"seed{seed}" / "validation.jsonl", validation_rows)
                write_json(OUTPUT / "data" / arm / interface / f"seed{seed}" / "GROUPS.json", {"groups": train_groups})
                result = train_model(arm, interface, seed, train_rows, train_groups, arm_dir)
                arms.append(result)
                model = load_model(arm_dir / "checkpoint.pt")
                for layout in EVAL_LAYOUTS:
                    eval_records = {"train": [row for row in train_rows if row["metadata"]["layout"] == layout] if layout in TRAIN_LAYOUTS else expanded_eval_records(base, interface, "train", layout), "validation": expanded_eval_records(base, interface, "validation", layout)}
                    rows, metrics, attention, arm_traces = adjacency.evaluate(model, eval_records, {"arm": arm, "arm_id": result["arm_id"], "interface": interface, "seed": seed, "checkpoint": result["checkpoint"], "checkpoint_sha256": result["checkpoint_sha256"]}, layout)
                    evaluations.append({"arm": arm, "arm_id": result["arm_id"], "interface": interface, "seed": seed, "layout": layout, "metrics": metrics, "attention": attention})
                    predictions.extend(rows)
                    traces.extend(arm_traces)
                    attention_summaries[f"{result['arm_id']}__{layout}"] = attention
                del model
                gc.collect()
                print(json.dumps({"event": "complete", "arm_id": result["arm_id"], "train_expanded": result["train_expanded_exact"]}, sort_keys=True), flush=True)
    with (OUTPUT / "PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(predictions[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    write_json(OUTPUT / "ATTENTION_SUMMARIES.json", attention_summaries)
    write_jsonl(OUTPUT / "ATTENTION_TRACES.jsonl", traces)
    summary = {"experiment": "crucible_paired_positional_exposure_20251025_v1", "status": "completed", "arms": arms, "evaluations": evaluations, "prediction_rows": len(predictions), "attention_trace_rows": len(traces), "sealed_test_accessed": False, "architecture_changed": False, "retrained": True}
    write_json(OUTPUT / "SUMMARY.json", summary)
    (OUTPUT / "REPORT_FINAL.md").write_text(report(summary), encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt":
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = {"status": "completed", "verification_pass": all(x["metrics"]["instrumentation_pass"] for x in evaluations), "sealed_test_accessed": False, "architecture_changed": False, "retrained": True, "arms": len(arms), "evaluations": len(evaluations), "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"), "prediction_sha256": sha256(OUTPUT / "PREDICTIONS.csv")}
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "completed", "output": str(OUTPUT.relative_to(ROOT)), "arms": len(arms), "evaluations": len(evaluations), "prediction_rows": len(predictions), "verification_pass": finalization["verification_pass"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
