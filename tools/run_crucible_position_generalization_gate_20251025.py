#!/usr/bin/env python3
"""Train matched fixed-position and position-varied CPU controls."""

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
OUTPUT = ROOT / "experiments" / "crucible_position_generalization_gate_20251025_v1"
POSITION_ROOT = ROOT / "experiments" / "crucible_operation_position_matrix_20251025_v1"
ROLE_SOURCE = ROOT / "tools" / "run_crucible_role_selection_competence_20251025.py"
ADJACENCY_SOURCE = ROOT / "tools" / "run_crucible_relevance_adjacency_gate_20251025.py"
INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
ARMS = ("FIXED_POSITION_CONTROL", "POSITION_VARIED_INTERVENTION")
EVAL_LAYOUTS = ("A3_EXACT_ADJACENT", "FIXED_NEAR_PADDED", "EARLY_PADDED", "MIDDLE_POSITION", "UNSEEN_BETWEEN_OPERANDS", "UNSEEN_EARLY_CONTEXT_FIRST")
TRAIN_STEPS = 500
TRAIN_THRESHOLD = 0.95
VALIDATION_THRESHOLD = 0.90
CURVE_STEPS = (1, 10, 25, 50, 100, 200, 300, 400, 500)
NEUTRAL = "context=neutral_17;"


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


role = import_module("position_generalization_role_runner", ROLE_SOURCE)
adjacency = import_module("position_generalization_adjacency_runner", ADJACENCY_SOURCE)
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


def source_records(interface: str, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    directory = POSITION_ROOT / "data" / "A3_MARKER_FIXED" / interface
    return {"train": load_jsonl(directory / "train.jsonl"), "validation": load_jsonl(directory / "validation.jsonl")}


def marker_field(operation: str) -> str:
    return ("F" if operation == "SELECT_FIRST" else "S") + ("_" * (len(operation.encode("utf-8")) - 1))


def operands(interface: str, a: int, b: int) -> str:
    return f"a={a}; b={b}." if interface == "canonical" else f"a_digit={a}; b_digit={b}."


def layout_input(interface: str, operation: str, a: int, b: int, layout: str) -> str:
    field = marker_field(operation)
    pair = operands(interface, a, b)
    if layout == "A3_EXACT_ADJACENT":
        return f"Select an operand. {pair} Return digit only. operation={field};"
    if layout == "FIXED_NEAR_PADDED":
        return f"Select an operand. {NEUTRAL} {pair} Return digit only. operation={field};"
    if layout == "EARLY_PADDED":
        return f"Select an operand. operation={field}; {NEUTRAL} {pair} Return digit only."
    if layout == "MIDDLE_POSITION":
        return f"Select an operand. {pair} {NEUTRAL} operation={field}; Return digit only."
    if layout == "UNSEEN_BETWEEN_OPERANDS":
        if interface == "canonical":
            pair_split = f"a={a}; operation={field}; {NEUTRAL} b={b}."
        else:
            pair_split = f"a_digit={a}; operation={field}; {NEUTRAL} b_digit={b}."
        return f"Select an operand. {pair_split} Return digit only."
    if layout == "UNSEEN_EARLY_CONTEXT_FIRST":
        return f"Select an operand. {NEUTRAL} operation={field}; {pair} Return digit only."
    raise ValueError(layout)


def clone_records(base: Mapping[str, Sequence[Mapping[str, Any]]], interface: str, arm: str, layout: str | None = None) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for split, records in base.items():
        result[split] = []
        for index, row in enumerate(records):
            a = int(row["metadata"]["a"])
            b = int(row["metadata"]["b"])
            operation = str(row["metadata"]["operation"])
            if split == "train" and arm == "POSITION_VARIED_INTERVENTION":
                chosen = ("FIXED_NEAR_PADDED", "EARLY_PADDED", "MIDDLE_POSITION")[index % 3]
            else:
                chosen = "FIXED_NEAR_PADDED" if layout is None else layout
            cloned = dict(row)
            cloned["record_id"] = hashlib.sha256(f"position_generalization|{arm}|{interface}|{split}|{index}|{a}|{b}|{operation}|{chosen}".encode()).hexdigest()[:20]
            cloned["input"] = layout_input(interface, operation, a, b, chosen)
            cloned["condition"] = arm
            cloned["metadata"] = {**row["metadata"], "layout": chosen, "arm": arm}
            result[split].append(cloned)
    return result


def train_model(arm: str, interface: str, seed: int, records: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    initial_digest = stage_a.canonical_state_digest(model.state_dict())
    optimizer = stage_a.optimizer_for(model)
    prepared = {split: ladder.prepare(values) for split, values in records.items()}
    order = list(range(len(prepared["train"])))
    random.Random(ladder.stable_seed(seed, "position_generalization_shared_order", interface)).shuffle(order)
    curves: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, TRAIN_STEPS + 1):
        stage_a.train_step(model, optimizer, prepared["train"], stage_a.batch_indices(order, step))
        if step in CURVE_STEPS:
            curves.append({"step": step, "train": stage_a.evaluate(model, prepared["train"]), "validation": stage_a.evaluate(model, prepared["validation"])})
    train_eval = stage_a.evaluate(model, prepared["train"])
    validation_eval = stage_a.evaluate(model, prepared["validation"])
    checkpoint = output / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint, model, optimizer, TRAIN_STEPS, order)
    result = {
        "arm_id": f"{arm}_{interface}_seed{seed}",
        "arm": arm,
        "interface": interface,
        "seed": seed,
        "steps": TRAIN_STEPS,
        "elapsed_seconds": time.perf_counter() - started,
        "initial_dense_state_digest": initial_digest,
        "train": train_eval,
        "validation": validation_eval,
        "train_exact": train_eval["teacher_forced_exact_accuracy"],
        "validation_exact": validation_eval["teacher_forced_exact_accuracy"],
        "fit_pass": train_eval["teacher_forced_exact_accuracy"] >= TRAIN_THRESHOLD,
        "validation_pass": validation_eval["teacher_forced_exact_accuracy"] >= VALIDATION_THRESHOLD,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "training_template_counts": {layout: sum(1 for row in records["train"] if row["metadata"]["layout"] == layout) for layout in ("FIXED_NEAR_PADDED", "EARLY_PADDED", "MIDDLE_POSITION")},
    }
    write_json(output / "RESULTS.json", result)
    write_json(output / "TRAINING_CURVE.json", {"curve_steps": list(CURVE_STEPS), "points": curves})
    del model, optimizer
    gc.collect()
    return result


def load_model(checkpoint: Path) -> Any:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = stage_a.DenseLM()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def aggregate_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Crucible Position-Generalization Gate — Final Report",
        "",
        "Matched CPU training comparison. No sealed-test access.",
        "",
        "## Training arms",
        "",
        "| Arm | Interface | Seed | Train exact | Validation exact | Fit gate |",
        "|---|---|---:|---:|---:|---|",
    ]
    for result in summary["arms"]:
        lines.append(f"| {result['arm']} | {result['interface']} | {result['seed']} | {result['train_exact']:.2%} | {result['validation_exact']:.2%} | {'PASS' if result['fit_pass'] else 'FAIL'} |")
    lines += [
        "",
        "## Layout evaluation",
        "",
        "| Layout | Fixed-control validation | Varied-intervention validation | Fixed paired | Varied paired |",
        "|---|---:|---:|---:|---:|",
    ]
    for layout in EVAL_LAYOUTS:
        fixed = [x for x in summary["evaluations"] if x["arm"] == "FIXED_POSITION_CONTROL" and x["layout"] == layout]
        varied = [x for x in summary["evaluations"] if x["arm"] == "POSITION_VARIED_INTERVENTION" and x["layout"] == layout]
        fixed_acc = sum(x["metrics"]["by_split"]["validation"]["accuracy"] for x in fixed) / len(fixed)
        varied_acc = sum(x["metrics"]["by_split"]["validation"]["accuracy"] for x in varied) / len(varied)
        fixed_pair = sum(x["metrics"]["by_split"]["validation"]["paired_unequal"]["both_correct_rate"] for x in fixed) / len(fixed)
        varied_pair = sum(x["metrics"]["by_split"]["validation"]["paired_unequal"]["both_correct_rate"] for x in varied) / len(varied)
        lines.append(f"| {layout} | {fixed_acc:.2%} | {varied_acc:.2%} | {fixed_pair:.2%} | {varied_pair:.2%} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Position-varied improvement would support learned robustness to the preregistered positional distribution. It would not establish position-independent reasoning beyond the tested layouts. Attention summaries are descriptive shadow Q/K probabilities while native SDPA remains on the output path.",
        "",
        "## Artifacts",
        "",
        "- [Protocol](../../docs/CRUCIBLE_POSITION_GENERALIZATION_GATE_PROTOCOL_20251025.md)",
        "- [Summary](SUMMARY.json)",
        "- [Predictions](PREDICTIONS.csv)",
        "- [Attention summaries](ATTENTION_SUMMARIES.json)",
        "- [Finalization](FINALIZATION.json)",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {"experiment": "crucible_position_generalization_gate_20251025_v1", "status": "frozen_before_training", "arms": list(ARMS), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "eval_layouts": list(EVAL_LAYOUTS), "steps": TRAIN_STEPS, "sealed_test_accessed": False, "architecture_changed": False})
    arms: List[Dict[str, Any]] = []
    evaluations: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []
    all_traces: List[Dict[str, Any]] = []
    attention_summaries: Dict[str, Any] = {}
    for arm in ARMS:
        for interface in INTERFACES:
            base_by_seed = {seed: source_records(interface, seed) for seed in SEEDS}
            # The pair split is the same for both seeds; records are rebuilt
            # from each seed's immutable A3 data to retain provenance.
            for seed in SEEDS:
                records = clone_records(base_by_seed[seed], interface, arm)
                arm_dir = OUTPUT / "arm_outputs" / arm / interface / f"seed{seed}"
                for split in ("train", "validation"):
                    write_jsonl(OUTPUT / "data" / arm / interface / f"seed{seed}" / f"{split}.jsonl", records[split])
                result = train_model(arm, interface, seed, records, arm_dir)
                result["records"] = {"train": len(records["train"]), "validation": len(records["validation"]) }
                arms.append(result)
                model = load_model(arm_dir / "checkpoint.pt")
                for layout in EVAL_LAYOUTS:
                    evaluation_records = clone_records(base_by_seed[seed], interface, arm, layout=layout)
                    rows, metrics, attention, traces = adjacency.evaluate(model, evaluation_records, {"arm_id": result["arm_id"], "arm": arm, "interface": interface, "seed": seed, "checkpoint": result["checkpoint"], "checkpoint_sha256": result["checkpoint_sha256"]}, layout)
                    evaluations.append({"arm": arm, "arm_id": result["arm_id"], "interface": interface, "seed": seed, "layout": layout, "metrics": metrics, "attention": attention})
                    all_predictions.extend(rows)
                    all_traces.extend(traces)
                    attention_summaries[f"{result['arm_id']}__{layout}"] = attention
                    write_json(OUTPUT / "evaluations" / arm / interface / f"seed{seed}" / f"{layout}.json", {"metrics": metrics, "attention": attention})
                del model
                gc.collect()
                print(json.dumps({"event": "arm_complete", "arm_id": result["arm_id"], "train": result["train_exact"], "validation": result["validation_exact"]}, sort_keys=True), flush=True)
    with (OUTPUT / "PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(all_predictions[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_predictions)
    write_json(OUTPUT / "ATTENTION_SUMMARIES.json", attention_summaries)
    write_jsonl(OUTPUT / "ATTENTION_TRACES.jsonl", all_traces)
    summary = {"experiment": "crucible_position_generalization_gate_20251025_v1", "status": "completed", "arms": arms, "evaluations": evaluations, "prediction_rows": len(all_predictions), "attention_trace_rows": len(all_traces), "sealed_test_accessed": False, "architecture_changed": False, "retrained": True}
    write_json(OUTPUT / "SUMMARY.json", summary)
    (OUTPUT / "REPORT_FINAL.md").write_text(aggregate_report(summary), encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt":
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = {"status": "completed", "verification_pass": all(e["metrics"]["instrumentation_pass"] for e in evaluations), "sealed_test_accessed": False, "architecture_changed": False, "retrained": True, "arms": len(arms), "evaluations": len(evaluations), "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"), "prediction_sha256": sha256(OUTPUT / "PREDICTIONS.csv")}
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "completed", "output": str(OUTPUT.relative_to(ROOT)), "arms": len(arms), "evaluations": len(evaluations), "prediction_rows": len(all_predictions), "verification_pass": finalization["verification_pass"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
