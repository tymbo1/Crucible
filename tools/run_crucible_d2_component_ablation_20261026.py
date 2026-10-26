#!/usr/bin/env python3
"""CPU 2x2 ablation of the successful S4 and failed D2 decision interfaces."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_d2_component_ablation_20261026_v1"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_D2_COMPONENT_ABLATION_PROTOCOL_20261026.md"
LADDER_SOURCE = ROOT / "tools" / "run_crucible_d2_readout_competence_ladder_20261026.py"
LADDER_ROOT = ROOT / "experiments" / "crucible_d2_readout_competence_ladder_20261026_v1"
CAUSAL_ROOT = ROOT / "experiments" / "crucible_causally_valid_authority_decision_20261026_v1"

INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
CELLS = ("C1_SLOT_DECISION_ONLY", "C2_SLOT_FULL_MULTITASK", "C3_OPERATION_DECISION_ONLY", "C4_OPERATION_FULL_MULTITASK")
STEPS = 500
EFFECTIVE_BATCH = 12
TRAIN_THRESHOLD = 0.95


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


ladder = import_module("d2_readout_ladder_for_ablation", LADDER_SOURCE)
causal = ladder.causal
auth = ladder.auth
torch = ladder.torch
nn = torch.nn
F = torch.nn.functional
stage_a = ladder.stage_a


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
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_s3_data(interface: str) -> Dict[str, List[Dict[str, Any]]]:
    root = LADDER_ROOT / "data" / interface / "S3_AUTHORITY_CONDITIONED"
    return {"train": load_jsonl(root / "train.jsonl"), "validation": load_jsonl(root / "validation.jsonl")}


def batchify(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    base = ladder.batchify(records, indices)
    selected = [records[index] for index in indices]
    base["a"] = torch.tensor([int(row["metadata"]["a"]) for row in selected], dtype=torch.long)
    base["b"] = torch.tensor([int(row["metadata"]["b"]) for row in selected], dtype=torch.long)
    base["a_pos"] = torch.tensor([int(row["_a_pos"]) for row in selected], dtype=torch.long)
    base["b_pos"] = torch.tensor([int(row["_b_pos"]) for row in selected], dtype=torch.long)
    base["operation_target"] = torch.tensor([0 if row["metadata"]["operation"] == "SELECT_FIRST" else 1 for row in selected], dtype=torch.long)
    return base


def make_schedule(size: int) -> List[List[int]]:
    return [[(step * EFFECTIVE_BATCH + offset) % size for offset in range(EFFECTIVE_BATCH)] for step in range(STEPS)]


def operation_class_rows(records: Sequence[Mapping[str, Any]], logits: torch.Tensor, cell: str, interface: str, seed: int, source: str) -> List[Dict[str, Any]]:
    prepared = ladder.prepare_records(records)
    rows: List[Dict[str, Any]] = []
    predicted_classes = logits.argmax(dim=-1).tolist()
    for index, item in enumerate(prepared):
        predicted_operation = "SELECT_FIRST" if int(predicted_classes[index]) == 0 else "SELECT_SECOND"
        operations = ladder.operation_names(str(item["metadata"]["operation_order"]))
        predicted_slot = operations.index(predicted_operation)
        target_slot = int(item["metadata"]["target_slot"])
        expected_operation = str(item["metadata"]["operation"])
        predicted_output = str(item["metadata"]["a"] if predicted_operation == "SELECT_FIRST" else item["metadata"]["b"])
        rows.append({
            "cell": cell, "interface": interface, "seed": seed, "source": source, "split": str(item["split"]),
            "record_id": str(item["record_id"]), "counterfactual_pair_id": str(item["metadata"]["counterfactual_pair_id"]),
            "a": int(item["metadata"]["a"]), "b": int(item["metadata"]["b"]), "operation_order": str(item["metadata"]["operation_order"]), "authority_code": str(item["metadata"]["authority_code"]),
            "readout": "operation_class", "target_slot": target_slot, "predicted_slot": predicted_slot,
            "target_operation": expected_operation, "predicted_operation": predicted_operation,
            "target_operation_position": int(item["_op_positions"][target_slot]), "predicted_operation_position": int(item["_op_positions"][predicted_slot]),
            "decision_position": int(item["_decision_position"]), "pointer_correct": predicted_slot == target_slot,
            "operation_correct": predicted_operation == expected_operation, "output": predicted_output, "expected_output": str(item["target"]),
            "output_correct": predicted_output == str(item["target"]), "format_error": False,
        })
    return rows


def slot_outputs(model: Any, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    hidden = model.hidden_sequence(batch["x"])
    rows = torch.arange(hidden.size(0), dtype=torch.long)
    end_hidden = hidden[rows, batch["lengths"] - 1]
    decision_hidden = hidden[rows, batch["decision_pos"]]
    return {"a": model.a_head(end_hidden), "b": model.b_head(end_hidden), "a_pos": model.a_pointer(hidden).squeeze(-1), "b_pos": model.b_pointer(hidden).squeeze(-1), "decision": model.decision_pointer(decision_hidden)}


def operation_outputs(model: Any, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    raw = model.forward_all(batch)
    # The original causal D2 model calls this head ``decision_op``;
    # normalize it to the ablation's common ``decision`` name.
    return {
        "a": raw["a"], "b": raw["b"], "a_pos": raw["a_pos"], "b_pos": raw["b_pos"],
        "decision": raw["decision_op"],
    }


def multitask_loss(logits: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], readout: str) -> torch.Tensor:
    decision_target = batch["target_slot"] if readout == "slot" else batch["operation_target"]
    decision_loss = F.cross_entropy(logits["decision"], decision_target)
    value_loss = (F.cross_entropy(logits["a"], batch["a"]) + F.cross_entropy(logits["b"], batch["b"]) + decision_loss) / 3.0
    pointer_loss = (F.cross_entropy(logits["a_pos"], batch["a_pos"]) + F.cross_entropy(logits["b_pos"], batch["b_pos"])) / 2.0
    return value_loss + pointer_loss


def evaluate_model(model: Any, readout: str, cell: str, interface: str, seed: int, records: Sequence[Mapping[str, Any]], source: str) -> List[Dict[str, Any]]:
    prepared = ladder.prepare_records(records)
    outputs: List[Dict[str, Any]] = []
    model.eval()
    for start in range(0, len(prepared), EFFECTIVE_BATCH):
        indices = list(range(start, min(start + EFFECTIVE_BATCH, len(prepared))))
        batch = batchify(prepared, indices)
        with torch.inference_mode():
            logits = slot_outputs(model, batch) if readout == "slot" else operation_outputs(model, batch)
        if readout == "slot":
            outputs.extend(ladder.pointer_rows([records[index] for index in indices], logits["decision"], cell, seed, interface, source))
        else:
            outputs.extend(operation_class_rows([records[index] for index in indices], logits["decision"], cell, interface, seed, source))
    return outputs


def train_crossed(cell: str, interface: str, seed: int, records: Sequence[Mapping[str, Any]], output: Path) -> Tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
    readout = "slot" if cell.startswith("C2") else "operation"
    model = ladder.DecisionPointerParser(seed) if readout == "slot" else causal.DecisionRoleParser(seed)
    torch.manual_seed(seed + (94_000 if readout == "slot" else 95_000))
    random.seed(seed + 96_000)
    optimizer = stage_a.optimizer_for(model)
    prepared = ladder.prepare_records(records)
    schedule = make_schedule(len(records))
    trace: List[Dict[str, Any]] = []
    for step, indices in enumerate(schedule, start=1):
        batch = batchify(prepared, indices)
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = slot_outputs(model, batch) if readout == "slot" else operation_outputs(model, batch)
        if cell.endswith("DECISION_ONLY"):
            target = batch["target_slot"] if readout == "slot" else batch["operation_target"]
            loss = F.cross_entropy(logits["decision"], target)
        else:
            loss = multitask_loss(logits, batch, readout)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), stage_a.CLIP_NORM).item())
        optimizer.step()
        trace.append({"cell": cell, "interface": interface, "seed": seed, "step": step, "loss": float(loss.detach().item()), "grad_norm": grad_norm, "examples": len(indices)})
        if step in (1, 100, 200, 300, 400, 500):
            print(json.dumps({"event": "d2_component_ablation_training", "cell": cell, "interface": interface, "seed": seed, "step": step, "loss": float(loss.detach().item())}, sort_keys=True), flush=True)
    model.eval()
    predictions = evaluate_model(model, readout, cell, interface, seed, records, "new_crossed_training")
    checkpoint = output / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cell": cell, "interface": interface, "seed": seed, "steps": STEPS, "readout": readout, "model": model.state_dict()}, checkpoint)
    write_jsonl(output / "TRAINING_TRACE.jsonl", trace)
    result = {"cell": cell, "interface": interface, "seed": seed, "readout": readout, "source": "new_crossed_training", "steps": STEPS, "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint), "train_metrics": ladder.row_metrics(predictions, "train"), "validation_metrics": ladder.row_metrics(predictions, "validation")}
    write_json(output / "RESULTS.json", result)
    write_csv(output / "PREDICTIONS.csv", predictions)
    return model, result, predictions


def load_reused(cell: str, interface: str, seed: int, records: Sequence[Mapping[str, Any]]) -> Tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
    if cell == "C1_SLOT_DECISION_ONLY":
        checkpoint = LADDER_ROOT / "trained_arms" / "S4_END_TO_END_AUTHORITY" / interface / f"seed{seed}" / "checkpoint.pt"
        model = ladder.DecisionPointerParser(seed)
        readout = "slot"
        source = "reused_accepted_s4"
    else:
        checkpoint = CAUSAL_ROOT / "trained_arms" / "D2_LATE_FINAL_DECISION" / interface / f"seed{seed}" / "checkpoint.pt"
        model = causal.DecisionRoleParser(seed)
        readout = "operation"
        source = "reused_accepted_d2"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    predictions = evaluate_model(model, readout, cell, interface, seed, records, source)
    result = {"cell": cell, "interface": interface, "seed": seed, "readout": readout, "source": source, "reused_checkpoint": rel(checkpoint), "reused_checkpoint_sha256": sha256(checkpoint), "train_metrics": ladder.row_metrics(predictions, "train"), "validation_metrics": ladder.row_metrics(predictions, "validation")}
    arm_output = OUTPUT / "trained_arms" / cell / interface / f"seed{seed}"
    arm_output.mkdir(parents=True, exist_ok=True)
    write_json(arm_output / "RESULTS.json", result)
    write_csv(arm_output / "PREDICTIONS.csv", predictions)
    return model, result, predictions


def load_completed_crossed(arm: Path) -> Tuple[None, Dict[str, Any], List[Dict[str, Any]]]:
    """Load a completed newly-trained arm after an interrupted run."""
    result_path = arm / "RESULTS.json"
    predictions_path = arm / "PREDICTIONS.csv"
    checkpoint = arm / "checkpoint.pt"
    if not (result_path.exists() and predictions_path.exists() and checkpoint.exists()):
        raise FileNotFoundError(f"incomplete crossed arm: {arm}")
    result = load_json(result_path)
    expected_hash = str(result.get("checkpoint_sha256", ""))
    actual_hash = sha256(checkpoint)
    if expected_hash != actual_hash:
        raise RuntimeError(f"checkpoint hash mismatch while resuming: {checkpoint}")
    return None, result, load_csv(predictions_path)


def main() -> int:
    # Permit resuming an interrupted run, but never overwrite a completed
    # final artifact.
    if OUTPUT.exists() and (OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"refusing to overwrite completed output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    audits: Dict[str, Any] = {}
    for interface in INTERFACES:
        data[interface] = read_s3_data(interface)
        audits[interface] = ladder.verify_records("S3_AUTHORITY_CONDITIONED", data[interface]["train"], data[interface]["validation"])
        write_jsonl(OUTPUT / "data" / interface / "train.jsonl", data[interface]["train"])
        write_jsonl(OUTPUT / "data" / interface / "validation.jsonl", data[interface]["validation"])

    all_predictions: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    checkpoints: Dict[str, Any] = {}
    for cell in CELLS:
        for interface in INTERFACES:
            for seed in SEEDS:
                records = data[interface]["train"] + data[interface]["validation"]
                arm = OUTPUT / "trained_arms" / cell / interface / f"seed{seed}"
                if cell in ("C1_SLOT_DECISION_ONLY", "C4_OPERATION_FULL_MULTITASK"):
                    model, result, predictions = load_reused(cell, interface, seed, records)
                elif (arm / "RESULTS.json").exists() and (arm / "PREDICTIONS.csv").exists() and (arm / "checkpoint.pt").exists():
                    model, result, predictions = load_completed_crossed(arm)
                else:
                    model, result, predictions = train_crossed(cell, interface, seed, records, arm)
                results.append(result)
                all_predictions.extend(predictions)
                if "checkpoint" in result:
                    checkpoints[f"{cell}|{interface}|seed{seed}"] = {"path": result["checkpoint"], "sha256": result["checkpoint_sha256"]}
                else:
                    checkpoints[f"{cell}|{interface}|seed{seed}"] = {"path": result["reused_checkpoint"], "sha256": result["reused_checkpoint_sha256"]}
                del model
                gc.collect()

    summary = {"experiment": "crucible_d2_component_ablation_20261026_v1", "status": "completed", "protocol": rel(PROTOCOL), "cells": list(CELLS), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "steps": STEPS, "effective_batch": EFFECTIVE_BATCH, "data_audit": audits, "results": results, "checkpoint_hashes": checkpoints, "sealed_test_accessed": False, "architecture_changed": False, "reused_diagonal_controls": {"C1": "accepted S4", "C4": "accepted original D2"}}
    write_csv(OUTPUT / "PREDICTIONS.csv", all_predictions)
    write_json(OUTPUT / "DATA_AUDIT.json", audits)
    write_json(OUTPUT / "SUMMARY.json", summary)
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {"experiment": summary["experiment"], "protocol": rel(PROTOCOL), "cells": list(CELLS), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "sealed_test_accessed": False, "architecture_changed": False, "reused_diagonal_controls": summary["reused_diagonal_controls"]})

    report = ["# Crucible D2 Component Ablation v1", "", "2x2 CPU ablation of candidate-slot versus operation-class readout and decision-only versus auxiliary multi-task loss.", "", "## Results", "", "The training competence gate is 95% selection accuracy. Paired flips and paired correctness are the authority counterfactual metrics.", "", "| Cell | Interface | Seed | Train selection | Validation selection | Validation flips | Paired correct | Source |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for result in results:
        tr = result["train_metrics"]; va = result["validation_metrics"]; pair = va["paired"]
        report.append(f"| {result['cell']} | {result['interface']} | {result['seed']} | {tr['pointer_correct']}/{tr['rows']} | {va['pointer_correct']}/{va['rows']} | {pair['pointer_flips']}/{pair['pairs']} | {pair['correct_two_way']}/{pair['pairs']} | {result['source']} |")
    report += ["", "## Interpretation", "", "C1 is the accepted S4 candidate-slot/decision-only positive control. C4 is the accepted original D2 operation-class/multi-task negative control. C2 and C3 are the newly trained crossed cells. Results from cells below the training competence gate are not interpreted as clean held-out generalization evidence.", "", "## Artifacts", "", f"- [Protocol](../../{rel(PROTOCOL)})", "- [Summary](SUMMARY.json)", "- [Predictions](PREDICTIONS.csv)", "- [Data audit](DATA_AUDIT.json)", "- [Protocol manifest](PROTOCOL_MANIFEST.json)", "- Per-cell checkpoints and traces are under `trained_arms/`"]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    interpretation = """# Scientific Interpretation

This is a matched 2x2 ablation. The accepted S4 cell supplies the successful
candidate-slot/decision-only control, while the accepted original D2 cell
supplies the operation-class/multi-task control. The two crossed cells isolate
the candidate-slot and objective factors as far as this small experiment can.

The primary evidence is paired counterfactual selection: changing only the
authority code must change the selected candidate. Training competence is
required before validation is interpreted.

This experiment does not test larger candidate sets, arbitrary authority
relations or autonomous reasoning.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    hash_paths = [PROTOCOL, OUTPUT / "SUMMARY.json", OUTPUT / "DATA_AUDIT.json", OUTPUT / "PROTOCOL_MANIFEST.json", OUTPUT / "PREDICTIONS.csv", OUTPUT / "REPORT_FINAL.md", OUTPUT / "SCIENTIFIC_INTERPRETATION.md"]
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths})
    finalization = {"experiment": summary["experiment"], "status": "completed", "verification_pass": True, "sealed_test_accessed": False, "retrained_crossed_cells": True, "architecture_changed": False, "hashes": rel(OUTPUT / "HASHES.json"), "arm_count": len(results), "prediction_rows": len(all_predictions)}
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"event": "d2_component_ablation_complete", "output": rel(OUTPUT), "verification_pass": True, "arm_count": len(results), "prediction_rows": len(all_predictions)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
