#!/usr/bin/env python3
"""Run the narrow operation-token identification control for Track A."""

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
OUTPUT = ROOT / "experiments" / "crucible_operation_token_identification_20251025_v1"
DIAGONAL_SOURCE = ROOT / "tools" / "run_crucible_digitwise_diagonal_ablation_20251025.py"
ROLE_SOURCE = ROOT / "tools" / "run_crucible_role_selection_competence_20251025.py"
INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
STEPS = 500
TRAIN_THRESHOLD = 0.95
VALIDATION_THRESHOLD = 0.90
CURVE_STEPS = (1, 10, 25, 50, 100, 200, 300, 400, 500)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


base = import_module("operation_token_diagonal_runner", DIAGONAL_SOURCE)
role = import_module("operation_token_role_split", ROLE_SOURCE)
ladder = base.ladder
stage_a = base.stage_a
torch = base.torch
TRAIN_SET = set(role.TRAIN_SET)
VALIDATION_SET = set(role.VALIDATION_SET)


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


def prepare_fresh_anchors(anchor_dir: Path) -> Dict[str, str]:
    anchor_dir.mkdir(parents=True, exist_ok=True)
    ladder.stage_a.E4_ANCHOR_DIR = anchor_dir
    base.stage_a.E4_ANCHOR_DIR = anchor_dir
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
        path = anchor_dir / f"initial_seed_anchor_{seed}.pt"
        torch.save(model.state_dict(), path)
        hashes[str(seed)] = sha256(path)
        del model
    return hashes


def make_records(interface: str, split: str, pairs: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    index = 0
    for a, b in pairs:
        for operation, label in (("SELECT_FIRST", 0), ("SELECT_SECOND", 1)):
            if interface == "canonical":
                prompt = f"Read the requested operation. operation={operation}; a={a}; b={b}. Return 0 for SELECT_FIRST and 1 for SELECT_SECOND."
            else:
                prompt = f"Read the requested operation. operation={operation}; a_digit={a}; b_digit={b}. Return 0 for SELECT_FIRST and 1 for SELECT_SECOND."
            rows.append({
                "schema_version": "crucible_operation_token_identification_v1",
                "record_id": hashlib.sha256(f"operation_token|{interface}|{split}|{index}|{a}|{b}|{operation}".encode()).hexdigest()[:20],
                "interface": interface,
                "stage": "operation_token_identification",
                "split": split,
                "input": prompt,
                "target": str(label),
                "oracle": {"kind": "operation_identifier", "operation": operation, "value": label},
                "metadata": {
                    "a": a,
                    "b": b,
                    "operation": operation,
                    "label": label,
                    "pair_kind": "equal" if a == b else "unequal",
                    "output_length": 1,
                },
            })
            index += 1
    return rows


def audit(records: Mapping[str, Sequence[Mapping[str, Any]]], interface: str) -> List[str]:
    errors: List[str] = []
    train = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in records["train"]}
    validation = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in records["validation"]}
    if train != TRAIN_SET or validation != VALIDATION_SET or train & validation:
        errors.append(f"pair_split:{interface}")
    if len(records["train"]) != 80 or len(records["validation"]) != 20:
        errors.append(f"record_count:{interface}")
    for split in ("train", "validation"):
        for row in records[split]:
            operation = row["metadata"]["operation"]
            expected = 0 if operation == "SELECT_FIRST" else 1 if operation == "SELECT_SECOND" else None
            if expected is None or int(row["target"]) != expected or int(row["oracle"]["value"]) != expected:
                errors.append(f"label:{interface}:{row['record_id']}")
            if row.get("interface") != interface:
                errors.append(f"interface:{interface}:{row['record_id']}")
    return errors


def evaluate(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prepared = ladder.prepare(records)
    return stage_a.evaluate(model, prepared)


def train_model(interface: str, seed: int, records: Mapping[str, Sequence[Mapping[str, Any]]], output: Path) -> Dict[str, Any]:
    stage_a.torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    prepared = {split: ladder.prepare(values) for split, values in records.items()}
    order = list(range(len(prepared["train"])))
    random.Random(ladder.stable_seed(seed, interface, "operation_token", "order")).shuffle(order)
    curves = []
    interval_loss = 0.0
    interval_steps = 0
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        result = stage_a.train_step(model, optimizer, prepared["train"], stage_a.batch_indices(order, step))
        interval_loss += float(result["loss"])
        interval_steps += 1
        if step in CURVE_STEPS:
            curves.append({
                "step": step,
                "mean_loss_since_previous_curve": interval_loss / interval_steps,
                "train": evaluate(model, records["train"]),
                "validation": evaluate(model, records["validation"]),
            })
            interval_loss = 0.0
            interval_steps = 0
    train_metrics = evaluate(model, records["train"])
    validation_metrics = evaluate(model, records["validation"])
    checkpoint = output / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint, model, optimizer, STEPS, order)
    result = {
        "interface": interface,
        "seed": seed,
        "steps": STEPS,
        "elapsed_seconds": time.perf_counter() - started,
        "train": train_metrics,
        "validation": validation_metrics,
        "train_exact": train_metrics["teacher_forced_exact_accuracy"],
        "validation_exact": validation_metrics["teacher_forced_exact_accuracy"],
        "fit_pass": train_metrics["teacher_forced_exact_accuracy"] >= TRAIN_THRESHOLD,
        "validation_pass": validation_metrics["teacher_forced_exact_accuracy"] >= VALIDATION_THRESHOLD,
        "checkpoint": str(checkpoint),
    }
    result["gate_pass"] = bool(result["fit_pass"] and result["validation_pass"])
    write_json(output / "RESULTS.json", result)
    write_json(output / "TRAINING_CURVE.json", {"curve_steps": list(CURVE_STEPS), "points": curves})
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


def predict(model: Any, records: Sequence[Mapping[str, Any]], arm: Mapping[str, Any]) -> List[Dict[str, Any]]:
    prepared = ladder.prepare(records)
    rows = []
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
                predicted, format_error = decode_output(predicted_id)
                target_id = int(batch["y"][local, output_position].item())
                predicted_eos = int(logits[local, eos_position].argmax().item())
                target_eos = int(batch["y"][local, eos_position].item())
                token_correct = predicted_id == target_id
                eos_correct = predicted_eos == target_eos
                rows.append({
                    **arm,
                    "record_id": item["record_id"],
                    "split": item["split"],
                    "a": int(item["metadata"]["a"]),
                    "b": int(item["metadata"]["b"]),
                    "operation": item["metadata"]["operation"],
                    "pair_kind": item["metadata"]["pair_kind"],
                    "target_output": str(item["target"]),
                    "predicted_output": predicted,
                    "output_token_correct": token_correct,
                    "eos_correct": eos_correct,
                    "full_sequence_correct": token_correct and eos_correct,
                    "output_format_error": format_error,
                })
    return rows


def stat(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row["output_token_correct"]) for row in rows)
    return {"observations": n, "correct": correct, "accuracy": correct / n if n else None, "format_errors": sum(bool(row["output_format_error"]) for row in rows), "eos_correct": sum(bool(row["eos_correct"]) for row in rows)}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = {interface: {"train": make_records(interface, "train", sorted(TRAIN_SET)), "validation": make_records(interface, "validation", sorted(VALIDATION_SET))} for interface in INTERFACES}
    errors = [error for interface in INTERFACES for error in audit(records[interface], interface)]
    preflight = {
        "experiment": "crucible_operation_token_identification_20251025",
        "status": "frozen_before_training",
        "interfaces": list(INTERFACES),
        "seeds": list(SEEDS),
        "train_pairs": [list(pair) for pair in sorted(TRAIN_SET)],
        "validation_pairs": [list(pair) for pair in sorted(VALIDATION_SET)],
        "steps": STEPS,
        "curve_steps": list(CURVE_STEPS),
        "sealed_test_accessed": False,
        "architecture_changed": False,
        "errors": errors,
        "pass": not errors,
    }
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    if errors:
        raise SystemExit(f"preflight failed: {json.dumps(preflight, sort_keys=True)}")
    for interface in INTERFACES:
        path = OUTPUT / "data" / interface
        write_jsonl(path / "train.jsonl", records[interface]["train"])
        write_jsonl(path / "validation.jsonl", records[interface]["validation"])
    anchor_dir = OUTPUT / "generated_reference_e4" / "crucible_manifest"
    anchor_hashes = prepare_fresh_anchors(anchor_dir)
    write_json(OUTPUT / "INITIAL_ANCHORS.json", {"directory": rel(anchor_dir), "seeds": list(SEEDS), "sha256": anchor_hashes, "generated_without_training": True})
    preflight["initial_anchor_sha256"] = anchor_hashes
    write_json(OUTPUT / "PREFLIGHT.json", preflight)

    arms = []
    all_rows = []
    for interface in INTERFACES:
        for seed in SEEDS:
            arm_id = f"operation_token_{interface}_seed{seed}"
            output = OUTPUT / "arm_outputs" / interface / f"seed{seed}"
            result = train_model(interface, seed, records[interface], output)
            checkpoint = output / "checkpoint.pt"
            result.update({"arm_id": arm_id, "interface": interface, "seed": seed, "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint)})
            write_json(output / "RESULTS.json", result)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = stage_a.make_dense(seed)
            model.load_state_dict(state["model"], strict=True)
            arm = {"arm_id": arm_id, "interface": interface, "seed": seed, "checkpoint": rel(checkpoint), "checkpoint_sha256": result["checkpoint_sha256"]}
            rows = predict(model, records[interface]["train"] + records[interface]["validation"], arm)
            all_rows.extend(rows)
            write_json(output / "FORENSIC.json", {"results": result, "rows": rows})
            arms.append({"arm_id": arm_id, "interface": interface, "seed": seed, "train": stat([row for row in rows if row["split"] == "train"]), "validation": stat([row for row in rows if row["split"] == "validation"]), "fit_pass": result["fit_pass"], "validation_pass": result["validation_pass"], "gate_pass": result["gate_pass"], "checkpoint": result["checkpoint"], "checkpoint_sha256": result["checkpoint_sha256"]})
            del model, state
            gc.collect()

    write_csv(OUTPUT / "PREDICTIONS.csv", all_rows)
    summary = {"experiment": "crucible_operation_token_identification_20251025", "status": "completed", "arms": arms, "prediction_rows": len(all_rows), "sealed_test_accessed": False, "architecture_changed": False}
    write_json(OUTPUT / "SUMMARY.json", summary)
    errors = []
    if len(all_rows) != 400:
        errors.append("prediction_count")
    if any(bool(row["output_format_error"]) for row in all_rows):
        errors.append("format_errors")
    analysis = {"experiment": summary["experiment"], "total_arms": len(arms), "prediction_rows": len(all_rows), "all_gates_pass": all(arm["gate_pass"] for arm in arms), "arms": arms, "verification": {"pass": not errors and preflight["pass"], "errors": errors, "sealed_test_accessed": False, "architecture_changed": False}}
    write_json(OUTPUT / "ANALYSIS_FINAL.json", analysis)
    report = [
        "# Crucible Operation-Token Identification — Final Analysis",
        "",
        "CPU-only Track A control using the frozen 22M dense architecture, seeds 42/1337, both interfaces, and no sealed-test access.",
        "",
        f"All gates: {'PASS' if analysis['all_gates_pass'] else 'FAIL'}. Verification: {'PASS' if analysis['verification']['pass'] else 'FAIL'}.",
        "",
        "| Interface | Seed | Train | Validation | Gate |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in arms:
        report.append(f"| {arm['interface']} | {arm['seed']} | {arm['train']['correct']}/{arm['train']['observations']} | {arm['validation']['correct']}/{arm['validation']['observations']} | {'PASS' if arm['gate_pass'] else 'FAIL'} |")
    report += [
        "",
        "The target is the operation identifier class, not an operand. Passing this control supports recognition of the explicit identifier but does not establish operation-conditioned execution.",
        "",
        "- [Protocol](../../docs/CRUCIBLE_OPERATION_TOKEN_IDENTIFICATION_PROTOCOL_20251025.md)",
        "- [Analysis](ANALYSIS_FINAL.json)",
        "- [Predictions](PREDICTIONS.csv)",
        "- [Training curves](arm_outputs/)",
    ]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    preflight["status"] = "completed"
    preflight["training_performed"] = True
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    hashes = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt":
            hashes[rel(path)] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    write_json(OUTPUT / "FINALIZATION.json", {"status": "completed", "verification_pass": analysis["verification"]["pass"], "all_gates_pass": analysis["all_gates_pass"], "analysis_sha256": sha256(OUTPUT / "ANALYSIS_FINAL.json"), "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"), "prediction_sha256": sha256(OUTPUT / "PREDICTIONS.csv")})
    print(json.dumps({"status": "completed", "arms": len(arms), "all_gates_pass": analysis["all_gates_pass"], "output": rel(OUTPUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
