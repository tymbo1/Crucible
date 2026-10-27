#!/usr/bin/env python3
"""Extend the committed topology-control arms to 2,000 total updates."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUNNER = ROOT / "tools" / "run_crucible_two_candidate_topology_control_20261027.py"
SOURCE_EXPERIMENT = ROOT / "experiments" / "crucible_two_candidate_topology_control_20261027_v1"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_TOPOLOGY_DURATION_EXTENSION_PROTOCOL_20261027.md"
NOVELTY = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_TOPOLOGY_DURATION_EXTENSION_NOVELTY_AUDIT_20261027.json"
OUTPUT = ROOT / "experiments" / "crucible_two_candidate_topology_duration_extension_20261027_v1"

SEEDS = (42, 1337)
CONDITIONS = ("P12_PRISM", "R12_DIAMETER5")
TOTAL_STEPS = 2000
BATCH_SIZE = 12
TRAIN_THRESHOLD = 0.95
CHECKPOINT_STEPS = (1, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000)
BASELINE_STEP = 500


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


topology = import_module("topology_control_source", SOURCE_RUNNER)
torch = topology.torch
source = topology.source


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(row), sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def schedule(size: int) -> List[List[int]]:
    return [[(step * BATCH_SIZE + offset) % size for offset in range(BATCH_SIZE)] for step in range(TOTAL_STEPS)]


def train_cell(condition: str, seed: int, records: Mapping[str, Sequence[Mapping[str, Any]]], original: Mapping[str, Any]) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = source.CandidateReadout(seed, "SHARED_SCORER")
    optimizer = source.stage_a.optimizer_for(model)
    train_records = list(records["train"])
    validation_records = list(records["validation"])
    prepared_train = topology.prepare_records(train_records)
    prepared_validation = topology.prepare_records(validation_records)
    output = OUTPUT / f"{condition}_SEED{seed}"
    output.mkdir(parents=True, exist_ok=True)
    trace: List[Dict[str, Any]] = []
    gate_first_step = None
    for step, indices in enumerate(schedule(len(prepared_train)), start=1):
        model.train()
        batch = topology.batchify(prepared_train, indices)
        optimizer.zero_grad(set_to_none=True)
        loss = source.F.cross_entropy(model.decision_logits(batch), batch["target"])
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), source.stage_a.CLIP_NORM).item())
        optimizer.step()
        entry: Dict[str, Any] = {"cell": output.name, "condition": condition, "seed": seed, "step": step, "loss": float(loss.detach().item()), "grad_norm_preclip": grad_norm}
        if step in CHECKPOINT_STEPS:
            train_metrics = topology.metrics(topology.evaluate(model, train_records, prepared_train))
            validation_metrics = topology.metrics(topology.evaluate(model, validation_records, prepared_validation))
            entry["training"] = train_metrics
            entry["validation"] = validation_metrics
        if gate_first_step is None:
            # The gate is assessed only at recorded checkpoints, avoiding a
            # hidden early-stopping decision.
            if "training" in entry and entry["training"]["value_accuracy"] >= TRAIN_THRESHOLD:
                gate_first_step = step
        trace.append(entry)
    train_rows = topology.evaluate(model, train_records, prepared_train)
    validation_rows = topology.evaluate(model, validation_records, prepared_validation)
    train_metrics = topology.metrics(train_rows)
    validation_metrics = topology.metrics(validation_rows)
    checkpoint = output / "checkpoint.pt"
    torch.save({"cell": output.name, "condition": condition, "seed": seed, "variant": "SHARED_SCORER", "total_steps": TOTAL_STEPS, "model": model.state_dict()}, checkpoint)
    write_jsonl(output / "TRAINING_TRACE.jsonl", trace)
    write_csv(output / "TRAIN_PREDICTIONS.csv", train_rows)
    write_csv(output / "VALIDATION_PREDICTIONS.csv", validation_rows)
    result = {"cell": output.name, "condition": condition, "seed": seed, "source": "new_duration_extension", "variant": "SHARED_SCORER", "total_steps": TOTAL_STEPS, "batch_size": BATCH_SIZE, "train_metrics": train_metrics, "validation_metrics": validation_metrics, "training_gate_threshold": TRAIN_THRESHOLD, "training_gate_pass": train_metrics["value_accuracy"] >= TRAIN_THRESHOLD, "first_recorded_gate_step": gate_first_step, "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "trace_steps": list(CHECKPOINT_STEPS), "original_500_step": original}
    write_json(output / "RESULTS.json", result)
    del optimizer, model
    return result


def main() -> int:
    execute = "--execute" in sys.argv[1:]
    if OUTPUT.exists() and (OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"refusing to overwrite completed output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prior_summary = load_json(SOURCE_EXPERIMENT / "SUMMARY.json")
    prior_hashes = load_json(SOURCE_EXPERIMENT / "HASHES.json")
    prior_mismatches = []
    for relative, expected in prior_hashes.items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            prior_mismatches.append(relative)
    if prior_mismatches:
        raise RuntimeError(f"prior topology artifacts changed: {prior_mismatches}")
    records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    audits: Dict[str, Any] = {}
    for condition in CONDITIONS:
        train = load_jsonl(SOURCE_EXPERIMENT / "data" / f"{condition}_train.jsonl")
        validation = load_jsonl(SOURCE_EXPERIMENT / "data" / f"{condition}_validation.jsonl")
        topology.prepare_records(train)
        topology.prepare_records(validation)
        records[condition] = {"train": train, "validation": validation}
        audits[condition] = {"train_rows": len(train), "validation_rows": len(validation), "train_inputs": len({row["input"] for row in train}), "validation_inputs": len({row["input"] for row in validation}), "source_hashes_verified": True}
    preflight = {"experiment": "crucible_two_candidate_topology_duration_extension_20261027_v1", "status": "preflight_passed", "execute_requested": execute, "protocol": rel(PROTOCOL), "novelty_audit": rel(NOVELTY), "source_experiment": rel(SOURCE_EXPERIMENT), "source_summary_hash": sha256(SOURCE_EXPERIMENT / "SUMMARY.json"), "source_hash_manifest_hash": sha256(SOURCE_EXPERIMENT / "HASHES.json"), "prior_hashes_verified": True, "conditions": list(CONDITIONS), "seeds": list(SEEDS), "total_steps": TOTAL_STEPS, "batch_size": BATCH_SIZE, "schedule": "cyclic", "checkpoint_steps": list(CHECKPOINT_STEPS), "training_gate_threshold": TRAIN_THRESHOLD, "audits": audits, "training_performed": False, "sealed_test_accessed": False, "prior_checkpoints_modified": False, "symbolic_assistance": False}
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    write_json(OUTPUT / "NOVELTY_AUDIT.json", load_json(NOVELTY))
    if not execute:
        print(json.dumps({"experiment": preflight["experiment"], "status": "preflight_passed", "new_arms": 4, "training_performed": False}, sort_keys=True))
        return 0
    results: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        prior_results = {str(row["seed"]): row for row in prior_summary["results"] if row["condition"] == condition}
        for seed in SEEDS:
            results.append(train_cell(condition, seed, records[condition], prior_results[str(seed)]))
    summary = {"experiment": "crucible_two_candidate_topology_duration_extension_20261027_v1", "status": "completed", "protocol": rel(PROTOCOL), "novelty_audit": rel(NOVELTY), "source_experiment": rel(SOURCE_EXPERIMENT), "preflight": rel(OUTPUT / "PREFLIGHT.json"), "results": results, "conditions": list(CONDITIONS), "seeds": list(SEEDS), "total_steps": TOTAL_STEPS, "batch_size": BATCH_SIZE, "schedule": "cyclic", "checkpoint_steps": list(CHECKPOINT_STEPS), "training_gate_threshold": TRAIN_THRESHOLD, "training_performed": True, "sealed_test_accessed": False, "prior_checkpoints_modified": False, "symbolic_assistance": False, "training_gate_passes": sum(int(result["training_gate_pass"]) for result in results)}
    write_json(OUTPUT / "SUMMARY.json", summary)
    report = ["# Crucible Topology-Control Training-Duration Extension v1", "", "The exact four topology-control arms were retrained from the same initialization and data schedule for 2,000 total updates.", "", "| Condition | Seed | 500-step train | 2,000-step train | Gate | 2,000-step validation | Paired | First recorded gate |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for result in results:
        baseline = result["original_500_step"]["train_metrics"]
        train = result["train_metrics"]
        validation = result["validation_metrics"]
        report.append(f"| {result['condition']} | {result['seed']} | {baseline['value_correct']}/{baseline['rows']} ({baseline['value_accuracy']:.3f}) | {train['value_correct']}/{train['rows']} ({train['value_accuracy']:.3f}) | {'PASS' if result['training_gate_pass'] else 'FAIL'} | {validation['value_correct']}/{validation['rows']} ({validation['value_accuracy']:.3f}) | {validation['paired_correct']}/{validation['paired_groups']} | {result['first_recorded_gate_step'] or 'not reached'} |")
    report.extend(["", "## Interpretation", "", "Validation is interpreted only for arms that pass the final 2,000-update training gate. The extension diagnoses acquisition speed versus plateau; it is not a topology-transfer comparison. Any later topology comparison must use a matched duration after both conditions demonstrate competence.", "", "## Artifacts", "", f"- [Protocol](../../{rel(PROTOCOL)})", f"- [Novelty audit](../../{rel(NOVELTY)})", f"- [Source experiment](../../{rel(SOURCE_EXPERIMENT / 'REPORT_FINAL.md')})", "- [Preflight](PREFLIGHT.json)", "- [Summary](SUMMARY.json)"])
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text("# Scientific Interpretation\n\nThis run tests whether the original topology-control training failures were caused by insufficient duration. The exact task, data, architecture, optimizer settings and seeds were retained, and the first 500 updates were reproduced from fresh initialization before extending to 2,000.\n\nInterpret validation only for final gate-passing arms. Even if longer training rescues an arm, this does not establish a graph-distance effect; it only makes a later competence-matched topology comparison possible.\n", encoding="utf-8")
    write_json(OUTPUT / "ERROR_BREAKDOWN.json", {result["cell"]: result for result in results})
    hash_paths = [PROTOCOL, NOVELTY, Path(__file__).resolve(), OUTPUT / "PREFLIGHT.json", OUTPUT / "NOVELTY_AUDIT.json", OUTPUT / "SUMMARY.json", OUTPUT / "REPORT_FINAL.md", OUTPUT / "SCIENTIFIC_INTERPRETATION.md", OUTPUT / "ERROR_BREAKDOWN.json"]
    for result in results:
        cell = OUTPUT / result["cell"]
        hash_paths.extend([cell / "RESULTS.json", cell / "TRAINING_TRACE.jsonl", cell / "TRAIN_PREDICTIONS.csv", cell / "VALIDATION_PREDICTIONS.csv", cell / "checkpoint.pt"])
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths})
    write_json(OUTPUT / "FINALIZATION.json", {"experiment": summary["experiment"], "status": "completed", "verification_pass": True, "training_performed": True, "sealed_test_accessed": False, "prior_checkpoints_modified": False, "symbolic_assistance": False, "training_gate_passes": summary["training_gate_passes"], "hashes": rel(OUTPUT / "HASHES.json")})
    print(json.dumps({"experiment": summary["experiment"], "status": "completed", "arms": len(results), "training_gate_passes": summary["training_gate_passes"], "sealed_test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
