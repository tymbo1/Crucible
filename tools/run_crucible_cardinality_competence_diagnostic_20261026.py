#!/usr/bin/env python3
"""Four-cell training-competence diagnostic for cardinality matching.

The experiment deliberately evaluates only the frozen canonical training
records from the prior cardinality run. It does not evaluate validation or
transfer data, access sealed tests, or modify the prior experiment.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUNNER = ROOT / "tools" / "run_crucible_shared_candidate_scoring_cardinality_20261026.py"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_CARDINALITY_COMPETENCE_DIAGNOSTIC_PROTOCOL_20261026.md"
NOVELTY = ROOT / "docs" / "CRUCIBLE_CARDINALITY_COMPETENCE_NOVELTY_AUDIT_20261026.json"
SOURCE_DATA = ROOT / "experiments" / "crucible_shared_candidate_scoring_cardinality_20261026_v1" / "data" / "canonical" / "train.jsonl"
OUTPUT = ROOT / "experiments" / "crucible_cardinality_competence_diagnostic_20261026_v1"

SEED = 42
BATCH_SIZE = 12
STEPS = 500
SHUFFLE_SEED = 20261026
TRAIN_THRESHOLD = 0.95
CELLS = (
    ("D0_DENSE_CYCLIC", "dense", "cyclic"),
    ("D1_DENSE_SHUFFLED", "dense", "shuffled"),
    ("D2_STRUCTURED_CYCLIC", "structured", "cyclic"),
    ("D3_STRUCTURED_SHUFFLED", "structured", "shuffled"),
)


def import_runner() -> Any:
    spec = importlib.util.spec_from_file_location("cardinality_runner_for_diagnostic", SOURCE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(SOURCE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(SOURCE_RUNNER)
    spec.loader.exec_module(module)
    return module


runner = import_runner()
torch = runner.torch
nn = torch.nn
F = runner.F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_records() -> List[Dict[str, Any]]:
    return [json.loads(line) for line in SOURCE_DATA.read_text(encoding="utf-8").splitlines() if line]


def schedule(size: int, mode: str) -> List[List[int]]:
    if mode == "cyclic":
        order = list(range(size))
    elif mode == "shuffled":
        order = list(range(size))
        random.Random(SHUFFLE_SEED).shuffle(order)
    else:
        raise ValueError(mode)
    return [[order[(step * BATCH_SIZE + offset) % size] for offset in range(BATCH_SIZE)] for step in range(STEPS)]


def oracle_target(row: Mapping[str, Any]) -> int:
    candidates = [int(value) for value in row["metadata"]["candidates"]]
    authority = int(row["metadata"]["authority_id"])
    return candidates.index(authority) if authority in candidates else 8


class StructuredSharedScorer(nn.Module):
    """Candidate-independent scorer over explicit non-answer features."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed + 91_000)
        # candidate id (0..9), query id (0..9), slot (0..7), count (2..8)
        feature_size = 10 + 10 + 8 + 7
        self.candidate = nn.Sequential(nn.Linear(feature_size, 64), nn.GELU(), nn.Linear(64, 1))
        self.abstain = nn.Sequential(nn.Linear(10 + 7, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        candidate_features = batch["candidate_features"]
        scores = self.candidate(candidate_features).squeeze(-1)
        scores = scores.masked_fill(~batch["candidate_mask"], -1e9)
        abstain_features = batch["abstain_features"]
        return torch.cat([scores, self.abstain(abstain_features)], dim=1)


def structured_batchify(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    rows = [records[index] for index in indices]
    candidate_features = torch.zeros((len(rows), 8, 35), dtype=torch.float32)
    candidate_mask = torch.zeros((len(rows), 8), dtype=torch.bool)
    abstain_features = torch.zeros((len(rows), 17), dtype=torch.float32)
    targets: List[int] = []
    for row_index, row in enumerate(rows):
        candidates = [int(value) for value in row["metadata"]["candidates"]]
        authority = int(row["metadata"]["authority_id"])
        count = len(candidates)
        query_slot = 10 + authority
        abstain_count_slot = 10 + (count - 2)
        abstain_features[row_index, authority] = 1.0
        abstain_features[row_index, abstain_count_slot] = 1.0
        for slot, candidate in enumerate(candidates):
            feature = candidate_features[row_index, slot]
            feature[candidate] = 1.0
            feature[10 + authority] = 1.0
            feature[20 + slot] = 1.0
            feature[28 + (count - 2)] = 1.0
            candidate_mask[row_index, slot] = True
        targets.append(int(row["metadata"]["target_slot"]))
    return {
        "candidate_features": candidate_features,
        "candidate_mask": candidate_mask,
        "abstain_features": abstain_features,
        "target": torch.tensor(targets, dtype=torch.long),
    }


def dense_batch(records: Sequence[Mapping[str, Any]], prepared: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    return runner.batchify(prepared, indices)


def row_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    correct = sum(int(row["pointer_correct"]) for row in rows)
    abstain = [row for row in rows if int(row["target_slot"]) == 8]
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["counterfactual_pair_id"]:
            groups[str(row["counterfactual_pair_id"])].append(row)
    complete = [values for values in groups.values() if len(values) == 2]
    return {
        "rows": len(rows),
        "pointer_correct": correct,
        "pointer_accuracy": correct / len(rows) if rows else None,
        "abstain_rows": len(abstain),
        "abstain_correct": sum(int(row["pointer_correct"]) for row in abstain),
        "abstain_accuracy": sum(int(row["pointer_correct"]) for row in abstain) / len(abstain) if abstain else None,
        "paired_groups": len(complete),
        "paired_correct": sum(int(values[0]["pointer_correct"] and values[1]["pointer_correct"]) for values in complete),
        "pointer_flips": sum(int(values[0]["predicted_slot"] != values[1]["predicted_slot"]) for values in complete),
    }


def evaluate(model: nn.Module, representation: str, records: Sequence[Mapping[str, Any]], prepared: Sequence[Mapping[str, Any]] | None) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(records), BATCH_SIZE):
            indices = list(range(start, min(start + BATCH_SIZE, len(records))))
            if representation == "dense":
                assert prepared is not None
                batch = dense_batch(records, prepared, indices)
                logits = model.decision_logits(batch)
            else:
                batch = structured_batchify(records, indices)
                logits = model(batch)
            predictions = logits.argmax(dim=-1).tolist()
            for local, index in enumerate(indices):
                item = records[index]
                target = int(item["metadata"]["target_slot"])
                rows.append({
                    "record_id": str(item["record_id"]),
                    "candidate_count": len(item["metadata"]["candidates"]),
                    "candidates": ",".join(str(value) for value in item["metadata"]["candidates"]),
                    "authority_id": int(item["metadata"]["authority_id"]),
                    "target_slot": target,
                    "predicted_slot": int(predictions[local]),
                    "pointer_correct": int(predictions[local]) == target,
                    "counterfactual_pair_id": item["metadata"]["counterfactual_pair_id"],
                })
    return rows


def model_for(representation: str) -> nn.Module:
    if representation == "dense":
        return runner.CandidateReadout(SEED, "SHARED_SCORER")
    return StructuredSharedScorer(SEED)


def train_cell(cell: str, representation: str, schedule_mode: str, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    model = model_for(representation)
    prepared = runner.prepare_records(records) if representation == "dense" else None
    optimizer = runner.stage_a.optimizer_for(model)
    batches = schedule(len(records), schedule_mode)
    trace: List[Dict[str, Any]] = []
    output = OUTPUT / cell
    output.mkdir(parents=True, exist_ok=True)
    for step, indices in enumerate(batches, start=1):
        model.train()
        batch = dense_batch(records, prepared, indices) if representation == "dense" else structured_batchify(records, indices)
        optimizer.zero_grad(set_to_none=True)
        logits = model.decision_logits(batch) if representation == "dense" else model(batch)
        loss = F.cross_entropy(logits, batch["target"])
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), runner.stage_a.CLIP_NORM).item())
        optimizer.step()
        entry: Dict[str, Any] = {"cell": cell, "step": step, "examples": len(indices), "loss": float(loss.detach().item()), "grad_norm_preclip": grad_norm}
        if step in (1, 50, 100, 200, 300, 400, 500):
            entry["full_training"] = row_metrics(evaluate(model, representation, records, prepared))
        trace.append(entry)
    final_rows = evaluate(model, representation, records, prepared)
    final_metrics = row_metrics(final_rows)
    checkpoint = output / "checkpoint.pt"
    torch.save({"cell": cell, "representation": representation, "schedule": schedule_mode, "seed": SEED, "model": model.state_dict()}, checkpoint)
    write_jsonl(output / "TRAINING_TRACE.jsonl", trace)
    write_csv(output / "PREDICTIONS.csv", final_rows)
    result = {
        "cell": cell,
        "representation": representation,
        "schedule": schedule_mode,
        "seed": SEED,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "source_rows": len(records),
        "train_metrics": final_metrics,
        "training_gate_threshold": TRAIN_THRESHOLD,
        "training_gate_pass": bool(final_metrics["pointer_accuracy"] >= TRAIN_THRESHOLD),
        "checkpoint": rel(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    write_json(output / "RESULTS.json", result)
    del optimizer, model
    return result


def main() -> int:
    if OUTPUT.exists() and (OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"refusing to overwrite completed output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    source_hash = sha256(SOURCE_DATA)
    if len(records) != 576:
        raise RuntimeError(f"expected 576 source rows, found {len(records)}")
    if any(int(row["metadata"]["target_slot"]) != oracle_target(row) for row in records):
        raise RuntimeError("source target mismatch")
    write_json(OUTPUT / "NOVELTY_AUDIT.json", json.loads(NOVELTY.read_text(encoding="utf-8")))
    write_json(OUTPUT / "DATA_AUDIT.json", {
        "source_data": rel(SOURCE_DATA),
        "source_sha256": source_hash,
        "rows": len(records),
        "unique_record_ids": len({row["record_id"] for row in records}),
        "unique_inputs": len({row["input"] for row in records}),
        "oracle_rows_correct": sum(int(int(row["metadata"]["target_slot"]) == oracle_target(row)) for row in records),
        "candidate_count_distribution": {str(key): value for key, value in sorted(__import__("collections").Counter(len(row["metadata"]["candidates"]) for row in records).items())},
        "target_slot_distribution": {str(key): value for key, value in sorted(__import__("collections").Counter(int(row["metadata"]["target_slot"]) for row in records).items())},
        "sealed_test_accessed": False,
    })
    results = [train_cell(cell, representation, schedule_mode, records) for cell, representation, schedule_mode in CELLS]
    summary = {
        "experiment": "crucible_cardinality_competence_diagnostic_20261026_v1",
        "status": "completed",
        "protocol": rel(PROTOCOL),
        "novelty_audit": rel(NOVELTY),
        "source_data": rel(SOURCE_DATA),
        "source_sha256": source_hash,
        "interface": "canonical",
        "seed": SEED,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "shuffle_seed": SHUFFLE_SEED,
        "training_gate_threshold": TRAIN_THRESHOLD,
        "sealed_test_accessed": False,
        "prior_experiment_modified": False,
        "results": results,
    }
    write_json(OUTPUT / "SUMMARY.json", summary)
    report = [
        "# Crucible Cardinality Competence Diagnostic v1",
        "",
        "Training-competence-only representation-by-schedule diagnostic on the frozen canonical cardinality training split.",
        "",
        "| Cell | Representation | Schedule | Train pointer | Paired groups | Gate |",
        "|---|---|---|---:|---:|---|",
    ]
    for result in results:
        metrics = result["train_metrics"]
        report.append(f"| {result['cell']} | {result['representation']} | {result['schedule']} | {metrics['pointer_correct']}/{metrics['rows']} ({metrics['pointer_accuracy']:.3f}) | {metrics['paired_correct']}/{metrics['paired_groups']} | {'PASS' if result['training_gate_pass'] else 'FAIL'} |")
    report += [
        "",
        "## Interpretation boundary",
        "",
        "This run evaluates only training competence. It does not test validation, cardinality transfer, candidate-count generalization or any sealed test set. A passed cell establishes that its representation and schedule can fit this training distribution; it does not establish transfer.",
        "",
        "## Artifacts",
        "",
        f"- [Protocol](../../{rel(PROTOCOL)})",
        f"- [Novelty audit](../../{rel(NOVELTY)})",
        "- [Summary](SUMMARY.json)",
        "- [Data audit](DATA_AUDIT.json)",
        "- Per-cell predictions, traces, results and checkpoints under the cell directories",
    ]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    interpretation = """# Scientific Interpretation

This diagnostic was designed to separate a representation burden from a
training-schedule contribution without reopening cardinality transfer. The
source split, labels, deterministic span metadata, one canonical interface,
one seed, optimizer family, update count and training gate were fixed before
execution. Structured cells use candidate/query/slot/count features but no
target or privileged equality label.

Only training competence is interpretable here. Validation and transfer were
intentionally not run. The full decision is recorded in REPORT_FINAL.md and
the machine-readable SUMMARY.json.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    hash_paths = [PROTOCOL, NOVELTY, SOURCE_DATA, OUTPUT / "SUMMARY.json", OUTPUT / "DATA_AUDIT.json", OUTPUT / "REPORT_FINAL.md", OUTPUT / "SCIENTIFIC_INTERPRETATION.md"]
    for cell, _, _ in CELLS:
        hash_paths.extend([OUTPUT / cell / "RESULTS.json", OUTPUT / cell / "PREDICTIONS.csv", OUTPUT / cell / "TRAINING_TRACE.jsonl", OUTPUT / cell / "checkpoint.pt"])
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths})
    write_json(OUTPUT / "FINALIZATION.json", {
        "experiment": summary["experiment"],
        "status": "completed",
        "verification_pass": True,
        "sealed_test_accessed": False,
        "prior_experiment_modified": False,
        "cell_count": len(results),
        "training_gate_passes": sum(int(result["training_gate_pass"]) for result in results),
        "hashes": rel(OUTPUT / "HASHES.json"),
    })
    print(json.dumps({"experiment": summary["experiment"], "cells": len(results), "gate_passes": sum(int(result["training_gate_pass"]) for result in results), "sealed_test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
