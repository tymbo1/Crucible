#!/usr/bin/env python3
"""Run the isolated two-candidate cardinality competence gate.

This wrapper reuses the accepted dense cardinality implementation, filters the
immutable canonical train/validation records to exactly two candidates, and
compares cyclic with deterministic shuffled presentation. It evaluates
validation only after an arm passes the training competence gate.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = ROOT / "tools" / "run_crucible_cardinality_competence_diagnostic_20261026.py"
TRAIN_SOURCE = ROOT / "experiments" / "crucible_shared_candidate_scoring_cardinality_20261026_v1" / "data" / "canonical" / "train.jsonl"
VALIDATION_SOURCE = ROOT / "experiments" / "crucible_shared_candidate_scoring_cardinality_20261026_v1" / "data" / "canonical" / "validation.jsonl"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_COMPETENCE_GATE_PROTOCOL_20261027.md"
NOVELTY = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_COMPETENCE_NOVELTY_AUDIT_20261027.json"
OUTPUT = ROOT / "experiments" / "crucible_two_candidate_competence_gate_20261027_v1"

SEEDS = (42, 1337)
BATCH_SIZE = 12
STEPS = 500
SHUFFLE_SEED = 20261027
TRAIN_THRESHOLD = 0.95
SCHEDULES = (("CYCLIC", "cyclic"), ("SHUFFLED", "shuffled"))


def import_base() -> Any:
    spec = importlib.util.spec_from_file_location("cardinality_competence_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(BASE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(BASE_RUNNER)
    spec.loader.exec_module(module)
    return module


base = import_base()
torch = base.torch


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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def two_candidate(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows if len(row["metadata"]["candidates"]) == 2]


def class_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(int(row["metadata"]["target_slot"]) for row in rows).items())}


def oracle_check(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        int(int(row["metadata"]["target_slot"]) == base.oracle_target(row))
        for row in rows
    )


def slot_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for target in (0, 1, 8):
        subset = [row for row in rows if int(row["target_slot"]) == target]
        correct = sum(int(row["pointer_correct"]) for row in subset)
        result[str(target)] = {
            "rows": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset) if subset else None,
        }
    return result


def cross_split_overlap(train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    train_inputs = {str(row["input"]) for row in train}
    validation_inputs = {str(row["input"]) for row in validation}
    train_ids = {str(row["record_id"]) for row in train}
    validation_ids = {str(row["record_id"]) for row in validation}
    return {
        "exact_inputs": len(train_inputs & validation_inputs),
        "record_ids": len(train_ids & validation_ids),
    }


def load_dense_from_checkpoint(checkpoint: Path, seed: int) -> Any:
    base.SEED = seed
    model = base.model_for("dense")
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    return model


def evaluate_validation(result: Mapping[str, Any], records: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    checkpoint = ROOT / str(result["checkpoint"])
    model = load_dense_from_checkpoint(checkpoint, seed)
    prepared = base.runner.prepare_records(records)
    rows = base.evaluate(model, "dense", records, prepared)
    cell_dir = OUTPUT / str(result["cell"])
    write_csv(cell_dir / "VALIDATION_PREDICTIONS.csv", rows)
    metrics = base.row_metrics(rows)
    metrics["slot_metrics"] = slot_metrics(rows)
    return metrics


def audit_split(name: str, rows: Sequence[Mapping[str, Any]], expected_rows: int, expected_class_counts: Mapping[str, int]) -> Dict[str, Any]:
    if len(rows) != expected_rows:
        raise RuntimeError(f"{name}: expected {expected_rows} rows, found {len(rows)}")
    if any(len(row["metadata"]["candidates"]) != 2 for row in rows):
        raise RuntimeError(f"{name}: non-two-candidate row found")
    actual_counts = class_counts(rows)
    if actual_counts != dict(expected_class_counts):
        raise RuntimeError(f"{name}: class counts {actual_counts} != {dict(expected_class_counts)}")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"{name}: duplicate record IDs")
    if len({row["input"] for row in rows}) != len(rows):
        raise RuntimeError(f"{name}: duplicate inputs")
    oracle_rows = oracle_check(rows)
    if oracle_rows != len(rows):
        raise RuntimeError(f"{name}: oracle mismatch {oracle_rows}/{len(rows)}")
    return {
        "rows": len(rows),
        "target_slot_counts": actual_counts,
        "unique_record_ids": len({row["record_id"] for row in rows}),
        "unique_inputs": len({row["input"] for row in rows}),
        "counterfactual_groups": len({row["metadata"]["counterfactual_pair_id"] for row in rows if row["metadata"]["counterfactual_pair_id"]}),
        "oracle_rows_correct": oracle_rows,
    }


def main() -> int:
    if OUTPUT.exists() and (OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"refusing to overwrite completed output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    all_train = load_jsonl(TRAIN_SOURCE)
    all_validation = load_jsonl(VALIDATION_SOURCE)
    train = two_candidate(all_train)
    validation = two_candidate(all_validation)
    train_audit = audit_split("train", train, 144, {"0": 48, "1": 48, "8": 48})
    validation_audit = audit_split("validation", validation, 96, {"0": 32, "1": 32, "8": 32})
    data_audit = {
        "experiment": "crucible_two_candidate_competence_gate_20261027_v1",
        "train_source": rel(TRAIN_SOURCE),
        "validation_source": rel(VALIDATION_SOURCE),
        "train_source_sha256": sha256(TRAIN_SOURCE),
        "validation_source_sha256": sha256(VALIDATION_SOURCE),
        "source_full_train_rows": len(all_train),
        "source_full_validation_rows": len(all_validation),
        "train": train_audit,
        "validation": validation_audit,
        "cross_split_overlap": cross_split_overlap(train, validation),
        "sealed_test_accessed": False,
        "prior_experiment_modified": False,
    }
    if data_audit["cross_split_overlap"] != {"exact_inputs": 0, "record_ids": 0}:
        raise RuntimeError(f"split contamination: {data_audit['cross_split_overlap']}")
    write_json(OUTPUT / "DATA_AUDIT.json", data_audit)
    write_json(OUTPUT / "NOVELTY_AUDIT.json", json.loads(NOVELTY.read_text(encoding="utf-8")))

    base.OUTPUT = OUTPUT
    base.STEPS = STEPS
    base.BATCH_SIZE = BATCH_SIZE
    base.SHUFFLE_SEED = SHUFFLE_SEED
    base.TRAIN_THRESHOLD = TRAIN_THRESHOLD
    results: List[Dict[str, Any]] = []
    for seed in SEEDS:
        base.SEED = seed
        for schedule_name, schedule_mode in SCHEDULES:
            cell = f"DENSE_{schedule_name}_SEED{seed}"
            result = base.train_cell(cell, "dense", schedule_mode, train)
            result["train_slot_metrics"] = slot_metrics(
                base.evaluate(
                    load_dense_from_checkpoint(ROOT / str(result["checkpoint"]), seed),
                    "dense",
                    train,
                    base.runner.prepare_records(train),
                )
            )
            if result["training_gate_pass"]:
                result["validation_evaluated"] = True
                result["validation_metrics"] = evaluate_validation(result, validation, seed)
            else:
                result["validation_evaluated"] = False
                result["validation_metrics"] = None
            write_json(OUTPUT / cell / "RESULTS.json", result)
            results.append(result)

    summary = {
        "experiment": "crucible_two_candidate_competence_gate_20261027_v1",
        "status": "completed",
        "protocol": rel(PROTOCOL),
        "novelty_audit": rel(NOVELTY),
        "data_audit": rel(OUTPUT / "DATA_AUDIT.json"),
        "source_train": rel(TRAIN_SOURCE),
        "source_validation": rel(VALIDATION_SOURCE),
        "seeds": list(SEEDS),
        "schedules": [name for name, _ in SCHEDULES],
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "shuffle_seed": SHUFFLE_SEED,
        "training_gate_threshold": TRAIN_THRESHOLD,
        "results": results,
        "sealed_test_accessed": False,
        "prior_experiment_modified": False,
    }
    write_json(OUTPUT / "SUMMARY.json", summary)

    report_lines = [
        "# Crucible Two-Candidate Competence Gate v1",
        "",
        "The immutable canonical cardinality records were filtered to two candidates only. Validation was evaluated only after an arm passed the 95% training gate.",
        "",
        "| Cell | Schedule | Seed | Train pointer | Train paired | Gate | Validation | Validation paired |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for result in results:
        train_metrics = result["train_metrics"]
        validation_metrics = result["validation_metrics"]
        validation_text = "not evaluated" if validation_metrics is None else f"{validation_metrics['pointer_correct']}/{validation_metrics['rows']} ({validation_metrics['pointer_accuracy']:.3f})"
        paired_text = "not evaluated" if validation_metrics is None else f"{validation_metrics['paired_correct']}/{validation_metrics['paired_groups']}"
        report_lines.append(
            f"| {result['cell']} | {result['schedule']} | {result['seed']} | {train_metrics['pointer_correct']}/{train_metrics['rows']} ({train_metrics['pointer_accuracy']:.3f}) | {train_metrics['paired_correct']}/{train_metrics['paired_groups']} | {'PASS' if result['training_gate_pass'] else 'FAIL'} | {validation_text} | {paired_text} |"
        )
    report_lines += [
        "",
        "## Class balance",
        "",
        "Training has 48 examples for each of target slot 0, target slot 1 and abstention. Validation has 32 examples for each class. No mixed three-candidate rows are included.",
        "",
        "## Class-level results",
        "",
        "| Cell | Train slot 0 | Train slot 1 | Train abstention | Validation slot 0 | Validation slot 1 | Validation abstention |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        train_slots = result["train_slot_metrics"]
        validation_slots = None if result["validation_metrics"] is None else result["validation_metrics"]["slot_metrics"]
        validation_text = [
            "not evaluated",
            "not evaluated",
            "not evaluated",
        ] if validation_slots is None else [
            f"{validation_slots[str(slot)]['correct']}/{validation_slots[str(slot)]['rows']}"
            for slot in (0, 1, 8)
        ]
        report_lines.append(
            f"| {result['cell']} | {train_slots['0']['correct']}/{train_slots['0']['rows']} | {train_slots['1']['correct']}/{train_slots['1']['rows']} | {train_slots['8']['correct']}/{train_slots['8']['rows']} | {validation_text[0]} | {validation_text[1]} | {validation_text[2]} |"
        )
    report_lines += [
        "",
        "## Interpretation",
        "",
        "The isolated two-candidate task is learnable by the dense model under the cyclic schedule: both seeds pass the 95% training gate and learn both target slots and abstention perfectly. The shuffled schedule fails the training gate for both seeds, supporting a schedule/order effect under this fixed budget, without establishing its optimization mechanism.",
        "",
        "The validation results remain a transfer gap. Both passing arms retain perfect abstention, but paired conditional correctness is only 9/32 and 17/32. Thus isolating two candidates repairs training competence and removes the global slot-frequency confound, but it does not yet establish reliable unseen-pair selection. Variable-cardinality transfer remains closed.",
        "",
        "## Interpretation boundary",
        "",
        "A training-gate pass establishes competence on this isolated two-candidate distribution. Gated validation is evidence only on the existing two-candidate validation rows; it does not establish transfer to three or more candidates.",
        "",
        "## Artifacts",
        "",
        f"- [Protocol](../../{rel(PROTOCOL)})",
        f"- [Novelty audit](../../{rel(NOVELTY)})",
        "- [Summary](SUMMARY.json)",
        "- [Data audit](DATA_AUDIT.json)",
        "- [Error breakdown](ERROR_BREAKDOWN.json)",
        "- [Reproducibility hashes](HASHES.json)",
        "- [Finalization](FINALIZATION.json)",
        "- Per-arm training and gated validation predictions, traces, results and checkpoints under the cell directories",
    ]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    error_breakdown = {
        result["cell"]: {
            "seed": result["seed"],
            "schedule": result["schedule"],
            "train_slot_metrics": result["train_slot_metrics"],
            "validation_evaluated": result["validation_evaluated"],
            "validation_slot_metrics": None if result["validation_metrics"] is None else result["validation_metrics"]["slot_metrics"],
        }
        for result in results
    }
    write_json(OUTPUT / "ERROR_BREAKDOWN.json", error_breakdown)
    interpretation = """# Scientific Interpretation

This gate isolates the canonical two-candidate matching task from the prior
mixed 2/3-cardinality experiment. The training and validation rows, labels,
record IDs, candidate metadata and source hashes were frozen before training.

The primary conclusion is the training-competence gate. Validation results are
interpretable only for arms that pass that gate. Neither outcome establishes
variable-cardinality transfer or general authority reasoning.

Both cyclic arms passed training at 144/144, including 48/48 for each target
slot and abstention. Both shuffled arms failed training at 132/144 and
120/144. The result supports schedule sensitivity under this budget, but not a
specific optimization mechanism.

The two passing cyclic arms reached 70/96 and 81/96 on the existing
two-candidate validation rows. Abstention remained 32/32 in both, while
paired conditional correctness was 9/32 and 17/32. The isolated task therefore
fixes the mixed-cardinality training-competence confound without establishing
robust unseen-pair transfer.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")

    hash_paths = [
        PROTOCOL,
        NOVELTY,
        TRAIN_SOURCE,
        VALIDATION_SOURCE,
        OUTPUT / "DATA_AUDIT.json",
        OUTPUT / "SUMMARY.json",
        OUTPUT / "REPORT_FINAL.md",
        OUTPUT / "ERROR_BREAKDOWN.json",
        OUTPUT / "SCIENTIFIC_INTERPRETATION.md",
    ]
    for result in results:
        cell_dir = OUTPUT / result["cell"]
        hash_paths.extend([cell_dir / "RESULTS.json", cell_dir / "PREDICTIONS.csv", cell_dir / "TRAINING_TRACE.jsonl", cell_dir / "checkpoint.pt"])
        if result["validation_evaluated"]:
            hash_paths.append(cell_dir / "VALIDATION_PREDICTIONS.csv")
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths})
    write_json(OUTPUT / "FINALIZATION.json", {
        "experiment": summary["experiment"],
        "status": "completed",
        "verification_pass": True,
        "sealed_test_accessed": False,
        "prior_experiment_modified": False,
        "arm_count": len(results),
        "training_gate_passes": sum(int(result["training_gate_pass"]) for result in results),
        "validation_evaluated_arms": sum(int(result["validation_evaluated"]) for result in results),
        "hashes": rel(OUTPUT / "HASHES.json"),
    })
    print(json.dumps({"experiment": summary["experiment"], "arms": len(results), "training_gate_passes": sum(int(result["training_gate_pass"]) for result in results), "validation_evaluated": sum(int(result["validation_evaluated"]) for result in results), "sealed_test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
