#!/usr/bin/env python3
"""Train and evaluate the preregistered rotated pair-coverage experiment."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = ROOT / "tools" / "run_crucible_cardinality_competence_diagnostic_20261026.py"
SOURCE_RUNNER = ROOT / "tools" / "run_crucible_shared_candidate_scoring_cardinality_20261026.py"
TRAIN_SOURCE = ROOT / "experiments" / "crucible_shared_candidate_scoring_cardinality_20261026_v1" / "data" / "canonical" / "train.jsonl"
VALIDATION_SOURCE = ROOT / "experiments" / "crucible_shared_candidate_scoring_cardinality_20261026_v1" / "data" / "canonical" / "validation.jsonl"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_PAIR_COVERAGE_PROTOCOL_20261027.md"
NOVELTY = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_PAIR_COVERAGE_NOVELTY_AUDIT_20261027.json"
OUTPUT = ROOT / "experiments" / "crucible_two_candidate_pair_coverage_20261027_v1"

SEEDS = (42, 1337)
BATCH_SIZE = 12
STEPS = 500
SHUFFLE_SEED = 20261027
TRAIN_THRESHOLD = 0.95
TRAIN_CONTEXTS = ((1, 6), (1, 8), (2, 7), (2, 9), (3, 8), (4, 9))
VALIDATION_CONTEXTS = ((1, 6), (2, 7), (3, 8), (4, 9))
PANELS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "M0": ((0, 1), (2, 3), (4, 5), (6, 7)),
    "M1": ((0, 2), (1, 3), (4, 6), (5, 7)),
    "M2": ((0, 3), (1, 2), (4, 7), (5, 6)),
    "M3": ((0, 4), (1, 5), (2, 6), (3, 7)),
}
PARTITIONS = {
    "P0": {"train_panel": "M0", "validation_panel": "M1"},
    "P1": {"train_panel": "M1", "validation_panel": "M2"},
    "P2": {"train_panel": "M2", "validation_panel": "M3"},
    "P3": {"train_panel": "M3", "validation_panel": "M0"},
}


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


base = import_module("pair_coverage_base_runner", BASE_RUNNER)
source = import_module("pair_coverage_source_runner", SOURCE_RUNNER)
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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def oracle_target(row: Mapping[str, Any]) -> int:
    candidates = [int(value) for value in row["metadata"]["candidates"]]
    authority = int(row["metadata"]["authority_id"])
    return candidates.index(authority) if authority in candidates else 8


def panel_signature(panel: Sequence[Sequence[int]]) -> List[str]:
    return [f"{min(pair)},{max(pair)}" for pair in panel]


def make_records(partition: str, split: str, panel_name: str, contexts: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ordinal = 0
    for context_index, (a, b) in enumerate(contexts):
        for pair_index, unordered in enumerate(PANELS[panel_name]):
            for candidates in (tuple(unordered), (unordered[1], unordered[0])):
                pair_key = f"pair_coverage|{partition}|{split}|{context_index}|{a}|{b}|{candidates}"
                counterfactual_pair_id = hashlib.sha256(pair_key.encode("utf-8")).hexdigest()[:20]
                for authority in (candidates[0], candidates[1], 9):
                    target = candidates.index(authority) if authority in candidates else 8
                    record_key = f"{pair_key}|{authority}|{ordinal}"
                    record_id = hashlib.sha256(record_key.encode("utf-8")).hexdigest()[:20]
                    rows.append({
                        "schema_version": "crucible_two_candidate_pair_coverage_v1",
                        "record_id": record_id,
                        "interface": "canonical",
                        "split": split,
                        "layout": "TRAIN_COMPACT",
                        "input": source.input_text("canonical", "TRAIN_COMPACT", a, b, candidates, authority, ordinal),
                        "target": str(target),
                        "oracle": {"kind": "identifier_match", "target_slot": target, "authority_id": authority},
                        "metadata": {
                            "a": a,
                            "b": b,
                            "candidates": list(candidates),
                            "authority_id": authority,
                            "target_slot": target,
                            "counterfactual_pair_id": counterfactual_pair_id if authority != 9 else None,
                            "partition": partition,
                            "panel": panel_name,
                            "unordered_pair": list(unordered),
                            "context_index": context_index,
                            "context_pool": "TRAIN_SOURCE_POOL",
                            "pair_index": pair_index,
                        },
                    })
                    ordinal += 1
    return rows


def class_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(int(row["metadata"]["target_slot"]) for row in rows).items())}


def audit_rows(name: str, rows: Sequence[Mapping[str, Any]], expected_rows: int, expected_panel: str, contexts: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    if len(rows) != expected_rows:
        raise RuntimeError(f"{name}: expected {expected_rows} rows, got {len(rows)}")
    if len({str(row["record_id"]) for row in rows}) != len(rows):
        raise RuntimeError(f"{name}: duplicate record IDs")
    if len({str(row["input"]) for row in rows}) != len(rows):
        raise RuntimeError(f"{name}: duplicate inputs")
    if any(str(row["layout"]) != "TRAIN_COMPACT" for row in rows):
        raise RuntimeError(f"{name}: layout changed")
    if any(len(row["metadata"]["candidates"]) != 2 for row in rows):
        raise RuntimeError(f"{name}: non-two-candidate row")
    if any(int(row["metadata"]["target_slot"]) != oracle_target(row) for row in rows):
        raise RuntimeError(f"{name}: oracle mismatch")
    if class_counts(rows) != ({"0": 48, "1": 48, "8": 48} if expected_rows == 144 else {"0": 32, "1": 32, "8": 32}):
        raise RuntimeError(f"{name}: class balance mismatch {class_counts(rows)}")
    actual_contexts = sorted({(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in rows})
    if actual_contexts != sorted(contexts):
        raise RuntimeError(f"{name}: context mismatch {actual_contexts}")
    actual_pairs = {tuple(sorted(int(value) for value in row["metadata"]["candidates"])) for row in rows}
    expected_pairs = {tuple(pair) for pair in PANELS[expected_panel]}
    if actual_pairs != expected_pairs:
        raise RuntimeError(f"{name}: pair-panel mismatch")
    return {
        "rows": len(rows),
        "target_slot_counts": class_counts(rows),
        "unique_record_ids": len({str(row["record_id"]) for row in rows}),
        "unique_inputs": len({str(row["input"]) for row in rows}),
        "counterfactual_groups": len({str(row["metadata"]["counterfactual_pair_id"]) for row in rows if row["metadata"]["counterfactual_pair_id"]}),
        "contexts": [f"{a},{b}" for a, b in sorted(actual_contexts)],
        "unordered_pairs": panel_signature(PANELS[expected_panel]),
        "oracle_rows_correct": len(rows),
    }


def cross_split_audit(train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        "record_ids": len({str(row["record_id"]) for row in train} & {str(row["record_id"]) for row in validation}),
        "exact_inputs": len({str(row["input"]) for row in train} & {str(row["input"]) for row in validation}),
        "unordered_pairs": len({tuple(sorted(int(value) for value in row["metadata"]["candidates"])) for row in train} & {tuple(sorted(int(value) for value in row["metadata"]["candidates"])) for row in validation}),
    }


def slot_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for target in (0, 1, 8):
        subset = [row for row in rows if int(row["target_slot"]) == target]
        correct = sum(int(bool(row["pointer_correct"])) for row in subset)
        result[str(target)] = {"rows": len(subset), "correct": correct, "accuracy": correct / len(subset) if subset else None}
    return result


def enrich_predictions(rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {str(record["record_id"]): record for record in records}
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        record = by_id[str(row["record_id"])]
        metadata = record["metadata"]
        enriched_row = dict(row)
        enriched_row.update({
            "partition": metadata["partition"],
            "panel": metadata["panel"],
            "unordered_pair": ",".join(str(value) for value in metadata["unordered_pair"]),
            "context_a": metadata["a"],
            "context_b": metadata["b"],
            "layout": record["layout"],
        })
        enriched.append(enriched_row)
    return enriched


def load_model(checkpoint: Path, seed: int) -> Any:
    base.SEED = seed
    model = base.model_for("dense")
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    return model


def metric_breakdown(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def block(subset: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        correct = sum(int(bool(row["pointer_correct"])) for row in subset)
        groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in subset:
            if row["counterfactual_pair_id"]:
                groups[str(row["counterfactual_pair_id"])].append(row)
        complete = [values for values in groups.values() if len(values) == 2]
        return {
            "rows": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset) if subset else None,
            "paired_groups": len(complete),
            "paired_correct": sum(int(all(bool(row["pointer_correct"]) for row in values)) for values in complete),
            "predicted_flips": sum(int(values[0]["predicted_slot"] != values[1]["predicted_slot"]) for values in complete),
            "abstention": slot_metrics(subset)["8"],
        }
    return {
        "overall": block(rows),
        "by_panel": {panel: block([row for row in rows if row["panel"] == panel]) for panel in sorted({str(row["panel"]) for row in rows})},
        "by_unordered_pair": {pair: block([row for row in rows if row["unordered_pair"] == pair]) for pair in sorted({str(row["unordered_pair"]) for row in rows})},
        "by_context": {f"{a},{b}": block([row for row in rows if int(row["context_a"]) == a and int(row["context_b"]) == b]) for a, b in sorted({(int(row["context_a"]), int(row["context_b"])) for row in rows})},
    }


def main() -> int:
    execute = "--execute" in sys.argv[1:]
    if OUTPUT.exists() and (OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"refusing to overwrite completed output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    source_train = load_jsonl(TRAIN_SOURCE)
    source_validation = load_jsonl(VALIDATION_SOURCE)
    source_train_contexts = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in source_train if len(row["metadata"]["candidates"]) == 2}
    if not set(TRAIN_CONTEXTS).issubset(source_train_contexts) or not set(VALIDATION_CONTEXTS).issubset(source_train_contexts):
        raise RuntimeError("configured contexts are not present in the canonical training context pool")
    if not set(VALIDATION_CONTEXTS).issubset(set(TRAIN_CONTEXTS)):
        raise RuntimeError("validation contexts are not represented in training")
    all_pairs = [tuple(pair) for panel in PANELS.values() for pair in panel]
    if len(set(all_pairs)) != 16:
        raise RuntimeError("pair panels are not disjoint")
    if any(sorted(sum(([a, b] for a, b in panel), [])) != list(range(8)) for panel in PANELS.values()):
        raise RuntimeError("panel is not a perfect matching over IDs 0..7")
    if any(PARTITIONS[name]["train_panel"] == PARTITIONS[name]["validation_panel"] for name in PARTITIONS):
        raise RuntimeError("partition reuses a panel in train and validation")

    partitions: Dict[str, Dict[str, Any]] = {}
    for partition, config in PARTITIONS.items():
        train_records = make_records(partition, "train", config["train_panel"], TRAIN_CONTEXTS)
        validation_records = make_records(partition, "validation", config["validation_panel"], VALIDATION_CONTEXTS)
        train_audit = audit_rows(f"{partition}_train", train_records, 144, config["train_panel"], TRAIN_CONTEXTS)
        validation_audit = audit_rows(f"{partition}_validation", validation_records, 96, config["validation_panel"], VALIDATION_CONTEXTS)
        overlap = cross_split_audit(train_records, validation_records)
        if overlap != {"record_ids": 0, "exact_inputs": 0, "unordered_pairs": 0}:
            raise RuntimeError(f"{partition}: train/validation contamination {overlap}")
        prepared_train = source.prepare_records(train_records)
        prepared_validation = source.prepare_records(validation_records)
        del prepared_train, prepared_validation
        partitions[partition] = {
            "config": config,
            "train_records": train_records,
            "validation_records": validation_records,
            "train_audit": train_audit,
            "validation_audit": validation_audit,
            "cross_split_overlap": overlap,
        }

    preflight = {
        "experiment": "crucible_two_candidate_pair_coverage_20261027_v1",
        "status": "preflight_passed",
        "execute_requested": execute,
        "train_source": rel(TRAIN_SOURCE),
        "validation_source": rel(VALIDATION_SOURCE),
        "train_source_sha256": sha256(TRAIN_SOURCE),
        "validation_source_sha256": sha256(VALIDATION_SOURCE),
        "train_contexts": [f"{a},{b}" for a, b in TRAIN_CONTEXTS],
        "validation_contexts": [f"{a},{b}" for a, b in VALIDATION_CONTEXTS],
        "panels": {name: panel_signature(panel) for name, panel in PANELS.items()},
        "partitions": {name: {"config": data["config"], "train_audit": data["train_audit"], "validation_audit": data["validation_audit"], "cross_split_overlap": data["cross_split_overlap"]} for name, data in partitions.items()},
        "architecture": "dense SHARED_SCORER",
        "seeds": list(SEEDS),
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "schedule": "cyclic",
        "training_gate_threshold": TRAIN_THRESHOLD,
        "training_performed": False,
        "sealed_test_accessed": False,
        "prior_checkpoints_modified": False,
    }
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    write_json(OUTPUT / "NOVELTY_AUDIT.json", json.loads(NOVELTY.read_text(encoding="utf-8")))
    for partition, data in partitions.items():
        write_jsonl(OUTPUT / "data" / f"{partition}_train.jsonl", data["train_records"])
        write_jsonl(OUTPUT / "data" / f"{partition}_validation.jsonl", data["validation_records"])
    if not execute:
        print(json.dumps({"experiment": preflight["experiment"], "status": "preflight_passed", "training_performed": False, "execute_with": "--execute"}, sort_keys=True))
        return 0

    base.OUTPUT = OUTPUT
    base.STEPS = STEPS
    base.BATCH_SIZE = BATCH_SIZE
    base.SHUFFLE_SEED = SHUFFLE_SEED
    base.TRAIN_THRESHOLD = TRAIN_THRESHOLD
    results: List[Dict[str, Any]] = []
    for partition, data in partitions.items():
        for seed in SEEDS:
            base.SEED = seed
            cell = f"{partition}_SEED{seed}"
            result = base.train_cell(cell, "dense", "cyclic", data["train_records"])
            train_rows = base.evaluate(
                load_model(OUTPUT / cell / "checkpoint.pt", seed),
                "dense",
                data["train_records"],
                base.runner.prepare_records(data["train_records"]),
            )
            train_enriched = enrich_predictions(train_rows, data["train_records"])
            write_csv(OUTPUT / cell / "TRAIN_PREDICTIONS_ENRICHED.csv", train_enriched)
            result["train_slot_metrics"] = slot_metrics(train_rows)
            val_model = load_model(OUTPUT / cell / "checkpoint.pt", seed)
            validation_rows = base.evaluate(
                val_model,
                "dense",
                data["validation_records"],
                base.runner.prepare_records(data["validation_records"]),
            )
            validation_enriched = enrich_predictions(validation_rows, data["validation_records"])
            write_csv(OUTPUT / cell / "VALIDATION_PREDICTIONS.csv", validation_enriched)
            result["validation_evaluated"] = True
            result["validation_metrics"] = base.row_metrics(validation_rows)
            result["validation_slot_metrics"] = slot_metrics(validation_rows)
            result["validation_breakdown"] = metric_breakdown(validation_enriched)
            result["train_breakdown"] = metric_breakdown(train_enriched)
            write_json(OUTPUT / cell / "RESULTS.json", result)
            results.append(result)

    summary = {
        "experiment": "crucible_two_candidate_pair_coverage_20261027_v1",
        "status": "completed",
        "protocol": rel(PROTOCOL),
        "novelty_audit": rel(NOVELTY),
        "preflight": rel(OUTPUT / "PREFLIGHT.json"),
        "data_directory": rel(OUTPUT / "data"),
        "train_source": rel(TRAIN_SOURCE),
        "validation_source": rel(VALIDATION_SOURCE),
        "train_source_sha256": sha256(TRAIN_SOURCE),
        "validation_source_sha256": sha256(VALIDATION_SOURCE),
        "seeds": list(SEEDS),
        "partitions": list(PARTITIONS),
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "schedule": "cyclic",
        "training_gate_threshold": TRAIN_THRESHOLD,
        "training_performed": True,
        "sealed_test_accessed": False,
        "prior_checkpoints_modified": False,
        "results": results,
    }
    write_json(OUTPUT / "SUMMARY.json", summary)

    report_lines = [
        "# Crucible Two-Candidate Pair-Coverage Experiment v1",
        "",
        "Rotated balanced unordered-pair holdouts using the unchanged dense cyclic training path. Validation contexts were represented during training and both layouts used `TRAIN_COMPACT`.",
        "",
        "| Partition | Train panel | Held-out panel | Seed | Train | Gate | Validation | Paired validation |",
        "|---|---|---|---:|---:|---|---:|---:|",
    ]
    for result in results:
        config = PARTITIONS[result["cell"].split("_SEED")[0]]
        train_metrics = result["train_metrics"]
        val_metrics = result["validation_metrics"]
        report_lines.append(f"| {result['cell'].split('_SEED')[0]} | {config['train_panel']} | {config['validation_panel']} | {result['seed']} | {train_metrics['pointer_correct']}/{train_metrics['rows']} ({train_metrics['pointer_accuracy']:.3f}) | {'PASS' if result['training_gate_pass'] else 'FAIL'} | {val_metrics['pointer_correct']}/{val_metrics['rows']} ({val_metrics['pointer_accuracy']:.3f}) | {val_metrics['paired_correct']}/{val_metrics['paired_groups']} |")
    report_lines += [
        "",
        "## Panel-level validation accuracy",
        "",
        "| Cell | Held-out panel | Pair accuracy | Ordered-pair rows |",
        "|---|---|---:|---|",
    ]
    for result in results:
        partition = result["cell"].split("_SEED")[0]
        val = result["validation_breakdown"]["by_panel"]
        panel = PARTITIONS[partition]["validation_panel"]
        block = val[panel]
        pair_text = ", ".join(f"{pair}={info['correct']}/{info['rows']}" for pair, info in sorted(result["validation_breakdown"]["by_unordered_pair"].items()))
        report_lines.append(f"| {result['cell']} | {panel} | {block['correct']}/{block['rows']} ({block['accuracy']:.3f}) | {pair_text} |")
    report_lines += [
        "",
        "## Decision boundary",
        "",
        "A validation result is interpreted only after the 95% training gate. Panel-following errors support exact pair-coverage dependence; repeated identifier-family failures across panels support a more persistent representation effect. These are behavioral interpretations, not neural-mechanism claims.",
        "",
        "## Artifacts",
        "",
        f"- [Protocol](../../{rel(PROTOCOL)})",
        f"- [Novelty audit](../../{rel(NOVELTY)})",
        "- [Preflight](PREFLIGHT.json)",
        "- [Summary](SUMMARY.json)",
        "- Per-partition generated data, checkpoints, traces and prediction matrices under `data/` and partition cell directories",
    ]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    error_breakdown = {
        result["cell"]: {
            "partition": result["cell"].split("_SEED")[0],
            "seed": result["seed"],
            "training_gate_pass": result["training_gate_pass"],
            "train": result["train_breakdown"],
            "validation": result["validation_breakdown"],
        }
        for result in results
    }
    write_json(OUTPUT / "ERROR_BREAKDOWN.json", error_breakdown)

    partition_summary: Dict[str, Any] = {}
    for partition in PARTITIONS:
        cells = [result for result in results if result["cell"].startswith(partition + "_")]
        partition_summary[partition] = {
            "validation_panel": PARTITIONS[partition]["validation_panel"],
            "training_gate_passes": sum(int(result["training_gate_pass"]) for result in cells),
            "mean_validation_accuracy": sum(result["validation_metrics"]["pointer_accuracy"] for result in cells) / len(cells),
            "mean_validation_paired": sum(result["validation_metrics"]["paired_correct"] for result in cells) / len(cells),
        }
    interpretation = """# Scientific Interpretation

This experiment keeps the dense 22M SHARED_SCORER, cyclic schedule, optimizer,
500-update budget, compact serialization and competence gate fixed while
rotating which unordered candidate-pair panel is held out. Validation contexts
are a subset of training contexts, so the principal unseen dimension is the
candidate-pair panel.

The primary comparison is whether errors follow the held-out panel or recur for
the same identifier families after those pairs are included in training in a
different partition. Training-gate failures are reported separately and do not
support transfer claims.

See SUMMARY.json and ERROR_BREAKDOWN.json for the complete per-seed, per-panel
denominators. This experiment cannot identify a neural mechanism; it only
distinguishes behavioral coverage patterns under the preregistered controls.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")

    hash_paths = [
        PROTOCOL,
        NOVELTY,
        Path(__file__).resolve(),
        TRAIN_SOURCE,
        VALIDATION_SOURCE,
        OUTPUT / "PREFLIGHT.json",
        OUTPUT / "SUMMARY.json",
        OUTPUT / "REPORT_FINAL.md",
        OUTPUT / "ERROR_BREAKDOWN.json",
        OUTPUT / "SCIENTIFIC_INTERPRETATION.md",
    ]
    for partition in PARTITIONS:
        hash_paths.extend([OUTPUT / "data" / f"{partition}_train.jsonl", OUTPUT / "data" / f"{partition}_validation.jsonl"])
        for seed in SEEDS:
            cell_dir = OUTPUT / f"{partition}_SEED{seed}"
            hash_paths.extend([cell_dir / "RESULTS.json", cell_dir / "PREDICTIONS.csv", cell_dir / "TRAIN_PREDICTIONS_ENRICHED.csv", cell_dir / "VALIDATION_PREDICTIONS.csv", cell_dir / "TRAINING_TRACE.jsonl", cell_dir / "checkpoint.pt"])
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths})
    write_json(OUTPUT / "FINALIZATION.json", {
        "experiment": summary["experiment"],
        "status": "completed",
        "verification_pass": True,
        "training_performed": True,
        "sealed_test_accessed": False,
        "prior_checkpoints_modified": False,
        "partition_count": len(PARTITIONS),
        "arm_count": len(results),
        "training_gate_passes": sum(int(result["training_gate_pass"]) for result in results),
        "hashes": rel(OUTPUT / "HASHES.json"),
    })
    print(json.dumps({"experiment": summary["experiment"], "status": "completed", "arms": len(results), "training_gate_passes": sum(int(result["training_gate_pass"]) for result in results), "sealed_test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
