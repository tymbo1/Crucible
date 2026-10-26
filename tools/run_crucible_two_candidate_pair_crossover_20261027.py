#!/usr/bin/env python3
"""Complete the two-candidate train-panel/held-out-panel crossover.

The four cyclic cells from the pair-coverage experiment are immutable prior
evidence. This runner trains only the eight missing ordered panel combinations
for seeds 42 and 1337, then emits a combined 24-arm analysis.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
PAIR_RUNNER = ROOT / "tools" / "run_crucible_two_candidate_pair_coverage_20261027.py"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_PAIR_CROSSOVER_PROTOCOL_20261027.md"
NOVELTY = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_PAIR_CROSSOVER_NOVELTY_AUDIT_20261027.json"
OUTPUT = ROOT / "experiments" / "crucible_two_candidate_pair_crossover_20261027_v1"
PRIOR_OUTPUT = ROOT / "experiments" / "crucible_two_candidate_pair_coverage_20261027_v1"

SEEDS = (42, 1337)
TRAIN_THRESHOLD = 0.95
PRIOR_CELL_BY_COMBO = {
    ("M0", "M1"): "P0",
    ("M1", "M2"): "P1",
    ("M2", "M3"): "P2",
    ("M3", "M0"): "P3",
}


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


pair = import_module("pair_coverage_for_crossover", PAIR_RUNNER)
base = pair.base
source = pair.source
torch = pair.torch
PANELS = pair.PANELS
TRAIN_CONTEXTS = pair.TRAIN_CONTEXTS
VALIDATION_CONTEXTS = pair.VALIDATION_CONTEXTS
BATCH_SIZE = pair.BATCH_SIZE
STEPS = pair.STEPS
SHUFFLE_SEED = pair.SHUFFLE_SEED
TRAIN_SOURCE = pair.TRAIN_SOURCE
VALIDATION_SOURCE = pair.VALIDATION_SOURCE

ALL_COMBOS = tuple(
    (train_panel, validation_panel)
    for train_panel in sorted(PANELS)
    for validation_panel in sorted(PANELS)
    if train_panel != validation_panel
)
NEW_COMBOS = tuple(combo for combo in ALL_COMBOS if combo not in PRIOR_CELL_BY_COMBO)


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


def cell_name(train_panel: str, validation_panel: str, seed: int) -> str:
    return f"T_{train_panel}_V_{validation_panel}_SEED{seed}"


def old_cell_name(train_panel: str, validation_panel: str, seed: int) -> str:
    return f"{PRIOR_CELL_BY_COMBO[(train_panel, validation_panel)]}_SEED{seed}"


def load_prior_hashes() -> Dict[str, str]:
    return json.loads((PRIOR_OUTPUT / "HASHES.json").read_text(encoding="utf-8"))


def verify_hashes(hashes: Mapping[str, str]) -> List[Dict[str, str]]:
    failures: List[Dict[str, str]] = []
    for name, expected in hashes.items():
        path = ROOT / name
        if not path.exists():
            failures.append({"path": name, "error": "missing"})
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append({"path": name, "expected": expected, "actual": actual})
    return failures


def combo_label(train_panel: str, validation_panel: str) -> str:
    return f"{train_panel}->{validation_panel}"


def audit_combo(train_panel: str, validation_panel: str) -> Dict[str, Any]:
    partition = f"T_{train_panel}_V_{validation_panel}"
    train_records = pair.make_records(partition, "train", train_panel, TRAIN_CONTEXTS)
    validation_records = pair.make_records(partition, "validation", validation_panel, VALIDATION_CONTEXTS)
    train_audit = pair.audit_rows(f"{partition}_train", train_records, 144, train_panel, TRAIN_CONTEXTS)
    validation_audit = pair.audit_rows(f"{partition}_validation", validation_records, 96, validation_panel, VALIDATION_CONTEXTS)
    overlap = pair.cross_split_audit(train_records, validation_records)
    if overlap != {"record_ids": 0, "exact_inputs": 0, "unordered_pairs": 0}:
        raise RuntimeError(f"{partition}: train/validation contamination {overlap}")
    source.prepare_records(train_records)
    source.prepare_records(validation_records)
    return {
        "train_panel": train_panel,
        "validation_panel": validation_panel,
        "train_records": train_records,
        "validation_records": validation_records,
        "train_audit": train_audit,
        "validation_audit": validation_audit,
        "cross_split_overlap": overlap,
    }


def compact_result(result: Mapping[str, Any], display_cell: str, source_kind: str, train_panel: str, validation_panel: str) -> Dict[str, Any]:
    return {
        "cell": display_cell,
        "source": source_kind,
        "train_panel": train_panel,
        "validation_panel": validation_panel,
        "seed": result["seed"],
        "checkpoint": result.get("checkpoint"),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "parameter_count": result.get("parameter_count"),
        "steps": result.get("steps"),
        "batch_size": result.get("batch_size"),
        "schedule": result.get("schedule"),
        "training_gate_pass": result["training_gate_pass"],
        "train_metrics": result["train_metrics"],
        "train_slot_metrics": result.get("train_slot_metrics"),
        "validation_metrics": result["validation_metrics"],
        "validation_slot_metrics": result.get("validation_slot_metrics"),
    }


def prior_result(train_panel: str, validation_panel: str, seed: int) -> Dict[str, Any]:
    old_cell = old_cell_name(train_panel, validation_panel, seed)
    payload = json.loads((PRIOR_OUTPUT / old_cell / "RESULTS.json").read_text(encoding="utf-8"))
    return compact_result(payload, cell_name(train_panel, validation_panel, seed), "prior", train_panel, validation_panel)


def renamed_breakdown(payload: Mapping[str, Any], display_cell: str, train_panel: str, validation_panel: str, seed: int) -> Dict[str, Any]:
    return {
        "cell": display_cell,
        "source": payload.get("source", "prior"),
        "partition": combo_label(train_panel, validation_panel),
        "train_panel": train_panel,
        "validation_panel": validation_panel,
        "seed": seed,
        "training_gate_pass": payload["training_gate_pass"],
        "train": payload["train"],
        "validation": payload["validation"],
    }


def aggregate_by_validation_panel(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for panel in sorted(PANELS):
        subset = [result for result in results if result["validation_panel"] == panel]
        rows = sum(result["validation_metrics"]["rows"] for result in subset)
        correct = sum(result["validation_metrics"]["pointer_correct"] for result in subset)
        paired = sum(result["validation_metrics"]["paired_correct"] for result in subset)
        groups = sum(result["validation_metrics"]["paired_groups"] for result in subset)
        abstain = sum(result["validation_metrics"]["abstain_correct"] for result in subset)
        abstain_rows = sum(result["validation_metrics"]["abstain_rows"] for result in subset)
        aggregate[panel] = {
            "arms": len(subset),
            "rows": rows,
            "correct": correct,
            "accuracy": correct / rows if rows else None,
            "paired_correct": paired,
            "paired_groups": groups,
            "abstention_correct": abstain,
            "abstention_rows": abstain_rows,
        }
    return aggregate


def main() -> int:
    execute = "--execute" in sys.argv[1:]
    if OUTPUT.exists() and (OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"refusing to overwrite completed output: {OUTPUT}")
    if not PRIOR_OUTPUT.exists() or not (PRIOR_OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"missing completed prior experiment: {PRIOR_OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    prior_hashes = load_prior_hashes()
    prior_hash_failures = verify_hashes(prior_hashes)
    if prior_hash_failures:
        raise SystemExit(f"prior hash verification failed: {prior_hash_failures[:3]}")

    source_train = load_jsonl(TRAIN_SOURCE)
    source_contexts = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in source_train if len(row["metadata"]["candidates"]) == 2}
    if not set(TRAIN_CONTEXTS).issubset(source_contexts) or not set(VALIDATION_CONTEXTS).issubset(source_contexts):
        raise RuntimeError("configured contexts are not present in the canonical training context pool")
    if not set(VALIDATION_CONTEXTS).issubset(set(TRAIN_CONTEXTS)):
        raise RuntimeError("validation contexts are not represented in training")
    all_pairs = [tuple(pair_value) for panel in PANELS.values() for pair_value in panel]
    if len(set(all_pairs)) != 16:
        raise RuntimeError("pair panels are not disjoint")
    if any(sorted(sum(([a, b] for a, b in panel), [])) != list(range(8)) for panel in PANELS.values()):
        raise RuntimeError("panel is not a perfect matching over IDs 0..7")
    if len(ALL_COMBOS) != 12 or len(NEW_COMBOS) != 8:
        raise RuntimeError(f"unexpected crossover counts: all={len(ALL_COMBOS)} new={len(NEW_COMBOS)}")

    combos: Dict[str, Dict[str, Any]] = {}
    for train_panel, validation_panel in NEW_COMBOS:
        combos[f"T_{train_panel}_V_{validation_panel}"] = audit_combo(train_panel, validation_panel)

    preflight = {
        "experiment": "crucible_two_candidate_pair_crossover_20261027_v1",
        "status": "preflight_passed",
        "execute_requested": execute,
        "prior_experiment": rel(PRIOR_OUTPUT),
        "prior_hash_files": len(prior_hashes),
        "prior_hash_verification_failures": prior_hash_failures,
        "train_source": rel(TRAIN_SOURCE),
        "validation_source": rel(VALIDATION_SOURCE),
        "train_source_sha256": sha256(TRAIN_SOURCE),
        "validation_source_sha256": sha256(VALIDATION_SOURCE),
        "panels": {name: pair.panel_signature(panel) for name, panel in PANELS.items()},
        "prior_combinations": {combo_label(train, validation): old for (train, validation), old in PRIOR_CELL_BY_COMBO.items()},
        "new_combinations": [combo_label(train, validation) for train, validation in NEW_COMBOS],
        "new_partitions": {
            name: {
                "train_panel": data["train_panel"],
                "validation_panel": data["validation_panel"],
                "train_audit": data["train_audit"],
                "validation_audit": data["validation_audit"],
                "cross_split_overlap": data["cross_split_overlap"],
            }
            for name, data in combos.items()
        },
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
    for name, data in combos.items():
        write_jsonl(OUTPUT / "data" / f"{name}_train.jsonl", data["train_records"])
        write_jsonl(OUTPUT / "data" / f"{name}_validation.jsonl", data["validation_records"])
    if not execute:
        print(json.dumps({"experiment": preflight["experiment"], "status": "preflight_passed", "new_arms": len(NEW_COMBOS) * len(SEEDS), "training_performed": False, "execute_with": "--execute"}, sort_keys=True))
        return 0

    base.OUTPUT = OUTPUT
    base.STEPS = STEPS
    base.BATCH_SIZE = BATCH_SIZE
    base.SHUFFLE_SEED = SHUFFLE_SEED
    base.TRAIN_THRESHOLD = TRAIN_THRESHOLD
    new_results: List[Dict[str, Any]] = []
    for name, data in combos.items():
        for seed in SEEDS:
            cell = f"{name}_SEED{seed}"
            base.SEED = seed
            result = base.train_cell(cell, "dense", "cyclic", data["train_records"])
            train_rows = base.evaluate(
                pair.load_model(OUTPUT / cell / "checkpoint.pt", seed),
                "dense",
                data["train_records"],
                base.runner.prepare_records(data["train_records"]),
            )
            train_enriched = pair.enrich_predictions(train_rows, data["train_records"])
            write_csv(OUTPUT / cell / "TRAIN_PREDICTIONS_ENRICHED.csv", train_enriched)
            result["train_slot_metrics"] = pair.slot_metrics(train_rows)
            validation_rows = base.evaluate(
                pair.load_model(OUTPUT / cell / "checkpoint.pt", seed),
                "dense",
                data["validation_records"],
                base.runner.prepare_records(data["validation_records"]),
            )
            validation_enriched = pair.enrich_predictions(validation_rows, data["validation_records"])
            write_csv(OUTPUT / cell / "VALIDATION_PREDICTIONS.csv", validation_enriched)
            result["validation_evaluated"] = True
            result["validation_metrics"] = base.row_metrics(validation_rows)
            result["validation_slot_metrics"] = pair.slot_metrics(validation_rows)
            result["validation_breakdown"] = pair.metric_breakdown(validation_enriched)
            result["train_breakdown"] = pair.metric_breakdown(train_enriched)
            result["train_panel"] = data["train_panel"]
            result["validation_panel"] = data["validation_panel"]
            result["source"] = "new"
            write_json(OUTPUT / cell / "RESULTS.json", result)
            new_results.append(result)

    combined: List[Dict[str, Any]] = []
    combined_breakdown: Dict[str, Any] = {}
    for train_panel, validation_panel in ALL_COMBOS:
        for seed in SEEDS:
            display = cell_name(train_panel, validation_panel, seed)
            if (train_panel, validation_panel) in PRIOR_CELL_BY_COMBO:
                old_cell = old_cell_name(train_panel, validation_panel, seed)
                old_payload = json.loads((PRIOR_OUTPUT / old_cell / "RESULTS.json").read_text(encoding="utf-8"))
                compact = compact_result(old_payload, display, "prior", train_panel, validation_panel)
                old_breakdown = json.loads((PRIOR_OUTPUT / "ERROR_BREAKDOWN.json").read_text(encoding="utf-8"))[old_cell]
                combined_breakdown[display] = renamed_breakdown(old_breakdown, display, train_panel, validation_panel, seed)
            else:
                new_cell = display
                new_payload = next(result for result in new_results if result["cell"] == new_cell)
                compact = compact_result(new_payload, display, "new", train_panel, validation_panel)
                combined_breakdown[display] = renamed_breakdown({"source": "new", "training_gate_pass": new_payload["training_gate_pass"], "train": new_payload["train_breakdown"], "validation": new_payload["validation_breakdown"]}, display, train_panel, validation_panel, seed)
            combined.append(compact)

    summary = {
        "experiment": "crucible_two_candidate_pair_crossover_20261027_v1",
        "status": "completed",
        "protocol": rel(PROTOCOL),
        "novelty_audit": rel(NOVELTY),
        "preflight": rel(OUTPUT / "PREFLIGHT.json"),
        "prior_experiment": rel(PRIOR_OUTPUT),
        "prior_arm_count": 8,
        "new_arm_count": len(new_results),
        "total_arm_count": len(combined),
        "all_combinations": [combo_label(train, validation) for train, validation in ALL_COMBOS],
        "new_combinations": [combo_label(train, validation) for train, validation in NEW_COMBOS],
        "seeds": list(SEEDS),
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "schedule": "cyclic",
        "training_gate_threshold": TRAIN_THRESHOLD,
        "training_performed": True,
        "sealed_test_accessed": False,
        "prior_checkpoints_modified": False,
        "training_gate_passes_new": sum(int(result["training_gate_pass"]) for result in new_results),
        "training_gate_passes_combined": sum(int(result["training_gate_pass"]) for result in combined),
        "results": combined,
        "validation_by_held_out_panel": aggregate_by_validation_panel(combined),
    }
    write_json(OUTPUT / "SUMMARY.json", summary)
    write_json(OUTPUT / "ERROR_BREAKDOWN.json", combined_breakdown)

    report_lines = [
        "# Crucible Two-Candidate Pair-Coverage Crossover v1",
        "",
        "Complete ordered train-panel/held-out-panel crossover. The four prior cyclic cells are included by immutable reference; only the eight missing combinations were trained here.",
        "",
        "| Train | Held out | Seed | Source | Train | Gate | Validation | Paired validation |",
        "|---|---|---:|---|---:|---|---:|---:|",
    ]
    for result in combined:
        report_lines.append(
            f"| {result['train_panel']} | {result['validation_panel']} | {result['seed']} | {result['source']} | {result['train_metrics']['pointer_correct']}/{result['train_metrics']['rows']} ({result['train_metrics']['pointer_accuracy']:.3f}) | {'PASS' if result['training_gate_pass'] else 'FAIL'} | {result['validation_metrics']['pointer_correct']}/{result['validation_metrics']['rows']} ({result['validation_metrics']['pointer_accuracy']:.3f}) | {result['validation_metrics']['paired_correct']}/{result['validation_metrics']['paired_groups']} |"
        )
    report_lines += [
        "",
        "## Validation by held-out panel",
        "",
        "| Held-out panel | Rows | Accuracy | Paired groups | Abstention |",
        "|---|---:|---:|---:|---:|",
    ]
    for panel, values in sorted(summary["validation_by_held_out_panel"].items()):
        report_lines.append(f"| {panel} | {values['correct']}/{values['rows']} | {values['accuracy']:.3f} | {values['paired_correct']}/{values['paired_groups']} | {values['abstention_correct']}/{values['abstention_rows']} |")
    report_lines += [
        "",
        "## Interpretation boundary",
        "",
        "Interpret validation only after the 95% training gate. Repeated weakness of the same unordered pair across the three training-partner panels would support pair-specific difficulty; changes with the training partner would support coverage-graph dependence; seed reversals under the same panel combination would support optimization or initialization sensitivity. These are behavioral interpretations, not neural-mechanism claims.",
        "",
        "## Provenance",
        "",
        f"- [Protocol](../../{rel(PROTOCOL)})",
        f"- [Novelty audit](../../{rel(NOVELTY)})",
        f"- Prior immutable run: [`{rel(PRIOR_OUTPUT)}`](../../{rel(PRIOR_OUTPUT)})",
        "- New arm data, checkpoints, traces and prediction matrices are under the cell directories.",
    ]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    interpretation = """# Scientific Interpretation

This crossover combines the four immutable cyclic cells from the prior
pair-coverage run with the eight missing ordered train-panel/held-out-panel
combinations trained here. Every new arm uses the same dense SHARED_SCORER,
cyclic schedule, compact layout, contexts, budget and competence gate.

The primary question is whether held-out pair performance is stable across the
three possible training-partner panels. Same-pair weakness across those panels
supports pair-specific difficulty; movement with the training partner supports
coverage-graph dependence; seed reversals under the same combination support
optimization or initialization sensitivity.

See SUMMARY.json and ERROR_BREAKDOWN.json for complete denominators. The prior
cells are referenced rather than retrained, and this experiment does not
identify a neural mechanism or establish variable-cardinality transfer.
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
        PRIOR_OUTPUT / "SUMMARY.json",
        PRIOR_OUTPUT / "ERROR_BREAKDOWN.json",
        PRIOR_OUTPUT / "HASHES.json",
        PRIOR_OUTPUT / "FINALIZATION.json",
    ]
    for name, data in combos.items():
        hash_paths.extend([OUTPUT / "data" / f"{name}_train.jsonl", OUTPUT / "data" / f"{name}_validation.jsonl"])
        for seed in SEEDS:
            cell_dir = OUTPUT / f"{name}_SEED{seed}"
            hash_paths.extend([cell_dir / "RESULTS.json", cell_dir / "PREDICTIONS.csv", cell_dir / "TRAIN_PREDICTIONS_ENRICHED.csv", cell_dir / "VALIDATION_PREDICTIONS.csv", cell_dir / "TRAINING_TRACE.jsonl", cell_dir / "checkpoint.pt"])
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths})
    write_json(OUTPUT / "FINALIZATION.json", {
        "experiment": summary["experiment"],
        "status": "completed",
        "verification_pass": True,
        "training_performed": True,
        "sealed_test_accessed": False,
        "prior_checkpoints_modified": False,
        "prior_hash_verification_failures": prior_hash_failures,
        "new_combination_count": len(NEW_COMBOS),
        "new_arm_count": len(new_results),
        "total_arm_count": len(combined),
        "training_gate_passes_new": sum(int(result["training_gate_pass"]) for result in new_results),
        "training_gate_passes_combined": sum(int(result["training_gate_pass"]) for result in combined),
        "hashes": rel(OUTPUT / "HASHES.json"),
    })
    print(json.dumps({"experiment": summary["experiment"], "status": "completed", "new_arms": len(new_results), "total_arms": len(combined), "training_gate_passes_new": sum(int(result["training_gate_pass"]) for result in new_results), "sealed_test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
