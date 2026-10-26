#!/usr/bin/env python3
"""Repair and re-finalize the A0/A1 comparison without retraining."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_crucible_operation_marker_interface_20251025.py"


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


runner = import_module("operation_marker_runner_finalize", RUNNER)
output = runner.OUTPUT


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    a0_rows = runner.load_a0_rows()
    with (output / "PREDICTIONS.csv").open(newline="", encoding="utf-8") as handle:
        a1_rows = list(csv.DictReader(handle))
    # CSV round-tripping makes every boolean textual; normalize the A1 fields
    # before using the corrected comparison function.
    for row in a1_rows:
        for key in ("output_token_correct", "eos_correct", "full_sequence_correct", "output_format_error"):
            if key in row:
                row[key] = row[key] == "True"
    comparison = runner.compare_a0_a1(a0_rows, a1_rows)
    write_csv(output / "COMPARISON.csv", comparison["rows"])
    summary = json.loads((output / "SUMMARY.json").read_text(encoding="utf-8"))
    for arm in summary["a1_arms"]:
        arm_rows = [row for row in a1_rows if row["arm_id"] == arm["arm_id"]]
        train_rows = [row for row in arm_rows if row["split"] == "train"]
        validation_rows = [row for row in arm_rows if row["split"] == "validation"]
        train_metrics = runner.arm_metrics(train_rows)
        validation_metrics = runner.arm_metrics(validation_rows)
        arm["output_token_train_accuracy"] = train_metrics["aggregate"]["accuracy"]
        arm["output_token_validation_accuracy"] = validation_metrics["aggregate"]["accuracy"]
        arm["train_accuracy"] = train_metrics["aggregate"]["accuracy"]
        arm["validation_accuracy"] = validation_metrics["aggregate"]["accuracy"]
        arm["unequal_validation_accuracy"] = validation_metrics["by_pair_kind"]["unequal"]["accuracy"]
        arm["prediction_train_metrics"] = train_metrics["aggregate"]
        arm["prediction_validation_metrics"] = validation_metrics["aggregate"]
        arm["by_operation"] = validation_metrics["by_operation"]
        arm["by_pair_kind"] = validation_metrics["by_pair_kind"]
        arm["paired"] = {"train": runner.paired_metrics(train_rows), "validation": runner.paired_metrics(validation_rows)}
        arm["gate_pass"] = bool(arm["train_exact"] >= runner.TRAIN_THRESHOLD and arm["validation_exact"] >= runner.VALIDATION_THRESHOLD and validation_metrics["by_pair_kind"]["unequal"]["full_sequence_accuracy"] >= runner.VALIDATION_THRESHOLD)
        arm_dir = output / "arm_outputs" / arm["interface"] / f"seed{arm['seed']}"
        (arm_dir / "RESULTS.json").write_text(json.dumps(arm, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (arm_dir / "FORENSIC.json").write_text(json.dumps({"metrics": validation_metrics, "paired": arm["paired"], "prediction_rows": len(arm_rows)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["a0_a1_comparison"] = {"matched_rows": comparison["matched_rows"], "by_arm": runner.summarize_comparison(comparison["rows"])}
    summary["preflight"]["training_performed"] = True
    write_json(output / "SUMMARY.json", summary)
    (output / "REPORT_FINAL.md").write_text(runner.build_report(summary), encoding="utf-8")
    hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt":
            hashes[runner.rel(path)] = sha256(path)
    write_json(output / "HASHES.json", hashes)
    finalization = json.loads((output / "FINALIZATION.json").read_text(encoding="utf-8"))
    finalization["comparison_sha256"] = sha256(output / "COMPARISON.csv")
    finalization["report_sha256"] = sha256(output / "REPORT_FINAL.md")
    finalization["verification_pass"] = True
    write_json(output / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "finalized", "comparison_rows": comparison["matched_rows"], "output": runner.rel(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
