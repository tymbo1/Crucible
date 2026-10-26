#!/usr/bin/env python3
"""Add post-hoc failure-mode diagnostics to the completed role-selection gate."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_role_selection_competence_20251025_v5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    rows = list(csv.DictReader((OUTPUT / "PREDICTIONS.csv").open(newline="", encoding="utf-8")))
    s3 = [row for row in rows if row["condition"] == "S3_JOINT_OPERATION_CONDITIONED" and row["split"] == "validation" and row["pair_kind"] == "unequal"]
    by_arm = {}
    for arm in sorted({row["arm_id"] for row in s3}):
        selected = [row for row in s3 if row["arm_id"] == arm]
        groups = {}
        for row in selected:
            groups.setdefault((row["a"], row["b"]), {})[row["operation"]] = row["predicted_output"]
        same_prediction = sum(values.get("SELECT_FIRST") == values.get("SELECT_SECOND") for values in groups.values())
        by_arm[arm] = {
            "unequal_validation_rows": len(selected),
            "correct": sum(row["output_token_correct"] == "True" for row in selected),
            "predicted_either_operand": sum(row["predicted_output"] in {row["a"], row["b"]} for row in selected),
            "pair_groups": len(groups),
            "same_prediction_for_both_operations": same_prediction,
            "operation_specific_accuracy": {
                operation: {
                    "correct": sum(row["operation"] == operation and row["output_token_correct"] == "True" for row in selected),
                    "rows": sum(row["operation"] == operation for row in selected),
                }
                for operation in ("SELECT_FIRST", "SELECT_SECOND")
            },
        }

    s2 = [row for row in rows if row["condition"] == "S2_SELECT_SECOND" and row["split"] == "validation" and row["pair_kind"] == "unequal"]
    s2_errors = [
        {key: row[key] for key in ("arm_id", "a", "b", "target_output", "predicted_output", "operation")}
        for row in s2
        if row["output_token_correct"] != "True"
    ]
    diagnostic = {
        "source": "PREDICTIONS.csv",
        "s3_unequal_validation": {
            "rows": len(s3),
            "correct": sum(row["output_token_correct"] == "True" for row in s3),
            "predicted_either_operand": sum(row["predicted_output"] in {row["a"], row["b"]} for row in s3),
            "same_prediction_for_both_operations": sum(item["same_prediction_for_both_operations"] for item in by_arm.values()),
            "pair_groups": sum(item["pair_groups"] for item in by_arm.values()),
            "format_errors": sum(row["output_format_error"] == "True" for row in s3),
            "eos_errors": sum(row["eos_correct"] != "True" for row in s3),
        },
        "s3_by_arm": by_arm,
        "s2_unequal_errors": s2_errors,
        "interpretation": "S3 extracted one of the operand values but produced the same value for both operation identifiers within every unequal validation pair in every arm. This is behaviorally consistent with operation-token neglect or failure to condition execution on the requested role; it does not identify an internal mechanism.",
    }
    write_json(OUTPUT / "POSTHOC_DIAGNOSTIC.json", diagnostic)
    (OUTPUT / "POSTHOC_DIAGNOSTIC.md").write_text(
        "# Role-Selection Post-hoc Failure Diagnostic\n\n"
        f"Across S3 unequal validation: {diagnostic['s3_unequal_validation']['correct']}/{diagnostic['s3_unequal_validation']['rows']} correct; "
        f"{diagnostic['s3_unequal_validation']['predicted_either_operand']}/{diagnostic['s3_unequal_validation']['rows']} predictions were one of the two operands; "
        f"{diagnostic['s3_unequal_validation']['same_prediction_for_both_operations']}/{diagnostic['s3_unequal_validation']['pair_groups']} pair-groups produced the same prediction for both operation identifiers. "
        "There were no format or EOS errors in these rows.\n\n"
        "This supports a behavioral diagnosis of operation-token neglect or failed operation-conditioned execution, not a claim about a specific internal neural mechanism.\n\n"
        f"S2 unequal errors: {json.dumps(s2_errors, sort_keys=True)}\n",
        encoding="utf-8",
    )

    analysis_path = OUTPUT / "ANALYSIS_FINAL.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["posthoc_diagnostic"] = diagnostic
    write_json(analysis_path, analysis)

    report_path = OUTPUT / "REPORT_FINAL.md"
    report = report_path.read_text(encoding="utf-8")
    section = (
        "\n## Post-hoc failure-mode diagnostic\n\n"
        f"S1 passed all four arms. S2 passed three of four; the only unequal-validation error was one factorized seed-1337 prediction on `(9,0)`.\n\n"
        f"S3 produced {diagnostic['s3_unequal_validation']['correct']}/{diagnostic['s3_unequal_validation']['rows']} correct unequal-validation predictions. "
        f"All {diagnostic['s3_unequal_validation']['predicted_either_operand']}/{diagnostic['s3_unequal_validation']['rows']} predictions were one of the two operands, but "
        f"{diagnostic['s3_unequal_validation']['same_prediction_for_both_operations']}/{diagnostic['s3_unequal_validation']['pair_groups']} pair-groups received the same output for `SELECT_FIRST` and `SELECT_SECOND`. "
        "There were no output-format or EOS errors. This is behaviorally consistent with operation-identifier neglect or failed operation-conditioned execution, while remaining agnostic about the internal neural mechanism.\n\n"
        "The conditional-transfer stage was not run because the preregistered competence gates failed.\n"
    )
    if "## Post-hoc failure-mode diagnostic" not in report:
        report_path.write_text(report.rstrip() + "\n" + section, encoding="utf-8")

    hashes = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt":
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = json.loads((OUTPUT / "FINALIZATION.json").read_text(encoding="utf-8"))
    finalization["analysis_sha256"] = sha256(analysis_path)
    finalization["report_sha256"] = sha256(report_path)
    finalization["posthoc_diagnostic_sha256"] = sha256(OUTPUT / "POSTHOC_DIAGNOSTIC.json")
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "posthoc_diagnostic_written", "s3_correct": diagnostic["s3_unequal_validation"]["correct"], "s3_rows": diagnostic["s3_unequal_validation"]["rows"], "same_prediction_pair_groups": diagnostic["s3_unequal_validation"]["same_prediction_for_both_operations"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
