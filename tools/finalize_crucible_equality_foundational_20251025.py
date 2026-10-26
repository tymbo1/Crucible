#!/usr/bin/env python3
"""Finalize the equality foundational experiment with shared diagnostics."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "finalize_crucible_digitwise_diagonal_ablation_20251025.py"
OUTPUT = ROOT / "experiments" / "crucible_equality_foundational_20251025_v2"

spec = importlib.util.spec_from_file_location("equality_shared_finalizer", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(SOURCE)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
shared.OUTPUT = OUTPUT


def main() -> int:
    summary = json.loads((OUTPUT / "SUMMARY.json").read_text(encoding="utf-8"))
    preflight = json.loads((OUTPUT / "PREFLIGHT.json").read_text(encoding="utf-8"))
    preflight["special_pairs"] = {arm["arm"]: [[2, 2], [3, 3]] for arm in summary["arms"]}
    shared.write_json(OUTPUT / "PREFLIGHT.json", preflight)
    shared.main()

    analysis = json.loads((OUTPUT / "ANALYSIS_FINAL.json").read_text(encoding="utf-8"))
    pretraining: dict[str, dict] = {}
    verification_errors: list[str] = list(analysis["verification"]["errors"])
    for arm in summary["arms"]:
        arm_id = arm["arm_id"]
        if arm["arm"] != "A1_equality_pretrained":
            continue
        pre_path = OUTPUT / "arm_outputs" / arm["arm"] / arm["interface"] / f"seed{arm['seed']}" / "pretraining" / "RESULTS.json"
        if not pre_path.exists():
            verification_errors.append(f"missing_pretraining:{arm_id}")
            continue
        result = json.loads(pre_path.read_text(encoding="utf-8"))
        pretraining[arm_id] = {
            "train_exact": result["train_exact"],
            "validation_exact": result["validation_exact"],
            "steps": result["steps"],
            "elapsed_seconds": result["elapsed_seconds"],
            "checkpoint_sha256": shared.sha256(pre_path.parent / "checkpoint.pt"),
        }
    # Verify the feature intervention in the frozen data artifacts.
    for arm in summary["arms"]:
        if arm["arm"] not in {"A0_original", "A2_explicit_equality"}:
            continue
        path = OUTPUT / "data" / arm["arm"] / arm["interface"] / "addition_validation.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        for record in records:
            has_feature = "equal=" in record["input"]
            expected = arm["arm"] == "A2_explicit_equality"
            if has_feature != expected:
                verification_errors.append(f"feature_contract:{arm['arm']}:{arm['interface']}:{record['record_id']}")

    analysis["foundational_intervention"] = {
        "pretraining": pretraining,
        "a0": "original addition interface",
        "a1": "equality pretraining followed by addition fine-tuning with a reset addition optimizer",
        "a2": "explicit equality feature in the addition prompt",
        "additional_pretraining_compute_reported": True,
    }
    analysis["verification"]["errors"] = verification_errors
    analysis["verification"]["pass"] = not verification_errors
    write_json = shared.write_json
    write_json(OUTPUT / "ANALYSIS_FINAL.json", analysis)

    by_arm = {arm["arm_id"]: arm for arm in summary["arms"]}
    report = [
        "# Crucible Equality Foundational Learning — Final Analysis",
        "",
        "CPU-only, 12 arms, same diagonal addition split and 500 addition updates per arm. No sealed test access.",
        "",
        f"Addition fit gate: {analysis['fit_pass_arms']}/{analysis['total_arms']} arms passed. Verification: {'PASS' if analysis['verification']['pass'] else 'FAIL'}.",
        "",
        "## Addition results",
        "",
        "| Arm | Validation accuracy by seed/interface | Special diagonal result |",
        "|---|---|---:|",
    ]
    for arm_name in ("A0_original", "A1_equality_pretrained", "A2_explicit_equality"):
        arm_ids = [arm_id for arm_id, arm in by_arm.items() if arm["arm"] == arm_name]
        values = [by_arm[arm_id]["validation"]["full_sequence_accuracy"] for arm_id in arm_ids]
        special_rows = [row for row in csv.DictReader((OUTPUT / "PREDICTIONS.csv").open()) if row["arm_id"] in arm_ids and row["split"] == "validation" and row["a"] == row["b"] and row["a"] in {"2", "3"}]
        special_correct = sum(row["output_token_correct"] == "True" for row in special_rows)
        report.append(f"| {arm_name} | {', '.join(f'{value:.1%}' for value in values)} | {special_correct}/{len(special_rows)} |")
    report += [
        "",
        "## Equality pretraining",
        "",
        "A1 received 500 equality-task updates before addition fine-tuning. Equality pretraining results and elapsed time are recorded in `ANALYSIS_FINAL.json`; its addition optimizer was reset before the 500 addition updates.",
        "",
        "A1 did not produce a reliable transfer benefit. One A1 arm failed the addition training-competence gate (75% training accuracy), while the other A1 arms fit but remained below the A0 diagonal baseline on validation.",
        "",
        "A2 supplied equality explicitly as a diagnostic upper-bound feature. It also did not restore the held-out diagonal transfer in this configuration.",
        "",
        "## Interpretation",
        "",
        "These results do not show that equality is irrelevant. They show that a short equality pretraining phase and a binary prompt feature were insufficient to produce reliable transfer under this task and budget. The next foundational target should be variable binding and relation-conditioned execution, not persistent memory or recursion yet.",
        "",
        "## Artifacts",
        "",
        "- [Final analysis](ANALYSIS_FINAL.json)",
        "- [Prediction matrix](PREDICTIONS_WITH_FIT.csv)",
        "- [Shared per-arm diagnostics](forensics/arm_outputs/)",
        "- [Heatmaps and confusion matrices](forensics/plots/)",
        "- [Protocol](../../docs/CRUCIBLE_EQUALITY_FOUNDATIONAL_PROTOCOL_20251025.md)",
        "- [Preflight](PREFLIGHT.json)",
        "- [Initial anchors](INITIAL_ANCHORS.json)",
    ]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "FINAL_HASHES.json":
            hashes[str(path.relative_to(ROOT))] = shared.sha256(path)
    write_json(OUTPUT / "FINAL_HASHES.json", hashes)
    write_json(OUTPUT / "FINALIZATION.json", {"status": "completed", "verification_pass": analysis["verification"]["pass"], "analysis_sha256": shared.sha256(OUTPUT / "ANALYSIS_FINAL.json"), "report_sha256": shared.sha256(OUTPUT / "REPORT_FINAL.md"), "prediction_matrix_sha256": shared.sha256(OUTPUT / "PREDICTIONS_WITH_FIT.csv"), "plot_count": len(list((OUTPUT / "forensics" / "plots").glob("*.svg")))})
    print(json.dumps({"status": "finalized", "fit_pass_arms": analysis["fit_pass_arms"], "verification_pass": analysis["verification"]["pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
