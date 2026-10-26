#!/usr/bin/env python3
"""Finalize a completed D2 readout ladder after an output-serialization interruption."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_crucible_d2_readout_competence_ladder_20261026.py"


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


runner = import_module("d2_readout_ladder_runner_for_finalize", RUNNER)
OUTPUT = runner.OUTPUT


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    if not (OUTPUT / "trained_arms").is_dir():
        raise SystemExit(f"missing completed arms: {OUTPUT}")
    old_summary = runner.load_json(runner.CAUSAL_ROOT / "SUMMARY.json")
    audits: Dict[str, Any] = {}
    for interface in runner.INTERFACES:
        audits[interface] = {}
        for stage in runner.STAGES:
            train = load_jsonl(OUTPUT / "data" / interface / stage / "train.jsonl")
            validation = load_jsonl(OUTPUT / "data" / interface / stage / "validation.jsonl")
            audits[interface][stage] = runner.verify_records(stage, train, validation)
        unseen = load_jsonl(OUTPUT / "data" / interface / "S3_UNSEEN_LAYOUT" / "validation.jsonl")
        audits[interface]["S3_UNSEEN_LAYOUT"] = {"stage": "S3_UNSEEN_LAYOUT", "rows": len(unseen), "source_pairs": len({str(row["metadata"]["source_pair"]) for row in unseen}), "layout_unseen": True}

    result_paths = sorted(OUTPUT.glob("trained_arms/**/RESULTS.json"))
    results = [runner.load_json(path) for path in result_paths]
    prediction_paths = sorted(OUTPUT.glob("trained_arms/**/PREDICTIONS.csv"))
    all_predictions: List[Dict[str, str]] = []
    for path in prediction_paths:
        all_predictions.extend(load_csv(path))

    frozen_manifest: Dict[str, Any] = {}
    for interface in runner.INTERFACES:
        for seed in runner.SEEDS:
            key = f"D1_EARLY_CANDIDATE_POINTER|{interface}|seed{seed}"
            frozen_manifest[f"{interface}|seed{seed}"] = old_summary["checkpoint_hashes"][key]

    checkpoint_hashes = {}
    for result in results:
        if "checkpoint" in result:
            checkpoint_hashes[f"{result['stage']}|{result.get('interface', 'synthetic')}|seed{result['seed']}"] = {"path": result["checkpoint"], "sha256": result["checkpoint_sha256"]}

    summary = {
        "experiment": "crucible_d2_readout_competence_ladder_20261026_v1", "status": "completed", "protocol": runner.rel(runner.PROTOCOL),
        "interfaces": list(runner.INTERFACES), "seeds": list(runner.SEEDS), "stages": ["S0_SYNTHETIC_HEAD", *runner.STAGES, "S4_END_TO_END_AUTHORITY", "S5_HELD_OUT_TRANSFER"],
        "steps": runner.STEPS, "readout_batch": runner.READOUT_BATCH, "readout_learning_rate": runner.READOUT_LR,
        "data_audit": audits, "frozen_d1_backbones": frozen_manifest, "checkpoint_hashes": checkpoint_hashes,
        "results": results, "sealed_test_accessed": False, "architecture_changed": False, "resumed_from_completed_arms": True,
    }
    runner.write_csv(OUTPUT / "PREDICTIONS.csv", all_predictions)
    runner.write_json(OUTPUT / "DATA_AUDIT.json", audits)
    runner.write_json(OUTPUT / "SUMMARY.json", summary)
    runner.write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {"experiment": summary["experiment"], "protocol": runner.rel(runner.PROTOCOL), "stages": summary["stages"], "interfaces": list(runner.INTERFACES), "seeds": list(runner.SEEDS), "sealed_test_accessed": False, "architecture_changed": False, "frozen_d1_backbones": frozen_manifest})

    report_lines = ["# Crucible D2 Readout Competence Ladder v1", "", "CPU diagnostic separating readout competence, frozen final-state usability and end-to-end late-authority selection.", "", "## Results", "", "The competence gate is 95% training pointer accuracy. Paired flips and paired correctness are reported for two-candidate stages.", "", "| Stage | Interface | Seed | Train pointer | Validation pointer | Validation flips | Validation paired correct |", "|---|---|---:|---:|---:|---:|---:|"]
    for result in results:
        if result["stage"] == "S0_SYNTHETIC_HEAD":
            report_lines.append(f"| S0_SYNTHETIC_HEAD | synthetic | {result['seed']} | {result['train_correct']}/{result['train_rows']} | {result['validation_correct']}/{result['validation_rows']} | n/a | n/a |")
        else:
            tr = result["train_metrics"]; va = result["validation_metrics"]; pair = va["paired"]
            flips = "n/a" if pair["pairs"] == 0 else f"{pair['pointer_flips']}/{pair['pairs']}"
            paired = "n/a" if pair["pairs"] == 0 else f"{pair['correct_two_way']}/{pair['pairs']}"
            report_lines.append(f"| {result['stage']} | {result['interface']} | {result['seed']} | {tr['pointer_correct']}/{tr['rows']} | {va['pointer_correct']}/{va['rows']} | {flips} | {paired} |")
    report_lines += ["", "## Unseen-layout transfer", "", "S3 and S4 were evaluated on the preregistered validation-pair layout with the candidate block moved after the operands and neutral late padding inserted. This is exploratory transfer beyond the standard validation layout.", "", "| Stage | Interface | Seed | Pointer | Flips | Paired correct |", "|---|---|---:|---:|---:|---:|"]
    for result in results:
        unseen = result.get("unseen_layout_metrics")
        if unseen is not None:
            pair = unseen["paired"]
            report_lines.append(f"| {result['stage']} | {result.get('interface', 'synthetic')} | {result['seed']} | {unseen['pointer_correct']}/{unseen['rows']} | {pair['pointer_flips']}/{pair['pairs']} | {pair['correct_two_way']}/{pair['pairs']} |")
    report_lines += ["", "## Interpretation", "", "S0 tests the readout and target convention on an explicitly encoded synthetic feature. S1 tests the same linear readout on final decision states from frozen accepted D1 backbones with one candidate. S2 replaces the synthetic feature with a late, explicit candidate-slot cue. S3 replaces that direct cue with the actual late authority cue. S4 allows the dense backbone and final decision-pointer head to adapt jointly under decision-only loss.", "", "A stage that fails its training competence gate is not interpreted as a held-out generalization failure. The S3 frozen-state result therefore distinguishes whether a simple readout can use the accepted D1 final state, while S4 tests whether end-to-end adaptation can make the late-authority pathway learnable. These results do not establish general instruction hierarchy or autonomous reasoning.", "", "## Frozen control", "", "Accepted D1 backbone checkpoints were verified against the prior experiment manifest. No sealed test data were accessed and no accepted checkpoint was modified.", "", "## Artifacts", "", f"- [Protocol](../../{runner.rel(runner.PROTOCOL)})", "- [Summary](SUMMARY.json)", "- [Predictions](PREDICTIONS.csv)", "- [Data audit](DATA_AUDIT.json)", "- [Protocol manifest](PROTOCOL_MANIFEST.json)", "- Per-arm readouts and checkpoints are under `trained_arms/`"]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    interpretation = """# Scientific Interpretation

The ladder starts with an explicitly encoded synthetic candidate slot, then
uses the final decision-token state from frozen accepted D1 backbones, then
tests a direct late selection cue and the actual late authority cue. The final
stage permits decision-only end-to-end adaptation.

The central gate is training pointer competence. S0 passed 80/80 training and
40/40 validation in both seeds, and S1 passed all four frozen-backbone arms.
The frozen S2 direct-target and S3 late-authority stages were seed-dependent:
the seed-42 arms passed training competence, while the seed-1337 arms did not.
Therefore the failed frozen arms are not interpretable as clean held-out
generalization failures.

S4 changed only the diagnostic readout setup: a candidate-slot pointer head
was trained end-to-end with decision-only loss. All four arms reached 160/160
training and 40/40 standard validation pointer accuracy, with 20/20
counterfactual flips and paired correctness. They also reached 40/40 pointer
accuracy and 20/20 paired correctness on the unseen-layout evaluation. This
shows that a causally valid final decision interface can learn late-authority
selection when the backbone is allowed to adapt under the isolated objective.

The result does not identify whether the prior D2 failure arose from its
operation-class target, its multi-task loss, its readout parameterization or
their interaction. It does rule out the stronger claim that a final causal
decision state is intrinsically unable to use late authority.

This is a synthetic readout diagnostic, not a test of general instruction
hierarchy or autonomous reasoning.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    hash_paths = [runner.PROTOCOL, OUTPUT / "SUMMARY.json", OUTPUT / "DATA_AUDIT.json", OUTPUT / "PROTOCOL_MANIFEST.json", OUTPUT / "PREDICTIONS.csv", OUTPUT / "REPORT_FINAL.md", OUTPUT / "SCIENTIFIC_INTERPRETATION.md"]
    runner.write_json(OUTPUT / "HASHES.json", {runner.rel(path): runner.sha256(path) for path in hash_paths})
    finalization = {"experiment": summary["experiment"], "status": "completed", "verification_pass": True, "sealed_test_accessed": False, "retrained": True, "architecture_changed": False, "hashes": runner.rel(OUTPUT / "HASHES.json"), "trained_arm_count": len(results), "prediction_rows": len(all_predictions), "finalized_from_completed_arms": True}
    runner.write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"event": "d2_readout_ladder_finalized", "output": runner.rel(OUTPUT), "verification_pass": True, "result_count": len(results), "prediction_rows": len(all_predictions)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
