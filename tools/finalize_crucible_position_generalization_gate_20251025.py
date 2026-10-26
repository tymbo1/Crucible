#!/usr/bin/env python3
"""Add actual mixed-training paired metrics without retraining."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_position_generalization_gate_20251025_v1"
RUNNER = ROOT / "tools" / "run_crucible_position_generalization_gate_20251025.py"


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


runner = import_module("position_generalization_finished_runner", RUNNER)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report(summary: Mapping[str, Any]) -> str:
    layouts = runner.EVAL_LAYOUTS
    lines = [
        "# Crucible Position-Generalization Gate — Final Report",
        "",
        "Matched CPU training comparison. No sealed-test access.",
        "",
        "## Training arms",
        "",
        "| Arm | Interface | Seed | Actual train exact | Actual train paired | Validation field in arm result | Fit gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in summary["arms"]:
        actual = summary["actual_training"][result["arm_id"]]
        train = actual["by_split"]["train"]
        lines.append(f"| {result['arm']} | {result['interface']} | {result['seed']} | {train['accuracy']:.2%} | {train['paired_unequal']['both_correct']}/{train['paired_unequal']['pair_groups']} | {result['validation_exact']:.2%} | {'PASS' if result['fit_pass'] else 'FAIL'} |")
    lines += [
        "",
        "The position-varied training set contains fixed-near, early-padded and middle-position records in a deterministic 27/27/26 split. The actual-training column evaluates those mixed records, rather than substituting a single validation layout.",
        "",
        "## Layout evaluation on unseen operand pairs",
        "",
        "| Layout | Fixed-control validation | Varied-intervention validation | Fixed paired | Varied paired |",
        "|---|---:|---:|---:|---:|",
    ]
    for layout in layouts:
        fixed = [x for x in summary["evaluations"] if x["arm"] == "FIXED_POSITION_CONTROL" and x["layout"] == layout]
        varied = [x for x in summary["evaluations"] if x["arm"] == "POSITION_VARIED_INTERVENTION" and x["layout"] == layout]
        fixed_acc = sum(x["metrics"]["by_split"]["validation"]["accuracy"] for x in fixed) / len(fixed)
        varied_acc = sum(x["metrics"]["by_split"]["validation"]["accuracy"] for x in varied) / len(varied)
        fixed_pair = sum(x["metrics"]["by_split"]["validation"]["paired_unequal"]["both_correct_rate"] for x in fixed) / len(fixed)
        varied_pair = sum(x["metrics"]["by_split"]["validation"]["paired_unequal"]["both_correct_rate"] for x in varied) / len(varied)
        lines.append(f"| {layout} | {fixed_acc:.2%} | {varied_acc:.2%} | {fixed_pair:.2%} | {varied_pair:.2%} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The position-varied intervention did not pass the positional-robustness gate at this training budget: its actual training paired competence was incomplete in aggregate and its held-out paired performance remained near zero. This does not establish that position variation cannot work; it leaves optimization, capacity and curriculum effects unresolved.",
        "",
        "Attention summaries are descriptive shadow Q/K probabilities while native SDPA remains on the output path.",
        "",
        "## Artifacts",
        "",
        "- [Protocol](../../docs/CRUCIBLE_POSITION_GENERALIZATION_GATE_PROTOCOL_20251025.md)",
        "- [Summary](SUMMARY.json)",
        "- [Actual training metrics](ACTUAL_TRAINING_METRICS.json)",
        "- [Predictions](PREDICTIONS.csv)",
        "- [Attention summaries](ATTENTION_SUMMARIES.json)",
        "- [Finalization](FINALIZATION.json)",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    summary = json.loads((OUTPUT / "SUMMARY.json").read_text(encoding="utf-8"))
    actual: Dict[str, Any] = {}
    for result in summary["arms"]:
        arm = result["arm"]
        interface = result["interface"]
        seed = int(result["seed"])
        records = {
            "train": runner.load_jsonl(OUTPUT / "data" / arm / interface / f"seed{seed}" / "train.jsonl"),
            "validation": runner.load_jsonl(OUTPUT / "data" / arm / interface / f"seed{seed}" / "validation.jsonl"),
        }
        model = runner.load_model(OUTPUT / "arm_outputs" / arm / interface / f"seed{seed}" / "checkpoint.pt")
        _, metrics, _, _ = runner.adjacency.evaluate(model, records, {"arm": arm, "interface": interface, "seed": seed, "checkpoint": result["checkpoint"]}, "ACTUAL_TRAINING_LAYOUTS")
        actual[result["arm_id"]] = metrics
        del model
    summary["actual_training"] = actual
    summary["postprocessed_without_retraining"] = True
    write_json(OUTPUT / "ACTUAL_TRAINING_METRICS.json", actual)
    write_json(OUTPUT / "SUMMARY.json", summary)
    (OUTPUT / "REPORT_FINAL.md").write_text(report(summary), encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt":
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = json.loads((OUTPUT / "FINALIZATION.json").read_text(encoding="utf-8"))
    finalization["postprocessed_without_retraining"] = True
    finalization["report_sha256"] = sha256(OUTPUT / "REPORT_FINAL.md")
    finalization["summary_sha256"] = sha256(OUTPUT / "SUMMARY.json")
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "postprocessed", "output": str(OUTPUT.relative_to(ROOT)), "verification_pass": finalization["verification_pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
