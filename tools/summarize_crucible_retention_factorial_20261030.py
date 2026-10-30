#!/usr/bin/env python3
"""Summarize the completed exploratory retention factorial from evaluator traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def by_policy(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(path):
        grouped[str(row["policy"])][str(row["record_id"])] = row
    return dict(grouped)


def transitions(old: Mapping[str, Mapping[str, Any]], new: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ids = sorted(set(old) & set(new))
    if set(old) != set(new):
        raise RuntimeError(f"record-id mismatch: old={len(old)} new={len(new)} overlap={len(ids)}")
    overall = Counter()
    domain: dict[str, Counter] = defaultdict(Counter)
    for record_id in ids:
        before = bool(old[record_id]["verified"])
        after = bool(new[record_id]["verified"])
        label = ("correct" if before else "wrong") + "->" + ("correct" if after else "wrong")
        overall[label] += 1
        domain[str(old[record_id]["domain"])][label] += 1
        if old[record_id].get("target_normalized") != new[record_id].get("target_normalized"):
            raise RuntimeError(f"target mismatch for {record_id}")
    return {
        "records": len(ids),
        "overall": dict(sorted(overall.items())),
        "by_domain": {key: dict(sorted(value.items())) for key, value in sorted(domain.items())},
        "accuracy_before": sum(bool(old[item]["verified"]) for item in ids),
        "accuracy_after": sum(bool(new[item]["verified"]) for item in ids),
    }


def policy_score(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["policy"]: {
            "correct": row["verified_correct"],
            "records": row["records"],
            "accuracy": row["accuracy"],
            "mean_model_tokens": row["mean_model_tokens"],
            "by_domain": row["by_domain"],
        }
        for row in payload["policies"]
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Crucible 120M retention factorial — paired readout",
        "",
        "Classification: exploratory; all arms started from the untouched step-15,000 checkpoint and trained to step 20,000. The sealed 304-record manifest was opened only after training.",
        "",
        "## Final policy scores",
        "",
        "| Condition | One pass | Fixed depth 2 | External OP | Tokens/s | Peak allocated VRAM | Safety |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in report["arms"].items():
        scores = item["scores"]
        train = item["training"]
        lines.append(
            f"| {name} | {scores['one_pass']['correct']}/{scores['one_pass']['records']} | {scores['fixed_depth_2']['correct']}/{scores['fixed_depth_2']['records']} | {scores['op_external']['correct']}/{scores['op_external']['records']} | {train['tokens_per_second']:.1f} | {train['peak_vram_allocated'] / 1e9:.2f} GB | {item['safety']['status']} |"
        )
    lines += ["", "## Paired one-pass transitions versus step 15,000", "", "| Condition | Correct→Wrong | Wrong→Correct | Correct→Correct | Wrong→Wrong | Net change |", "|---|---:|---:|---:|---:|---:|"]
    for name, item in report["arms"].items():
        t = item["paired"]["one_pass"]
        overall = t["overall"]
        cw = overall.get("correct->wrong", 0)
        wc = overall.get("wrong->correct", 0)
        cc = overall.get("correct->correct", 0)
        ww = overall.get("wrong->wrong", 0)
        lines.append(f"| {name} | {cw} | {wc} | {cc} | {ww} | {wc - cw:+d} |")
    lines += ["", "## Interpretation", "", "- The protected one-pass gate is zero correct→wrong transitions relative to step 15,000; passing it requires preserving the tested competency, not merely improving aggregate accuracy.", "- Aggregate score changes are not treated as causal evidence until paired transitions and domain distributions are considered.", "- A3 external OP is an inference-time result and does not establish that the retention loss improved the learned model.", "- This run tests a five-thousand-update continuation, one seed and a finite protected manifest; it is not evidence of lifelong retention or universal capability preservation.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-traces", type=Path, required=True)
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--arm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = by_policy(args.baseline_traces.resolve())
    report: Dict[str, Any] = {
        "schema_version": "crucible-retention-factorial-readout-v1-20261030",
        "classification": "exploratory_retention_factorial_paired_readout",
        "baseline": {"eval": str(args.baseline_eval.resolve()), "scores": policy_score(args.baseline_eval.resolve())},
        "arms": {},
        "controls": {
            "same_test_manifest": True,
            "sealed_target_used_for_training": False,
            "start_step": 15000,
            "end_step": 20000,
            "one_pass_protected_gate": "zero correct->wrong transitions relative to step-15,000 checkpoint",
        },
    }
    for name in ("A0", "A1", "A2", "A3"):
        root = args.arm_root.resolve() / name
        eval_path = root / "test_eval" / "CHECKPOINT_OP_EVAL.json"
        traces = by_policy(root / "test_eval" / "CHECKPOINT_OP_TRACES.jsonl")
        train = json.loads((root / "RETENTION_TRAINING_REPORT.json").read_text(encoding="utf-8"))
        paired = {policy: transitions(baseline[policy], traces[policy]) for policy in sorted(baseline)}
        report["arms"][name] = {
            "scores": policy_score(eval_path),
            "paired": paired,
            "training": {
                "tokens_per_second": train["training"]["tokens_per_second"],
                "training_tokens_this_run": train["training"]["training_tokens_this_run"],
                "peak_vram_allocated": train["training"]["peak_vram_allocated"],
                "peak_vram_reserved": train["training"]["peak_vram_reserved"],
                "finite": train["safety"]["checkpoint_model_optimizer_finite_before_atomic_commit"],
                "loss_scale_recovery_events": train["safety"]["loss_scale_recovery_events"],
            },
            "safety": {"status": "PASS" if train["safety"]["checkpoint_model_optimizer_finite_before_atomic_commit"] and train["safety"]["loss_scale_recovery_events"] == 0 else "REVIEW"},
        }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "RETENTION_FACTORIAL_PAIRED_READOUT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "RETENTION_FACTORIAL_PAIRED_READOUT.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
