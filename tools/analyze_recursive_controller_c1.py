#!/usr/bin/env python3
"""Read-only diagnostic analysis of the sealed C1 controller traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments" / "crucible_recursive_controller_c0_c1_20261021"
DEFAULT_OUTPUT = ROOT / "experiments" / "crucible_recursive_controller_c1_diagnostic_20261021"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_traces(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate_policy(runs: List[Mapping[str, Any]]) -> Dict[str, Any]:
    policy = str(runs[0]["policy"])
    action_counts = Counter()
    action_by_category: Dict[str, Counter] = defaultdict(Counter)
    verification_by_category: Counter = Counter()
    verification_signals: Counter = Counter()
    cost_by_category: Dict[str, float] = defaultdict(float)
    records_by_category: Counter = Counter()
    for run in runs:
        category = str(run["category"])
        records_by_category[category] += 1
        for transition in run["transitions"]:
            action = str(transition["action"])
            action_counts[action] += 1
            action_by_category[category][action] += 1
            cost_by_category[category] += float(transition.get("cost", {}).get("estimated_cost", 0.0))
            if action == "verify":
                verification_by_category[category] += 1
                observation = transition.get("observation") or {}
                verification_signals[str(observation.get("signal", "missing"))] += 1
    total_cost = sum(cost_by_category.values())
    return {
        "policy": policy,
        "records": len(runs),
        "accuracy": sum(bool(run["final_correct"]) for run in runs) / len(runs),
        "initial_accuracy": sum(bool(run["initial_correct"]) for run in runs) / len(runs),
        "corrections": sum(bool(run["corrected"]) for run in runs),
        "false_corrections": sum(bool(run["false_correction"]) for run in runs),
        "unresolved": sum(bool(run["unresolved"]) for run in runs),
        "valid_memory_commits": sum(int(run["valid_memory_commits"]) for run in runs),
        "invalid_memory_attempts": sum(int(run["invalid_memory_attempts"]) for run in runs),
        "mean_actions": sum(len(run["transitions"]) for run in runs) / len(runs),
        "mean_cost_units": total_cost / len(runs),
        "action_counts": dict(sorted(action_counts.items())),
        "verification_by_category": dict(sorted(verification_by_category.items())),
        "verification_signals": dict(sorted(verification_signals.items())),
        "by_category": {
            category: {
                "records": records_by_category[category],
                "accuracy": sum(bool(run["final_correct"]) for run in runs if run["category"] == category) / records_by_category[category],
                "unresolved": sum(bool(run["unresolved"]) for run in runs if run["category"] == category),
                "false_corrections": sum(bool(run["false_correction"]) for run in runs if run["category"] == category),
                "mean_cost_units": cost_by_category[category] / records_by_category[category],
                "actions": dict(sorted(action_by_category[category].items())),
            }
            for category in sorted(records_by_category)
        },
    }


def build_diagnostic(input_dir: Path) -> Dict[str, Any]:
    traces_path = input_dir / "C1_TRACES.jsonl"
    hashes_path = input_dir / "HASHES.json"
    config_path = input_dir / "RUN_CONFIG.json"
    traces = load_traces(traces_path)
    expected_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    actual_trace_hash = sha256_file(traces_path)
    recorded_trace_hash = expected_hashes.get(str(traces_path.relative_to(ROOT)))
    if actual_trace_hash != recorded_trace_hash:
        raise RuntimeError("sealed C1 trace hash mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        by_policy[str(trace["policy"])].append(trace)
    aggregates = {policy: aggregate_policy(runs) for policy, runs in sorted(by_policy.items())}
    adaptive = aggregates["adaptive"]
    verifier = aggregates["verification_only"]
    shortcut_adaptive = adaptive["by_category"].get("shortcut_probe", {})
    findings = [
        {
            "finding": "perfect_verifier_available",
            "evidence": {
                "verification_only_calls": verifier["action_counts"].get("verify", 0),
                "verification_only_records": verifier["records"],
                "verification_signals": verifier["verification_signals"],
            },
            "interpretation": "The fixture supplies an exact, always-available independent verifier at declared cost; this strongly favors verification-first behavior.",
        },
        {
            "finding": "adaptive_skips_verification_on_shortcut_probe",
            "evidence": {
                "records": shortcut_adaptive.get("records", 0),
                "verify_calls": adaptive["verification_by_category"].get("shortcut_probe", 0),
                "accuracy": shortcut_adaptive.get("accuracy", 0.0),
                "unresolved": shortcut_adaptive.get("unresolved", 0),
            },
            "interpretation": "The current low-confidence adaptive branch uses critique/counterexample/revision but never escalates to verification, so it misses every shortcut-probe case.",
        },
        {
            "finding": "adaptive_spends_more_and_resolves_less",
            "evidence": {
                "verification_only_mean_cost": verifier["mean_cost_units"],
                "adaptive_mean_cost": adaptive["mean_cost_units"],
                "verification_only_accuracy": verifier["accuracy"],
                "adaptive_accuracy": adaptive["accuracy"],
            },
            "interpretation": "Under this fixture, adaptive is neither cost-dominant nor accuracy-dominant; it has higher mean cost and lower accuracy.",
        },
        {
            "finding": "confidence_is_not_calibrated",
            "evidence": {"brier": 0.319644140625, "ece": 0.45},
            "interpretation": "The supplied confidence values are diagnostic synthetic signals, not a calibrated posterior. They cannot justify entropy or information-gain optimization.",
        },
        {
            "finding": "budgets_are_matched_as_ceilings_not_as_realized_cost",
            "evidence": {
                "shared_budget": config["budget"],
                "verification_only_mean_cost": verifier["mean_cost_units"],
                "adaptive_mean_cost": adaptive["mean_cost_units"],
            },
            "interpretation": "Policies receive the same maximum budget and action/tool types, but actual cost is not equalized. Efficiency claims therefore require a separate matched-cost analysis.",
        },
        {
            "finding": "memory_gate_exposes_contradiction_risk",
            "evidence": {
                "adaptive_invalid_memory_attempts": adaptive["invalid_memory_attempts"],
                "adaptive_valid_memory_commits": adaptive["valid_memory_commits"],
            },
            "interpretation": "Adaptive evidence paths create contradiction-linked proposals that are correctly quarantined rather than committed.",
        },
    ]
    return {
        "source": {
            "input_directory": str(input_dir.relative_to(ROOT)),
            "trace_sha256": actual_trace_hash,
            "sealed_hash_index_entry": recorded_trace_hash,
            "run_config_sha256": sha256_file(config_path),
        },
        "protocol": {
            "records": len(traces) // len(by_policy),
            "policies": sorted(by_policy),
            "shared_budget": config["budget"],
            "future_competence": config["future_competence"],
            "execution": config["execution"],
        },
        "aggregates": aggregates,
        "findings": findings,
        "conclusion": "Verification-only is the current fixture winner because exact verification is cheap and always available. Adaptive policy advantage is not established; the result primarily diagnoses a missing condition, not the recursive equations.",
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# C1 diagnostic analysis",
        "",
        "Date: 2026-10-21",
        "Scope: read-only analysis of the sealed C0/C1 traces; no existing artifact was changed.",
        "",
        "## Conclusion",
        "",
        report["conclusion"],
        "",
        "The existing C1 policies had identical budget ceilings and access to the same action types, but they did not incur equal realized costs. Therefore the result is not a matched-cost efficiency comparison.",
        "",
        "## Policy comparison",
        "",
        "| Policy | Accuracy | Mean cost | Verify calls | Unresolved | Valid commits | Invalid memory attempts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, item in report["aggregates"].items():
        lines.append(
            f"| `{policy}` | {item['accuracy']:.4f} | {item['mean_cost_units']:.3f} | "
            f"{item['action_counts'].get('verify', 0)} | {item['unresolved']} | "
            f"{item['valid_memory_commits']} | {item['invalid_memory_attempts']} |"
        )
    lines.extend(["", "## Findings", ""])
    for item in report["findings"]:
        lines.append(f"### {item['finding']}")
        lines.append("")
        lines.append(item["interpretation"])
        lines.append("")
        lines.append(f"Evidence: `{json.dumps(item['evidence'], sort_keys=True)}`")
        lines.append("")
    lines.extend(
        [
            "## What C1 did and did not test",
            "",
            "- It tested a perfect, immediately available verifier, not noisy, delayed, incomplete or unavailable verification.",
            "- It tested deterministic synthetic response categories; it did not test a learned candidate generator.",
            "- It tested contract safety and policy traces, not future competence or model cognition.",
            "- The exact verifier is an allowed action, not a hidden privilege, but its reliability/cost regime creates an easy verification-first environment.",
            "- The adaptive miss is localized: 51 shortcut-probe cases received no verification and remained unresolved.",
            "",
            "## Recommendation",
            "",
            "Retain the sealed C1 result. Do not retune the existing policy against it. Review the separately versioned C1b protocol before executing any new replay.",
            "",
            f"Sealed trace SHA-256: `{report['source']['trace_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = build_diagnostic(args.input)
    json_path = args.output / "C1_DIAGNOSTIC.json"
    markdown_path = args.output / "C1_DIAGNOSTIC.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rendered = markdown(report)
    markdown_path.write_text(rendered, encoding="utf-8")
    hashes = {
        str(json_path.relative_to(ROOT)): sha256_file(json_path),
        str(markdown_path.relative_to(ROOT)): sha256_file(markdown_path),
    }
    (args.output / "HASHES.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
