#!/usr/bin/env python3
"""Run the preregistered CPU-only C0/C1 recursive-controller gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from recursive_controller import (  # noqa: E402
    POLICIES,
    ReasoningBudget,
    Scenario,
    UtilityConfig,
    aggregate_runs,
    canonical_json,
    calibration_metrics,
    make_scenario,
    run_c0_checks,
    run_policy,
    sha256_value,
)
from validate_cognitive_manifests import load_records, validate  # noqa: E402


DEFAULT_MANIFESTS = ROOT / "experiments" / "crucible_cognitive_eval_v2_20261021" / "manifests"
DEFAULT_OUTPUT = ROOT / "experiments" / "crucible_recursive_controller_c0_c1_20261021"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_c1(records: Iterable[Mapping[str, Any]], config: UtilityConfig, budget: ReasoningBudget) -> Dict[str, Any]:
    scenarios = [make_scenario(record) for record in records]
    all_runs: Dict[str, List[Any]] = {policy: [] for policy in POLICIES}
    traces: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for policy in POLICIES:
            run = run_policy(scenario, policy, budget=budget, engine=None)
            all_runs[policy].append(run)
            traces.append(run.to_dict())
    aggregates = {policy: aggregate_runs(runs) for policy, runs in all_runs.items()}

    shortcut_checks = []
    for scenario in scenarios:
        altered = Scenario(
            scenario_id=scenario.scenario_id,
            family=scenario.family,
            prompt=scenario.prompt + " [whitespace-only counterfactual]",
            gold_answer=scenario.gold_answer,
            wrong_answer=scenario.wrong_answer,
            initial_answer=scenario.initial_answer,
            initial_confidence=scenario.initial_confidence,
            category=scenario.category,
            candidate_revision_answer=scenario.candidate_revision_answer,
        )
        for policy in POLICIES:
            first = run_policy(scenario, policy, budget=budget)
            second = run_policy(altered, policy, budget=budget)
            first_actions = [transition.action for transition in first.transitions]
            second_actions = [transition.action for transition in second.transitions]
            shortcut_checks.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "policy": policy,
                    "invariant": first_actions == second_actions and first.final_state.answer == second.final_state.answer,
                }
            )

    simple_policies = [policy for policy in POLICIES if policy != "adaptive"]
    best_simple_accuracy = max(aggregates[policy]["accuracy"] for policy in simple_policies)
    adaptive = aggregates["adaptive"]
    policy_advantage = {
        "adaptive_accuracy": adaptive["accuracy"],
        "best_simple_accuracy": best_simple_accuracy,
        "adaptive_estimated_cost_units": adaptive["estimated_cost_units"],
        "best_simple_policy_by_accuracy": max(simple_policies, key=lambda policy: aggregates[policy]["accuracy"]),
        "status": (
            "not_established"
            if adaptive["accuracy"] <= best_simple_accuracy
            else "candidate_advantage_requires_independent_replay"
        ),
        "interpretation": "synthetic policy advantage is not evidence of cognitive superiority",
    }
    shortcut_passed = sum(item["invariant"] for item in shortcut_checks)
    return {
        "protocol": {
            "records": len(scenarios),
            "policies": list(POLICIES),
            "budget": budget.to_dict(),
            "utility_config": config.to_dict(),
            "candidate_engine": "deterministic imperfect fixture; target hidden from policy actions",
            "future_competence": "disabled",
        },
        "aggregates": aggregates,
        "policy_advantage": policy_advantage,
        "shortcut_audit": {
            "checks": len(shortcut_checks),
            "passed": shortcut_passed,
            "failed": len(shortcut_checks) - shortcut_passed,
            "all_passed": shortcut_passed == len(shortcut_checks),
        },
        "traces": traces,
    }


def markdown_summary(
    *,
    c0: Mapping[str, Any],
    c1: Mapping[str, Any],
    manifest_report: Mapping[str, Any],
    elapsed_seconds: float,
    hashes: Mapping[str, str],
    source_hashes: Mapping[str, str],
) -> str:
    lines = [
        "# Recursive reasoning controller C0/C1 results",
        "",
        "Date: 2026-10-21",
        "Execution boundary: CPU-only; no model, CUDA, network or cloud execution.",
        "",
        "## Verdict",
        "",
        f"- C0 contract gate: **{'PASS' if c0['passed'] else 'FAIL'}** ({c0['count']} invariant checks).",
        f"- C1 frozen-task replay: **{'PASS' if c1['shortcut_audit']['all_passed'] else 'FAIL'}** for shortcut invariance.",
        f"- Policy advantage: **{c1['policy_advantage']['status']}**; adaptive policy was not treated as superior by default.",
        f"- Wall time for this CPU run: {elapsed_seconds:.3f} seconds.",
        "",
        "C0 contract success and C1 policy advantage are separate claims. The synthetic replay validates implementation contracts and exposes failure modes; it does not establish cognitive superiority or future competence.",
        "",
        "## C0 contract checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for item in c0["checks"]:
        result = "PASS" if item["passed"] else f"FAIL: {item.get('error', 'unknown error')}"
        lines.append(f"| `{item['name']}` | {result} |")
    lines.extend(
        [
            "",
            "## Manifest and C1 protocol",
            "",
            f"- Manifest validator oracle checks: {manifest_report['oracle_checks']}",
            f"- Records replayed from frozen test split: {c1['protocol']['records']}",
            f"- Policies: {', '.join(c1['protocol']['policies'])}",
            f"- Shared remaining budget: `{json.dumps(c1['protocol']['budget'], sort_keys=True)}`",
            "- Future-competence term: disabled for C0/C1.",
            "- Candidate responses include clean, detectable-error, misleading-evidence, contradiction and shortcut-probe cases.",
            "",
            "## C1 policy results",
            "",
            "| Policy | Accuracy | Initial accuracy | Corrections | False corrections | Unresolved | Valid memory commits | Cost units |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy in POLICIES:
        item = c1["aggregates"][policy]
        lines.append(
            f"| `{policy}` | {item['accuracy']:.4f} | {item['initial_accuracy']:.4f} | "
            f"{item['corrections']} | {item['false_corrections']} | {item['unresolved']} | "
            f"{item['valid_memory_commits']} | {item['estimated_cost_units']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Required limitations",
            "",
            "- The candidate engine is deterministic and synthetic; it is not a language model.",
            "- The exact task oracle is used only by the independent verifier and evaluator, not by ordinary policy selection.",
            "- Confidence values are intentionally imperfect fixtures; calibration metrics are diagnostic and do not authorize entropy optimization.",
            "- No calibrated likelihood transition was supplied, so information gain was recorded as unavailable rather than fabricated.",
            "- The future-competence objective was disabled and no learning intervention occurred.",
            "- C1 results do not justify C2 model evaluation, neural training or a claim that adaptive control is beneficial.",
            "",
            "## Artifact hashes",
            "",
        ]
    )
    for path, digest in sorted(hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    lines.extend(["", "## Governing source hashes", ""])
    for path, digest in sorted(source_hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, default=DEFAULT_MANIFESTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    records = load_records(args.manifests)
    manifest_report = validate(records)
    c0 = run_c0_checks()
    config = UtilityConfig()
    budget = ReasoningBudget()
    c1 = run_c1(records["test"], config, budget)

    write_json(args.output / "C0_RESULTS.json", c0)
    c1_without_traces = {key: value for key, value in c1.items() if key != "traces"}
    write_json(args.output / "C1_RESULTS.json", c1_without_traces)
    with (args.output / "C1_TRACES.jsonl").open("w", encoding="utf-8") as handle:
        for trace in c1["traces"]:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
    write_json(args.output / "MANIFEST_VALIDATION.json", manifest_report)
    write_json(
        args.output / "RUN_CONFIG.json",
        {
            "date": "2026-10-21",
            "manifests": str(args.manifests.relative_to(ROOT)),
            "policy_order": list(POLICIES),
            "utility_config": config.to_dict(),
            "budget": budget.to_dict(),
            "future_competence": "disabled",
            "execution": "CPU-only; no model, CUDA, network or cloud execution",
        },
    )
    source_paths = [
        ROOT / "docs" / "CRUCIBLE_RECURSIVE_CONTROLLER_OPERATIONAL_SPEC_20261021.md",
        ROOT / "tools" / "recursive_controller.py",
        ROOT / "tools" / "run_recursive_controller_c0_c1.py",
        ROOT / "tools" / "test_recursive_controller.py",
        ROOT / "tools" / "validate_cognitive_manifests.py",
        args.manifests / "manifest.json",
        args.manifests / "manifest.sha256",
        args.manifests / "train.jsonl",
        args.manifests / "validation.jsonl",
        args.manifests / "test.jsonl",
    ]
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}
    write_json(args.output / "SOURCE_HASHES.json", source_hashes)
    hashes = {}
    for path in sorted(args.output.iterdir()):
        # The hash index is self-excluded, and the summary cannot contain its
        # own digest without a self-referential hash.  Its actual digest is
        # added to HASHES.json after the summary is written.
        if path.is_file() and path.name not in {"HASHES.json", "RESULTS_SUMMARY.md"}:
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(args.output / "HASHES.json", hashes)
    elapsed = perf_counter() - started
    summary = markdown_summary(
        c0=c0,
        c1=c1,
        manifest_report=manifest_report,
        elapsed_seconds=elapsed,
        hashes=hashes,
        source_hashes=source_hashes,
    )
    (args.output / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")
    hashes[str((args.output / "RESULTS_SUMMARY.md").relative_to(ROOT))] = sha256_file(args.output / "RESULTS_SUMMARY.md")
    write_json(args.output / "HASHES.json", hashes)
    print(summary)
    return 0 if c0["passed"] and c1["shortcut_audit"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
