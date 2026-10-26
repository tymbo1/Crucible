#!/usr/bin/env python3
"""Run C1r: prediction and intervention testing on CPU."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_crucible_c1n_relational_transfer_20261021 as c1n


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_c1r_prediction_intervention_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
C1P_OUTPUT = ROOT / "experiments" / "crucible_c1p_active_information_20261021"
C1Q_OUTPUT = ROOT / "experiments" / "crucible_c1q_evidence_corroboration_20261021"
EXPERIMENT = "crucible_c1r_prediction_intervention_20261021"
TASKS_PER_SEED = 12_000
DISCOVERY_FRACTION = 0.25
TASK_SEEDS = (42, 1337, 2026)
CONDITIONS = (
    "correlated_misinformation",
    "prediction_only",
    "intervention_only",
    "inconclusive_test",
    "misleading_prediction",
    "misleading_intervention",
    "truthful_consensus",
    "source_conflict",
    "unavailable_tests",
    "role_break",
)
TEMPLATES = ("balanced", "asymmetric", "deep", "wide")
POLICIES = ("corroboration_only", "prediction_testing", "intervention_testing")
BASE_COST = 0.15
SOURCE_A_COST = 0.20
SOURCE_B_COST = 0.25
SOURCE_C_COST = 0.80
PREDICTION_COST = 0.35
INTERVENTION_COST = 0.50


@dataclass(frozen=True)
class TestingTask:
    task_id: str
    seed: int
    index: int
    phase: str
    condition: str
    template: str
    source_tree: Dict[str, Any]
    target_tree: Dict[str, Any]
    source_rule: Optional[str]
    candidate_rule: Optional[str]
    alternate_rule: Optional[str]
    true_rule: Optional[str]
    source_relational_signature: Tuple[Any, ...]
    target_relational_signature: Tuple[Any, ...]
    sources: List[Dict[str, Any]]
    tests: Dict[str, Dict[str, Any]]
    target_output: Optional[int]
    valid_transfer: bool
    truthful_test_available: bool
    oracle_source_kinds: Dict[str, str]
    oracle_test_kinds: Dict[str, str]


def stable_int(*parts: Any) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:16], 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_index(index_path: Path) -> Dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in sorted(index.items()):
        path = ROOT / relative
        actual = "missing" if not path.exists() else sha256_file(path)
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    if failures:
        raise RuntimeError(f"Frozen artifact hash failure: {index_path}: {failures}")
    return {
        "index": str(index_path.relative_to(ROOT)),
        "indexed_files": len(index),
        "hash_failures": 0,
        "index_sha256": sha256_file(index_path),
    }


def clone_tree(tree: Mapping[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(tree, sort_keys=True))


def children(tree: Mapping[str, Any]) -> Dict[int, List[int]]:
    return c1n.children(tree)


def role_position(tree: Mapping[str, Any], role: str) -> Optional[str]:
    return c1n.role_position(tree, role)


def apply_rule(tree: Mapping[str, Any], rule: str) -> int:
    return c1n.apply_rule(tree, rule)


def build_source_tree(seed: int, index: int, template: str) -> Dict[str, Any]:
    tree = c1n.build_source_tree(seed, index, template)
    tree["c1r_family"] = "hypothesis_consequence_testing"
    return tree


def target_tree_for(source: Mapping[str, Any], condition: str) -> Dict[str, Any]:
    target = c1n.relabel_isomorphism(source, 0, 0)
    if condition == "role_break":
        node_by_id = {int(node["id"]): node for node in target["nodes"]}
        anchor_position = role_position(target, "anchor")
        root_children = children(target).get(int(target["root"]), [])
        branch_id = root_children[0 if anchor_position == "left" else 1]
        node_by_id[branch_id]["role"] = "distractor"
    return target


def rule_for_role(tree: Mapping[str, Any], role: str) -> Optional[str]:
    position = role_position(tree, role)
    return None if position is None else f"{position}_leaf_count"


def make_equal_tree(target: Mapping[str, Any]) -> Dict[str, Any]:
    root = int(target["root"])
    target_children = children(target).get(root, [])[:2]
    node_by_id = {int(node["id"]): node for node in target["nodes"]}
    nodes = [{"id": root, "role": "root", "label": 1, "marker": 0}]
    edges: List[List[int]] = []
    next_id = root + 1
    for branch_id in target_children:
        new_branch = next_id
        next_id += 1
        source_node = node_by_id[branch_id]
        nodes.append({"id": new_branch, "role": source_node["role"], "label": source_node.get("label", 0), "marker": 0})
        edges.append([root, new_branch])
        leaf_id = next_id
        next_id += 1
        nodes.append({"id": leaf_id, "role": "leaf", "label": 4, "marker": 0})
        edges.append([new_branch, leaf_id])
    return {"root": root, "nodes": nodes, "edges": edges, "hidden_mode": target.get("hidden_mode", "anchor")}


def make_intervention_tree(target: Mapping[str, Any], candidate: Optional[str]) -> Dict[str, Any]:
    result = clone_tree(target)
    if candidate is None:
        return result
    root_children = children(result).get(int(result["root"]), [])
    position = "left" if candidate.startswith("left") else "right"
    branch_id = root_children[0 if position == "left" else 1]
    next_id = max(int(node["id"]) for node in result["nodes"]) + 1
    result["nodes"].append({"id": next_id, "role": "leaf", "label": 8, "marker": 0})
    result["edges"].append([branch_id, next_id])
    result["intervention"] = {"operation": "add_leaf", "position": position}
    return result


def source_record(name: str, group: str, cost: float, tree: Mapping[str, Any], candidate: Optional[str], alternate: Optional[str], observed: Optional[int], available: bool) -> Dict[str, Any]:
    candidate_prediction = None if candidate is None else apply_rule(tree, candidate)
    alternate_prediction = None if alternate is None else apply_rule(tree, alternate)
    return {
        "name": name,
        "group": group,
        "cost": cost,
        "tree": clone_tree(tree),
        "available": available,
        "observed_output": observed,
        "candidate_prediction": candidate_prediction,
        "alternate_prediction": alternate_prediction,
        "informative": candidate_prediction is not None and alternate_prediction is not None and candidate_prediction != alternate_prediction,
    }


def test_record(name: str, kind: str, cost: float, tree: Mapping[str, Any], candidate: Optional[str], alternate: Optional[str], observed: Optional[int], available: bool) -> Dict[str, Any]:
    candidate_prediction = None if candidate is None else apply_rule(tree, candidate)
    alternate_prediction = None if alternate is None else apply_rule(tree, alternate)
    return {
        "name": name,
        "kind": kind,
        "cost": cost,
        "tree": clone_tree(tree),
        "available": available,
        "observed_output": observed,
        "candidate_prediction": candidate_prediction,
        "alternate_prediction": alternate_prediction,
        "informative": candidate_prediction is not None and alternate_prediction is not None and candidate_prediction != alternate_prediction,
    }


def build_task(seed: int, index: int) -> TestingTask:
    condition = CONDITIONS[stable_int("c1r-condition", seed, index) % len(CONDITIONS)]
    template = TEMPLATES[stable_int("c1r-template", seed, index) % len(TEMPLATES)]
    source = build_source_tree(seed, index, template)
    target = target_tree_for(source, condition)
    source_rule = rule_for_role(source, "anchor")
    candidate_rule = rule_for_role(target, "anchor")
    alternate_rule = rule_for_role(target, "support")
    true_mode = "anchor" if stable_int("c1r-true-mode", seed, index) % 2 == 0 else "support"
    if condition == "role_break":
        true_mode = "support"
    true_rule = None if alternate_rule is None else (candidate_rule if true_mode == "anchor" else alternate_rule)
    target_output = None if true_rule is None else apply_rule(target, true_rule)
    prediction_tree = c1n.relabel_isomorphism(target, 7, index)
    intervention_tree = make_intervention_tree(target, candidate_rule)
    inconclusive_tree = make_equal_tree(target)
    oracle_source_kinds: Dict[str, str] = {}
    oracle_test_kinds: Dict[str, str] = {}

    def source_observation(kind: str, tree: Mapping[str, Any]) -> Optional[int]:
        if not candidate_rule or not alternate_rule:
            return None
        if kind == "truthful":
            return apply_rule(tree, true_rule) if true_rule else None
        return apply_rule(tree, candidate_rule if true_mode == "support" else alternate_rule)

    if condition in {"truthful_consensus"}:
        source_specs = [("A1", "A", SOURCE_A_COST, True, "truthful"), ("B1", "B", SOURCE_B_COST, True, "truthful")]
    elif condition == "source_conflict":
        source_specs = [("A1", "A", SOURCE_A_COST, True, "truthful"), ("B1", "B", SOURCE_B_COST, True, "misleading")]
    elif condition == "role_break":
        source_specs = []
    else:
        source_specs = [("A1", "A", SOURCE_A_COST, True, "misleading"), ("B1", "B", SOURCE_B_COST, True, "misleading")]
    sources: List[Dict[str, Any]] = []
    for name, group, cost, available, kind in source_specs:
        sources.append(source_record(name, group, cost, target, candidate_rule, alternate_rule, source_observation(kind, target) if available else None, available))
        oracle_source_kinds[name] = kind

    if condition == "correlated_misinformation":
        test_specs = [
            ("prediction", "truthful", PREDICTION_COST, prediction_tree, True),
            ("intervention", "truthful", INTERVENTION_COST, intervention_tree, True),
        ]
    elif condition == "prediction_only":
        test_specs = [
            ("prediction", "truthful", PREDICTION_COST, prediction_tree, True),
            ("intervention", "inconclusive", INTERVENTION_COST, inconclusive_tree, True),
        ]
    elif condition == "intervention_only":
        test_specs = [
            ("prediction", "inconclusive", PREDICTION_COST, inconclusive_tree, True),
            ("intervention", "truthful", INTERVENTION_COST, intervention_tree, True),
        ]
    elif condition == "inconclusive_test":
        test_specs = [
            ("prediction", "inconclusive", PREDICTION_COST, inconclusive_tree, True),
            ("intervention", "inconclusive", INTERVENTION_COST, inconclusive_tree, True),
        ]
    elif condition == "misleading_prediction":
        test_specs = [
            ("prediction", "misleading", PREDICTION_COST, prediction_tree, True),
            ("intervention", "inconclusive", INTERVENTION_COST, inconclusive_tree, True),
        ]
    elif condition == "misleading_intervention":
        test_specs = [
            ("prediction", "inconclusive", PREDICTION_COST, inconclusive_tree, True),
            ("intervention", "misleading", INTERVENTION_COST, intervention_tree, True),
        ]
    elif condition == "truthful_consensus":
        test_specs = [
            ("prediction", "truthful", PREDICTION_COST, prediction_tree, True),
            ("intervention", "truthful", INTERVENTION_COST, intervention_tree, True),
        ]
    elif condition == "source_conflict":
        test_specs = [
            ("prediction", "truthful", PREDICTION_COST, prediction_tree, True),
            ("intervention", "truthful", INTERVENTION_COST, intervention_tree, True),
        ]
    elif condition == "unavailable_tests":
        test_specs = [
            ("prediction", "unavailable", PREDICTION_COST, prediction_tree, False),
            ("intervention", "unavailable", INTERVENTION_COST, intervention_tree, False),
        ]
    elif condition == "role_break":
        test_specs = []
    else:
        raise ValueError(condition)

    tests: Dict[str, Dict[str, Any]] = {}
    for name, kind, cost, tree, available in test_specs:
        if not available:
            observed = None
        elif kind == "truthful":
            observed = apply_rule(tree, true_rule) if true_rule else None
        elif kind == "misleading":
            observed = apply_rule(tree, candidate_rule) if candidate_rule else None
        else:
            observed = apply_rule(tree, candidate_rule) if candidate_rule else None
        tests[name] = test_record(name, kind, cost, tree, candidate_rule, alternate_rule, observed, available)
        oracle_test_kinds[name] = kind
    phase = "discovery" if index < int(TASKS_PER_SEED * DISCOVERY_FRACTION) else "heldout"
    truthful_test_available = any(item["available"] and item["informative"] and item["kind"] == "truthful" for item in tests.values())
    return TestingTask(
        task_id=f"c1r-{seed}-{index}-{condition}-{template}",
        seed=seed,
        index=index,
        phase=phase,
        condition=condition,
        template=template,
        source_tree=source,
        target_tree=target,
        source_rule=source_rule,
        candidate_rule=candidate_rule,
        alternate_rule=alternate_rule,
        true_rule=true_rule,
        source_relational_signature=c1n.relational_signature(source),
        target_relational_signature=c1n.relational_signature(target),
        sources=sources,
        tests=tests,
        target_output=target_output,
        valid_transfer=true_mode == "anchor" and candidate_rule is not None,
        truthful_test_available=truthful_test_available,
        oracle_source_kinds=oracle_source_kinds,
        oracle_test_kinds=oracle_test_kinds,
    )


def generate_tasks(seed: int) -> List[TestingTask]:
    return [build_task(seed, index) for index in range(TASKS_PER_SEED)]


def oracle_check(task: TestingTask) -> bool:
    if task.condition != "role_break" and task.candidate_rule is None:
        return False
    if task.true_rule is None:
        return task.condition == "role_break"
    if task.target_output != apply_rule(task.target_tree, task.true_rule):
        return False
    for source in task.sources:
        kind = task.oracle_source_kinds[source["name"]]
        if kind == "truthful":
            expected = apply_rule(task.target_tree, task.true_rule)
        else:
            expected = apply_rule(task.target_tree, task.candidate_rule if task.true_rule == task.alternate_rule else task.alternate_rule)
        if source["observed_output"] != expected:
            return False
    for name, test in task.tests.items():
        kind = task.oracle_test_kinds[name]
        if not test["available"]:
            if test["observed_output"] is not None:
                return False
            continue
        if kind == "truthful":
            expected = apply_rule(test["tree"], task.true_rule)
        elif kind == "misleading":
            expected = apply_rule(test["tree"], task.candidate_rule)
        else:
            expected = test["candidate_prediction"]
        if test["observed_output"] != expected:
            return False
    return True


def observed_hypothesis(item: Mapping[str, Any]) -> Optional[str]:
    if not item.get("available") or not item.get("informative"):
        return None
    observed = item.get("observed_output")
    matches = []
    if observed == item.get("candidate_prediction"):
        matches.append("candidate")
    if observed == item.get("alternate_prediction"):
        matches.append("alternate")
    return matches[0] if len(matches) == 1 else None


def corroboration(task: TestingTask) -> Tuple[Optional[str], List[Mapping[str, Any]], float, str]:
    available = [source for source in task.sources if source.get("available") and source.get("informative")]
    by_group: Dict[str, Mapping[str, Any]] = {}
    for source in sorted(available, key=lambda item: (float(item["cost"]), str(item["name"]))):
        by_group.setdefault(str(source["group"]), source)
    selected = sorted(by_group.values(), key=lambda item: (float(item["cost"]), str(item["name"])))[:2]
    cost = sum(float(source["cost"]) for source in selected)
    hypotheses = [hypothesis for source in selected if (hypothesis := observed_hypothesis(source)) is not None]
    if len(selected) < 2 or len(hypotheses) != 2:
        return None, selected, cost, "insufficient_sources"
    if hypotheses[0] != hypotheses[1]:
        return None, selected, cost, "source_conflict"
    return hypotheses[0], selected, cost, "source_agreement"


def evaluate(policy: str, tasks: Sequence[TestingTask], seed: int) -> Dict[str, Any]:
    outcomes: List[Dict[str, Any]] = []
    for task in tasks:
        hypothesis, selected_sources, source_cost, source_reason = corroboration(task)
        cost = BASE_COST + source_cost
        selected_rule: Optional[str] = None
        decision = "abstain"
        reason = source_reason
        test_attempted = False
        test_kind: Optional[str] = None
        committed_predictions: Optional[Dict[str, Any]] = None
        observed_test_output: Optional[int] = None
        if hypothesis is not None:
            if policy == "corroboration_only":
                selected_rule = task.candidate_rule if hypothesis == "candidate" else task.alternate_rule
                decision = "accept_corrobated_hypothesis"
                reason = "source_agreement_only"
            else:
                test_kind = "prediction" if policy == "prediction_testing" else "intervention"
                test = task.tests.get(test_kind)
                if test is None or not test.get("available"):
                    reason = "test_unavailable"
                else:
                    committed_predictions = {
                        "candidate": test.get("candidate_prediction"),
                        "alternate": test.get("alternate_prediction"),
                    }
                    # The commitment is recorded before the environment's
                    # observed output is consulted.
                    test_attempted = True
                    observed_test_output = test.get("observed_output")
                    cost += float(test["cost"])
                    if not test.get("informative"):
                        reason = "test_inconclusive"
                    else:
                        result_hypothesis = observed_hypothesis(test)
                        if result_hypothesis is None:
                            reason = "test_observation_unresolved"
                        elif result_hypothesis != hypothesis:
                            selected_rule = task.candidate_rule if result_hypothesis == "candidate" else task.alternate_rule
                            decision = "revise_after_test"
                            reason = "test_overrode_source_consensus"
                        else:
                            selected_rule = task.candidate_rule if hypothesis == "candidate" else task.alternate_rule
                            decision = "accept_after_test"
                            reason = "test_supported_source_consensus"
        accepted = selected_rule is not None
        correct = accepted and task.target_output is not None and apply_rule(task.target_tree, selected_rule) == task.target_output
        harmful = accepted and not correct
        false_abstention = task.valid_transfer and not accepted
        outcome = {
            "task_id": task.task_id,
            "seed": seed,
            "index": task.index,
            "phase": task.phase,
            "condition": task.condition,
            "template": task.template,
            "valid_transfer": task.valid_transfer,
            "truthful_test_available": task.truthful_test_available,
            "selected_rule": selected_rule,
            "decision": decision,
            "reason": reason,
            "accepted": accepted,
            "correct": correct,
            "harmful_transfer": harmful,
            "false_abstention": false_abstention,
            "source_hypothesis": hypothesis,
            "selected_sources": [source["name"] for source in selected_sources],
            "selected_groups": [source["group"] for source in selected_sources],
            "source_cost": source_cost,
            "test_attempted": test_attempted,
            "test_kind": test_kind,
            "committed_predictions": committed_predictions,
            "observed_test_output": observed_test_output,
            "cost": cost,
        }
        if harmful:
            outcome["trace"] = {
                "sources": task.sources,
                "tests": task.tests,
                "source_reason": source_reason,
                "committed_predictions": committed_predictions,
            }
        outcomes.append(outcome)
    return summarize(policy, seed, outcomes)


def summarize_slice(outcomes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    accepted = [item for item in outcomes if item["accepted"]]
    valid = [item for item in outcomes if item["valid_transfer"]]
    harmful = [item for item in outcomes if item["harmful_transfer"]]
    truthful = [item for item in outcomes if item["truthful_test_available"]]
    return {
        "tasks": len(outcomes),
        "accuracy": sum(item["correct"] for item in outcomes) / max(1, len(outcomes)),
        "coverage": len(accepted) / max(1, len(outcomes)),
        "conditional_accuracy": sum(item["correct"] for item in accepted) / max(1, len(accepted)),
        "mean_cost": sum(float(item["cost"]) for item in outcomes) / max(1, len(outcomes)),
        "harmful_transfer_count": len(harmful),
        "harmful_transfer_rate_all_tasks": len(harmful) / max(1, len(outcomes)),
        "harmful_transfer_rate_accepted": len(harmful) / max(1, len(accepted)),
        "valid_transfer_recall": sum(item["accepted"] and item["correct"] for item in valid) / max(1, len(valid)),
        "false_abstentions": sum(item["false_abstention"] for item in valid),
        "truthful_resolution_count": sum(item["truthful_test_available"] and item["correct"] for item in outcomes),
        "truthful_resolution_rate": sum(item["truthful_test_available"] and item["correct"] for item in outcomes) / max(1, len(truthful)),
        "test_attempts": sum(item["test_attempted"] for item in outcomes),
        "test_matches": sum(item["test_attempted"] and item["correct"] for item in outcomes),
        "memory_contamination": len(harmful),
        "inconclusive_abstentions": sum(item["reason"] == "test_inconclusive" for item in outcomes),
    }


def summarize(policy: str, seed: int, outcomes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary = summarize_slice(outcomes)
    summary.update({
        "policy": policy,
        "seed": seed,
        "phase_summaries": {
            phase: summarize_slice([item for item in outcomes if item["phase"] == phase])
            for phase in ("discovery", "heldout")
        },
        "outcomes": list(outcomes),
    })
    return summary


def aggregate_summaries(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    numeric_mean = (
        "accuracy", "coverage", "conditional_accuracy", "mean_cost",
        "harmful_transfer_rate_all_tasks", "harmful_transfer_rate_accepted",
        "valid_transfer_recall", "truthful_resolution_rate",
    )
    integer_sum = (
        "tasks", "harmful_transfer_count", "false_abstentions", "truthful_resolution_count",
        "test_attempts", "test_matches", "memory_contamination", "inconclusive_abstentions",
    )
    result: Dict[str, Any] = {"runs": len(summaries)}
    for field in numeric_mean:
        result[field] = sum(float(item.get(field, 0.0)) for item in summaries) / max(1, len(summaries))
    for field in integer_sum:
        result[field] = sum(int(item.get(field, 0)) for item in summaries)
    return result


def group_results(runs: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for outcome in run["outcomes"]:
            grouped[str(outcome[field])][run["policy"]].append(outcome)
    return {
        group: {policy: summarize_slice(outcomes) for policy, outcomes in sorted(policies.items())}
        for group, policies in sorted(grouped.items())
    }


def source_hashes() -> Dict[str, str]:
    paths = [
        PROTOCOL,
        ROOT / "tools" / "run_crucible_c1r_prediction_intervention_20261021.py",
        C1P_OUTPUT / "RESULTS.json",
        C1P_OUTPUT / "HASHES.json",
        C1Q_OUTPUT / "RESULTS.json",
        C1Q_OUTPUT / "HASHES.json",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def render_summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1r prediction and intervention testing",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; hand-coded symbolic consequence-testing experiment.",
        "",
        "## Aggregate results",
        "",
        "| Policy | Accuracy | Coverage | Conditional accuracy | Mean cost | Harmful/all | Harmful/accepted | Truthful resolution |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, item in result["aggregate"].items():
        lines.append(
            f"| `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | {item['conditional_accuracy']:.4f} | "
            f"{item['mean_cost']:.3f} | {item['harmful_transfer_rate_all_tasks']:.4f} | "
            f"{item['harmful_transfer_rate_accepted']:.4f} | {item['truthful_resolution_rate']:.4f} |"
        )
    lines.extend([
        "",
        "Harmful/all is harmful transfer divided by all tasks. Harmful/accepted is the conditional rate among accepted transfers.",
        "Truthful resolution is correct resolution divided by tasks with a truthful discriminating test available.",
        "",
        "## Results by condition",
        "",
        "| Condition | Policy | Accuracy | Coverage | Harmful/all | Truthful resolution | Test attempts | Mean cost |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for condition, policies in result["by_condition"].items():
        for policy, item in policies.items():
            lines.append(
                f"| `{condition}` | `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | "
                f"{item['harmful_transfer_rate_all_tasks']:.4f} | {item['truthful_resolution_rate']:.4f} | "
                f"{item['test_attempts']} | {item['mean_cost']:.3f} |"
            )
    lines.extend(["", "## Results by phase", "", "| Phase | Policy | Accuracy | Coverage | Harmful/all | Truthful resolution | Mean cost |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for phase, policies in result["by_phase"].items():
        for policy, item in policies.items():
            lines.append(
                f"| `{phase}` | `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | "
                f"{item['harmful_transfer_rate_all_tasks']:.4f} | {item['truthful_resolution_rate']:.4f} | {item['mean_cost']:.3f} |"
            )
    lines.extend(["", "## Testing diagnostics", "", "| Policy | Test attempts | Test matches | Inconclusive abstentions | Memory contamination |", "| --- | ---: | ---: | ---: | ---: |"])
    for policy, item in result["aggregate"].items():
        lines.append(f"| `{policy}` | {item['test_attempts']} | {item['test_matches']} | {item['inconclusive_abstentions']} | {item['memory_contamination']} |")
    lines.extend(["", "## Gates", ""])
    for name, value in result["gates"].items():
        lines.append(f"- `{name}`: `{value}`.")
    lines.extend([
        "",
        "This is an explicit symbolic consequence-testing result, not evidence that a neural model learned causal reasoning.",
        "",
        "## Harmful-transfer traces",
        "",
        f"Stored harmful-transfer traces: `{result['harmful_trace_count']}`.",
        "",
        "## Source hashes",
        "",
    ])
    for path, digest in sorted(hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen_audit = {
        "c1p": verify_index(C1P_OUTPUT / "HASHES.json"),
        "c1q": verify_index(C1Q_OUTPUT / "HASHES.json"),
    }
    all_tasks: List[TestingTask] = []
    manifests: Dict[str, Any] = {}
    oracle_checks = 0
    oracle_failures: List[str] = []
    for seed in TASK_SEEDS:
        tasks = generate_tasks(seed)
        all_tasks.extend(tasks)
        checks = [oracle_check(task) for task in tasks]
        oracle_checks += len(checks)
        oracle_failures.extend(task.task_id for task, passed in zip(tasks, checks) if not passed)
        manifests[str(seed)] = {
            "count": len(tasks),
            "task_digest": hashlib.sha256("|".join(task.task_id for task in tasks).encode()).hexdigest(),
            "phase_counts": dict(sorted(Counter(task.phase for task in tasks).items())),
            "conditions": dict(sorted(Counter(task.condition for task in tasks).items())),
            "templates": dict(sorted(Counter(task.template for task in tasks).items())),
        }
    if oracle_failures:
        raise RuntimeError(f"C1r oracle generation failure: {oracle_failures[:5]}")
    write_json(OUTPUT / "TASK_MANIFEST.json", {
        "experiment": EXPERIMENT,
        "seeds": manifests,
        "task_count_total": len(all_tasks),
        "oracle_checks": oracle_checks,
        "oracle_failures": len(oracle_failures),
        "hidden_labels_used_by_policy": False,
    })

    all_runs: List[Dict[str, Any]] = []
    for seed in TASK_SEEDS:
        tasks = [task for task in all_tasks if task.seed == seed]
        for policy in POLICIES:
            all_runs.append(evaluate(policy, tasks, seed))

    aggregate = {
        policy: aggregate_summaries([run for run in all_runs if run["policy"] == policy])
        for policy in POLICIES
    }
    condition_results = group_results(all_runs, "condition")
    phase_results = group_results(all_runs, "phase")
    result: Dict[str, Any] = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "aggregate": aggregate,
        "by_condition": condition_results,
        "by_phase": phase_results,
        "frozen_audit": frozen_audit,
        "task_manifest": manifests,
        "oracle_checks": {"checks": oracle_checks, "failures": len(oracle_failures)},
        "gates": {
            "oracle_and_manifest_reproducible": len(oracle_failures) == 0 and oracle_checks == TASKS_PER_SEED * len(TASK_SEEDS),
            "prediction_reduces_correlated_harm": condition_results["correlated_misinformation"]["prediction_testing"]["harmful_transfer_rate_all_tasks"] < condition_results["correlated_misinformation"]["corroboration_only"]["harmful_transfer_rate_all_tasks"],
            "intervention_reduces_correlated_harm": condition_results["correlated_misinformation"]["intervention_testing"]["harmful_transfer_rate_all_tasks"] < condition_results["correlated_misinformation"]["corroboration_only"]["harmful_transfer_rate_all_tasks"],
            "prediction_resolves_prediction_only": condition_results["prediction_only"]["prediction_testing"]["truthful_resolution_rate"] > condition_results["prediction_only"]["corroboration_only"]["truthful_resolution_rate"],
            "intervention_resolves_intervention_only": condition_results["intervention_only"]["intervention_testing"]["truthful_resolution_rate"] > condition_results["intervention_only"]["corroboration_only"]["truthful_resolution_rate"],
            "testing_abstains_inconclusive_and_unavailable": condition_results["inconclusive_test"]["prediction_testing"]["inconclusive_abstentions"] > 0 and condition_results["unavailable_tests"]["intervention_testing"]["coverage"] == 0.0,
            "misleading_test_failure_visible": condition_results["misleading_prediction"]["prediction_testing"]["harmful_transfer_rate_all_tasks"] > 0.0 or condition_results["misleading_intervention"]["intervention_testing"]["harmful_transfer_rate_all_tasks"] > 0.0,
            "hand_coded_not_learned": True,
        },
        "config": {
            "task_seeds": list(TASK_SEEDS),
            "tasks_per_seed": TASKS_PER_SEED,
            "discovery_fraction": DISCOVERY_FRACTION,
            "conditions": list(CONDITIONS),
            "templates": list(TEMPLATES),
            "policies": list(POLICIES),
            "base_cost": BASE_COST,
            "source_costs": {"A": SOURCE_A_COST, "B": SOURCE_B_COST, "C": SOURCE_C_COST},
            "prediction_cost": PREDICTION_COST,
            "intervention_cost": INTERVENTION_COST,
            "execution": "CPU-only",
            "neural_training_updates": 0,
        },
    }
    harmful_traces = []
    for run in all_runs:
        for outcome in run["outcomes"]:
            if outcome["harmful_transfer"]:
                harmful_traces.append({
                    "policy": run["policy"],
                    "seed": run["seed"],
                    "task_id": outcome["task_id"],
                    "phase": outcome["phase"],
                    "condition": outcome["condition"],
                    "reason": outcome["reason"],
                    "selected_sources": outcome["selected_sources"],
                    "selected_groups": outcome["selected_groups"],
                    "test_kind": outcome["test_kind"],
                    "committed_predictions": outcome["committed_predictions"],
                })
    result["harmful_trace_count"] = len(harmful_traces)
    write_json(OUTPUT / "HARMFUL_TRANSFER_TRACES.json", harmful_traces)
    result["harmful_trace_sha256"] = sha256_file(OUTPUT / "HARMFUL_TRANSFER_TRACES.json")
    write_json(OUTPUT / "RESULTS.json", result)
    write_json(OUTPUT / "RUN_CONFIG.json", result["config"])
    with (OUTPUT / "RUNS.jsonl").open("w", encoding="utf-8") as handle:
        for run in all_runs:
            handle.write(json.dumps(run, sort_keys=True) + "\n")
    hashes = source_hashes()
    (OUTPUT / "RESULTS_SUMMARY.md").write_text(render_summary(result, hashes), encoding="utf-8")
    file_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "HASHES.json"
    }
    write_json(OUTPUT / "HASHES.json", file_hashes)
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "runs": len(all_runs), "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
