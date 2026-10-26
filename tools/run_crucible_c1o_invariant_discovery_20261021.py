#!/usr/bin/env python3
"""Run C1o: invariant discovery and counterexample-guided refinement on CPU."""

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
OUTPUT = ROOT / "experiments" / "crucible_c1o_invariant_discovery_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
C1N_OUTPUT = ROOT / "experiments" / "crucible_c1n_relational_transfer_20261021"
C1M_OUTPUT = ROOT / "experiments" / "crucible_c1m_exact_reuse_adaptation_20261021"
EXPERIMENT = "crucible_c1o_invariant_discovery_20261021"
TASKS_PER_SEED = 12_000
DISCOVERY_FRACTION = 0.25
TASK_SEEDS = (42, 1337, 2026)
CONDITIONS = (
    "exact_role",
    "depth_extension",
    "role_permutation",
    "marker_guard_valid",
    "marker_guard_invalid",
    "unobservable_ambiguity",
    "misleading_evidence",
    "role_break",
)
TEMPLATES = ("balanced", "asymmetric", "deep", "wide")
POLICIES = ("frozen_relational", "counterexample_refinement", "active_evidence")
VALID_CONDITIONS = {"exact_role", "depth_extension", "role_permutation", "marker_guard_valid"}
DISCOVERABLE_INVALID = {"marker_guard_invalid"}
UNRESOLVABLE_INVALID = {"unobservable_ambiguity", "misleading_evidence"}
BASE_COST = 0.15
COUNTEREXAMPLE_COST = 0.20
PROBE_COST = 0.35
REFINEMENT_COST = 0.10


@dataclass(frozen=True)
class InvariantTask:
    task_id: str
    seed: int
    index: int
    phase: str
    condition: str
    template: str
    source_tree: Dict[str, Any]
    target_tree: Dict[str, Any]
    probe_tree: Dict[str, Any]
    source_rule: str
    true_rule: str
    source_relational_signature: Tuple[Any, ...]
    target_relational_signature: Tuple[Any, ...]
    source_features: Dict[str, Any]
    target_features: Dict[str, Any]
    probe_available: bool
    probe_output: Optional[int]
    target_output: int
    valid_transfer: bool


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


def build_source_tree(seed: int, index: int, template: str) -> Dict[str, Any]:
    tree = c1n.build_source_tree(seed, index, template)
    # C1n's marker is deliberately neutral; C1o adds an observable feature
    # that can become a discovered guard without exposing hidden_mode.
    for node in tree["nodes"]:
        if node["role"] in {"anchor", "support"}:
            node["marker"] = 0
    tree["c1o_family"] = "leaf_transfer"
    return tree


def children(tree: Mapping[str, Any]) -> Dict[int, List[int]]:
    return c1n.children(tree)


def role_position(tree: Mapping[str, Any], role: str) -> Optional[str]:
    return c1n.role_position(tree, role)


def apply_rule(tree: Mapping[str, Any], rule: str) -> int:
    return c1n.apply_rule(tree, rule)


def branch_marker(tree: Mapping[str, Any], role: str) -> Optional[int]:
    node_by_id = {int(node["id"]): node for node in tree["nodes"]}
    position = role_position(tree, role)
    if position is None:
        return None
    root_children = children(tree).get(int(tree["root"]), [])
    branch_id = root_children[0 if position == "left" else 1]
    return int(node_by_id[branch_id].get("marker", 0))


def branch_leaf_counts(tree: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    root_children = children(tree).get(int(tree["root"]), [])
    if len(root_children) < 2:
        return (None, None)
    return (c1n.count_leaves(tree, root_children[0]), c1n.count_leaves(tree, root_children[1]))


def visible_features(tree: Mapping[str, Any]) -> Dict[str, Any]:
    left_count, right_count = branch_leaf_counts(tree)
    return {
        "anchor_position": role_position(tree, "anchor"),
        "support_position": role_position(tree, "support"),
        "anchor_marker": branch_marker(tree, "anchor"),
        "support_marker": branch_marker(tree, "support"),
        "left_leaf_count": left_count,
        "right_leaf_count": right_count,
        "node_count": len(tree["nodes"]),
        "edge_count": len(tree["edges"]),
        "reachable": c1n.reachable(tree),
    }


def feature_differences(source: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        key: {"source": source.get(key), "target": target.get(key)}
        for key in sorted(set(source) | set(target))
        if source.get(key) != target.get(key)
    }


def target_for_condition(source: Mapping[str, Any], condition: str) -> Dict[str, Any]:
    target = clone_tree(source)
    node_by_id = {int(node["id"]): node for node in target["nodes"]}
    root_children = children(target).get(int(target["root"]), [])
    if condition == "exact_role":
        return c1n.relabel_isomorphism(target, 0, 0)
    if condition == "depth_extension":
        anchor_position = role_position(target, "anchor")
        branch_id = root_children[0 if anchor_position == "left" else 1]
        next_id = max(int(node["id"]) for node in target["nodes"]) + 1
        target["nodes"].append({"id": next_id, "role": "leaf", "label": 9, "marker": 0})
        target["edges"].append([branch_id, next_id])
    elif condition == "role_permutation":
        left, right = root_children[:2]
        node_by_id[left]["role"], node_by_id[right]["role"] = node_by_id[right]["role"], node_by_id[left]["role"]
    elif condition == "marker_guard_valid":
        pass
    elif condition == "marker_guard_invalid":
        anchor_position = role_position(target, "anchor")
        branch_id = root_children[0 if anchor_position == "left" else 1]
        node_by_id[branch_id]["marker"] = 1
    elif condition == "unobservable_ambiguity":
        target["hidden_mode"] = "support"
    elif condition == "misleading_evidence":
        target["hidden_mode"] = "support"
    elif condition == "role_break":
        anchor_position = role_position(target, "anchor")
        branch_id = root_children[0 if anchor_position == "left" else 1]
        node_by_id[branch_id]["role"] = "distractor"
    else:
        raise ValueError(condition)
    return target


def true_rule_for(source: Mapping[str, Any], target: Mapping[str, Any], condition: str) -> str:
    if condition in {"marker_guard_invalid", "unobservable_ambiguity", "misleading_evidence", "role_break"}:
        position = role_position(target, "support") or "left"
    else:
        position = role_position(target, "anchor") or "left"
    return f"{position}_leaf_count"


def build_task(seed: int, index: int) -> InvariantTask:
    condition = CONDITIONS[stable_int("c1o-condition", seed, index) % len(CONDITIONS)]
    template = TEMPLATES[stable_int("c1o-template", seed, index) % len(TEMPLATES)]
    source = build_source_tree(seed, index, template)
    target = target_for_condition(source, condition)
    source_anchor = role_position(source, "anchor") or "left"
    source_rule = f"{source_anchor}_leaf_count"
    true_rule = true_rule_for(source, target, condition)
    probe = clone_tree(target)
    target_output = apply_rule(target, true_rule)
    probe_available = condition in {
        "exact_role",
        "depth_extension",
        "role_permutation",
        "marker_guard_valid",
        "marker_guard_invalid",
        "misleading_evidence",
    }
    probe_output: Optional[int]
    if not probe_available:
        probe_output = None
    elif condition == "misleading_evidence":
        candidate_position = role_position(target, "anchor") or "left"
        probe_output = apply_rule(probe, f"{candidate_position}_leaf_count")
    else:
        probe_output = target_output
    phase = "discovery" if index < int(TASKS_PER_SEED * DISCOVERY_FRACTION) else "heldout"
    return InvariantTask(
        task_id=f"c1o-{seed}-{index}-{condition}-{template}",
        seed=seed,
        index=index,
        phase=phase,
        condition=condition,
        template=template,
        source_tree=source,
        target_tree=target,
        probe_tree=probe,
        source_rule=source_rule,
        true_rule=true_rule,
        source_relational_signature=c1n.relational_signature(source),
        target_relational_signature=c1n.relational_signature(target),
        source_features=visible_features(source),
        target_features=visible_features(target),
        probe_available=probe_available,
        probe_output=probe_output,
        target_output=target_output,
        valid_transfer=condition in VALID_CONDITIONS,
    )


def generate_tasks(seed: int) -> List[InvariantTask]:
    return [build_task(seed, index) for index in range(TASKS_PER_SEED)]


def oracle_check(task: InvariantTask) -> bool:
    expected_target = apply_rule(task.target_tree, task.true_rule)
    if expected_target != task.target_output:
        return False
    if task.probe_available and task.probe_output is None:
        return False
    if task.condition == "marker_guard_invalid":
        return task.target_features["anchor_marker"] != task.source_features["anchor_marker"]
    if task.condition in {"unobservable_ambiguity", "misleading_evidence"}:
        return task.target_tree.get("hidden_mode") == "support"
    return True


def relational_candidate(task: InvariantTask) -> Optional[str]:
    if task.source_relational_signature != task.target_relational_signature:
        return None
    target_anchor = role_position(task.target_tree, "anchor")
    target_support = role_position(task.target_tree, "support")
    if target_anchor is None or target_support is None:
        return None
    return f"{target_anchor}_leaf_count"


def initial_state() -> Dict[str, Any]:
    return {
        "guards": {},
        "records": [],
        "refinements": 0,
        "counterexamples": 0,
        "probe_attempts": 0,
        "probe_passes": 0,
        "guard_conflicts": 0,
    }


def guard_mismatches(state: Mapping[str, Any], task: InvariantTask) -> Dict[str, Dict[str, Any]]:
    mismatches: Dict[str, Dict[str, Any]] = {}
    guards = state.get("guards", {})
    for key, expected in sorted(guards.items()):
        actual = task.target_features.get(key)
        if actual != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    return mismatches


def learn_from_counterexample(state: Dict[str, Any], task: InvariantTask) -> Dict[str, Any]:
    differences = feature_differences(task.source_features, task.target_features)
    added: Dict[str, Any] = {}
    guards = state["guards"]
    for key, values in differences.items():
        expected = values["source"]
        if key in guards and guards[key] != expected:
            state["guard_conflicts"] += 1
            continue
        if key not in guards:
            guards[key] = expected
            added[key] = expected
    state["counterexamples"] += 1
    state["refinements"] += 1 if added else 0
    state["records"].append({
        "status": "revised",
        "task_id": task.task_id,
        "condition": task.condition,
        "feature_differences": differences,
        "added_guards": added,
    })
    return added


def evaluate(policy: str, tasks: Sequence[InvariantTask], seed: int) -> Dict[str, Any]:
    state = initial_state()
    outcomes: List[Dict[str, Any]] = []
    phase_order = {"discovery": 0, "heldout": 1}
    for task in tasks:
        selected = relational_candidate(task)
        decision = "abstain"
        reason = "no_relational_candidate" if selected is None else ""
        accepted = False
        correct = False
        cost = 0.0
        probe_attempted = False
        probe_passed = False
        counterexample_detected = False
        refinement_added: Dict[str, Any] = {}
        guard_snapshot = dict(state["guards"])
        cost += BASE_COST
        if selected is not None:
            mismatch = guard_mismatches(state, task)
            if mismatch:
                decision = "abstain"
                reason = "retained_guard_mismatch"
            elif policy == "frozen_relational":
                accepted = True
                decision = "reuse_or_adapt"
                reason = "frozen_relational_match"
            else:
                require_probe = policy == "active_evidence"
                if task.probe_available:
                    probe_attempted = True
                    state["probe_attempts"] += 1
                    cost += PROBE_COST
                    probe_passed = apply_rule(task.probe_tree, selected) == task.probe_output
                    if probe_passed:
                        state["probe_passes"] += 1
                        accepted = True
                        decision = "reuse_or_adapt"
                        reason = "probe_supported"
                    else:
                        counterexample_detected = True
                        cost += COUNTEREXAMPLE_COST + REFINEMENT_COST
                        refinement_added = learn_from_counterexample(state, task)
                        decision = "abstain"
                        reason = "counterexample_rejected_candidate"
                elif require_probe:
                    decision = "abstain"
                    reason = "evidence_unavailable"
                else:
                    accepted = True
                    decision = "reuse_or_adapt"
                    reason = "no_counterexample_observed"
        if accepted:
            correct = apply_rule(task.target_tree, selected) == task.target_output
        harmful = accepted and not correct
        invalid_acceptance = (not task.valid_transfer) and accepted
        false_abstention = task.valid_transfer and not accepted
        outcome = {
            "task_id": task.task_id,
            "seed": seed,
            "index": task.index,
            "phase": task.phase,
            "condition": task.condition,
            "template": task.template,
            "valid_transfer": task.valid_transfer,
            "source_rule": task.source_rule,
            "selected_rule": selected,
            "true_rule": task.true_rule,
            "decision": decision,
            "reason": reason,
            "accepted": accepted,
            "correct": correct,
            "harmful_transfer": harmful,
            "invalid_acceptance": invalid_acceptance,
            "false_abstention": false_abstention,
            "unresolvable_case": task.condition in UNRESOLVABLE_INVALID,
            "discoverable_case": task.condition in DISCOVERABLE_INVALID,
            "probe_available": task.probe_available,
            "probe_attempted": probe_attempted,
            "probe_passed": probe_passed,
            "counterexample_detected": counterexample_detected,
            "refinement_added": refinement_added,
            "guard_snapshot": guard_snapshot,
            "guards_after": dict(state["guards"]),
            "source_features": task.source_features,
            "target_features": task.target_features,
            "feature_differences": feature_differences(task.source_features, task.target_features),
            "cost": cost,
        }
        outcomes.append(outcome)
    return summarize(policy, seed, outcomes, state)


def summarize(policy: str, seed: int, outcomes: Sequence[Mapping[str, Any]], state: Mapping[str, Any]) -> Dict[str, Any]:
    count = len(outcomes)
    accepted = [item for item in outcomes if item["accepted"]]
    valid = [item for item in outcomes if item["valid_transfer"]]
    invalid = [item for item in outcomes if not item["valid_transfer"]]
    harmful = [item for item in outcomes if item["harmful_transfer"]]
    phase_summaries = {
        phase: summarize_slice([item for item in outcomes if item["phase"] == phase])
        for phase in ("discovery", "heldout")
    }
    return {
        "policy": policy,
        "seed": seed,
        "tasks": count,
        "accuracy": sum(item["correct"] for item in outcomes) / max(1, count),
        "coverage": len(accepted) / max(1, count),
        "conditional_accuracy": sum(item["correct"] for item in accepted) / max(1, len(accepted)),
        "mean_cost": sum(float(item["cost"]) for item in outcomes) / max(1, count),
        "harmful_transfer_count": len(harmful),
        "harmful_transfer_rate_all_tasks": len(harmful) / max(1, count),
        "harmful_transfer_rate_accepted": len(harmful) / max(1, len(accepted)),
        "valid_transfer_recall": sum(item["accepted"] and item["correct"] for item in valid) / max(1, len(valid)),
        "false_abstentions": sum(item["false_abstention"] for item in valid),
        "invalid_transfer_rejection": sum(not item["accepted"] for item in invalid) / max(1, len(invalid)),
        "invalid_acceptances": sum(item["invalid_acceptance"] for item in invalid),
        "unresolvable_tasks": sum(item["unresolvable_case"] for item in outcomes),
        "unresolvable_abstentions": sum(item["unresolvable_case"] and not item["accepted"] for item in outcomes),
        "counterexample_detections": sum(item["counterexample_detected"] for item in outcomes),
        "probe_attempts": sum(item["probe_attempted"] for item in outcomes),
        "probe_passes": sum(item["probe_passed"] for item in outcomes),
        "refinements": int(state["refinements"]),
        "guard_conflicts": int(state["guard_conflicts"]),
        "guards_final": dict(state["guards"]),
        "action_counts": dict(sorted(Counter(item["decision"] for item in outcomes).items())),
        "phase_summaries": phase_summaries,
        "outcomes": list(outcomes),
        "state_records": list(state["records"]),
    }


def summarize_slice(outcomes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    accepted = [item for item in outcomes if item["accepted"]]
    valid = [item for item in outcomes if item["valid_transfer"]]
    invalid = [item for item in outcomes if not item["valid_transfer"]]
    harmful = [item for item in outcomes if item["harmful_transfer"]]
    return {
        "tasks": len(outcomes),
        "accuracy": sum(item["correct"] for item in outcomes) / max(1, len(outcomes)),
        "coverage": len(accepted) / max(1, len(outcomes)),
        "conditional_accuracy": sum(item["correct"] for item in accepted) / max(1, len(accepted)),
        "mean_cost": sum(float(item["cost"]) for item in outcomes) / max(1, len(outcomes)),
        "harmful_transfer_count": len(harmful),
        "harmful_transfer_rate_all_tasks": len(harmful) / max(1, len(outcomes)),
        "valid_transfer_recall": sum(item["accepted"] and item["correct"] for item in valid) / max(1, len(valid)),
        "false_abstentions": sum(item["false_abstention"] for item in valid),
        "invalid_transfer_rejection": sum(not item["accepted"] for item in invalid) / max(1, len(invalid)),
        "counterexample_detections": sum(item["counterexample_detected"] for item in outcomes),
        "probe_attempts": sum(item["probe_attempted"] for item in outcomes),
        "unresolvable_abstentions": sum(item["unresolvable_case"] and not item["accepted"] for item in outcomes),
    }


def aggregate_summaries(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not summaries:
        return {"runs": 0}
    numeric_mean = (
        "accuracy", "coverage", "conditional_accuracy", "mean_cost",
        "harmful_transfer_rate_all_tasks", "harmful_transfer_rate_accepted",
        "valid_transfer_recall", "invalid_transfer_rejection",
    )
    integer_sum = (
        "tasks", "harmful_transfer_count", "false_abstentions", "invalid_acceptances",
        "unresolvable_tasks", "unresolvable_abstentions", "counterexample_detections",
        "probe_attempts", "probe_passes", "refinements", "guard_conflicts",
    )
    result: Dict[str, Any] = {"runs": len(summaries)}
    for field in numeric_mean:
        result[field] = sum(float(item.get(field, 0.0)) for item in summaries) / len(summaries)
    for field in integer_sum:
        result[field] = sum(int(item.get(field, 0)) for item in summaries)
    return result


def by_group(runs: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for outcome in run["outcomes"]:
            grouped[str(outcome[field])][run["policy"]].append(outcome)
    result: Dict[str, Dict[str, Any]] = {}
    for group, policies in sorted(grouped.items()):
        result[group] = {}
        for policy, outcomes in sorted(policies.items()):
            result[group][policy] = summarize_slice(outcomes)
    return result


def source_hashes() -> Dict[str, str]:
    paths = [
        PROTOCOL,
        ROOT / "tools" / "run_crucible_c1o_invariant_discovery_20261021.py",
        C1N_OUTPUT / "RESULTS.json",
        C1N_OUTPUT / "HASHES.json",
        C1M_OUTPUT / "RESULTS.json",
        C1M_OUTPUT / "HASHES.json",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def render_summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1o invariant discovery and counterexample-guided refinement",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; hand-coded symbolic mechanism test.",
        "",
        "## Aggregate results",
        "",
        "| Policy | Accuracy | Coverage | Conditional accuracy | Mean cost | Harmful/all | Harmful/accepted | Valid recall | Invalid rejection |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, item in result["aggregate"].items():
        lines.append(
            f"| `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | {item['conditional_accuracy']:.4f} | "
            f"{item['mean_cost']:.3f} | {item['harmful_transfer_rate_all_tasks']:.4f} | "
            f"{item['harmful_transfer_rate_accepted']:.4f} | {item['valid_transfer_recall']:.4f} | "
            f"{item['invalid_transfer_rejection']:.4f} |"
        )
    lines.extend([
        "",
        "Harmful/all is harmful transfer divided by all tasks. Harmful/accepted is the conditional rate among accepted transfers.",
        "",
        "## Results by condition",
        "",
        "| Condition | Policy | Accuracy | Coverage | Harmful/all | Valid recall | Invalid rejection | Mean cost |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for condition, policies in result["by_condition"].items():
        for policy, item in policies.items():
            lines.append(
                f"| `{condition}` | `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | "
                f"{item['harmful_transfer_rate_all_tasks']:.4f} | {item['valid_transfer_recall']:.4f} | "
                f"{item['invalid_transfer_rejection']:.4f} | {item['mean_cost']:.3f} |"
            )
    lines.extend(["", "## Results by phase", "", "| Phase | Policy | Accuracy | Coverage | Harmful/all | Valid recall | Mean cost |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for phase, policies in result["by_phase"].items():
        for policy, item in policies.items():
            lines.append(
                f"| `{phase}` | `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | "
                f"{item['harmful_transfer_rate_all_tasks']:.4f} | {item['valid_transfer_recall']:.4f} | {item['mean_cost']:.3f} |"
            )
    lines.extend(["", "## Refinement diagnostics", "", "| Policy | Counterexamples | Refinements | Probe attempts | Unresolvable abstentions | Final guards |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for policy, item in result["aggregate"].items():
        guard_text = json.dumps(item.get("guards_final", {}), sort_keys=True)
        lines.append(f"| `{policy}` | {item.get('counterexample_detections', 0)} | {item.get('refinements', 0)} | {item.get('probe_attempts', 0)} | {item.get('unresolvable_abstentions', 0)} | `{guard_text}` |")
    lines.extend(["", "## Gates", ""])
    for name, value in result["gates"].items():
        lines.append(f"- `{name}`: `{value}`.")
    lines.extend([
        "",
        "This is an explicit symbolic invariant-refinement result, not evidence that a neural model learned relational reasoning.",
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
        "c1n": verify_index(C1N_OUTPUT / "HASHES.json"),
        "c1m": verify_index(C1M_OUTPUT / "HASHES.json"),
    }
    all_tasks: List[InvariantTask] = []
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
        raise RuntimeError(f"C1o oracle generation failure: {oracle_failures[:5]}")
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
            run = evaluate(policy, tasks, seed)
            all_runs.append(run)

    aggregate = {
        policy: aggregate_summaries([run for run in all_runs if run["policy"] == policy])
        for policy in POLICIES
    }
    for policy in POLICIES:
        policy_runs = [run for run in all_runs if run["policy"] == policy]
        aggregate[policy]["guards_final_by_seed"] = [run["guards_final"] for run in policy_runs]
        aggregate[policy]["guards_final"] = policy_runs[-1]["guards_final"] if policy_runs else {}

    condition_groups = by_group(all_runs, "condition")
    phase_groups = by_group(all_runs, "phase")
    frozen = aggregate["frozen_relational"]
    refinement = aggregate["counterexample_refinement"]
    active = aggregate["active_evidence"]
    discoverable_invalid = condition_groups["marker_guard_invalid"]
    unobservable = condition_groups["unobservable_ambiguity"]
    misleading = condition_groups["misleading_evidence"]
    result: Dict[str, Any] = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "aggregate": aggregate,
        "by_condition": condition_groups,
        "by_phase": phase_groups,
        "frozen_audit": frozen_audit,
        "task_manifest": manifests,
        "oracle_checks": {"checks": oracle_checks, "failures": len(oracle_failures)},
        "gates": {
            "oracle_and_manifest_reproducible": len(oracle_failures) == 0 and oracle_checks == TASKS_PER_SEED * len(TASK_SEEDS),
            "refinement_discoverable_recall_or_tradeoff": refinement["valid_transfer_recall"] >= frozen["valid_transfer_recall"] or refinement["harmful_transfer_rate_all_tasks"] < frozen["harmful_transfer_rate_all_tasks"],
            "refinement_reduces_discoverable_harm": discoverable_invalid["counterexample_refinement"]["harmful_transfer_rate_all_tasks"] < discoverable_invalid["frozen_relational"]["harmful_transfer_rate_all_tasks"],
            "active_reduces_unobservable_harm": unobservable["active_evidence"]["harmful_transfer_rate_all_tasks"] < unobservable["counterexample_refinement"]["harmful_transfer_rate_all_tasks"],
            "active_not_trivial_always_abstain": active["coverage"] > 0.0,
            "misleading_evidence_visible": misleading["active_evidence"]["harmful_transfer_rate_all_tasks"] > 0.0,
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
            "counterexample_cost": COUNTEREXAMPLE_COST,
            "probe_cost": PROBE_COST,
            "refinement_cost": REFINEMENT_COST,
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
                    "guard_snapshot": outcome["guard_snapshot"],
                    "guards_after": outcome["guards_after"],
                    "feature_differences": outcome["feature_differences"],
                    "probe_available": outcome["probe_available"],
                    "probe_attempted": outcome["probe_attempted"],
                })
    result["harmful_trace_count"] = len(harmful_traces)
    write_json(OUTPUT / "HARMFUL_TRANSFER_TRACES.json", harmful_traces)
    result["harmful_trace_sha256"] = sha256_file(OUTPUT / "HARMFUL_TRANSFER_TRACES.json")
    write_json(OUTPUT / "RESULTS.json", result)
    write_json(OUTPUT / "RUN_CONFIG.json", result["config"])
    with (OUTPUT / "RUNS.jsonl").open("w", encoding="utf-8") as handle:
        for run in all_runs:
            record = {key: value for key, value in run.items() if key not in {"outcomes", "state_records"}}
            record["outcomes"] = run["outcomes"]
            record["state_records"] = run["state_records"]
            handle.write(json.dumps(record, sort_keys=True) + "\n")
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
