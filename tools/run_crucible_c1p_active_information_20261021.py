#!/usr/bin/env python3
"""Run C1p: active information acquisition on CPU."""

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
OUTPUT = ROOT / "experiments" / "crucible_c1p_active_information_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
C1N_OUTPUT = ROOT / "experiments" / "crucible_c1n_relational_transfer_20261021"
C1O_OUTPUT = ROOT / "experiments" / "crucible_c1o_invariant_discovery_20261021"
EXPERIMENT = "crucible_c1p_active_information_20261021"
TASKS_PER_SEED = 12_000
DISCOVERY_FRACTION = 0.25
TASK_SEEDS = (42, 1337, 2026)
CONDITIONS = (
    "ambiguous_distractor",
    "ambiguous_expensive",
    "ambiguous_multi",
    "no_information",
    "unavailable_evidence",
    "misleading_evidence",
    "role_break",
)
TEMPLATES = ("balanced", "asymmetric", "deep", "wide")
POLICIES = ("passive_abstention", "fixed_probe", "targeted_probe", "no_information_control")
BASE_COST = 0.15
DISTRACTOR_COST = 0.05
INFORMATIVE_COST = 0.35
EXPENSIVE_COST = 1.50


@dataclass(frozen=True)
class InformationTask:
    task_id: str
    seed: int
    index: int
    phase: str
    condition: str
    template: str
    source_tree: Dict[str, Any]
    target_tree: Dict[str, Any]
    source_rule: str
    candidate_rule: Optional[str]
    alternate_rule: Optional[str]
    true_rule: Optional[str]
    source_relational_signature: Tuple[Any, ...]
    target_relational_signature: Tuple[Any, ...]
    probes: List[Dict[str, Any]]
    target_output: Optional[int]
    valid_transfer: bool
    ambiguous: bool


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
    tree["c1p_family"] = "two_hypothesis_leaf_transfer"
    return tree


def make_equal_probe(target: Mapping[str, Any]) -> Dict[str, Any]:
    root = int(target["root"])
    target_children = children(target).get(root, [])[:2]
    node_by_id = {int(node["id"]): node for node in target["nodes"]}
    nodes = [{"id": root, "role": "root", "label": 1, "marker": 0}]
    edges: List[List[int]] = []
    next_id = root + 1
    branch_ids: List[int] = []
    for branch_id in target_children:
        new_branch = next_id
        next_id += 1
        branch_ids.append(new_branch)
        source_node = node_by_id[branch_id]
        nodes.append({"id": new_branch, "role": source_node["role"], "label": source_node.get("label", 0), "marker": source_node.get("marker", 0)})
        edges.append([root, new_branch])
        leaf_id = next_id
        next_id += 1
        nodes.append({"id": leaf_id, "role": "leaf", "label": 4, "marker": 0})
        edges.append([new_branch, leaf_id])
    return {"root": root, "nodes": nodes, "edges": edges, "hidden_mode": target.get("hidden_mode", "anchor")}


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


def probe_record(name: str, cost: float, tree: Mapping[str, Any], candidate: Optional[str], alternate: Optional[str], observed: Optional[int], available: bool) -> Dict[str, Any]:
    candidate_prediction = None if candidate is None else apply_rule(tree, candidate)
    alternate_prediction = None if alternate is None else apply_rule(tree, alternate)
    return {
        "name": name,
        "cost": cost,
        "tree": clone_tree(tree),
        "available": available,
        "observed_output": observed,
        "candidate_prediction": candidate_prediction,
        "alternate_prediction": alternate_prediction,
        "informative": candidate_prediction is not None and alternate_prediction is not None and candidate_prediction != alternate_prediction,
    }


def build_task(seed: int, index: int) -> InformationTask:
    condition = CONDITIONS[stable_int("c1p-condition", seed, index) % len(CONDITIONS)]
    template = TEMPLATES[stable_int("c1p-template", seed, index) % len(TEMPLATES)]
    source = build_source_tree(seed, index, template)
    target = target_tree_for(source, condition)
    source_rule = rule_for_role(source, "anchor")
    candidate_rule = rule_for_role(target, "anchor")
    alternate_rule = rule_for_role(target, "support")
    true_mode = "anchor" if stable_int("c1p-true-mode", seed, index) % 2 == 0 else "support"
    if condition == "role_break":
        true_mode = "support"
    true_rule = None if alternate_rule is None else (candidate_rule if true_mode == "anchor" else alternate_rule)
    target_output = None if true_rule is None else apply_rule(target, true_rule)
    equal_probe = make_equal_probe(target)
    informative_probe = clone_tree(target)
    misleading = condition == "misleading_evidence"
    truthful_observation = None if candidate_rule is None else (apply_rule(informative_probe, candidate_rule) if true_mode == "anchor" else apply_rule(informative_probe, alternate_rule))
    misleading_observation = None if candidate_rule is None else apply_rule(informative_probe, candidate_rule)
    probes: List[Dict[str, Any]] = []
    if condition == "ambiguous_distractor":
        probes = [
            probe_record("cheap_distractor", DISTRACTOR_COST, equal_probe, candidate_rule, alternate_rule, apply_rule(equal_probe, candidate_rule) if candidate_rule else None, True),
            probe_record("informative", INFORMATIVE_COST, informative_probe, candidate_rule, alternate_rule, truthful_observation, True),
        ]
    elif condition == "ambiguous_expensive":
        probes = [
            probe_record("cheap_distractor", DISTRACTOR_COST, equal_probe, candidate_rule, alternate_rule, apply_rule(equal_probe, candidate_rule) if candidate_rule else None, True),
            probe_record("expensive_informative", EXPENSIVE_COST, informative_probe, candidate_rule, alternate_rule, truthful_observation, True),
        ]
    elif condition == "ambiguous_multi":
        second_probe = clone_tree(informative_probe)
        probes = [
            probe_record("cheap_distractor", DISTRACTOR_COST, equal_probe, candidate_rule, alternate_rule, apply_rule(equal_probe, candidate_rule) if candidate_rule else None, True),
            probe_record("medium_informative", INFORMATIVE_COST, informative_probe, candidate_rule, alternate_rule, truthful_observation, True),
            probe_record("expensive_informative", EXPENSIVE_COST, second_probe, candidate_rule, alternate_rule, truthful_observation, True),
        ]
    elif condition == "no_information":
        probes = [
            probe_record("distractor_a", DISTRACTOR_COST, equal_probe, candidate_rule, alternate_rule, apply_rule(equal_probe, candidate_rule) if candidate_rule else None, True),
            probe_record("distractor_b", INFORMATIVE_COST, equal_probe, candidate_rule, alternate_rule, apply_rule(equal_probe, candidate_rule) if candidate_rule else None, True),
        ]
    elif condition == "unavailable_evidence":
        probes = [
            probe_record("unavailable", INFORMATIVE_COST, informative_probe, candidate_rule, alternate_rule, None, False),
        ]
    elif condition == "misleading_evidence":
        probes = [
            probe_record("misleading_informative", INFORMATIVE_COST, informative_probe, candidate_rule, alternate_rule, misleading_observation, True),
            probe_record("distractor", DISTRACTOR_COST, equal_probe, candidate_rule, alternate_rule, apply_rule(equal_probe, candidate_rule) if candidate_rule else None, True),
        ]
    elif condition == "role_break":
        probes = []
    else:
        raise ValueError(condition)
    phase = "discovery" if index < int(TASKS_PER_SEED * DISCOVERY_FRACTION) else "heldout"
    valid_transfer = true_mode == "anchor" and candidate_rule is not None
    ambiguous = condition != "role_break"
    return InformationTask(
        task_id=f"c1p-{seed}-{index}-{condition}-{template}",
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
        probes=probes,
        target_output=target_output,
        valid_transfer=valid_transfer,
        ambiguous=ambiguous,
    )


def generate_tasks(seed: int) -> List[InformationTask]:
    return [build_task(seed, index) for index in range(TASKS_PER_SEED)]


def oracle_check(task: InformationTask) -> bool:
    if task.condition != "role_break" and task.candidate_rule is None:
        return False
    if task.true_rule is None:
        return task.condition == "role_break"
    if task.target_output != apply_rule(task.target_tree, task.true_rule):
        return False
    for probe in task.probes:
        if probe["available"] and probe["informative"] and probe["name"] != "misleading_informative":
            expected = probe["candidate_prediction"] if task.true_rule == task.candidate_rule else probe["alternate_prediction"]
            if probe["observed_output"] != expected:
                return False
        if probe["name"] == "misleading_informative" and probe["observed_output"] != probe["candidate_prediction"]:
            return False
    return True


def choose_observation(probe: Mapping[str, Any]) -> Optional[str]:
    candidate_prediction = probe.get("candidate_prediction")
    alternate_prediction = probe.get("alternate_prediction")
    observed = probe.get("observed_output")
    if not probe.get("available") or not probe.get("informative"):
        return None
    matches = []
    if observed == candidate_prediction:
        matches.append("candidate")
    if observed == alternate_prediction:
        matches.append("alternate")
    return matches[0] if len(matches) == 1 else None


def evaluate(policy: str, tasks: Sequence[InformationTask], seed: int) -> Dict[str, Any]:
    outcomes: List[Dict[str, Any]] = []
    for task in tasks:
        cost = BASE_COST
        selected_rule: Optional[str] = None
        decision = "abstain"
        reason = ""
        probe_selected: Optional[Dict[str, Any]] = None
        probe_attempted = False
        probe_informative = False
        resolved_hypothesis: Optional[str] = None
        if task.candidate_rule is None:
            reason = "no_relational_candidate"
        elif policy == "passive_abstention":
            reason = "passive_uncertainty"
        elif policy == "no_information_control":
            selected_rule = task.candidate_rule
            decision = "reuse_candidate"
            reason = "unverified_reuse"
        else:
            available = [probe for probe in task.probes if probe.get("available")]
            if policy == "fixed_probe":
                probe_selected = available[0] if available else None
            elif policy == "targeted_probe":
                informative = [probe for probe in available if probe.get("informative")]
                probe_selected = min(informative, key=lambda probe: (float(probe["cost"]), str(probe["name"]))) if informative else None
            if probe_selected is None:
                reason = "no_available_informative_probe"
            else:
                probe_attempted = True
                probe_informative = bool(probe_selected.get("informative"))
                cost += float(probe_selected["cost"])
                resolved_hypothesis = choose_observation(probe_selected)
                if resolved_hypothesis == "candidate":
                    selected_rule = task.candidate_rule
                    decision = "reuse_candidate"
                    reason = "probe_selected_candidate"
                elif resolved_hypothesis == "alternate":
                    selected_rule = task.alternate_rule
                    decision = "revise_to_alternate"
                    reason = "probe_selected_alternate"
                elif not probe_informative:
                    reason = "selected_probe_uninformative"
                else:
                    reason = "probe_ambiguous_or_misleading"
        accepted = selected_rule is not None
        correct = accepted and task.target_output is not None and apply_rule(task.target_tree, selected_rule) == task.target_output
        harmful = accepted and not correct
        false_abstention = task.valid_transfer and not accepted
        resolvable = any(probe.get("available") and probe.get("informative") for probe in task.probes) and task.condition != "misleading_evidence"
        outcome = {
            "task_id": task.task_id,
            "seed": seed,
            "index": task.index,
            "phase": task.phase,
            "condition": task.condition,
            "template": task.template,
            "valid_transfer": task.valid_transfer,
            "ambiguous": task.ambiguous,
            "resolvable_by_truthful_probe": resolvable,
            "candidate_rule": task.candidate_rule,
            "alternate_rule": task.alternate_rule,
            "selected_rule": selected_rule,
            "decision": decision,
            "reason": reason,
            "accepted": accepted,
            "correct": correct,
            "harmful_transfer": harmful,
            "false_abstention": false_abstention,
            "probe_selected": None if probe_selected is None else probe_selected["name"],
            "probe_attempted": probe_attempted,
            "probe_informative": probe_informative,
            "probe_cost": None if probe_selected is None else probe_selected["cost"],
            "resolved_hypothesis": resolved_hypothesis,
            "no_information_case": task.condition in {"no_information", "unavailable_evidence"},
            "misleading_evidence_case": task.condition == "misleading_evidence",
            "cost": cost,
        }
        if harmful:
            outcome["trace"] = {
                "probes": task.probes,
                "selected_probe": probe_selected,
                "source_relational_signature": task.source_relational_signature,
                "target_relational_signature": task.target_relational_signature,
            }
        outcomes.append(outcome)
    return summarize(policy, seed, outcomes)


def summarize_slice(outcomes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    accepted = [item for item in outcomes if item["accepted"]]
    valid = [item for item in outcomes if item["valid_transfer"]]
    harmful = [item for item in outcomes if item["harmful_transfer"]]
    resolvable = [item for item in outcomes if item["resolvable_by_truthful_probe"]]
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
        "resolved_correctly": sum(item["resolvable_by_truthful_probe"] and item["correct"] for item in outcomes),
        "resolvable_tasks": len(resolvable),
        "resolution_rate": sum(item["resolvable_by_truthful_probe"] and item["correct"] for item in outcomes) / max(1, len(resolvable)),
        "unresolved": sum(not item["accepted"] for item in outcomes),
        "no_information_abstentions": sum(item["no_information_case"] and not item["accepted"] for item in outcomes),
        "probe_attempts": sum(item["probe_attempted"] for item in outcomes),
        "informative_probe_attempts": sum(item["probe_attempted"] and item["probe_informative"] for item in outcomes),
        "misleading_harmful": sum(item["misleading_evidence_case"] and item["harmful_transfer"] for item in outcomes),
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
        "valid_transfer_recall", "resolution_rate",
    )
    integer_sum = (
        "tasks", "harmful_transfer_count", "false_abstentions", "resolved_correctly",
        "resolvable_tasks", "unresolved", "no_information_abstentions",
        "probe_attempts", "informative_probe_attempts", "misleading_harmful",
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
        ROOT / "tools" / "run_crucible_c1p_active_information_20261021.py",
        C1N_OUTPUT / "RESULTS.json",
        C1N_OUTPUT / "HASHES.json",
        C1O_OUTPUT / "RESULTS.json",
        C1O_OUTPUT / "HASHES.json",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def render_summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1p active information acquisition",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; hand-coded symbolic probe-selection test.",
        "",
        "## Aggregate results",
        "",
        "| Policy | Accuracy | Coverage | Conditional accuracy | Mean cost | Harmful/all | Harmful/accepted | Resolution rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, item in result["aggregate"].items():
        lines.append(
            f"| `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | {item['conditional_accuracy']:.4f} | "
            f"{item['mean_cost']:.3f} | {item['harmful_transfer_rate_all_tasks']:.4f} | "
            f"{item['harmful_transfer_rate_accepted']:.4f} | {item['resolution_rate']:.4f} |"
        )
    lines.extend([
        "",
        "Harmful/all is harmful transfer divided by all tasks. Harmful/accepted is the conditional rate among accepted transfers.",
        "Resolution rate is correct resolution divided by tasks with a truthful informative probe available.",
        "",
        "## Results by condition",
        "",
        "| Condition | Policy | Accuracy | Coverage | Harmful/all | Resolution rate | Probe attempts | Mean cost |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for condition, policies in result["by_condition"].items():
        for policy, item in policies.items():
            lines.append(
                f"| `{condition}` | `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | "
                f"{item['harmful_transfer_rate_all_tasks']:.4f} | {item['resolution_rate']:.4f} | "
                f"{item['probe_attempts']} | {item['mean_cost']:.3f} |"
            )
    lines.extend(["", "## Results by phase", "", "| Phase | Policy | Accuracy | Coverage | Harmful/all | Resolution rate | Mean cost |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for phase, policies in result["by_phase"].items():
        for policy, item in policies.items():
            lines.append(
                f"| `{phase}` | `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | "
                f"{item['harmful_transfer_rate_all_tasks']:.4f} | {item['resolution_rate']:.4f} | {item['mean_cost']:.3f} |"
            )
    lines.extend(["", "## Investigation diagnostics", "", "| Policy | Probe attempts | Informative probes | No-information abstentions | Misleading harmful cases |", "| --- | ---: | ---: | ---: | ---: |"])
    for policy, item in result["aggregate"].items():
        lines.append(f"| `{policy}` | {item['probe_attempts']} | {item['informative_probe_attempts']} | {item['no_information_abstentions']} | {item['misleading_harmful']} |")
    lines.extend(["", "## Gates", ""])
    for name, value in result["gates"].items():
        lines.append(f"- `{name}`: `{value}`.")
    lines.extend([
        "",
        "This is an explicit symbolic probe-selection result, not evidence that a neural model learned active reasoning.",
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
        "c1o": verify_index(C1O_OUTPUT / "HASHES.json"),
    }
    all_tasks: List[InformationTask] = []
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
        raise RuntimeError(f"C1p oracle generation failure: {oracle_failures[:5]}")
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
    ambiguous_conditions = [condition for condition in CONDITIONS if condition != "role_break"]
    targeted_ambiguous = [outcome for run in all_runs if run["policy"] == "targeted_probe" for outcome in run["outcomes"] if outcome["condition"] in ambiguous_conditions]
    passive_ambiguous = [outcome for run in all_runs if run["policy"] == "passive_abstention" for outcome in run["outcomes"] if outcome["condition"] in ambiguous_conditions]
    no_info_ambiguous = [outcome for run in all_runs if run["policy"] == "no_information_control" for outcome in run["outcomes"] if outcome["condition"] in ambiguous_conditions]
    targeted_info_cases = condition_results["ambiguous_distractor"]["targeted_probe"]["informative_probe_attempts"] + condition_results["ambiguous_multi"]["targeted_probe"]["informative_probe_attempts"]
    fixed_info_cases = condition_results["ambiguous_distractor"]["fixed_probe"]["informative_probe_attempts"] + condition_results["ambiguous_multi"]["fixed_probe"]["informative_probe_attempts"]
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
            "targeted_resolves_more_than_passive": aggregate["targeted_probe"]["resolved_correctly"] > aggregate["passive_abstention"]["resolved_correctly"],
            "targeted_harm_below_unverified_control": sum(item["harmful_transfer"] for item in targeted_ambiguous) < sum(item["harmful_transfer"] for item in no_info_ambiguous),
            "targeted_abstains_without_information": condition_results["no_information"]["targeted_probe"]["no_information_abstentions"] > 0 and condition_results["unavailable_evidence"]["targeted_probe"]["no_information_abstentions"] > 0,
            "targeted_selects_more_informative_than_fixed": targeted_info_cases > fixed_info_cases,
            "misleading_evidence_visible": aggregate["targeted_probe"]["misleading_harmful"] > 0,
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
            "distractor_cost": DISTRACTOR_COST,
            "informative_cost": INFORMATIVE_COST,
            "expensive_cost": EXPENSIVE_COST,
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
                    "selected_probe": outcome["probe_selected"],
                    "probe_attempted": outcome["probe_attempted"],
                    "probe_informative": outcome["probe_informative"],
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
