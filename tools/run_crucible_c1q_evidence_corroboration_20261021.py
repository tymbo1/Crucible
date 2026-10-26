#!/usr/bin/env python3
"""Run C1q: evidence reliability and corroboration on CPU."""

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
OUTPUT = ROOT / "experiments" / "crucible_c1q_evidence_corroboration_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
C1N_OUTPUT = ROOT / "experiments" / "crucible_c1n_relational_transfer_20261021"
C1O_OUTPUT = ROOT / "experiments" / "crucible_c1o_invariant_discovery_20261021"
C1P_OUTPUT = ROOT / "experiments" / "crucible_c1p_active_information_20261021"
EXPERIMENT = "crucible_c1q_evidence_corroboration_20261021"
TASKS_PER_SEED = 12_000
DISCOVERY_FRACTION = 0.25
TASK_SEEDS = (42, 1337, 2026)
CONDITIONS = (
    "single_misleading",
    "correlated_misinformation",
    "independent_contradiction",
    "truthful_agreement",
    "correlated_truth",
    "no_trustworthy_evidence",
    "unavailable_sources",
    "role_break",
)
TEMPLATES = ("balanced", "asymmetric", "deep", "wide")
POLICIES = ("single_targeted", "naive_corroboration", "independence_aware", "reliability_aware")
BASE_COST = 0.15


@dataclass(frozen=True)
class EvidenceTask:
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
    oracle_kinds: Dict[str, str]
    target_output: Optional[int]
    valid_transfer: bool
    trustworthy_evidence_available: bool


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
    tree["c1q_family"] = "two_hypothesis_evidence_transfer"
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


def source_record(name: str, group: str, cost: float, tree: Mapping[str, Any], candidate: Optional[str], alternate: Optional[str], observed: Optional[int], available: bool, kind: str) -> Dict[str, Any]:
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


def build_task(seed: int, index: int) -> EvidenceTask:
    condition = CONDITIONS[stable_int("c1q-condition", seed, index) % len(CONDITIONS)]
    template = TEMPLATES[stable_int("c1q-template", seed, index) % len(TEMPLATES)]
    source = build_source_tree(seed, index, template)
    target = target_tree_for(source, condition)
    source_rule = rule_for_role(source, "anchor")
    candidate_rule = rule_for_role(target, "anchor")
    alternate_rule = rule_for_role(target, "support")
    true_mode = "anchor" if stable_int("c1q-true-mode", seed, index) % 2 == 0 else "support"
    if condition == "role_break":
        true_mode = "support"
    true_rule = None if alternate_rule is None else (candidate_rule if true_mode == "anchor" else alternate_rule)
    target_output = None if true_rule is None else apply_rule(target, true_rule)
    target_probe = clone_tree(target)

    def observation(kind: str) -> Optional[int]:
        if not candidate_rule or not alternate_rule:
            return None
        truthful = apply_rule(target_probe, true_rule) if true_rule else None
        misleading = apply_rule(target_probe, candidate_rule if true_mode == "support" else alternate_rule)
        return truthful if kind == "truthful" else misleading

    specs: List[Tuple[str, str, float, bool, str]]
    if condition == "single_misleading":
        specs = [("A1", "A", 0.20, True, "misleading"), ("B1", "B", 0.80, True, "truthful")]
    elif condition == "correlated_misinformation":
        specs = [("A1", "A", 0.20, True, "misleading"), ("A2", "A", 0.25, True, "misleading"), ("B1", "B", 1.00, True, "truthful")]
    elif condition == "independent_contradiction":
        specs = [("A1", "A", 0.20, True, "truthful"), ("B1", "B", 0.25, True, "misleading"), ("C1", "C", 0.80, True, "truthful")]
    elif condition == "truthful_agreement":
        specs = [("A1", "A", 0.20, True, "truthful"), ("B1", "B", 0.25, True, "truthful"), ("C1", "C", 0.90, True, "misleading")]
    elif condition == "correlated_truth":
        specs = [("A1", "A", 0.20, True, "truthful"), ("A2", "A", 0.25, True, "truthful"), ("B1", "B", 0.80, True, "misleading")]
    elif condition == "no_trustworthy_evidence":
        specs = [("A1", "A", 0.20, True, "misleading"), ("B1", "B", 0.25, True, "misleading"), ("C1", "C", 0.80, False, "misleading")]
    elif condition == "unavailable_sources":
        specs = [("A1", "A", 0.20, False, "truthful"), ("B1", "B", 0.25, False, "truthful")]
    elif condition == "role_break":
        specs = []
    else:
        raise ValueError(condition)
    sources: List[Dict[str, Any]] = []
    oracle_kinds: Dict[str, str] = {}
    for name, group, cost, available, kind in specs:
        sources.append(source_record(name, group, cost, target_probe, candidate_rule, alternate_rule, observation(kind) if available else None, available, kind))
        oracle_kinds[name] = kind
    phase = "discovery" if index < int(TASKS_PER_SEED * DISCOVERY_FRACTION) else "heldout"
    trustworthy = condition in {"single_misleading", "correlated_misinformation", "independent_contradiction", "truthful_agreement", "correlated_truth"}
    return EvidenceTask(
        task_id=f"c1q-{seed}-{index}-{condition}-{template}",
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
        oracle_kinds=oracle_kinds,
        target_output=target_output,
        valid_transfer=true_mode == "anchor" and candidate_rule is not None,
        trustworthy_evidence_available=trustworthy,
    )


def generate_tasks(seed: int) -> List[EvidenceTask]:
    return [build_task(seed, index) for index in range(TASKS_PER_SEED)]


def oracle_check(task: EvidenceTask) -> bool:
    if task.condition != "role_break" and task.candidate_rule is None:
        return False
    if task.true_rule is None:
        return task.condition == "role_break"
    if task.target_output != apply_rule(task.target_tree, task.true_rule):
        return False
    for source in task.sources:
        if not source["available"]:
            if source["observed_output"] is not None:
                return False
            continue
        kind = task.oracle_kinds[source["name"]]
        expected = source["alternate_prediction"] if kind == "misleading" and task.true_rule == task.candidate_rule else source["candidate_prediction"] if kind == "misleading" else (source["candidate_prediction"] if task.true_rule == task.candidate_rule else source["alternate_prediction"])
        if source["observed_output"] != expected:
            return False
    return True


def source_hypothesis(source: Mapping[str, Any]) -> Optional[str]:
    if not source.get("available") or not source.get("informative"):
        return None
    observed = source.get("observed_output")
    matches = []
    if observed == source.get("candidate_prediction"):
        matches.append("candidate")
    if observed == source.get("alternate_prediction"):
        matches.append("alternate")
    return matches[0] if len(matches) == 1 else None


def representative_sources(sources: Sequence[Mapping[str, Any]], state: Mapping[str, Any], distinct: bool = False) -> List[Mapping[str, Any]]:
    available = [source for source in sources if source.get("available") and source.get("informative")]
    if not distinct:
        return sorted(available, key=lambda item: (float(item["cost"]), str(item["name"])))
    best_by_group: Dict[str, Mapping[str, Any]] = {}
    for source in available:
        group = str(source["group"])
        current = best_by_group.get(group)
        if current is None or (float(source["cost"]), str(source["name"])) < (float(current["cost"]), str(current["name"])):
            best_by_group[group] = source
    scores = state.get("group_scores", {})
    return sorted(best_by_group.values(), key=lambda item: (-float(scores.get(str(item["group"]), 0.5)), float(item["cost"]), str(item["name"])))


def update_consensus_scores(state: Dict[str, Any], selected: Sequence[Mapping[str, Any]], accepted_hypothesis: Optional[str]) -> None:
    if accepted_hypothesis is None:
        return
    scores = state.setdefault("group_scores", {})
    for source in selected:
        group = str(source["group"])
        scores.setdefault(group, 0.5)
        observed = source_hypothesis(source)
        if observed == accepted_hypothesis:
            scores[group] = min(0.99, scores[group] + 0.05)
        elif observed is not None:
            scores[group] = max(0.01, scores[group] - 0.05)


def evaluate(policy: str, tasks: Sequence[EvidenceTask], seed: int) -> Dict[str, Any]:
    state: Dict[str, Any] = {"group_scores": {}, "consensus_updates": 0, "contradictions": 0, "third_source_attempts": 0}
    outcomes: List[Dict[str, Any]] = []
    for task in tasks:
        cost = BASE_COST
        selected_rule: Optional[str] = None
        decision = "abstain"
        reason = ""
        selected_sources: List[Mapping[str, Any]] = []
        hypotheses: List[str] = []
        if task.candidate_rule is None:
            reason = "no_relational_candidate"
        elif policy == "single_targeted":
            ordered = representative_sources(task.sources, state, distinct=False)
            if ordered:
                selected_sources = [ordered[0]]
                cost += float(ordered[0]["cost"])
                hypothesis = source_hypothesis(ordered[0])
                if hypothesis == "candidate":
                    selected_rule = task.candidate_rule
                    decision = "accept_candidate"
                elif hypothesis == "alternate":
                    selected_rule = task.alternate_rule
                    decision = "revise_to_alternate"
                reason = "single_source_observation" if hypothesis else "single_source_unresolved"
            else:
                reason = "no_informative_source"
        elif policy == "naive_corroboration":
            ordered = representative_sources(task.sources, state, distinct=False)
            selected_sources = ordered[:2]
            cost += sum(float(source["cost"]) for source in selected_sources)
            hypotheses = [hypothesis for source in selected_sources if (hypothesis := source_hypothesis(source)) is not None]
            if len(selected_sources) >= 2 and len(hypotheses) == 2 and hypotheses[0] == hypotheses[1]:
                if hypotheses[0] == "candidate":
                    selected_rule = task.candidate_rule
                    decision = "accept_candidate"
                else:
                    selected_rule = task.alternate_rule
                    decision = "revise_to_alternate"
                reason = "same_output_corroboration"
            elif selected_sources:
                state["contradictions"] += 1
                reason = "corroboration_conflict_or_unresolved"
            else:
                reason = "no_informative_source"
        elif policy == "independence_aware":
            ordered = representative_sources(task.sources, state, distinct=True)
            selected_sources = ordered[:2]
            cost += sum(float(source["cost"]) for source in selected_sources)
            hypotheses = [hypothesis for source in selected_sources if (hypothesis := source_hypothesis(source)) is not None]
            if len(selected_sources) >= 2 and len(hypotheses) == 2 and hypotheses[0] == hypotheses[1]:
                if hypotheses[0] == "candidate":
                    selected_rule = task.candidate_rule
                    decision = "accept_candidate"
                else:
                    selected_rule = task.alternate_rule
                    decision = "revise_to_alternate"
                reason = "independent_agreement"
                update_consensus_scores(state, selected_sources, hypotheses[0])
                state["consensus_updates"] += 1
            elif selected_sources:
                state["contradictions"] += 1
                reason = "independent_contradiction"
            else:
                reason = "no_independent_sources"
        elif policy == "reliability_aware":
            ordered = representative_sources(task.sources, state, distinct=True)
            selected_sources = ordered[:2]
            cost += sum(float(source["cost"]) for source in selected_sources)
            hypotheses = [hypothesis for source in selected_sources if (hypothesis := source_hypothesis(source)) is not None]
            if len(selected_sources) >= 2 and len(hypotheses) == 2 and hypotheses[0] == hypotheses[1]:
                selected_hypothesis = hypotheses[0]
                if selected_hypothesis == "candidate":
                    selected_rule = task.candidate_rule
                    decision = "accept_candidate"
                else:
                    selected_rule = task.alternate_rule
                    decision = "revise_to_alternate"
                reason = "independent_agreement"
                update_consensus_scores(state, selected_sources, selected_hypothesis)
                state["consensus_updates"] += 1
            elif len(selected_sources) >= 2:
                state["contradictions"] += 1
                remaining_groups = {str(source["group"]) for source in ordered[2:]}
                if remaining_groups:
                    third = ordered[2]
                    selected_sources.append(third)
                    state["third_source_attempts"] += 1
                    cost += float(third["cost"])
                    hypotheses = [hypothesis for source in selected_sources if (hypothesis := source_hypothesis(source)) is not None]
                    counts = Counter(hypotheses)
                    majority = [hypothesis for hypothesis, count in counts.items() if count >= 2]
                    if len(majority) == 1:
                        selected_hypothesis = majority[0]
                        if selected_hypothesis == "candidate":
                            selected_rule = task.candidate_rule
                            decision = "accept_candidate"
                        else:
                            selected_rule = task.alternate_rule
                            decision = "revise_to_alternate"
                        reason = "third_source_majority"
                        update_consensus_scores(state, selected_sources, selected_hypothesis)
                        state["consensus_updates"] += 1
                    else:
                        reason = "three_source_conflict"
                else:
                    reason = "independent_contradiction_no_tiebreaker"
            else:
                reason = "no_independent_sources"
        else:
            raise ValueError(policy)
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
            "trustworthy_evidence_available": task.trustworthy_evidence_available,
            "selected_rule": selected_rule,
            "decision": decision,
            "reason": reason,
            "accepted": accepted,
            "correct": correct,
            "harmful_transfer": harmful,
            "false_abstention": false_abstention,
            "selected_sources": [source["name"] for source in selected_sources],
            "selected_groups": [source["group"] for source in selected_sources],
            "source_hypotheses": [source_hypothesis(source) for source in selected_sources],
            "source_cost": sum(float(source["cost"]) for source in selected_sources),
            "contradiction": len(set(hypotheses)) > 1,
            "third_source_attempted": len(selected_sources) >= 3,
            "group_scores_after": dict(state["group_scores"]),
            "cost": cost,
        }
        if harmful:
            outcome["trace"] = {
                "sources": list(task.sources),
                "selected_sources": outcome["selected_sources"],
                "selected_groups": outcome["selected_groups"],
                "source_hypotheses": outcome["source_hypotheses"],
            }
        outcomes.append(outcome)
    return summarize(policy, seed, outcomes, state)


def summarize_slice(outcomes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    accepted = [item for item in outcomes if item["accepted"]]
    valid = [item for item in outcomes if item["valid_transfer"]]
    harmful = [item for item in outcomes if item["harmful_transfer"]]
    trustworthy = [item for item in outcomes if item["trustworthy_evidence_available"]]
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
        "trustworthy_resolution_count": sum(item["trustworthy_evidence_available"] and item["correct"] for item in outcomes),
        "trustworthy_resolution_rate": sum(item["trustworthy_evidence_available"] and item["correct"] for item in outcomes) / max(1, len(trustworthy)),
        "contradictions": sum(item["contradiction"] for item in outcomes),
        "third_source_attempts": sum(item["third_source_attempted"] for item in outcomes),
        "no_trustworthy_harmful": sum(item["condition"] == "no_trustworthy_evidence" and item["harmful_transfer"] for item in outcomes),
        "unavailable_abstentions": sum(item["condition"] == "unavailable_sources" and not item["accepted"] for item in outcomes),
    }


def summarize(policy: str, seed: int, outcomes: Sequence[Mapping[str, Any]], state: Mapping[str, Any]) -> Dict[str, Any]:
    summary = summarize_slice(outcomes)
    summary.update({
        "policy": policy,
        "seed": seed,
        "phase_summaries": {
            phase: summarize_slice([item for item in outcomes if item["phase"] == phase])
            for phase in ("discovery", "heldout")
        },
        "group_scores_final": dict(state["group_scores"]),
        "consensus_updates": int(state["consensus_updates"]),
        "state_contradictions": int(state["contradictions"]),
        "third_source_attempts_state": int(state["third_source_attempts"]),
        "outcomes": list(outcomes),
    })
    return summary


def aggregate_summaries(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    numeric_mean = (
        "accuracy", "coverage", "conditional_accuracy", "mean_cost",
        "harmful_transfer_rate_all_tasks", "harmful_transfer_rate_accepted",
        "valid_transfer_recall", "trustworthy_resolution_rate",
    )
    integer_sum = (
        "tasks", "harmful_transfer_count", "false_abstentions", "trustworthy_resolution_count",
        "contradictions", "third_source_attempts", "no_trustworthy_harmful", "unavailable_abstentions",
        "consensus_updates", "state_contradictions", "third_source_attempts_state",
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
        ROOT / "tools" / "run_crucible_c1q_evidence_corroboration_20261021.py",
        C1N_OUTPUT / "RESULTS.json",
        C1N_OUTPUT / "HASHES.json",
        C1O_OUTPUT / "RESULTS.json",
        C1O_OUTPUT / "HASHES.json",
        C1P_OUTPUT / "RESULTS.json",
        C1P_OUTPUT / "HASHES.json",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def render_summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1q evidence reliability and corroboration",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; hand-coded symbolic evidence-handling test.",
        "",
        "## Aggregate results",
        "",
        "| Policy | Accuracy | Coverage | Conditional accuracy | Mean cost | Harmful/all | Harmful/accepted | Trustworthy resolution |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, item in result["aggregate"].items():
        lines.append(
            f"| `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | {item['conditional_accuracy']:.4f} | "
            f"{item['mean_cost']:.3f} | {item['harmful_transfer_rate_all_tasks']:.4f} | "
            f"{item['harmful_transfer_rate_accepted']:.4f} | {item['trustworthy_resolution_rate']:.4f} |"
        )
    lines.extend([
        "",
        "Harmful/all is harmful transfer divided by all tasks. Harmful/accepted is the conditional rate among accepted transfers.",
        "Trustworthy resolution is correct resolution divided by tasks with trustworthy evidence available.",
        "",
        "## Results by condition",
        "",
        "| Condition | Policy | Accuracy | Coverage | Harmful/all | Trustworthy resolution | Contradictions | Mean cost |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for condition, policies in result["by_condition"].items():
        for policy, item in policies.items():
            lines.append(
                f"| `{condition}` | `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | "
                f"{item['harmful_transfer_rate_all_tasks']:.4f} | {item['trustworthy_resolution_rate']:.4f} | "
                f"{item['contradictions']} | {item['mean_cost']:.3f} |"
            )
    lines.extend(["", "## Results by phase", "", "| Phase | Policy | Accuracy | Coverage | Harmful/all | Trustworthy resolution | Mean cost |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for phase, policies in result["by_phase"].items():
        for policy, item in policies.items():
            lines.append(
                f"| `{phase}` | `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | "
                f"{item['harmful_transfer_rate_all_tasks']:.4f} | {item['trustworthy_resolution_rate']:.4f} | {item['mean_cost']:.3f} |"
            )
    lines.extend(["", "## Corroboration diagnostics", "", "| Policy | Contradictions | Third-source attempts | No-trustworthy harmful | Unavailable abstentions | Consensus updates |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for policy, item in result["aggregate"].items():
        lines.append(f"| `{policy}` | {item['contradictions']} | {item['third_source_attempts']} | {item['no_trustworthy_harmful']} | {item['unavailable_abstentions']} | {item['consensus_updates']} |")
    lines.extend(["", "## Gates", ""])
    for name, value in result["gates"].items():
        lines.append(f"- `{name}`: `{value}`.")
    lines.extend([
        "",
        "This is an explicit symbolic evidence-handling result, not evidence that a neural model learned source reliability.",
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
        "c1p": verify_index(C1P_OUTPUT / "HASHES.json"),
    }
    all_tasks: List[EvidenceTask] = []
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
        raise RuntimeError(f"C1q oracle generation failure: {oracle_failures[:5]}")
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
    correlated = {policy: condition_results["correlated_misinformation"][policy] for policy in POLICIES}
    contradiction = {policy: condition_results["independent_contradiction"][policy] for policy in POLICIES}
    unavailable = condition_results["unavailable_sources"]
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
            "independence_reduces_correlated_harm": correlated["independence_aware"]["harmful_transfer_rate_all_tasks"] < correlated["naive_corroboration"]["harmful_transfer_rate_all_tasks"],
            "reliability_resolves_third_source_contradictions": aggregate["reliability_aware"]["trustworthy_resolution_count"] > aggregate["independence_aware"]["trustworthy_resolution_count"],
            "safe_policies_abstain_unavailable": unavailable["independence_aware"]["unavailable_abstentions"] > 0 and unavailable["reliability_aware"]["unavailable_abstentions"] > 0,
            "not_always_abstain": aggregate["reliability_aware"]["coverage"] > 0.0,
            "no_trustworthy_failure_visible": aggregate["reliability_aware"]["no_trustworthy_harmful"] > 0,
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
                    "source_hypotheses": outcome["source_hypotheses"],
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
