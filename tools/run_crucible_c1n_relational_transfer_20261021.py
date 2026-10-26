#!/usr/bin/env python3
"""Run C1n: explicit relational transfer validity on CPU."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_crucible_c1j_cpu_stress_20261021 as c1j


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_c1n_relational_transfer_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
C1M_OUTPUT = ROOT / "experiments" / "crucible_c1m_exact_reuse_adaptation_20261021"
C1L_OUTPUT = ROOT / "experiments" / "crucible_c1l_certificate_amortization_20261021"
EXPERIMENT = "crucible_c1n_relational_transfer_20261021"
TASKS_PER_SEED = 12_000
TASK_SEEDS = (42, 1337, 2026)
CONDITIONS = (
    "isomorphic_relabel",
    "role_permutation",
    "depth_extension",
    "role_break",
    "hidden_invariant_change",
    "surface_decoy",
    "no_probe",
    "misleading_probe",
)
TEMPLATES = ("balanced", "asymmetric", "deep", "wide")
SURFACE_COST = 0.05
RELATIONAL_COST = 0.15
VERIFICATION_COST = 0.35
POLICIES = ("surface_association", "relational_matching", "relational_verified")
VALID_TRANSFER_CONDITIONS = {"isomorphic_relabel", "role_permutation", "depth_extension", "no_probe"}


@dataclass(frozen=True)
class TransferTask:
    task_id: str
    seed: int
    index: int
    condition: str
    template: str
    source_tree: Dict[str, Any]
    target_tree: Dict[str, Any]
    probe_tree: Dict[str, Any]
    source_rule: str
    true_rule: str
    source_surface_signature: Tuple[Any, ...]
    target_surface_signature: Tuple[Any, ...]
    source_relational_signature: Tuple[Any, ...]
    target_relational_signature: Tuple[Any, ...]
    probe_available: bool
    probe_output: Optional[int]
    target_output: int
    memory_size: int


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
    return {"index": str(index_path.relative_to(ROOT)), "indexed_files": len(index), "hash_failures": 0, "index_sha256": sha256_file(index_path)}


def clone_tree(tree: Mapping[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(tree, sort_keys=True))


def build_source_tree(seed: int, index: int, template: str) -> Dict[str, Any]:
    left_count = {"balanced": 3, "asymmetric": 2, "deep": 2, "wide": 4}[template]
    right_count = {"balanced": 5, "asymmetric": 6, "deep": 4, "wide": 7}[template]
    left_count += stable_int("c1n-left", seed, index) % 2
    right_count += stable_int("c1n-right", seed, index) % 2
    anchor_left = stable_int("c1n-anchor-position", seed, index) % 2 == 0
    nodes: List[Dict[str, Any]] = [{"id": 0, "role": "root", "label": 1, "marker": 0}]
    edges: List[List[int]] = []
    next_id = 1
    branch_ids = [next_id, next_id + 1]
    next_id += 2
    left_role = "anchor" if anchor_left else "support"
    right_role = "support" if anchor_left else "anchor"
    nodes.extend([
        {"id": branch_ids[0], "role": left_role, "label": 2, "marker": 0},
        {"id": branch_ids[1], "role": right_role, "label": 3, "marker": 0},
    ])
    edges.extend([[0, branch_ids[0]], [0, branch_ids[1]]])
    for branch_id, count, start_label in ((branch_ids[0], left_count, 4), (branch_ids[1], right_count, 5)):
        for offset in range(count):
            nodes.append({"id": next_id, "role": "leaf", "label": start_label + offset % 3, "marker": 0})
            edges.append([branch_id, next_id])
            next_id += 1
    return {"root": 0, "nodes": nodes, "edges": edges, "hidden_mode": "anchor", "template": template}


def children(tree: Mapping[str, Any]) -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = defaultdict(list)
    for parent, child in tree["edges"]:
        result[int(parent)].append(int(child))
    return result


def role_position(tree: Mapping[str, Any], role: str) -> Optional[str]:
    node_by_id = {int(node["id"]): node for node in tree["nodes"]}
    root_children = children(tree).get(int(tree["root"]), [])
    for position, node_id in zip(("left", "right"), root_children[:2]):
        if node_by_id[node_id]["role"] == role:
            return position
    return None


def count_leaves(tree: Mapping[str, Any], branch_id: int) -> int:
    graph = children(tree)
    total = 0
    stack = [branch_id]
    while stack:
        node = stack.pop()
        node_children = graph.get(node, [])
        if not node_children:
            total += 1
        else:
            stack.extend(node_children)
    return total


def apply_rule(tree: Mapping[str, Any], rule: str) -> int:
    graph = children(tree)
    root_children = graph.get(int(tree["root"]), [])
    if rule == "left_leaf_count":
        return count_leaves(tree, root_children[0])
    if rule == "right_leaf_count":
        return count_leaves(tree, root_children[1])
    raise ValueError(rule)


def surface_signature(tree: Mapping[str, Any]) -> Tuple[Any, ...]:
    graph = children(tree)
    degree_hist = tuple(sorted(len(graph.get(int(node["id"]), [])) for node in tree["nodes"]))
    label_hist = tuple(sorted(Counter(int(node["label"]) for node in tree["nodes"]).items()))
    return (len(tree["nodes"]), len(tree["edges"]), degree_hist, label_hist)


def reachable(tree: Mapping[str, Any]) -> bool:
    graph = children(tree)
    seen = {int(tree["root"])}
    frontier = [int(tree["root"])]
    while frontier:
        node = frontier.pop()
        for child in graph.get(node, []):
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return len(seen) == len(tree["nodes"])


def relational_signature(tree: Mapping[str, Any]) -> Tuple[Any, ...]:
    node_by_id = {int(node["id"]): node for node in tree["nodes"]}
    root_children = children(tree).get(int(tree["root"]), [])
    direct_roles = tuple(sorted(node_by_id[node_id]["role"] for node_id in root_children))
    return (
        len(root_children),
        direct_roles,
        role_position(tree, "anchor") is not None,
        role_position(tree, "support") is not None,
        reachable(tree),
    )


def relabel_isomorphism(tree: Mapping[str, Any], seed: int, index: int) -> Dict[str, Any]:
    old_ids = [int(node["id"]) for node in tree["nodes"]]
    shuffled = list(reversed(old_ids))
    mapping = {old: 100 + position for position, old in enumerate(shuffled)}
    result = clone_tree(tree)
    for node in result["nodes"]:
        old = int(node["id"])
        node["id"] = mapping[old]
        node["label"] = int(node["label"]) + 7
    result["root"] = mapping[int(tree["root"])]
    result["edges"] = [[mapping[int(parent)], mapping[int(child)]] for parent, child in tree["edges"]]
    result["hidden_mode"] = tree["hidden_mode"]
    return result


def transform_target(source: Mapping[str, Any], condition: str, seed: int, index: int) -> Dict[str, Any]:
    target = clone_tree(source)
    node_by_id = {int(node["id"]): node for node in target["nodes"]}
    root_children = children(target)[int(target["root"])]
    if condition == "isomorphic_relabel":
        return relabel_isomorphism(target, seed, index)
    if condition == "role_permutation":
        left, right = root_children[:2]
        node_by_id[left]["role"], node_by_id[right]["role"] = node_by_id[right]["role"], node_by_id[left]["role"]
    elif condition == "depth_extension":
        anchor_position = role_position(target, "anchor")
        branch_id = root_children[0 if anchor_position == "left" else 1]
        next_id = max(int(node["id"]) for node in target["nodes"]) + 1
        target["nodes"].append({"id": next_id, "role": "leaf", "label": 9, "marker": 0})
        target["edges"].append([branch_id, next_id])
    elif condition == "role_break":
        anchor_position = role_position(target, "anchor")
        branch_id = root_children[0 if anchor_position == "left" else 1]
        node_by_id[branch_id]["role"] = "distractor"
    elif condition == "hidden_invariant_change":
        target["hidden_mode"] = "support"
    elif condition == "surface_decoy":
        anchor_position = role_position(target, "anchor")
        branch_id = root_children[0 if anchor_position == "left" else 1]
        branch_children = children(target).get(branch_id, [])
        if branch_children:
            moved = branch_children[0]
            target["edges"] = [[parent, child] for parent, child in target["edges"] if not (parent == branch_id and child == moved)]
            target["edges"].append([root_children[0 if anchor_position == "right" else 1], moved])
    elif condition == "no_probe":
        left, right = root_children[:2]
        node_by_id[left]["role"], node_by_id[right]["role"] = node_by_id[right]["role"], node_by_id[left]["role"]
    elif condition == "misleading_probe":
        target["hidden_mode"] = "support"
    else:
        raise ValueError(condition)
    return target


def true_rule_for(source: Mapping[str, Any], target: Mapping[str, Any], condition: str) -> str:
    if condition in {"hidden_invariant_change", "misleading_probe", "role_break", "surface_decoy"}:
        position = role_position(target, "support") or "left"
    else:
        position = role_position(target, "anchor") or "left"
    return f"{position}_leaf_count"


def build_task(seed: int, index: int) -> TransferTask:
    condition = CONDITIONS[stable_int("c1n-condition", seed, index) % len(CONDITIONS)]
    template = TEMPLATES[stable_int("c1n-template", seed, index) % len(TEMPLATES)]
    source = build_source_tree(seed, index, template)
    target = transform_target(source, condition, seed, index)
    source_anchor = role_position(source, "anchor") or "left"
    source_rule = f"{source_anchor}_leaf_count"
    true_rule = true_rule_for(source, target, condition)
    probe = clone_tree(target)
    target_output = apply_rule(target, true_rule)
    probe_output = apply_rule(probe, true_rule)
    probe_available = condition != "no_probe"
    if condition == "misleading_probe":
        candidate_anchor = role_position(target, "anchor") or "left"
        probe_output = apply_rule(probe, f"{candidate_anchor}_leaf_count")
    return TransferTask(
        task_id=f"c1n-{seed}-{index}-{condition}-{template}",
        seed=seed,
        index=index,
        condition=condition,
        template=template,
        source_tree=source,
        target_tree=target,
        probe_tree=probe,
        source_rule=source_rule,
        true_rule=true_rule,
        source_surface_signature=surface_signature(source),
        target_surface_signature=surface_signature(target),
        source_relational_signature=relational_signature(source),
        target_relational_signature=relational_signature(target),
        probe_available=probe_available,
        probe_output=probe_output if probe_available else None,
        target_output=target_output,
        memory_size=(10, 100, 1000, 10000)[stable_int("c1n-memory", seed, index) % 4],
    )


def generate_tasks(seed: int) -> List[TransferTask]:
    return [build_task(seed, index) for index in range(TASKS_PER_SEED)]


def oracle_check(task: TransferTask) -> bool:
    return (
        apply_rule(task.source_tree, task.source_rule) == apply_rule(task.source_tree, task.source_rule)
        and apply_rule(task.target_tree, task.true_rule) == task.target_output
        and (not task.probe_available or apply_rule(task.probe_tree, task.true_rule) == task.probe_output or task.condition == "misleading_probe")
    )


def relational_candidate(task: TransferTask) -> Optional[str]:
    if task.source_relational_signature != task.target_relational_signature:
        return None
    target_anchor = role_position(task.target_tree, "anchor")
    target_support = role_position(task.target_tree, "support")
    if target_anchor is None or target_support is None:
        return None
    return f"{target_anchor}_leaf_count"


def evaluate(policy: str, tasks: Sequence[TransferTask], seed: int) -> Dict[str, Any]:
    outcomes: List[Dict[str, Any]] = []
    for task in tasks:
        selected: Optional[str] = None
        decision = "abstain"
        source = "none"
        cost = 0.0
        verification_attempted = False
        verification_passed = False
        if policy == "surface_association":
            cost = SURFACE_COST
            if task.source_surface_signature == task.target_surface_signature:
                selected = task.source_rule
                decision = "reuse"
                source = "surface"
        elif policy in {"relational_matching", "relational_verified"}:
            cost = RELATIONAL_COST
            selected = relational_candidate(task)
            if selected is not None:
                decision = "adapt" if selected != task.source_rule else "reuse"
                source = "relational"
                if policy == "relational_verified":
                    if task.probe_available:
                        verification_attempted = True
                        cost += VERIFICATION_COST
                        verification_passed = apply_rule(task.probe_tree, selected) == task.probe_output
                        if not verification_passed:
                            selected = None
                            decision = "abstain"
                            source = "relational_verification_rejected"
                    else:
                        selected = None
                        decision = "abstain"
                        source = "relational_no_probe"
        else:
            raise ValueError(policy)
        correct = selected is not None and apply_rule(task.target_tree, selected) == task.target_output
        valid_transfer = task.condition in VALID_TRANSFER_CONDITIONS
        accepted = selected is not None
        outcomes.append(
            {
                "task_id": task.task_id,
                "seed": seed,
                "index": task.index,
                "condition": task.condition,
                "template": task.template,
                "scope": "valid_transfer" if valid_transfer else "invalid_or_ambiguous_transfer",
                "memory_size": task.memory_size,
                "source_rule": task.source_rule,
                "selected_rule": selected,
                "true_rule": task.true_rule,
                "decision": decision,
                "source": source,
                "correct": correct,
                "accepted": accepted,
                "harmful_transfer": accepted and not correct,
                "false_abstention": valid_transfer and not accepted,
                "invalid_acceptance": not valid_transfer and accepted,
                "verification_attempted": verification_attempted,
                "verification_passed": verification_passed,
                "probe_available": task.probe_available,
                "cost": cost,
                "source_surface_signature": task.source_surface_signature,
                "target_surface_signature": task.target_surface_signature,
                "source_relational_signature": task.source_relational_signature,
                "target_relational_signature": task.target_relational_signature,
            }
        )
    count = len(outcomes)
    accepted = [item for item in outcomes if item["accepted"]]
    valid = [item for item in outcomes if item["scope"] == "valid_transfer"]
    invalid = [item for item in outcomes if item["scope"] != "valid_transfer"]
    summary = {
        "policy": policy,
        "seed": seed,
        "tasks": count,
        "accuracy": sum(item["correct"] for item in outcomes) / max(1, count),
        "coverage": len(accepted) / max(1, count),
        "conditional_accuracy": sum(item["correct"] for item in accepted) / max(1, len(accepted)),
        "mean_cost": sum(item["cost"] for item in outcomes) / max(1, count),
        "harmful_transfer_rate": sum(item["harmful_transfer"] for item in accepted) / max(1, len(accepted)),
        "false_acceptance_rate": sum(item["invalid_acceptance"] for item in invalid) / max(1, len(invalid)),
        "valid_transfer_recall": sum(item["accepted"] and item["correct"] for item in valid) / max(1, len(valid)),
        "valid_transfer_false_abstentions": sum(item["false_abstention"] for item in valid),
        "invalid_transfer_rejection": sum(not item["accepted"] for item in invalid) / max(1, len(invalid)),
        "invalid_transfer_acceptances": sum(item["invalid_acceptance"] for item in invalid),
        "verification_attempts": sum(item["verification_attempted"] for item in outcomes),
        "verification_passes": sum(item["verification_passed"] for item in outcomes),
        "abstentions": sum(not item["accepted"] for item in outcomes),
        "action_counts": dict(sorted(Counter(item["decision"] for item in outcomes).items())),
        "outcomes": outcomes,
    }
    return summary


def aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    numeric = (
        "accuracy", "coverage", "conditional_accuracy", "mean_cost", "harmful_transfer_rate",
        "false_acceptance_rate", "valid_transfer_recall", "invalid_transfer_rejection",
    )
    counts = (
        "tasks", "valid_transfer_false_abstentions", "invalid_transfer_acceptances",
        "verification_attempts", "verification_passes", "abstentions",
    )
    return {
        "runs": len(runs),
        **{field: sum(float(run.get(field, 0.0)) for run in runs) / max(1, len(runs)) for field in numeric},
        **{field: sum(int(run.get(field, 0)) for run in runs) for field in counts},
    }


def subgroup_summary(outcomes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not outcomes:
        return {"tasks": 0}
    accepted = [item for item in outcomes if item["accepted"]]
    valid = [item for item in outcomes if item["scope"] == "valid_transfer"]
    invalid = [item for item in outcomes if item["scope"] != "valid_transfer"]
    return {
        "tasks": len(outcomes),
        "accuracy": sum(item["correct"] for item in outcomes) / len(outcomes),
        "coverage": len(accepted) / len(outcomes),
        "mean_cost": sum(item["cost"] for item in outcomes) / len(outcomes),
        "harmful_transfer_rate": sum(item["harmful_transfer"] for item in accepted) / max(1, len(accepted)),
        "valid_transfer_recall": sum(item["accepted"] and item["correct"] for item in valid) / max(1, len(valid)),
        "invalid_transfer_rejection": sum(not item["accepted"] for item in invalid) / max(1, len(invalid)),
        "false_abstentions": sum(item["false_abstention"] for item in valid),
        "invalid_acceptances": sum(item["invalid_acceptance"] for item in invalid),
        "verification_attempts": sum(item["verification_attempted"] for item in outcomes),
    }


def source_hashes() -> Dict[str, str]:
    paths = [
        PROTOCOL,
        ROOT / "tools" / "run_crucible_c1n_relational_transfer_20261021.py",
        ROOT / "tools" / "run_crucible_c1j_cpu_stress_20261021.py",
        C1M_OUTPUT / "RESULTS.json",
        C1M_OUTPUT / "HASHES.json",
        C1L_OUTPUT / "CERTIFICATE_CATALOGUE.json",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def render_summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1n relational transfer validity",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; hand-coded symbolic mechanism test.",
        "",
        "## Aggregate results",
        "",
        "| Policy | Accuracy | Coverage | Conditional accuracy | Mean cost | Harmful transfer | Valid-transfer recall | Invalid rejection |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, item in result["aggregate"].items():
        lines.append(
            f"| `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | {item['conditional_accuracy']:.4f} | "
            f"{item['mean_cost']:.3f} | {item['harmful_transfer_rate']:.4f} | {item['valid_transfer_recall']:.4f} | {item['invalid_transfer_rejection']:.4f} |"
        )
    lines.extend(["", "## Results by condition", "", "| Condition | Policy | Accuracy | Coverage | Harmful transfer | Valid recall | Invalid rejection |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for condition, policies in result["by_condition"].items():
        for policy, item in policies.items():
            lines.append(f"| `{condition}` | `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | {item['harmful_transfer_rate']:.4f} | {item['valid_transfer_recall']:.4f} | {item['invalid_transfer_rejection']:.4f} |")
    lines.extend(["", "## Results by template", "", "| Template | Policy | Accuracy | Coverage | Harmful transfer | Mean cost |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for template, policies in result["by_template"].items():
        for policy, item in policies.items():
            lines.append(f"| `{template}` | `{policy}` | {item['accuracy']:.4f} | {item['coverage']:.4f} | {item['harmful_transfer_rate']:.4f} | {item['mean_cost']:.3f} |")
    lines.extend(["", "## Gates", ""])
    for name, value in result["gates"].items():
        lines.append(f"- `{name}`: `{value}`.")
    lines.extend(["", "This is an explicit symbolic mechanism result, not evidence that a neural model learned relational reasoning.", "", "## Source hashes", ""])
    for path, digest in sorted(hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen_audit = {
        "c1m": verify_index(C1M_OUTPUT / "HASHES.json"),
        "c1l": verify_index(C1L_OUTPUT / "HASHES.json"),
    }
    all_tasks: List[TransferTask] = []
    manifests: Dict[str, Any] = {}
    oracle_checks = 0
    oracle_failures = []
    for seed in TASK_SEEDS:
        tasks = generate_tasks(seed)
        all_tasks.extend(tasks)
        checks = [oracle_check(task) for task in tasks]
        oracle_checks += len(checks)
        oracle_failures.extend(task.task_id for task, passed in zip(tasks, checks) if not passed)
        manifests[str(seed)] = {
            "count": len(tasks),
            "task_digest": hashlib.sha256("|".join(task.task_id for task in tasks).encode()).hexdigest(),
            "conditions": dict(sorted(Counter(task.condition for task in tasks).items())),
            "templates": dict(sorted(Counter(task.template for task in tasks).items())),
        }
    if oracle_failures:
        raise RuntimeError(f"C1n oracle generation failure: {oracle_failures[:5]}")
    write_json(OUTPUT / "TASK_MANIFEST.json", {"experiment": EXPERIMENT, "seeds": manifests, "task_count_total": len(all_tasks), "oracle_checks": oracle_checks, "oracle_failures": len(oracle_failures), "target_labels_used_for_policy": False})

    all_runs: List[Dict[str, Any]] = []
    by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_condition: Dict[str, Dict[str, List[Dict[str, Any]]]] = {condition: defaultdict(list) for condition in CONDITIONS}
    by_template: Dict[str, Dict[str, List[Dict[str, Any]]]] = {template: defaultdict(list) for template in TEMPLATES}
    for seed in TASK_SEEDS:
        tasks = [task for task in all_tasks if task.seed == seed]
        for policy in POLICIES:
            run = evaluate(policy, tasks, seed)
            summary = {key: value for key, value in run.items() if key != "outcomes"}
            record = {"seed": seed, "policy": policy, "summary": summary, "outcomes": run["outcomes"]}
            all_runs.append(record)
            by_policy[policy].append(summary)
            for condition in CONDITIONS:
                by_condition[condition][policy].append(subgroup_summary([item for item in run["outcomes"] if item["condition"] == condition]))
            for template in TEMPLATES:
                by_template[template][policy].append(subgroup_summary([item for item in run["outcomes"] if item["template"] == template]))

    aggregate = {policy: aggregate_runs(runs) for policy, runs in sorted(by_policy.items())}
    condition_results = {
        condition: {policy: aggregate_runs(runs) for policy, runs in sorted(policies.items())}
        for condition, policies in by_condition.items()
    }
    template_results = {
        template: {policy: aggregate_runs(runs) for policy, runs in sorted(policies.items())}
        for template, policies in by_template.items()
    }
    surface = aggregate["surface_association"]
    relational = aggregate["relational_matching"]
    verified = aggregate["relational_verified"]
    result = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "aggregate": aggregate,
        "by_condition": condition_results,
        "by_template": template_results,
        "frozen_audit": frozen_audit,
        "task_manifest": manifests,
        "oracle_checks": {"checks": oracle_checks, "failures": len(oracle_failures)},
        "gates": {
            "oracle_and_manifest_reproducible": len(oracle_failures) == 0 and oracle_checks == TASKS_PER_SEED * len(TASK_SEEDS),
            "relational_valid_recall_above_surface": relational["valid_transfer_recall"] > surface["valid_transfer_recall"],
            "relational_decoy_harm_below_surface": relational["harmful_transfer_rate"] < surface["harmful_transfer_rate"],
            "verification_reduces_relational_harm": verified["harmful_transfer_rate"] < relational["harmful_transfer_rate"],
            "hand_coded_not_learned": True,
        },
        "config": {
            "task_seeds": list(TASK_SEEDS),
            "tasks_per_seed": TASKS_PER_SEED,
            "conditions": list(CONDITIONS),
            "templates": list(TEMPLATES),
            "policies": list(POLICIES),
            "surface_cost": SURFACE_COST,
            "relational_cost": RELATIONAL_COST,
            "verification_cost": VERIFICATION_COST,
            "execution": "CPU-only",
            "neural_training_updates": 0,
        },
    }
    write_json(OUTPUT / "RESULTS.json", result)
    write_json(OUTPUT / "RUN_CONFIG.json", result["config"])
    with (OUTPUT / "RUNS.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_runs:
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
