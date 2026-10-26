#!/usr/bin/env python3
"""CPU-only C1h held-out structural generalization experiment."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_c1h_heldout_generalization_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
EXPERIMENT = "crucible_c1h_heldout_generalization_20261021"
SEEDS = (42, 1337, 2026)
FAMILIES = ("vector", "grid", "graph", "set")
CONDITIONS = (
    "novel_surface",
    "novel_structure",
    "adaptation",
    "invalidated",
    "contradictory",
    "misleading_similarity",
    "unresolvable",
)
APPLICABLE_CONDITIONS = ("novel_surface", "adaptation")
EPISODES_PER_CONDITION = 6
PROBE_COST = 0.8
VERIFY_COST = 0.5
RULES = {
    "vector": ("vector_reverse", "vector_rotate"),
    "grid": ("grid_transpose", "grid_reflect"),
    "graph": ("graph_reach", "graph_direct"),
    "set": ("set_union", "set_intersection"),
}


def stable_int(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def solve(rule: str, family: str, value: Any) -> Any:
    if family == "vector":
        values = list(value["values"])
        if rule == "vector_reverse":
            return list(reversed(values))
        if rule == "vector_rotate":
            return values[1:] + values[:1]
    elif family == "grid":
        grid = [list(row) for row in value["grid"]]
        if rule == "grid_transpose":
            return [[grid[row][column] for row in range(len(grid))] for column in range(len(grid[0]))]
        if rule == "grid_reflect":
            return [list(reversed(row)) for row in grid]
    elif family == "graph":
        edges = {tuple(edge) for edge in value["edges"]}
        start = value["start"]
        goal = value["goal"]
        if rule == "graph_direct":
            return (start, goal) in edges
        if rule == "graph_reach":
            seen = {start}
            frontier = [start]
            while frontier:
                node = frontier.pop(0)
                for source, target in edges:
                    if source == node and target not in seen:
                        seen.add(target)
                        frontier.append(target)
            return goal in seen
    elif family == "set":
        left = set(value["left"])
        right = set(value["right"])
        if rule == "set_union":
            return sorted(left | right)
        if rule == "set_intersection":
            return sorted(left & right)
    raise ValueError(f"unknown rule/family: {rule}/{family}")


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    variant: str
    condition: str
    phase: str
    surface_group: str
    structural_signature: str
    environment_version: int
    true_rule: str
    candidate_rules: Tuple[str, ...]
    demonstrations: Tuple[Tuple[Any, Any], ...]
    probe_input: Any
    probe_output: Any
    target_input: Any
    target_output: Any
    probe_available: bool
    verifier_available: bool


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    family: str
    variant: str
    surface_group: str
    structural_signature: str
    environment_version: int
    rule: str
    conditions: Mapping[str, Any]
    evidence: Mapping[str, Any]
    provenance: Mapping[str, Any]
    validation_status: str
    confidence: float


@dataclass
class MemoryStore:
    records: List[MemoryRecord] = field(default_factory=list)
    writes: int = 0
    contradiction_records: int = 0
    provisional_records: int = 0
    failed_records: int = 0

    def add(self, record: MemoryRecord) -> bool:
        self.records.append(record)
        if record.validation_status == "validated":
            self.writes += 1
            return True
        if record.validation_status == "contradicted":
            self.contradiction_records += 1
        elif record.validation_status == "provisional":
            self.provisional_records += 1
        elif record.validation_status == "failed":
            self.failed_records += 1
        return False

    def safe_exact(self, task: Task) -> Optional[MemoryRecord]:
        candidates = [
            record
            for record in self.records
            if record.validation_status == "validated"
            and record.family == task.family
            and record.structural_signature == task.structural_signature
            and record.environment_version == task.environment_version
        ]
        return candidates[-1] if candidates else None

    def safe_related(self, task: Task) -> Optional[MemoryRecord]:
        candidates = [
            record
            for record in self.records
            if record.validation_status == "validated"
            and record.family == task.family
            and record.structural_signature == task.structural_signature
        ]
        return candidates[-1] if candidates else None

    def persistent_variant(self, task: Task) -> Optional[MemoryRecord]:
        candidates = [
            record
            for record in self.records
            if record.validation_status == "validated"
            and record.family == task.family
            and record.variant == task.variant
        ]
        return candidates[-1] if candidates else None

    def seed_shuffled(self, seed: int) -> None:
        for family in FAMILIES:
            wrong_rule = RULES[family][(stable_int("shuffled", seed, family) + 1) % len(RULES[family])]
            self.add(MemoryRecord(
                record_id=f"shuffled-{family}",
                family=family,
                variant="source",
                surface_group="red",
                structural_signature="base",
                environment_version=0,
                rule=wrong_rule,
                conditions={"family": family, "structural_signature": "base", "environment_version": 0},
                evidence={"type": "shuffled_memory_control"},
                provenance={"source": "shuffled_memory_control"},
                validation_status="validated",
                confidence=1.0,
            ))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "records": [record.__dict__ for record in self.records],
            "writes": self.writes,
            "contradiction_records": self.contradiction_records,
            "provisional_records": self.provisional_records,
            "failed_records": self.failed_records,
        }


def task_inputs(family: str, base: int, holdout: bool, ambiguous: bool) -> Tuple[Any, Any]:
    if family == "vector":
        if ambiguous:
            return {"values": [base]}, {"values": [base, base + 4]}
        length = 4 if holdout else 3
        probe = {"values": [base + index for index in range(length)]}
        target = {"values": [base + 2, base + 5, base + 8, base + 11, base + 14][: length + 1]}
        return probe, {"values": target["values"]}
    if family == "grid":
        if ambiguous:
            return {"grid": [[base]]}, {"grid": [[base, base + 1], [base + 2, base + 3]]}
        if holdout:
            probe_grid = [[base, base + 1, base + 2], [base + 3, base + 4, base + 5]]
            target_grid = [[base + 6, base + 7, base + 8], [base + 9, base + 10, base + 11]]
        else:
            probe_grid = [[base, base + 1], [base + 2, base + 3]]
            target_grid = [[base + 4, base + 5], [base + 6, base + 7]]
        return {"grid": probe_grid}, {"grid": target_grid}
    if family == "graph":
        if ambiguous:
            return {"edges": [[0, 1]], "start": 0, "goal": 1}, {"edges": [[0, 1], [1, 2]], "start": 0, "goal": 2}
        if holdout:
            probe_edges = [[0, 1], [1, 2], [2, 3]]
            target_edges = [[0, 1], [1, 2], [2, 3], [3, 4]]
        else:
            probe_edges = [[0, 1], [1, 2]]
            target_edges = [[0, 1], [1, 2], [2, 3]]
        return {"edges": probe_edges, "start": 0, "goal": probe_edges[-1][1]}, {"edges": target_edges, "start": 0, "goal": target_edges[-1][1]}
    if ambiguous:
        return {"left": [base], "right": [base]}, {"left": [base, base + 1], "right": [base + 1]}
    offset = 1 if holdout else 0
    return {"left": [base, base + 1 + offset], "right": [base + 1 + offset, base + 2]}, {"left": [base, base + 2, base + 4], "right": [base + 2, base + 5]}


def make_task(seed: int, family: str, condition: str, episode: int, phase: str = "evaluation") -> Task:
    rules = RULES[family]
    stable_rule = rules[stable_int("c1h-rule", seed, family) % len(rules)]
    changed_rule = rules[1 - rules.index(stable_rule)]
    if phase == "warmup":
        true_rule = stable_rule
        variant = "source"
        structural_signature = "base"
        environment_version = 0
    else:
        true_rule = stable_rule if condition in {"novel_surface", "novel_structure", "adaptation", "unresolvable"} else changed_rule
        variant = "source" if condition in {"invalidated", "contradictory", "unresolvable"} else "holdout"
        structural_signature = (
            "base" if condition in {"novel_surface", "adaptation", "invalidated", "contradictory"}
            else "holdout" if condition == "novel_structure"
            else "decoy" if condition == "misleading_similarity"
            else "unknown"
        )
        environment_version = 1 if condition in {"adaptation", "invalidated"} else 2 if condition == "unresolvable" else 0
    surface_group = "blue" if condition == "unresolvable" else "red"
    base = stable_int("c1h-surface", seed, family, condition, episode, phase) % 20 + 1
    holdout = structural_signature == "holdout" or variant == "holdout"
    ambiguous = condition == "unresolvable"
    probe_input, target_input = task_inputs(family, base, holdout, ambiguous)
    demo_input, second_demo = task_inputs(family, base + 2, holdout, False)
    demonstrations = (
        (demo_input, solve(true_rule, family, demo_input)),
        (second_demo, solve(true_rule, family, second_demo)),
    )
    probe_output = solve(true_rule, family, probe_input)
    return Task(
        task_id=f"{family}-{condition}-{episode}-{phase}",
        family=family,
        variant=variant,
        condition=condition,
        phase=phase,
        surface_group=surface_group,
        structural_signature=structural_signature,
        environment_version=environment_version,
        true_rule=true_rule,
        candidate_rules=rules,
        demonstrations=demonstrations,
        probe_input=probe_input,
        probe_output=probe_output,
        target_input=target_input,
        target_output=solve(true_rule, family, target_input),
        probe_available=not ambiguous,
        verifier_available=not ambiguous,
    )


def make_tasks(seed: int) -> List[Task]:
    tasks = [make_task(seed, family, "warmup", index, phase="warmup") for index, family in enumerate(FAMILIES)]
    for family in FAMILIES:
        for condition in CONDITIONS:
            for episode in range(EPISODES_PER_CONDITION):
                tasks.append(make_task(seed, family, condition, episode))
    return tasks


def verify(task: Task, rule: str) -> Optional[bool]:
    if not task.verifier_available:
        return None
    return solve(rule, task.family, task.target_input) == task.target_output


def matches_probe(task: Task, rule: str) -> bool:
    return solve(rule, task.family, task.probe_input) == task.probe_output


def record_for(task: Task, rule: str, status: str, source: str, evidence: Mapping[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        record_id=f"{source}-{task.task_id}-{len(evidence)}",
        family=task.family,
        variant=task.variant,
        surface_group=task.surface_group,
        structural_signature=task.structural_signature,
        environment_version=task.environment_version,
        rule=rule,
        conditions={"family": task.family, "structural_signature": task.structural_signature, "environment_version": task.environment_version},
        evidence=dict(evidence),
        provenance={"source": source, "task_id": task.task_id},
        validation_status=status,
        confidence=1.0 if status == "validated" else 0.0,
    )


def investigate(task: Task, store: MemoryStore, write_enabled: bool, counters: Dict[str, int]) -> Tuple[Optional[str], float, str]:
    if not task.probe_available:
        counters["abstentions"] += 1
        counters["safe_abstentions"] += 1
        store.add(record_for(task, task.candidate_rules[0], "provisional", "unresolvable", {"reason": "no_discriminating_probe"}))
        return None, 0.0, "abstain"
    counters["probes"] += 1
    cost = PROBE_COST
    matches = [rule for rule in task.candidate_rules if matches_probe(task, rule)]
    candidates = matches if len(matches) == 1 else list(task.candidate_rules)
    for candidate in candidates:
        checked = verify(task, candidate)
        if checked is not None:
            counters["verifications"] += 1
            cost += VERIFY_COST
        if checked is True:
            if write_enabled:
                store.add(record_for(task, candidate, "validated", "verified_investigation", {"probe_output": task.probe_output}))
            return candidate, cost, "investigate"
    store.add(record_for(task, task.candidate_rules[0], "provisional", "ambiguous_investigation", {"probe_output": task.probe_output}))
    counters["abstentions"] += 1
    counters["false_abstentions"] += 1
    return None, cost, "abstain"


def run_arm(tasks: Sequence[Task], arm: str, seed: int) -> Dict[str, Any]:
    store = MemoryStore()
    if arm == "shuffled_memory":
        store.seed_shuffled(seed)
    outcomes: List[Dict[str, Any]] = []
    counters = defaultdict(int)
    for task in tasks:
        if arm == "oracle":
            selected = task.true_rule
            decision = "oracle"
            cost = 0.0
            memory_source = "none"
        elif arm == "persistent":
            record = store.persistent_variant(task)
            if record is not None:
                selected = record.rule
                decision = "reuse"
                cost = 0.0
                memory_source = "persistent_memory"
                counters["retrievals"] += 1
            else:
                selected, cost, decision = investigate(task, store, True, counters)
                memory_source = "none"
        elif arm in {"safe_adaptive", "shuffled_memory"}:
            exact = store.safe_exact(task)
            related = None if exact is not None else store.safe_related(task)
            record = exact or related
            memory_source = "exact" if exact is not None else "related" if related is not None else "none"
            if record is None:
                selected, cost, decision = investigate(task, store, True, counters)
            else:
                counters["retrievals"] += 1
                if matches_probe(task, record.rule):
                    if memory_source == "related":
                        checked = verify(task, record.rule)
                        cost = VERIFY_COST if checked is not None else 0.0
                        counters["verifications"] += int(checked is not None)
                        if checked is True:
                            counters["adaptation_attempts"] += 1
                            counters["adaptation_successes"] += 1
                            selected = record.rule
                            decision = "adapt"
                            store.add(record_for(task, selected, "validated", "validated_adaptation", {"from": record.record_id}))
                        else:
                            selected, extra_cost, decision = investigate(task, store, True, counters)
                            cost += extra_cost
                    else:
                        selected = record.rule
                        decision = "reuse"
                        cost = 0.0
                else:
                    store.add(record_for(task, record.rule, "contradicted", "contradiction_detected", {"retrieved": record.record_id, "current_probe": task.probe_output}))
                    selected, cost, decision = investigate(task, store, True, counters)
        elif arm in {"memory_disabled", "stateless"}:
            selected, cost, decision = investigate(task, store, False, counters)
            memory_source = "none"
        else:
            raise ValueError(arm)
        correct = selected is not None and solve(selected, task.family, task.target_input) == task.target_output
        if selected is not None and memory_source != "none" and decision in {"reuse", "adapt"} and not correct:
            counters["harmful_transfer"] += 1
        if task.phase != "warmup":
            counters["eval_tasks"] += 1
            counters["eval_correct"] += int(correct)
            counters["eval_cost"] += cost
            if task.condition in APPLICABLE_CONDITIONS:
                counters["applicable_tasks"] += 1
                counters["reuse_or_adapt"] += int(decision in {"reuse", "adapt"})
            if task.condition == "novel_structure":
                counters["novel_structure_tasks"] += 1
                counters["novel_structure_correct"] += int(correct)
            if decision == "abstain":
                counters["eval_abstentions"] += 1
                counters["safe_eval_abstentions"] += int(task.condition == "unresolvable")
                counters["false_eval_abstentions"] += int(task.condition != "unresolvable")
        outcomes.append({
            "task_id": task.task_id,
            "family": task.family,
            "variant": task.variant,
            "condition": task.condition,
            "phase": task.phase,
            "true_rule": task.true_rule,
            "selected_rule": selected,
            "decision": decision,
            "memory_source": memory_source,
            "correct": correct,
            "cost": cost,
        })
    eval_tasks = counters["eval_tasks"]
    return {
        "arm": arm,
        "seed": seed,
        "tasks": eval_tasks,
        "accuracy": counters["eval_correct"] / max(1, eval_tasks),
        "mean_cost": counters["eval_cost"] / max(1, eval_tasks),
        "novel_structure_accuracy": counters["novel_structure_correct"] / max(1, counters["novel_structure_tasks"]),
        "reuse_coverage": counters["reuse_or_adapt"] / max(1, counters["applicable_tasks"]),
        "adaptation_attempts": counters["adaptation_attempts"],
        "adaptation_successes": counters["adaptation_successes"],
        "adaptation_success_rate": counters["adaptation_successes"] / max(1, counters["adaptation_attempts"]),
        "retrievals": counters["retrievals"],
        "harmful_transfer_rate": counters["harmful_transfer"] / max(1, counters["retrievals"]),
        "abstentions": counters["eval_abstentions"],
        "abstention_precision": counters["safe_eval_abstentions"] / max(1, counters["eval_abstentions"]),
        "false_abstentions": counters["false_eval_abstentions"],
        "probes": counters["probes"],
        "verifications": counters["verifications"],
        "validated_writes": store.writes,
        "contradiction_records": store.contradiction_records,
        "provisional_records": store.provisional_records,
        "failed_records": store.failed_records,
        "final_memory_size": sum(record.validation_status == "validated" for record in store.records),
        "memory": store.snapshot(),
        "outcomes": outcomes,
    }


def aggregate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    numeric = ("accuracy", "mean_cost", "novel_structure_accuracy", "reuse_coverage", "adaptation_success_rate", "harmful_transfer_rate", "abstention_precision", "final_memory_size")
    counts = ("tasks", "adaptation_attempts", "adaptation_successes", "retrievals", "abstentions", "false_abstentions", "probes", "verifications", "validated_writes", "contradiction_records", "provisional_records", "failed_records")
    return {
        "runs": len(records),
        **{field: sum(float(record.get(field, 0.0)) for record in records) / len(records) for field in numeric},
        **{field: sum(int(record.get(field, 0)) for record in records) for field in counts},
    }


def source_hashes() -> Dict[str, str]:
    paths = [
        PROTOCOL,
        ROOT / "tools" / "run_crucible_c1f_persistent_reuse_20261021.py",
        ROOT / "tools" / "run_crucible_c1g_safe_reuse_20261021.py",
        ROOT / "tools" / "run_crucible_c1h_heldout_generalization_20261021.py",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def render_summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1h held-out structural generalization",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; no GPU, cloud or Crucible neural training.",
        "",
        "## Aggregate results",
        "",
        "| Arm | Accuracy | Novel-structure accuracy | Reuse coverage | Adaptation success | Harmful transfer | Abstention precision | False abstentions | Mean cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, item in result["aggregate"].items():
        lines.append(f"| `{arm}` | {item['accuracy']:.4f} | {item['novel_structure_accuracy']:.4f} | {item['reuse_coverage']:.4f} | {item['adaptation_success_rate']:.4f} | {item['harmful_transfer_rate']:.4f} | {item['abstention_precision']:.4f} | {item['false_abstentions']} | {item['mean_cost']:.3f} |")
    lines.extend(["", "## Condition results", ""])
    for condition, arms in result["by_condition"].items():
        lines.extend([f"### {condition}", "", "| Arm | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for arm, item in arms.items():
            lines.append(f"| `{arm}` | {item['accuracy']:.4f} | {item['reuse_coverage']:.4f} | {item['harmful_transfer_rate']:.4f} | {item['abstention_precision']:.4f} | {item['mean_cost']:.3f} |")
        lines.append("")
    lines.extend(["## Memory transactions", "", "| Arm | Validated writes | Contradiction records | Provisional records | Failed records |", "| --- | ---: | ---: | ---: | ---: |"])
    for arm, item in result["aggregate"].items():
        lines.append(f"| `{arm}` | {item['validated_writes']} | {item['contradiction_records']} | {item['provisional_records']} | {item['failed_records']} |")
    lines.extend([
        "",
        "## Preregistered gates",
        "",
        f"- Primary safe-generalization gate: `{result['gates']['primary_gate']}`.",
        f"- Novel-structure gate: `{result['gates']['novel_structure_gate']}`.",
        f"- Adaptation gate: `{result['gates']['adaptation_gate']}`.",
        f"- Shuffled-memory harmful-transfer gate: `{result['gates']['shuffled_memory_gate']}`.",
        f"- Contradiction-preservation gate: `{result['gates']['contradiction_preservation']}`.",
        "- False abstentions are reported separately; abstention alone is not treated as success.",
        "",
        "## Interpretation",
        "",
        "A passing result transfers the generic controller contract to these synthetic held-out families. It does not demonstrate learned applicability, neural reasoning or CMS2-assisted training.",
        "",
        "## Source hashes",
        "",
    ])
    for path, digest in sorted(hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    arms = ("persistent", "safe_adaptive", "memory_disabled", "shuffled_memory", "stateless", "oracle")
    full_runs: List[Dict[str, Any]] = []
    summary_runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    condition_runs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {condition: defaultdict(list) for condition in CONDITIONS}
    for seed in SEEDS:
        tasks = make_tasks(seed)
        for arm in arms:
            run = run_arm(tasks, arm, seed)
            summary = {key: value for key, value in run.items() if key not in {"outcomes", "memory"}}
            full_runs.append({"seed": seed, "arm": arm, "summary": summary, "memory": run["memory"], "outcomes": run["outcomes"]})
            summary_runs[arm].append(summary)
            for condition in CONDITIONS:
                condition_outcomes = [outcome for outcome in run["outcomes"] if outcome["phase"] != "warmup" and outcome["condition"] == condition]
                memory_items = [item for item in condition_outcomes if item["decision"] in {"reuse", "adapt"}]
                condition_runs[condition][arm].append({
                    "tasks": len(condition_outcomes),
                    "accuracy": sum(item["correct"] for item in condition_outcomes) / max(1, len(condition_outcomes)),
                    "reuse_coverage": sum(item["decision"] in {"reuse", "adapt"} for item in condition_outcomes) / max(1, len(condition_outcomes)),
                    "harmful_transfer_rate": sum(item["decision"] in {"reuse", "adapt"} and not item["correct"] for item in condition_outcomes) / max(1, len(memory_items)),
                    "abstention_precision": sum(item["decision"] == "abstain" and condition == "unresolvable" for item in condition_outcomes) / max(1, sum(item["decision"] == "abstain" for item in condition_outcomes)),
                    "mean_cost": sum(float(item["cost"]) for item in condition_outcomes) / max(1, len(condition_outcomes)),
                })
    aggregate_results = {arm: aggregate(records) for arm, records in sorted(summary_runs.items())}
    aggregate_conditions = {condition: {arm: aggregate(records) for arm, records in sorted(arms_for_condition.items())} for condition, arms_for_condition in condition_runs.items()}
    safe = aggregate_results["safe_adaptive"]
    persistent = aggregate_results["persistent"]
    disabled = aggregate_results["memory_disabled"]
    harmful_conditions = ("invalidated", "contradictory", "misleading_similarity")
    safe_harm = sum(aggregate_conditions[condition]["safe_adaptive"]["harmful_transfer_rate"] for condition in harmful_conditions) / len(harmful_conditions)
    persistent_harm = sum(aggregate_conditions[condition]["persistent"]["harmful_transfer_rate"] for condition in harmful_conditions) / len(harmful_conditions)
    safe_novel_structure = aggregate_conditions["novel_structure"]["safe_adaptive"]["accuracy"]
    disabled_novel_structure = aggregate_conditions["novel_structure"]["memory_disabled"]["accuracy"]
    primary_gate = (
        safe["accuracy"] >= persistent["accuracy"]
        and safe_harm <= persistent_harm * 0.5
        and safe["reuse_coverage"] >= 0.60
        and safe["abstention_precision"] >= 0.80
    )
    novel_structure_gate = safe_novel_structure >= disabled_novel_structure
    adaptation_gate = safe["adaptation_attempts"] > 0 and safe["adaptation_successes"] == safe["adaptation_attempts"]
    shuffled_gate = aggregate_results["shuffled_memory"]["harmful_transfer_rate"] <= 0.10
    contradiction_preservation = all(run["memory"]["contradiction_records"] > 0 for run in full_runs if run["arm"] == "safe_adaptive")
    result = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "aggregate": aggregate_results,
        "by_condition": aggregate_conditions,
        "gates": {
            "primary_gate": primary_gate,
            "novel_structure_gate": novel_structure_gate,
            "adaptation_gate": adaptation_gate,
            "shuffled_memory_gate": shuffled_gate,
            "contradiction_preservation": contradiction_preservation,
            "persistent_harmful_transfer": persistent_harm,
            "safe_harmful_transfer": safe_harm,
            "safe_novel_structure_accuracy": safe_novel_structure,
            "memory_disabled_novel_structure_accuracy": disabled_novel_structure,
        },
        "seeds": list(SEEDS),
        "families": list(FAMILIES),
        "conditions": list(CONDITIONS),
        "episodes_per_condition": EPISODES_PER_CONDITION,
        "arms": list(arms),
    }
    write_json(OUTPUT / "RESULTS.json", result)
    with (OUTPUT / "RUNS.jsonl").open("w", encoding="utf-8") as handle:
        for run in full_runs:
            handle.write(json.dumps(run, sort_keys=True) + "\n")
    write_json(OUTPUT / "RUN_CONFIG.json", {
        "experiment": EXPERIMENT,
        "seeds": list(SEEDS),
        "families": list(FAMILIES),
        "conditions": list(CONDITIONS),
        "episodes_per_condition": EPISODES_PER_CONDITION,
        "probe_cost": PROBE_COST,
        "verify_cost": VERIFY_COST,
        "execution": "CPU-only",
    })
    hashes = source_hashes()
    (OUTPUT / "RESULTS_SUMMARY.md").write_text(render_summary(result, hashes), encoding="utf-8")
    file_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path.name != "HASHES.json"}
    write_json(OUTPUT / "HASHES.json", file_hashes)
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "runs": len(SEEDS) * len(arms), "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
