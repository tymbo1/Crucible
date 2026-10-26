#!/usr/bin/env python3
"""CPU-only C1g evidence-grounded memory reuse experiment."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_c1g_safe_reuse_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
EXPERIMENT = "crucible_c1g_safe_reuse_20261021"
SEEDS = (42, 1337, 2026)
FAMILIES = ("arithmetic", "sequence", "logic")
VARIANTS = ("compact", "extended")
CONDITIONS = ("applicable", "misleading_similarity", "changed_environment", "contradictory", "unresolvable")
EPISODES_PER_CONDITION = 8
PROBE_COST = 0.8
VERIFY_COST = 0.5

sys.path.insert(0, str(ROOT))
from tools.run_crucible_c1f_persistent_reuse_20261021 import RULES, solve, stable_int  # noqa: E402


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        candidates = [record for record in self.records if record.validation_status == "validated" and record.family == task.family and record.structural_signature == task.structural_signature and record.environment_version == task.environment_version]
        return candidates[-1] if candidates else None

    def safe_related(self, task: Task) -> Optional[MemoryRecord]:
        candidates = [record for record in self.records if record.validation_status == "validated" and record.family == task.family and record.structural_signature == task.structural_signature]
        return candidates[-1] if candidates else None

    def broad_surface(self, task: Task) -> Optional[MemoryRecord]:
        candidates = [record for record in self.records if record.validation_status == "validated" and record.family == task.family and record.surface_group == task.surface_group]
        return candidates[-1] if candidates else None

    def persistent_variant(self, task: Task) -> Optional[MemoryRecord]:
        """C1f-style persistence: reuse by family/variant, ignoring context."""
        candidates = [record for record in self.records if record.validation_status == "validated" and record.family == task.family and record.variant == task.variant]
        return candidates[-1] if candidates else None

    def seed_shuffled(self, seed: int) -> None:
        for family in FAMILIES:
            rules = RULES[family]
            for variant_index, variant in enumerate(VARIANTS):
                wrong_rule = rules[(variant_index + seed + 1) % len(rules)]
                self.add(MemoryRecord(
                    record_id=f"shuffled-{family}-{variant}",
                    family=family,
                    variant=variant,
                    surface_group="red",
                    structural_signature="base",
                    environment_version=0,
                    rule=wrong_rule,
                    conditions={"family": family, "structural_signature": "base", "environment_version": 0},
                    evidence={"type": "shuffled_control"},
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


def make_task(seed: int, family: str, variant: str, condition: str, episode: int, phase: str = "evaluation") -> Task:
    rules = RULES[family]
    stable_rule = rules[stable_int("c1g-rule", seed, family) % len(rules)]
    changed_rule = rules[1 - rules.index(stable_rule)]
    true_rule = stable_rule if condition in {"applicable", "unresolvable"} else changed_rule
    if phase == "warmup":
        true_rule = stable_rule
    surface_group = "blue" if condition == "unresolvable" else "red"
    structural_signature = "base" if condition in {"applicable", "changed_environment", "contradictory"} or phase == "warmup" else "alternate" if condition == "misleading_similarity" else "unknown"
    # Extended applicable tasks preserve the underlying rule but arrive under
    # a changed observable condition. They therefore exercise validated
    # adaptation rather than exact replay. Changed-environment tasks use the
    # same version number but a different rule, so the applicability gate must
    # reject the retrieved record and investigate.
    environment_version = 1 if condition == "changed_environment" or (condition == "applicable" and variant == "extended") else 2 if condition == "unresolvable" else 0
    base = stable_int("c1g-surface", seed, family, variant, condition, episode) % 20
    if family == "arithmetic":
        demonstrations = (
            ({"start": 2 + base, "operations": (("add", 2), ("sub", 1))}, None),
            ({"start": 4 + base, "operations": (("sub", 1), ("add", 3))}, None),
        )
        probe_input = {"start": 2 + base, "operations": (("add", 3), ("mul", 2), ("sub", 1))}
        target_input = {"start": 3 + base, "operations": (("add", 1), ("mul", 3), ("sub", 2))}
    elif family == "sequence":
        demonstrations = (([1 + base, 2 + base, 2 + base], None), ([3 + base, 4 + base, 4 + base], None))
        probe_input = [1 + base, 2 + base, 3 + base]
        target_input = [4 + base, 7 + base, 9 + base, 10 + base]
    else:
        demonstrations = (([True], None), ([False], None))
        probe_input = [True, False]
        target_input = [True, True, False]
    demonstrations = tuple((value, solve(true_rule, family, value)) for value, _ in demonstrations)
    probe_output = solve(true_rule, family, demonstrations[0][0]) if condition == "unresolvable" else solve(true_rule, family, probe_input)
    return Task(
        task_id=f"{family}-{variant}-{condition}-{episode}-{phase}",
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
        probe_available=condition != "unresolvable",
        verifier_available=condition != "unresolvable",
    )


def make_tasks(seed: int) -> List[Task]:
    tasks: List[Task] = []
    for family in FAMILIES:
        for variant in VARIANTS:
            tasks.append(make_task(seed, family, variant, "applicable", 0, phase="warmup"))
    for family in FAMILIES:
        for variant in VARIANTS:
            for condition in CONDITIONS:
                for episode in range(EPISODES_PER_CONDITION):
                    tasks.append(make_task(seed, family, variant, condition, episode))
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
    if len(matches) == 1:
        candidate = matches[0]
        checked = verify(task, candidate)
        if checked is True:
            counters["verifications"] += 1
            cost += VERIFY_COST
            if write_enabled:
                store.add(record_for(task, candidate, "validated", "verified_investigation", {"probe_output": task.probe_output}))
            return candidate, cost, "investigate"
        if checked is False:
            counters["verifications"] += 1
            store.add(record_for(task, candidate, "failed", "failed_investigation", {"probe_output": task.probe_output}))
            counters["abstentions"] += 1
            counters["false_abstentions"] += 1
            return None, cost + VERIFY_COST, "abstain"
    for candidate in task.candidate_rules:
        checked = verify(task, candidate)
        if checked is not None:
            counters["verifications"] += 1
            cost += VERIFY_COST
        if checked is True:
            if write_enabled:
                store.add(record_for(task, candidate, "validated", "verified_fallback", {"probe_output": task.probe_output}))
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
                memory_source = "surface_memory"
                counters["retrievals"] += 1
            else:
                selected, cost, decision = investigate(task, store, True, counters)
                memory_source = "none"
                if selected is not None:
                    store.add(record_for(task, selected, "validated", "persistent_surface", {"source_decision": decision}))
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
            if task.condition == "applicable":
                counters["applicable_tasks"] += 1
                counters["reuse_or_adapt"] += int(decision in {"reuse", "adapt"})
            if decision == "abstain":
                counters["eval_abstentions"] += 1
                counters["safe_eval_abstentions"] += int(task.condition == "unresolvable")
                counters["false_eval_abstentions"] += int(task.condition != "unresolvable")
            counters["eval_cost"] += cost
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
        "memory": store.snapshot(),
        "final_memory_size": sum(record.validation_status == "validated" for record in store.records),
        "outcomes": outcomes,
    }


def aggregate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    numeric = ("accuracy", "mean_cost", "reuse_coverage", "adaptation_success_rate", "harmful_transfer_rate", "abstention_precision", "final_memory_size")
    counts = ("tasks", "adaptation_attempts", "adaptation_successes", "retrievals", "abstentions", "false_abstentions", "probes", "verifications", "validated_writes", "contradiction_records", "provisional_records", "failed_records")
    # Per-condition summaries intentionally contain only condition-relevant
    # fields. Missing counters are zero rather than a schema failure; the
    # top-level arm summaries still emit the complete run-level schema.
    return {
        "runs": len(records),
        **{field: sum(float(record.get(field, 0.0)) for record in records) / len(records) for field in numeric},
        **{field: sum(int(record.get(field, 0)) for record in records) for field in counts},
    }


def source_hashes() -> Dict[str, str]:
    c1f_script = ROOT / "tools" / "run_crucible_c1f_persistent_reuse_20261021.py"
    paths = [PROTOCOL, ROOT / "tools" / "run_crucible_c1g_safe_reuse_20261021.py", c1f_script]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def render_summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1g evidence-grounded adaptive reuse",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; no GPU, cloud or Crucible neural training.",
        "",
        "## Aggregate results",
        "",
        "| Arm | Accuracy | Reuse coverage | Adaptation success | Harmful transfer | Abstention precision | False abstentions | Mean cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, item in result["aggregate"].items():
        lines.append(f"| `{arm}` | {item['accuracy']:.4f} | {item['reuse_coverage']:.4f} | {item['adaptation_success_rate']:.4f} | {item['harmful_transfer_rate']:.4f} | {item['abstention_precision']:.4f} | {item['false_abstentions']} | {item['mean_cost']:.3f} |")
    lines.extend(["", "## Condition results", ""])
    for condition, arms in result["by_condition"].items():
        lines.extend([f"### {condition}", "", "| Arm | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for arm, item in arms.items():
            lines.append(f"| `{arm}` | {item['accuracy']:.4f} | {item['reuse_coverage']:.4f} | {item['harmful_transfer_rate']:.4f} | {item['abstention_precision']:.4f} | {item['mean_cost']:.3f} |")
        lines.append("")
    lines.extend(["## Memory transactions", "", "| Arm | Validated writes | Contradiction records | Provisional records | Failed records |", "| --- | ---: | ---: | ---: | ---: |"])
    for arm, item in result["aggregate"].items():
        lines.append(f"| `{arm}` | {item['validated_writes']} | {item['contradiction_records']} | {item['provisional_records']} | {item['failed_records']} |")
    lines.append("")
    lines.extend([
        "## Preregistered gates",
        "",
        f"- Primary safe-reuse gate: `{result['gates']['primary_gate']}`.",
        f"- Shuffled-memory harmful-transfer gate: `{result['gates']['shuffled_memory_gate']}`.",
        f"- Contradiction preservation: `{result['gates']['contradiction_preservation']}`.",
        "- False abstentions are reported separately; abstention alone is not treated as success.",
        "",
        "## Interpretation",
        "",
        "The persistent arm is the primary C1f baseline. Safe adaptive reuse is only accepted if it improves applicability decisions without increasing unsafe transfer. A failed gate is preserved as an experimental result.",
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
            full_runs.append({"seed": seed, "arm": arm, "summary": {key: value for key, value in run.items() if key not in {"outcomes", "memory"}}, "memory": run["memory"], "outcomes": run["outcomes"]})
            summary_runs[arm].append({key: value for key, value in run.items() if key not in {"outcomes", "memory"}})
            for condition in CONDITIONS:
                condition_outcomes = [outcome for outcome in run["outcomes"] if outcome["phase"] != "warmup" and outcome["condition"] == condition]
                correct = sum(item["correct"] for item in condition_outcomes)
                memory_items = [item for item in condition_outcomes if item["decision"] in {"reuse", "adapt"}]
                condition_runs[condition][arm].append({
                    "tasks": len(condition_outcomes),
                    "accuracy": correct / max(1, len(condition_outcomes)),
                    "reuse_coverage": sum(item["decision"] in {"reuse", "adapt"} for item in condition_outcomes) / max(1, len(condition_outcomes)),
                    "harmful_transfer_rate": sum(item["decision"] in {"reuse", "adapt"} and not item["correct"] for item in condition_outcomes) / max(1, len(memory_items)),
                    "abstention_precision": sum(item["decision"] == "abstain" and condition == "unresolvable" for item in condition_outcomes) / max(1, sum(item["decision"] == "abstain" for item in condition_outcomes)),
                    "mean_cost": sum(float(item["cost"]) for item in condition_outcomes) / max(1, len(condition_outcomes)),
                })
    aggregate_results = {arm: aggregate(records) for arm, records in sorted(summary_runs.items())}
    aggregate_conditions = {condition: {arm: aggregate(records) for arm, records in sorted(arms_for_condition.items())} for condition, arms_for_condition in condition_runs.items()}
    safe = aggregate_results["safe_adaptive"]
    persistent = aggregate_results["persistent"]
    changed_conditions = ("misleading_similarity", "changed_environment", "contradictory")
    safe_harm = sum(aggregate_conditions[condition]["safe_adaptive"]["harmful_transfer_rate"] for condition in changed_conditions) / len(changed_conditions)
    persistent_harm = sum(aggregate_conditions[condition]["persistent"]["harmful_transfer_rate"] for condition in changed_conditions) / len(changed_conditions)
    primary_gate = (
        safe["accuracy"] >= persistent["accuracy"]
        and safe_harm <= persistent_harm * 0.5
        and safe["reuse_coverage"] >= 0.60
        and safe["abstention_precision"] >= 0.80
    )
    shuffled_gate = aggregate_results["shuffled_memory"]["harmful_transfer_rate"] <= 0.10
    contradiction_preservation = all(run["memory"]["contradiction_records"] > 0 for run in full_runs if run["arm"] == "safe_adaptive")
    result = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "aggregate": aggregate_results,
        "by_condition": aggregate_conditions,
        "gates": {"primary_gate": primary_gate, "shuffled_memory_gate": shuffled_gate, "contradiction_preservation": contradiction_preservation, "safe_changed_harmful_transfer": safe_harm, "persistent_changed_harmful_transfer": persistent_harm},
        "seeds": list(SEEDS),
        "arms": list(arms),
        "conditions": list(CONDITIONS),
        "episodes_per_condition": EPISODES_PER_CONDITION,
    }
    write_json(OUTPUT / "RESULTS.json", result)
    write_json(OUTPUT / "RUN_CONFIG.json", {"experiment": EXPERIMENT, "seeds": list(SEEDS), "families": list(FAMILIES), "variants": list(VARIANTS), "conditions": list(CONDITIONS), "episodes_per_condition": EPISODES_PER_CONDITION, "probe_cost": PROBE_COST, "verify_cost": VERIFY_COST, "arms": list(arms), "execution": "CPU-only"})
    with (OUTPUT / "RUNS.jsonl").open("w", encoding="utf-8") as handle:
        for run in full_runs:
            handle.write(json.dumps(run, sort_keys=True) + "\n")
    hashes = source_hashes()
    (OUTPUT / "RESULTS_SUMMARY.md").write_text(render_summary(result, hashes), encoding="utf-8")
    file_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path.name != "HASHES.json"}
    write_json(OUTPUT / "HASHES.json", file_hashes)
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "runs": len(full_runs), "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
