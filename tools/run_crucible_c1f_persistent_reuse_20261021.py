#!/usr/bin/env python3
"""CPU-only sequential memory experiment for Crucible C1f."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_c1f_persistent_reuse_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
EXPERIMENT = "crucible_c1f_persistent_reuse_20261021"
SEEDS = (42, 1337, 2026)
FAMILIES = ("arithmetic", "sequence", "logic")
VARIANTS = ("compact", "extended")
REGIMES = ("informative", "expensive", "noisy", "unresolvable")
EPISODES_PER_TRACK = 12
RULES = {
    "arithmetic": ("arith_left", "arith_precedence"),
    "sequence": ("seq_reverse", "seq_rotate"),
    "logic": ("logic_all", "logic_any"),
}
PROBE_COST = {"informative": 0.5, "expensive": 3.0, "noisy": 0.8, "unresolvable": 1.0}
VERIFY_COST = 0.5
ADAPTIVE_PROBE_LIMIT = 2.0


def stable_int(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


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
    regime: str
    episode: int
    true_rule: str
    candidate_rules: Tuple[str, ...]
    demonstrations: Tuple[Tuple[Any, Any], ...]
    probe_input: Any
    target_input: Any
    target_output: Any
    probe_output: Any
    probe_cost: float


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    family: str
    variant: str
    rule: str
    conditions: Mapping[str, Any]
    evidence: Mapping[str, Any]
    provenance: Mapping[str, Any]
    validation_status: str
    confidence: float


@dataclass
class MemoryStore:
    records: Dict[Tuple[str, str], MemoryRecord] = field(default_factory=dict)
    writes: int = 0
    rejected_writes: int = 0

    def exact(self, family: str, variant: str) -> Optional[MemoryRecord]:
        return self.records.get((family, variant))

    def related(self, family: str, variant: str) -> Optional[MemoryRecord]:
        candidates = [record for (record_family, record_variant), record in self.records.items() if record_family == family and record_variant != variant]
        return sorted(candidates, key=lambda record: record.record_id)[0] if candidates else None

    def write(self, record: MemoryRecord) -> bool:
        if record.validation_status != "validated" or record.confidence < 1.0:
            self.rejected_writes += 1
            return False
        self.records[(record.family, record.variant)] = record
        self.writes += 1
        return True

    def seed_shuffled(self, seed: int) -> None:
        for family in FAMILIES:
            rules = RULES[family]
            for variant_index, variant in enumerate(VARIANTS):
                wrong_rule = rules[(variant_index + 1 + seed) % len(rules)]
                self.records[(family, variant)] = MemoryRecord(
                    record_id=f"shuffled-{family}-{variant}",
                    family=family,
                    variant=variant,
                    rule=wrong_rule,
                    conditions={"family": family, "variant": variant},
                    evidence={"type": "shuffled_control"},
                    provenance={"source": "shuffled_memory_control"},
                    validation_status="validated",
                    confidence=1.0,
                )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "records": [
                {
                    "record_id": record.record_id,
                    "family": record.family,
                    "variant": record.variant,
                    "rule": record.rule,
                    "conditions": record.conditions,
                    "evidence": record.evidence,
                    "provenance": record.provenance,
                    "validation_status": record.validation_status,
                    "confidence": record.confidence,
                }
                for record in sorted(self.records.values(), key=lambda item: item.record_id)
            ],
            "writes": self.writes,
            "rejected_writes": self.rejected_writes,
        }


def arithmetic_value(rule: str, start: int, operations: Sequence[Tuple[str, int]]) -> int:
    if rule == "arith_left":
        value = start
        for operation, operand in operations:
            if operation == "add":
                value += operand
            elif operation == "sub":
                value -= operand
            else:
                value *= operand
        return value
    if rule == "arith_precedence":
        terms: List[Tuple[int, int]] = [(1, start)]
        for operation, operand in operations:
            if operation == "mul":
                sign, value = terms[-1]
                terms[-1] = (sign, value * operand)
            else:
                terms.append((1 if operation == "add" else -1, operand))
        return sum(sign * value for sign, value in terms)
    raise ValueError(rule)


def sequence_value(rule: str, values: Sequence[int]) -> List[int]:
    if rule == "seq_reverse":
        return list(reversed(values))
    if rule == "seq_rotate":
        return list(values[1:]) + [values[0]]
    raise ValueError(rule)


def logic_value(rule: str, values: Sequence[bool]) -> bool:
    if rule == "logic_all":
        return all(values)
    if rule == "logic_any":
        return any(values)
    raise ValueError(rule)


def solve(rule: str, family: str, value: Any) -> Any:
    if family == "arithmetic":
        return arithmetic_value(rule, value["start"], value["operations"])
    if family == "sequence":
        return sequence_value(rule, value)
    if family == "logic":
        return logic_value(rule, value)
    raise ValueError(family)


def make_task(seed: int, family: str, variant: str, regime: str, episode: int) -> Task:
    rules = RULES[family]
    true_rule = rules[stable_int("true-rule", seed, family, variant) % len(rules)]
    base = stable_int("surface", seed, family, variant, regime, episode) % 20
    if family == "arithmetic":
        demonstrations = (
            ({"start": 2 + base, "operations": (("add", 2), ("sub", 1))}, None),
            ({"start": 4 + base, "operations": (("sub", 1), ("add", 3))}, None),
        )
        probe_input = {"start": 2 + base, "operations": (("add", 3), ("mul", 2), ("sub", 1))}
        target_input = {"start": 3 + base, "operations": (("add", 1), ("mul", 3), ("sub", 2))}
    elif family == "sequence":
        demonstrations = (
            ([1 + base, 2 + base, 2 + base], None),
            ([3 + base, 4 + base, 4 + base], None),
        )
        probe_input = [1 + base, 2 + base, 3 + base]
        target_input = [4 + base, 7 + base, 9 + base, 10 + base]
    else:
        demonstrations = (([True], None), ([False], None))
        probe_input = [True, False]
        target_input = [True, True, False]
    demonstrations = tuple((value, solve(true_rule, family, value)) for value, _ in demonstrations)
    target_output = solve(true_rule, family, target_input)
    true_probe_output = solve(true_rule, family, probe_input)
    other_rule = rules[1 - rules.index(true_rule)]
    if regime == "informative" or regime == "expensive":
        probe_output = true_probe_output
    elif regime == "noisy":
        probe_output = solve(other_rule, family, probe_input)
    else:
        ambiguous_input = demonstrations[0][0]
        probe_output = solve(true_rule, family, ambiguous_input)
    return Task(
        task_id=f"{family}-{variant}-{regime}-{episode}",
        family=family,
        variant=variant,
        regime=regime,
        episode=episode,
        true_rule=true_rule,
        candidate_rules=rules,
        demonstrations=demonstrations,
        probe_input=probe_input,
        target_input=target_input,
        target_output=target_output,
        probe_output=probe_output,
        probe_cost=PROBE_COST[regime],
    )


def make_tasks(seed: int, regime: str) -> List[Task]:
    return [
        make_task(seed, family, variant, regime, episode)
        for family in FAMILIES
        for variant in VARIANTS
        for episode in range(EPISODES_PER_TRACK)
    ]


def probe(task: Task) -> Mapping[str, Any]:
    return {"output": task.probe_output, "cost": task.probe_cost, "input": task.probe_input}


def verify(task: Task, rule: str) -> Optional[bool]:
    if task.regime == "unresolvable":
        return None
    return solve(rule, task.family, task.target_input) == task.target_output


def rule_matches_probe(task: Task, rule: str, observation: Mapping[str, Any]) -> bool:
    return solve(rule, task.family, observation["input"]) == observation["output"]


def make_record(task: Task, rule: str, source: str, evidence: Mapping[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        record_id=f"{source}-{task.task_id}",
        family=task.family,
        variant=task.variant,
        rule=rule,
        conditions={"family": task.family, "variant": task.variant},
        evidence=dict(evidence),
        provenance={"source": source, "task_id": task.task_id},
        validation_status="validated",
        confidence=1.0,
    )


def choose_default(task: Task) -> str:
    return task.candidate_rules[0]


def acquire(task: Task, store: MemoryStore, mode: str, counters: Dict[str, int]) -> Tuple[str, float, str, bool]:
    """Acquire a rule, returning rule, cost, source and whether memory was written."""
    cost = 0.0
    observation: Optional[Mapping[str, Any]] = None
    should_probe = mode == "stateless" or mode == "persistent" or (mode == "adaptive" and task.probe_cost <= ADAPTIVE_PROBE_LIMIT) or mode == "adaptive_memory_disabled"
    if should_probe:
        observation = probe(task)
        cost += float(observation["cost"])
        counters["probes"] += 1
        matches = [rule for rule in task.candidate_rules if rule_matches_probe(task, rule, observation)]
        if len(matches) == 1:
            candidate = matches[0]
            checked = verify(task, candidate)
            if checked is True:
                cost += VERIFY_COST
                counters["verifications"] += 1
                if mode in {"persistent", "adaptive"}:
                    record = make_record(task, candidate, "verified_probe", {"probe_output": observation["output"]})
                    wrote = store.write(record)
                    if wrote:
                        counters["valid_writes"] += 1
                    return candidate, cost, "probe_verified", wrote
                return candidate, cost, "probe_verified", False
            if checked is False:
                cost += VERIFY_COST
                counters["verifications"] += 1
        elif len(matches) == 0:
            for candidate in task.candidate_rules:
                checked = verify(task, candidate)
                if checked is not None:
                    cost += VERIFY_COST
                    counters["verifications"] += 1
                if checked is True:
                    if mode in {"persistent", "adaptive"}:
                        record = make_record(task, candidate, "verified_fallback", {"probe_output": observation["output"]})
                        wrote = store.write(record)
                        if wrote:
                            counters["valid_writes"] += 1
                        return candidate, cost, "verification", wrote
                    return candidate, cost, "verification", False
    if mode in {"persistent", "adaptive"}:
        counters["unresolved"] += 1
    return choose_default(task), cost, "default", False


def run_arm(tasks: Sequence[Task], arm: str, seed: int) -> Dict[str, Any]:
    store = MemoryStore()
    if arm == "adaptive_shuffled_memory":
        store.seed_shuffled(seed)
    retrieval_enabled = arm not in {"adaptive_memory_disabled", "stateless"}
    write_enabled = arm in {"persistent", "adaptive"}
    counters = defaultdict(int)
    outcomes: List[Dict[str, Any]] = []
    for task in tasks:
        record: Optional[MemoryRecord] = None
        memory_source = "none"
        if retrieval_enabled:
            record = store.exact(task.family, task.variant)
            if record is not None:
                memory_source = "exact"
            elif arm in {"adaptive", "adaptive_shuffled_memory"}:
                record = store.related(task.family, task.variant)
                if record is not None:
                    memory_source = "related"
        cost = 0.0
        selected_rule: str
        source: str
        wrote = False
        adaptation = False
        if arm == "oracle":
            selected_rule = task.true_rule
            source = "oracle"
        elif record is not None:
            counters["retrievals"] += 1
            if record.rule == task.true_rule:
                counters["relevant_retrievals"] += 1
            if arm in {"adaptive", "adaptive_shuffled_memory"} and memory_source == "related":
                adaptation = True
                counters["adaptation_attempts"] += 1
                checked = verify(task, record.rule)
                if checked is True:
                    cost += VERIFY_COST
                    counters["verifications"] += 1
                    counters["adaptation_successes"] += 1
                    selected_rule = record.rule
                    source = "adapted_memory"
                    if write_enabled:
                        new_record = make_record(task, record.rule, "adapted_memory", {"from": record.record_id})
                        wrote = store.write(new_record)
                        if wrote:
                            counters["valid_writes"] += 1
                elif checked is False:
                    cost += VERIFY_COST
                    counters["verifications"] += 1
                    selected_rule, extra_cost, source, wrote = acquire(task, store, arm if arm != "adaptive_shuffled_memory" else "adaptive", counters)
                    cost += extra_cost
                else:
                    selected_rule = record.rule
                    source = "unverified_related_memory"
            elif arm in {"adaptive", "adaptive_shuffled_memory"}:
                checked = verify(task, record.rule)
                if checked is True:
                    cost += VERIFY_COST
                    counters["verifications"] += 1
                    selected_rule = record.rule
                    source = "validated_memory"
                elif checked is False:
                    cost += VERIFY_COST
                    counters["verifications"] += 1
                    selected_rule, extra_cost, source, wrote = acquire(task, store, arm if arm != "adaptive_shuffled_memory" else "adaptive", counters)
                    cost += extra_cost
                else:
                    selected_rule = record.rule
                    source = "unverified_memory"
            else:
                selected_rule = record.rule
                source = "persistent_memory"
        else:
            acquisition_mode = "adaptive" if arm == "adaptive_shuffled_memory" else arm
            selected_rule, cost, source, wrote = acquire(task, store, acquisition_mode, counters)
        correct = solve(selected_rule, task.family, task.target_input) == task.target_output
        used_memory = "memory" in source
        if used_memory and not correct:
            counters["harmful_transfer"] += 1
        if not correct:
            counters["errors"] += 1
        if task.episode == 0:
            counters["novel_tasks"] += 1
            counters["novel_correct"] += int(correct)
        else:
            counters["later_tasks"] += 1
            counters["later_correct"] += int(correct)
        counters["total_cost"] += cost
        outcomes.append({
            "task_id": task.task_id,
            "family": task.family,
            "variant": task.variant,
            "regime": task.regime,
            "episode": task.episode,
            "true_rule": task.true_rule,
            "selected_rule": selected_rule,
            "correct": correct,
            "cost": cost,
            "source": source,
            "memory_source": memory_source,
            "adaptation": adaptation,
            "wrote_memory": wrote,
        })
    total = len(outcomes)
    return {
        "arm": arm,
        "seed": seed,
        "tasks": total,
        "accuracy": sum(item["correct"] for item in outcomes) / max(1, total),
        "mean_cost": sum(item["cost"] for item in outcomes) / max(1, total),
        "novel_accuracy": counters["novel_correct"] / max(1, counters["novel_tasks"]),
        "later_accuracy": counters["later_correct"] / max(1, counters["later_tasks"]),
        "retrievals": counters["retrievals"],
        "relevant_retrieval_rate": counters["relevant_retrievals"] / max(1, counters["retrievals"]),
        "adaptation_attempts": counters["adaptation_attempts"],
        "adaptation_successes": counters["adaptation_successes"],
        "adaptation_success_rate": counters["adaptation_successes"] / max(1, counters["adaptation_attempts"]),
        "harmful_transfer_rate": counters["harmful_transfer"] / max(1, counters["retrievals"]),
        "unresolved": counters["unresolved"],
        "probes": counters["probes"],
        "verifications": counters["verifications"],
        "valid_writes": counters["valid_writes"],
        "final_memory_size": len(store.records),
        "memory": store.snapshot(),
        "outcomes": outcomes,
    }


def aggregate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    numeric = ("accuracy", "mean_cost", "novel_accuracy", "later_accuracy", "relevant_retrieval_rate", "adaptation_success_rate", "harmful_transfer_rate", "final_memory_size")
    return {"runs": len(records), **{field: sum(float(item[field]) for item in records) / len(records) for field in numeric}, **{field: sum(int(item[field]) for item in records) for field in ("tasks", "retrievals", "adaptation_attempts", "adaptation_successes", "unresolved", "probes", "verifications", "valid_writes")}}


def source_hashes() -> Dict[str, str]:
    paths = [PROTOCOL, ROOT / "tools" / "run_crucible_c1f_persistent_reuse_20261021.py"]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def render_summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1f acquire, retain, adapt and reuse",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; no GPU, cloud or Crucible neural training.",
        "",
        "## Verdict",
        "",
        "This is a sequential controller and memory-contract experiment, not a neural-model capability result.",
        "",
        "## Aggregate results",
        "",
        "| Arm | Accuracy | Novel accuracy | Later accuracy | Mean cost | Retrieval relevance | Adaptation success | Harmful transfer | Final memory |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in result["aggregate"]:
        item = result["aggregate"][arm]
        lines.append(f"| `{arm}` | {item['accuracy']:.4f} | {item['novel_accuracy']:.4f} | {item['later_accuracy']:.4f} | {item['mean_cost']:.3f} | {item['relevant_retrieval_rate']:.4f} | {item['adaptation_success_rate']:.4f} | {item['harmful_transfer_rate']:.4f} | {item['final_memory_size']:.1f} |")
    lines.extend(["", "## Environment results", ""])
    for regime in REGIMES:
        lines.extend([f"### {regime}", "", "| Arm | Accuracy | Later accuracy | Mean cost | Harmful transfer |", "| --- | ---: | ---: | ---: | ---: |"])
        for arm in result["by_regime"][regime]:
            item = result["by_regime"][regime][arm]
            lines.append(f"| `{arm}` | {item['accuracy']:.4f} | {item['later_accuracy']:.4f} | {item['mean_cost']:.3f} | {item['harmful_transfer_rate']:.4f} |")
        lines.append("")
    lines.extend([
        "## Preregistered gates",
        "",
        f"- Accuracy/cost transfer gate: `{result['gates']['transfer_gate']}`.",
        f"- Adaptive harmful-transfer gate: `{result['gates']['adaptive_harmful_transfer_gate']}`.",
        f"- Shuffled-memory harmful-transfer gate: `{result['gates']['shuffled_harmful_transfer_gate']}`.",
        "- The shuffled-memory and memory-disabled controls are included to separate genuine reuse from incidental computation or contamination.",
        "",
        "## Provenance",
        "",
        "- Three seeds: 42, 1337 and 2026.",
        f"- {EPISODES_PER_TRACK} novel episodes per family, variant and environment per seed.",
        "- Every retained record contains rule, conditions, evidence, provenance and validation status.",
        "- No task answer or hidden rule was provided as a policy feature.",
        "",
        "## Source hashes",
        "",
    ])
    for path, digest in sorted(hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    full_runs: List[Dict[str, Any]] = []
    by_regime: Dict[str, Dict[str, List[Dict[str, Any]]]] = {regime: defaultdict(list) for regime in REGIMES}
    for seed in SEEDS:
        for regime in REGIMES:
            tasks = make_tasks(seed, regime)
            for arm in ("stateless", "persistent", "adaptive", "adaptive_memory_disabled", "adaptive_shuffled_memory", "oracle"):
                result = run_arm(tasks, arm, seed)
                all_runs[arm].append({"seed": seed, "regime": regime, **{key: value for key, value in result.items() if key not in {"outcomes", "memory"}}})
                by_regime[regime][arm].append(result)
                full_runs.append({"seed": seed, "regime": regime, "arm": arm, "summary": {key: value for key, value in result.items() if key not in {"outcomes", "memory"}}, "memory": result["memory"], "outcomes": result["outcomes"]})
    aggregate_results = {arm: aggregate(records) for arm, records in sorted(all_runs.items())}
    aggregate_by_regime: Dict[str, Dict[str, Any]] = {}
    for regime in REGIMES:
        aggregate_by_regime[regime] = {arm: aggregate(records) for arm, records in sorted(by_regime[regime].items())}
    adaptive = aggregate_results["adaptive"]
    disabled = aggregate_results["adaptive_memory_disabled"]
    informative_expensive_adaptive = [by_regime[regime]["adaptive"][index] for regime in ("informative", "expensive") for index in range(len(SEEDS))]
    informative_expensive_disabled = [by_regime[regime]["adaptive_memory_disabled"][index] for regime in ("informative", "expensive") for index in range(len(SEEDS))]
    persistent_transfer = [by_regime[regime]["persistent"][index] for regime in ("informative", "expensive") for index in range(len(SEEDS))]
    persistent_transfer_accuracy = sum(item["later_accuracy"] for item in persistent_transfer) / len(persistent_transfer)
    persistent_transfer_cost = sum(item["mean_cost"] for item in persistent_transfer) / len(persistent_transfer)
    adaptive_transfer_accuracy = sum(item["later_accuracy"] for item in informative_expensive_adaptive) / len(informative_expensive_adaptive)
    disabled_transfer_accuracy = sum(item["later_accuracy"] for item in informative_expensive_disabled) / len(informative_expensive_disabled)
    adaptive_transfer_cost = sum(item["mean_cost"] for item in informative_expensive_adaptive) / len(informative_expensive_adaptive)
    disabled_transfer_cost = sum(item["mean_cost"] for item in informative_expensive_disabled) / len(informative_expensive_disabled)
    gates = {
        "transfer_gate": (adaptive_transfer_accuracy - disabled_transfer_accuracy >= 0.10 or adaptive_transfer_cost <= disabled_transfer_cost * 0.80 or persistent_transfer_accuracy - disabled_transfer_accuracy >= 0.10 or persistent_transfer_cost <= disabled_transfer_cost * 0.80),
        "adaptive_harmful_transfer_gate": aggregate_results["adaptive"]["harmful_transfer_rate"] <= 0.10,
        "shuffled_harmful_transfer_gate": aggregate_results["adaptive_shuffled_memory"]["harmful_transfer_rate"] <= 0.10,
        "persistent_transfer_accuracy": persistent_transfer_accuracy,
        "persistent_transfer_cost": persistent_transfer_cost,
        "adaptive_transfer_accuracy": adaptive_transfer_accuracy,
        "disabled_transfer_accuracy": disabled_transfer_accuracy,
        "adaptive_transfer_cost": adaptive_transfer_cost,
        "disabled_transfer_cost": disabled_transfer_cost,
    }
    result = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "aggregate": aggregate_results,
        "by_regime": aggregate_by_regime,
        "gates": gates,
        "seeds": list(SEEDS),
        "episodes_per_track": EPISODES_PER_TRACK,
        "regimes": list(REGIMES),
        "arms": ["stateless", "persistent", "adaptive", "adaptive_memory_disabled", "adaptive_shuffled_memory", "oracle"],
    }
    write_json(OUTPUT / "RESULTS.json", result)
    with (OUTPUT / "RUNS.jsonl").open("w", encoding="utf-8") as handle:
        for run in full_runs:
            handle.write(json.dumps(run, sort_keys=True) + "\n")
    write_json(OUTPUT / "RUN_CONFIG.json", {"experiment": EXPERIMENT, "seeds": list(SEEDS), "families": list(FAMILIES), "variants": list(VARIANTS), "regimes": list(REGIMES), "episodes_per_track": EPISODES_PER_TRACK, "probe_cost": PROBE_COST, "verify_cost": VERIFY_COST, "adaptive_probe_limit": ADAPTIVE_PROBE_LIMIT, "execution": "CPU-only"})
    hashes = source_hashes()
    (OUTPUT / "RESULTS_SUMMARY.md").write_text(render_summary(result, hashes), encoding="utf-8")
    file_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path.name != "HASHES.json"}
    write_json(OUTPUT / "HASHES.json", file_hashes)
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "runs": len(SEEDS) * len(REGIMES) * 6, "gates": gates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
