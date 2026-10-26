#!/usr/bin/env python3
"""Run the CPU-only C1j learned-applicability stress experiment.

This runner is intentionally standalone.  It does not import the earlier
C1f/C1g/C1h/C1i generators, so the held-out-family and memory-size stress
conditions remain independently auditable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_c1j_cpu_stress_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
EXPERIMENT = "crucible_c1j_cpu_stress_20261021"

LEARNER_SEEDS = (42, 1337, 2026)
SOURCE_FAMILIES = ("vector", "grid", "graph", "set")
HELDOUT_FAMILIES = ("interval", "multiset", "tree", "matrix")
ALL_FAMILIES = SOURCE_FAMILIES + HELDOUT_FAMILIES
CONDITIONS = (
    "applicable",
    "adaptation",
    "invalidated",
    "contradictory",
    "misleading_similarity",
    "noisy_evidence",
    "provenance_corrupt",
    "unresolvable",
)
MEMORY_SIZES = (10, 100, 1000, 10000)
TRAIN_CANDIDATES = 20_000
VALIDATION_CANDIDATES = 5_000
TEST_TASKS = 10_000
TRAIN_UPDATES = 3_000
CHECKPOINTS = (0, 500, 1000, 1500, 2000, 2500, 3000)

PROBE_COST = 0.80
VERIFY_COST = 0.50
FEATURE_COST = 0.10
DETERMINISTIC_GATE_COST = 0.18

RULES: Dict[str, Tuple[str, str]] = {
    "vector": ("vector_reverse", "vector_rotate"),
    "grid": ("grid_transpose", "grid_reflect"),
    "graph": ("graph_reach", "graph_direct"),
    "set": ("set_union", "set_intersection"),
    "interval": ("interval_overlap", "interval_contains"),
    "multiset": ("multiset_union", "multiset_unique"),
    "tree": ("tree_depth", "tree_leaf_count"),
    "matrix": ("matrix_transpose", "matrix_rotate"),
}

FEATURE_NAMES = (
    "same_family",
    "variant_match",
    "surface_match",
    "structural_match",
    "environment_match",
    "probe_available",
    "verifier_available",
    "probe_agrees",
    "evidence_quality",
    "provenance_valid",
    "contradiction_flag",
    "retrieval_rank_normalized",
    "log10_distractors_normalized",
    "memory_size_normalized",
)


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


def safe_solve(rule: str, family: str, value: Any) -> Any:
    try:
        return solve(rule, family, value)
    except (KeyError, TypeError, ValueError, IndexError):
        return None


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
    elif family == "interval":
        left = tuple(value["left"])
        right = tuple(value["right"])
        if rule == "interval_overlap":
            return max(left[0], right[0]) < min(left[1], right[1])
        if rule == "interval_contains":
            return left[0] <= right[0] and left[1] >= right[1]
    elif family == "multiset":
        left = list(value["left"])
        right = list(value["right"])
        if rule == "multiset_union":
            return sorted(left + right)
        if rule == "multiset_unique":
            return sorted(set(left + right))
    elif family == "tree":
        edges = [tuple(edge) for edge in value["edges"]]
        root = value["root"]
        children: Dict[int, List[int]] = defaultdict(list)
        for parent, child in edges:
            children[parent].append(child)
        if rule == "tree_depth":
            depths = [(root, 0)]
            maximum = 0
            while depths:
                node, depth = depths.pop()
                maximum = max(maximum, depth)
                depths.extend((child, depth + 1) for child in children.get(node, []))
            return maximum
        if rule == "tree_leaf_count":
            nodes = {root} | {node for edge in edges for node in edge}
            return sum(not children.get(node) for node in nodes)
    elif family == "matrix":
        matrix = [list(row) for row in value["matrix"]]
        if rule == "matrix_transpose":
            return [[matrix[row][column] for row in range(len(matrix))] for column in range(len(matrix[0]))]
        if rule == "matrix_rotate":
            transposed = [[matrix[row][column] for row in range(len(matrix))] for column in range(len(matrix[0]))]
            return [list(reversed(row)) for row in transposed]
    raise ValueError(f"unknown rule/family: {rule}/{family}")


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    condition: str
    phase: str
    variant: str
    surface_group: str
    structural_signature: str
    environment_version: int
    true_rule: str
    candidate_rules: Tuple[str, ...]
    probe_input: Any
    probe_output: Any
    target_input: Any
    target_output: Any
    probe_available: bool
    verifier_available: bool
    memory_size: int
    distractor_count: int


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    family: str
    rule: str
    variant: str
    surface_group: str
    structural_signature: str
    environment_version: int
    evidence_quality: float
    provenance_valid: bool
    contradiction_flag: bool
    retrieval_rank: int
    memory_size: int
    distractor_count: int


@dataclass(frozen=True)
class Candidate:
    split: str
    index: int
    task: Task
    record: MemoryRecord
    features: Tuple[float, ...]
    label: int


def task_inputs(family: str, base: int, structural: str) -> Tuple[Any, Any]:
    wide = structural in {"transfer", "holdout"}
    if family == "vector":
        size = 5 if wide else 4
        values = [base + 2 * index for index in range(size)]
        return {"values": values}, {"values": [base + 3 + 2 * index for index in range(size + 1)]}
    if family == "grid":
        rows = 3 if wide else 2
        cols = 3 if wide else 2
        probe = [[base + row * cols + column for column in range(cols)] for row in range(rows)]
        target = [[base + 20 + row * cols + column for column in range(cols)] for row in range(rows)]
        return {"grid": probe}, {"grid": target}
    if family == "graph":
        length = 4 if wide else 3
        probe_edges = [[node, node + 1] for node in range(length)]
        target_edges = [[node, node + 1] for node in range(length + 1)]
        return {"edges": probe_edges, "start": 0, "goal": length}, {"edges": target_edges, "start": 0, "goal": length + 1}
    if family == "set":
        offset = 2 if wide else 1
        return {"left": [base, base + offset], "right": [base + offset, base + 4]}, {"left": [base + 1, base + 3], "right": [base + 3, base + 6]}
    if family == "interval":
        left = [base, base + (8 if wide else 5)]
        right = [base + 3, base + (6 if wide else 10)]
        target_left = [base, base + 10]
        target_right = [base + 2, base + (4 if wide else 12)]
        return {"left": left, "right": right}, {"left": target_left, "right": target_right}
    if family == "multiset":
        probe = {"left": [base, base + 1, base + 1], "right": [base + 1, base + 2]}
        target = {"left": [base + 2, base + 3, base + 3], "right": [base + 3, base + 4]}
        return probe, target
    if family == "tree":
        branch = 3 if wide else 2
        probe_edges = [[0, 1], [1, 2], [1, branch]]
        target_edges = [[0, 1], [1, 2], [1, branch], [2, 4], [branch, 5]]
        return {"edges": probe_edges, "root": 0}, {"edges": target_edges, "root": 0}
    if family == "matrix":
        rows = 3 if wide else 2
        cols = 2 if wide else 3
        probe = [[base + row * cols + column for column in range(cols)] for row in range(rows)]
        target = [[base + 30 + row * cols + column for column in range(cols)] for row in range(rows)]
        return {"matrix": probe}, {"matrix": target}
    raise ValueError(family)


def retrieval_is_correct(seed: int, family: str, condition: str, index: int, memory_size: int) -> bool:
    rates = {10: 0.98, 100: 0.95, 1000: 0.88, 10000: 0.75}
    draw = stable_int("c1j-retrieval", seed, family, condition, index, memory_size) % 10_000
    return draw < int(rates[memory_size] * 10_000)


def make_task(seed: int, family: str, condition: str, index: int, phase: str, memory_size: int) -> Tuple[Task, MemoryRecord]:
    rules = RULES[family]
    stable_rule = rules[stable_int("c1j-stable-rule", seed, family) % len(rules)]
    changed_rule = rules[1 - rules.index(stable_rule)]
    stable_conditions = {"applicable", "adaptation", "noisy_evidence", "provenance_corrupt", "unresolvable"}
    true_rule = stable_rule if condition in stable_conditions else changed_rule
    structural = {
        "applicable": "core",
        "adaptation": "transfer",
        "invalidated": "core",
        "contradictory": "core",
        "misleading_similarity": "decoy",
        "noisy_evidence": "core",
        "provenance_corrupt": "core",
        "unresolvable": "unknown",
    }[condition]
    environment = 1 if condition == "invalidated" else 2 if condition == "unresolvable" else 0
    variant = "holdout" if condition == "adaptation" else "source"
    surface = "red" if condition != "misleading_similarity" else "red"
    base = stable_int("c1j-input", seed, family, condition, index, phase) % 50 + 1
    probe_input, target_input = task_inputs(family, base, structural)
    target_output = solve(true_rule, family, target_input)
    true_probe_output = solve(true_rule, family, probe_input)
    observed_probe_output = true_probe_output
    if condition == "noisy_evidence":
        observed_probe_output = solve(changed_rule, family, probe_input)
    probe_available = condition != "unresolvable"
    verifier_available = condition != "unresolvable"
    task = Task(
        task_id=f"{phase}-{family}-{condition}-{index}-{memory_size}",
        family=family,
        condition=condition,
        phase=phase,
        variant=variant,
        surface_group=surface,
        structural_signature=structural,
        environment_version=environment,
        true_rule=true_rule,
        candidate_rules=rules,
        probe_input=probe_input,
        probe_output=observed_probe_output,
        target_input=target_input,
        target_output=target_output,
        probe_available=probe_available,
        verifier_available=verifier_available,
        memory_size=memory_size,
        distractor_count=memory_size - 1,
    )

    correct_retrieval = retrieval_is_correct(seed, family, condition, index, memory_size)
    record_family = family
    record_rule = stable_rule
    if not correct_retrieval:
        if stable_int("c1j-distractor-family", seed, family, condition, index, memory_size) % 2:
            other_families = [candidate for candidate in ALL_FAMILIES if candidate != family]
            record_family = other_families[stable_int("c1j-other-family", seed, index, memory_size) % len(other_families)]
            record_rule = RULES[record_family][stable_int("c1j-other-rule", seed, index) % 2]
        else:
            record_rule = changed_rule
    record_structural = "transfer" if condition == "adaptation" else "core"
    record_environment = 0
    evidence_quality = 1.0
    provenance_valid = True
    contradiction_flag = False
    if condition == "noisy_evidence":
        evidence_quality = 0.35
    elif condition == "provenance_corrupt":
        provenance_valid = False
    elif condition == "contradictory":
        contradiction_flag = True
    retrieval_rank = 1 if correct_retrieval else 2 + (stable_int("c1j-rank", seed, index, memory_size) % 7)
    record = MemoryRecord(
        record_id=f"record-{task.task_id}",
        family=record_family,
        rule=record_rule,
        variant="source",
        surface_group=surface,
        structural_signature=record_structural,
        environment_version=record_environment,
        evidence_quality=evidence_quality,
        provenance_valid=provenance_valid,
        contradiction_flag=contradiction_flag,
        retrieval_rank=retrieval_rank,
        memory_size=memory_size,
        distractor_count=memory_size - 1,
    )
    return task, record


def feature_vector(task: Task, record: MemoryRecord) -> Tuple[float, ...]:
    same_family = task.family == record.family
    probe_agrees = task.probe_available and safe_solve(record.rule, task.family, task.probe_input) == task.probe_output
    return (
        float(same_family),
        float(task.variant == record.variant),
        float(task.surface_group == record.surface_group),
        float(task.structural_signature == record.structural_signature),
        float(task.environment_version == record.environment_version),
        float(task.probe_available),
        float(task.verifier_available),
        float(probe_agrees),
        float(record.evidence_quality),
        float(record.provenance_valid),
        float(record.contradiction_flag),
        1.0 / max(1.0, float(record.retrieval_rank)),
        math.log10(1.0 + record.distractor_count) / 4.0,
        math.log10(float(record.memory_size)) / 4.0,
    )


def label_for(task: Task, record: MemoryRecord) -> int:
    correct_rule = safe_solve(record.rule, task.family, task.target_input) == task.target_output
    structural_ok = task.structural_signature == record.structural_signature
    evidence_ok = record.evidence_quality >= 0.8 and record.provenance_valid and not record.contradiction_flag
    probe_ok = task.probe_available and safe_solve(record.rule, task.family, task.probe_input) == task.probe_output
    return int(
        task.family == record.family
        and correct_rule
        and structural_ok
        and task.environment_version == record.environment_version
        and evidence_ok
        and probe_ok
    )


def generate_candidates(split: str, count: int, seed: int, families: Sequence[str]) -> List[Candidate]:
    candidates: List[Candidate] = []
    for index in range(count):
        family = families[stable_int("c1j-family", split, seed, index) % len(families)]
        condition = CONDITIONS[stable_int("c1j-condition", split, seed, index) % len(CONDITIONS)]
        memory_size = MEMORY_SIZES[stable_int("c1j-memory-size", split, seed, index) % len(MEMORY_SIZES)]
        task_seed = stable_int("c1j-task-seed", split, seed, index)
        task, record = make_task(task_seed, family, condition, index, split, memory_size)
        features = feature_vector(task, record)
        candidates.append(Candidate(split, index, task, record, features, label_for(task, record)))
    return candidates


@dataclass
class LogisticApplicability:
    weights: List[float]
    bias: float
    updates: int
    curve: List[Dict[str, float]]

    @classmethod
    def initialize(cls, seed: int, feature_count: int) -> "LogisticApplicability":
        weights = [((stable_int("c1j-init", seed, index) % 200) - 100) / 100_000.0 for index in range(feature_count)]
        return cls(weights=weights, bias=0.0, updates=0, curve=[])

    def probability(self, features: Sequence[float]) -> float:
        logit = self.bias + sum(weight * feature for weight, feature in zip(self.weights, features))
        logit = max(-40.0, min(40.0, logit))
        return 1.0 / (1.0 + math.exp(-logit))

    def train(self, candidates: Sequence[Candidate], seed: int) -> None:
        learning_rate = 0.08
        l2 = 0.0005
        for update in range(1, TRAIN_UPDATES + 1):
            candidate = candidates[stable_int("c1j-order", seed, update) % len(candidates)]
            probability = self.probability(candidate.features)
            error = probability - candidate.label
            for feature_index, feature in enumerate(candidate.features):
                self.weights[feature_index] -= learning_rate * (error * feature + l2 * self.weights[feature_index])
            self.bias -= learning_rate * error
            self.updates = update
            if update in CHECKPOINTS[1:]:
                loss = -(candidate.label * math.log(max(probability, 1e-12)) + (1 - candidate.label) * math.log(max(1 - probability, 1e-12)))
                self.curve.append({"update": float(update), "sample_loss": float(loss)})


def model_metrics(model: LogisticApplicability, candidates: Sequence[Candidate]) -> Dict[str, float]:
    probabilities = [model.probability(candidate.features) for candidate in candidates]
    labels = [candidate.label for candidate in candidates]
    return {
        "candidates": float(len(candidates)),
        "label_rate": sum(labels) / max(1, len(labels)),
        "classification_accuracy": sum((probability >= 0.5) == bool(label) for probability, label in zip(probabilities, labels)) / max(1, len(labels)),
        "brier": sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / max(1, len(labels)),
        "mean_probability": sum(probabilities) / max(1, len(probabilities)),
    }


def observed_probe_agrees(task: Task, record: MemoryRecord) -> bool:
    return bool(task.probe_available and safe_solve(record.rule, task.family, task.probe_input) == task.probe_output)


def hard_evidence(task: Task, record: MemoryRecord) -> bool:
    return bool(
        task.family == record.family
        and task.structural_signature == record.structural_signature
        and task.environment_version == record.environment_version
        and task.probe_available
        and observed_probe_agrees(task, record)
        and record.evidence_quality >= 0.8
        and record.provenance_valid
        and not record.contradiction_flag
    )


def investigate(task: Task) -> Tuple[Optional[str], float, str]:
    if not task.probe_available or not task.verifier_available:
        return None, 0.0, "abstain"
    cost = PROBE_COST
    matching = [rule for rule in task.candidate_rules if safe_solve(rule, task.family, task.probe_input) == task.probe_output]
    ordered = matching + [rule for rule in task.candidate_rules if rule not in matching]
    for rule in ordered:
        cost += VERIFY_COST
        if safe_solve(rule, task.family, task.target_input) == task.target_output:
            return rule, cost, "investigate"
    return None, cost, "abstain"


def policy_decision(task: Task, record: MemoryRecord, policy: str, model: Optional[LogisticApplicability]) -> Tuple[Optional[str], float, str, str, Optional[float]]:
    probability: Optional[float] = None
    if policy == "oracle":
        return task.true_rule, 0.0, "oracle", "none", None
    if policy == "stateless":
        selected, cost, decision = investigate(task)
        return selected, cost, decision, "none", None
    if policy == "persistent":
        return record.rule, 0.0, "reuse", "persistent", None
    if policy == "deterministic":
        if hard_evidence(task, record):
            if task.variant == record.variant and task.structural_signature == record.structural_signature:
                return record.rule, DETERMINISTIC_GATE_COST, "reuse", "deterministic", None
            checked = safe_solve(record.rule, task.family, task.target_input) == task.target_output
            if checked:
                return record.rule, DETERMINISTIC_GATE_COST + VERIFY_COST, "adapt", "deterministic", None
        selected, cost, decision = investigate(task)
        return selected, cost + DETERMINISTIC_GATE_COST, decision, "none", None
    if policy in {"learned", "hybrid"}:
        if model is None:
            raise ValueError("learned and hybrid policies require a model")
        probability = model.probability(feature_vector(task, record))
        if probability >= 0.5 and (policy == "learned" or hard_evidence(task, record)):
            decision = "reuse" if task.variant == record.variant and task.structural_signature == record.structural_signature else "adapt"
            return record.rule, FEATURE_COST, decision, policy, probability
        selected, cost, decision = investigate(task)
        return selected, cost + FEATURE_COST, decision, "none", probability
    raise ValueError(policy)


def safe_correct(task: Task, selected: Optional[str]) -> bool:
    return selected is not None and safe_solve(selected, task.family, task.target_input) == task.target_output


def group_metrics(outcomes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(outcomes)
    reuse = [item for item in outcomes if item["decision"] in {"reuse", "adapt"}]
    abstentions = [item for item in outcomes if item["decision"] == "abstain"]
    harmful = [item for item in reuse if not item["correct"]]
    applicable = [item for item in outcomes if item["condition"] in {"applicable", "adaptation"}]
    adaptations = [item for item in outcomes if item["decision"] == "adapt"]
    resolvable = [item for item in outcomes if item["condition"] != "unresolvable"]
    insufficient = [item for item in outcomes if item["condition"] == "unresolvable"]
    probabilities = [item for item in outcomes if item.get("probability") is not None]
    return {
        "tasks": count,
        "accuracy": sum(bool(item["correct"]) for item in outcomes) / max(1, count),
        "mean_cost": sum(float(item["cost"]) for item in outcomes) / max(1, count),
        "harmful_transfer_rate": sum(not item["correct"] for item in harmful) / max(1, len(reuse)),
        "reuse_coverage": sum(item["decision"] in {"reuse", "adapt"} for item in applicable) / max(1, len(applicable)),
        "adaptation_attempts": len(adaptations),
        "adaptation_successes": sum(bool(item["correct"]) for item in adaptations),
        "adaptation_success_rate": sum(bool(item["correct"]) for item in adaptations) / max(1, len(adaptations)),
        "abstentions": len(abstentions),
        "abstention_precision": sum(item["condition"] == "unresolvable" for item in abstentions) / max(1, len(abstentions)),
        "false_abstentions": sum(item["condition"] != "unresolvable" for item in abstentions),
        "resolvable_accuracy": sum(bool(item["correct"]) for item in resolvable) / max(1, len(resolvable)),
        "insufficient_evidence_accuracy": sum(bool(item["correct"]) for item in insufficient) / max(1, len(insufficient)),
        "brier": sum((float(item["probability"]) - int(item["label"])) ** 2 for item in probabilities) / max(1, len(probabilities)),
        "probability_count": len(probabilities),
        "retrievals": len(outcomes),
        "action_counts": dict(sorted(Counter(item["decision"] for item in outcomes).items())),
    }


def evaluate_policy(tasks: Sequence[Candidate], policy: str, model: Optional[LogisticApplicability], seed: int) -> Dict[str, Any]:
    outcomes: List[Dict[str, Any]] = []
    for candidate in tasks:
        task = candidate.task
        record = candidate.record
        selected, cost, decision, source, probability = policy_decision(task, record, policy, model)
        correct = safe_correct(task, selected)
        outcomes.append({
            "split": candidate.split,
            "index": candidate.index,
            "task_id": task.task_id,
            "family": task.family,
            "condition": task.condition,
            "memory_size": task.memory_size,
            "distractor_count": task.distractor_count,
            "record_family": record.family,
            "record_rule": record.rule,
            "selected_rule": selected,
            "decision": decision,
            "source": source,
            "correct": correct,
            "harmful_transfer": decision in {"reuse", "adapt"} and not correct,
            "cost": cost,
            "probability": probability,
            "label": candidate.label,
        })
    summary = group_metrics(outcomes)
    summary.update({"policy": policy, "seed": seed, "outcomes": outcomes})
    return summary


def aggregate_summaries(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    numeric = (
        "accuracy", "mean_cost", "harmful_transfer_rate", "reuse_coverage",
        "adaptation_success_rate", "abstention_precision", "resolvable_accuracy",
        "insufficient_evidence_accuracy", "brier",
    )
    counts = (
        "tasks", "adaptation_attempts", "adaptation_successes", "abstentions",
        "false_abstentions", "probability_count", "retrievals",
    )
    if not records:
        return {"runs": 0}
    return {
        "runs": len(records),
        **{field: sum(float(item.get(field, 0.0)) for item in records) / len(records) for field in numeric},
        **{field: sum(int(item.get(field, 0)) for item in records) for field in counts},
    }


def subgroup_summaries(run: Mapping[str, Any], key: str, values: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    outcomes = run["outcomes"]
    result: Dict[str, Dict[str, Any]] = {}
    for value in values:
        subset = [item for item in outcomes if item[key] == value]
        result[str(value)] = group_metrics(subset)
    return result


def expected_calibration_error(outcomes: Sequence[Mapping[str, Any]], bins: int = 10) -> float:
    probabilities = [item for item in outcomes if item.get("probability") is not None]
    if not probabilities:
        return 0.0
    total = 0.0
    for bucket in range(bins):
        lower = bucket / bins
        upper = (bucket + 1) / bins
        members = [item for item in probabilities if lower <= float(item["probability"]) < upper or (bucket == bins - 1 and float(item["probability"]) == 1.0)]
        if members:
            confidence = sum(float(item["probability"]) for item in members) / len(members)
            frequency = sum(int(item["label"]) for item in members) / len(members)
            total += len(members) / len(probabilities) * abs(confidence - frequency)
    return total


def source_hashes() -> Dict[str, str]:
    paths = [PROTOCOL, ROOT / "tools" / "run_crucible_c1j_cpu_stress_20261021.py"]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def render_summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1j CPU memory-applicability stress test",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; no GPU, cloud or Crucible neural training.",
        "",
        "## Frozen scale",
        "",
        f"- Training candidates: `{result['config']['train_candidates']}`.",
        f"- Validation candidates: `{result['config']['validation_candidates']}`.",
        f"- Untouched held-out test tasks: `{result['config']['test_tasks']}`.",
        f"- Learner updates per seed: `{result['config']['train_updates']}`.",
        f"- Held-out families: `{', '.join(result['config']['heldout_families'])}`.",
        f"- Virtual memory sizes: `{', '.join(map(str, result['config']['memory_sizes']))}`.",
        "",
        "## Aggregate test results",
        "",
        "| Policy | Accuracy | Resolvable accuracy | Mean cost | Harmful transfer | Reuse coverage | Abstention precision | False abstentions | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, item in result["aggregate"].items():
        lines.append(
            f"| `{policy}` | {item['accuracy']:.4f} | {item['resolvable_accuracy']:.4f} | "
            f"{item['mean_cost']:.3f} | {item['harmful_transfer_rate']:.4f} | "
            f"{item['reuse_coverage']:.4f} | {item['abstention_precision']:.4f} | "
            f"{item['false_abstentions']} | {item['brier']:.4f} |"
        )
    lines.extend(["", "## Held-out-family results", "", "| Family | Policy | Accuracy | Mean cost | Harmful transfer | Brier |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for family, policies in result["by_family"].items():
        for policy, item in policies.items():
            lines.append(f"| `{family}` | `{policy}` | {item['accuracy']:.4f} | {item['mean_cost']:.3f} | {item['harmful_transfer_rate']:.4f} | {item['brier']:.4f} |")
    lines.extend(["", "## Memory-size results", "", "| Memory size | Policy | Accuracy | Mean cost | Harmful transfer | Reuse coverage |", "| ---: | --- | ---: | ---: | ---: | ---: |"])
    for memory_size, policies in result["by_memory_size"].items():
        for policy, item in policies.items():
            lines.append(f"| `{memory_size}` | `{policy}` | {item['accuracy']:.4f} | {item['mean_cost']:.3f} | {item['harmful_transfer_rate']:.4f} | {item['reuse_coverage']:.4f} |")
    lines.extend(["", "## Condition results", "", "| Condition | Policy | Accuracy | Resolvable accuracy | Mean cost | Harmful transfer | False abstentions |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for condition, policies in result["by_condition"].items():
        for policy, item in policies.items():
            lines.append(f"| `{condition}` | `{policy}` | {item['accuracy']:.4f} | {item['resolvable_accuracy']:.4f} | {item['mean_cost']:.3f} | {item['harmful_transfer_rate']:.4f} | {item['false_abstentions']} |")
    lines.extend([
        "",
        "## Preregistered gates",
        "",
        f"- Deterministic safety gate: `{result['gates']['deterministic_safety_gate']}`.",
        f"- Learned primary gate: `{result['gates']['learned_primary_gate']}`.",
        f"- Hybrid safety gate: `{result['gates']['hybrid_safety_gate']}`.",
        f"- Test separation: `{result['gates']['test_separation']}`.",
        "",
        "A failed gate is retained as a valid result. These synthetic outcomes do not establish neural reasoning or CMS2 training value.",
        "",
        "## Source hashes",
        "",
    ])
    for path, digest in sorted(hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = generate_candidates("train", TRAIN_CANDIDATES, 101, SOURCE_FAMILIES)
    validation = generate_candidates("validation", VALIDATION_CANDIDATES, 202, SOURCE_FAMILIES)
    test = generate_candidates("test", TEST_TASKS, 303, HELDOUT_FAMILIES)

    split_manifest = {
        "experiment": EXPERIMENT,
        "train": {"count": len(train), "families": list(SOURCE_FAMILIES), "task_digest": hashlib.sha256("|".join(candidate.task.task_id for candidate in train).encode()).hexdigest()},
        "validation": {"count": len(validation), "families": list(SOURCE_FAMILIES), "task_digest": hashlib.sha256("|".join(candidate.task.task_id for candidate in validation).encode()).hexdigest()},
        "test": {"count": len(test), "families": list(HELDOUT_FAMILIES), "task_digest": hashlib.sha256("|".join(candidate.task.task_id for candidate in test).encode()).hexdigest()},
        "memory_sizes": list(MEMORY_SIZES),
        "conditions": list(CONDITIONS),
        "feature_names": list(FEATURE_NAMES),
        "family_one_hot_features": False,
        "test_labels_not_used_as_features": True,
        "test_labels_not_used_for_selection": True,
    }
    write_json(OUTPUT / "SPLIT_MANIFEST.json", split_manifest)

    policies = ("deterministic", "learned", "hybrid", "persistent", "stateless", "oracle")
    all_runs: List[Dict[str, Any]] = []
    by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_condition: Dict[str, Dict[str, List[Dict[str, Any]]]] = {condition: defaultdict(list) for condition in CONDITIONS}
    by_family: Dict[str, Dict[str, List[Dict[str, Any]]]] = {family: defaultdict(list) for family in HELDOUT_FAMILIES}
    by_memory: Dict[str, Dict[str, List[Dict[str, Any]]]] = {str(size): defaultdict(list) for size in MEMORY_SIZES}
    training_reports: Dict[str, Any] = {}
    models: Dict[str, Any] = {}

    for learner_seed in LEARNER_SEEDS:
        model = LogisticApplicability.initialize(learner_seed, len(FEATURE_NAMES))
        model.train(train, learner_seed)
        training_reports[str(learner_seed)] = {
            "updates": model.updates,
            "checkpoints": list(CHECKPOINTS),
            "validation": model_metrics(model, validation),
            "train": model_metrics(model, train),
            "curve": model.curve,
        }
        models[str(learner_seed)] = {"weights": model.weights, "bias": model.bias, "updates": model.updates}
        for policy in policies:
            run = evaluate_policy(test, policy, model if policy in {"learned", "hybrid"} else None, learner_seed)
            summary = {key: value for key, value in run.items() if key != "outcomes"}
            summary["expected_calibration_error"] = expected_calibration_error(run["outcomes"])
            all_runs.append({"seed": learner_seed, "policy": policy, "summary": summary, "outcomes": run["outcomes"]})
            by_policy[policy].append(summary)
            for condition in CONDITIONS:
                subgroup = [item for item in run["outcomes"] if item["condition"] == condition]
                by_condition[condition][policy].append(group_metrics(subgroup))
            for family in HELDOUT_FAMILIES:
                subgroup = [item for item in run["outcomes"] if item["family"] == family]
                by_family[family][policy].append(group_metrics(subgroup))
            for memory_size in MEMORY_SIZES:
                subgroup = [item for item in run["outcomes"] if item["memory_size"] == memory_size]
                by_memory[str(memory_size)][policy].append(group_metrics(subgroup))

    aggregate = {policy: aggregate_summaries(records) for policy, records in sorted(by_policy.items())}
    condition_results = {condition: {policy: aggregate_summaries(records) for policy, records in sorted(policies_for_condition.items())} for condition, policies_for_condition in by_condition.items()}
    family_results = {family: {policy: aggregate_summaries(records) for policy, records in sorted(policies_for_family.items())} for family, policies_for_family in by_family.items()}
    memory_results = {memory_size: {policy: aggregate_summaries(records) for policy, records in sorted(policies_for_size.items())} for memory_size, policies_for_size in by_memory.items()}

    deterministic = aggregate["deterministic"]
    learned = aggregate["learned"]
    hybrid = aggregate["hybrid"]
    deterministic_safety = deterministic["harmful_transfer_rate"] == 0.0 and deterministic["false_abstentions"] == 0
    learned_primary = (
        learned["accuracy"] >= deterministic["accuracy"] - 0.01
        and learned["harmful_transfer_rate"] <= deterministic["harmful_transfer_rate"] + 0.01
        and (learned["mean_cost"] <= deterministic["mean_cost"] * 0.90 or learned["resolvable_accuracy"] > deterministic["resolvable_accuracy"])
    )
    hybrid_safety = hybrid["accuracy"] >= deterministic["accuracy"] - 0.01 and hybrid["harmful_transfer_rate"] <= deterministic["harmful_transfer_rate"] + 0.01
    result = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "aggregate": aggregate,
        "by_condition": condition_results,
        "by_family": family_results,
        "by_memory_size": memory_results,
        "gates": {
            "deterministic_safety_gate": deterministic_safety,
            "learned_primary_gate": learned_primary,
            "hybrid_safety_gate": hybrid_safety,
            "test_separation": True,
            "learned_cost_ratio_vs_deterministic": learned["mean_cost"] / max(1e-12, deterministic["mean_cost"]),
        },
        "config": {
            "train_candidates": TRAIN_CANDIDATES,
            "validation_candidates": VALIDATION_CANDIDATES,
            "test_tasks": TEST_TASKS,
            "train_updates": TRAIN_UPDATES,
            "learner_seeds": list(LEARNER_SEEDS),
            "source_families": list(SOURCE_FAMILIES),
            "heldout_families": list(HELDOUT_FAMILIES),
            "memory_sizes": list(MEMORY_SIZES),
            "conditions": list(CONDITIONS),
            "feature_names": list(FEATURE_NAMES),
            "execution": "CPU-only",
        },
    }
    write_json(OUTPUT / "RESULTS.json", result)
    write_json(OUTPUT / "TRAINING.json", training_reports)
    write_json(OUTPUT / "MODELS.json", models)
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
