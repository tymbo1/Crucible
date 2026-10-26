#!/usr/bin/env python3
"""CPU-only C1i learned-versus-deterministic applicability experiment."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_crucible_c1h_heldout_generalization_20261021 import (
    CONDITIONS,
    FAMILIES,
    MemoryRecord,
    MemoryStore,
    Task,
    make_task,
    matches_probe,
    record_for,
    solve,
    stable_int,
)


OUTPUT = ROOT / "experiments" / "crucible_c1i_learned_applicability_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
EXPERIMENT = "crucible_c1i_learned_applicability_20261021"
LEARNER_SEEDS = (42, 1337, 2026)
TRAIN_CANDIDATES = 6000
VALIDATION_CANDIDATES = 1500
TEST_TASKS = 1000
TRAIN_UPDATES = 3000
PROBE_COST = 0.8
VERIFY_COST = 0.5
FEATURE_PROBE_COST = 0.1
FEATURE_NAMES = (
    "family_vector",
    "family_grid",
    "family_graph",
    "family_set",
    "variant_match",
    "surface_match",
    "structural_match",
    "environment_match",
    "probe_available",
    "verifier_available",
    "probe_agrees",
    "confidence",
    "provenance_valid",
)


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
class Candidate:
    split: str
    index: int
    task: Task
    record: MemoryRecord
    features: Tuple[float, ...]
    label: int


@dataclass
class LogisticApplicability:
    weights: List[float]
    bias: float
    updates: int
    training_curve: List[Dict[str, float]]

    @classmethod
    def initialize(cls, seed: int, feature_count: int) -> "LogisticApplicability":
        weights = [((stable_int("c1i-init", seed, index) % 200) - 100) / 100000.0 for index in range(feature_count)]
        return cls(weights=weights, bias=0.0, updates=0, training_curve=[])

    def probability(self, features: Sequence[float]) -> float:
        value = self.bias + sum(weight * feature for weight, feature in zip(self.weights, features))
        value = max(-40.0, min(40.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def train(self, candidates: Sequence[Candidate], seed: int) -> None:
        learning_rate = 0.08
        l2 = 0.0005
        for update in range(1, TRAIN_UPDATES + 1):
            index = stable_int("c1i-order", seed, update) % len(candidates)
            candidate = candidates[index]
            probability = self.probability(candidate.features)
            error = probability - candidate.label
            for feature_index, feature in enumerate(candidate.features):
                self.weights[feature_index] -= learning_rate * (error * feature + l2 * self.weights[feature_index])
            self.bias -= learning_rate * error
            self.updates = update
            if update in {0, 500, 1000, 1500, 2000, 2500, 3000}:
                sample_loss = -(candidate.label * math.log(max(probability, 1e-12)) + (1 - candidate.label) * math.log(max(1 - probability, 1e-12)))
                self.training_curve.append({"update": float(update), "sample_loss": float(sample_loss)})


def source_record(seed: int, family: str) -> MemoryRecord:
    warmup = make_task(seed, family, "warmup", 0, phase="warmup")
    return record_for(warmup, warmup.true_rule, "validated", "source_warmup", {"source_seed": seed})


def feature_vector(task: Task, record: MemoryRecord) -> Tuple[float, ...]:
    probe_agrees = float(task.probe_available and matches_probe(task, record.rule))
    family_features = [float(task.family == family) for family in FAMILIES]
    return tuple(family_features + [
        float(task.variant == record.variant),
        float(task.surface_group == record.surface_group),
        float(task.structural_signature == record.structural_signature),
        float(task.environment_version == record.environment_version),
        float(task.probe_available),
        float(task.verifier_available),
        probe_agrees,
        float(record.confidence),
        float(record.validation_status == "validated" and bool(record.provenance)),
    ])


def label_for(task: Task, record: MemoryRecord) -> int:
    if not task.probe_available:
        return 0
    return int(solve(record.rule, task.family, task.target_input) == task.target_output)


def generate_candidates(split: str, count: int, seed: int) -> List[Candidate]:
    candidates: List[Candidate] = []
    for index in range(count):
        task_seed = stable_int("c1i-task", split, seed, index)
        family = FAMILIES[stable_int("c1i-family", split, seed, index) % len(FAMILIES)]
        condition = CONDITIONS[stable_int("c1i-condition", split, seed, index) % len(CONDITIONS)]
        task = make_task(task_seed, family, condition, index)
        record = source_record(task_seed, family)
        candidates.append(Candidate(split, index, task, record, feature_vector(task, record), label_for(task, record)))
    return candidates


def evaluate_model(model: LogisticApplicability, candidates: Sequence[Candidate]) -> Dict[str, float]:
    probabilities = [model.probability(candidate.features) for candidate in candidates]
    brier = sum((probability - candidate.label) ** 2 for probability, candidate in zip(probabilities, candidates)) / max(1, len(candidates))
    accuracy = sum((probability >= 0.5) == bool(candidate.label) for probability, candidate in zip(probabilities, candidates)) / max(1, len(candidates))
    return {"accuracy": accuracy, "brier": brier, "mean_probability": sum(probabilities) / max(1, len(probabilities))}


def investigate(task: Task) -> Tuple[Optional[str], float, str]:
    if not task.probe_available:
        return None, 0.0, "abstain"
    cost = PROBE_COST
    matches = [rule for rule in task.candidate_rules if matches_probe(task, rule)]
    candidates = matches if len(matches) == 1 else list(task.candidate_rules)
    for candidate in candidates:
        cost += VERIFY_COST
        if solve(candidate, task.family, task.target_input) == task.target_output:
            return candidate, cost, "investigate"
    return None, cost, "abstain"


def generic_record(task: Task) -> MemoryRecord:
    return source_record(stable_int("c1i-task-source", task.task_id), task.family)


def deterministic_policy(task: Task, record: MemoryRecord) -> Tuple[Optional[str], float, str, str]:
    exact = record.structural_signature == task.structural_signature and record.environment_version == task.environment_version
    related = record.structural_signature == task.structural_signature
    if exact or related:
        if task.probe_available and matches_probe(task, record.rule):
            if related and not exact:
                cost = VERIFY_COST
                if solve(record.rule, task.family, task.target_input) == task.target_output:
                    return record.rule, cost, "adapt", "related"
            elif exact:
                return record.rule, 0.0, "reuse", "exact"
    selected, cost, decision = investigate(task)
    return selected, cost, decision, "none"


def learned_policy(task: Task, record: MemoryRecord, model: LogisticApplicability) -> Tuple[Optional[str], float, str, str, float]:
    probability = model.probability(feature_vector(task, record))
    feature_cost = FEATURE_PROBE_COST if task.probe_available else 0.0
    if probability >= 0.5:
        return record.rule, feature_cost, "reuse", "learned", probability
    selected, cost, decision = investigate(task)
    return selected, feature_cost + cost, decision, "none", probability


def hybrid_policy(task: Task, record: MemoryRecord, model: LogisticApplicability) -> Tuple[Optional[str], float, str, str, float]:
    probability = model.probability(feature_vector(task, record))
    feature_cost = FEATURE_PROBE_COST if task.probe_available else 0.0
    hard_evidence = task.probe_available and matches_probe(task, record.rule)
    if probability >= 0.5 and hard_evidence:
        exact = record.structural_signature == task.structural_signature and record.environment_version == task.environment_version
        if exact:
            return record.rule, feature_cost, "reuse", "hybrid", probability
        if record.structural_signature == task.structural_signature:
            if solve(record.rule, task.family, task.target_input) == task.target_output:
                return record.rule, feature_cost + VERIFY_COST, "adapt", "hybrid", probability
    selected, cost, decision = investigate(task)
    return selected, feature_cost + cost, decision, "none", probability


def persistent_policy(task: Task, record: MemoryRecord) -> Tuple[Optional[str], float, str, str]:
    if record.variant == task.variant:
        return record.rule, 0.0, "reuse", "persistent"
    selected, cost, decision = investigate(task)
    return selected, cost, decision, "none"


def run_policy(tasks: Sequence[Candidate], policy: str, model: Optional[LogisticApplicability], seed: int) -> Dict[str, Any]:
    outcomes: List[Dict[str, Any]] = []
    counters = Counter()
    action_counts = Counter()
    brier_terms: List[float] = []
    for candidate in tasks:
        task = candidate.task
        record = candidate.record
        probability: Optional[float] = None
        if policy == "deterministic":
            selected, cost, decision, source = deterministic_policy(task, record)
        elif policy == "learned":
            assert model is not None
            selected, cost, decision, source, probability = learned_policy(task, record, model)
        elif policy == "hybrid":
            assert model is not None
            selected, cost, decision, source, probability = hybrid_policy(task, record, model)
        elif policy == "persistent":
            selected, cost, decision, source = persistent_policy(task, record)
        elif policy == "stateless":
            selected, cost, decision = investigate(task)
            source = "none"
        elif policy == "oracle":
            selected, cost, decision, source = task.true_rule, 0.0, "oracle", "none"
        else:
            raise ValueError(policy)
        correct = selected is not None and solve(selected, task.family, task.target_input) == task.target_output
        used_memory = source != "none" and decision in {"reuse", "adapt"}
        harmful = used_memory and not correct
        action_counts[decision] += 1
        counters["tasks"] += 1
        counters["correct"] += int(correct)
        counters["cost"] += cost
        counters["harmful"] += int(harmful)
        counters["retrievals"] += int(source != "none" or policy in {"deterministic", "learned", "hybrid", "persistent"})
        counters["applicable"] += int(task.condition in {"novel_surface", "adaptation"})
        counters["reused_applicable"] += int(task.condition in {"novel_surface", "adaptation"} and decision in {"reuse", "adapt"})
        counters["adaptation_attempts"] += int(decision == "adapt")
        counters["adaptation_successes"] += int(decision == "adapt" and correct)
        counters["abstentions"] += int(decision == "abstain")
        counters["safe_abstentions"] += int(decision == "abstain" and task.condition == "unresolvable")
        counters["false_abstentions"] += int(decision == "abstain" and task.condition != "unresolvable")
        if task.condition == "novel_structure":
            counters["novel_structure_tasks"] += 1
            counters["novel_structure_correct"] += int(correct)
        if probability is not None:
            brier_terms.append((probability - candidate.label) ** 2)
        outcomes.append({
            "split": candidate.split,
            "index": candidate.index,
            "task_id": task.task_id,
            "family": task.family,
            "condition": task.condition,
            "true_rule": task.true_rule,
            "selected_rule": selected,
            "decision": decision,
            "source": source,
            "correct": correct,
            "harmful_transfer": harmful,
            "cost": cost,
            "probability": probability,
        })
    tasks_count = counters["tasks"]
    return {
        "policy": policy,
        "seed": seed,
        "tasks": tasks_count,
        "accuracy": counters["correct"] / max(1, tasks_count),
        "mean_cost": counters["cost"] / max(1, tasks_count),
        "novel_structure_accuracy": counters["novel_structure_correct"] / max(1, counters["novel_structure_tasks"]),
        "reuse_coverage": counters["reused_applicable"] / max(1, counters["applicable"]),
        "adaptation_attempts": counters["adaptation_attempts"],
        "adaptation_successes": counters["adaptation_successes"],
        "adaptation_success_rate": counters["adaptation_successes"] / max(1, counters["adaptation_attempts"]),
        "harmful_transfer_rate": counters["harmful"] / max(1, counters["retrievals"]),
        "abstentions": counters["abstentions"],
        "abstention_precision": counters["safe_abstentions"] / max(1, counters["abstentions"]),
        "false_abstentions": counters["false_abstentions"],
        "brier": sum(brier_terms) / max(1, len(brier_terms)),
        "actions": dict(sorted(action_counts.items())),
        "outcomes": outcomes,
    }


def aggregate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    numeric = ("accuracy", "mean_cost", "novel_structure_accuracy", "reuse_coverage", "adaptation_success_rate", "harmful_transfer_rate", "abstention_precision", "brier")
    counts = ("tasks", "adaptation_attempts", "adaptation_successes", "abstentions", "false_abstentions")
    return {
        "runs": len(records),
        **{field: sum(float(record.get(field, 0.0)) for record in records) / len(records) for field in numeric},
        **{field: sum(int(record.get(field, 0)) for record in records) for field in counts},
    }


def condition_aggregate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selected = [record for record in records]
    return aggregate(selected)


def source_hashes() -> Dict[str, str]:
    paths = [
        PROTOCOL,
        ROOT / "tools" / "run_crucible_c1h_heldout_generalization_20261021.py",
        ROOT / "tools" / "run_crucible_c1i_learned_applicability_20261021.py",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def render_summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1i learned versus deterministic applicability",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; no GPU, cloud or Crucible neural training.",
        "",
        "## Aggregate test results",
        "",
        "| Policy | Accuracy | Novel-structure accuracy | Reuse coverage | Harmful transfer | Abstention precision | False abstentions | Brier | Mean cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, item in result["aggregate"].items():
        lines.append(f"| `{policy}` | {item['accuracy']:.4f} | {item['novel_structure_accuracy']:.4f} | {item['reuse_coverage']:.4f} | {item['harmful_transfer_rate']:.4f} | {item['abstention_precision']:.4f} | {item['false_abstentions']} | {item['brier']:.4f} | {item['mean_cost']:.3f} |")
    lines.extend(["", "## Condition results", ""])
    for condition, policies in result["by_condition"].items():
        lines.extend([f"### {condition}", "", "| Policy | Accuracy | Reuse coverage | Harmful transfer | Abstention precision | Mean cost |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for policy, item in policies.items():
            lines.append(f"| `{policy}` | {item['accuracy']:.4f} | {item['reuse_coverage']:.4f} | {item['harmful_transfer_rate']:.4f} | {item['abstention_precision']:.4f} | {item['mean_cost']:.3f} |")
        lines.append("")
    lines.extend([
        "## Learner and gates",
        "",
        f"- Training candidates: `{result['config']['train_candidates']}`.",
        f"- Validation candidates: `{result['config']['validation_candidates']}`.",
        f"- Untouched test tasks: `{result['config']['test_tasks']}`.",
        f"- Updates per learner seed: `{result['config']['train_updates']}`.",
        f"- Primary deterministic-contract gate: `{result['gates']['deterministic_gate']}`.",
        f"- Learned safety gate: `{result['gates']['learned_gate']}`.",
        f"- Hybrid safety/cost gate: `{result['gates']['hybrid_gate']}`.",
        f"- Learned policy reported without test retuning: `{result['gates']['test_separation']}`.",
        "",
        "A failed learned or hybrid gate is a valid result; the deterministic controller remains the reference.",
        "",
        "## Source hashes",
        "",
    ])
    for path, digest in sorted(hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = generate_candidates("train", TRAIN_CANDIDATES, 101)
    validation = generate_candidates("validation", VALIDATION_CANDIDATES, 202)
    test = generate_candidates("test", TEST_TASKS, 303)
    split_manifest = {
        "train": {"count": len(train), "task_digest": hashlib.sha256("|".join(candidate.task.task_id for candidate in train).encode()).hexdigest()},
        "validation": {"count": len(validation), "task_digest": hashlib.sha256("|".join(candidate.task.task_id for candidate in validation).encode()).hexdigest()},
        "test": {"count": len(test), "task_digest": hashlib.sha256("|".join(candidate.task.task_id for candidate in test).encode()).hexdigest()},
        "feature_names": list(FEATURE_NAMES),
        "test_labels_not_used_as_features": True,
    }
    write_json(OUTPUT / "SPLIT_MANIFEST.json", split_manifest)
    models: Dict[str, Dict[str, Any]] = {}
    all_runs: List[Dict[str, Any]] = []
    aggregate_runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    condition_runs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {condition: defaultdict(list) for condition in CONDITIONS}
    training_reports: Dict[str, Any] = {}
    policies = ("deterministic", "learned", "hybrid", "persistent", "stateless", "oracle")
    for seed in LEARNER_SEEDS:
        model = LogisticApplicability.initialize(seed, len(FEATURE_NAMES))
        model.train(train, seed)
        validation_metrics = evaluate_model(model, validation)
        test_metrics = evaluate_model(model, test)
        models[str(seed)] = {"weights": model.weights, "bias": model.bias, "updates": model.updates}
        training_reports[str(seed)] = {"validation": validation_metrics, "test": test_metrics, "curve": model.training_curve}
        for policy in policies:
            run = run_policy(test, policy, model if policy in {"learned", "hybrid"} else None, seed)
            summary = {key: value for key, value in run.items() if key != "outcomes"}
            all_runs.append({"seed": seed, "summary": summary, "outcomes": run["outcomes"]})
            aggregate_runs[policy].append(summary)
            for condition in CONDITIONS:
                condition_outcomes = [outcome for outcome in run["outcomes"] if outcome["condition"] == condition]
                memory_items = [item for item in condition_outcomes if item["decision"] in {"reuse", "adapt"}]
                condition_runs[condition][policy].append({
                    "tasks": len(condition_outcomes),
                    "accuracy": sum(item["correct"] for item in condition_outcomes) / max(1, len(condition_outcomes)),
                    "reuse_coverage": sum(item["decision"] in {"reuse", "adapt"} for item in condition_outcomes) / max(1, len(condition_outcomes)),
                    "harmful_transfer_rate": sum(item["harmful_transfer"] for item in condition_outcomes) / max(1, len(memory_items)),
                    "abstention_precision": sum(item["decision"] == "abstain" and condition == "unresolvable" for item in condition_outcomes) / max(1, sum(item["decision"] == "abstain" for item in condition_outcomes)),
                    "mean_cost": sum(float(item["cost"]) for item in condition_outcomes) / max(1, len(condition_outcomes)),
                })
    aggregate_results = {policy: aggregate(records) for policy, records in sorted(aggregate_runs.items())}
    aggregate_conditions = {condition: {policy: aggregate(records) for policy, records in sorted(policies_for_condition.items())} for condition, policies_for_condition in condition_runs.items()}
    deterministic = aggregate_results["deterministic"]
    learned = aggregate_results["learned"]
    hybrid = aggregate_results["hybrid"]
    deterministic_harm = sum(aggregate_conditions[condition]["deterministic"]["harmful_transfer_rate"] for condition in ("invalidated", "contradictory", "misleading_similarity")) / 3
    hybrid_harm = sum(aggregate_conditions[condition]["hybrid"]["harmful_transfer_rate"] for condition in ("invalidated", "contradictory", "misleading_similarity")) / 3
    deterministic_gate = deterministic["accuracy"] >= aggregate_results["stateless"]["accuracy"] and deterministic_harm == 0.0
    learned_gate = learned["accuracy"] >= deterministic["accuracy"] and learned["harmful_transfer_rate"] <= deterministic["harmful_transfer_rate"]
    hybrid_gate = hybrid["accuracy"] >= deterministic["accuracy"] and hybrid_harm <= deterministic_harm and (hybrid["mean_cost"] <= deterministic["mean_cost"] * 0.90 or aggregate_conditions["unresolvable"]["hybrid"]["accuracy"] > aggregate_conditions["unresolvable"]["deterministic"]["accuracy"])
    result = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "aggregate": aggregate_results,
        "by_condition": aggregate_conditions,
        "gates": {
            "deterministic_gate": deterministic_gate,
            "learned_gate": learned_gate,
            "hybrid_gate": hybrid_gate,
            "test_separation": True,
            "deterministic_harmful_transfer": deterministic_harm,
            "hybrid_harmful_transfer": hybrid_harm,
        },
        "config": {
            "train_candidates": TRAIN_CANDIDATES,
            "validation_candidates": VALIDATION_CANDIDATES,
            "test_tasks": TEST_TASKS,
            "train_updates": TRAIN_UPDATES,
            "learner_seeds": list(LEARNER_SEEDS),
            "feature_names": list(FEATURE_NAMES),
            "execution": "CPU-only",
        },
    }
    write_json(OUTPUT / "RESULTS.json", result)
    write_json(OUTPUT / "TRAINING.json", training_reports)
    write_json(OUTPUT / "MODELS.json", models)
    with (OUTPUT / "RUNS.jsonl").open("w", encoding="utf-8") as handle:
        for run in all_runs:
            handle.write(json.dumps(run, sort_keys=True) + "\n")
    hashes = source_hashes()
    (OUTPUT / "RESULTS_SUMMARY.md").write_text(render_summary(result, hashes), encoding="utf-8")
    file_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path.name != "HASHES.json"}
    write_json(OUTPUT / "HASHES.json", file_hashes)
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "runs": len(all_runs), "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
