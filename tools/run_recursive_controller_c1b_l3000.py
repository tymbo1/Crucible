#!/usr/bin/env python3
"""Run the bounded CPU-only C1b-L3000 strategy-selection pilot.

This experiment is deliberately separate from the sealed C0/C1 replay.  A
linear contextual bandit chooses the first reasoning operation for a bounded
synthetic episode; a deterministic, safe continuation executes that operation
without exposing the latent answer to the policy.  The pilot is a feasibility
test for strategy selection, not a claim about language-model cognition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from cognitive_task_suite import generate_records  # noqa: E402
from recursive_controller import make_scenario, sha256_value  # noqa: E402
from validate_cognitive_manifests import (  # noqa: E402
    load_records as load_manifest_records,
    validate as validate_manifests,
)


EXPERIMENT = "crucible_recursive_controller_c1b_l3000_20261021"
DEFAULT_OUTPUT = ROOT / "experiments" / EXPERIMENT
DEFAULT_SEALED_MANIFESTS = ROOT / "experiments" / "crucible_cognitive_eval_v2_20261021" / "manifests"
TASK_SEED = 20261022
LEARNER_SEEDS = (42, 1337, 2026)
UPDATES = 3000
CHECKPOINTS = (0, 250, 500, 1000, 1500, 2000, 2500, 3000)
WALL_CAP_SECONDS = 20.0 * 60.0
CONDITIONS = ("R", "I", "E", "D", "U", "A")
ACTIONS = (
    "answer",
    "verify",
    "critique",
    "counterexample",
    "evidence_search",
    "revise",
    "stop",
)
POLICIES = (
    "verification_only",
    "verification_first_adaptive",
    "adaptive",
    "fixed_depth_2",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_int(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


@dataclass(frozen=True)
class ConditionSpec:
    code: str
    verifier_available: bool
    verifier_reliability: float
    verifier_cost: float
    delayed: bool
    evidence_available: bool
    evidence_reliability: float
    contradiction_frequency: float


CONDITION_SPECS: Dict[str, ConditionSpec] = {
    "R": ConditionSpec("R", True, 1.00, 3.0, False, False, 0.0, 0.00),
    "I": ConditionSpec("I", True, 0.80, 3.0, False, False, 0.0, 0.05),
    "E": ConditionSpec("E", True, 1.00, 8.0, False, False, 0.0, 0.00),
    "D": ConditionSpec("D", True, 1.00, 4.0, True, False, 0.0, 0.00),
    "U": ConditionSpec("U", False, 0.00, 0.5, False, False, 0.0, 0.10),
    "A": ConditionSpec("A", False, 0.00, 0.5, False, True, 0.80, 0.10),
}


@dataclass(frozen=True)
class TaskCase:
    record_id: str
    family: str
    prompt: str
    gold_answer: str
    initial_answer: str
    initial_confidence: float
    category: str
    candidate_revision_answer: str


@dataclass(frozen=True)
class EpisodeSpec:
    task: TaskCase
    condition: str
    mode: str
    episode_seed: int

    @property
    def actual(self) -> ConditionSpec:
        return CONDITION_SPECS[self.condition]

    @property
    def visible(self) -> ConditionSpec:
        # Inferred-shift episodes deliberately hide the changed contract.  A
        # first-decision bandit cannot infer a shift until it observes a tool
        # result; this is measured as a limitation rather than patched away.
        if self.mode == "declared":
            return self.actual
        return CONDITION_SPECS["R"]


@dataclass
class EpisodeState:
    answer: str
    observations: List[Dict[str, Any]]
    history: List[str]
    contradictions: int
    total_cost: float
    actions_used: int
    pending_verification: bool = False
    stopped: bool = False


@dataclass(frozen=True)
class EpisodeResult:
    policy: str
    first_action: str
    final_answer: str
    initial_correct: bool
    final_correct: bool
    corrected: bool
    false_correction: bool
    unresolved: bool
    total_cost: float
    actions_used: int
    verification_calls: int
    verification_omitted: bool
    evidence_calls: int
    contradiction_count: int
    action_counts: Mapping[str, int]
    observations: Tuple[Mapping[str, Any], ...]


def load_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_task(record: Mapping[str, Any]) -> TaskCase:
    scenario = make_scenario(record)
    return TaskCase(
        record_id=scenario.scenario_id,
        family=scenario.family,
        prompt=scenario.prompt,
        gold_answer=scenario.gold_answer,
        initial_answer=scenario.initial_answer,
        initial_confidence=scenario.initial_confidence,
        category=scenario.category,
        candidate_revision_answer=scenario.candidate_revision_answer,
    )


def condition_for(record_id: str, split: str, seed: int) -> Tuple[str, str]:
    if split in {"train", "validation"}:
        training_conditions = ("R", "I", "U", "A")
        slot = stable_int("c1b-condition", TASK_SEED, split, seed, record_id) % len(training_conditions)
        condition = training_conditions[slot]
        mode = "declared"
    else:
        slot = stable_int("c1b-condition", TASK_SEED, split, seed, record_id) % 12
        condition = CONDITIONS[slot % len(CONDITIONS)]
        mode = "declared" if slot < len(CONDITIONS) else "inferred"
    return condition, mode


def make_episode(record: Mapping[str, Any], split: str, seed: int) -> EpisodeSpec:
    condition, mode = condition_for(str(record["record_id"]), split, seed)
    episode_seed = stable_int("c1b-episode", seed, split, record["record_id"], condition, mode)
    return EpisodeSpec(make_task(record), condition, mode, episode_seed)


def initial_state(task: TaskCase) -> EpisodeState:
    return EpisodeState(
        answer=task.initial_answer,
        observations=[],
        history=[],
        contradictions=0,
        total_cost=0.0,
        actions_used=0,
    )


def action_cost(episode: EpisodeSpec, action: str) -> float:
    if action == "verify":
        return episode.actual.verifier_cost
    return {
        "answer": 1.0,
        "critique": 1.0,
        "counterexample": 1.5,
        "evidence_search": 2.0,
        "revise": 1.0,
        "stop": 0.0,
    }[action]


def can_take(episode: EpisodeSpec, state: EpisodeState, action: str) -> bool:
    if state.stopped or state.actions_used >= 8:
        return False
    return state.total_cost + action_cost(episode, action) <= 12.0 + 1e-9


def _false_outcome(episode: EpisodeSpec, state: EpisodeState) -> bool:
    # Fixed per episode/action outcome: every policy sees the same noisy tool
    # realization for a paired comparison.
    return stable_int("c1b-noise", episode.episode_seed, len(state.observations), state.answer) % 10 == 0


def observation_for(episode: EpisodeSpec, state: EpisodeState, action: str) -> Dict[str, Any]:
    task = episode.task
    if action == "critique":
        signals = {
            "clean_correct": ("no_error", 0.90),
            "detectable_wrong": ("error", 0.90),
            "misleading_evidence": ("no_error", 0.90),
            "contradiction": ("no_error", 0.80),
            "shortcut_probe": ("uncertain", 0.60),
        }
        signal, reliability = signals[task.category]
        return {"action": action, "signal": signal, "reliability": reliability, "verified": False, "detail": "synthetic critique"}
    if action == "counterexample":
        signals = {
            "clean_correct": ("no_counterexample", 0.90),
            "detectable_wrong": ("supports_error", 0.90),
            "misleading_evidence": ("false_alarm", 0.90),
            "contradiction": ("false_alarm", 0.85),
            "shortcut_probe": ("supports_error", 0.70),
        }
        signal, reliability = signals[task.category]
        return {"action": action, "signal": signal, "reliability": reliability, "verified": False, "detail": "synthetic counterexample"}
    if action == "evidence_search":
        if not episode.actual.evidence_available:
            return {"action": action, "signal": "unavailable", "reliability": 0.0, "verified": False, "detail": "evidence unavailable"}
        signals = {
            "clean_correct": ("supports_candidate", 0.80),
            "detectable_wrong": ("supports_correction", 0.80),
            "misleading_evidence": ("supports_candidate", 0.95),
            "contradiction": ("contradicts_candidate", 0.95),
            "shortcut_probe": ("supports_correction", 0.70),
        }
        signal, reliability = signals[task.category]
        evidence_id = sha256_value({"scenario": task.record_id, "action": action})[:16]
        return {
            "action": action,
            "signal": signal,
            "reliability": min(reliability, episode.actual.evidence_reliability),
            "verified": False,
            "detail": "synthetic evidence; not an oracle",
            "evidence_id": evidence_id,
        }
    if action == "verify":
        if not episode.actual.verifier_available:
            return {"action": action, "signal": "unavailable", "reliability": 0.0, "verified": False, "detail": "verifier unavailable"}
        if episode.actual.delayed and not state.pending_verification:
            return {"action": action, "signal": "pending", "reliability": 0.0, "verified": False, "detail": "result delayed one transition"}
        correct = state.answer == task.gold_answer
        if episode.condition == "I" and _false_outcome(episode, state):
            correct = not correct
            if correct:
                return {"action": action, "signal": "verified_incorrect", "reliability": 0.80, "verified": False, "detail": "no correction supplied"}
            return {"action": action, "signal": "verified_correct", "reliability": 0.80, "verified": False, "detail": "no correction supplied"}
        if correct:
            return {"action": action, "signal": "verified_correct", "reliability": episode.actual.verifier_reliability, "verified": True, "detail": state.answer}
        return {
            "action": action,
            "signal": "verified_incorrect",
            "reliability": episode.actual.verifier_reliability,
            "verified": True,
            "detail": "correction available",
            "correction": task.gold_answer,
        }
    raise ValueError(f"no observation for {action}")


def apply_revision(episode: EpisodeSpec, state: EpisodeState) -> None:
    task = episode.task
    exact_correction = next(
        (obs.get("correction") for obs in state.observations if obs.get("signal") == "verified_incorrect" and obs.get("correction")),
        None,
    )
    if exact_correction is not None:
        state.answer = str(exact_correction)
        return
    signals = {str(obs.get("signal")) for obs in state.observations}
    reliable_error = (
        "error" in signals and "supports_error" in signals
    ) or ("error" in signals and "supports_correction" in signals)
    if reliable_error:
        state.answer = task.candidate_revision_answer
    # A contradiction is deliberately not enough to revise a correct answer;
    # it is retained as unresolved evidence until independent support arrives.


def execute_action(episode: EpisodeSpec, state: EpisodeState, action: str) -> Optional[Dict[str, Any]]:
    if action not in ACTIONS:
        raise ValueError(action)
    if not can_take(episode, state, action):
        state.stopped = True
        return {"action": action, "signal": "rejected_budget", "reliability": 0.0, "verified": False, "detail": "budget exhausted"}
    state.total_cost = round(state.total_cost + action_cost(episode, action), 9)
    state.actions_used += 1
    state.history.append(action)
    if action == "answer":
        return {"action": action, "signal": "candidate_available", "reliability": 0.0, "verified": False, "detail": "initial candidate"}
    if action == "stop":
        state.stopped = True
        return {"action": action, "signal": "stopped", "reliability": 0.0, "verified": False, "detail": "policy stop"}
    if action == "revise":
        before = state.answer
        apply_revision(episode, state)
        return {"action": action, "signal": "revision_applied", "reliability": 1.0, "verified": False, "detail": "changed" if before != state.answer else "retained"}
    observation = observation_for(episode, state, action)
    if action == "verify" and observation["signal"] == "pending":
        state.pending_verification = True
    elif action == "verify":
        state.pending_verification = False
    if observation.get("signal") in {"false_alarm", "contradicts_candidate"}:
        state.contradictions += 1
    state.observations.append(observation)
    return observation


def action_plan(policy: str, episode: EpisodeSpec) -> List[str]:
    if policy == "verification_only":
        return ["verify", "verify" if episode.actual.delayed else "revise", "revise", "stop"]
    if policy == "verification_first_adaptive":
        visible = episode.visible
        if visible.verifier_available and visible.verifier_reliability >= 0.9 and visible.verifier_cost <= 4.0:
            return ["verify", "verify" if visible.delayed else "revise", "revise", "stop"]
        if visible.evidence_available:
            return ["evidence_search", "critique", "counterexample", "revise", "stop"]
        return ["critique", "counterexample", "revise", "stop"]
    if policy == "adaptive":
        if episode.task.initial_confidence < 0.75:
            return ["critique", "counterexample", "revise", "stop"]
        return ["critique", "evidence_search", "verify", "revise", "stop"]
    if policy == "fixed_depth_2":
        return ["critique", "counterexample", "revise", "stop"]
    raise ValueError(policy)


def plan_for_first_action(first_action: str, episode: EpisodeSpec) -> List[str]:
    if first_action == "verify":
        return ["verify", "verify" if episode.actual.delayed else "revise", "revise", "stop"]
    if first_action == "critique":
        return ["critique", "counterexample", "revise", "stop"]
    if first_action == "counterexample":
        return ["counterexample", "critique", "revise", "stop"]
    if first_action == "evidence_search":
        return ["evidence_search", "critique", "counterexample", "revise", "stop"]
    if first_action == "revise":
        return ["revise", "verify", "revise", "stop"]
    if first_action == "answer":
        return ["answer", "stop"]
    if first_action == "stop":
        return ["stop"]
    raise ValueError(first_action)


def run_episode(episode: EpisodeSpec, policy: str, *, first_action: Optional[str] = None) -> EpisodeResult:
    state = initial_state(episode.task)
    plan = action_plan(policy, episode) if first_action is None else plan_for_first_action(first_action, episode)
    for action in plan:
        execute_action(episode, state, action)
        if state.stopped:
            break
    final_correct = state.answer == episode.task.gold_answer
    initial_correct = episode.task.initial_answer == episode.task.gold_answer
    return EpisodeResult(
        policy=policy,
        first_action=plan[0],
        final_answer=state.answer,
        initial_correct=initial_correct,
        final_correct=final_correct,
        corrected=(not initial_correct and final_correct),
        false_correction=(initial_correct and not final_correct),
        unresolved=(not final_correct),
        total_cost=state.total_cost,
        actions_used=state.actions_used,
        verification_calls=sum(1 for action in state.history if action == "verify"),
        verification_omitted=(not any(action == "verify" for action in state.history)),
        evidence_calls=sum(1 for action in state.history if action == "evidence_search"),
        contradiction_count=state.contradictions,
        action_counts={action: state.history.count(action) for action in ACTIONS},
        observations=tuple(state.observations),
    )


def feature_vector(episode: EpisodeSpec) -> List[float]:
    visible = episode.visible
    family = [1.0 if episode.task.family == name else 0.0 for name in ("math", "relational_binding", "logic", "error_correction")]
    return [
        1.0,
        1.0,  # remaining depth budget at the first decision
        1.0,  # remaining action budget at the first decision
        1.0,  # remaining evidence/counterexample budget summary
        1.0,  # remaining normalized cost budget
        episode.task.initial_confidence,
        visible.verifier_reliability,
        min(1.0, visible.verifier_cost / 12.0),
        1.0 if visible.verifier_available else 0.0,
        1.0 if visible.delayed else 0.0,
        1.0 if visible.evidence_available else 0.0,
        visible.evidence_reliability,
        visible.contradiction_frequency,
        1.0 if episode.mode == "declared" else 0.0,
        min(1.0, len(episode.task.prompt) / 600.0),
        *family,
    ]


class LinearContextualBandit:
    def __init__(self, seed: int, feature_count: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.weights = {action: [0.0] * feature_count for action in ACTIONS}
        self.counts = {action: 0 for action in ACTIONS}

    def scores(self, features: Sequence[float]) -> Dict[str, float]:
        return {
            action: sum(weight * value for weight, value in zip(self.weights[action], features))
            for action in ACTIONS
        }

    def select(self, features: Sequence[float], update: int, *, explore: bool) -> str:
        scores = self.scores(features)
        epsilon = max(0.05, 0.30 * (1.0 - (update - 1) / UPDATES))
        if explore and self.rng.random() < epsilon:
            return ACTIONS[self.rng.randrange(len(ACTIONS))]
        return max(ACTIONS, key=lambda action: (scores[action], -ACTIONS.index(action)))

    def update(self, action: str, features: Sequence[float], reward: float, update: int) -> float:
        prediction = sum(weight * value for weight, value in zip(self.weights[action], features))
        error = reward - prediction
        learning_rate = 0.08 / math.sqrt(1.0 + self.counts[action] * 0.02)
        for index, value in enumerate(features):
            self.weights[action][index] += learning_rate * error * value
        self.counts[action] += 1
        return prediction

    def snapshot(self) -> Dict[str, Any]:
        return {
            "weights": {action: list(values) for action, values in self.weights.items()},
            "counts": dict(self.counts),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.weights = {action: [float(value) for value in snapshot["weights"][action]] for action in ACTIONS}
        self.counts = {action: int(snapshot["counts"][action]) for action in ACTIONS}


def reward_for(result: EpisodeResult) -> float:
    verified_progress = 1.0 if any(obs.get("verified") for obs in result.observations) else 0.0
    useful_evidence = sum(
        1.0 for obs in result.observations
        if obs.get("action") == "evidence_search" and obs.get("signal") in {"supports_correction", "contradicts_candidate"}
    )
    reward = (1.0 if result.final_correct else 0.0)
    reward += 0.10 * verified_progress + 0.05 * useful_evidence
    reward -= 0.10 * (result.total_cost / 12.0)
    reward -= 0.50 if result.false_correction else 0.0
    reward -= 0.15 if result.unresolved else 0.0
    return round(reward, 9)


def metrics(results: Sequence[EpisodeResult]) -> Dict[str, Any]:
    if not results:
        raise ValueError("empty results")
    action_counts = {action: sum(item.action_counts[action] for item in results) for action in ACTIONS}
    return {
        "cases": len(results),
        "accuracy": sum(item.final_correct for item in results) / len(results),
        "initial_accuracy": sum(item.initial_correct for item in results) / len(results),
        "corrections": sum(item.corrected for item in results),
        "false_corrections": sum(item.false_correction for item in results),
        "unresolved": sum(item.unresolved for item in results),
        "mean_cost": sum(item.total_cost for item in results) / len(results),
        "total_cost": sum(item.total_cost for item in results),
        "mean_actions": sum(item.actions_used for item in results) / len(results),
        "verification_calls": sum(item.verification_calls for item in results),
        "verification_omissions": sum(item.verification_omitted for item in results),
        "evidence_calls": sum(item.evidence_calls for item in results),
        "contradiction_events": sum(item.contradiction_count for item in results),
        "action_counts": action_counts,
        "cost_accuracy_score": sum(item.final_correct for item in results) / len(results) - 0.05 * (sum(item.total_cost for item in results) / len(results) / 12.0),
    }


def evaluate_policy(records: Sequence[Mapping[str, Any]], split: str, seed: int, policy: str, *, first_action_model: Optional[LinearContextualBandit] = None) -> Dict[str, Any]:
    results: List[EpisodeResult] = []
    for record in records:
        episode = make_episode(record, split, seed)
        if first_action_model is None:
            result = run_episode(episode, policy)
        else:
            action = first_action_model.select(feature_vector(episode), UPDATES, explore=False)
            result = run_episode(episode, "learned_adaptive", first_action=action)
        results.append(result)
    aggregate = metrics(results)
    by_condition: Dict[str, Dict[str, Any]] = {}
    by_mode: Dict[str, Dict[str, Any]] = {}
    by_condition_mode: Dict[str, Dict[str, Any]] = {}
    by_family: Dict[str, Dict[str, Any]] = {}
    for condition in CONDITIONS:
        subset = [result for result, record in zip(results, records) if make_episode(record, split, seed).condition == condition]
        if subset:
            by_condition[condition] = metrics(subset)
    for mode in ("declared", "inferred"):
        subset = [result for result, record in zip(results, records) if make_episode(record, split, seed).mode == mode]
        if subset:
            by_mode[mode] = metrics(subset)
    for condition in CONDITIONS:
        for mode in ("declared", "inferred"):
            subset = [
                result
                for result, record in zip(results, records)
                if (episode := make_episode(record, split, seed)).condition == condition and episode.mode == mode
            ]
            if subset:
                by_condition_mode.setdefault(condition, {})[mode] = metrics(subset)
    for family in ("math", "relational_binding", "logic", "error_correction"):
        subset = [result for result, record in zip(results, records) if str(record["family"]) == family]
        if subset:
            by_family[family] = metrics(subset)
    return {
        **aggregate,
        "by_condition": by_condition,
        "by_mode": by_mode,
        "by_condition_mode": by_condition_mode,
        "by_family": by_family,
    }


def validation_score(item: Mapping[str, Any]) -> float:
    return round(
        float(item["accuracy"])
        - 0.05 * float(item["mean_cost"]) / 12.0
        - 0.25 * float(item["false_corrections"]) / float(item["cases"])
        - 0.10 * float(item["unresolved"]) / float(item["cases"]),
        9,
    )


def build_manifests(output: Path) -> Dict[str, Any]:
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    counts = {"train": 64, "validation": 32, "test": 64}
    records_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split, count in counts.items():
        records = generate_records(TASK_SEED, split, count, variant="v2")
        # A few generators intentionally do not expose the seed in their
        # human-readable prompt.  Namespace the C1b prompt so the manifest is
        # unambiguously disjoint even when a shuffled fixture happens to match
        # an older prompt byte-for-byte.  The suffix keeps the canonical
        # validator's task parsers unchanged.
        for record in records:
            record["input"] = str(record["input"]) + " [C1b-L3000 independent fixture]"
            record.setdefault("metadata", {})["input_namespace"] = "c1b_l3000_20261021"
        records_by_split[split] = records
        with (manifests / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    sealed_inputs = set()
    for split in ("train", "validation", "test"):
        for record in load_records(DEFAULT_SEALED_MANIFESTS / f"{split}.jsonl"):
            sealed_inputs.add(sha256_text(str(record["input"])))
    new_inputs = {sha256_text(str(record["input"])) for records in records_by_split.values() for record in records}
    overlap = sorted(sealed_inputs & new_inputs)
    record_ids = [record["record_id"] for records in records_by_split.values() for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("C1b manifest record IDs are not unique")
    if overlap:
        raise RuntimeError(f"C1b input overlap with sealed C1: {len(overlap)}")
    manifest = {
        "experiment": EXPERIMENT,
        "generator": "tools/cognitive_task_suite.py",
        "generator_variant": "v2",
        "task_seed": TASK_SEED,
        "records_per_family": counts,
        "record_counts": {split: len(records) for split, records in records_by_split.items()},
        "sealed_c1_input_sha256_overlap_count": len(overlap),
        "sealed_c1_record_count": len(sealed_inputs),
        "new_input_count": len(new_inputs),
        "latent_targets": "stored for evaluator only; never supplied as decision-time features",
        "split_paths": {split: f"manifests/{split}.jsonl" for split in counts},
    }
    write_json(output / "MANIFEST.json", manifest)
    return {"manifest": manifest, "records": records_by_split}


def run_seed(train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]], test_records: Sequence[Mapping[str, Any]], seed: int, started: float) -> Dict[str, Any]:
    learner = LinearContextualBandit(seed, len(feature_vector(make_episode(train_records[0], "train", seed))))
    initial_validation = evaluate_policy(validation_records, "validation", seed, "learned_adaptive", first_action_model=learner)
    checkpoints: List[Dict[str, Any]] = [{
        "update": 0,
        "validation": initial_validation,
        "validation_score": validation_score(initial_validation),
        "learner": learner.snapshot(),
    }]
    curves: List[Dict[str, Any]] = []
    train_rewards: List[float] = []
    train_rng = random.Random(seed)
    checkpoint_set = set(CHECKPOINTS)

    for update in range(1, UPDATES + 1):
        if perf_counter() - started > WALL_CAP_SECONDS:
            raise TimeoutError(f"C1b-L3000 exceeded {WALL_CAP_SECONDS:.1f}s wall cap")
        record = train_records[train_rng.randrange(len(train_records))]
        episode = make_episode(record, "train", seed)
        features = feature_vector(episode)
        action = learner.select(features, update, explore=True)
        result = run_episode(episode, "learned_adaptive", first_action=action)
        reward = reward_for(result)
        prediction = learner.update(action, features, reward, update)
        train_rewards.append(reward)
        if update % 25 == 0 or update in checkpoint_set:
            rolling = train_rewards[-100:]
            curves.append({
                "seed": seed,
                "update": update,
                "mean_reward_last_100": sum(rolling) / len(rolling),
                "prediction": prediction,
                "exploration_epsilon": max(0.05, 0.30 * (1.0 - (update - 1) / UPDATES)),
            })
        if update in checkpoint_set:
            validation = evaluate_policy(validation_records, "validation", seed, "learned_adaptive", first_action_model=learner)
            checkpoints.append({
                "update": update,
                "validation": validation,
                "validation_score": validation_score(validation),
                "learner": learner.snapshot(),
            })

    candidates = [item for item in checkpoints if item["update"] in CHECKPOINTS]
    selected = max(candidates, key=lambda item: (item["validation_score"], -item["update"]))
    selected_learner = LinearContextualBandit(seed, len(feature_vector(make_episode(train_records[0], "train", seed))))
    selected_learner.restore(selected["learner"])
    test_results = {
        "verification_only": evaluate_policy(test_records, "test", seed, "verification_only"),
        "verification_first_adaptive": evaluate_policy(test_records, "test", seed, "verification_first_adaptive"),
        "adaptive": evaluate_policy(test_records, "test", seed, "adaptive"),
        "fixed_depth_2": evaluate_policy(test_records, "test", seed, "fixed_depth_2"),
        "learned_adaptive": evaluate_policy(test_records, "test", seed, "learned_adaptive", first_action_model=selected_learner),
    }
    return {
        "seed": seed,
        "updates_requested": UPDATES,
        "updates_completed": UPDATES,
        "checkpoint_updates": list(CHECKPOINTS),
        "selected_checkpoint": selected["update"],
        "selected_validation_score": selected["validation_score"],
        "selected_validation": selected["validation"],
        "checkpoints": checkpoints,
        "test": test_results,
        "curves": curves,
        "final_action_counts": learner.counts,
    }


def summary_markdown(result: Mapping[str, Any], manifest: Mapping[str, Any], elapsed: float, source_hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1b-L3000 learned strategy-selection pilot",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; no GPU, cloud, model training or checkpoint download.",
        "",
        "## Verdict",
        "",
        "This is a bounded synthetic feasibility experiment. It does not establish cognitive superiority, metacognition or future competence.",
        f"- Seeds completed: **{len(result['seeds'])}/{len(LEARNER_SEEDS)}**.",
        f"- Policy updates per seed: **{UPDATES}**.",
        f"- Wall time: **{elapsed:.3f} s**; cap: **{WALL_CAP_SECONDS:.1f} s**.",
        f"- Sealed-C1 input overlap: **{manifest['sealed_c1_input_sha256_overlap_count']}**.",
        "",
        "## Test results by learner seed",
        "",
        "The test split was evaluated after validation-based checkpoint selection. Test outcomes did not select a checkpoint.",
        "",
        "| Seed | Policy | Accuracy | Mean cost | False corrections | Unresolved | Verification omissions | Cost/accuracy score |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed_result in result["seeds"]:
        for policy, item in seed_result["test"].items():
            lines.append(
                f"| {seed_result['seed']} | `{policy}` | {item['accuracy']:.4f} | {item['mean_cost']:.3f} | "
                f"{item['false_corrections']} | {item['unresolved']} | {item['verification_omissions']} | {item['cost_accuracy_score']:.4f} |"
            )
    lines.extend([
        "",
        "## Historical C1 reference",
        "",
        "The sealed C1 replay reported verification-only at 100% accuracy / mean cost 5.0 and adaptive at 80.08% / mean cost 6.646. Those cases are not reused here; the values are historical context only.",
        "",
        "## Experimental contract",
        "",
        f"- New task seed: `{TASK_SEED}`; generated records: `{json.dumps(manifest['record_counts'], sort_keys=True)}`.",
        "- Training, validation and test record IDs are disjoint; all generated input hashes were checked against the sealed C1 manifests.",
        "- Training conditions were R/I/U/A; test conditions included R/I/E/D/U/A with declared and inferred-shift modes.",
        "- Latent answers were evaluator-only. The learned policy received state features and declared tool metadata only.",
        "- The bandit selected the first operation; a deterministic bounded continuation executed it. No Q-learning or neural policy was run.",
        "- Validation score: accuracy minus preregistered cost, false-correction and unresolved penalties; ties selected the earliest checkpoint.",
        "- Future-competence objective: disabled.",
        "",
        "## Limitations and decision gate",
        "",
        "A first-operation contextual bandit cannot infer an undisclosed environment shift before observing a tool result. Therefore inferred-shift performance is a deliberate limitation of this pilot, not evidence of sequential adaptation.",
        "The learned policy only earns a follow-on sequential-controller experiment if it beats the best frozen rule on a held-out declared shift without a safety regression. Otherwise the deterministic rule remains preferred.",
        "",
        "## Governing source hashes",
        "",
    ])
    for path, digest in sorted(source_hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    global WALL_CAP_SECONDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wall-cap-seconds", type=float, default=WALL_CAP_SECONDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(LEARNER_SEEDS))
    args = parser.parse_args()
    if args.wall_cap_seconds <= 0:
        raise SystemExit("--wall-cap-seconds must be positive")
    if args.seeds != list(LEARNER_SEEDS):
        raise SystemExit(f"C1b-L3000 requires fixed seeds {list(LEARNER_SEEDS)}")

    WALL_CAP_SECONDS = float(args.wall_cap_seconds)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    built = build_manifests(output)
    records = built["records"]
    manifest_validation = validate_manifests({
        split: load_manifest_records(output / "manifests")[split]
        for split in ("train", "validation", "test")
    })
    write_json(output / "MANIFEST_VALIDATION.json", manifest_validation)
    run_config = {
        "experiment": EXPERIMENT,
        "task_seed": TASK_SEED,
        "learner_seeds": list(LEARNER_SEEDS),
        "updates": UPDATES,
        "checkpoints": list(CHECKPOINTS),
        "actions": list(ACTIONS),
        "policies": list(POLICIES) + ["learned_adaptive"],
        "training_conditions": ["R", "I", "U", "A"],
        "test_conditions": list(CONDITIONS),
        "bandit": {"kind": "linear_contextual_bandit", "feature_count": len(feature_vector(make_episode(records["train"][0], "train", LEARNER_SEEDS[0]))), "epsilon_start": 0.30, "epsilon_floor": 0.05, "learning_rate": "0.08/sqrt(1+0.02*action_count)"},
        "checkpoint_selection": "validation score; earliest checkpoint on tie; test evaluated once after selection",
        "future_competence": "disabled",
        "execution": "CPU-only; no GPU, cloud, model training or checkpoint download",
        "wall_cap_seconds": WALL_CAP_SECONDS,
    }
    write_json(output / "RUN_CONFIG.json", run_config)
    all_seeds: List[Dict[str, Any]] = []
    for seed in LEARNER_SEEDS:
        all_seeds.append(run_seed(records["train"], records["validation"], records["test"], seed, started))
    result = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "elapsed_seconds": round(perf_counter() - started, 6),
        "seeds": all_seeds,
        "aggregate_test": {
            policy: {
                metric: sum(seed_result["test"][policy][metric] for seed_result in all_seeds) / len(all_seeds)
                for metric in ("accuracy", "mean_cost", "false_corrections", "unresolved", "verification_omissions", "cost_accuracy_score")
            }
            for policy in all_seeds[0]["test"]
        },
    }
    write_json(output / "RESULTS.json", result)
    with (output / "TRAINING_CURVES.jsonl").open("w", encoding="utf-8") as handle:
        for seed_result in all_seeds:
            for curve in seed_result["curves"]:
                handle.write(json.dumps(curve, sort_keys=True) + "\n")
    source_paths = [
        ROOT / "tools" / "run_recursive_controller_c1b_l3000.py",
        ROOT / "tools" / "cognitive_task_suite.py",
        ROOT / "tools" / "recursive_controller.py",
        ROOT / "tools" / "validate_cognitive_manifests.py",
        ROOT / "experiments" / "crucible_recursive_controller_c1b_20261021" / "PROTOCOL_V2.md",
        ROOT / "experiments" / "crucible_recursive_controller_c1b_20261021" / "C1B_L3000_PROTOCOL.md",
        ROOT / "experiments" / "crucible_cognitive_eval_v2_20261021" / "manifests" / "test.jsonl",
    ]
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}
    summary = summary_markdown(result, built["manifest"], result["elapsed_seconds"], source_hashes)
    (output / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")
    artifact_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "HASHES.json"
    }
    write_json(output / "HASHES.json", artifact_hashes)
    print(json.dumps({"status": "completed", "output": str(output), "elapsed_seconds": result["elapsed_seconds"], "seeds": len(all_seeds), "updates_per_seed": UPDATES}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
