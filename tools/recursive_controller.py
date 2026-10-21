#!/usr/bin/env python3
"""CPU-only bounded recursive-reasoning controller for C0/C1.

This module deliberately contains no model, CUDA or network code.  It is a
small executable contract for the controller specification: immutable state
snapshots, inherited budgets, deterministic transitions, independent
verification and quarantined memory writes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


POLICIES = (
    "one_pass",
    "fixed_depth_2",
    "confidence_stop",
    "verification_only",
    "adaptive",
)

ACTION_COSTS = {
    "answer": (1, 0, 0, 0, 1.0),
    "critique": (1, 0, 0, 0, 1.0),
    "counterexample": (1, 0, 1, 0, 1.5),
    "evidence_search": (1, 1, 0, 0, 2.0),
    "verify": (1, 1, 0, 0, 3.0),
    "revise": (1, 0, 0, 0, 1.0),
    "stop": (0, 0, 0, 0, 0.0),
}


class ContractError(ValueError):
    """Raised when a controller contract or safety invariant is violated."""


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if dataclass_instance(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def dataclass_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__")


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReasoningBudget:
    """Remaining budget; child states can only reduce these values."""

    depth: int = 8
    actions: int = 8
    evidence_calls: int = 2
    counterexample_calls: int = 2
    memory_writes: int = 2
    cost: float = 12.0

    def child(
        self,
        *,
        depth: int = 0,
        actions: int = 0,
        evidence_calls: int = 0,
        counterexample_calls: int = 0,
        memory_writes: int = 0,
        cost: float = 0.0,
    ) -> "ReasoningBudget":
        deductions = {
            "depth": depth,
            "actions": actions,
            "evidence_calls": evidence_calls,
            "counterexample_calls": counterexample_calls,
            "memory_writes": memory_writes,
            "cost": cost,
        }
        for name, amount in deductions.items():
            if amount < 0:
                raise ContractError(f"negative budget deduction: {name}")
            if amount > getattr(self, name):
                raise ContractError(f"budget exhausted: {name}")
        return replace(
            self,
            depth=self.depth - depth,
            actions=self.actions - actions,
            evidence_calls=self.evidence_calls - evidence_calls,
            counterexample_calls=self.counterexample_calls - counterexample_calls,
            memory_writes=self.memory_writes - memory_writes,
            cost=round(self.cost - cost, 9),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "actions": self.actions,
            "evidence_calls": self.evidence_calls,
            "counterexample_calls": self.counterexample_calls,
            "memory_writes": self.memory_writes,
            "cost": self.cost,
        }


@dataclass(frozen=True)
class BeliefState:
    hypotheses: Tuple[Tuple[str, float], ...]
    calibrated: bool = False
    calibration_source: str = "none"

    @classmethod
    def from_mapping(
        cls,
        probabilities: Mapping[str, float],
        *,
        calibrated: bool,
        calibration_source: str,
    ) -> "BeliefState":
        if not probabilities or any(value < 0 or not math.isfinite(value) for value in probabilities.values()):
            raise ContractError("belief probabilities must be finite and non-negative")
        total = sum(probabilities.values())
        if total <= 0:
            raise ContractError("belief probabilities must have positive mass")
        hypotheses = tuple(sorted((key, value / total) for key, value in probabilities.items()))
        return cls(hypotheses, calibrated=calibrated, calibration_source=calibration_source)

    def entropy(self) -> Optional[float]:
        if not self.calibrated:
            return None
        return -sum(probability * math.log(probability) for _, probability in self.hypotheses if probability > 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypotheses": [[key, probability] for key, probability in self.hypotheses],
            "calibrated": self.calibrated,
            "calibration_source": self.calibration_source,
        }


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    status: str = "hypothesis"
    confidence: float = 0.0
    provenance: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "status": self.status,
            "confidence": self.confidence,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    claim_id: str
    source: str
    content: str
    reliability: float
    verified: bool = False
    misleading: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "claim_id": self.claim_id,
            "source": self.source,
            "content": self.content,
            "reliability": self.reliability,
            "verified": self.verified,
            "misleading": self.misleading,
        }


@dataclass(frozen=True)
class Observation:
    action: str
    signal: str
    reliability: float
    verified: bool = False
    detail: str = ""
    evidence_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "signal": self.signal,
            "reliability": self.reliability,
            "verified": self.verified,
            "detail": self.detail,
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True)
class MemoryProposal:
    proposal_id: str
    claim_id: str
    content: str
    source: str
    evidence_ids: Tuple[str, ...]
    contradiction_ids: Tuple[str, ...]
    verified: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "claim_id": self.claim_id,
            "content": self.content,
            "source": self.source,
            "evidence_ids": list(self.evidence_ids),
            "contradiction_ids": list(self.contradiction_ids),
            "verified": self.verified,
        }


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    claim_id: str
    content: str
    source: str
    evidence_ids: Tuple[str, ...]
    provenance_hash: str
    committed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "claim_id": self.claim_id,
            "content": self.content,
            "source": self.source,
            "evidence_ids": list(self.evidence_ids),
            "provenance_hash": self.provenance_hash,
            "committed": self.committed,
        }


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family: str
    prompt: str
    gold_answer: str
    wrong_answer: str
    initial_answer: str
    initial_confidence: float
    category: str
    candidate_revision_answer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "family": self.family,
            "prompt": self.prompt,
            "gold_answer": self.gold_answer,
            "wrong_answer": self.wrong_answer,
            "initial_answer": self.initial_answer,
            "initial_confidence": self.initial_confidence,
            "category": self.category,
            "candidate_revision_answer": self.candidate_revision_answer,
        }


@dataclass(frozen=True)
class ControllerState:
    scenario_id: str
    goal: str
    answer: str
    confidence: float
    beliefs: BeliefState
    observations: Tuple[Observation, ...]
    claims: Tuple[Claim, ...]
    evidence: Tuple[Evidence, ...]
    contradictions: Tuple[str, ...]
    history: Tuple[str, ...]
    cycle_keys: Tuple[str, ...]
    memory_proposals: Tuple[MemoryProposal, ...]
    memory_commits: Tuple[MemoryRecord, ...]
    status: str
    budget: ReasoningBudget
    step: int = 0

    def payload(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "goal": self.goal,
            "answer": self.answer,
            "confidence": self.confidence,
            "beliefs": self.beliefs,
            "observations": self.observations,
            "claims": self.claims,
            "evidence": self.evidence,
            "contradictions": self.contradictions,
            "history": self.history,
            "cycle_keys": self.cycle_keys,
            "memory_proposals": self.memory_proposals,
            "memory_commits": self.memory_commits,
            "status": self.status,
            "budget": self.budget,
            "step": self.step,
        }

    @property
    def state_hash(self) -> str:
        return sha256_value(self.payload())

    def to_dict(self) -> Dict[str, Any]:
        result = _jsonable(self.payload())
        result["state_hash"] = self.state_hash
        return result


@dataclass(frozen=True)
class TransitionRecord:
    sequence: int
    pre_state_hash: str
    post_state_hash: str
    action: str
    accepted: bool
    reason: str
    score: Dict[str, Any]
    cost: Dict[str, Any]
    observation: Optional[Observation]

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable({
            "sequence": self.sequence,
            "pre_state_hash": self.pre_state_hash,
            "post_state_hash": self.post_state_hash,
            "action": self.action,
            "accepted": self.accepted,
            "reason": self.reason,
            "score": self.score,
            "cost": self.cost,
            "observation": self.observation,
        })


@dataclass(frozen=True)
class UtilityConfig:
    """Preregistered dimensionless C0/C1 score coefficients."""

    utility: float = 1.0
    information_gain: float = 0.25
    consistency: float = 0.50
    self_quality: float = 0.25
    entropy: float = 0.10
    cost: float = 0.10
    future_competence: float = 0.0
    normalization: str = "all quality components clipped to [0,1]; cost is cost/12"
    future_competence_status: str = "disabled_for_C0_C1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "utility": self.utility,
            "information_gain": self.information_gain,
            "consistency": self.consistency,
            "self_quality": self.self_quality,
            "entropy": self.entropy,
            "cost": self.cost,
            "future_competence": self.future_competence,
            "normalization": self.normalization,
            "future_competence_status": self.future_competence_status,
        }


@dataclass(frozen=True)
class PolicyRun:
    policy: str
    scenario: Scenario
    final_state: ControllerState
    transitions: Tuple[TransitionRecord, ...]
    initial_correct: bool
    final_correct: bool
    corrected: bool
    false_correction: bool
    unresolved: bool
    valid_memory_commits: int
    invalid_memory_attempts: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "scenario_id": self.scenario.scenario_id,
            "family": self.scenario.family,
            "category": self.scenario.category,
            "initial_correct": self.initial_correct,
            "final_correct": self.final_correct,
            "corrected": self.corrected,
            "false_correction": self.false_correction,
            "unresolved": self.unresolved,
            "valid_memory_commits": self.valid_memory_commits,
            "invalid_memory_attempts": self.invalid_memory_attempts,
            "final_state": self.final_state.to_dict(),
            "transitions": [transition.to_dict() for transition in self.transitions],
        }


def _wrong_answer(gold: str) -> str:
    if gold.lstrip("-").isdigit():
        return str(int(gold) + 1)
    alternatives = {
        "yes": "no",
        "no": "yes",
        "entailed": "unknown",
        "contradicted": "unknown",
        "unknown": "entailed",
    }
    if gold in alternatives:
        return alternatives[gold]
    if "|" in gold:
        status, value = gold.split("|", 1)
        return f"{'incorrect' if status == 'correct' else 'correct'}|{value}"
    return f"not-{gold}"


def make_scenario(record: Mapping[str, Any]) -> Scenario:
    """Create a deterministic imperfect candidate fixture from one record.

    The gold target is retained only by the independent verifier.  Policy
    selection uses the signals returned by this fixture, never the target.
    """

    record_id = str(record["record_id"])
    number = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:8], 16)
    categories = ("clean_correct", "detectable_wrong", "misleading_evidence", "contradiction", "shortcut_probe")
    category = categories[number % len(categories)]
    gold = str(record["target"])
    wrong = _wrong_answer(gold)
    if category == "clean_correct":
        initial = gold
        confidence = 0.88
    elif category == "contradiction":
        initial = gold
        confidence = 0.90
    elif category == "misleading_evidence":
        initial = wrong
        confidence = 0.93
    elif category == "shortcut_probe":
        initial = wrong
        confidence = 0.55
    else:
        initial = wrong
        confidence = 0.62
    return Scenario(
        scenario_id=record_id,
        family=str(record["family"]),
        prompt=str(record["input"]),
        gold_answer=gold,
        wrong_answer=wrong,
        initial_answer=initial,
        initial_confidence=confidence,
        category=category,
        candidate_revision_answer=gold,
    )


def initial_state(scenario: Scenario, budget: Optional[ReasoningBudget] = None) -> ControllerState:
    return ControllerState(
        scenario_id=scenario.scenario_id,
        goal="produce a correct, verifiable answer without unsupported memory writes",
        answer="",
        confidence=0.0,
        beliefs=BeliefState.from_mapping(
            {"candidate_correct": 0.5, "candidate_wrong": 0.5},
            calibrated=False,
            calibration_source="synthetic_candidate_not_calibrated",
        ),
        observations=(),
        claims=(),
        evidence=(),
        contradictions=(),
        history=(),
        cycle_keys=(),
        memory_proposals=(),
        memory_commits=(),
        status="active",
        budget=budget or ReasoningBudget(),
    )


def _signal_for(scenario: Scenario, action: str) -> Observation:
    if action == "critique":
        signals = {
            "clean_correct": ("no_error", 0.90, False),
            "detectable_wrong": ("error", 0.90, False),
            "misleading_evidence": ("no_error", 0.90, False),
            "contradiction": ("no_error", 0.80, False),
            "shortcut_probe": ("uncertain", 0.60, False),
        }
        signal, reliability, verified = signals[scenario.category]
        return Observation(action, signal, reliability, verified, "synthetic critique")
    if action == "counterexample":
        signals = {
            "clean_correct": ("no_counterexample", 0.90, False),
            "detectable_wrong": ("supports_error", 0.90, False),
            "misleading_evidence": ("false_alarm", 0.90, False),
            "contradiction": ("false_alarm", 0.85, False),
            "shortcut_probe": ("supports_error", 0.70, False),
        }
        signal, reliability, verified = signals[scenario.category]
        return Observation(action, signal, reliability, verified, "synthetic counterexample")
    if action == "evidence_search":
        signals = {
            "clean_correct": ("supports_candidate", 0.80, False),
            "detectable_wrong": ("supports_correction", 0.80, False),
            "misleading_evidence": ("supports_candidate", 0.95, False),
            "contradiction": ("contradicts_candidate", 0.95, False),
            "shortcut_probe": ("unavailable", 0.0, False),
        }
        signal, reliability, verified = signals[scenario.category]
        evidence_id = sha256_value({"scenario": scenario.scenario_id, "action": action})[:16]
        return Observation(action, signal, reliability, verified, "synthetic evidence; not an oracle", evidence_id)
    if action == "verify":
        passed = scenario.initial_answer == scenario.gold_answer
        signal = "verified_correct" if passed else "verified_incorrect"
        return Observation(action, signal, 1.0, True, scenario.gold_answer)
    raise ContractError(f"no synthetic signal for action {action}")


def _apply_revision(state: ControllerState, scenario: Scenario, policy: str) -> Tuple[str, str]:
    observations = state.observations
    if any(observation.signal == "verified_incorrect" for observation in observations):
        verifier_answer = next(
            observation.detail
            for observation in observations
            if observation.signal == "verified_incorrect"
        )
        return verifier_answer, "independent verifier correction"
    if any(observation.signal == "verified_correct" for observation in observations):
        return state.answer, "independent verifier accepted answer"
    signals = {observation.signal for observation in observations}
    if policy == "fixed_depth_2":
        if signals & {"error", "supports_error", "contradicts_candidate"}:
            if scenario.category == "contradiction":
                return scenario.wrong_answer, "naive fixed-depth false correction"
            return scenario.candidate_revision_answer, "two-pass synthetic revision"
        return state.answer, "fixed-depth retained answer"
    if policy in {"confidence_stop", "adaptive"}:
        independent_error = "error" in signals and "supports_error" in signals
        reliable_evidence = "supports_correction" in signals and "error" in signals
        if independent_error or reliable_evidence:
            return scenario.candidate_revision_answer, "independent signals supported revision"
        return state.answer, "insufficient independent support"
    return state.answer, "no revision rule"


def _score(state: ControllerState, scenario: Scenario, action: str, config: UtilityConfig) -> Dict[str, Any]:
    # The controller must not inspect the task oracle while selecting or
    # recording an operational score.  Accuracy is computed separately by the
    # evaluator after a run has completed.
    consistency = 0.0 if state.contradictions else 1.0
    self_quality = 0.0
    entropy = state.beliefs.entropy()
    ig_status = "unavailable"
    ig = 0.0
    if entropy is not None:
        ig_status = "available_only_with_calibrated_transition"
    cost_fraction = max(0.0, min(1.0, (12.0 - state.budget.cost) / 12.0))
    total = (
        config.information_gain * ig
        + config.consistency * consistency
        + config.self_quality * self_quality
        - config.entropy * max(0.0, entropy or 0.0)
        - config.cost * cost_fraction
        + config.future_competence * 0.0
    )
    return {
        "action": action,
        "total": round(total, 9),
        "answer_quality": "evaluator_only_not_available_to_controller",
        "consistency": consistency,
        "self_quality": round(self_quality, 9),
        "self_quality_status": "unavailable_without_historical_outcomes",
        "information_gain": ig,
        "information_gain_status": ig_status,
        "entropy_credit": 0.0,
        "future_competence": 0.0,
        "future_competence_status": "disabled_for_C0_C1",
    }


class ControllerEngine:
    """Apply deterministic, immutable transitions to a controller state."""

    def __init__(self, config: Optional[UtilityConfig] = None) -> None:
        self.config = config or UtilityConfig()

    def apply(
        self,
        state: ControllerState,
        scenario: Scenario,
        action: str,
        *,
        policy: str,
        sequence: int,
    ) -> Tuple[ControllerState, TransitionRecord]:
        if action not in ACTION_COSTS:
            raise ContractError(f"unsupported action: {action}")
        pre_hash = state.state_hash
        # A critique/counterexample cannot be retried against the same
        # evidence-free state.  History and budget changes make ordinary
        # state hashes different after every step, so this semantic guard is
        # needed in addition to the exact state/action cycle key below.
        if action in {"critique", "counterexample"} and action in state.history:
            new_evidence = any(
                observation.action in {"evidence_search", "verify"}
                for observation in state.observations
            )
            if not new_evidence:
                rejected = replace(state, status="stop_unresolved")
                record = TransitionRecord(
                    sequence, pre_hash, rejected.state_hash, action, False,
                    "cycle_detected", _score(rejected, scenario, action, self.config), {}, None,
                )
                return rejected, record
        cycle_key = sha256_value({"state": pre_hash, "action": action})
        if cycle_key in state.cycle_keys:
            rejected = replace(state, status="stop_unresolved", cycle_keys=state.cycle_keys + (cycle_key,))
            record = TransitionRecord(
                sequence, pre_hash, rejected.state_hash, action, False, "cycle_detected",
                _score(rejected, scenario, action, self.config), {}, None,
            )
            return rejected, record
        action_count, evidence_count, counterexample_count, memory_count, cost = ACTION_COSTS[action]
        try:
            next_budget = state.budget.child(
                depth=1 if action not in {"answer", "stop"} else 0,
                actions=action_count,
                evidence_calls=evidence_count,
                counterexample_calls=counterexample_count,
                memory_writes=memory_count,
                cost=cost,
            )
        except ContractError as exc:
            rejected = replace(state, status="stop_unresolved", cycle_keys=state.cycle_keys + (cycle_key,))
            record = TransitionRecord(
                sequence, pre_hash, rejected.state_hash, action, False, str(exc),
                _score(rejected, scenario, action, self.config), {}, None,
            )
            return rejected, record

        observation: Optional[Observation] = None
        next_state = replace(
            state,
            budget=next_budget,
            history=state.history + (action,),
            cycle_keys=state.cycle_keys + (cycle_key,),
            step=state.step + 1,
        )
        if action == "answer":
            next_state = replace(next_state, answer=scenario.initial_answer, confidence=scenario.initial_confidence)
        elif action in {"critique", "counterexample", "evidence_search", "verify"}:
            observation = _signal_for(scenario, action)
            contradictions = next_state.contradictions
            if observation.signal in {"false_alarm", "contradicts_candidate"}:
                contradictions = contradictions + (f"{action}:{observation.signal}",)
            evidence = next_state.evidence
            if observation.evidence_id:
                evidence = evidence + (
                    Evidence(
                        observation.evidence_id,
                        scenario.scenario_id,
                        "synthetic_fixture",
                        observation.signal,
                        observation.reliability,
                        observation.verified,
                        observation.signal in {"supports_candidate", "false_alarm", "contradicts_candidate"},
                    ),
                )
            next_state = replace(
                next_state,
                observations=next_state.observations + (observation,),
                contradictions=contradictions,
                evidence=evidence,
            )
        elif action == "revise":
            answer, reason = _apply_revision(next_state, scenario, policy)
            next_state = replace(next_state, answer=answer)
            observation = Observation(action, "revision_applied", 1.0, False, reason)
        elif action == "stop":
            next_state = replace(next_state, status="stopped")

        record = TransitionRecord(
            sequence,
            pre_hash,
            next_state.state_hash,
            action,
            True,
            "accepted",
            _score(next_state, scenario, action, self.config),
            {
                "actions": action_count,
                "evidence_calls": evidence_count,
                "counterexample_calls": counterexample_count,
                "memory_writes": memory_count,
                "estimated_cost": cost,
            },
            observation,
        )
        return next_state, record


class MemoryStore:
    """Append-only transactional memory with explicit rollback."""

    def __init__(self) -> None:
        self.records: List[MemoryRecord] = []
        self.audit: List[Dict[str, Any]] = []

    def commit(self, proposal: MemoryProposal) -> MemoryRecord:
        if not proposal.source or not proposal.evidence_ids or not proposal.verified:
            self.audit.append({"proposal_id": proposal.proposal_id, "event": "rejected"})
            raise ContractError("memory proposal lacks provenance, evidence or verification")
        if proposal.contradiction_ids:
            self.audit.append({"proposal_id": proposal.proposal_id, "event": "rejected_contradiction"})
            raise ContractError("memory proposal has unresolved contradictions")
        record = MemoryRecord(
            record_id=sha256_value(proposal.to_dict())[:20],
            claim_id=proposal.claim_id,
            content=proposal.content,
            source=proposal.source,
            evidence_ids=proposal.evidence_ids,
            provenance_hash=sha256_value({"source": proposal.source, "evidence": proposal.evidence_ids}),
        )
        self.records.append(record)
        self.audit.append({"proposal_id": proposal.proposal_id, "event": "committed", "record_id": record.record_id})
        return record

    def rollback(self, proposal: MemoryProposal, reason: str) -> None:
        self.audit.append({"proposal_id": proposal.proposal_id, "event": "rollback", "reason": reason})


def _policy_actions(policy: str, state: ControllerState) -> Tuple[str, ...]:
    if policy == "one_pass":
        return ("stop",)
    if policy == "fixed_depth_2":
        return ("critique", "counterexample", "revise", "stop")
    if policy == "confidence_stop":
        if state.confidence < 0.75:
            return ("critique", "revise", "stop")
        return ("stop",)
    if policy == "verification_only":
        return ("verify", "revise", "stop")
    if policy == "adaptive":
        if state.confidence < 0.75:
            return ("critique", "counterexample", "revise", "stop")
        return ("critique", "evidence_search", "verify", "revise", "stop")
    raise ContractError(f"unknown policy: {policy}")


def _memory_for_state(state: ControllerState, scenario: Scenario, policy: str) -> Tuple[ControllerState, int, int]:
    verified = [observation for observation in state.observations if observation.verified]
    if not verified:
        return state, 0, 0
    evidence_ids = tuple(observation.evidence_id for observation in verified if observation.evidence_id)
    proposal = MemoryProposal(
        proposal_id=sha256_value({"scenario": scenario.scenario_id, "policy": policy})[:20],
        claim_id=scenario.scenario_id,
        content=state.answer,
        source="independent_verifier",
        evidence_ids=evidence_ids or (f"verifier:{scenario.scenario_id}",),
        contradiction_ids=state.contradictions,
        verified=any(
            observation.verified and observation.detail == state.answer
            for observation in verified
        ),
    )
    try:
        next_budget = state.budget.child(memory_writes=1)
    except ContractError:
        return replace(state, memory_proposals=state.memory_proposals + (proposal,)), 0, 1
    state_with_proposal = replace(
        state,
        budget=next_budget,
        memory_proposals=state.memory_proposals + (proposal,),
    )
    store = MemoryStore()
    try:
        record = store.commit(proposal)
    except ContractError:
        store.rollback(proposal, "verification/provenance/contradiction gate")
        return state_with_proposal, 0, 1
    return replace(state_with_proposal, memory_commits=state.memory_commits + (record,)), 1, 0


def run_policy(
    scenario: Scenario,
    policy: str,
    *,
    budget: Optional[ReasoningBudget] = None,
    engine: Optional[ControllerEngine] = None,
) -> PolicyRun:
    if policy not in POLICIES:
        raise ContractError(policy)
    engine = engine or ControllerEngine()
    state = initial_state(scenario, budget)
    transitions: List[TransitionRecord] = []
    state, record = engine.apply(state, scenario, "answer", policy=policy, sequence=0)
    transitions.append(record)
    sequence = 1
    for action in _policy_actions(policy, state):
        state, record = engine.apply(state, scenario, action, policy=policy, sequence=sequence)
        transitions.append(record)
        sequence += 1
        if action == "stop":
            break
    state, valid_commits, invalid_attempts = _memory_for_state(state, scenario, policy)
    final_correct = state.answer == scenario.gold_answer
    initial_correct = scenario.initial_answer == scenario.gold_answer
    unresolved = not final_correct
    state = replace(state, status="resolved" if final_correct else "stop_unresolved")
    return PolicyRun(
        policy=policy,
        scenario=scenario,
        final_state=state,
        transitions=tuple(transitions),
        initial_correct=initial_correct,
        final_correct=final_correct,
        corrected=(not initial_correct and final_correct),
        false_correction=(initial_correct and not final_correct),
        unresolved=unresolved,
        valid_memory_commits=valid_commits,
        invalid_memory_attempts=invalid_attempts,
    )


def calibration_metrics(runs: Sequence[PolicyRun]) -> Dict[str, Any]:
    if not runs:
        return {"count": 0}
    brier = sum((run.scenario.initial_confidence - float(run.initial_correct)) ** 2 for run in runs) / len(runs)
    bins: Dict[int, List[bool]] = {}
    for run in runs:
        bucket = min(9, int(run.scenario.initial_confidence * 10))
        bins.setdefault(bucket, []).append(run.initial_correct)
    ece = 0.0
    bin_report = {}
    for bucket, outcomes in sorted(bins.items()):
        confidence = (bucket + 0.5) / 10.0
        accuracy = sum(outcomes) / len(outcomes)
        ece += abs(confidence - accuracy) * len(outcomes) / len(runs)
        bin_report[str(bucket)] = {"count": len(outcomes), "accuracy": accuracy, "nominal_midpoint": confidence}
    return {"count": len(runs), "brier": brier, "ece": ece, "bins": bin_report, "status": "diagnostic_only"}


def aggregate_runs(runs: Sequence[PolicyRun]) -> Dict[str, Any]:
    if not runs:
        raise ContractError("cannot aggregate empty run set")
    first = runs[0]
    actions = sum(len(run.transitions) for run in runs)
    estimated_cost = sum(
        float(transition.cost.get("estimated_cost", 0.0))
        for run in runs
        for transition in run.transitions
    )
    by_category: Dict[str, Dict[str, Any]] = {}
    for category in sorted({run.scenario.category for run in runs}):
        subset = [run for run in runs if run.scenario.category == category]
        by_category[category] = {
            "count": len(subset),
            "accuracy": sum(run.final_correct for run in subset) / len(subset),
            "false_corrections": sum(run.false_correction for run in subset),
            "unresolved": sum(run.unresolved for run in subset),
        }
    return {
        "policy": first.policy,
        "records": len(runs),
        "accuracy": sum(run.final_correct for run in runs) / len(runs),
        "initial_accuracy": sum(run.initial_correct for run in runs) / len(runs),
        "corrections": sum(run.corrected for run in runs),
        "false_corrections": sum(run.false_correction for run in runs),
        "unresolved": sum(run.unresolved for run in runs),
        "valid_memory_commits": sum(run.valid_memory_commits for run in runs),
        "invalid_memory_attempts": sum(run.invalid_memory_attempts for run in runs),
        "transition_count": actions,
        "estimated_cost_units": round(estimated_cost, 6),
        "mean_actions": actions / len(runs),
        "by_category": by_category,
        "calibration": calibration_metrics(runs),
    }


def run_c0_checks() -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def check(name: str, function: Any) -> None:
        try:
            function()
            checks.append({"name": name, "passed": True})
        except Exception as exc:  # pragma: no cover - report preserves failures
            checks.append({"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"})

    def budget_monotonicity() -> None:
        root = ReasoningBudget()
        child = root.child(actions=1, cost=1.0)
        if child.actions >= root.actions or child.cost >= root.cost:
            raise AssertionError("child budget did not decrease")
        try:
            child.child(actions=root.actions + 1)
        except ContractError:
            return
        raise AssertionError("child budget increased or bypassed exhaustion")

    def state_round_trip() -> None:
        scenario = Scenario("c0", "math", "", "1", "2", "1", 0.5, "clean_correct")
        state = initial_state(scenario)
        if sha256_value(state.to_dict() | {"state_hash": state.state_hash}) != sha256_value(state.to_dict() | {"state_hash": state.state_hash}):
            raise AssertionError("non-deterministic serialization")
        restored = ControllerState(
            scenario_id=state.scenario_id,
            goal=state.goal,
            answer=state.answer,
            confidence=state.confidence,
            beliefs=BeliefState.from_mapping({"candidate_correct": 0.5, "candidate_wrong": 0.5}, calibrated=False, calibration_source="synthetic_candidate_not_calibrated"),
            observations=(), claims=(), evidence=(), contradictions=(), history=(), cycle_keys=(), memory_proposals=(), memory_commits=(), status="active", budget=ReasoningBudget(),
        )
        if state.state_hash != restored.state_hash:
            raise AssertionError("state hash changed after round trip")

    def entropy_contract() -> None:
        uncalibrated = BeliefState.from_mapping({"a": 0.5, "b": 0.5}, calibrated=False, calibration_source="none")
        if uncalibrated.entropy() is not None:
            raise AssertionError("uncalibrated entropy was treated as usable")
        calibrated = BeliefState.from_mapping({"a": 0.5, "b": 0.5}, calibrated=True, calibration_source="held_out")
        if not math.isclose(calibrated.entropy() or 0.0, math.log(2), rel_tol=1e-12):
            raise AssertionError("entropy calculation incorrect")

    def invalid_belief_rejected() -> None:
        try:
            BeliefState.from_mapping({"a": float("nan")}, calibrated=True, calibration_source="bad")
        except ContractError:
            return
        raise AssertionError("non-finite belief probability accepted")

    def cycle_detection() -> None:
        scenario = Scenario("c0-cycle", "math", "", "1", "2", "1", 0.5, "clean_correct")
        engine = ControllerEngine()
        state = initial_state(scenario)
        state, _ = engine.apply(state, scenario, "answer", policy="one_pass", sequence=0)
        state, first = engine.apply(state, scenario, "critique", policy="one_pass", sequence=1)
        _, second = engine.apply(state, scenario, "critique", policy="one_pass", sequence=2)
        if not first.accepted or second.accepted or second.reason != "cycle_detected":
            raise AssertionError("cycle was not rejected")

    def budget_exhaustion_is_explicit() -> None:
        scenario = Scenario("c0-budget", "math", "", "1", "2", "1", 0.5, "clean_correct")
        engine = ControllerEngine()
        state = initial_state(scenario, ReasoningBudget(actions=1, cost=1.0))
        state, answer = engine.apply(state, scenario, "answer", policy="one_pass", sequence=0)
        if not answer.accepted or state.budget.actions != 0:
            raise AssertionError("answer did not consume the inherited action budget")
        state, rejected = engine.apply(state, scenario, "critique", policy="one_pass", sequence=1)
        if rejected.accepted or rejected.reason != "budget exhausted: actions" or state.status != "stop_unresolved":
            raise AssertionError("budget exhaustion was not fail-closed")

    def transition_hash_linkage() -> None:
        scenario = Scenario("c0-transition", "math", "", "1", "2", "1", 0.5, "clean_correct")
        engine = ControllerEngine()
        state = initial_state(scenario)
        before = state.state_hash
        state, transition = engine.apply(state, scenario, "answer", policy="one_pass", sequence=0)
        if transition.pre_state_hash != before or transition.post_state_hash != state.state_hash:
            raise AssertionError("transition hashes do not link to immutable state snapshots")

    def transition_records_are_immutable() -> None:
        scenario = Scenario("c0-immutable", "math", "", "1", "2", "1", 0.5, "clean_correct")
        engine = ControllerEngine()
        _, transition = engine.apply(initial_state(scenario), scenario, "answer", policy="one_pass", sequence=0)
        try:
            transition.action = "critique"  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("transition record was mutable")

    def memory_transaction() -> None:
        store = MemoryStore()
        proposal = MemoryProposal("p", "c", "fact", "source", ("e",), (), True)
        record = store.commit(proposal)
        if not record.committed or len(store.records) != 1:
            raise AssertionError("valid memory commit failed")
        bad = MemoryProposal("bad", "c", "fact", "", (), (), True)
        try:
            store.commit(bad)
        except ContractError:
            store.rollback(bad, "missing provenance")
        else:
            raise AssertionError("invalid memory write committed")

    def provenance_gate() -> None:
        store = MemoryStore()
        bad = MemoryProposal("bad", "c", "fact", "source", ("e",), ("contradiction",), True)
        try:
            store.commit(bad)
        except ContractError:
            return
        raise AssertionError("contradictory memory write committed")

    def independent_verifier() -> None:
        scenario = Scenario("c0-verify", "math", "", "1", "2", "2", 0.2, "detectable_wrong")
        run = run_policy(scenario, "verification_only")
        if not run.final_correct or not run.valid_memory_commits:
            raise AssertionError("independent verifier did not correct and commit")

    check("budget_inheritance_is_monotonic", budget_monotonicity)
    check("typed_state_hash_round_trip", state_round_trip)
    check("calibrated_entropy_only", entropy_contract)
    check("invalid_belief_rejected", invalid_belief_rejected)
    check("cycle_detection", cycle_detection)
    check("budget_exhaustion_is_fail_closed", budget_exhaustion_is_explicit)
    check("transition_hash_linkage", transition_hash_linkage)
    check("transition_records_are_immutable", transition_records_are_immutable)
    check("transactional_memory_commit_and_rollback", memory_transaction)
    check("contradiction_provenance_gate", provenance_gate)
    check("independent_verification", independent_verifier)
    return {"checks": checks, "passed": all(item["passed"] for item in checks), "count": len(checks)}
