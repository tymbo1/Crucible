#!/usr/bin/env python3
"""Minimal deterministic procedural-learning lifecycle for Crucible v0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DOMAIN: Tuple[Tuple[int, int], ...] = tuple((a, b) for a in range(10) for b in range(10))
RULE_NAMES: Tuple[str, ...] = (
    "ADD_MOD10",
    "SUB_MOD10",
    "MUL_MOD10",
    "FIRST",
    "SECOND",
    "XOR_MOD10",
    "MAX",
    "MIN",
)


def apply_rule(rule_name: str, a: int, b: int) -> int:
    if rule_name == "ADD_MOD10":
        return (a + b) % 10
    if rule_name == "SUB_MOD10":
        return (a - b) % 10
    if rule_name == "MUL_MOD10":
        return (a * b) % 10
    if rule_name == "FIRST":
        return a
    if rule_name == "SECOND":
        return b
    if rule_name == "XOR_MOD10":
        return (a ^ b) % 10
    if rule_name == "MAX":
        return max(a, b)
    if rule_name == "MIN":
        return min(a, b)
    raise ValueError(f"unknown rule: {rule_name}")


@dataclass(frozen=True)
class Observation:
    observation_id: str
    a: int
    b: int
    output: int
    source: str
    provenance: Mapping[str, object]

    @classmethod
    def make(cls, a: int, b: int, output: int, source: str, provenance: Mapping[str, object]) -> "Observation":
        material = json.dumps({"a": a, "b": b, "output": output, "source": source, "provenance": provenance}, sort_keys=True)
        return cls(hashlib.sha256(material.encode("utf-8")).hexdigest()[:20], a, b, output, source, dict(provenance))

    def as_ref(self) -> str:
        return self.observation_id


def observation_from_oracle(a: int, b: int, oracle: Callable[[int, int], int], source: str, provenance: Mapping[str, object]) -> Observation:
    return Observation.make(a, b, int(oracle(a, b)), source, provenance)


def compatible_rules(observations: Sequence[Observation], rule_names: Sequence[str] = RULE_NAMES) -> List[str]:
    return [
        rule_name
        for rule_name in rule_names
        if all(apply_rule(rule_name, observation.a, observation.b) == observation.output for observation in observations)
    ]


class PersistentStore:
    """Versioned durable procedure metadata; raw demonstrations are not stored."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        self.state: Dict[str, object] = {
            "schema_version": "crucible_learning_lifecycle_v0",
            "procedures": {},
            "events": [],
        }
        if path is not None and path.exists():
            self.state = json.loads(path.read_text(encoding="utf-8"))

    @property
    def procedures(self) -> Dict[str, List[Dict[str, object]]]:
        return self.state.setdefault("procedures", {})  # type: ignore[return-value]

    @property
    def events(self) -> List[Dict[str, object]]:
        return self.state.setdefault("events", [])  # type: ignore[return-value]

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def promote(
        self,
        alias: str,
        rule_name: str,
        evidence_refs: Sequence[str],
        verification_refs: Sequence[str],
        provenance: Mapping[str, object],
        confidence: float = 1.0,
    ) -> Dict[str, object]:
        versions = self.procedures.setdefault(alias, [])
        version = len(versions) + 1
        record: Dict[str, object] = {
            "alias": alias,
            "version": version,
            "rule_name": rule_name,
            "applicability": "digits_0_9_mod10_binary",
            "status": "validated",
            "confidence": confidence,
            "evidence_refs": list(evidence_refs),
            "verification_refs": list(verification_refs),
            "provenance": dict(provenance),
            "revision_history": [],
        }
        versions.append(record)
        self.events.append({"kind": "promotion", "alias": alias, "version": version, "rule_name": rule_name})
        self.save()
        return record

    def active(self, alias: str) -> Optional[Dict[str, object]]:
        versions = self.procedures.get(alias, [])
        for record in reversed(versions):
            if record.get("status") == "validated":
                return record
        return None

    def invalidate(self, alias: str, contradiction: Mapping[str, object], reason: str) -> bool:
        record = self.active(alias)
        if record is None:
            return False
        record["status"] = "invalidated"
        history = record.setdefault("revision_history", [])
        if isinstance(history, list):
            history.append({"kind": "contradiction", "reason": reason, "evidence": dict(contradiction)})
        self.events.append({"kind": "invalidation", "alias": alias, "version": record["version"], "reason": reason})
        self.save()
        return True

    def record_event(self, event: Mapping[str, object]) -> None:
        self.events.append(dict(event))
        self.save()

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.state, sort_keys=True).encode("utf-8")).hexdigest()


class ProceduralLearner:
    def __init__(self, store: Optional[PersistentStore], probe_budget: int = 4, rule_names: Sequence[str] = RULE_NAMES) -> None:
        self.store = store
        self.probe_budget = probe_budget
        self.rule_names = tuple(rule_names)
        self.transient_hypotheses: Dict[str, List[str]] = {}
        self.transient_observations: Dict[str, List[Observation]] = {}

    def reset_context(self) -> None:
        self.transient_hypotheses.clear()
        self.transient_observations.clear()

    def _choose_probe(self, candidates: Sequence[str], observations: Sequence[Observation], probe_pool: Sequence[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        seen = {(observation.a, observation.b) for observation in observations}
        for a, b in probe_pool:
            if (a, b) in seen:
                continue
            outputs = {apply_rule(rule_name, a, b) for rule_name in candidates}
            if len(outputs) > 1:
                return a, b
        return None

    def learn(
        self,
        alias: str,
        demonstrations: Sequence[Observation],
        verification_pairs: Sequence[Tuple[int, int]],
        oracle: Callable[[int, int], int],
        probe_pool: Sequence[Tuple[int, int]],
        provenance: Mapping[str, object],
    ) -> Dict[str, object]:
        observations = list(demonstrations)
        candidates = compatible_rules(observations, self.rule_names)
        probes: List[Observation] = []
        candidate_trace = [{"stage": "demonstrations", "candidates": list(candidates)}]
        for probe_index in range(self.probe_budget):
            if len(candidates) <= 1:
                break
            probe = self._choose_probe(candidates, observations, probe_pool)
            if probe is None:
                break
            observation = observation_from_oracle(probe[0], probe[1], oracle, "active_probe", {"alias": alias, "probe_index": probe_index})
            probes.append(observation)
            observations.append(observation)
            candidates = compatible_rules(observations, self.rule_names)
            candidate_trace.append({"stage": "probe", "probe_ref": observation.observation_id, "candidates": list(candidates)})

        self.transient_hypotheses[alias] = list(candidates)
        self.transient_observations[alias] = list(observations)
        cost = {"demonstrations": len(demonstrations), "probes": len(probes), "verification_checks": 0, "candidate_evaluations": len(self.rule_names) * len(observations)}
        if not candidates:
            result = {"status": "failed", "reason": "no_compatible_hypothesis", "candidates": [], "candidate_trace": candidate_trace, "cost": cost}
            if self.store is not None:
                self.store.record_event({"kind": "failed_strategy", "alias": alias, "reason": result["reason"], "provenance": dict(provenance)})
            return result
        if len(candidates) != 1:
            result = {"status": "uncertain", "reason": "multiple_compatible_hypotheses", "candidates": list(candidates), "candidate_trace": candidate_trace, "cost": cost}
            if self.store is not None:
                self.store.record_event({"kind": "uncertain_hypothesis", "alias": alias, "candidates": list(candidates), "provenance": dict(provenance)})
            return result

        rule_name = candidates[0]
        verification_refs: List[str] = []
        for index, (a, b) in enumerate(verification_pairs):
            expected = int(oracle(a, b))
            verification = Observation.make(a, b, expected, "independent_verification", {"alias": alias, "verification_index": index})
            verification_refs.append(verification.observation_id)
            cost["verification_checks"] = int(cost["verification_checks"]) + 1
            if apply_rule(rule_name, a, b) != expected:
                result = {"status": "failed", "reason": "verification_failed", "rule_name": rule_name, "candidates": list(candidates), "candidate_trace": candidate_trace, "verification_refs": verification_refs, "cost": cost}
                if self.store is not None:
                    self.store.record_event({"kind": "failed_verification", "alias": alias, "rule_name": rule_name, "verification_refs": verification_refs, "provenance": dict(provenance)})
                return result

        evidence_refs = [observation.as_ref() for observation in observations]
        if self.store is None:
            result = {"status": "validated_ephemeral", "rule_name": rule_name, "candidates": list(candidates), "candidate_trace": candidate_trace, "verification_refs": verification_refs, "cost": cost}
            return result
        record = self.store.promote(alias, rule_name, evidence_refs, verification_refs, provenance)
        return {"status": "validated", "rule_name": rule_name, "candidates": list(candidates), "candidate_trace": candidate_trace, "verification_refs": verification_refs, "record": record, "cost": cost}

    def execute(self, alias: str, a: int, b: int) -> Optional[int]:
        if self.store is None:
            return None
        record = self.store.active(alias)
        if record is None:
            return None
        return apply_rule(str(record["rule_name"]), a, b)

    def invalidate(self, alias: str, a: int, b: int, observed_output: int, reason: str) -> bool:
        if self.store is None:
            return False
        return self.store.invalidate(alias, {"a": a, "b": b, "observed_output": observed_output}, reason)


class RawExampleRetriever:
    def __init__(self, examples: Sequence[Observation]) -> None:
        self.examples = list(examples)

    def execute(self, a: int, b: int) -> Optional[int]:
        for example in self.examples:
            if example.a == a and example.b == b:
                return example.output
        return None

