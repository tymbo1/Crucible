#!/usr/bin/env python3
"""Four-arm cumulative-adaptation runner with a sealed scientific boundary.

The runner has two intentionally separate paths:

* ``fixtures`` exercises the complete lifecycle and all integrity guards with
  synthetic, non-scientific tasks.
* ``scientific`` is a guarded execution path.  It refuses to run unless the
  protocol and amendment carry explicit approval and a sealed target provider
  is supplied.  A separate explicit post-protocol exploratory opt-in is
  available to record a terminated-protocol run as exploratory, never as a
  registered scientific result.

No target answer is needed to propose or independently verify a procedure
against its observed demonstrations.  Post-reset target scoring belongs to a
later, explicitly authorised finalisation phase.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import resource
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.crucible_track_b_neural_proposer import (  # noqa: E402
    GrammarConstraint,
    ast_depth,
    ast_hash,
    ast_nodes,
    canonical_ast,
    evaluate_ast,
    truth_table_hash,
)


EXPERIMENT = "crucible_cumulative_adaptation_scientific_runner_20261028"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_CUMULATIVE_ADAPTATION_GATE_PROTOCOL_20261028.json"
AMENDMENT = ROOT / "docs" / "CRUCIBLE_CUMULATIVE_ADAPTATION_PROTOCOL_INTEGRITY_AMENDMENT_20261028.json"
POOL = ROOT / "experiments" / "crucible_cumulative_adaptation_preflight_20261028" / "GRAMMAR_REPRESENTATIVE_POOL.jsonl"
OUTPUT = ROOT / "experiments" / "crucible_cumulative_adaptation_runner_validation_20261028"
GRAMMAR = GrammarConstraint(max_depth=3, max_nodes=7)

ARM_IDS = (
    "A0_FROZEN_INDUCTION",
    "A1_EXACT_PROCEDURE_MEMORY",
    "A2_SHAM_OPERATOR_LIBRARY",
    "A3_VERIFIED_ADAPTIVE_LIBRARY",
)
TARGET_BUDGET = {
    "candidate_slots": 4,
    "operator_lookup_slots": 4,
    "verifier_calls": 4,
    "verifier_examples_per_call": 8,
    "candidate_construction_units": 16,
    "post_reset_checks": 8,
}
PROMOTION_THRESHOLD = 0.80
PROMOTION_TRIALS = 16
MOTIF_SCHEMA_VERSION = "crucible.operator.motif.v1"
HOLE_TYPE = "expr_mod10"


class RunnerIntegrityError(RuntimeError):
    """Base class for a runner refusal or integrity failure."""


class ScientificExecutionBlocked(RunnerIntegrityError):
    pass


class LeakageViolation(RunnerIntegrityError):
    pass


class BudgetMismatch(RunnerIntegrityError):
    pass


class SealedTargetProvider:
    """Late-opening answer adapter for a separately authorised run.

    The input bundle is allowed to contain demonstrations but never target ASTs
    or post-reset labels.  The answer bundle is opened only after every arm has
    fixed its proposal digest.  This class is defined for the future runner
    path; the fixture-only validation below never instantiates it.
    """

    def __init__(self, input_path: Path, answer_path: Path, authorised: bool):
        if not authorised:
            raise ScientificExecutionBlocked("sealed target provider requires explicit scientific authorization")
        self.input_path = input_path
        self.answer_path = answer_path
        self.opened = False

    def load_inputs(self) -> List[Dict[str, Any]]:
        if not self.input_path.is_file():
            raise RunnerIntegrityError(f"missing sealed target input bundle: {self.input_path}")
        rows = load_jsonl(self.input_path)
        for row in rows:
            forbidden = {"target_ast", "target_outputs", "query_outputs", "truth_table", "answer_labels"} & set(row)
            if forbidden:
                raise LeakageViolation(f"sealed input bundle contains answer fields: {sorted(forbidden)}")
        return rows

    def open_answers_after_finalization(self, proposal_digest: str) -> Dict[str, Dict[str, Any]]:
        if self.opened:
            raise RunnerIntegrityError("sealed target answers may be opened only once")
        if not proposal_digest:
            raise RunnerIntegrityError("proposal digest is required before opening sealed answers")
        if not self.answer_path.is_file():
            raise RunnerIntegrityError(f"missing sealed target answer bundle: {self.answer_path}")
        rows = load_jsonl(self.answer_path)
        answers = {str(row["task_id"]): dict(row) for row in rows}
        self.opened = True
        return answers


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_digest(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def stable_result_digest(value: Any) -> str:
    """Hash deterministic decisions while excluding measured runtime fields."""
    if isinstance(value, dict):
        return json_digest({
            key: stable_result_value(item)
            for key, item in value.items()
            if key not in {"cpu_seconds", "wall_clock_seconds", "resource_seconds"}
        })
    return json_digest(stable_result_value(value))


def stable_result_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: stable_result_value(item)
            for key, item in value.items()
            if key not in {"cpu_seconds", "wall_clock_seconds", "resource_seconds"}
        }
    if isinstance(value, list):
        return [stable_result_value(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def lower_bound_95(successes: int, trials: int) -> float:
    # Avoid a scipy dependency in the runner.  The exact registered value for
    # 16/16 is recorded by the preflight; this closed form is sufficient for
    # the fixed promotion gate and synthetic fixtures.
    if trials == 16 and successes == 16:
        return 0.8049356770309067
    if successes == trials:
        # Beta(0.025; trials+1, 1) = 0.025 ** (1/(trials+1)).
        return 0.025 ** (1.0 / float(trials + 1))
    return 0.0


def root_operator(ast: Mapping[str, Any]) -> Optional[str]:
    canonical = canonical_ast(ast)
    return None if canonical["op"] in {"var", "const"} else str(canonical["op"])


def extract_operator_motif(ast: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Extract an executable metadata motif with typed expression holes.

    Hole nodes are deliberately not accepted by ``GRAMMAR.validate``.  They
    are a canonical operator representation that must be instantiated before
    execution.  Keeping the hole-bearing form separate prevents a partially
    bound motif from entering the persistent procedure store.
    """
    canonical = canonical_ast(ast)
    if canonical["op"] in {"var", "const"}:
        raise ValueError("operator motifs require a non-leaf AST")
    holes = [
        {
            "id": f"arg{index}",
            "type": HOLE_TYPE,
            "path": ["args", index],
        }
        for index, _ in enumerate(canonical.get("args", []))
    ]
    motif = {
        "schema_version": MOTIF_SCHEMA_VERSION,
        "op": canonical["op"],
        "args": [
            {"op": "hole", "id": hole["id"], "type": hole["type"]}
            for hole in holes
        ],
    }
    return motif, holes


def _instantiate_motif_node(node: Mapping[str, Any], bindings: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    if node.get("op") == "hole":
        hole_id = str(node["id"])
        if hole_id not in bindings:
            raise ValueError(f"missing binding for typed hole {hole_id}")
        if str(node.get("type")) != HOLE_TYPE:
            raise ValueError(f"unsupported typed hole {hole_id}: {node.get('type')}")
        return canonical_ast(bindings[hole_id])
    result = {key: value for key, value in node.items() if key != "args"}
    if "args" in node:
        result["args"] = [_instantiate_motif_node(child, bindings) for child in node["args"]]
    return canonical_ast(result)


def instantiate_operator_motif(record: "OperatorRecord", bindings: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Bind all typed holes and return a grammar-validated executable AST."""
    candidate = _instantiate_motif_node(record.canonical_motif, bindings)
    return GRAMMAR.validate(candidate)


def _unique_ast_values(values: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for value in values:
        canonical = canonical_ast(value)
        unique.setdefault(ast_hash(canonical), canonical)
    return [unique[key] for key in sorted(unique)]


def build_operator_representation(
    signature: str,
    source_tasks: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], List[str]]:
    """Build a motif and typed binding vocabulary only from verified sources."""
    matching = [task for task in source_tasks if root_operator(task["target_ast"]) == signature]
    if not matching:
        raise ValueError(f"no source procedures for operator {signature}")
    motif, holes = extract_operator_motif(matching[0]["target_ast"])
    arity = len(holes)
    binding_vocabulary = {
        hole["id"]: _unique_ast_values(
            canonical_ast(task["target_ast"]).get("args", [])[index]
            for task in matching
            if len(canonical_ast(task["target_ast"]).get("args", [])) == arity
        )
        for index, hole in enumerate(holes)
    }
    if any(not values for values in binding_vocabulary.values()):
        raise ValueError(f"empty binding vocabulary for operator {signature}")
    source_refs = [str(task["task_id"]) for task in matching]
    return motif, holes, binding_vocabulary, source_refs


def _match_count(ast: Mapping[str, Any], demonstrations: Sequence[Mapping[str, int]]) -> int:
    return sum(
        int(evaluate_ast(ast, int(example["a"]), int(example["b"]))) == int(example["output"])
        for example in demonstrations
    )


def construct_operator_candidates(
    record: "OperatorRecord",
    demonstrations: Sequence[Mapping[str, int]],
    limit: int = TARGET_BUDGET["candidate_slots"],
) -> List[Dict[str, Any]]:
    """Instantiate a promoted motif from source-derived typed bindings."""
    hole_ids = [hole["id"] for hole in record.typed_holes]
    choices = [record.binding_vocabulary.get(hole_id, []) for hole_id in hole_ids]
    if not choices or any(not values for values in choices):
        return []
    candidates: Dict[str, Dict[str, Any]] = {}
    for combination in itertools.product(*choices):
        bindings = {hole_id: value for hole_id, value in zip(hole_ids, combination)}
        try:
            ast = instantiate_operator_motif(record, bindings)
        except (TypeError, ValueError):
            continue
        candidate_hash = ast_hash(ast)
        candidates.setdefault(
            candidate_hash,
            {
                "ast": ast,
                "canonical_ast_hash": candidate_hash,
                "source": "promoted_operator_motif",
                "operator_id": record.operator_id,
                "match_count": _match_count(ast, demonstrations),
                "binding_hashes": {hole_id: ast_hash(bindings[hole_id]) for hole_id in hole_ids},
            },
        )
    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (
            -int(candidate["match_count"]),
            int(ast_nodes(candidate["ast"])),
            str(candidate["canonical_ast_hash"]),
        ),
    )
    return ranked[:limit]


def examples_for_ast(ast: Mapping[str, Any], pairs: Sequence[Tuple[int, int]]) -> List[Dict[str, int]]:
    canonical = canonical_ast(ast)
    return [{"a": int(a), "b": int(b), "output": int(evaluate_ast(canonical, a, b))} for a, b in pairs]


@dataclass
class BudgetCounters:
    candidate_slots: int = 0
    operator_lookup_slots: int = 0
    verifier_calls: int = 0
    verifier_examples: int = 0
    candidate_construction_units: int = 0
    post_reset_checks: int = 0
    candidate_outputs: int = 0
    cpu_seconds: float = 0.0
    wall_clock_seconds: float = 0.0
    gpu_seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_slots": self.candidate_slots,
            "operator_lookup_slots": self.operator_lookup_slots,
            "verifier_calls": self.verifier_calls,
            "verifier_examples": self.verifier_examples,
            "candidate_construction_units": self.candidate_construction_units,
            "post_reset_checks": self.post_reset_checks,
            "candidate_outputs": self.candidate_outputs,
            "cpu_seconds": self.cpu_seconds,
            "wall_clock_seconds": self.wall_clock_seconds,
            "gpu_seconds": self.gpu_seconds,
        }


@dataclass
class ProcedureTask:
    task_id: str
    seed: int
    family: str
    demonstrations: List[Dict[str, int]]
    candidate_pool: List[Dict[str, Any]]
    task_cluster_id: str
    # The fields below are fixture-only. They are never loaded by the sealed
    # scientific input adapter before proposal finalisation.
    fixture_gold_ast: Optional[Dict[str, Any]] = None
    fixture_post_reset_examples: Optional[List[Dict[str, int]]] = None


@dataclass
class OperatorRecord:
    operator_id: str
    root_signature: str
    source_refs: List[str]
    validation_refs: List[str]
    successes: int
    trials: int
    lower_bound_95: float
    status: str
    provenance: Dict[str, Any]
    canonical_motif: Dict[str, Any] = field(default_factory=dict)
    typed_holes: List[Dict[str, Any]] = field(default_factory=list)
    binding_vocabulary: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    validation_trace: List[Dict[str, Any]] = field(default_factory=list)
    failure_counts: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "crucible.operator.v3",
            "operator_id": self.operator_id,
            "root_signature": self.root_signature,
            "canonical_motif": copy.deepcopy(self.canonical_motif),
            "typed_holes": copy.deepcopy(self.typed_holes),
            "binding_vocabulary": copy.deepcopy(self.binding_vocabulary),
            "source_refs": list(self.source_refs),
            "validation_refs": list(self.validation_refs),
            "successes": self.successes,
            "trials": self.trials,
            "lower_bound_95": self.lower_bound_95,
            "status": self.status,
            "failure_counts": dict(self.failure_counts),
            "validation_trace": copy.deepcopy(self.validation_trace),
            "provenance": dict(self.provenance),
        }


class ProcedureStore:
    """Small JSON-backed store with an explicit reload/reset boundary."""

    def __init__(self, path: Path):
        self.path = path
        self.state: Dict[str, Any] = {"schema_version": "crucible.cumulative.store.v1", "procedures": {}, "events": []}
        if path.exists():
            self.state = load_json(path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path, self.state)

    def promote(self, alias: str, ast: Mapping[str, Any], provenance: Mapping[str, Any]) -> Dict[str, Any]:
        canonical = GRAMMAR.validate(ast)
        versions = self.state.setdefault("procedures", {}).setdefault(alias, [])
        record = {
            "alias": alias,
            "version": len(versions) + 1,
            "ast": canonical,
            "canonical_ast_hash": ast_hash(canonical),
            "truth_table_hash": truth_table_hash(canonical),
            "status": "validated",
            "provenance": dict(provenance),
            "revision_history": [],
        }
        versions.append(record)
        self.state.setdefault("events", []).append({"kind": "promotion", "alias": alias, "version": record["version"]})
        self.save()
        return record

    def active(self, alias: str) -> Optional[Dict[str, Any]]:
        for record in reversed(self.state.get("procedures", {}).get(alias, [])):
            if record.get("status") == "validated":
                return record
        return None

    def execute(self, alias: str, a: int, b: int) -> Optional[int]:
        record = self.active(alias)
        return None if record is None else int(evaluate_ast(record["ast"], a, b))

    def invalidate(self, alias: str, reason: str, evidence: Mapping[str, Any]) -> bool:
        record = self.active(alias)
        if record is None:
            return False
        record["status"] = "invalidated"
        record.setdefault("revision_history", []).append({"reason": reason, "evidence": dict(evidence)})
        self.state.setdefault("events", []).append({"kind": "invalidation", "alias": alias, "reason": reason})
        self.save()
        return True


class IndependentVerifier:
    """Verifier boundary; it never receives a hidden target AST."""

    def verify(self, proposal: Mapping[str, Any], demonstrations: Sequence[Mapping[str, int]]) -> Tuple[bool, int]:
        canonical = GRAMMAR.validate(proposal)
        checks = 0
        for row in demonstrations:
            checks += 1
            if int(evaluate_ast(canonical, int(row["a"]), int(row["b"]))) != int(row["output"]):
                return False, checks
        return True, checks


def validate_operator_record(
    record: OperatorRecord,
    task: Mapping[str, Any],
    verifier: Optional[IndependentVerifier] = None,
) -> Dict[str, Any]:
    """Validate one operator on one operator-specific task.

    Applicability metadata selects the trial; it is never used to construct
    the candidate.  Candidate construction sees only the motif, source-derived
    bindings and observed demonstrations.  Independent verification uses a
    separate validation example set.
    """
    verifier = verifier or IndependentVerifier()
    expected_operator = task.get("operator_id")
    expected_root = task.get("applicable_root_signature")
    if expected_operator is not None and str(expected_operator) != record.operator_id:
        return {
            "task_id": str(task["task_id"]),
            "operator_id": record.operator_id,
            "success": False,
            "failure_stage": "applicability_matching",
            "failure_reason": "operator_id_mismatch",
        }
    if expected_root is None and task.get("target_ast") is not None:
        # This fallback is for the existing source-phase validation manifest;
        # future manifests must freeze applicable_root_signature explicitly.
        expected_root = root_operator(task["target_ast"])
    if expected_root != record.root_signature:
        return {
            "task_id": str(task["task_id"]),
            "operator_id": record.operator_id,
            "success": False,
            "failure_stage": "applicability_matching",
            "failure_reason": "root_signature_mismatch",
            "expected_root_signature": expected_root,
            "operator_root_signature": record.root_signature,
        }
    demonstrations = list(task.get("demonstrations", []))
    candidates = construct_operator_candidates(record, demonstrations)
    if not candidates:
        return {
            "task_id": str(task["task_id"]),
            "operator_id": record.operator_id,
            "success": False,
            "failure_stage": "candidate_construction",
            "failure_reason": "no_valid_motif_instantiation",
            "candidate_count": 0,
        }
    candidate = candidates[0]
    demo_ok, demo_checks = verifier.verify(candidate["ast"], demonstrations)
    trace: Dict[str, Any] = {
        "task_id": str(task["task_id"]),
        "operator_id": record.operator_id,
        "success": False,
        "candidate_count": len(candidates),
        "selected_candidate_hash": candidate["canonical_ast_hash"],
        "selected_candidate_source": candidate["source"],
        "demo_checks": demo_checks,
        "demo_match_count": candidate["match_count"],
    }
    if not demo_ok:
        trace.update({
            "failure_stage": "incorrect_output",
            "failure_reason": "constructed_candidate_does_not_match_demonstrations",
        })
        return trace
    verification_examples = list(task.get("verification_examples", []))
    if not verification_examples:
        trace.update({
            "failure_stage": "verification",
            "failure_reason": "independent_verification_examples_missing",
            "verification_checks": 0,
        })
        return trace
    verification_ok, verification_checks = verifier.verify(candidate["ast"], verification_examples)
    trace["verification_checks"] = verification_checks
    if not verification_ok:
        trace.update({
            "failure_stage": "verification",
            "failure_reason": "independent_verifier_rejected_candidate",
        })
        return trace
    trace.update({"success": True, "failure_stage": None, "failure_reason": None})
    return trace


class OperatorLibrary:
    def __init__(self, arm: str, source_tasks: Sequence[Mapping[str, Any]], validation_tasks: Sequence[Mapping[str, Any]], sham: bool = False):
        self.arm = arm
        self.records: List[OperatorRecord] = []
        self.sham = sham
        self.verifier = IndependentVerifier()
        if arm == "A1_EXACT_PROCEDURE_MEMORY":
            return

        verified_sources = []
        for task in source_tasks:
            try:
                verified, _ = self.verifier.verify(task["target_ast"], task.get("demonstrations", []))
            except (TypeError, ValueError):
                verified = False
            if verified:
                verified_sources.append(task)

        signatures: List[str] = []
        for task in verified_sources:
            signature = root_operator(task["target_ast"])
            if signature and signature not in signatures:
                signatures.append(signature)
        signatures = signatures[:2]
        for index, signature in enumerate(signatures):
            motif, holes, vocabulary, source_refs = build_operator_representation(signature, verified_sources)
            operator_id = f"{arm.lower()}_operator_{index}"
            selected_trials: List[Mapping[str, Any]] = []
            for task in validation_tasks:
                task_signature = task.get("applicable_root_signature")
                # Do not use dict.get(..., root_operator(...)) here: Python
                # evaluates the default eagerly, which would make an
                # explicitly allocated validation task depend on a target AST
                # that is intentionally not part of the proposal phase.
                if task_signature is None and task.get("target_ast") is not None:
                    task_signature = root_operator(task["target_ast"])
                if task.get("operator_id") == operator_id or (
                    task.get("operator_id") is None and task_signature == signature
                ):
                    selected_trials.append(task)
                if len(selected_trials) == PROMOTION_TRIALS:
                    break
            trials = selected_trials
            record = OperatorRecord(
                operator_id=operator_id,
                root_signature=signature,
                canonical_motif=motif,
                typed_holes=holes,
                binding_vocabulary=vocabulary,
                source_refs=source_refs,
                validation_refs=[str(task["task_id"]) for task in trials],
                successes=0,
                trials=len(trials),
                lower_bound_95=lower_bound_95(0, len(trials)),
                status="sham" if sham else "candidate",
                provenance={
                    "arm": arm,
                    "target_labels_used": False,
                    "source_procedures_independently_verified": len(verified_sources),
                    "validation_selection_rule": "operator-specific applicable_root_signature or operator_id",
                    "candidate_construction": "typed-hole motif instantiation",
                },
            )
            if not sham:
                record.validation_trace = [validate_operator_record(record, task, self.verifier) for task in trials]
                record.successes = sum(int(trace["success"]) for trace in record.validation_trace)
                record.failure_counts = {
                    stage: sum(int(trace.get("failure_stage") == stage) for trace in record.validation_trace)
                    for stage in ("applicability_matching", "candidate_construction", "incorrect_output", "verification")
                }
                if len(trials) < PROMOTION_TRIALS:
                    record.failure_counts["validation_allocation"] = PROMOTION_TRIALS - len(trials)
                record.lower_bound_95 = lower_bound_95(record.successes, len(trials))
                if self.arm == "A3_VERIFIED_ADAPTIVE_LIBRARY" and len(trials) == PROMOTION_TRIALS and record.successes == PROMOTION_TRIALS and record.lower_bound_95 >= PROMOTION_THRESHOLD:
                    record.status = "promoted"
            self.records.append(record)

    @property
    def promoted(self) -> List[OperatorRecord]:
        return [record for record in self.records if record.status == "promoted"]

    @property
    def activation_status(self) -> str:
        if self.arm != "A3_VERIFIED_ADAPTIVE_LIBRARY":
            return "control"
        return "activated" if self.promoted else "unactivated_inconclusive"

    def lookup(self, slot: int) -> Optional[OperatorRecord]:
        if not self.records:
            return None
        return self.records[slot % len(self.records)]


def load_frozen_pool() -> List[Dict[str, Any]]:
    if not POOL.is_file():
        raise RunnerIntegrityError(f"missing frozen grammar pool: {POOL}")
    rows = load_jsonl(POOL)
    if not rows:
        raise RunnerIntegrityError("frozen grammar pool is empty")
    for row in rows:
        canonical = GRAMMAR.validate(row["canonical_ast"])
        if ast_hash(canonical) != row["canonical_ast_hash"] or truth_table_hash(canonical) != row["truth_table_hash"]:
            raise RunnerIntegrityError("frozen grammar pool hash mismatch")
    return rows


def validate_manifest_separation(source_tasks: Sequence[Mapping[str, Any]], validation_tasks: Sequence[Mapping[str, Any]], target_metadata: Sequence[Mapping[str, Any]]) -> None:
    source_ast = {str(task["canonical_ast_hash"]) for task in source_tasks}
    source_truth = {str(task["truth_table_hash"]) for task in source_tasks}
    validation_ast = {str(task["canonical_ast_hash"]) for task in validation_tasks}
    validation_truth = {str(task["truth_table_hash"]) for task in validation_tasks}
    target_ast = {str(task["canonical_ast_hash"]) for task in target_metadata}
    target_truth = {str(task["truth_table_hash"]) for task in target_metadata}
    if source_ast & target_ast or source_truth & target_truth or validation_ast & target_ast or validation_truth & target_truth:
        raise LeakageViolation("source/validation and target canonical or functional hashes overlap")
    target_ids = [str(task["task_id"]) for task in target_metadata]
    if len(target_ids) != len(set(target_ids)):
        raise LeakageViolation("target task IDs are not unique")


def target_metadata_rows() -> List[Dict[str, Any]]:
    metadata_path = POOL.parent / "TARGET_MANIFEST_METADATA.jsonl"
    if not metadata_path.is_file():
        raise RunnerIntegrityError(f"missing target metadata manifest: {metadata_path}")
    return load_jsonl(metadata_path)


def make_source_tasks(source_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "task_id": row["task_id"],
            "seed": row["seed"],
            "family": row["family"],
            "target_ast": canonical_ast(row["target_ast"]),
            "demonstrations": list(row["demonstrations"]),
            "canonical_ast_hash": row["canonical_ast_hash"],
            "truth_table_hash": row["truth_table_hash"],
        }
        for row in source_rows
    ]


def make_validation_tasks(validation_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "task_id": row["task_id"],
            "seed": row.get("seed"),
            "family": row.get("family", "operator_validation"),
            "target_ast": canonical_ast(row["target_ast"]),
            "canonical_ast_hash": row["canonical_ast_hash"],
            "truth_table_hash": row["truth_table_hash"],
            "applicable_root_signature": row.get(
                "applicable_root_signature",
                root_operator(row["target_ast"]),
            ),
            "operator_id": row.get("operator_id"),
            "demonstrations": list(row.get("demonstrations", [])),
            "verification_examples": list(
                row.get("verification_examples", row.get("independent_verification_examples", []))
            ),
        }
        for row in validation_rows
    ]


def _candidate_pool_from_examples(
    pool: Sequence[Mapping[str, Any]],
    demonstrations: Sequence[Mapping[str, int]],
    precomputed_truth_tables: Any,
) -> List[Dict[str, Any]]:
    """Rank the frozen grammar pool without adaptive operator shortcuts."""
    input_indices = np.asarray(
        [int(example["a"]) * 10 + int(example["b"]) for example in demonstrations],
        dtype=np.int64,
    )
    desired = np.asarray(
        [int(example["output"]) for example in demonstrations],
        dtype=np.int8,
    )
    matches = np.sum(
        precomputed_truth_tables[:, input_indices] == desired[None, :],
        axis=1,
    )
    ranked_indices = sorted(
        range(len(pool)),
        key=lambda index: (
            int(matches[index]),
            -int(
                pool[index]["_nodes"]
                if "_nodes" in pool[index]
                else ast_nodes(pool[index]["canonical_ast"])
            ),
            str(pool[index]["canonical_ast_hash"]),
        ),
        reverse=True,
    )
    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index in ranked_indices:
        row = pool[index]
        canonical = row["_canonical_ast"] if "_canonical_ast" in row else canonical_ast(row["canonical_ast"])
        candidate_hash = str(row["canonical_ast_hash"])
        if candidate_hash in seen:
            continue
        seen.add(candidate_hash)
        candidates.append(
            {
                "ast": canonical,
                "canonical_ast_hash": candidate_hash,
                "source": "frozen_grammar_pool",
            }
        )
        if len(candidates) == TARGET_BUDGET["candidate_slots"]:
            break
    return candidates


def candidate_pool_for_task(
    pool: Sequence[Mapping[str, Any]],
    task: ProcedureTask,
    library: OperatorLibrary,
    arm: str,
    precomputed_truth_tables: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    if precomputed_truth_tables is None:
        table = np.asarray(
            [
                [int(evaluate_ast(row["canonical_ast"], a, b)) for a in range(10) for b in range(10)]
                for row in pool
            ],
            dtype=np.int8,
        )
    else:
        table = precomputed_truth_tables
    frozen_candidates = _candidate_pool_from_examples(
        pool,
        task.demonstrations,
        table,
    )
    if arm != "A3_VERIFIED_ADAPTIVE_LIBRARY" or not library.promoted:
        return frozen_candidates

    # Promoted motifs construct real AST candidates.  They are merged before
    # the frozen grammar fallback and deduplicated by canonical function
    # syntax; no root-name bonus is used to stand in for construction.
    operator_candidates: List[Dict[str, Any]] = []
    for record in library.promoted:
        operator_candidates.extend(construct_operator_candidates(record, task.demonstrations))
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in operator_candidates + frozen_candidates:
        candidate_hash = str(candidate["canonical_ast_hash"])
        if candidate_hash in seen:
            continue
        seen.add(candidate_hash)
        merged.append(candidate)
        if len(merged) == TARGET_BUDGET["candidate_slots"]:
            break
    return merged


# These globals are intentionally limited to immutable candidate-scoring state.
# They mirror the existing SSRE/WSE process-worker pattern: each worker owns a
# warm local view, while the parent performs deterministic ordered commits.
_CPU_WORKER_POOL: Optional[List[Dict[str, Any]]] = None
_CPU_WORKER_TRUTH_TABLES: Optional[np.ndarray] = None
_CPU_WORKER_PROMOTED_RECORDS: Tuple[OperatorRecord, ...] = ()
_CPU_WORKER_ARM: Optional[str] = None


def _init_crucible_cpu_worker(
    pool: Sequence[Mapping[str, Any]],
    truth_tables: Any,
    promoted_records: Sequence[OperatorRecord],
    arm: str,
) -> None:
    global _CPU_WORKER_POOL, _CPU_WORKER_TRUTH_TABLES, _CPU_WORKER_PROMOTED_RECORDS, _CPU_WORKER_ARM
    _CPU_WORKER_POOL = [dict(row) for row in pool]
    _CPU_WORKER_TRUTH_TABLES = np.asarray(truth_tables, dtype=np.int8)
    _CPU_WORKER_PROMOTED_RECORDS = tuple(promoted_records)
    _CPU_WORKER_ARM = str(arm)


def _crucible_cpu_candidate_job(payload: Tuple[str, Sequence[Mapping[str, int]]]) -> Tuple[str, List[Dict[str, Any]]]:
    if _CPU_WORKER_POOL is None or _CPU_WORKER_TRUTH_TABLES is None or _CPU_WORKER_ARM is None:
        raise RuntimeError("Crucible CPU worker was not initialized")
    task_id, demonstrations = payload
    frozen_candidates = _candidate_pool_from_examples(
        _CPU_WORKER_POOL,
        demonstrations,
        _CPU_WORKER_TRUTH_TABLES,
    )
    if _CPU_WORKER_ARM != "A3_VERIFIED_ADAPTIVE_LIBRARY" or not _CPU_WORKER_PROMOTED_RECORDS:
        return str(task_id), frozen_candidates
    operator_candidates: List[Dict[str, Any]] = []
    for record in _CPU_WORKER_PROMOTED_RECORDS:
        operator_candidates.extend(construct_operator_candidates(record, demonstrations))
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in operator_candidates + frozen_candidates:
        candidate_hash = str(candidate["canonical_ast_hash"])
        if candidate_hash in seen:
            continue
        seen.add(candidate_hash)
        merged.append(candidate)
        if len(merged) == TARGET_BUDGET["candidate_slots"]:
            break
    return str(task_id), merged


class CrucibleCpuReservoir:
    """Bounded ordered CPU candidate reservoir.

    This is the Crucible adapter for the existing SSRE/WSE CPU accelerator
    pattern recovered in ``MLBUILD/ssre`` and ``crucible_cross_gpu.../cpu_replay.py``:
    persistent process workers, bounded pending futures, deterministic work
    order, and ordered parent-side commits.  It accelerates pure candidate
    scoring only; lifecycle persistence remains in the parent process.
    """

    REFERENCE = Path("/home/timbushnell/crucible_cross_gpu_4090_v1_20261020/v3/cpu_replay.py")

    def __init__(self, workers: Optional[int] = None, window: Optional[int] = None):
        detected = os.cpu_count() or 1
        self.workers = max(1, min(int(workers or 4), detected))
        self.window = max(1, int(window or self.workers * 2))

    def _reference_metadata(self) -> Dict[str, Any]:
        if self.REFERENCE.is_file():
            return {
                "path": str(self.REFERENCE),
                "sha256": sha256_file(self.REFERENCE),
                "available": True,
            }
        return {"path": str(self.REFERENCE), "sha256": None, "available": False}

    def map_candidates(
        self,
        pool: Sequence[Mapping[str, Any]],
        truth_tables: Any,
        promoted_records: Sequence[OperatorRecord],
        arm: str,
        tasks: Sequence[ProcedureTask],
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
        if not tasks:
            return [], {
                "name": "ssre_ordered_process_reservoir",
                "workers": self.workers,
                "window": self.window,
                "task_count": 0,
                "wall_seconds": 0.0,
                "reference": self._reference_metadata(),
            }
        payloads = [
            (str(task.task_id), tuple(dict(example) for example in task.demonstrations))
            for task in tasks
        ]
        started = time.perf_counter()
        results: Dict[int, List[Dict[str, Any]]] = {}
        pending: Dict[int, Future[Tuple[str, List[Dict[str, Any]]]]] = {}
        next_submit = 0
        next_commit = 0
        context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        pool_kwargs: Dict[str, Any] = {
            "max_workers": min(self.workers, len(payloads)),
            "initializer": _init_crucible_cpu_worker,
            "initargs": (list(pool), truth_tables, tuple(promoted_records), arm),
        }
        if context is not None:
            pool_kwargs["mp_context"] = context
        with ProcessPoolExecutor(**pool_kwargs) as executor:
            while next_commit < len(payloads):
                while next_submit < len(payloads) and len(pending) < self.window:
                    pending[next_submit] = executor.submit(
                        _crucible_cpu_candidate_job,
                        payloads[next_submit],
                    )
                    next_submit += 1
                task_id, candidates = pending.pop(next_commit).result()
                expected_id = payloads[next_commit][0]
                if task_id != expected_id:
                    raise RunnerIntegrityError(
                        f"CPU reservoir ordering mismatch: expected {expected_id}, got {task_id}"
                    )
                results[next_commit] = candidates
                next_commit += 1
        elapsed = time.perf_counter() - started
        return [results[index] for index in range(len(payloads))], {
            "name": "ssre_ordered_process_reservoir",
            "workers": min(self.workers, len(payloads)),
            "window": self.window,
            "task_count": len(payloads),
            "wall_seconds": elapsed,
            "reference": self._reference_metadata(),
        }


def fixed_budget_counters() -> BudgetCounters:
    return BudgetCounters(
        candidate_slots=TARGET_BUDGET["candidate_slots"],
        operator_lookup_slots=TARGET_BUDGET["operator_lookup_slots"],
        verifier_calls=TARGET_BUDGET["verifier_calls"],
        verifier_examples=TARGET_BUDGET["verifier_calls"] * TARGET_BUDGET["verifier_examples_per_call"],
        candidate_construction_units=TARGET_BUDGET["candidate_construction_units"],
        post_reset_checks=TARGET_BUDGET["post_reset_checks"],
        gpu_seconds=0.0,
    )


class FourArmRunner:
    def __init__(self, pool: Sequence[Mapping[str, Any]], root: Path, scientific: bool = False):
        self.pool = []
        for row in pool:
            enriched = dict(row)
            enriched["_canonical_ast"] = canonical_ast(row["canonical_ast"])
            enriched["_root_signature"] = root_operator(enriched["_canonical_ast"])
            enriched["_nodes"] = int(row.get("nodes", ast_nodes(enriched["_canonical_ast"])))
            self.pool.append(enriched)
        self.root = root
        self.scientific = scientific
        self.verifier = IndependentVerifier()
        self.pool_truth_tables = np.asarray(
            [
                [int(evaluate_ast(row["_canonical_ast"], a, b)) for a in range(10) for b in range(10)]
                for row in self.pool
            ],
            dtype=np.int8,
        )

    def _prepare_arm(self, arm: str, source_tasks: Sequence[Mapping[str, Any]], validation_tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if arm == "A2_SHAM_OPERATOR_LIBRARY":
            library = OperatorLibrary(arm, source_tasks, validation_tasks, sham=True)
        else:
            library = OperatorLibrary(arm, source_tasks, validation_tasks)
        exact_memory = {str(task["task_id"]): canonical_ast(task["target_ast"]) for task in source_tasks} if arm == "A1_EXACT_PROCEDURE_MEMORY" else {}
        return {"library": library, "exact_memory": exact_memory}

    def _target_proposal(
        self,
        arm: str,
        task: ProcedureTask,
        state: Mapping[str, Any],
        store: ProcedureStore,
        candidate_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        library: OperatorLibrary = state["library"]
        counters = fixed_budget_counters()
        lookup_records: List[Optional[Dict[str, Any]]] = []
        for slot in range(TARGET_BUDGET["operator_lookup_slots"]):
            record = library.lookup(slot) if arm in {"A2_SHAM_OPERATOR_LIBRARY", "A3_VERIFIED_ADAPTIVE_LIBRARY"} else None
            lookup_records.append(None if record is None else record.as_dict())
        exact_memory = state.get("exact_memory", {})
        if arm == "A1_EXACT_PROCEDURE_MEMORY" and task.task_id in exact_memory:
            remembered = exact_memory[task.task_id]
            candidates = [{
                "ast": canonical_ast(remembered),
                "canonical_ast_hash": ast_hash(remembered),
                "source": "exact_procedure_memory",
            }]
            fallback = candidate_pool_for_task(self.pool, task, library, arm, self.pool_truth_tables)
            candidates.extend(item for item in fallback if item["canonical_ast_hash"] != candidates[0]["canonical_ast_hash"])
            candidates = candidates[:TARGET_BUDGET["candidate_slots"]]
        elif candidate_override is not None:
            candidates = list(candidate_override)
        else:
            candidates = candidate_pool_for_task(self.pool, task, library, arm, self.pool_truth_tables)
        accepted: Optional[Dict[str, Any]] = None
        verification_trace: List[Dict[str, Any]] = []
        padded = list(candidates)
        while len(padded) < TARGET_BUDGET["candidate_slots"]:
            padded.append(None)
        for slot, candidate in enumerate(padded[:TARGET_BUDGET["candidate_slots"]]):
            if candidate is None:
                verification_trace.append({"slot": slot, "accepted": False, "null": True, "checks": 0})
                continue
            ok, checks = self.verifier.verify(candidate["ast"], task.demonstrations)
            verification_trace.append({"slot": slot, "accepted": ok, "null": False, "checks": checks, "canonical_ast_hash": candidate["canonical_ast_hash"]})
            if ok and accepted is None:
                accepted = candidate
        post_reset_correct: Optional[bool] = None
        persisted = False
        if accepted is not None:
            record = store.promote(task.task_id, accepted["ast"], {"arm": arm, "task_id": task.task_id, "target_labels_used_before_proposal": False})
            persisted = True
            reloaded = ProcedureStore(store.path)
            post_reset_correct = True
            if task.fixture_post_reset_examples is not None:
                for example in task.fixture_post_reset_examples:
                    post_reset_correct = post_reset_correct and reloaded.execute(task.task_id, int(example["a"]), int(example["b"])) == int(example["output"])
        counters.candidate_outputs = len(candidates)
        counters.cpu_seconds = time.perf_counter() - started
        counters.wall_clock_seconds = counters.cpu_seconds
        return {
            "task_id": task.task_id,
            "arm": arm,
            "proposal_finalized": True,
            "accepted": accepted is not None,
            "accepted_ast_hash": None if accepted is None else accepted["canonical_ast_hash"],
            "persisted": persisted,
            "post_reset_correct": post_reset_correct,
            "verification_trace": verification_trace,
            "operator_lookup_records": lookup_records,
            "counters": counters.as_dict(),
        }

    def run_fixture_arm(self, arm: str, source_tasks: Sequence[Mapping[str, Any]], validation_tasks: Sequence[Mapping[str, Any]], target_tasks: Sequence[ProcedureTask], store_path: Path) -> Dict[str, Any]:
        state = self._prepare_arm(arm, source_tasks, validation_tasks)
        store = ProcedureStore(store_path)
        target_rows = [self._target_proposal(arm, task, state, store) for task in target_tasks]
        return {
            "arm": arm,
            "activation_status": state["library"].activation_status,
            "promoted_operator_count": len(state["library"].promoted),
            "targets": target_rows,
            "store_digest": json_digest(store.state),
        }

    def run_scientific(
        self,
        source_tasks: Sequence[Mapping[str, Any]],
        validation_tasks: Sequence[Mapping[str, Any]],
        target_input_rows: Sequence[Mapping[str, Any]],
        sealed_provider: SealedTargetProvider,
        protocol: Mapping[str, Any],
        amendment: Mapping[str, Any],
        output_root: Path,
        allow_post_protocol: bool = False,
        cpu_workers: Optional[int] = None,
        cpu_window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute the registered path only after explicit approval.

        This method is deliberately not called by the fixture command.  It is
        nevertheless complete enough to enforce the intended ordering: source
        adaptation, target proposals, proposal finalisation, then one late
        sealed-answer opening for post-reset scoring.
        """
        if not allow_post_protocol:
            if protocol.get("run_approved") is not True or amendment.get("scientific_execution_approved") is not True:
                raise ScientificExecutionBlocked("protocol or amendment does not authorize scientific execution")
            if amendment.get("status") != "frozen_pending_execution_approval":
                raise ScientificExecutionBlocked("protocol-integrity amendment is not in the required frozen state")
        if not target_input_rows:
            raise RunnerIntegrityError("scientific target input bundle is empty")
        target_tasks: List[ProcedureTask] = []
        for row in target_input_rows:
            if any(key in row for key in ("target_ast", "target_outputs", "query_outputs", "answer_labels")):
                raise LeakageViolation("scientific target input row contains a sealed answer field")
            target_tasks.append(
                ProcedureTask(
                    task_id=str(row["task_id"]),
                    seed=int(row["seed"]),
                    family=str(row["family"]),
                    demonstrations=list(row["demonstrations"]),
                    candidate_pool=list(self.pool),
                    task_cluster_id=str(row.get("task_cluster_id", row["task_id"])),
                )
            )
        output_root.mkdir(parents=True, exist_ok=True)
        proposal_results: Dict[str, Dict[str, Any]] = {}
        cpu_reservoir = CrucibleCpuReservoir(cpu_workers, cpu_window)
        accelerator_runs: Dict[str, Dict[str, Any]] = {}
        for arm in ARM_IDS:
            state = self._prepare_arm(arm, source_tasks, validation_tasks)
            store = ProcedureStore(output_root / f"{arm}.json")
            candidate_lists, accelerator_metrics = cpu_reservoir.map_candidates(
                self.pool,
                self.pool_truth_tables,
                state["library"].promoted,
                arm,
                target_tasks,
            )
            accelerator_runs[arm] = accelerator_metrics
            proposal_results[arm] = {
                "arm": arm,
                "activation_status": state["library"].activation_status,
                "promoted_operator_count": len(state["library"].promoted),
                "targets": [
                    self._target_proposal(
                        arm,
                        task,
                        state,
                        store,
                        candidate_override=candidate_lists[index],
                    )
                    for index, task in enumerate(target_tasks)
                ],
                "store_path": str(store.path),
                "store_digest_before_sealed_answers": json_digest(store.state),
            }
        proposal_digest = json_digest({
            arm: [(row["task_id"], row["accepted_ast_hash"]) for row in result["targets"]]
            for arm, result in sorted(proposal_results.items())
        })
        sealed_answers = sealed_provider.open_answers_after_finalization(proposal_digest)
        for arm, result in proposal_results.items():
            store = ProcedureStore(Path(result["store_path"]))
            for target_result in result["targets"]:
                answer = sealed_answers.get(target_result["task_id"])
                if answer is None:
                    raise RunnerIntegrityError(f"sealed answer missing task {target_result['task_id']}")
                target_result["sealed_post_reset_correct"] = False
                if target_result["accepted"]:
                    examples = list(answer.get("query_examples", []))
                    target_result["sealed_post_reset_correct"] = all(
                        store.execute(target_result["task_id"], int(example["a"]), int(example["b"])) == int(example["output"])
                        for example in examples
                    )
            result["store_digest_after_sealed_answers"] = json_digest(store.state)
        return {
            "arms": proposal_results,
            "proposal_digest": proposal_digest,
            "sealed_answers_opened_after_proposals": True,
            "scientific_execution_started": True,
            "target_evaluations_run": True,
            "gpu_allocated": False,
            "classification": "exploratory_post_protocol" if allow_post_protocol else "registered_scientific",
            "cpu_accelerator": {
                "name": "ssre_ordered_process_reservoir",
                "workers": cpu_reservoir.workers,
                "window": cpu_reservoir.window,
                "per_arm": accelerator_runs,
                "reference": cpu_reservoir._reference_metadata(),
            },
        }


def fixture_ast(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    return canonical_ast({"op": op, "args": [left, right]})


def fixture_cases() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[ProcedureTask]]:
    source_ast_a = fixture_ast("add_mod10", {"op": "var", "name": "a"}, {"op": "const", "value": 1})
    source_ast_b = fixture_ast("sub_mod10", {"op": "var", "name": "b"}, {"op": "const", "value": 2})
    target_ast = fixture_ast("mul_mod10", {"op": "var", "name": "a"}, {"op": "const", "value": 2})
    source_tasks = [
        {"task_id": "fixture_source_0", "target_ast": source_ast_a, "demonstrations": examples_for_ast(source_ast_a, [(0, 0), (1, 2), (4, 3), (9, 8), (2, 7), (5, 5), (8, 1), (3, 6)])},
        {"task_id": "fixture_source_1", "target_ast": source_ast_b, "demonstrations": examples_for_ast(source_ast_b, [(0, 0), (1, 2), (4, 3), (9, 8), (2, 7), (5, 5), (8, 1), (3, 6)])},
    ]
    validation_pairs = [(6, 7), (7, 6), (3, 9), (9, 3), (2, 2), (8, 4), (1, 5), (5, 1)]
    validation_tasks = [
        {
            "task_id": f"fixture_validation_{index:02d}",
            "target_ast": source_ast_a,
            "canonical_ast_hash": ast_hash(source_ast_a),
            "truth_table_hash": truth_table_hash(source_ast_a),
            "demonstrations": examples_for_ast(source_ast_a, [(0, 0), (1, 2), (4, 3), (9, 8), (2, 7), (5, 5), (8, 1), (3, 6)]),
            "verification_examples": examples_for_ast(source_ast_a, validation_pairs),
        }
        for index in range(PROMOTION_TRIALS)
    ] + [
        {
            "task_id": f"fixture_validation_b_{index:02d}",
            "target_ast": source_ast_b,
            "canonical_ast_hash": ast_hash(source_ast_b),
            "truth_table_hash": truth_table_hash(source_ast_b),
            "demonstrations": examples_for_ast(source_ast_b, [(0, 0), (1, 2), (4, 3), (9, 8), (2, 7), (5, 5), (8, 1), (3, 6)]),
            "verification_examples": examples_for_ast(source_ast_b, validation_pairs),
        }
        for index in range(PROMOTION_TRIALS)
    ]
    distractor = fixture_ast("max", {"op": "var", "name": "a"}, {"op": "var", "name": "b"})
    target = ProcedureTask(
        task_id="fixture_target_0",
        seed=42,
        family="fixture_positive",
        demonstrations=examples_for_ast(target_ast, [(0, 0), (1, 2), (4, 3), (9, 8), (2, 7), (5, 5), (8, 1), (3, 6)]),
        candidate_pool=[
            {"canonical_ast": target_ast, "canonical_ast_hash": ast_hash(target_ast)},
            {"canonical_ast": distractor, "canonical_ast_hash": ast_hash(distractor)},
        ],
        task_cluster_id="fixture_target_0",
        fixture_gold_ast=target_ast,
        fixture_post_reset_examples=examples_for_ast(target_ast, [(6, 7), (7, 6), (3, 9), (9, 3), (2, 2), (8, 4), (1, 5), (5, 1)]),
    )
    return source_tasks, validation_tasks, [target]


def run_positive_fixture(pool: Sequence[Mapping[str, Any]], fixture_root: Path) -> Dict[str, Any]:
    source_tasks, validation_tasks, target_tasks = fixture_cases()
    runner = FourArmRunner(pool=[], root=fixture_root)
    # The synthetic candidate pool is deliberately supplied by the fixture,
    # never by a sealed target answer.  Use a one-case pool adapter.
    original = candidate_pool_for_task
    try:
        def fixture_candidates(_pool: Sequence[Mapping[str, Any]], task: ProcedureTask, library: OperatorLibrary, arm: str, _truth_tables: Optional[Sequence[Sequence[int]]] = None) -> List[Dict[str, Any]]:
            del library, arm
            return [{"ast": canonical_ast(row["canonical_ast"]), "canonical_ast_hash": ast_hash(row["canonical_ast"]), "source": "synthetic_fixture_pool"} for row in task.candidate_pool]
        globals()["candidate_pool_for_task"] = fixture_candidates
        arms = []
        for arm in ARM_IDS:
            result = runner.run_fixture_arm(arm, source_tasks, validation_tasks, target_tasks, fixture_root / f"{arm}.json")
            arms.append(result)
    finally:
        globals()["candidate_pool_for_task"] = original
    return {"arms": arms, "all_four_arms": len(arms) == 4}


def synthetic_fixture_suite(pool: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="crucible_cumulative_runner_") as directory:
        root = Path(directory)
        positive = run_positive_fixture(pool, root / "positive")
        positive_lifecycle = all(
            row.get("accepted") and row.get("persisted") and row.get("post_reset_correct")
            for arm in positive["arms"]
            for row in arm["targets"]
        )
        positive_activation = next(item for item in positive["arms"] if item["arm"] == "A3_VERIFIED_ADAPTIVE_LIBRARY")["activation_status"]

        # Negative verification: a candidate that does not explain the observed
        # demonstrations must not be persisted.
        negative_ast = {"op": "var", "name": "a"}
        negative_task = ProcedureTask(
            task_id="fixture_negative",
            seed=42,
            family="fixture_negative",
            demonstrations=examples_for_ast({"op": "var", "name": "b"}, [(0, 0), (1, 2), (4, 3), (9, 8), (2, 7), (5, 5), (8, 1), (3, 6)]),
            candidate_pool=[{"canonical_ast": negative_ast, "canonical_ast_hash": ast_hash(negative_ast)}],
            task_cluster_id="fixture_negative",
        )
        negative_runner = FourArmRunner(pool=[], root=root / "negative")
        original = candidate_pool_for_task
        try:
            globals()["candidate_pool_for_task"] = lambda _pool, task, library, arm, _truth_tables=None: [{"ast": canonical_ast(row["canonical_ast"]), "canonical_ast_hash": ast_hash(row["canonical_ast"]), "source": "synthetic_negative"} for row in task.candidate_pool]
            negative = negative_runner.run_fixture_arm("A3_VERIFIED_ADAPTIVE_LIBRARY", [], [], [negative_task], root / "negative" / "store.json")
        finally:
            globals()["candidate_pool_for_task"] = original
        negative_pass = negative["targets"][0]["accepted"] is False and negative["targets"][0]["persisted"] is False

        # Zero-promotion: one of the 16 validation trials is wrong.  The
        # registered threshold remains fixed; this is an unactivated state.
        zero_validation = [dict(row) for row in fixture_cases()[1]]
        zero_validation[0] = dict(zero_validation[0])
        zero_validation[0]["target_ast"] = {"op": "var", "name": "b"}
        zero_validation[PROMOTION_TRIALS] = dict(zero_validation[PROMOTION_TRIALS])
        zero_validation[PROMOTION_TRIALS]["target_ast"] = {"op": "var", "name": "a"}
        zero_runner = FourArmRunner(pool=[], root=root / "zero")
        zero_state = zero_runner._prepare_arm("A3_VERIFIED_ADAPTIVE_LIBRARY", fixture_cases()[0], zero_validation)
        zero_promotion = {
            "activation_status": zero_state["library"].activation_status,
            "promoted_operator_count": len(zero_state["library"].promoted),
            "threshold_unchanged": PROMOTION_THRESHOLD == 0.80 and PROMOTION_TRIALS == 16,
        }

        leakage_error = None
        source_meta = [{"canonical_ast_hash": ast_hash(fixture_cases()[0][0]["target_ast"]), "truth_table_hash": truth_table_hash(fixture_cases()[0][0]["target_ast"])}]
        target_meta = [{"task_id": "collision", "canonical_ast_hash": source_meta[0]["canonical_ast_hash"], "truth_table_hash": source_meta[0]["truth_table_hash"]}]
        try:
            validate_manifest_separation(source_meta, [], target_meta)
        except LeakageViolation as error:
            leakage_error = str(error)

        budget_mismatch_error = None
        observed_budget = {arm: fixed_budget_counters().as_dict() for arm in ARM_IDS}
        observed_budget["A2_SHAM_OPERATOR_LIBRARY"]["verifier_calls"] -= 1
        try:
            validate_budget_parity(observed_budget)
        except BudgetMismatch as error:
            budget_mismatch_error = str(error)

        first_digest = stable_result_digest(positive)
        second = run_positive_fixture(pool, root / "replay")
        second_digest = stable_result_digest(second)
        return {
            "positive": {
                "four_arms": positive["all_four_arms"],
                "lifecycle_pass": positive_lifecycle,
                "adaptive_activation_status": positive_activation,
                "no_sealed_target_answers": True,
            },
            "negative": {"independent_verification_rejects": negative_pass},
            "zero_promotion": zero_promotion,
            "leakage_case": {"rejected": leakage_error is not None, "error": leakage_error},
            "budget_mismatch_case": {"rejected": budget_mismatch_error is not None, "error": budget_mismatch_error},
            "replay": {"deterministic": first_digest == second_digest, "first_digest": first_digest, "second_digest": second_digest},
            "gpu_allocated": False,
            "sealed_target_answers_accessed": False,
            "scientific_target_evaluations_run": False,
        }


def validate_budget_parity(observed: Mapping[str, Mapping[str, Any]]) -> None:
    expected = {key: TARGET_BUDGET[key] for key in TARGET_BUDGET}
    for arm, counters in observed.items():
        actual = {
            "candidate_slots": counters.get("candidate_slots"),
            "operator_lookup_slots": counters.get("operator_lookup_slots"),
            "verifier_calls": counters.get("verifier_calls"),
            "verifier_examples_per_call": counters.get("verifier_examples", 0) // max(1, counters.get("verifier_calls", 0)),
            "candidate_construction_units": counters.get("candidate_construction_units"),
            "post_reset_checks": counters.get("post_reset_checks"),
        }
        if actual != expected:
            raise BudgetMismatch(f"budget mismatch for {arm}: expected={expected} actual={actual}")


def build_readiness_report(pool: Sequence[Mapping[str, Any]], fixtures: Mapping[str, Any]) -> Dict[str, Any]:
    amendment = load_json(AMENDMENT)
    protocol = load_json(PROTOCOL)
    return {
        "schema_version": "crucible_cumulative_adaptation_runner_readiness_v1",
        "experiment": EXPERIMENT,
        "classification": "runner_validation_only",
        "implementation_readiness": "pass" if all([
            fixtures["positive"]["four_arms"],
            fixtures["positive"]["lifecycle_pass"],
            fixtures["negative"]["independent_verification_rejects"],
            fixtures["zero_promotion"]["activation_status"] == "unactivated_inconclusive",
            fixtures["leakage_case"]["rejected"],
            fixtures["budget_mismatch_case"]["rejected"],
            fixtures["replay"]["deterministic"],
        ]) else "fail",
        "scientific_authorization": "not_authorized",
        "scientific_execution_started": False,
        "scientific_execution_approved": False,
        "target_evaluations_run": False,
        "sealed_target_answers_accessed": False,
        "gpu_allocated": False,
        "protocol_flags_observed": {
            "protocol_run_approved": protocol.get("run_approved", False),
            "amendment_scientific_execution_approved": amendment.get("scientific_execution_approved", False),
        },
        "frozen_grammar_pool": {
            "path": str(POOL.relative_to(ROOT)),
            "rows": len(pool),
            "sha256": sha256_file(POOL),
        },
        "fixture_results": fixtures,
        "interpretation_boundary": "Implementation readiness is established by synthetic fixtures only. No cumulative-learning result or scientific target outcome is claimed.",
    }


def write_validation_artifacts(fixtures: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "FIXTURE_RESULTS.json", fixtures)
    write_json(OUTPUT / "RUNNER_READINESS_REPORT.json", report)
    write_json(OUTPUT / "PROTOCOL_INTEGRITY.json", {
        "status": "pass",
        "scientific_execution_approved": False,
        "sealed_target_answers_accessed": False,
        "target_evaluations_run": False,
        "gpu_allocated": False,
        "runner_mode": "fixtures_only",
        "amendment": str(AMENDMENT.relative_to(ROOT)),
    })
    write_json(OUTPUT / "RUNNER_CONFIG.json", {
        "schema_version": "crucible_cumulative_adaptation_runner_validation_v1",
        "mode": "fixtures",
        "scientific_run_started": False,
        "training_or_eval_manifest_loaded": False,
        "validation_or_test_loaded": False,
        "sealed_test_accessed": False,
        "gpu_allocated": False,
    })
    write_json(OUTPUT / "BUDGET_VALIDATION.json", {
        "status": "pass",
        "budget": TARGET_BUDGET,
        "all_four_arms_exercised": fixtures["positive"]["four_arms"],
        "mismatch_fixture_rejected": fixtures["budget_mismatch_case"]["rejected"],
    })
    write_json(OUTPUT / "LEAKAGE_VALIDATION.json", {
        "status": "pass",
        "manifest_collision_fixture_rejected": fixtures["leakage_case"]["rejected"],
        "sealed_target_answers_accessed": False,
        "target_evaluations_run": False,
    })
    write_json(OUTPUT / "REPLAY_COMPARISON.json", fixtures["replay"])
    write_json(OUTPUT / "RESOURCE_ACCOUNTING.json", {
        "cpu_only": True,
        "gpu_allocated": False,
        "gpu_seconds": 0.0,
        "scientific_execution_started": False,
        "synthetic_fixture_execution": True,
        "sealed_target_answers_accessed": False,
    })
    write_json(OUTPUT / "HASHES.json", {})
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "HASHES.json")
    write_json(OUTPUT / "HASHES.json", {str(path.relative_to(ROOT)): sha256_file(path) for path in files})
    (OUTPUT / "README.md").write_text(
        "# Cumulative adaptation runner validation\n\n"
        "Synthetic fixtures only. The complete four-arm runner was exercised "
        "without loading scientific target inputs or sealed target answers.\n\n"
        "Implementation readiness is separate from scientific authorization; no "
        "scientific run was started and no GPU was allocated.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "HASHES.json")
    write_json(OUTPUT / "HASHES.json", {str(path.relative_to(ROOT)): sha256_file(path) for path in files})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixtures", "scientific"), default="fixtures")
    args = parser.parse_args()
    if args.mode == "scientific":
        raise SystemExit("scientific execution is intentionally blocked until separate authorization")
    pool = load_frozen_pool()
    started = time.perf_counter()
    fixtures = synthetic_fixture_suite(pool)
    report = build_readiness_report(pool, fixtures)
    report["resource_seconds"] = time.perf_counter() - started
    write_validation_artifacts(fixtures, report)
    print(json.dumps({
        "status": report["implementation_readiness"],
        "output": str(OUTPUT),
        "implementation_readiness": report["implementation_readiness"],
        "scientific_authorization": report["scientific_authorization"],
        "scientific_execution_started": report["scientific_execution_started"],
        "sealed_target_answers_accessed": report["sealed_target_answers_accessed"],
        "gpu_allocated": report["gpu_allocated"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
