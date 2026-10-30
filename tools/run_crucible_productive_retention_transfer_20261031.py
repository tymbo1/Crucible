#!/usr/bin/env python3
"""CPU lifecycle runner and integrity harness for productive-retention transfer.

This runner deliberately has two modes:

* ``fixtures`` validates the exact frozen public manifest and exercises all
  C0--C3 state transitions without reading sealed answers or training a neural
  model.  It is the only mode permitted while the protocol is unapproved.
* ``scientific`` is a fail-closed placeholder for the later neural backend.
  It refuses to run while the frozen protocol says scientific authorization is
  false.  This prevents a readiness check from becoming an accidental run.

The fixture backend is an instrumentation backend, not evidence of transfer.
It records the state boundary that a future neural backend must implement:
source competence certificate, source reset, source-only memory retrieval,
frozen source teacher anchoring, matched target exposure and persistence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "experiments" / "crucible_productive_retention_transfer_20261031"
DEFAULT_MANIFEST = DEFAULT_EXPERIMENT / "manifest"
SCHEMA = "crucible-productive-retention-transfer-runner-v1-20261031"
CONDITIONS = ("C0", "C1", "C2", "C3")
EXPECTED_SPLIT_COUNTS = {
    "source_train": 32,
    "source_validation": 16,
    "target_dependent_train": 64,
    "target_dependent_validation": 32,
    "target_dependent_sealed": 64,
    "target_independent_train": 64,
    "target_independent_validation": 32,
    "target_independent_sealed": 64,
}
SOURCE_GATE_THRESHOLD = 0.80
SOURCE_GATE_MIN_CORRECT = 13
SOURCE_GATE_TRIALS = 16
SOURCE_UPDATES = 1000
TARGET_UPDATES = 3000
TARGET_CHECKPOINTS = (250, 500, 1000, 2000, 3000)
FIXTURE_CANDIDATE_BUDGET = 4
FIXTURE_VERIFIER_BUDGET = 4


class RunnerIntegrityError(RuntimeError):
    """Base class for fail-closed runner errors."""


class ScientificExecutionBlocked(RunnerIntegrityError):
    pass


class LeakageViolation(RunnerIntegrityError):
    pass


class BudgetMismatch(RunnerIntegrityError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items() if key not in {"wall_seconds", "cpu_seconds"}}
    if isinstance(value, list):
        return [stable(item) for item in value]
    return value


@dataclass
class ManifestView:
    root: Path
    manifest: Dict[str, Any]
    rows: Dict[str, List[Dict[str, Any]]]
    public_digest: str

    @classmethod
    def load(cls, root: Path) -> "ManifestView":
        root = root.resolve()
        manifest_path = root / "MANIFEST.json"
        if not manifest_path.is_file():
            raise RunnerIntegrityError(f"missing manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        if manifest.get("scientific_authorization") is not False:
            raise ScientificExecutionBlocked("manifest does not carry scientific_authorization=false")
        if manifest.get("classification") != "cpu_manifest_and_leakage_preflight_only":
            raise RunnerIntegrityError("unexpected manifest classification")

        rows: Dict[str, List[Dict[str, Any]]] = {}
        for relative, expected_hash in manifest.get("public_files", {}).items():
            path = root / relative
            if not path.is_file():
                raise RunnerIntegrityError(f"missing public manifest file: {relative}")
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise RunnerIntegrityError(f"public manifest hash mismatch: {relative}")
            seed = path.parent.name
            split = path.name.removesuffix("_public.jsonl").removesuffix(".jsonl")
            key = f"{seed}/{split}"
            loaded = read_jsonl(path)
            rows[key] = loaded
            for row in loaded:
                required = {
                    "task_id",
                    "family",
                    "split",
                    "canonical_ast_sha256",
                    "truth_table_sha256",
                    "demonstrations",
                    "query",
                }
                if not required.issubset(row):
                    raise RunnerIntegrityError(f"incomplete public row: {row.get('task_id', '<unknown>')}")
                if row["split"].endswith("sealed") and "query_output" in row:
                    raise LeakageViolation(f"sealed answer exposed in public manifest: {row['task_id']}")

        validation_report = root / "INDEPENDENT_VALIDATION_REPORT.json"
        if not validation_report.is_file() or read_json(validation_report).get("status") != "PASS":
            raise RunnerIntegrityError("independent manifest validation is not PASS")

        for seed in manifest.get("seeds", []):
            for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
                key = f"seed_{seed}/{split}"
                actual_count = len(rows.get(key, []))
                if actual_count != expected_count:
                    raise RunnerIntegrityError(
                        f"manifest count mismatch for {key}: expected {expected_count}, got {actual_count}"
                    )
        return cls(root=root, manifest=manifest, rows=rows, public_digest=digest(manifest))

    def split(self, seed: int, name: str) -> List[Dict[str, Any]]:
        return list(self.rows[f"seed_{seed}/{name}"])

    def all_target_training_ids(self, seed: int) -> Tuple[str, ...]:
        rows = self.split(seed, "target_dependent_train") + self.split(seed, "target_independent_train")
        return tuple(row["task_id"] for row in rows)

    def all_target_validation_ids(self, seed: int) -> Tuple[str, ...]:
        rows = self.split(seed, "target_dependent_validation") + self.split(seed, "target_independent_validation")
        return tuple(row["task_id"] for row in rows)


@dataclass
class RunnerState:
    condition: str
    seed: int
    theta_digest: str
    optimizer_digest: str
    rng_digest: str
    source_skill: bool = False
    source_certificate: Dict[str, Any] | None = None
    source_memory: Tuple[str, ...] = ()
    source_teacher_digest: str | None = None
    target_transient_ids: List[str] = field(default_factory=list)
    source_updates: int = 0
    target_updates: int = 0
    candidate_calls: int = 0
    verifier_calls: int = 0
    event_log: List[Dict[str, Any]] = field(default_factory=list)

    def persistent_payload(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "theta_digest": self.theta_digest,
            "optimizer_digest": self.optimizer_digest,
            "rng_digest": self.rng_digest,
            "source_skill": self.source_skill,
            "source_certificate": self.source_certificate,
            "source_memory": list(self.source_memory),
            "source_teacher_digest": self.source_teacher_digest,
            "source_updates": self.source_updates,
            "target_updates": self.target_updates,
            "candidate_calls": self.candidate_calls,
            "verifier_calls": self.verifier_calls,
        }

    def transient_payload(self) -> Dict[str, Any]:
        return {"target_transient_ids": list(self.target_transient_ids)}


class FixtureBackend:
    """Synthetic state backend used only to validate runner boundaries."""

    def __init__(self, condition: str, seed: int, manifest: ManifestView):
        if condition not in CONDITIONS:
            raise RunnerIntegrityError(f"unknown condition {condition}")
        self.condition = condition
        self.seed = seed
        self.manifest = manifest
        theta0 = digest({"kind": "theta0", "seed": seed, "manifest": manifest.public_digest})
        self.state = RunnerState(
            condition=condition,
            seed=seed,
            theta_digest=theta0,
            optimizer_digest=digest({"kind": "optimizer0", "seed": seed}),
            rng_digest=digest({"kind": "rng0", "seed": seed}),
        )
        self.theta0 = self.state.theta_digest
        self.optimizer0 = self.state.optimizer_digest
        self.rng0 = self.state.rng_digest
        self._source_teacher_snapshot: str | None = None
        self._target_schedule: Tuple[str, ...] = ()

    def _record(self, event: str, **payload: Any) -> None:
        self.state.event_log.append({"event": event, **payload})

    def acquire_source(self) -> None:
        source_train = self.manifest.split(self.seed, "source_train")
        if len(source_train) != 32:
            raise RunnerIntegrityError("source training count is not frozen at 32")
        self.state.source_updates = SOURCE_UPDATES
        self.state.source_skill = True
        self.state.theta_digest = digest({"kind": "theta_source", "theta0": self.theta0, "seed": self.seed})
        self.state.optimizer_digest = digest({"kind": "optimizer_source", "seed": self.seed})
        self.state.rng_digest = digest({"kind": "rng_source", "seed": self.seed})
        self._record("source_acquired", task_ids=[row["task_id"] for row in source_train], updates=SOURCE_UPDATES)

    def source_gate(self) -> Dict[str, Any]:
        validation = self.manifest.split(self.seed, "source_validation")
        # This oracle is deliberately fixture-only. It tests gate accounting and
        # state routing; it is never used by scientific mode or target scoring.
        correct = len(validation)
        result = {
            "correct": correct,
            "trials": len(validation),
            "accuracy": correct / max(1, len(validation)),
            "threshold": SOURCE_GATE_THRESHOLD,
            "pass": correct / max(1, len(validation)) >= SOURCE_GATE_THRESHOLD,
            "fixture_oracle": True,
        }
        if not result["pass"]:
            raise RunnerIntegrityError("source competence gate failed")
        self.state.source_certificate = {
            "family": "shallow_boolean_composition",
            "validation_manifest_digest": digest(validation),
            "accuracy": result["accuracy"],
            "fixture_only": True,
        }
        self._record("source_gate_passed", **result)
        return result

    def configure_condition(self) -> None:
        if self.state.source_certificate is None:
            raise RunnerIntegrityError("condition configuration requires source certificate")
        if self.condition == "C0":
            self.reset_to_theta0(reason="source_skill_ablation")
        elif self.condition == "C1":
            self._record("condition_configured", source_skill=True, source_memory=False, teacher=False)
        elif self.condition == "C2":
            source_ids = tuple(row["task_id"] for row in self.manifest.split(self.seed, "source_train"))
            self.state.source_skill = False
            self.state.source_memory = source_ids
            self.state.source_teacher_digest = None
            self._record("condition_configured", source_skill=False, source_memory=True, teacher=False)
        elif self.condition == "C3":
            self.state.source_teacher_digest = digest(
                {"source_checkpoint": self.state.theta_digest, "source_certificate": self.state.source_certificate}
            )
            self._source_teacher_snapshot = self.state.source_teacher_digest
            self._record("condition_configured", source_skill=True, source_memory=False, teacher=True)

    def reset_to_theta0(self, reason: str) -> None:
        self.state.theta_digest = self.theta0
        self.state.optimizer_digest = self.optimizer0
        self.state.rng_digest = self.rng0
        self.state.source_skill = False
        self.state.source_memory = ()
        self.state.source_teacher_digest = None
        self._record("source_reset", reason=reason, theta_digest=self.theta0, optimizer_digest=self.optimizer0, rng_digest=self.rng0)
        if self.state.theta_digest != self.theta0 or self.state.optimizer_digest != self.optimizer0 or self.state.rng_digest != self.rng0:
            raise RunnerIntegrityError("C0 reset did not restore theta, optimizer and RNG state")

    def target_training(self) -> Tuple[str, ...]:
        dependent = self.manifest.split(self.seed, "target_dependent_train")
        independent = self.manifest.split(self.seed, "target_independent_train")
        schedule = tuple(row["task_id"] for row in dependent + independent)
        self._target_schedule = schedule
        total_tasks = len(schedule)
        base_updates, remainder_updates = divmod(TARGET_UPDATES, total_tasks)
        for task_index, row in enumerate(dependent + independent):
            self.state.target_transient_ids.append(row["task_id"])
            self.state.target_updates += base_updates + int(task_index < remainder_updates)
            self.state.candidate_calls += FIXTURE_CANDIDATE_BUDGET
            self.state.verifier_calls += FIXTURE_VERIFIER_BUDGET
            if self.condition == "C2":
                retrieved = set(self.state.source_memory)
                if not retrieved.issubset({item["task_id"] for item in self.manifest.split(self.seed, "source_train")}):
                    raise LeakageViolation("C2 retrieved non-source memory")
        if self.condition == "C3" and self.state.source_teacher_digest != self._source_teacher_snapshot:
            raise RunnerIntegrityError("C3 source teacher changed during target training")
        if self.state.target_updates != TARGET_UPDATES:
            raise BudgetMismatch(f"target update budget mismatch: {self.state.target_updates}")
        self._record(
            "target_training_complete",
            task_ids=list(schedule),
            updates=self.state.target_updates,
            candidate_calls=self.state.candidate_calls,
            verifier_calls=self.state.verifier_calls,
            source_skill=self.state.source_skill,
            source_memory_count=len(self.state.source_memory),
            teacher_digest=self.state.source_teacher_digest,
        )
        return schedule

    def validation_schedule(self) -> Dict[str, Tuple[str, ...]]:
        dependent = self.manifest.split(self.seed, "target_dependent_validation")
        independent = self.manifest.split(self.seed, "target_independent_validation")
        if len(dependent) != 32 or len(independent) != 32:
            raise RunnerIntegrityError("target validation counts are not frozen at 32 per family")
        return {
            "dependent": tuple(row["task_id"] for row in dependent),
            "independent": tuple(row["task_id"] for row in independent),
        }

    def persist_and_reset(self) -> Dict[str, Any]:
        persisted = self.state.persistent_payload()
        persisted_digest = digest(persisted)
        reloaded = json.loads(json.dumps(persisted, sort_keys=True))
        if digest(reloaded) != persisted_digest:
            raise RunnerIntegrityError("persistent state failed deterministic round-trip")
        self.state.target_transient_ids.clear()
        if self.state.transient_payload()["target_transient_ids"]:
            raise RunnerIntegrityError("target transient state survived reset")
        self._record("post_reset_execution", persisted_digest=persisted_digest, transient_cleared=True)
        return {"persisted_digest": persisted_digest, "transient_cleared": True}

    def result(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "source_gate": self.state.source_certificate,
            "state": self.state.persistent_payload(),
            "target_schedule_digest": digest(self._target_schedule),
            "event_digest": digest(stable(self.state.event_log)),
            "events": self.state.event_log,
        }


def assert_condition_routing(result: Mapping[str, Any]) -> None:
    condition = result["condition"]
    state = result["state"]
    if condition == "C0":
        if state["source_skill"] or state["source_memory"] or state["source_teacher_digest"]:
            raise RunnerIntegrityError("C0 retained source state after reset")
    elif condition == "C1":
        if not state["source_skill"] or state["source_memory"] or state["source_teacher_digest"]:
            raise RunnerIntegrityError("C1 state routing mismatch")
    elif condition == "C2":
        if state["source_skill"] or not state["source_memory"] or state["source_teacher_digest"]:
            raise RunnerIntegrityError("C2 state routing mismatch")
    elif condition == "C3":
        if not state["source_skill"] or state["source_memory"] or not state["source_teacher_digest"]:
            raise RunnerIntegrityError("C3 state routing mismatch")


def assert_equal_target_schedule(results: Sequence[Mapping[str, Any]]) -> None:
    schedules = {result["target_schedule_digest"] for result in results}
    if len(schedules) != 1:
        raise BudgetMismatch("conditions did not receive identical target exposure order")
    budgets = {
        (
            result["state"]["target_updates"],
            result["state"]["candidate_calls"],
            result["state"]["verifier_calls"],
        )
        for result in results
    }
    if len(budgets) != 1:
        raise BudgetMismatch("conditions did not receive matched target budgets")


def run_condition(manifest: ManifestView, condition: str, seed: int) -> Dict[str, Any]:
    backend = FixtureBackend(condition, seed, manifest)
    backend.acquire_source()
    backend.source_gate()
    backend.configure_condition()
    backend.target_training()
    backend.validation_schedule()
    backend.persist_and_reset()
    result = backend.result()
    assert_condition_routing(result)
    return result


def run_negative_assertion_fixtures(manifest: ManifestView, seed: int) -> Dict[str, str]:
    outcomes: Dict[str, str] = {}

    try:
        state = FixtureBackend("C0", seed, manifest)
        state.acquire_source()
        state.source_gate()
        state.state.source_skill = True
        assert_condition_routing(state.result())
    except RunnerIntegrityError:
        outcomes["c0_reset_violation"] = "caught"
    else:
        raise RunnerIntegrityError("negative fixture failed to catch C0 reset violation")

    try:
        assert_equal_target_schedule(
            [
                {"target_schedule_digest": "a", "state": {"target_updates": 1, "candidate_calls": 1, "verifier_calls": 1}},
                {"target_schedule_digest": "b", "state": {"target_updates": 1, "candidate_calls": 1, "verifier_calls": 1}},
            ]
        )
    except BudgetMismatch:
        outcomes["target_schedule_mismatch"] = "caught"
    else:
        raise RunnerIntegrityError("negative fixture failed to catch target schedule mismatch")

    try:
        leaked = dict(manifest.rows[f"seed_{seed}/target_dependent_sealed"][0])
        leaked["query_output"] = 1
        if leaked["split"].endswith("sealed") and "query_output" in leaked:
            raise LeakageViolation("synthetic sealed answer leakage")
    except LeakageViolation:
        outcomes["sealed_answer_leak"] = "caught"
    else:
        raise RunnerIntegrityError("negative fixture failed to catch sealed-answer leakage")

    try:
        c3 = FixtureBackend("C3", seed, manifest)
        c3.acquire_source()
        c3.source_gate()
        c3.configure_condition()
        c3.state.source_teacher_digest = None
        c3.target_training()
    except RunnerIntegrityError:
        outcomes["c3_teacher_mutation"] = "caught"
    else:
        raise RunnerIntegrityError("negative fixture failed to catch C3 teacher mutation")
    return outcomes


def run_fixtures(manifest_dir: Path, output_dir: Path) -> Dict[str, Any]:
    manifest = ManifestView.load(manifest_dir)
    try:
        manifest_report_path = str(manifest.root.relative_to(ROOT))
    except ValueError:
        manifest_report_path = "<external-manifest-root>"
    all_results: List[Dict[str, Any]] = []
    replay_results: List[Dict[str, Any]] = []
    for seed in (42, 1337):
        first = [run_condition(manifest, condition, seed) for condition in CONDITIONS]
        second = [run_condition(manifest, condition, seed) for condition in CONDITIONS]
        assert_equal_target_schedule(first)
        assert_equal_target_schedule(second)
        for left, right in zip(first, second):
            if left["event_digest"] != right["event_digest"]:
                raise RunnerIntegrityError(f"deterministic replay mismatch for {left['condition']} seed {seed}")
        all_results.extend(first)
        replay_results.extend(second)

    negative = run_negative_assertion_fixtures(manifest, 42)
    report = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "classification": "cpu_fixture_runner_validation_only",
        "scientific_authorization": False,
        "scientific_execution": False,
        "sealed_answers_opened": False,
        "sealed_answer_access_count": 0,
        "manifest": {
            "path": manifest_report_path,
            "digest": manifest.public_digest,
            "independent_validation_report": read_json(manifest.root / "INDEPENDENT_VALIDATION_REPORT.json").get("status"),
        },
        "frozen_schedule": {
            "source_updates": SOURCE_UPDATES,
            "target_updates": TARGET_UPDATES,
            "target_checkpoints": list(TARGET_CHECKPOINTS),
            "candidate_budget": FIXTURE_CANDIDATE_BUDGET,
            "verifier_budget": FIXTURE_VERIFIER_BUDGET,
        },
        "condition_assertions": {
            "C0_source_reset_restores_theta_optimizer_rng": True,
            "C1_source_weights_retained_without_memory_or_teacher": True,
            "C2_source_weights_discarded_and_source_memory_only": True,
            "C3_source_weights_retained_with_frozen_teacher_and_no_raw_memory": True,
            "equal_target_exposure_order": True,
            "equal_target_budgets": True,
            "persistence_round_trip": True,
            "post_reset_transient_state_cleared": True,
            "sealed_answer_isolation": True,
        },
        "negative_fixture_assertions": negative,
        "replay": {
            "runs": len(replay_results),
            "deterministic": True,
            "compared_event_digests": len(all_results),
        },
        "results": all_results,
        "limitations": [
            "The fixture backend is not a neural learner and its source gate uses an explicitly marked fixture oracle.",
            "No source or target scientific accuracy is reported by this validation run.",
            "This report validates runner boundaries and matched accounting only; it does not authorize the experiment.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "RUNNER_VALIDATION_REPORT.json", report)
    (output_dir / "RUNNER_VALIDATION_REPORT.md").write_text(
        "# Productive-retention runner validation\n\n"
        "Status: PASS — CPU fixture validation only; scientific execution is not authorized.\n\n"
        + json.dumps(report, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixtures", "scientific"), default="fixtures")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT / "runner_validation",
    )
    args = parser.parse_args()
    if args.mode == "scientific":
        raise ScientificExecutionBlocked(
            "scientific mode is blocked: frozen protocol and manifest carry scientific_authorization=false"
        )
    started = time.perf_counter()
    report = run_fixtures(args.manifest_dir, args.output_dir)
    report["wall_seconds"] = time.perf_counter() - started
    write_json(args.output_dir / "RUNNER_VALIDATION_REPORT.json", report)
    print(json.dumps({"status": report["status"], "scientific_authorization": False, "sealed_answers_opened": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
