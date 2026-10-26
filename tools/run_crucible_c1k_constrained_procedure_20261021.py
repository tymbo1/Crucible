#!/usr/bin/env python3
"""Run the CPU-only C1k constrained-procedure replay.

C1k reuses the frozen C1j test tasks and model weights.  It does not train a
new policy.  The new arm lets the learned policy request reuse, but an
independent evidence contract controls whether a deterministic procedure may
actually execute.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_crucible_c1j_cpu_stress_20261021 as c1j


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_c1k_constrained_procedure_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
C1J_OUTPUT = ROOT / "experiments" / "crucible_c1j_cpu_stress_20261021"
EXPERIMENT = "crucible_c1k_constrained_procedure_20261021"
POLICIES = (
    "deterministic",
    "learned_unconstrained",
    "hybrid_unconstrained",
    "learned_constrained",
    "persistent",
    "stateless",
    "oracle",
)
FEATURE_COST = c1j.FEATURE_COST
VERIFY_COST = c1j.VERIFY_COST
DETERMINISTIC_GATE_COST = c1j.DETERMINISTIC_GATE_COST


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def procedure_fingerprint(family: str, rule: str) -> str:
    return hashlib.sha256(f"procedure-v1|{family}|{rule}".encode()).hexdigest()


def verify_c1j_artifacts() -> Dict[str, Any]:
    index_path = C1J_OUTPUT / "HASHES.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in sorted(index.items()):
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    if failures:
        raise RuntimeError(f"C1j artifact hash failure: {failures}")
    return {"indexed_files": len(index), "hash_failures": 0, "hash_index_sha256": sha256_file(index_path)}


def load_c1j_models() -> Dict[str, c1j.LogisticApplicability]:
    raw = json.loads((C1J_OUTPUT / "MODELS.json").read_text(encoding="utf-8"))
    models: Dict[str, c1j.LogisticApplicability] = {}
    for seed, data in raw.items():
        models[seed] = c1j.LogisticApplicability(
            weights=[float(value) for value in data["weights"]],
            bias=float(data["bias"]),
            updates=int(data["updates"]),
            curve=[],
        )
    return models


def load_frozen_tasks() -> Tuple[List[c1j.Candidate], Dict[str, Any]]:
    manifest = json.loads((C1J_OUTPUT / "SPLIT_MANIFEST.json").read_text(encoding="utf-8"))
    tasks = c1j.generate_candidates("test", c1j.TEST_TASKS, 303, c1j.HELDOUT_FAMILIES)
    digest = hashlib.sha256("|".join(candidate.task.task_id for candidate in tasks).encode()).hexdigest()
    if digest != manifest["test"]["task_digest"]:
        raise RuntimeError("Regenerated C1j test digest does not match the frozen manifest")
    return tasks, {
        "count": len(tasks),
        "task_digest": digest,
        "manifest_digest": manifest["test"]["task_digest"],
        "families": manifest["test"]["families"],
        "memory_sizes": manifest["memory_sizes"],
        "conditions": manifest["conditions"],
    }


def load_frozen_failure_events() -> Dict[str, Any]:
    path = C1J_OUTPUT / "RUNS.jsonl"
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        run = json.loads(line)
        if run["policy"] != "learned":
            continue
        for outcome in run["outcomes"]:
            if outcome["harmful_transfer"]:
                events.append({
                    "seed": run["seed"],
                    "task_id": outcome["task_id"],
                    "family": outcome["family"],
                    "condition": outcome["condition"],
                    "memory_size": outcome["memory_size"],
                    "decision": outcome["decision"],
                    "probability": outcome["probability"],
                })

    def counts(key: str) -> Dict[str, int]:
        return dict(sorted(Counter(str(item[key]) for item in events).items()))

    return {
        "event_count": len(events),
        "by_condition": counts("condition"),
        "by_family": counts("family"),
        "by_memory_size": counts("memory_size"),
        "by_decision": counts("decision"),
        "events": events,
    }


@dataclass
class ProcedureLedger:
    contract_passes: int = 0
    contract_rejections: int = 0
    procedure_executions: int = 0
    procedure_failures: int = 0
    verified_adaptations: int = 0
    validated_promotions: int = 0
    provisional_events: int = 0
    contradiction_events: int = 0
    fingerprints: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.fingerprints is None:
            self.fingerprints = []

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract_passes": self.contract_passes,
            "contract_rejections": self.contract_rejections,
            "procedure_executions": self.procedure_executions,
            "procedure_failures": self.procedure_failures,
            "verified_adaptations": self.verified_adaptations,
            "validated_promotions": self.validated_promotions,
            "provisional_events": self.provisional_events,
            "contradiction_events": self.contradiction_events,
            "unique_procedure_fingerprints": sorted(set(self.fingerprints)),
        }


def mandatory_contract(task: c1j.Task, record: c1j.MemoryRecord) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    if task.family != record.family:
        failures.append("family_mismatch")
    if task.structural_signature != record.structural_signature:
        failures.append("structural_mismatch")
    if task.environment_version != record.environment_version:
        failures.append("environment_mismatch")
    if not task.probe_available:
        failures.append("probe_unavailable")
    elif not c1j.observed_probe_agrees(task, record):
        failures.append("probe_disagreement")
    if record.evidence_quality < 0.8:
        failures.append("insufficient_evidence_quality")
    if not record.provenance_valid:
        failures.append("invalid_provenance")
    if record.contradiction_flag:
        failures.append("contradictory_record")
    return not failures, failures


def constrained_decision(
    task: c1j.Task,
    record: c1j.MemoryRecord,
    model: c1j.LogisticApplicability,
    ledger: ProcedureLedger,
) -> Tuple[Optional[str], float, str, str, Optional[float], bool, List[str], Optional[str]]:
    probability = model.probability(c1j.feature_vector(task, record))
    contract_ok, failures = mandatory_contract(task, record)
    if probability < 0.5:
        selected, cost, decision = c1j.investigate(task)
        return selected, FEATURE_COST + cost, decision, "none", probability, False, ["learner_declined_reuse"], None
    if not contract_ok:
        ledger.contract_rejections += 1
        if "contradictory_record" in failures:
            ledger.contradiction_events += 1
        if "invalid_provenance" in failures or "insufficient_evidence_quality" in failures:
            ledger.provisional_events += 1
        selected, cost, decision = c1j.investigate(task)
        return selected, FEATURE_COST + cost, decision, "contract_fallback", probability, False, failures, None

    ledger.contract_passes += 1
    fingerprint = procedure_fingerprint(task.family, record.rule)
    ledger.fingerprints.append(fingerprint)
    if task.variant != record.variant:
        ledger.verified_adaptations += 1
        verified = c1j.safe_solve(record.rule, task.family, task.target_input) == task.target_output
        if not verified:
            ledger.procedure_failures += 1
            ledger.provisional_events += 1
            selected, cost, decision = c1j.investigate(task)
            return selected, FEATURE_COST + VERIFY_COST + cost, decision, "verification_fallback", probability, True, ["adaptation_verification_failed"], fingerprint
        ledger.procedure_executions += 1
        ledger.validated_promotions += 1
        decision = "adapt"
        return record.rule, FEATURE_COST + VERIFY_COST, decision, "validated_procedure", probability, True, [], fingerprint

    ledger.procedure_executions += 1
    selected = record.rule
    return selected, FEATURE_COST, "reuse", "validated_procedure", probability, True, [], fingerprint


def policy_arm(policy: str) -> str:
    return {
        "learned_unconstrained": "learned",
        "hybrid_unconstrained": "hybrid",
    }.get(policy, policy)


def evaluate_policy(
    tasks: Sequence[c1j.Candidate],
    policy: str,
    model: Optional[c1j.LogisticApplicability],
    seed: int,
) -> Dict[str, Any]:
    ledger = ProcedureLedger()
    outcomes: List[Dict[str, Any]] = []
    for candidate in tasks:
        task = candidate.task
        record = candidate.record
        contract_pass = None
        contract_failures: List[str] = []
        fingerprint = None
        if policy == "learned_constrained":
            if model is None:
                raise ValueError("constrained policy requires a model")
            selected, cost, decision, source, probability, contract_pass, contract_failures, fingerprint = constrained_decision(task, record, model, ledger)
        else:
            selected, cost, decision, source, probability = c1j.policy_decision(task, record, policy_arm(policy), model)
            if policy == "deterministic":
                contract_pass, contract_failures = mandatory_contract(task, record)
            elif policy in {"learned_unconstrained", "hybrid_unconstrained"}:
                contract_pass, contract_failures = mandatory_contract(task, record)
            if decision in {"reuse", "adapt"} and selected is not None:
                fingerprint = procedure_fingerprint(task.family, selected)
        correct = c1j.safe_solve(selected, task.family, task.target_input) == task.target_output if selected is not None else False
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
            "contract_pass": contract_pass,
            "contract_failures": contract_failures,
            "procedure_fingerprint": fingerprint,
            "procedure_executed": source == "validated_procedure" and decision in {"reuse", "adapt"},
        })
    metrics = c1j.group_metrics(outcomes)
    metrics.update({
        "policy": policy,
        "seed": seed,
        "ledger": ledger.as_dict(),
        "contract_pass_rate": sum(item["contract_pass"] is True for item in outcomes) / max(1, sum(item["contract_pass"] is not None for item in outcomes)),
        "outcomes": outcomes,
    })
    return metrics


def aggregate_runs(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    base = c1j.aggregate_summaries(records)
    if not records:
        return base
    ledger_fields = (
        "contract_passes", "contract_rejections", "procedure_executions",
        "procedure_failures", "verified_adaptations", "validated_promotions",
        "provisional_events", "contradiction_events",
    )
    for field in ledger_fields:
        base[field] = sum(int(item.get("ledger", {}).get(field, 0)) for item in records)
    base["contract_pass_rate"] = sum(float(item.get("contract_pass_rate", 0.0)) for item in records) / len(records)
    base["unique_procedure_fingerprints"] = len({fingerprint for item in records for fingerprint in item.get("ledger", {}).get("unique_procedure_fingerprints", [])})
    return base


def aggregate_group(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    base = c1j.aggregate_summaries(records)
    if not records:
        return base
    ledger_fields = (
        "contract_passes", "contract_rejections", "procedure_executions",
        "procedure_failures", "verified_adaptations", "validated_promotions",
        "provisional_events", "contradiction_events",
    )
    for field in ledger_fields:
        base[field] = sum(int(item.get(field, 0)) for item in records)
    base["contract_pass_rate"] = sum(float(item.get("contract_pass_rate", 0.0)) for item in records) / len(records)
    base["unique_procedure_fingerprints"] = sum(int(item.get("unique_procedure_fingerprints", 0)) for item in records)
    return base


def subgroup_metrics(outcomes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = c1j.group_metrics(outcomes)
    contract_observations = [item for item in outcomes if item.get("contract_pass") is not None]
    metrics.update({
        "contract_pass_rate": sum(item.get("contract_pass") is True for item in contract_observations) / max(1, len(contract_observations)),
        "contract_passes": sum(item.get("contract_pass") is True for item in contract_observations),
        "contract_rejections": sum(item.get("source") == "contract_fallback" for item in outcomes),
        "procedure_executions": sum(bool(item.get("procedure_executed")) for item in outcomes),
        "procedure_failures": sum("adaptation_verification_failed" in item.get("contract_failures", []) for item in outcomes),
        "verified_adaptations": sum(item.get("source") == "validated_procedure" and item.get("decision") == "adapt" for item in outcomes),
        "validated_promotions": sum(item.get("source") == "validated_procedure" and item.get("decision") == "adapt" for item in outcomes),
        "provisional_events": sum(
            item.get("source") == "verification_fallback"
            or any(reason in {"invalid_provenance", "insufficient_evidence_quality"} for reason in item.get("contract_failures", []))
            for item in outcomes
        ),
        "contradiction_events": sum("contradictory_record" in item.get("contract_failures", []) for item in outcomes),
        "unique_procedure_fingerprints": len({item["procedure_fingerprint"] for item in outcomes if item.get("procedure_fingerprint")}),
    })
    return metrics


def source_hashes() -> Dict[str, str]:
    paths = [
        PROTOCOL,
        ROOT / "tools" / "run_crucible_c1j_cpu_stress_20261021.py",
        ROOT / "tools" / "run_crucible_c1k_constrained_procedure_20261021.py",
        C1J_OUTPUT / "RESULTS.json",
        C1J_OUTPUT / "MODELS.json",
        C1J_OUTPUT / "SPLIT_MANIFEST.json",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def render_summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1k constrained learned procedure execution",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only replay; no new learner training, GPU, cloud or Crucible neural training.",
        "",
        "## Aggregate paired results",
        "",
        "| Policy | Accuracy | Resolvable accuracy | Mean cost | Harmful transfer | Contract pass rate | Procedure failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, item in result["aggregate"].items():
        lines.append(
            f"| `{policy}` | {item['accuracy']:.4f} | {item['resolvable_accuracy']:.4f} | "
            f"{item['mean_cost']:.3f} | {item['harmful_transfer_rate']:.4f} | "
            f"{item.get('contract_pass_rate', 0.0):.4f} | {item.get('procedure_failures', 0)} |"
        )
    lines.extend(["", "## C1j failure attribution", "", f"- Frozen learned harmful-transfer events: `{result['failure_attribution']['frozen_event_count']}`.", f"- Replayed learned harmful-transfer events: `{result['failure_attribution']['replayed_event_count']}`.", f"- Exact event replay: `{result['failure_attribution']['exact_replay']}`.", ""])
    for group_name in ("by_condition", "by_family", "by_memory_size", "by_decision"):
        lines.append(f"### Failures by {group_name.replace('by_', '')}")
        for key, value in result["failure_attribution"]["frozen"].get(group_name, {}).items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(["## Constrained-policy results by memory size", "", "| Memory size | Accuracy | Mean cost | Harmful transfer | Contract rejections |", "| ---: | ---: | ---: | ---: | ---: |"])
    for memory_size, item in result["by_memory_size"]["learned_constrained"].items():
        lines.append(f"| `{memory_size}` | {item['accuracy']:.4f} | {item['mean_cost']:.3f} | {item['harmful_transfer_rate']:.4f} | {item.get('contract_rejections', 0)} |")
    lines.extend([
        "",
        "## Gates",
        "",
        f"- C1j failure-replay gate: `{result['gates']['failure_replay_gate']}`.",
        f"- Constrained accuracy gate: `{result['gates']['constrained_accuracy_gate']}`.",
        f"- Constrained zero-harm gate: `{result['gates']['constrained_zero_harm_gate']}`.",
        f"- Constrained efficiency gate: `{result['gates']['constrained_efficiency_gate']}`.",
        f"- Contract enforcement gate: `{result['gates']['contract_enforcement_gate']}`.",
        "",
        "A failed gate remains a valid result. Passing synthetic gates does not establish neural reasoning or CMS2 training value.",
        "",
        "## Source hashes",
        "",
    ])
    for path, digest in sorted(hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    c1j_audit = verify_c1j_artifacts()
    tasks, input_audit = load_frozen_tasks()
    models = load_c1j_models()
    frozen_failures = load_frozen_failure_events()
    all_runs: List[Dict[str, Any]] = []
    by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_condition: Dict[str, Dict[str, List[Dict[str, Any]]]] = {condition: defaultdict(list) for condition in c1j.CONDITIONS}
    by_family: Dict[str, Dict[str, List[Dict[str, Any]]]] = {family: defaultdict(list) for family in c1j.HELDOUT_FAMILIES}
    by_memory: Dict[str, Dict[str, List[Dict[str, Any]]]] = {str(size): defaultdict(list) for size in c1j.MEMORY_SIZES}

    for seed in c1j.LEARNER_SEEDS:
        model = models[str(seed)]
        if model.updates != c1j.TRAIN_UPDATES:
            raise RuntimeError(f"C1j model for seed {seed} has unexpected update count {model.updates}")
        for policy in POLICIES:
            run = evaluate_policy(tasks, policy, model if policy in {"learned_unconstrained", "hybrid_unconstrained", "learned_constrained"} else None, seed)
            summary = {key: value for key, value in run.items() if key != "outcomes"}
            all_runs.append({"seed": seed, "policy": policy, "summary": summary, "outcomes": run["outcomes"]})
            by_policy[policy].append(summary)
            for condition in c1j.CONDITIONS:
                by_condition[condition][policy].append(subgroup_metrics([item for item in run["outcomes"] if item["condition"] == condition]))
            for family in c1j.HELDOUT_FAMILIES:
                by_family[family][policy].append(subgroup_metrics([item for item in run["outcomes"] if item["family"] == family]))
            for memory_size in c1j.MEMORY_SIZES:
                by_memory[str(memory_size)][policy].append(subgroup_metrics([item for item in run["outcomes"] if item["memory_size"] == memory_size]))

    aggregate = {policy: aggregate_runs(records) for policy, records in sorted(by_policy.items())}
    condition_results = {condition: {policy: aggregate_group(records) for policy, records in sorted(policies.items())} for condition, policies in by_condition.items()}
    family_results = {family: {policy: aggregate_group(records) for policy, records in sorted(policies.items())} for family, policies in by_family.items()}
    memory_results = {policy: {memory_size: aggregate_group(by_memory[memory_size][policy]) for memory_size in sorted(by_memory)} for policy in POLICIES}

    replay_events = []
    for run in all_runs:
        if run["policy"] != "learned_unconstrained":
            continue
        for outcome in run["outcomes"]:
            if outcome["harmful_transfer"]:
                replay_events.append({
                    "seed": run["seed"],
                    "task_id": outcome["task_id"],
                    "family": outcome["family"],
                    "condition": outcome["condition"],
                    "memory_size": outcome["memory_size"],
                    "decision": outcome["decision"],
                    "probability": outcome["probability"],
                })
    frozen_signature = sorted((item["seed"], item["task_id"], item["family"], item["condition"], item["memory_size"], item["decision"]) for item in frozen_failures["events"])
    replay_signature = sorted((item["seed"], item["task_id"], item["family"], item["condition"], item["memory_size"], item["decision"]) for item in replay_events)
    failure_attribution = {
        "frozen_event_count": len(frozen_failures["events"]),
        "replayed_event_count": len(replay_events),
        "exact_replay": frozen_signature == replay_signature,
        "frozen": {
            "by_condition": frozen_failures["by_condition"],
            "by_family": frozen_failures["by_family"],
            "by_memory_size": frozen_failures["by_memory_size"],
            "by_decision": frozen_failures["by_decision"],
        },
    }
    deterministic = aggregate["deterministic"]
    constrained = aggregate["learned_constrained"]
    constrained_outcomes = [run["outcomes"] for run in all_runs if run["policy"] == "learned_constrained"]
    constrained_flat = [item for group in constrained_outcomes for item in group]
    contract_enforcement = all(
        not (item["decision"] in {"reuse", "adapt"} and item["contract_pass"] is not True)
        and not (item["decision"] == "adapt" and item["source"] == "validated_procedure" and "adaptation_verification_failed" in item["contract_failures"])
        for item in constrained_flat
    )
    result = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "aggregate": aggregate,
        "by_condition": condition_results,
        "by_family": family_results,
        "by_memory_size": memory_results,
        "failure_attribution": failure_attribution,
        "gates": {
            "failure_replay_gate": failure_attribution["exact_replay"],
            "constrained_accuracy_gate": constrained["accuracy"] >= deterministic["accuracy"] - 0.01,
            "constrained_zero_harm_gate": constrained["harmful_transfer_rate"] == 0.0,
            "constrained_efficiency_gate": constrained["mean_cost"] <= deterministic["mean_cost"] * 0.90,
            "contract_enforcement_gate": contract_enforcement,
        },
        "config": {
            "source_experiment": c1j.EXPERIMENT,
            "test_tasks": len(tasks),
            "learner_seeds": list(c1j.LEARNER_SEEDS),
            "policies": list(POLICIES),
            "threshold": 0.5,
            "new_training_updates": 0,
            "execution": "CPU-only",
        },
    }
    write_json(OUTPUT / "C1J_INPUT_AUDIT.json", {"artifact_audit": c1j_audit, "input_audit": input_audit, "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in [C1J_OUTPUT / "RESULTS.json", C1J_OUTPUT / "MODELS.json", C1J_OUTPUT / "SPLIT_MANIFEST.json"]}})
    write_json(OUTPUT / "FAILURE_ATTRIBUTION.json", failure_attribution)
    write_json(OUTPUT / "RESULTS.json", result)
    write_json(OUTPUT / "RUN_CONFIG.json", result["config"])
    write_json(OUTPUT / "MODEL_REFERENCES.json", {str(seed): {"source": str((C1J_OUTPUT / "MODELS.json").relative_to(ROOT)), "updates": models[str(seed)].updates, "threshold": 0.5} for seed in c1j.LEARNER_SEEDS})
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
