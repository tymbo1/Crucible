#!/usr/bin/env python3
"""Run C1l: verification amortization and certificate invalidation.

This is a CPU-only replay of the frozen C1j test corpus.  It issues
condition-bound certificates from an independent seed split, then compares
cheap certificate execution with the C1k deterministic contract.  No model is
trained and no neural checkpoint is modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_crucible_c1j_cpu_stress_20261021 as c1j
from tools import run_crucible_c1k_constrained_procedure_20261021 as c1k


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_c1l_certificate_amortization_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
C1J_OUTPUT = ROOT / "experiments" / "crucible_c1j_cpu_stress_20261021"
C1K_OUTPUT = ROOT / "experiments" / "crucible_c1k_constrained_procedure_20261021"
C1K_REPEAT_OUTPUT = ROOT / "experiments" / "crucible_c1k_repeat_constrained_procedure_20261021"
EXPERIMENT = "crucible_c1l_certificate_amortization_20261021"

CERTIFICATE_SEED = 404
CERTIFICATE_SEED_CANDIDATES = 50_000
CERTIFICATE_LOOKUP_COST = 0.02
CERTIFICATE_VALIDITY_COST = 0.04
CERTIFICATE_USE_COST = CERTIFICATE_LOOKUP_COST + CERTIFICATE_VALIDITY_COST
CERTIFICATE_ISSUANCE_COST = 1.98
CERTIFICATE_FORMAT = "c1l-cert-v1"

POLICIES = (
    "deterministic_reference",
    "certificate_amortized",
    "certificate_corrupted",
    "stale_certificate_control",
    "learned_constrained_reference",
    "stateless_reference",
)
CERTIFICATE_POLICIES = {
    "certificate_amortized",
    "certificate_corrupted",
    "stale_certificate_control",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest_value(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def task_digest(tasks: Sequence[c1j.Candidate]) -> str:
    return hashlib.sha256("|".join(candidate.task.task_id for candidate in tasks).encode()).hexdigest()


def certificate_key(family: str, rule: str, structural: str, environment: int) -> str:
    return "|".join((CERTIFICATE_FORMAT, family, rule, structural, str(environment)))


def input_schema(family: str, structural: str) -> str:
    return f"{family}-input-v1-{structural}"


def procedure_fingerprint(family: str, rule: str) -> str:
    return c1k.procedure_fingerprint(family, rule)


def evidence_digest(task: c1j.Task, record: c1j.MemoryRecord) -> str:
    """Digest only decision-time evidence and applicability conditions."""

    return digest_value(
        {
            "format": CERTIFICATE_FORMAT,
            "task_family": task.family,
            "record_family": record.family,
            "record_rule": record.rule,
            "task_structural": task.structural_signature,
            "record_structural": record.structural_signature,
            "task_environment": task.environment_version,
            "record_environment": record.environment_version,
            "input_schema": input_schema(task.family, task.structural_signature),
            "probe_available": bool(task.probe_available),
            "probe_agrees": bool(c1j.observed_probe_agrees(task, record)),
            "evidence_quality": round(float(record.evidence_quality), 6),
            "provenance_valid": bool(record.provenance_valid),
            "contradiction_flag": bool(record.contradiction_flag),
        }
    )


def verify_index(index_path: Path) -> Dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    failures: List[Dict[str, str]] = []
    for relative, expected in sorted(index.items()):
        path = ROOT / relative
        if not path.exists():
            failures.append({"path": relative, "expected": expected, "actual": "missing"})
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    if failures:
        raise RuntimeError(f"Frozen artifact hash failure in {index_path}: {failures}")
    return {
        "index": str(index_path.relative_to(ROOT)),
        "indexed_files": len(index),
        "hash_failures": 0,
        "index_sha256": sha256_file(index_path),
    }


def build_certificate_catalogue() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[c1j.Candidate]]:
    """Issue certificates from an independent, non-test validation split."""

    seed_candidates = c1j.generate_candidates(
        "certificate_seed",
        CERTIFICATE_SEED_CANDIDATES,
        CERTIFICATE_SEED,
        c1j.HELDOUT_FAMILIES,
    )
    catalogue: Dict[str, Dict[str, Any]] = {}
    for candidate in seed_candidates:
        task = candidate.task
        record = candidate.record
        if task.condition not in {"applicable", "adaptation"}:
            continue
        if not c1j.hard_evidence(task, record):
            continue
        if not c1j.safe_correct(task, record.rule):
            continue
        key = certificate_key(task.family, record.rule, task.structural_signature, task.environment_version)
        if key in catalogue:
            continue
        certificate_id = digest_value({"format": CERTIFICATE_FORMAT, "key": key})
        catalogue[key] = {
            "certificate_id": certificate_id,
            "format": CERTIFICATE_FORMAT,
            "key": key,
            "family": task.family,
            "rule": record.rule,
            "procedure_fingerprint": procedure_fingerprint(task.family, record.rule),
            "structural_signature": task.structural_signature,
            "environment_version": task.environment_version,
            "input_schema": input_schema(task.family, task.structural_signature),
            "condition_digest": evidence_digest(task, record),
            "source_task_id": task.task_id,
            "source_task_digest": digest_value(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "rule": record.rule,
                    "structural": task.structural_signature,
                    "environment": task.environment_version,
                }
            ),
            "issuance_cost": CERTIFICATE_ISSUANCE_COST,
        }

    required = {
        certificate_key(family, rule, structural, 0)
        for family in c1j.HELDOUT_FAMILIES
        for rule in c1j.RULES[family]
        for structural in ("core", "transfer")
    }
    missing = sorted(required - set(catalogue))
    audit = {
        "seed": CERTIFICATE_SEED,
        "candidate_count": len(seed_candidates),
        "candidate_task_digest": task_digest(seed_candidates),
        "required_entries": len(required),
        "catalogue_entries": len(catalogue),
        "missing_entries": missing,
        "complete": not missing,
        "test_labels_used_for_issuance": False,
    }
    if missing:
        raise RuntimeError(f"Certificate catalogue incomplete: {missing}")
    return catalogue, audit, seed_candidates


def current_certificate_lookup(
    task: c1j.Task,
    record: c1j.MemoryRecord,
    catalogue: Mapping[str, Mapping[str, Any]],
    corrupted_ids: Sequence[str],
) -> Tuple[Optional[Mapping[str, Any]], str, str]:
    key = certificate_key(task.family, record.rule, task.structural_signature, task.environment_version)
    certificate = catalogue.get(key)
    if certificate is None:
        return None, "no_matching_certificate", key
    if certificate["certificate_id"] in set(corrupted_ids):
        return None, "corrupted_certificate", key
    if task.family != record.family:
        return None, "record_family_mismatch", key
    if task.structural_signature != record.structural_signature:
        return None, "record_structure_mismatch", key
    if task.environment_version != record.environment_version:
        return None, "record_environment_mismatch", key
    if not task.probe_available:
        return None, "probe_unavailable", key
    if not c1j.observed_probe_agrees(task, record):
        return None, "probe_disagreement", key
    if record.evidence_quality < 0.8:
        return None, "insufficient_evidence_quality", key
    if not record.provenance_valid:
        return None, "invalid_provenance", key
    if record.contradiction_flag:
        return None, "contradictory_record", key
    if certificate["input_schema"] != input_schema(task.family, task.structural_signature):
        return None, "input_schema_mismatch", key
    if certificate["condition_digest"] != evidence_digest(task, record):
        return None, "evidence_digest_mismatch", key
    return certificate, "valid", key


def stale_certificate_lookup(
    task: c1j.Task,
    record: c1j.MemoryRecord,
    catalogue: Mapping[str, Mapping[str, Any]],
) -> Tuple[Optional[Mapping[str, Any]], str, str]:
    """Unsafe control: use the cached record conditions, not current ones."""

    key = certificate_key(record.family, record.rule, record.structural_signature, record.environment_version)
    certificate = catalogue.get(key)
    if certificate is None:
        return None, "no_cached_certificate", key
    return certificate, "stale_used", key


def certificate_decision(task: c1j.Task, record: c1j.MemoryRecord) -> str:
    if task.variant == record.variant and task.structural_signature == record.structural_signature:
        return "reuse"
    return "adapt"


def deterministic_fallback(task: c1j.Task, record: c1j.MemoryRecord) -> Tuple[Optional[str], float, str]:
    selected, cost, decision, _source, _probability = c1j.policy_decision(task, record, "deterministic", None)
    return selected, cost, decision


def evaluate_certificate_policy(
    tasks: Sequence[c1j.Candidate],
    policy: str,
    seed: int,
    catalogue: Mapping[str, Mapping[str, Any]],
    corrupted_ids: Sequence[str],
) -> Dict[str, Any]:
    outcomes: List[Dict[str, Any]] = []
    invalidation_reasons: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    for candidate in tasks:
        task = candidate.task
        record = candidate.record
        certificate = None
        certificate_status = "not_applicable"
        certificate_key_used = None
        selected: Optional[str]
        decision: str
        source: str
        if policy == "stale_certificate_control":
            certificate, certificate_status, certificate_key_used = stale_certificate_lookup(task, record, catalogue)
            if certificate is not None:
                counters["certificate_lookups"] += 1
                counters["stale_certificate_uses"] += 1
                selected = record.rule
                decision = certificate_decision(task, record)
                source = "stale_certificate"
                cost = CERTIFICATE_LOOKUP_COST
            else:
                counters["certificate_lookups"] += 1
                selected, fallback_cost, decision = deterministic_fallback(task, record)
                source = "stale_certificate_fallback"
                cost = CERTIFICATE_LOOKUP_COST + fallback_cost
                counters["fallbacks"] += 1
        else:
            certificate, certificate_status, certificate_key_used = current_certificate_lookup(task, record, catalogue, corrupted_ids if policy == "certificate_corrupted" else ())
            counters["certificate_lookups"] += 1
            counters["validity_checks"] += 1
            if certificate is not None:
                counters["valid_certificate_uses"] += 1
                selected = record.rule
                decision = certificate_decision(task, record)
                source = "certificate"
                cost = CERTIFICATE_USE_COST
            else:
                counters["invalidations"] += 1
                invalidation_reasons[certificate_status] += 1
                selected, fallback_cost, decision = deterministic_fallback(task, record)
                source = "certificate_fallback"
                cost = CERTIFICATE_USE_COST + fallback_cost
                counters["fallbacks"] += 1
                if certificate_status == "corrupted_certificate":
                    counters["corrupted_lookups"] += 1
                    counters["corrupted_rejections"] += 1

        correct = c1j.safe_correct(task, selected)
        harmful = decision in {"reuse", "adapt"} and not correct
        outcomes.append(
            {
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
                "harmful_transfer": harmful,
                "cost": cost,
                "probability": None,
                "label": candidate.label,
                "certificate_status": certificate_status,
                "certificate_key": certificate_key_used,
                "certificate_id": certificate["certificate_id"] if certificate is not None else None,
                "certificate_valid": certificate is not None and certificate_status == "valid",
                "certificate_fallback": source.endswith("fallback"),
                "procedure_executed": source in {"certificate", "stale_certificate"} and decision in {"reuse", "adapt"},
            }
        )

    summary = c1j.group_metrics(outcomes)
    summary.update(
        {
            "policy": policy,
            "seed": seed,
            "outcomes": outcomes,
            "certificate_lookups": counters["certificate_lookups"],
            "validity_checks": counters["validity_checks"],
            "valid_certificate_uses": counters["valid_certificate_uses"],
            "stale_certificate_uses": counters["stale_certificate_uses"],
            "invalidations": counters["invalidations"],
            "fallbacks": counters["fallbacks"],
            "corrupted_lookups": counters["corrupted_lookups"],
            "corrupted_rejections": counters["corrupted_rejections"],
            "invalidation_reasons": dict(sorted(invalidation_reasons.items())),
        }
    )
    return summary


def reference_summary(run: Mapping[str, Any], policy: str, seed: int) -> Dict[str, Any]:
    summary = {key: value for key, value in run.items() if key != "outcomes"}
    summary["policy"] = policy
    summary["seed"] = seed
    summary["certificate_lookups"] = 0
    summary["validity_checks"] = 0
    summary["valid_certificate_uses"] = 0
    summary["stale_certificate_uses"] = 0
    summary["invalidations"] = 0
    summary["fallbacks"] = 0
    summary["corrupted_lookups"] = 0
    summary["corrupted_rejections"] = 0
    summary["invalidation_reasons"] = {}
    return summary


def aggregate_records(records: Sequence[Mapping[str, Any]], policy: str, shared_issuance: bool) -> Dict[str, Any]:
    base = c1j.aggregate_summaries(records)
    if not records:
        return base
    count_fields = (
        "certificate_lookups",
        "validity_checks",
        "valid_certificate_uses",
        "stale_certificate_uses",
        "invalidations",
        "fallbacks",
        "corrupted_lookups",
        "corrupted_rejections",
    )
    for field in count_fields:
        base[field] = sum(int(item.get(field, 0)) for item in records)
    reason_counts: Counter[str] = Counter()
    for item in records:
        reason_counts.update(item.get("invalidation_reasons", {}))
    base["invalidation_reasons"] = dict(sorted(reason_counts.items()))
    total_task_cost = sum(float(item.get("mean_cost", 0.0)) * int(item.get("tasks", 0)) for item in records)
    total_tasks = sum(int(item.get("tasks", 0)) for item in records)
    issuance = CERTIFICATE_ISSUANCE_COST * 32 if shared_issuance and policy in CERTIFICATE_POLICIES else 0.0
    base["total_task_cost"] = total_task_cost
    base["certificate_issuance_cost"] = issuance
    base["total_cost_including_shared_issuance"] = total_task_cost + issuance
    base["mean_cost_including_shared_issuance"] = (total_task_cost + issuance) / max(1, total_tasks)
    base["certificate_use_cost"] = CERTIFICATE_USE_COST
    return base


def outcome_metrics(outcomes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute subgroup metrics directly from outcomes, not run summaries."""

    base = c1j.group_metrics(outcomes)
    if not outcomes:
        return base
    reason_counts: Counter[str] = Counter(
        str(item["certificate_status"])
        for item in outcomes
        if item.get("certificate_status") not in {None, "not_applicable", "valid", "stale_used"}
    )
    base.update(
        {
            "certificate_lookups": sum(bool(item.get("certificate_key")) for item in outcomes),
            "validity_checks": sum(item.get("certificate_status") not in {None, "not_applicable"} for item in outcomes),
            "valid_certificate_uses": sum(bool(item.get("certificate_valid")) for item in outcomes),
            "stale_certificate_uses": sum(item.get("source") == "stale_certificate" for item in outcomes),
            "invalidations": sum(bool(item.get("certificate_fallback")) for item in outcomes),
            "fallbacks": sum(bool(item.get("certificate_fallback")) for item in outcomes),
            "corrupted_lookups": sum(item.get("certificate_status") == "corrupted_certificate" for item in outcomes),
            "corrupted_rejections": sum(item.get("certificate_status") == "corrupted_certificate" for item in outcomes),
            "invalidation_reasons": dict(sorted(reason_counts.items())),
        }
    )
    return base


def add_group(
    run: Mapping[str, Any],
    by_condition: Dict[str, Dict[str, List[Dict[str, Any]]]],
    by_family: Dict[str, Dict[str, List[Dict[str, Any]]]],
    by_memory: Dict[str, Dict[str, List[Dict[str, Any]]]],
    policy: str,
) -> None:
    outcomes = run["outcomes"]
    for condition in c1j.CONDITIONS:
        by_condition[condition][policy].append(outcome_metrics([item for item in outcomes if item["condition"] == condition]))
    for family in c1j.HELDOUT_FAMILIES:
        by_family[family][policy].append(outcome_metrics([item for item in outcomes if item["family"] == family]))
    for memory_size in c1j.MEMORY_SIZES:
        by_memory[str(memory_size)][policy].append(outcome_metrics([item for item in outcomes if item["memory_size"] == memory_size]))


def source_hashes() -> Dict[str, str]:
    paths = [
        PROTOCOL,
        ROOT / "tools" / "run_crucible_c1l_certificate_amortization_20261021.py",
        ROOT / "tools" / "run_crucible_c1j_cpu_stress_20261021.py",
        ROOT / "tools" / "run_crucible_c1k_constrained_procedure_20261021.py",
        C1J_OUTPUT / "RESULTS.json",
        C1J_OUTPUT / "MODELS.json",
        C1J_OUTPUT / "SPLIT_MANIFEST.json",
        C1K_OUTPUT / "RESULTS.json",
        C1K_REPEAT_OUTPUT / "REPEAT_COMPARISON.json",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def failure_attribution(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    events = [
        {
            "seed": record["seed"],
            "task_id": outcome["task_id"],
            "family": outcome["family"],
            "condition": outcome["condition"],
            "memory_size": outcome["memory_size"],
            "decision": outcome["decision"],
            "source": outcome["source"],
            "certificate_status": outcome.get("certificate_status"),
            "certificate_id": outcome.get("certificate_id"),
        }
        for record in records
        for outcome in record["outcomes"]
        if outcome.get("harmful_transfer")
    ]

    def counts(field: str) -> Dict[str, int]:
        return dict(sorted(Counter(str(event[field]) for event in events).items()))

    return {
        "event_count": len(events),
        "by_condition": counts("condition"),
        "by_family": counts("family"),
        "by_memory_size": counts("memory_size"),
        "by_decision": counts("decision"),
        "by_source": counts("source"),
        "by_certificate_status": counts("certificate_status"),
        "events": events,
    }


def render_summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1l verification amortization and certificate invalidation",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; no GPU, cloud or Crucible neural training.",
        "",
        "## Aggregate paired results",
        "",
        "| Policy | Accuracy | Mean task cost | Mean cost incl. shared issuance | Harmful transfer | Valid certificate uses | Invalidations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, item in result["aggregate"].items():
        lines.append(
            f"| `{policy}` | {item['accuracy']:.4f} | {item['mean_cost']:.3f} | "
            f"{item.get('mean_cost_including_shared_issuance', item['mean_cost']):.3f} | "
            f"{item['harmful_transfer_rate']:.4f} | {item.get('valid_certificate_uses', 0)} | {item.get('invalidations', 0)} |"
        )
    lines.extend([
        "",
        "## Certificate catalogue",
        "",
        f"- Independent seed candidates: `{result['certificate_catalogue']['candidate_count']}`.",
        f"- Catalogue entries: `{result['certificate_catalogue']['catalogue_entries']}` / `{result['certificate_catalogue']['required_entries']}`.",
        f"- Catalogue complete: `{result['certificate_catalogue']['complete']}`.",
        f"- Test-task overlap: `{result['test_separation']['overlap_count']}`.",
        "",
        "## Cost decomposition",
        "",
        f"- Certificate lookup cost: `{CERTIFICATE_LOOKUP_COST}`.",
        f"- Certificate validity cost: `{CERTIFICATE_VALIDITY_COST}`.",
        f"- Issuance cost per certificate: `{CERTIFICATE_ISSUANCE_COST}`.",
        f"- Shared catalogue issuance cost: `{CERTIFICATE_ISSUANCE_COST * 32:.2f}`.",
        "",
        "## Invalidation reasons",
        "",
    ])
    for reason, count in result["aggregate"]["certificate_amortized"].get("invalidation_reasons", {}).items():
        lines.append(f"- `{reason}`: `{count}`.")
    lines.extend(["", "## Harmful-transfer attribution", ""])
    for policy, attribution in result["failure_attribution"].items():
        lines.append(f"### `{policy}` — `{attribution['event_count']}` events")
        for field in ("by_condition", "by_family", "by_memory_size", "by_decision", "by_source", "by_certificate_status"):
            lines.append(f"- `{field}`: `{attribution[field]}`.")
    lines.extend(["", "## Results by condition", "", "| Condition | Policy | Accuracy | Mean cost | Harmful transfer |", "| --- | --- | ---: | ---: | ---: |"])
    for condition, policies in result["by_condition"].items():
        for policy, item in policies.items():
            lines.append(f"| `{condition}` | `{policy}` | {item['accuracy']:.4f} | {item['mean_cost']:.3f} | {item['harmful_transfer_rate']:.4f} |")
    lines.extend(["", "## Results by held-out family", "", "| Family | Policy | Accuracy | Mean cost | Harmful transfer |", "| --- | --- | ---: | ---: | ---: |"])
    for family, policies in result["by_family"].items():
        for policy, item in policies.items():
            lines.append(f"| `{family}` | `{policy}` | {item['accuracy']:.4f} | {item['mean_cost']:.3f} | {item['harmful_transfer_rate']:.4f} |")
    lines.extend(["", "## Results by memory size", "", "| Memory size | Policy | Accuracy | Mean cost | Harmful transfer |", "| ---: | --- | ---: | ---: | ---: |"])
    for memory_size, policies in result["by_memory_size"].items():
        for policy, item in policies.items():
            lines.append(f"| `{memory_size}` | `{policy}` | {item['accuracy']:.4f} | {item['mean_cost']:.3f} | {item['harmful_transfer_rate']:.4f} |")
    lines.extend([
        "",
        "## Gates",
        "",
    ])
    for name, value in result["gates"].items():
        lines.append(f"- `{name}`: `{value}`.")
    lines.extend([
        "",
        "Failed gates remain valid results. These synthetic outcomes do not establish neural reasoning or CMS2 training value.",
        "",
        "## Source hashes",
        "",
    ])
    for path, digest in sorted(hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen_audit = {
        "c1j": verify_index(C1J_OUTPUT / "HASHES.json"),
        "c1k": verify_index(C1K_OUTPUT / "HASHES.json"),
        "c1k_repeat": verify_index(C1K_REPEAT_OUTPUT / "HASHES.json"),
    }
    tasks, input_audit = c1k.load_frozen_tasks()
    models = c1k.load_c1j_models()
    catalogue, catalogue_audit, certificate_seed_tasks = build_certificate_catalogue()
    corrupted_ids = sorted(
        certificate["certificate_id"]
        for certificate in catalogue.values()
        if c1j.stable_int("c1l-corruption", certificate["certificate_id"]) % 5 == 0
    )
    test_ids = {candidate.task.task_id for candidate in tasks}
    seed_ids = {candidate.task.task_id for candidate in certificate_seed_tasks}
    overlap = sorted(test_ids & seed_ids)

    write_json(OUTPUT / "CERTIFICATE_CATALOGUE.json", {"catalogue": catalogue, "corrupted_certificate_ids": corrupted_ids})
    write_json(OUTPUT / "CERTIFICATE_SEED_AUDIT.json", catalogue_audit)
    write_json(OUTPUT / "C1J_INPUT_AUDIT.json", {"artifact_audit": frozen_audit, "input_audit": input_audit})

    all_runs: List[Dict[str, Any]] = []
    by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_condition: Dict[str, Dict[str, List[Dict[str, Any]]]] = {condition: defaultdict(list) for condition in c1j.CONDITIONS}
    by_family: Dict[str, Dict[str, List[Dict[str, Any]]]] = {family: defaultdict(list) for family in c1j.HELDOUT_FAMILIES}
    by_memory: Dict[str, Dict[str, List[Dict[str, Any]]]] = {str(size): defaultdict(list) for size in c1j.MEMORY_SIZES}

    for seed in c1j.LEARNER_SEEDS:
        model = models[str(seed)]
        if model.updates != c1j.TRAIN_UPDATES:
            raise RuntimeError(f"Unexpected frozen model update count for seed {seed}: {model.updates}")
        reference_specs = (
            ("deterministic_reference", "deterministic", None),
            ("learned_constrained_reference", "learned_constrained", model),
            ("stateless_reference", "stateless", None),
        )
        for output_policy, c1k_policy, policy_model in reference_specs:
            run = c1k.evaluate_policy(tasks, c1k_policy, policy_model, seed)
            summary = reference_summary(run, output_policy, seed)
            record = {"seed": seed, "policy": output_policy, "summary": summary, "outcomes": run["outcomes"]}
            all_runs.append(record)
            by_policy[output_policy].append(summary)
            add_group(record, by_condition, by_family, by_memory, output_policy)
        for policy in ("certificate_amortized", "certificate_corrupted", "stale_certificate_control"):
            run = evaluate_certificate_policy(tasks, policy, seed, catalogue, corrupted_ids)
            summary = {key: value for key, value in run.items() if key != "outcomes"}
            record = {"seed": seed, "policy": policy, "summary": summary, "outcomes": run["outcomes"]}
            all_runs.append(record)
            by_policy[policy].append(summary)
            add_group(record, by_condition, by_family, by_memory, policy)

    aggregate = {
        policy: aggregate_records(records, policy, True)
        for policy, records in sorted(by_policy.items())
    }
    condition_results = {
        condition: {policy: aggregate_records(records, policy, False) for policy, records in sorted(policies.items())}
        for condition, policies in by_condition.items()
    }
    family_results = {
        family: {policy: aggregate_records(records, policy, False) for policy, records in sorted(policies.items())}
        for family, policies in by_family.items()
    }
    memory_results = {
        memory_size: {policy: aggregate_records(records, policy, False) for policy, records in sorted(policies.items())}
        for memory_size, policies in by_memory.items()
    }

    deterministic = aggregate["deterministic_reference"]
    primary = aggregate["certificate_amortized"]
    corrupted = aggregate["certificate_corrupted"]
    stale = aggregate["stale_certificate_control"]
    primary_safety = primary["harmful_transfer_rate"] == 0.0 and primary["accuracy"] >= deterministic["accuracy"] - 0.01
    primary_efficiency = primary["mean_cost_including_shared_issuance"] <= deterministic["mean_cost"] * 0.90
    corrupted_safety = corrupted["harmful_transfer_rate"] == 0.0 and corrupted["accuracy"] >= deterministic["accuracy"] - 0.01
    corrupted_detection = aggregate["certificate_corrupted"]["corrupted_rejections"] == aggregate["certificate_corrupted"]["corrupted_lookups"]

    frozen_deterministic = json.loads((C1K_OUTPUT / "RESULTS.json").read_text(encoding="utf-8"))["aggregate"]["deterministic"]
    reference_fields = ("accuracy", "mean_cost", "harmful_transfer_rate", "resolvable_accuracy")
    reference_exact = all(abs(float(deterministic[field]) - float(frozen_deterministic[field])) < 1e-12 for field in reference_fields)

    result = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "aggregate": aggregate,
        "by_condition": condition_results,
        "by_family": family_results,
        "by_memory_size": memory_results,
        "certificate_catalogue": catalogue_audit,
        "failure_attribution": {
            policy: failure_attribution([run for run in all_runs if run["policy"] == policy])
            for policy in ("certificate_amortized", "certificate_corrupted", "stale_certificate_control")
        },
        "certificate_controls": {
            "corrupted_certificate_count": len(corrupted_ids),
            "corrupted_certificate_ids": corrupted_ids,
            "issuance_cost_per_certificate": CERTIFICATE_ISSUANCE_COST,
            "shared_issuance_cost": CERTIFICATE_ISSUANCE_COST * len(catalogue),
            "lookup_cost": CERTIFICATE_LOOKUP_COST,
            "validity_cost": CERTIFICATE_VALIDITY_COST,
        },
        "test_separation": {
            "test_task_count": len(test_ids),
            "certificate_seed_task_count": len(seed_ids),
            "overlap_count": len(overlap),
            "overlap_task_ids": overlap,
            "test_labels_used_at_decision": False,
        },
        "reference_audit": {
            "c1k_deterministic_exact_match": reference_exact,
            "c1k_repeat_reproducible": json.loads((C1K_REPEAT_OUTPUT / "REPEAT_COMPARISON.json").read_text(encoding="utf-8"))["reproducible"],
        },
        "gates": {
            "catalogue_completeness": catalogue_audit["complete"],
            "test_separation": len(overlap) == 0,
            "primary_safety": primary_safety,
            "primary_efficiency_10_percent": primary_efficiency,
            "invalidation_safety": corrupted_safety,
            "corruption_detection": corrupted_detection,
            "stale_control_exposed_failure": stale["harmful_transfer_rate"] > 0.0,
            "frozen_reference_exact": reference_exact,
        },
        "config": {
            "source_experiment": c1j.EXPERIMENT,
            "source_safety_experiment": c1k.EXPERIMENT,
            "test_tasks_per_seed": len(tasks),
            "learner_seeds": list(c1j.LEARNER_SEEDS),
            "certificate_seed": CERTIFICATE_SEED,
            "certificate_seed_candidates": CERTIFICATE_SEED_CANDIDATES,
            "policies": list(POLICIES),
            "new_training_updates": 0,
            "execution": "CPU-only",
        },
    }

    write_json(OUTPUT / "RESULTS.json", result)
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
