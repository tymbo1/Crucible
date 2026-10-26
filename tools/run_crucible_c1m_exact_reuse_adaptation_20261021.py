#!/usr/bin/env python3
"""Run C1m: exact certificate reuse versus independently verified adaptation."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_crucible_c1j_cpu_stress_20261021 as c1j
from tools import run_crucible_c1k_constrained_procedure_20261021 as c1k
from tools import run_crucible_c1l_certificate_amortization_20261021 as c1l


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_c1m_exact_reuse_adaptation_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
C1J_OUTPUT = ROOT / "experiments" / "crucible_c1j_cpu_stress_20261021"
C1K_OUTPUT = ROOT / "experiments" / "crucible_c1k_constrained_procedure_20261021"
C1K_REPEAT_OUTPUT = ROOT / "experiments" / "crucible_c1k_repeat_constrained_procedure_20261021"
C1L_OUTPUT = ROOT / "experiments" / "crucible_c1l_certificate_amortization_20261021"
EXPERIMENT = "crucible_c1m_exact_reuse_adaptation_20261021"
POLICIES = (
    "deterministic_reference",
    "exact_certificate_only",
    "broad_certificate_control",
    "learned_constrained_reference",
    "stateless_reference",
)
CERTIFICATE_POLICIES = {"exact_certificate_only", "broad_certificate_control"}
CERTIFICATE_ISSUANCE_COST = c1l.CERTIFICATE_ISSUANCE_COST
CERTIFICATE_USE_COST = c1l.CERTIFICATE_USE_COST


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_index(index_path: Path) -> Dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in sorted(index.items()):
        path = ROOT / relative
        actual = "missing" if not path.exists() else sha256_file(path)
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    if failures:
        raise RuntimeError(f"Frozen artifact hash failure: {index_path}: {failures}")
    return {
        "index": str(index_path.relative_to(ROOT)),
        "indexed_files": len(index),
        "hash_failures": 0,
        "index_sha256": sha256_file(index_path),
    }


def scope_category(task: c1j.Task) -> str:
    if task.condition == "applicable":
        return "exact_core_reuse"
    if task.condition == "adaptation":
        return "adaptation_transfer"
    return "changed_or_untrusted"


def exact_scope_eligible(task: c1j.Task, record: c1j.MemoryRecord) -> bool:
    """Observable gate for exact core reuse; no hidden target label is used."""

    return bool(
        task.family == record.family
        and task.variant == record.variant
        and task.structural_signature == "core"
        and record.structural_signature == "core"
        and task.environment_version == record.environment_version
    )


def deterministic_fallback(task: c1j.Task, record: c1j.MemoryRecord):
    return c1j.policy_decision(task, record, "deterministic", None)


def exact_certificate_run(
    tasks: Sequence[c1j.Candidate],
    seed: int,
    catalogue: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    outcomes: List[Dict[str, Any]] = []
    counters: Counter[str] = Counter()
    invalidation_reasons: Counter[str] = Counter()
    for candidate in tasks:
        task = candidate.task
        record = candidate.record
        selected: Optional[str]
        decision: str
        source: str
        certificate_status = "out_of_scope"
        certificate = None
        certificate_key = None
        if exact_scope_eligible(task, record):
            certificate, certificate_status, certificate_key = c1l.current_certificate_lookup(task, record, catalogue, ())
            counters["certificate_lookups"] += 1
            counters["validity_checks"] += 1
            if certificate is not None:
                counters["valid_certificate_uses"] += 1
                selected = record.rule
                decision = "reuse"
                source = "exact_certificate"
                cost = CERTIFICATE_USE_COST
            else:
                invalidation_reasons[certificate_status] += 1
                counters["invalidations"] += 1
                selected, fallback_cost, decision, _fallback_source, _probability = deterministic_fallback(task, record)
                source = "exact_certificate_fallback"
                cost = CERTIFICATE_USE_COST + fallback_cost
                counters["fallbacks"] += 1
        else:
            selected, cost, decision, _fallback_source, _probability = deterministic_fallback(task, record)
            source = "out_of_scope_deterministic"
            counters["out_of_scope_fallbacks"] += 1

        correct = c1j.safe_correct(task, selected)
        outcomes.append(
            {
                "split": candidate.split,
                "index": candidate.index,
                "task_id": task.task_id,
                "family": task.family,
                "condition": task.condition,
                "scope_category": scope_category(task),
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
                "probability": None,
                "label": candidate.label,
                "certificate_status": certificate_status,
                "certificate_key": certificate_key,
                "certificate_id": certificate["certificate_id"] if certificate is not None else None,
                "certificate_valid": certificate is not None,
                "certificate_fallback": source == "exact_certificate_fallback",
                "certificate_authorized_adaptation": False,
                "independent_adaptation_verification": task.condition == "adaptation",
                "procedure_executed": source == "exact_certificate" and decision == "reuse",
            }
        )
    summary = c1j.group_metrics(outcomes)
    summary.update(
        {
            "policy": "exact_certificate_only",
            "seed": seed,
            "outcomes": outcomes,
            "certificate_lookups": counters["certificate_lookups"],
            "validity_checks": counters["validity_checks"],
            "valid_certificate_uses": counters["valid_certificate_uses"],
            "invalidations": counters["invalidations"],
            "fallbacks": counters["fallbacks"],
            "out_of_scope_fallbacks": counters["out_of_scope_fallbacks"],
            "certificate_authorized_adaptations": 0,
            "independent_adaptation_verifications": sum(item["condition"] == "adaptation" for item in outcomes),
            "invalidation_reasons": dict(sorted(invalidation_reasons.items())),
        }
    )
    return summary


def annotate_reference(run: Mapping[str, Any], policy: str, seed: int) -> Dict[str, Any]:
    outcomes = []
    for item in run["outcomes"]:
        copied = dict(item)
        copied["scope_category"] = scope_category(c1j.Task(
            task_id=copied["task_id"],
            family=copied["family"],
            condition=copied["condition"],
            phase=copied["split"],
            variant="source" if copied["condition"] != "adaptation" else "holdout",
            surface_group="",
            structural_signature="transfer" if copied["condition"] == "adaptation" else "core",
            environment_version=0,
            true_rule="",
            candidate_rules=(),
            probe_input=None,
            probe_output=None,
            target_input=None,
            target_output=None,
            probe_available=True,
            verifier_available=True,
            memory_size=copied["memory_size"],
            distractor_count=copied["distractor_count"],
        ))
        copied.setdefault("certificate_lookups", 0)
        copied.setdefault("certificate_valid", False)
        copied.setdefault("certificate_fallback", False)
        copied.setdefault("certificate_authorized_adaptation", False)
        copied.setdefault("independent_adaptation_verification", copied["condition"] == "adaptation")
        copied.setdefault("certificate_status", "not_applicable")
        copied.setdefault("certificate_key", None)
        outcomes.append(copied)
    summary = c1j.group_metrics(outcomes)
    summary.update(
        {
            "policy": policy,
            "seed": seed,
            "outcomes": outcomes,
            "certificate_lookups": 0,
            "validity_checks": 0,
            "valid_certificate_uses": 0,
            "invalidations": 0,
            "fallbacks": 0,
            "out_of_scope_fallbacks": 0,
            "certificate_authorized_adaptations": 0,
            "independent_adaptation_verifications": sum(item["condition"] == "adaptation" for item in outcomes),
            "invalidation_reasons": {},
        }
    )
    return summary


def annotate_broad(run: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    outcomes = []
    for item in run["outcomes"]:
        copied = dict(item)
        copied["seed"] = seed
        copied["scope_category"] = "exact_core_reuse" if item["condition"] == "applicable" else "adaptation_transfer" if item["condition"] == "adaptation" else "changed_or_untrusted"
        copied["certificate_authorized_adaptation"] = copied.get("source") == "certificate" and copied.get("decision") == "adapt"
        copied["independent_adaptation_verification"] = False
        outcomes.append(copied)
    summary = c1j.group_metrics(outcomes)
    original = {key: value for key, value in run.items() if key != "outcomes"}
    summary.update(original)
    summary.update({
        "policy": "broad_certificate_control",
        "seed": seed,
        "outcomes": outcomes,
        "certificate_authorized_adaptations": sum(item["certificate_authorized_adaptation"] for item in outcomes),
        "independent_adaptation_verifications": 0,
    })
    return summary


def outcome_metrics(outcomes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    base = c1j.group_metrics(outcomes)
    if not outcomes:
        return base
    base.update(
        {
            "valid_certificate_uses": sum(bool(item.get("certificate_valid")) for item in outcomes),
            "certificate_authorized_adaptations": sum(bool(item.get("certificate_authorized_adaptation")) for item in outcomes),
            "independent_adaptation_verifications": sum(bool(item.get("independent_adaptation_verification")) for item in outcomes),
            "certificate_fallbacks": sum(bool(item.get("certificate_fallback")) for item in outcomes),
        }
    )
    return base


def aggregate_records(records: Sequence[Mapping[str, Any]], policy: str) -> Dict[str, Any]:
    base = c1j.aggregate_summaries(records)
    if not records:
        return base
    for field in (
        "certificate_lookups",
        "validity_checks",
        "valid_certificate_uses",
        "invalidations",
        "fallbacks",
        "out_of_scope_fallbacks",
        "certificate_authorized_adaptations",
        "independent_adaptation_verifications",
        "certificate_fallbacks",
    ):
        base[field] = sum(int(item.get(field, 0)) for item in records)
    total_task_cost = sum(float(item.get("mean_cost", 0.0)) * int(item.get("tasks", 0)) for item in records)
    total_tasks = sum(int(item.get("tasks", 0)) for item in records)
    issuance = CERTIFICATE_ISSUANCE_COST * 16 if policy in CERTIFICATE_POLICIES else 0.0
    base["total_task_cost"] = total_task_cost
    base["certificate_issuance_cost"] = issuance
    base["mean_cost_including_shared_issuance"] = (total_task_cost + issuance) / max(1, total_tasks)
    return base


def add_groups(
    record: Mapping[str, Any],
    policy: str,
    by_scope: Dict[str, Dict[str, List[Dict[str, Any]]]],
    by_family: Dict[str, Dict[str, List[Dict[str, Any]]]],
    by_memory: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> None:
    outcomes = record["outcomes"]
    for scope in ("exact_core_reuse", "adaptation_transfer", "changed_or_untrusted"):
        by_scope[scope][policy].append(outcome_metrics([item for item in outcomes if item["scope_category"] == scope]))
    for family in c1j.HELDOUT_FAMILIES:
        by_family[family][policy].append(outcome_metrics([item for item in outcomes if item["family"] == family]))
    for memory_size in c1j.MEMORY_SIZES:
        by_memory[str(memory_size)][policy].append(outcome_metrics([item for item in outcomes if item["memory_size"] == memory_size]))


def source_hashes() -> Dict[str, str]:
    paths = [
        PROTOCOL,
        ROOT / "tools" / "run_crucible_c1m_exact_reuse_adaptation_20261021.py",
        ROOT / "tools" / "run_crucible_c1j_cpu_stress_20261021.py",
        ROOT / "tools" / "run_crucible_c1k_constrained_procedure_20261021.py",
        ROOT / "tools" / "run_crucible_c1l_certificate_amortization_20261021.py",
        C1J_OUTPUT / "RESULTS.json",
        C1J_OUTPUT / "MODELS.json",
        C1J_OUTPUT / "SPLIT_MANIFEST.json",
        C1K_OUTPUT / "RESULTS.json",
        C1K_REPEAT_OUTPUT / "REPEAT_COMPARISON.json",
        C1L_OUTPUT / "RESULTS.json",
        C1L_OUTPUT / "CERTIFICATE_CATALOGUE.json",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def render_summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1m exact reuse versus adaptation",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; no GPU, cloud or Crucible neural training.",
        "",
        "## Aggregate paired results",
        "",
        "| Policy | Accuracy | Mean task cost | Mean cost incl. issuance | Harmful transfer | Valid cert uses | Cert-authorized adaptations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, item in result["aggregate"].items():
        lines.append(
            f"| `{policy}` | {item['accuracy']:.4f} | {item['mean_cost']:.3f} | "
            f"{item.get('mean_cost_including_shared_issuance', item['mean_cost']):.3f} | "
            f"{item['harmful_transfer_rate']:.4f} | {item.get('valid_certificate_uses', 0)} | "
            f"{item.get('certificate_authorized_adaptations', 0)} |"
        )
    lines.extend(["", "## Scope-category results", "", "| Scope | Policy | Accuracy | Mean cost | Harmful transfer | Independent adaptation verifications |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for scope, policies in result["by_scope"].items():
        for policy, item in policies.items():
            lines.append(f"| `{scope}` | `{policy}` | {item['accuracy']:.4f} | {item['mean_cost']:.3f} | {item['harmful_transfer_rate']:.4f} | {item.get('independent_adaptation_verifications', 0)} |")
    lines.extend(["", "## Broad-control failure attribution", "", f"- Broad-control harmful events: `{result['broad_failure_attribution']['event_count']}`.", f"- By family: `{result['broad_failure_attribution']['by_family']}`.", f"- By scope: `{result['broad_failure_attribution']['by_scope']}`.", ""])
    lines.extend(["## Gates", ""])
    for name, value in result["gates"].items():
        lines.append(f"- `{name}`: `{value}`.")
    lines.extend(["", "C1m does not establish neural reasoning or CMS2 training value. Failed gates remain valid results.", "", "## Source hashes", ""])
    for path, digest in sorted(hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    return "\n".join(lines) + "\n"


def broad_failure_attribution(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    events = [
        outcome
        for record in records
        for outcome in record["outcomes"]
        if outcome.get("harmful_transfer")
    ]
    return {
        "event_count": len(events),
        "by_family": dict(sorted(Counter(item["family"] for item in events).items())),
        "by_scope": dict(sorted(Counter(item["scope_category"] for item in events).items())),
        "by_memory_size": dict(sorted(Counter(str(item["memory_size"]) for item in events).items())),
        "by_decision": dict(sorted(Counter(item["decision"] for item in events).items())),
        "events": [
            {"task_id": item["task_id"], "seed": item.get("seed"), "family": item["family"], "condition": item["condition"]}
            for item in events
        ],
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen_audit = {
        "c1j": verify_index(C1J_OUTPUT / "HASHES.json"),
        "c1k": verify_index(C1K_OUTPUT / "HASHES.json"),
        "c1k_repeat": verify_index(C1K_REPEAT_OUTPUT / "HASHES.json"),
        "c1l": verify_index(C1L_OUTPUT / "HASHES.json"),
    }
    tasks, input_audit = c1k.load_frozen_tasks()
    models = c1k.load_c1j_models()
    frozen_catalogue_file = json.loads((C1L_OUTPUT / "CERTIFICATE_CATALOGUE.json").read_text(encoding="utf-8"))
    catalogue = frozen_catalogue_file["catalogue"]
    certificate_seed_audit = json.loads((C1L_OUTPUT / "CERTIFICATE_SEED_AUDIT.json").read_text(encoding="utf-8"))
    regenerated_catalogue, regenerated_audit, certificate_seed_tasks = c1l.build_certificate_catalogue()
    if regenerated_audit["candidate_task_digest"] != certificate_seed_audit["candidate_task_digest"]:
        raise RuntimeError("C1l certificate-seed digest changed")
    if set(regenerated_catalogue) != set(catalogue):
        raise RuntimeError("C1l certificate catalogue keys changed")
    test_ids = {candidate.task.task_id for candidate in tasks}
    seed_ids = {candidate.task.task_id for candidate in certificate_seed_tasks}
    overlap = sorted(test_ids & seed_ids)

    write_json(OUTPUT / "C1J_INPUT_AUDIT.json", {"artifact_audit": frozen_audit, "input_audit": input_audit})
    write_json(OUTPUT / "C1L_CATALOGUE_AUDIT.json", {"frozen": certificate_seed_audit, "regenerated": regenerated_audit, "catalogue_entries": len(catalogue), "test_seed_overlap": overlap})

    all_runs: List[Dict[str, Any]] = []
    by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_scope: Dict[str, Dict[str, List[Dict[str, Any]]]] = {scope: defaultdict(list) for scope in ("exact_core_reuse", "adaptation_transfer", "changed_or_untrusted")}
    by_family: Dict[str, Dict[str, List[Dict[str, Any]]]] = {family: defaultdict(list) for family in c1j.HELDOUT_FAMILIES}
    by_memory: Dict[str, Dict[str, List[Dict[str, Any]]]] = {str(size): defaultdict(list) for size in c1j.MEMORY_SIZES}

    for seed in c1j.LEARNER_SEEDS:
        model = models[str(seed)]
        reference_specs = (
            ("deterministic_reference", "deterministic", None),
            ("learned_constrained_reference", "learned_constrained", model),
            ("stateless_reference", "stateless", None),
        )
        for output_policy, c1k_policy, policy_model in reference_specs:
            run = c1k.evaluate_policy(tasks, c1k_policy, policy_model, seed)
            summary = annotate_reference(run, output_policy, seed)
            record = {"seed": seed, "policy": output_policy, "summary": {key: value for key, value in summary.items() if key != "outcomes"}, "outcomes": summary["outcomes"]}
            all_runs.append(record)
            by_policy[output_policy].append(record["summary"])
            add_groups(record, output_policy, by_scope, by_family, by_memory)

        exact = exact_certificate_run(tasks, seed, catalogue)
        exact_record = {"seed": seed, "policy": "exact_certificate_only", "summary": {key: value for key, value in exact.items() if key != "outcomes"}, "outcomes": exact["outcomes"]}
        all_runs.append(exact_record)
        by_policy["exact_certificate_only"].append(exact_record["summary"])
        add_groups(exact_record, "exact_certificate_only", by_scope, by_family, by_memory)

        broad_raw = c1l.evaluate_certificate_policy(tasks, "certificate_amortized", seed, catalogue, ())
        broad = annotate_broad(broad_raw, seed)
        broad_record = {"seed": seed, "policy": "broad_certificate_control", "summary": {key: value for key, value in broad.items() if key != "outcomes"}, "outcomes": broad["outcomes"]}
        all_runs.append(broad_record)
        by_policy["broad_certificate_control"].append(broad_record["summary"])
        add_groups(broad_record, "broad_certificate_control", by_scope, by_family, by_memory)

    aggregate = {policy: aggregate_records(records, policy) for policy, records in sorted(by_policy.items())}
    scope_results = {scope: {policy: aggregate_records(records, policy) for policy, records in sorted(policies.items())} for scope, policies in by_scope.items()}
    family_results = {family: {policy: aggregate_records(records, policy) for policy, records in sorted(policies.items())} for family, policies in by_family.items()}
    memory_results = {memory: {policy: aggregate_records(records, policy) for policy, records in sorted(policies.items())} for memory, policies in by_memory.items()}

    deterministic = aggregate["deterministic_reference"]
    exact = aggregate["exact_certificate_only"]
    broad_records = [record for record in all_runs if record["policy"] == "broad_certificate_control"]
    broad_attr = broad_failure_attribution(broad_records)
    c1l_result = json.loads((C1L_OUTPUT / "RESULTS.json").read_text(encoding="utf-8"))
    prior_attr = c1l_result["failure_attribution"]["certificate_amortized"]
    broad_signature = sorted((event["seed"], event["task_id"]) for event in broad_attr["events"])
    prior_signature = sorted((event["seed"], event["task_id"]) for event in prior_attr["events"])
    frozen_deterministic = json.loads((C1K_OUTPUT / "RESULTS.json").read_text(encoding="utf-8"))["aggregate"]["deterministic"]
    reference_exact = all(abs(float(deterministic[field]) - float(frozen_deterministic[field])) < 1e-12 for field in ("accuracy", "mean_cost", "harmful_transfer_rate", "resolvable_accuracy"))

    result = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "aggregate": aggregate,
        "by_scope": scope_results,
        "by_family": family_results,
        "by_memory_size": memory_results,
        "broad_failure_attribution": broad_attr,
        "reference_audit": {
            "c1k_deterministic_exact_match": reference_exact,
            "broad_c1l_failure_signature_exact": broad_signature == prior_signature,
            "broad_event_count": len(broad_signature),
            "prior_c1l_event_count": len(prior_signature),
        },
        "test_separation": {
            "test_task_count": len(test_ids),
            "certificate_seed_task_count": len(seed_ids),
            "overlap_count": len(overlap),
            "overlap_task_ids": overlap,
            "test_labels_used_for_decision": False,
        },
        "gates": {
            "frozen_input_audit": all(item["hash_failures"] == 0 for item in frozen_audit.values()),
            "frozen_reference_exact": reference_exact,
            "test_separation": len(overlap) == 0,
            "exact_safety": exact["harmful_transfer_rate"] == 0.0 and exact["accuracy"] >= deterministic["accuracy"] - 0.01,
            "no_certificate_authorized_adaptations": exact.get("certificate_authorized_adaptations", 0) == 0,
            "broad_failure_signature_reproduced": broad_signature == prior_signature,
        },
        "config": {
            "source_experiments": [c1j.EXPERIMENT, c1k.EXPERIMENT, c1l.EXPERIMENT],
            "test_tasks_per_seed": len(tasks),
            "learner_seeds": list(c1j.LEARNER_SEEDS),
            "policies": list(POLICIES),
            "certificate_issuance_cost": CERTIFICATE_ISSUANCE_COST,
            "certificate_use_cost": CERTIFICATE_USE_COST,
            "new_training_updates": 0,
            "execution": "CPU-only",
        },
    }

    write_json(OUTPUT / "RESULTS.json", result)
    write_json(OUTPUT / "RUN_CONFIG.json", result["config"])
    with (OUTPUT / "RUNS.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_runs:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
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
