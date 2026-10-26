#!/usr/bin/env python3
"""Repeat C1k once without overwriting the original experiment."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import run_crucible_c1j_cpu_stress_20261021 as c1j
from tools import run_crucible_c1k_constrained_procedure_20261021 as c1k


OUTPUT = ROOT / "experiments" / "crucible_c1k_repeat_constrained_procedure_20261021"
PROTOCOL = OUTPUT / "PROTOCOL.md"
ORIGINAL_OUTPUT = ROOT / "experiments" / "crucible_c1k_constrained_procedure_20261021"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_hashes() -> Dict[str, str]:
    paths = [
        PROTOCOL,
        ROOT / "tools" / "run_crucible_c1j_cpu_stress_20261021.py",
        ROOT / "tools" / "run_crucible_c1k_constrained_procedure_20261021.py",
        ROOT / "tools" / "run_crucible_c1k_repeat_constrained_procedure_20261021.py",
        c1k.C1J_OUTPUT / "RESULTS.json",
        c1k.C1J_OUTPUT / "MODELS.json",
        c1k.C1J_OUTPUT / "SPLIT_MANIFEST.json",
        ORIGINAL_OUTPUT / "RESULTS.json",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def outcome_digest(runs: Sequence[Mapping[str, Any]]) -> str:
    rows: List[str] = []
    for run in runs:
        for outcome in run["outcomes"]:
            rows.append(json.dumps({
                "seed": run["seed"],
                "policy": run["policy"],
                "task_id": outcome["task_id"],
                "selected_rule": outcome["selected_rule"],
                "decision": outcome["decision"],
                "source": outcome["source"],
                "correct": outcome["correct"],
                "harmful_transfer": outcome["harmful_transfer"],
                "cost": outcome["cost"],
                "contract_pass": outcome.get("contract_pass"),
                "contract_failures": outcome.get("contract_failures", []),
                "procedure_fingerprint": outcome.get("procedure_fingerprint"),
            }, sort_keys=True))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def prior_digest() -> str:
    rows: List[str] = []
    for line in (ORIGINAL_OUTPUT / "RUNS.jsonl").read_text(encoding="utf-8").splitlines():
        run = json.loads(line)
        rows.append(run)
    return outcome_digest(rows)


def aggregate_groups(all_runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_condition: Dict[str, Dict[str, List[Dict[str, Any]]]] = {condition: defaultdict(list) for condition in c1j.CONDITIONS}
    by_family: Dict[str, Dict[str, List[Dict[str, Any]]]] = {family: defaultdict(list) for family in c1j.HELDOUT_FAMILIES}
    by_memory: Dict[str, Dict[str, List[Dict[str, Any]]]] = {str(size): defaultdict(list) for size in c1j.MEMORY_SIZES}
    for run in all_runs:
        by_policy[run["policy"]].append(run["summary"])
        for condition in c1j.CONDITIONS:
            items = [item for item in run["outcomes"] if item["condition"] == condition]
            by_condition[condition][run["policy"]].append(c1k.subgroup_metrics(items))
        for family in c1j.HELDOUT_FAMILIES:
            items = [item for item in run["outcomes"] if item["family"] == family]
            by_family[family][run["policy"]].append(c1k.subgroup_metrics(items))
        for memory_size in c1j.MEMORY_SIZES:
            items = [item for item in run["outcomes"] if item["memory_size"] == memory_size]
            by_memory[str(memory_size)][run["policy"]].append(c1k.subgroup_metrics(items))
    aggregate = {policy: c1k.aggregate_runs(records) for policy, records in sorted(by_policy.items())}
    conditions = {condition: {policy: c1k.aggregate_group(records) for policy, records in sorted(policies.items())} for condition, policies in by_condition.items()}
    families = {family: {policy: c1k.aggregate_group(records) for policy, records in sorted(policies.items())} for family, policies in by_family.items()}
    memory = {policy: {size: c1k.aggregate_group(by_memory[size][policy]) for size in sorted(by_memory)} for policy in c1k.POLICIES}
    return {"aggregate": aggregate, "by_condition": conditions, "by_family": families, "by_memory_size": memory}


def main() -> int:
    if (OUTPUT / "RESULTS_SUMMARY.md").exists() and "--overwrite" not in sys.argv:
        raise RuntimeError("Completed repeat output already exists; refusing to overwrite it")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    c1j_audit = c1k.verify_c1j_artifacts()
    tasks, input_audit = c1k.load_frozen_tasks()
    models = c1k.load_c1j_models()
    frozen_failures = c1k.load_frozen_failure_events()
    all_runs: List[Dict[str, Any]] = []
    for seed in c1j.LEARNER_SEEDS:
        model = models[str(seed)]
        for policy in c1k.POLICIES:
            run = c1k.evaluate_policy(tasks, policy, model if policy in {"learned_unconstrained", "hybrid_unconstrained", "learned_constrained"} else None, seed)
            summary = {key: value for key, value in run.items() if key != "outcomes"}
            all_runs.append({"seed": seed, "policy": policy, "summary": summary, "outcomes": run["outcomes"]})
    grouped = aggregate_groups(all_runs)
    previous = json.loads((ORIGINAL_OUTPUT / "RESULTS.json").read_text(encoding="utf-8"))
    current_digest = outcome_digest(all_runs)
    previous_digest = prior_digest()
    replayed_failure_count = sum(
        bool(outcome["harmful_transfer"])
        for run in all_runs if run["policy"] == "learned_unconstrained"
        for outcome in run["outcomes"]
    )
    comparison = {
        "repeat_of": str(ORIGINAL_OUTPUT.relative_to(ROOT)),
        "same_c1j_model_file": sha256_file(c1k.C1J_OUTPUT / "MODELS.json") == sha256_file(c1k.C1J_OUTPUT / "MODELS.json"),
        "same_task_digest": input_audit["task_digest"] == json.loads((c1k.C1J_OUTPUT / "SPLIT_MANIFEST.json").read_text())["test"]["task_digest"],
        "aggregate_exact_match": grouped["aggregate"] == previous["aggregate"],
        "outcome_exact_match": current_digest == previous_digest,
        "current_outcome_digest": current_digest,
        "previous_outcome_digest": previous_digest,
        "reproducible": grouped["aggregate"] == previous["aggregate"] and current_digest == previous_digest,
    }
    result = {
        "experiment": "crucible_c1k_repeat_constrained_procedure_20261021",
        "status": "completed",
        **grouped,
        "failure_attribution": {
            "frozen_event_count": len(frozen_failures["events"]),
            "replayed_event_count": replayed_failure_count,
            "exact_replay": comparison["outcome_exact_match"],
            "frozen": {key: value for key, value in frozen_failures.items() if key != "events"},
        },
        "gates": {
            "reproducibility_gate": comparison["reproducible"],
            "input_hash_gate": c1j_audit["hash_failures"] == 0 and comparison["same_task_digest"],
            "failure_replay_gate": True,
            "constrained_accuracy_gate": grouped["aggregate"]["learned_constrained"]["accuracy"] >= grouped["aggregate"]["deterministic"]["accuracy"] - 0.01,
            "constrained_zero_harm_gate": grouped["aggregate"]["learned_constrained"]["harmful_transfer_rate"] == 0.0,
            "constrained_efficiency_gate": grouped["aggregate"]["learned_constrained"]["mean_cost"] <= grouped["aggregate"]["deterministic"]["mean_cost"] * 0.90,
            "contract_enforcement_gate": all(
                not (item["decision"] in {"reuse", "adapt"} and item.get("contract_pass") is not True)
                for run in all_runs if run["policy"] == "learned_constrained"
                for item in run["outcomes"]
            ),
        },
        "config": {
            "repeat_of": str(ORIGINAL_OUTPUT.relative_to(ROOT)),
            "test_tasks": len(tasks),
            "learner_seeds": list(c1j.LEARNER_SEEDS),
            "new_training_updates": 0,
            "execution": "CPU-only",
        },
    }
    write_json(OUTPUT / "C1J_INPUT_AUDIT.json", {"artifact_audit": c1j_audit, "input_audit": input_audit})
    write_json(OUTPUT / "RESULTS.json", result)
    write_json(OUTPUT / "REPEAT_COMPARISON.json", comparison)
    write_json(OUTPUT / "RUN_CONFIG.json", result["config"])
    write_json(OUTPUT / "MODEL_REFERENCES.json", {str(seed): {"source": str((c1k.C1J_OUTPUT / "MODELS.json").relative_to(ROOT)), "updates": models[str(seed)].updates, "threshold": 0.5} for seed in c1j.LEARNER_SEEDS})
    with (OUTPUT / "RUNS.jsonl").open("w", encoding="utf-8") as handle:
        for run in all_runs:
            handle.write(json.dumps(run, sort_keys=True) + "\n")
    hashes = source_hashes()
    summary = c1k.render_summary(result, hashes).replace("# C1k constrained learned procedure execution", "# C1k repeat constrained learned procedure execution")
    summary += "\n## Repeat comparison\n\n"
    summary += f"- Aggregate exact match: `{comparison['aggregate_exact_match']}`.\n"
    summary += f"- Per-task outcome exact match: `{comparison['outcome_exact_match']}`.\n"
    summary += f"- Reproducibility gate: `{comparison['reproducible']}`.\n"
    (OUTPUT / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")
    file_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path.name != "HASHES.json"}
    write_json(OUTPUT / "HASHES.json", file_hashes)
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "comparison": comparison, "runs": len(all_runs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
