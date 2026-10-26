#!/usr/bin/env python3
"""Run the four-arm C1b persistent-scratchpad factorial ablation."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_recursive_controller_c1b_scratchpad_5x10k as base  # noqa: E402


EXPERIMENT = "crucible_recursive_controller_c1b_scratchpad_ablation_v2_20261021"
OUTPUT = ROOT / "experiments" / EXPERIMENT
ARMS = ("baseline", "memory_only", "skip_only", "full")
SEEDS = base.SEEDS
PHASES = base.PHASES
STEPS_PER_PHASE = base.STEPS_PER_PHASE
TOTAL_UPDATES = base.TOTAL_UPDATES


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def value_hash(value: Any) -> str:
    return base.value_hash(value)


def make_scratchpad(seed: int, arm: str) -> Optional[base.PersistentScratchpad]:
    if arm in {"memory_only", "full", "skip_only"}:
        return base.PersistentScratchpad(seed)
    return None


def accessible_scratchpad(arm: str, scratchpad: Optional[base.PersistentScratchpad]) -> Optional[base.PersistentScratchpad]:
    return scratchpad if arm in {"memory_only", "full"} else None


def skip_scratchpad(arm: str, scratchpad: Optional[base.PersistentScratchpad]) -> Optional[base.PersistentScratchpad]:
    return scratchpad if arm in {"skip_only", "full"} else None


def run_phase(
    records: Sequence[Mapping[str, Any]],
    seed: int,
    arm: str,
    phase: int,
    schedule: Sequence[int],
    model: Any,
    scratchpad: Optional[base.PersistentScratchpad],
    selection_rng: random.Random,
) -> Dict[str, Any]:
    phase_start = (phase - 1) * STEPS_PER_PHASE
    rewards: List[float] = []
    source_counts: Counter[str] = Counter()
    skipped = 0
    model_updates = 0
    reopened = 0
    exposed = accessible_scratchpad(arm, scratchpad)
    store = scratchpad
    detector = skip_scratchpad(arm, scratchpad)
    for local_step in range(STEPS_PER_PHASE):
        global_update = phase_start + local_step + 1
        record = records[schedule[phase_start + local_step]]
        episode = base.pilot.make_episode(record, "train", seed)
        features = base.pilot.feature_vector(episode)
        key = base.state_key(features)
        prior = detector.association(key, count_lookup=True) if detector is not None else None
        action, source, _ = base.select_action(
            model,
            exposed,
            features,
            key,
            global_update,
            selection_rng,
            training=True,
        )
        source_counts[source] += 1
        result = base.pilot.run_episode(episode, "learned_adaptive", first_action=action)
        reward = base.pilot.reward_for(result)
        rewards.append(reward)
        stable_repeat = (
            detector is not None
            and prior is not None
            and action == prior["action"]
            and abs(float(reward) - float(prior["mean"])) <= base.REWARD_STABILITY_DELTA
        )
        if stable_repeat:
            skipped += 1
            detector.stats["skipped_relearning_updates"] += 1
        else:
            model.update(action, features, reward, global_update)
            model_updates += 1
            if store is not None:
                store.stats["model_updates_due_to_novel_evidence"] += 1
        if store is not None:
            observed = store.observe(key, features, action, reward, phase, global_update)
            reopened += int(observed["contradictory"])
    return {
        "phase": phase,
        "seed": seed,
        "arm": arm,
        "updates": STEPS_PER_PHASE,
        "global_update_start": phase_start + 1,
        "global_update_end": phase_start + STEPS_PER_PHASE,
        "mean_training_reward": sum(rewards) / len(rewards),
        "model_updates_applied": model_updates,
        "relearning_updates_skipped": skipped,
        "selection_sources": dict(source_counts),
        "associations_reopened": reopened,
    }


def evaluate_arm(
    records: Sequence[Mapping[str, Any]],
    split: str,
    seed: int,
    arm: str,
    model: Any,
    scratchpad: Optional[base.PersistentScratchpad],
) -> Dict[str, Any]:
    exposed = accessible_scratchpad(arm, scratchpad)
    return base.evaluate_arm(records, split, seed, model, exposed)


def phase_record(
    validation_records: Sequence[Mapping[str, Any]],
    seed: int,
    arm: str,
    phase_info: Mapping[str, Any],
    model: Any,
    scratchpad: Optional[base.PersistentScratchpad],
) -> Dict[str, Any]:
    result = dict(phase_info)
    result["validation"] = evaluate_arm(validation_records, "validation", seed, arm, model, scratchpad)
    result["model_snapshot_sha256"] = value_hash(model.snapshot())
    result["scratchpad_entries"] = len(scratchpad.entries) if scratchpad is not None else 0
    result["scratchpad_mature_entries"] = (
        sum(bool(entry.get("mature")) for entry in scratchpad.entries.values()) if scratchpad is not None else 0
    )
    result["scratchpad_stats"] = dict(scratchpad.stats) if scratchpad is not None else {}
    return result


def source_hashes(protocol: Path, runner: Path, start_results: Path, start_hashes: Path, manifest_root: Path) -> Dict[str, str]:
    paths = [
        protocol,
        runner,
        start_results,
        start_hashes,
        manifest_root / "train.jsonl",
        manifest_root / "validation.jsonl",
        manifest_root / "test.jsonl",
        ROOT / "tools" / "run_recursive_controller_c1b_scratchpad_5x10k.py",
        ROOT / "tools" / "run_recursive_controller_c1b_l3000.py",
        ROOT / "tools" / "recursive_controller.py",
        ROOT / "tools" / "cognitive_task_suite.py",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def summary_markdown(result: Mapping[str, Any], sources: Mapping[str, str]) -> str:
    lines = [
        "# C1b scratchpad factorial ablation",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; four paired arms, five sequential 10,000-update phases.",
        "",
        "## Final untouched-test comparison",
        "",
        "| Arm | Accuracy | Mean cost | False corrections | Unresolved | Score | Scratchpad hit rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        item = result["aggregate_final"][arm]
        lines.append(
            f"| `{arm}` | {item['accuracy']:.4f} | {item['mean_cost']:.3f} | {item['false_corrections']:.2f} | "
            f"{item['unresolved']:.2f} | {item['cost_accuracy_score']:.4f} | {item['scratchpad_hit_rate']:.4f} |"
        )
    lines.extend([
        "",
        "Memory arms were also evaluated with scratchpad access disabled after phase five; those read-only comparisons are stored in `RESULTS.json`.",
        "",
        "## Validation progression",
        "",
        "| Seed | Arm | Phase | Accuracy | Mean cost | Score | Applied updates | Skipped updates | Mature entries |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for seed_result in result["seeds"]:
        for arm in ARMS:
            for phase in seed_result["arms"][arm]["phases"]:
                validation = phase["validation"]
                lines.append(
                    f"| {seed_result['seed']} | `{arm}` | {phase['phase']} | {validation['accuracy']:.4f} | {validation['mean_cost']:.3f} | "
                    f"{validation['cost_accuracy_score']:.4f} | {phase['model_updates_applied']} | {phase['relearning_updates_skipped']} | {phase['scratchpad_mature_entries']} |"
                )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "`memory_only` isolates accessible retained associations. `skip_only` uses the same private redundancy detector as `full` but cannot use stored associations to choose actions. `full` combines both mechanisms. This is a synthetic linear-bandit result, not neural memory or cumulative language learning.",
        "",
        "## Source hashes",
        "",
    ])
    for path, digest in sorted(sources.items()):
        lines.append(f"- `{path}`: `{digest}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    existing = [path for path in OUTPUT.iterdir() if path.name != "PROTOCOL.md"] if OUTPUT.exists() else []
    if existing:
        raise SystemExit(f"refusing to overwrite existing output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for child in ("model_states", "scratchpads"):
        (OUTPUT / child).mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    snapshots, start_metadata = base.load_start_snapshots()
    records = base.load_manifests()
    schedules = {str(seed): base.make_schedule(seed, len(records["train"])) for seed in SEEDS}
    schedule_metadata = {str(seed): {"count": len(schedules[str(seed)]), "sha256": value_hash(schedules[str(seed)])} for seed in SEEDS}
    write_json(OUTPUT / "START_STATE.json", start_metadata)
    write_json(OUTPUT / "MANIFEST_AUDIT.json", {
        "source_manifest_root": str(base.MANIFEST_ROOT.relative_to(ROOT)),
        "record_counts": {split: len(items) for split, items in records.items()},
        "validation": base.pilot.validate_manifests(records),
        "training_schedule": schedule_metadata,
    })
    write_json(OUTPUT / "RUN_CONFIG.json", {
        "experiment": EXPERIMENT,
        "starting_experiment": base.START_EXPERIMENT,
        "execution": "CPU-only; no GPU, cloud, checkpoint download or neural training",
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "phases": PHASES,
        "updates_per_phase": STEPS_PER_PHASE,
        "total_updates_per_arm_and_seed": TOTAL_UPDATES,
        "schedule_namespace": base.EXPERIMENT,
        "final_test_scratchpad_disabled_comparison": True,
    })
    for seed in SEEDS:
        write_json(OUTPUT / "model_states" / f"seed_{seed}_start.json", snapshots[seed])
    seed_results: List[Dict[str, Any]] = []
    with (OUTPUT / "RUNS.jsonl").open("w", encoding="utf-8") as run_handle:
        for seed in SEEDS:
            arm_results: Dict[str, Any] = {}
            for arm in ARMS:
                model = base.make_model(seed, snapshots[seed], arm)
                scratchpad = make_scratchpad(seed, arm)
                schedule = schedules[str(seed)]
                selection_rng = random.Random(base.stable_int(base.EXPERIMENT, "selection", seed))
                phases: List[Dict[str, Any]] = []
                for phase in range(1, PHASES + 1):
                    phase_info = run_phase(records["train"], seed, arm, phase, schedule, model, scratchpad, selection_rng)
                    record = phase_record(records["validation"], seed, arm, phase_info, model, scratchpad)
                    phases.append(record)
                    write_json(OUTPUT / "model_states" / f"seed_{seed}_{arm}_phase_{phase:02d}.json", model.snapshot())
                    if scratchpad is not None:
                        write_json(OUTPUT / "scratchpads" / f"seed_{seed}_{arm}_phase_{phase:02d}.json", scratchpad.payload())
                    run_handle.write(json.dumps(record, sort_keys=True) + "\n")
                    run_handle.flush()
                final_test = evaluate_arm(records["test"], "test", seed, arm, model, scratchpad)
                final_disabled = None
                if arm in {"memory_only", "full"}:
                    final_disabled = evaluate_arm(records["test"], "test", seed, arm, model, None)
                arm_results[arm] = {
                    "phases": phases,
                    "final_validation": phases[-1]["validation"],
                    "final_test": final_test,
                    "final_test_scratchpad_disabled": final_disabled,
                    "final_model_snapshot_sha256": value_hash(model.snapshot()),
                    "final_scratchpad_sha256": value_hash(scratchpad.payload()) if scratchpad is not None else None,
                    "final_scratchpad_stats": dict(scratchpad.stats) if scratchpad is not None else {},
                }
                if scratchpad is not None:
                    write_json(OUTPUT / "scratchpads" / f"seed_{seed}_{arm}_final.json", scratchpad.payload())
            seed_results.append({"seed": seed, "arms": arm_results})
    metric_names = ("accuracy", "mean_cost", "false_corrections", "unresolved", "cost_accuracy_score", "scratchpad_hit_rate")
    aggregate_final = {
        arm: {
            metric: sum(float(seed_result["arms"][arm]["final_test"][metric]) for seed_result in seed_results) / len(seed_results)
            for metric in metric_names
        }
        for arm in ARMS
    }
    result = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "elapsed_seconds": round(perf_counter() - started, 6),
        "seeds": seed_results,
        "aggregate_final": aggregate_final,
        "schedule_metadata": schedule_metadata,
    }
    write_json(OUTPUT / "RESULTS.json", result)
    protocol = ROOT / "experiments" / EXPERIMENT / "PROTOCOL.md"
    runner = ROOT / "tools" / "run_recursive_controller_c1b_scratchpad_ablation_20261021.py"
    sources = source_hashes(protocol, runner, base.START_RESULTS, base.START_HASHES, base.MANIFEST_ROOT)
    (OUTPUT / "RESULTS_SUMMARY.md").write_text(summary_markdown(result, sources), encoding="utf-8")
    hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "HASHES.json"
    }
    write_json(OUTPUT / "HASHES.json", hashes)
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "elapsed_seconds": result["elapsed_seconds"], "arms": len(ARMS), "seeds": len(SEEDS), "phases": PHASES, "updates_per_phase": STEPS_PER_PHASE}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
