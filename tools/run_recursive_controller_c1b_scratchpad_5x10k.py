#!/usr/bin/env python3
"""Run five sequential 10,000-update CPU continuations with a scratchpad.

The current validation-selected C1b-L100000 bandit snapshots are the starting
models.  A paired continuation arm carries the same model state without a
scratchpad; the second arm additionally persists feature-keyed action/reward
associations across five phases.  The prior experiment is never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_recursive_controller_c1b_l3000 as pilot  # noqa: E402


EXPERIMENT = "crucible_recursive_controller_c1b_scratchpad_5x10k_20261021"
OUTPUT = ROOT / "experiments" / EXPERIMENT
START_EXPERIMENT = "crucible_recursive_controller_c1b_l100000_20261021"
START_ROOT = ROOT / "experiments" / START_EXPERIMENT
START_RESULTS = START_ROOT / "RESULTS.json"
START_HASHES = START_ROOT / "HASHES.json"
MANIFEST_ROOT = START_ROOT / "manifests"
PROTOCOL = ROOT / "experiments" / EXPERIMENT / "PROTOCOL.md"

SEEDS = (42, 1337, 2026)
ARMS = ("no_scratchpad", "persistent_scratchpad")
PHASES = 5
STEPS_PER_PHASE = 10_000
TOTAL_UPDATES = PHASES * STEPS_PER_PHASE
EPSILON_START = 0.30
EPSILON_FLOOR = 0.05
MIN_TOTAL_OBSERVATIONS = 8
MIN_ACTION_OBSERVATIONS = 4
MIN_REWARD_MARGIN = 0.05
REWARD_STABILITY_DELTA = 0.10
FEATURE_ROUND_DIGITS = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_int(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def value_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def state_key(features: Sequence[float]) -> str:
    normalized = [round(float(value), FEATURE_ROUND_DIGITS) for value in features]
    return value_hash({"feature_vector": normalized})


def epsilon(global_update: int) -> float:
    return max(EPSILON_FLOOR, EPSILON_START * (1.0 - (global_update - 1) / TOTAL_UPDATES))


def greedy_action(model: Any, features: Sequence[float]) -> str:
    scores = model.scores(features)
    return max(pilot.ACTIONS, key=lambda action: (scores[action], -pilot.ACTIONS.index(action)))


class PersistentScratchpad:
    """Small JSON-serializable store of observed decision-time associations."""

    def __init__(self, seed: int, payload: Optional[Mapping[str, Any]] = None) -> None:
        self.seed = seed
        raw = payload or {}
        self.entries: Dict[str, Dict[str, Any]] = json.loads(json.dumps(raw.get("entries", {})))
        self.stats: Dict[str, int] = {
            "lookups": 0,
            "mature_hits": 0,
            "new_associations": 0,
            "reopened_associations": 0,
            "skipped_relearning_updates": 0,
            "model_updates_due_to_novel_evidence": 0,
            "training_observations": 0,
        }
        self.stats.update({str(key): int(value) for key, value in raw.get("stats", {}).items()})

    def association(self, key: str, *, count_lookup: bool = False) -> Optional[Dict[str, Any]]:
        if count_lookup:
            self.stats["lookups"] += 1
        entry = self.entries.get(key)
        if not entry or not entry.get("mature") or not entry.get("best_action"):
            return None
        if count_lookup:
            self.stats["mature_hits"] += 1
        return {
            "action": str(entry["best_action"]),
            "mean": float(entry["best_mean"]),
            "margin": float(entry["margin"]),
            "observations": int(entry["total_observations"]),
        }

    def observe(
        self,
        key: str,
        features: Sequence[float],
        action: str,
        reward: float,
        phase: int,
        global_update: int,
    ) -> Dict[str, Any]:
        self.stats["training_observations"] += 1
        normalized = [round(float(value), FEATURE_ROUND_DIGITS) for value in features]
        entry = self.entries.get(key)
        if entry is None:
            entry = {
                "feature_vector": normalized,
                "actions": {},
                "total_observations": 0,
                "best_action": None,
                "best_mean": None,
                "margin": 0.0,
                "mature": False,
                "status": "provisional",
                "first_phase": phase,
                "first_update": global_update,
                "last_phase": phase,
                "last_update": global_update,
                "reopen_count": 0,
            }
            self.entries[key] = entry
        prior_mature = bool(entry.get("mature"))
        prior_action = entry.get("best_action")
        prior_mean = entry.get("best_mean")
        action_stats = entry["actions"].setdefault(
            action,
            {"count": 0, "reward_sum": 0.0, "reward_mean": 0.0, "last_reward": 0.0},
        )
        action_stats["count"] = int(action_stats["count"]) + 1
        action_stats["reward_sum"] = float(action_stats["reward_sum"]) + float(reward)
        action_stats["reward_mean"] = action_stats["reward_sum"] / action_stats["count"]
        action_stats["last_reward"] = float(reward)
        entry["total_observations"] = int(entry["total_observations"]) + 1
        entry["last_phase"] = phase
        entry["last_update"] = global_update

        observed = [
            (name, stats)
            for name, stats in entry["actions"].items()
            if int(stats["count"]) >= MIN_ACTION_OBSERVATIONS
        ]
        observed.sort(key=lambda item: (float(item[1]["reward_mean"]), -pilot.ACTIONS.index(item[0])), reverse=True)
        if observed:
            best_action, best_stats = observed[0]
            second_mean = float(observed[1][1]["reward_mean"]) if len(observed) > 1 else float("-inf")
            margin = float(best_stats["reward_mean"]) - second_mean if len(observed) > 1 else 0.0
            mature = (
                int(entry["total_observations"]) >= MIN_TOTAL_OBSERVATIONS
                and len(observed) >= 2
                and margin >= MIN_REWARD_MARGIN
            )
            entry["best_action"] = best_action
            entry["best_mean"] = float(best_stats["reward_mean"])
            entry["margin"] = margin
            entry["mature"] = mature
            entry["status"] = "mature" if mature else "provisional"

        contradictory = (
            prior_mature
            and prior_action == action
            and prior_mean is not None
            and abs(float(reward) - float(prior_mean)) > REWARD_STABILITY_DELTA
        )
        if contradictory:
            entry["mature"] = False
            entry["status"] = "reopened"
            entry["reopen_count"] = int(entry.get("reopen_count", 0)) + 1
            self.stats["reopened_associations"] += 1
        if not prior_mature and entry.get("mature"):
            self.stats["new_associations"] += 1
        return {
            "contradictory": contradictory,
            "mature_after": bool(entry.get("mature")),
            "best_action_after": entry.get("best_action"),
            "best_mean_after": entry.get("best_mean"),
        }

    def payload(self) -> Dict[str, Any]:
        return {
            "schema": "c1b-persistent-scratchpad-v1",
            "seed": self.seed,
            "parameters": {
                "min_total_observations": MIN_TOTAL_OBSERVATIONS,
                "min_action_observations": MIN_ACTION_OBSERVATIONS,
                "min_reward_margin": MIN_REWARD_MARGIN,
                "reward_stability_delta": REWARD_STABILITY_DELTA,
                "feature_round_digits": FEATURE_ROUND_DIGITS,
            },
            "stats": dict(self.stats),
            "entries": self.entries,
        }


def load_start_snapshots() -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    source = json.loads(START_RESULTS.read_text(encoding="utf-8"))
    snapshots: Dict[int, Dict[str, Any]] = {}
    metadata: Dict[str, Any] = {
        "source_results": str(START_RESULTS.relative_to(ROOT)),
        "source_results_sha256": sha256_file(START_RESULTS),
        "source_hashes": str(START_HASHES.relative_to(ROOT)),
        "source_hashes_sha256": sha256_file(START_HASHES),
        "selected": {},
    }
    for seed_result in source["seeds"]:
        seed = int(seed_result["seed"])
        selected_update = int(seed_result["selected_checkpoint"])
        matching = [item for item in seed_result["checkpoints"] if int(item["update"]) == selected_update]
        if len(matching) != 1:
            raise RuntimeError(f"expected one selected checkpoint for seed {seed}")
        snapshot = matching[0]["learner"]
        snapshots[seed] = snapshot
        metadata["selected"][str(seed)] = {
            "checkpoint_update": selected_update,
            "snapshot_sha256": value_hash(snapshot),
            "selected_validation_score": seed_result["selected_validation_score"],
        }
    if tuple(sorted(snapshots)) != tuple(SEEDS):
        raise RuntimeError("starting learner seed set mismatch")
    return snapshots, metadata


def load_manifests() -> Dict[str, List[Dict[str, Any]]]:
    records = {
        split: pilot.load_records(MANIFEST_ROOT / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    validation = pilot.validate_manifests(records)
    if validation.get("input_leakage") or int(validation.get("oracle_checks", 0)) != 640:
        raise RuntimeError(f"manifest validation did not pass: {validation}")
    return records


def make_model(seed: int, snapshot: Mapping[str, Any], arm: str) -> Any:
    feature_count = len(snapshot["weights"][pilot.ACTIONS[0]])
    model = pilot.LinearContextualBandit(seed, feature_count)
    model.restore(snapshot)
    model.rng = random.Random(stable_int(EXPERIMENT, "exploration", seed, arm))
    return model


def make_schedule(seed: int, count: int) -> List[int]:
    rng = random.Random(stable_int(EXPERIMENT, "training-schedule", seed))
    return [rng.randrange(count) for _ in range(TOTAL_UPDATES)]


def select_action(
    model: Any,
    scratchpad: Optional[PersistentScratchpad],
    features: Sequence[float],
    key: str,
    global_update: int,
    rng: random.Random,
    *,
    training: bool,
) -> Tuple[str, str, float]:
    current_epsilon = epsilon(global_update)
    if training and rng.random() < current_epsilon:
        return pilot.ACTIONS[rng.randrange(len(pilot.ACTIONS))], "exploration", current_epsilon
    if scratchpad is not None:
        association = scratchpad.association(key, count_lookup=training)
        if association is not None:
            return association["action"], "scratchpad", current_epsilon
    return greedy_action(model, features), "model", current_epsilon


def evaluate_arm(
    records: Sequence[Mapping[str, Any]],
    split: str,
    seed: int,
    model: Any,
    scratchpad: Optional[PersistentScratchpad],
) -> Dict[str, Any]:
    results = []
    sources: Counter[str] = Counter()
    for record in records:
        episode = pilot.make_episode(record, split, seed)
        features = pilot.feature_vector(episode)
        key = state_key(features)
        action, source, _ = select_action(
            model,
            scratchpad,
            features,
            key,
            TOTAL_UPDATES,
            random.Random(0),
            training=False,
        )
        sources[source] += 1
        results.append(pilot.run_episode(episode, "learned_adaptive", first_action=action))
    metrics = pilot.metrics(results)
    metrics["selection_sources"] = dict(sources)
    metrics["scratchpad_hit_rate"] = sources["scratchpad"] / max(1, len(records))
    return metrics


def run_phase(
    records: Sequence[Mapping[str, Any]],
    seed: int,
    arm: str,
    phase: int,
    schedule: Sequence[int],
    model: Any,
    scratchpad: Optional[PersistentScratchpad],
    selection_rng: random.Random,
) -> Dict[str, Any]:
    phase_start = (phase - 1) * STEPS_PER_PHASE
    rewards: List[float] = []
    source_counts: Counter[str] = Counter()
    skipped = 0
    model_updates = 0
    reopened = 0
    for local_step in range(STEPS_PER_PHASE):
        global_update = phase_start + local_step + 1
        record = records[schedule[phase_start + local_step]]
        episode = pilot.make_episode(record, "train", seed)
        features = pilot.feature_vector(episode)
        key = state_key(features)
        prior = scratchpad.association(key, count_lookup=True) if scratchpad is not None else None
        action, source, _ = select_action(
            model,
            scratchpad,
            features,
            key,
            global_update,
            selection_rng,
            training=True,
        )
        source_counts[source] += 1
        result = pilot.run_episode(episode, "learned_adaptive", first_action=action)
        reward = pilot.reward_for(result)
        rewards.append(reward)
        stable_repeat = (
            scratchpad is not None
            and prior is not None
            and action == prior["action"]
            and abs(float(reward) - float(prior["mean"])) <= REWARD_STABILITY_DELTA
        )
        if stable_repeat:
            skipped += 1
            scratchpad.stats["skipped_relearning_updates"] += 1
        else:
            model.update(action, features, reward, global_update)
            model_updates += 1
            if scratchpad is not None:
                scratchpad.stats["model_updates_due_to_novel_evidence"] += 1
        if scratchpad is not None:
            observed = scratchpad.observe(key, features, action, reward, phase, global_update)
            reopened += int(observed["contradictory"])
    return {
        "phase": phase,
        "seed": seed,
        "arm": arm,
        "updates": STEPS_PER_PHASE,
        "global_update_start": phase_start + 1,
        "global_update_end": phase_start + STEPS_PER_PHASE,
        "mean_training_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "model_updates_applied": model_updates,
        "relearning_updates_skipped": skipped,
        "selection_sources": dict(source_counts),
        "associations_reopened": reopened,
    }


def phase_record(
    records: Sequence[Mapping[str, Any]],
    seed: int,
    arm: str,
    phase_info: Mapping[str, Any],
    model: Any,
    scratchpad: Optional[PersistentScratchpad],
) -> Dict[str, Any]:
    validation = evaluate_arm(records, "validation", seed, model, scratchpad)
    result = dict(phase_info)
    result["validation"] = validation
    result["model_snapshot_sha256"] = value_hash(model.snapshot())
    result["scratchpad_entries"] = len(scratchpad.entries) if scratchpad is not None else 0
    result["scratchpad_mature_entries"] = (
        sum(bool(entry.get("mature")) for entry in scratchpad.entries.values()) if scratchpad is not None else 0
    )
    result["scratchpad_stats"] = dict(scratchpad.stats) if scratchpad is not None else {}
    return result


def summary_markdown(result: Mapping[str, Any], start_metadata: Mapping[str, Any], source_hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1b persistent scratchpad continuation",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; five sequential 10,000-update phases per seed.",
        "",
        "## Verdict",
        "",
        "This is a small policy-learner continuation experiment. It does not establish neural memory, cumulative language learning or general intelligence.",
        f"- Seeds: **{len(result['seeds'])}/{len(SEEDS)}**.",
        f"- Phases per arm and seed: **{PHASES}**.",
        f"- Updates per phase: **{STEPS_PER_PHASE}**.",
        f"- Total continuation updates per arm and seed: **{TOTAL_UPDATES}**.",
        "",
        "## Final untouched-test comparison",
        "",
        "| Arm | Accuracy | Mean cost | False corrections | Unresolved | Cost/accuracy score | Scratchpad hit rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        aggregate = result["aggregate_final"][arm]
        lines.append(
            f"| `{arm}` | {aggregate['accuracy']:.4f} | {aggregate['mean_cost']:.3f} | "
            f"{aggregate['false_corrections']:.2f} | {aggregate['unresolved']:.2f} | "
            f"{aggregate['cost_accuracy_score']:.4f} | {aggregate['scratchpad_hit_rate']:.4f} |"
        )
    lines.extend(["", "## Validation progression", "", "| Seed | Arm | Phase | Accuracy | Mean cost | Score | Scratchpad entries | Mature entries | Skipped updates |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for seed_result in result["seeds"]:
        for arm in ARMS:
            for phase in seed_result["arms"][arm]["phases"]:
                validation = phase["validation"]
                lines.append(
                    f"| {seed_result['seed']} | `{arm}` | {phase['phase']} | {validation['accuracy']:.4f} | "
                    f"{validation['mean_cost']:.3f} | {validation['cost_accuracy_score']:.4f} | "
                    f"{phase['scratchpad_entries']} | {phase['scratchpad_mature_entries']} | {phase['relearning_updates_skipped']} |"
                )
    lines.extend([
        "",
        "## Starting models",
        "",
        f"Source: `{start_metadata['source_results']}`; source SHA-256: `{start_metadata['source_results_sha256']}`.",
        "The three starts are the validation-selected C1b-L100000 learner snapshots; prior artifacts were not modified.",
        "",
        "| Seed | Selected checkpoint | Snapshot SHA-256 |",
        "| ---: | ---: | --- |",
    ])
    for seed in SEEDS:
        item = start_metadata["selected"][str(seed)]
        lines.append(f"| {seed} | {item['checkpoint_update']} | `{item['snapshot_sha256']}` |")
    lines.extend([
        "",
        "## Scratchpad interpretation",
        "",
        "The scratchpad records only decision-time feature keys, selected actions and observed training rewards. It is read-only during validation and test evaluation. Mature associations can bypass a stable redundant model update; materially different outcomes reopen the association.",
        "",
        "## Limitations",
        "",
        "This experiment uses the synthetic C1b corpus and a linear contextual bandit. A scratchpad hit is not proof that the underlying procedure is correct, and reduced updates are not equivalent to improved neural learning. The result does not transfer automatically to the 170M Crucible model or CMS2.",
        "",
        "## Source hashes",
        "",
    ])
    for path, digest in sorted(source_hashes.items()):
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
    start_snapshots, start_metadata = load_start_snapshots()
    records = load_manifests()
    schedule_by_seed = {str(seed): make_schedule(seed, len(records["train"])) for seed in SEEDS}
    schedule_metadata = {
        str(seed): {"count": len(schedule_by_seed[str(seed)]), "sha256": value_hash(schedule_by_seed[str(seed)])}
        for seed in SEEDS
    }
    write_json(OUTPUT / "START_STATE.json", start_metadata)
    write_json(OUTPUT / "MANIFEST_AUDIT.json", {
        "source_manifest_root": str(MANIFEST_ROOT.relative_to(ROOT)),
        "record_counts": {split: len(values) for split, values in records.items()},
        "validation": pilot.validate_manifests(records),
        "training_schedule": schedule_metadata,
    })
    write_json(OUTPUT / "RUN_CONFIG.json", {
        "experiment": EXPERIMENT,
        "starting_experiment": START_EXPERIMENT,
        "execution": "CPU-only; no GPU, cloud, checkpoint download or neural training",
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "phases": PHASES,
        "updates_per_phase": STEPS_PER_PHASE,
        "total_updates_per_arm_and_seed": TOTAL_UPDATES,
        "training_manifest": str((MANIFEST_ROOT / "train.jsonl").relative_to(ROOT)),
        "validation_manifest": str((MANIFEST_ROOT / "validation.jsonl").relative_to(ROOT)),
        "test_manifest": str((MANIFEST_ROOT / "test.jsonl").relative_to(ROOT)),
        "scratchpad": {
            "schema": "c1b-persistent-scratchpad-v1",
            "min_total_observations": MIN_TOTAL_OBSERVATIONS,
            "min_action_observations": MIN_ACTION_OBSERVATIONS,
            "min_reward_margin": MIN_REWARD_MARGIN,
            "reward_stability_delta": REWARD_STABILITY_DELTA,
        },
        "test_policy": "final phase only; never used for updates or selection",
    })
    for seed in SEEDS:
        for arm in ARMS:
            write_json(OUTPUT / "model_states" / f"seed_{seed}_start.json", start_snapshots[seed]) if arm == ARMS[0] else None
    seed_results: List[Dict[str, Any]] = []
    runs_jsonl = OUTPUT / "RUNS.jsonl"
    with runs_jsonl.open("w", encoding="utf-8") as run_handle:
        for seed in SEEDS:
            arm_results: Dict[str, Any] = {}
            for arm in ARMS:
                model = make_model(seed, start_snapshots[seed], arm)
                scratchpad = PersistentScratchpad(seed) if arm == "persistent_scratchpad" else None
                schedule = schedule_by_seed[str(seed)]
                selection_rng = random.Random(stable_int(EXPERIMENT, "selection", seed))
                phases: List[Dict[str, Any]] = []
                for phase in range(1, PHASES + 1):
                    phase_info = run_phase(records["train"], seed, arm, phase, schedule, model, scratchpad, selection_rng)
                    record = phase_record(records["validation"], seed, arm, phase_info, model, scratchpad)
                    phases.append(record)
                    write_json(OUTPUT / "model_states" / f"seed_{seed}_{arm}_phase_{phase:02d}.json", model.snapshot())
                    if scratchpad is not None:
                        write_json(OUTPUT / "scratchpads" / f"seed_{seed}_phase_{phase:02d}.json", scratchpad.payload())
                    run_handle.write(json.dumps(record, sort_keys=True) + "\n")
                    run_handle.flush()
                final_test = evaluate_arm(records["test"], "test", seed, model, scratchpad)
                final_validation = phases[-1]["validation"]
                arm_results[arm] = {
                    "phases": phases,
                    "final_validation": final_validation,
                    "final_test": final_test,
                    "final_model_snapshot_sha256": value_hash(model.snapshot()),
                    "final_scratchpad_sha256": value_hash(scratchpad.payload()) if scratchpad is not None else None,
                    "final_scratchpad_stats": dict(scratchpad.stats) if scratchpad is not None else {},
                }
                if scratchpad is not None:
                    write_json(OUTPUT / "scratchpads" / f"seed_{seed}_final.json", scratchpad.payload())
            seed_results.append({"seed": seed, "arms": arm_results})
    aggregate_final: Dict[str, Dict[str, float]] = {}
    metric_names = ("accuracy", "mean_cost", "false_corrections", "unresolved", "cost_accuracy_score", "scratchpad_hit_rate")
    for arm in ARMS:
        aggregate_final[arm] = {
            metric: sum(float(seed_result["arms"][arm]["final_test"][metric]) for seed_result in seed_results) / len(seed_results)
            for metric in metric_names
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
    source_paths = [
        START_RESULTS,
        START_HASHES,
        MANIFEST_ROOT / "train.jsonl",
        MANIFEST_ROOT / "validation.jsonl",
        MANIFEST_ROOT / "test.jsonl",
        ROOT / "tools" / "run_recursive_controller_c1b_scratchpad_5x10k.py",
        ROOT / "tools" / "run_recursive_controller_c1b_l3000.py",
        ROOT / "tools" / "recursive_controller.py",
        ROOT / "tools" / "cognitive_task_suite.py",
        PROTOCOL,
    ]
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}
    (OUTPUT / "RESULTS_SUMMARY.md").write_text(summary_markdown(result, start_metadata, source_hashes), encoding="utf-8")
    artifact_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "HASHES.json"
    }
    write_json(OUTPUT / "HASHES.json", artifact_hashes)
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "elapsed_seconds": result["elapsed_seconds"], "seeds": len(SEEDS), "phases": PHASES, "updates_per_phase": STEPS_PER_PHASE}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
