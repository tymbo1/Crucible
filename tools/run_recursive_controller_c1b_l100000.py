#!/usr/bin/env python3
"""Run the separately versioned 100,000-update C1b CPU pilot.

The implementation is intentionally delegated to the frozen C1b-L3000 runner
after changing only the experiment name, output directory, update count and
checkpoint schedule.  The earlier C1b-L3000 source and artifacts are not
modified by this wrapper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_recursive_controller_c1b_l3000 as pilot  # noqa: E402


EXPERIMENT = "crucible_recursive_controller_c1b_l100000_20261021"
OUTPUT = ROOT / "experiments" / EXPERIMENT
UPDATES = 100000
CHECKPOINTS = (0, 5000, 10000, 25000, 50000, 75000, 90000, 100000)
SEEDS = (42, 1337, 2026)
WALL_CAP_SECONDS = 20.0 * 60.0


def main() -> int:
    pilot.EXPERIMENT = EXPERIMENT
    pilot.DEFAULT_OUTPUT = OUTPUT
    pilot.UPDATES = UPDATES
    pilot.CHECKPOINTS = CHECKPOINTS
    pilot.LEARNER_SEEDS = SEEDS
    pilot.WALL_CAP_SECONDS = WALL_CAP_SECONDS

    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    built = pilot.build_manifests(OUTPUT)
    records = built["records"]
    manifest_validation = pilot.validate_manifests({
        split: pilot.load_manifest_records(OUTPUT / "manifests")[split]
        for split in ("train", "validation", "test")
    })
    pilot.write_json(OUTPUT / "MANIFEST_VALIDATION.json", manifest_validation)

    feature_count = len(pilot.feature_vector(pilot.make_episode(records["train"][0], "train", SEEDS[0])))
    run_config = {
        "experiment": EXPERIMENT,
        "task_seed": pilot.TASK_SEED,
        "learner_seeds": list(SEEDS),
        "updates": UPDATES,
        "checkpoints": list(CHECKPOINTS),
        "actions": list(pilot.ACTIONS),
        "policies": list(pilot.POLICIES) + ["learned_adaptive"],
        "training_conditions": ["R", "I", "U", "A"],
        "test_conditions": list(pilot.CONDITIONS),
        "bandit": {
            "kind": "linear_contextual_bandit",
            "feature_count": feature_count,
            "epsilon_start": 0.30,
            "epsilon_floor": 0.05,
            "learning_rate": "0.08/sqrt(1+0.02*action_count)",
        },
        "checkpoint_selection": "validation score; earliest checkpoint on tie; test evaluated once after selection",
        "future_competence": "disabled",
        "execution": "CPU-only; no GPU, cloud, model training or checkpoint download",
        "wall_cap_seconds": WALL_CAP_SECONDS,
        "governing_protocol": "experiments/crucible_recursive_controller_c1b_20261021/C1B_L100000_PROTOCOL.md",
    }
    pilot.write_json(OUTPUT / "RUN_CONFIG.json", run_config)

    seed_results = []
    for seed in SEEDS:
        seed_results.append(pilot.run_seed(records["train"], records["validation"], records["test"], seed, started))

    elapsed = perf_counter() - started
    result = {
        "experiment": EXPERIMENT,
        "status": "completed",
        "elapsed_seconds": round(elapsed, 6),
        "seeds": seed_results,
        "aggregate_test": {
            policy: {
                metric: sum(seed_result["test"][policy][metric] for seed_result in seed_results) / len(seed_results)
                for metric in ("accuracy", "mean_cost", "false_corrections", "unresolved", "verification_omissions", "cost_accuracy_score")
            }
            for policy in seed_results[0]["test"]
        },
    }
    pilot.write_json(OUTPUT / "RESULTS.json", result)
    with (OUTPUT / "TRAINING_CURVES.jsonl").open("w", encoding="utf-8") as handle:
        for seed_result in seed_results:
            for curve in seed_result["curves"]:
                handle.write(json.dumps(curve, sort_keys=True) + "\n")

    source_paths = [
        ROOT / "tools" / "run_recursive_controller_c1b_l100000.py",
        ROOT / "tools" / "run_recursive_controller_c1b_l3000.py",
        ROOT / "tools" / "cognitive_task_suite.py",
        ROOT / "tools" / "recursive_controller.py",
        ROOT / "tools" / "validate_cognitive_manifests.py",
        ROOT / "experiments" / "crucible_recursive_controller_c1b_20261021" / "PROTOCOL_V2.md",
        ROOT / "experiments" / "crucible_recursive_controller_c1b_20261021" / "C1B_L100000_PROTOCOL.md",
        ROOT / "experiments" / "crucible_cognitive_eval_v2_20261021" / "manifests" / "test.jsonl",
    ]
    source_hashes = {str(path.relative_to(ROOT)): pilot.sha256_file(path) for path in source_paths}
    summary = pilot.summary_markdown(result, built["manifest"], result["elapsed_seconds"], source_hashes)
    summary = summary.replace("# C1b-L3000 learned strategy-selection pilot", "# C1b-L100000 learned strategy-selection pilot")
    (OUTPUT / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")
    artifact_hashes = {
        str(path.relative_to(ROOT)): pilot.sha256_file(path)
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "HASHES.json"
    }
    pilot.write_json(OUTPUT / "HASHES.json", artifact_hashes)
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "elapsed_seconds": result["elapsed_seconds"], "seeds": len(seed_results), "updates_per_seed": UPDATES}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
