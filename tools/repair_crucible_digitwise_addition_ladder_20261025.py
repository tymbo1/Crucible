#!/usr/bin/env python3
"""Complete missing Stage-1 arms without writing large checkpoints."""

from __future__ import annotations

import gc
import importlib.util
import json
import random
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LADDER_PATH = ROOT / "tools" / "run_crucible_digitwise_addition_ladder_20261025.py"
OUTPUT = ROOT / "experiments" / "crucible_digitwise_addition_ladder_20261025" / "repair"


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ladder = import_module("digitwise_ladder_for_repair", LADDER_PATH)

CASES = {
    "curriculum_factorized_seed42": ("curriculum", "factorized", 42),
    "direct_factorized_seed1337": ("direct", "factorized", 1337),
    "curriculum_factorized_seed1337": ("curriculum", "factorized", 1337),
}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_case(case: str):
    arm, interface, seed = CASES[case]
    records = ladder.build_records()[interface][ladder.STAGES[0]]
    stage_a = ladder.stage_a
    torch = stage_a.torch
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    prepared = {split: ladder.prepare(values) for split, values in records.items()}
    order = list(range(len(prepared["train"])))
    random.Random(ladder.stable_seed(seed, interface, ladder.STAGES[0], "order")).shuffle(order)
    started = time.perf_counter()
    for step in range(1, ladder.STAGE_STEPS + 1):
        stage_a.train_step(model, optimizer, prepared["train"], stage_a.batch_indices(order, step))
    train_metrics = ladder.metrics(model, prepared["train"])
    validation_metrics = ladder.metrics(model, prepared["validation"])
    result = {
        "arm": arm,
        "interface": interface,
        "stage": ladder.STAGES[0],
        "seed": seed,
        "steps": ladder.STAGE_STEPS,
        "elapsed_seconds": time.perf_counter() - started,
        "train": train_metrics,
        "validation": validation_metrics,
        "train_exact": train_metrics["aggregate"]["teacher_forced_exact_accuracy"],
        "validation_exact": validation_metrics["aggregate"]["teacher_forced_exact_accuracy"],
        "fit_pass": train_metrics["aggregate"]["teacher_forced_exact_accuracy"] >= ladder.TRAIN_THRESHOLD,
        "advancement_pass": train_metrics["aggregate"]["teacher_forced_exact_accuracy"] >= ladder.TRAIN_THRESHOLD and validation_metrics["aggregate"]["teacher_forced_exact_accuracy"] >= ladder.VALIDATION_THRESHOLD,
        "checkpoint_written": False,
    }
    write_json(OUTPUT / case / "RESULTS.json", result)
    write_json(OUTPUT / case / "RUN_CONFIG.json", {"experiment": "crucible_digitwise_addition_ladder_20261025_repair", "case": case, "arm": arm, "interface": interface, "seed": seed, "steps": ladder.STAGE_STEPS, "checkpoint_written": False, "gpu_used": False, "test_evaluated": False})
    del model, optimizer
    gc.collect()
    print(json.dumps({"case": case, "train_exact": result["train_exact"], "validation_exact": result["validation_exact"], "status": "completed"}, sort_keys=True), flush=True)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    args = parser.parse_args()
    run_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
