#!/usr/bin/env python3
"""Finalize the digitwise ladder after checkpoint-light repair of missing arms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_digitwise_addition_ladder_20261025"


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def result(path: Path, arm: str, interface: str, seed: int):
    value = read(path)
    value["arm"] = arm
    value["interface"] = interface
    value["seed"] = seed
    value["source"] = str(path.relative_to(ROOT))
    return value


def main() -> int:
    results = []
    for arm in ("direct", "curriculum"):
        for interface in ("canonical", "factorized"):
            for seed in (42, 1337):
                original = OUTPUT / arm / interface / f"seed{seed}" / "single_digit_no_carry" / "RESULTS.json"
                repair = OUTPUT / "repair" / f"{arm}_{interface}_seed{seed}" / "RESULTS.json"
                path = original if original.is_file() else repair
                if not path.is_file():
                    raise SystemExit(f"missing result: {original} or {repair}")
                results.append(result(path, arm, interface, seed))
    stage1_pass = all(item["advancement_pass"] for item in results)
    summary = {
        "experiment": "crucible_digitwise_addition_ladder_20261025",
        "status": "completed",
        "preflight_pass": bool(read(OUTPUT / "PREFLIGHT.json")["all_checks_pass"]),
        "stage1_advancement_pass": stage1_pass,
        "stage2_ran": False,
        "test_evaluated": False,
        "repair_mode": True,
        "results": [
            {
                "arm": item["arm"],
                "interface": item["interface"],
                "stage": item["stage"],
                "seed": item["seed"],
                "steps": item["steps"],
                "train_exact": item["train_exact"],
                "validation_exact": item["validation_exact"],
                "fit_pass": item["fit_pass"],
                "advancement_pass": item["advancement_pass"],
                "elapsed_seconds": item["elapsed_seconds"],
                "source": item["source"],
            }
            for item in results
        ],
    }
    write_json(OUTPUT / "SUMMARY.json", summary)
    hashes = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(OUTPUT / "HASHES.json", hashes)
    print(json.dumps({"status": "completed", "arms": len(results), "stage2_ran": False, "stage1_advancement_pass": stage1_pass}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
