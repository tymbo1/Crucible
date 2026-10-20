#!/usr/bin/env python3
"""Prefix reproducibility and cross-variant hash diagnostic.

Runs a short frozen-runner prefix twice for the unchanged and grouped V2
variants.  It records terminal model, gradient and optimizer hashes per seed.
No frozen source, manifest, checkpoint or result is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_equivalence_gate import configure_strict_diagnostic_cuda, invoke_runner, sha256_file


VARIANTS = ("original", "grouped_rank_preserving")


def by_seed(runtime):
    return {int(item["seed"]): item for item in runtime}


def hash_equal(reference, candidate):
    keys = ("model_state_sha256", "gradient_sha256", "optimizer_state_sha256")
    return {
        key: reference[key] == candidate[key]
        for key in keys
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument(
        "--telemetry",
        choices=("disabled", "enabled"),
        default="disabled",
    )
    parser.add_argument("--warn-only", action="store_true")
    parser.add_argument("--repetitions", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()

    config_path = args.config_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_sha256 = sha256_file(config_path)
    if config_sha256 != args.config_sha256.strip().lower():
        raise ValueError("Config SHA-256 mismatch")

    configure_strict_diagnostic_cuda(warn_only=args.warn_only)

    runs = {}
    repetition_values = tuple(range(1, args.repetitions + 1))
    for variant in VARIANTS:
        runs[variant] = []
        for repetition in repetition_values:
            output_path = output_dir / variant / f"repeat_{repetition}" / "metrics.jsonl"
            print(f"running {variant} repeat {repetition} through step {args.max_steps}", flush=True)
            result = invoke_runner(
                config_path,
                config_sha256,
                output_path,
                args.max_steps,
                variant,
                disable_telemetry=args.telemetry == "disabled",
            )
            runs[variant].append({
                "repetition": repetition,
                "metrics_path": str(output_path.resolve()),
                "metrics_sha256": sha256_file(output_path),
                "runtime": result["runtime"],
            })

    reproducibility = {}
    if args.repetitions == 2:
        for variant in VARIANTS:
            first = by_seed(runs[variant][0]["runtime"])
            second = by_seed(runs[variant][1]["runtime"])
            reproducibility[variant] = {
                str(seed): hash_equal(first[seed], second[seed])
                for seed in sorted(first)
            }
    else:
        reproducibility = {variant: None for variant in VARIANTS}

    original = by_seed(runs["original"][0]["runtime"])
    grouped = by_seed(runs["grouped_rank_preserving"][0]["runtime"])
    cross_variant = {
        str(seed): hash_equal(original[seed], grouped[seed])
        for seed in sorted(original)
    }

    report = {
        "schema_version": "crucible-v2-prefix-reproducibility-20261018",
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "max_steps": args.max_steps,
        "variants": list(VARIANTS),
        "repetitions_per_variant": args.repetitions,
        "deterministic_settings": {
            "torch_use_deterministic_algorithms": True,
            "warn_only": args.warn_only,
            "flash_sdp": False,
            "mem_efficient_sdp": False,
            "math_sdp": True,
        },
        "telemetry": args.telemetry,
        "warn_only": args.warn_only,
        "runs": runs,
        "within_variant_reproducibility": reproducibility,
        "cross_variant_terminal_hashes": cross_variant,
        "interpretation": {
            "reproducible_original": (
                None if args.repetitions == 1 else all(
                    all(values.values()) for values in reproducibility["original"].values()
                )
            ),
            "reproducible_grouped": (
                None if args.repetitions == 1 else all(
                    all(values.values()) for values in reproducibility["grouped_rank_preserving"].values()
                )
            ),
            "cross_variant_equal_through_prefix": all(
                all(values.values()) for values in cross_variant.values()
            ),
        },
    }
    report_path = output_dir / "prefix_reproducibility_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    hashes = {
        "prefix_reproducibility_report.json": sha256_file(report_path),
        "run_prefix_reproducibility.py": sha256_file(Path(__file__).resolve()),
        "run_equivalence_gate.py": sha256_file(Path(__file__).with_name("run_equivalence_gate.py")),
    }
    for variant in VARIANTS:
        for repetition in repetition_values:
            key = f"{variant}/repeat_{repetition}/metrics.jsonl"
            hashes[key] = runs[variant][repetition - 1]["metrics_sha256"]
    hashes_path = output_dir / "prefix_reproducibility_hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    accepted = report["interpretation"]["cross_variant_equal_through_prefix"]
    if args.repetitions == 2:
        accepted = accepted and report["interpretation"]["reproducible_original"] and report["interpretation"]["reproducible_grouped"]
    print(json.dumps({"accepted": accepted, "report": str(report_path)}, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
