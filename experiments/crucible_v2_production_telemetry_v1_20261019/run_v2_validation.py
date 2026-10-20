#!/usr/bin/env python3
"""Run the frozen Crucible context runner with V2 grouped MoE in memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


CONTEXT_SHARED = Path("/home/timbushnell/crucible_v2_production_telemetry_v1_20261019/shared")
V1_DIR = Path("/home/timbushnell/gpu_neural_ssre_v1_grouped_expert_20261018")
V2_DIR = Path("/home/timbushnell/gpu_neural_ssre_v2_strategy_landscape_20261018")
PACKAGE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(CONTEXT_SHARED))
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(V2_DIR))

import context_runner  # noqa: E402
import strategy_landscape_benchmark as v2  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--metadata-path", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    original_constructor = context_runner.BraidPrototypeLM

    def patched_constructor(*constructor_args, **constructor_kwargs):
        model = original_constructor(*constructor_args, **constructor_kwargs)
        v2.patch_model(model, "grouped_rank_preserving")
        return model

    context_runner.BraidPrototypeLM = patched_constructor

    delegated = [
        "context_runner.py",
        "--config-path",
        str(args.config_path.resolve()),
        "--config-sha256",
        args.config_sha256.strip().lower(),
        "--output-path",
        str(args.output_path.resolve()),
    ]
    if args.max_steps is not None:
        delegated.extend(["--max-steps", str(args.max_steps)])

    saved_argv = sys.argv
    try:
        sys.argv = delegated
        return_code = context_runner.main()
    finally:
        sys.argv = saved_argv
        context_runner.BraidPrototypeLM = original_constructor

    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.config_path.resolve()
    metrics_path = args.output_path.resolve()
    metadata = {
        "schema_version": "crucible-v2-production-telemetry-sustained-validation-20261019",
        "execution_variant": "grouped_rank_preserving",
        "execution_scope": "all MoE layers",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "context_runner_sha256": sha256_file(CONTEXT_SHARED / "context_runner.py"),
        "model_core_sha256": sha256_file(CONTEXT_SHARED / "model_core.py"),
        "v1_grouped_source_sha256": sha256_file(V1_DIR / "grouped_expert_benchmark.py"),
        "v2_source_sha256": sha256_file(V2_DIR / "strategy_landscape_benchmark.py"),
        "measurement_protocol_sha256": sha256_file(PACKAGE_DIR / "MEASUREMENT_PROTOCOL.md"),
        "readme_sha256": sha256_file(PACKAGE_DIR / "README.md"),
        "max_steps": args.max_steps,
        "note": (
            "The frozen runner performed all scientific checks; this wrapper "
            "only patched model construction in memory."
        ),
    }
    args.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
