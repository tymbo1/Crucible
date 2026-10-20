#!/usr/bin/env python3
"""Manifest-bound disposable preflight for the expert-count package."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "crucible_runner.py"
PREVIEW_STEPS = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return math.isfinite(float(value))


def verify_variant(variant: dict[str, Any]) -> dict[str, Any]:
    variant_id = variant["id"]
    variant_dir = ROOT / variant_id
    config_path = variant_dir / "crucible_manifest" / "experiment_config.json"
    if sha256_file(config_path) != variant["manifest_sha256"]:
        raise AssertionError(f"manifest hash mismatch: {variant_id}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    hp = config["hyperparameters"]
    if hp["num_experts"] != variant["num_experts"]:
        raise AssertionError(f"expert-count mismatch: {variant_id}")
    if hp["top_k"] != variant["top_k"]:
        raise AssertionError(f"top-k mismatch: {variant_id}")
    if hp["d_ff"] != variant["d_ff"]:
        raise AssertionError(f"expert-width mismatch: {variant_id}")
    if sha256_file(RUNNER) != config["code_provenance"]["runner_sha256"]:
        raise AssertionError(f"runner provenance mismatch: {variant_id}")
    if sha256_file(ROOT / "model_core.py") != config["code_provenance"]["model_core_sha256"]:
        raise AssertionError(f"model provenance mismatch: {variant_id}")

    sys.path.insert(0, str(ROOT))
    from model_core import BraidPrototypeLM

    model = BraidPrototypeLM(
        hp["vocab_size"],
        hp["n_layers"],
        hp["d_model"],
        hp["n_heads"],
        hp["latent_dim_kv"],
        hp["latent_dim_q"],
        hp["pe_dim"],
        hp["num_experts"],
        hp["d_ff"],
        top_k=hp["top_k"],
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != hp["expected_parameter_geometry"]:
        raise AssertionError(
            f"parameter geometry mismatch for {variant_id}: "
            f"{parameter_count} != {hp['expected_parameter_geometry']}"
        )

    for seed in config["contract"]["seeds"]:
        info = config["anchors"][str(seed)]
        anchor_path = variant_dir / "crucible_manifest" / info["relative_path"]
        if sha256_file(anchor_path) != info["binary_payload_sha256"]:
            raise AssertionError(f"anchor hash mismatch: {variant_id}/{seed}")
        model.load_state_dict(
            torch.load(anchor_path, map_location="cpu", weights_only=True),
            strict=True,
        )

    output_dir = ROOT / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{variant_id}_metrics.jsonl"
    log_path = output_dir / f"{variant_id}.log"
    command = [
        sys.executable,
        str(RUNNER),
        "--config-path",
        str(config_path),
        "--config-sha256",
        variant["manifest_sha256"],
        "--output-path",
        str(output_path),
        "--max-steps",
        str(PREVIEW_STEPS),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=variant_dir,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"preflight failed for {variant_id}; see {log_path}")

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_keys = {
        (42, "B", 0.0003, PREVIEW_STEPS),
        (1337, "B", 0.0003, PREVIEW_STEPS),
    }
    actual_keys = {
        (
            int(row["seed"]),
            str(row["branch"]),
            float(row["peak_lr"]),
            int(row["interval_step_end"]),
        )
        for row in rows
    }
    if actual_keys != expected_keys:
        raise AssertionError(f"preflight matrix mismatch: {variant_id}")
    for row in rows:
        if row["variant_id"] != variant_id:
            raise AssertionError(f"row variant mismatch: {variant_id}")
        for field in (
            "validation_nll",
            "interval_train_ce",
            "router_interval_cv",
            "router_interval_entropy",
            "router_interval_load_ratio",
            "interval_branch_b_aux_loss",
        ):
            if not finite(row[field]):
                raise AssertionError(f"nonfinite {field}: {variant_id}")
        if int(row["interval_nan_inf_count"]) != 0:
            raise AssertionError(f"nonfinite event recorded: {variant_id}")

    return {
        "variant_id": variant_id,
        "parameter_count": parameter_count,
        "rows": len(rows),
        "metrics_sha256": sha256_file(output_path),
        "log_sha256": sha256_file(log_path),
        "preflight_steps": PREVIEW_STEPS,
        "full_training_started": False,
    }


def main() -> int:
    plan = json.loads((ROOT / "package_plan.json").read_text(encoding="utf-8"))
    results = [verify_variant(variant) for variant in plan["variants"]]
    report = {
        "protocol": "crucible_expert_count_preflight_v1",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "variants": results,
        "full_training_started": False,
        "runner_sha256": sha256_file(RUNNER),
        "model_core_sha256": sha256_file(ROOT / "model_core.py"),
    }
    output = ROOT / "preflight" / "preflight_report.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
