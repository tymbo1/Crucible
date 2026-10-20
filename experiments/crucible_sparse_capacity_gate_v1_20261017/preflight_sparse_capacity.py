#!/usr/bin/env python3
"""Disposable 100-step CUDA preflight for the E4/E8/E16 package."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "package_plan.json"
RUNNER = ROOT / "crucible_runner.py"
OUT = ROOT / "preflight"
STEPS = 100


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_rows(
    path: Path,
    variant: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[int, str]:
    rows = [read for read in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()) if read]
    if len(rows) != 2:
        raise ValueError(f"{variant['id']}: expected two seed rows, found {len(rows)}")
    seen = set()
    prefix = variant["manifest_sha256"][:16].lower()
    for row in rows:
        key = (int(row["seed"]), str(row["branch"]), float(row["peak_lr"]), int(row["interval_step_end"]))
        if key in seen:
            raise ValueError(f"{variant['id']}: duplicate row {key}")
        seen.add(key)
        if int(row["interval_step_end"]) != STEPS:
            raise ValueError(f"{variant['id']}: unexpected step")
        if row.get("variant_id") != variant["id"]:
            raise ValueError(f"{variant['id']}: variant binding mismatch")
        if str(row.get("master_config_sha256", "")).lower() != prefix:
            raise ValueError(f"{variant['id']}: manifest binding mismatch")
        if int(row["num_experts"]) != int(variant["num_experts"]):
            raise ValueError(f"{variant['id']}: expert-count mismatch")
        if int(row["top_k"]) != int(variant["top_k"]):
            raise ValueError(f"{variant['id']}: top-k mismatch")
        if int(row["d_ff"]) != int(variant["d_ff"]):
            raise ValueError(f"{variant['id']}: width mismatch")
        for field in (
            "interval_train_ce",
            "validation_nll",
            "byte_perplexity",
            "router_interval_cv",
            "router_interval_entropy",
            "router_interval_load_ratio",
        ):
            if field not in row or not math.isfinite(float(row[field])):
                raise ValueError(f"{variant['id']}: nonfinite/missing {field}")
        if int(row.get("interval_nan_inf_count", 0)) != 0:
            raise ValueError(f"{variant['id']}: recorded nonfinite event")
        layers = row.get("layer_telemetry")
        if not isinstance(layers, dict):
            raise ValueError(f"{variant['id']}: missing layer telemetry")
        expected_layers = int(manifest["hyperparameters"]["n_layers"])
        for layer_index in range(expected_layers):
            payload = layers.get(f"layer_{layer_index}_moe")
            if not isinstance(payload, dict):
                raise ValueError(f"{variant['id']}: missing layer {layer_index} telemetry")
            affinity = payload.get("_token_affinity")
            selected_tokens = payload.get("selected_token_counts")
            top1 = payload.get("top1_counts")
            top2 = payload.get("top2_counts")
            activation = payload.get("mean_expert_output_norm")
            confidence = payload.get("mean_selected_probability")
            expert_count = int(variant["num_experts"])
            if not isinstance(affinity, list) or len(affinity) != expert_count:
                raise ValueError(f"{variant['id']}: invalid affinity expert dimension")
            if any(not isinstance(row_values, list) or len(row_values) != 259 for row_values in affinity):
                raise ValueError(f"{variant['id']}: invalid affinity vocabulary dimension")
            if not isinstance(selected_tokens, list) or len(selected_tokens) != expert_count:
                raise ValueError(f"{variant['id']}: invalid selected-token matrix")
            if any(not isinstance(row_values, list) or len(row_values) != 259 for row_values in selected_tokens):
                raise ValueError(f"{variant['id']}: invalid selected-token vocabulary dimension")
            for vector_name, vector in (
                ("top1_counts", top1),
                ("top2_counts", top2),
                ("mean_expert_output_norm", activation),
                ("mean_selected_probability", confidence),
            ):
                if not isinstance(vector, list) or len(vector) != expert_count:
                    raise ValueError(f"{variant['id']}: invalid {vector_name}")
                if any(not math.isfinite(float(value)) for value in vector):
                    raise ValueError(f"{variant['id']}: nonfinite {vector_name}")
            if any(
                not math.isfinite(float(value)) or float(value) < 0
                for row_values in affinity
                for value in row_values
            ):
                raise ValueError(f"{variant['id']}: invalid affinity values")
            tokens_routed = float(payload["tokens_routed"])
            top_k = int(payload["top_k"])
            affinity_mass = float(payload["affinity_mass_total"])
            if not math.isclose(affinity_mass, tokens_routed, rel_tol=1e-6, abs_tol=1e-5):
                raise ValueError(f"{variant['id']}: affinity mass conservation failure")
            if not math.isclose(sum(float(x) for x in top1), tokens_routed, rel_tol=1e-6, abs_tol=1e-5):
                raise ValueError(f"{variant['id']}: top-1 count invariant failure")
            if not math.isclose(sum(float(x) for x in top2), tokens_routed * top_k, rel_tol=1e-6, abs_tol=1e-5):
                raise ValueError(f"{variant['id']}: top-k count invariant failure")
            if not math.isclose(
                sum(float(value) for row_values in selected_tokens for value in row_values),
                tokens_routed * top_k,
                rel_tol=1e-6,
                abs_tol=1e-5,
            ):
                raise ValueError(f"{variant['id']}: selected-token invariant failure")
    return len(rows), sha256(path)


def main() -> int:
    plan = read_json(PLAN)
    if plan.get("training_started") is not False:
        raise ValueError("package plan is not marked pre-training")
    if sha256(RUNNER) != plan["runner_sha256"]:
        raise ValueError("runner hash mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "crucible_sparse_capacity_preflight_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "full_training_started": False,
        "package_plan_sha256": sha256(PLAN),
        "runner_sha256": sha256(RUNNER),
        "variants": [],
    }
    for variant in plan["variants"]:
        variant_dir = ROOT / variant["id"]
        config = variant_dir / "crucible_manifest" / "experiment_config.json"
        if sha256(config) != variant["manifest_sha256"]:
            raise ValueError(f"{variant['id']}: manifest hash mismatch")
        manifest = read_json(config)
        if manifest["meta"].get("training_started") is not False:
            raise ValueError(f"{variant['id']}: manifest is not pre-training")
        output = OUT / f"{variant['id']}_metrics.jsonl"
        log = OUT / f"{variant['id']}.log"
        for path in (output, log):
            if path.exists():
                path.unlink()
        command = [
            sys.executable,
            str(RUNNER),
            "--config-path", str(config),
            "--config-sha256", variant["manifest_sha256"],
            "--output-path", str(output),
            "--max-steps", str(STEPS),
        ]
        with log.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=str(variant_dir),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "0"},
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"{variant['id']} preflight failed; see {log}")
        rows, digest = validate_rows(output, variant, manifest)
        report["variants"].append(
            {
                "id": variant["id"],
                "rows": rows,
                "metrics_sha256": digest,
                "log_sha256": sha256(log),
                "parameter_count": variant["parameter_count"],
                "preflight_steps": STEPS,
                "full_training_started": False,
            }
        )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report_path = OUT / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
