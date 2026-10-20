#!/usr/bin/env python3
"""Run and seal the isolated Crucible expert-count matrix.

This launcher does not alter any frozen source package.  Each variant is
executed in a separate process, writes to a temporary JSONL, is checked
against its exact manifest matrix, and is renamed into place only after the
check passes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "package_plan.json"
RUNNER_PATH = ROOT / "crucible_runner.py"
FULL_ROOT = ROOT / "full_run"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_metrics(
    metrics_path: Path,
    manifest: dict[str, Any],
    variant: dict[str, Any],
) -> tuple[int, str]:
    checkpoints = [
        int(x) for x in manifest["hyperparameters"]["checkpoint_intervals"]
    ]
    seeds = [int(x) for x in manifest["contract"]["seeds"]]
    branches = [str(x) for x in manifest["contract"]["architectural_branches"]]
    learning_rates = [
        float(x) for x in manifest["contract"]["learning_rate_candidates"]
    ]
    expected = {
        (seed, branch, lr, step)
        for seed in seeds
        for branch in branches
        for lr in learning_rates
        for step in checkpoints
    }

    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: row is not an object")
            rows.append(row)

    if len(rows) != len(expected):
        raise ValueError(
            f"expected {len(expected)} rows, found {len(rows)}"
        )

    seen: set[tuple[int, str, float, int]] = set()
    manifest_prefix = str(variant["manifest_sha256"])[:16].lower()
    for line_no, row in enumerate(rows, start=1):
        key = (
            int(row["seed"]),
            str(row["branch"]),
            float(row["peak_lr"]),
            int(row["interval_step_end"]),
        )
        if key not in expected:
            raise ValueError(f"line {line_no}: unexpected matrix key {key}")
        if key in seen:
            raise ValueError(f"line {line_no}: duplicate matrix key {key}")
        seen.add(key)

        if row.get("variant_id") != variant["id"]:
            raise ValueError(f"line {line_no}: variant binding mismatch")
        if str(row.get("master_config_sha256", "")).lower() != manifest_prefix:
            raise ValueError(f"line {line_no}: manifest binding mismatch")
        if int(row["num_experts"]) != int(variant["num_experts"]):
            raise ValueError(f"line {line_no}: expert-count mismatch")
        if int(row["top_k"]) != int(variant["top_k"]):
            raise ValueError(f"line {line_no}: top-k mismatch")
        if int(row["d_ff"]) != int(variant["d_ff"]):
            raise ValueError(f"line {line_no}: expert-width mismatch")

        for field in (
            "interval_train_ce",
            "validation_nll",
            "byte_perplexity",
            "router_interval_cv",
            "router_interval_entropy",
            "router_interval_load_ratio",
        ):
            if field not in row or not math.isfinite(float(row[field])):
                raise ValueError(f"line {line_no}: nonfinite/missing {field}")
        if int(row.get("interval_nan_inf_count", 0)) != 0:
            raise ValueError(f"line {line_no}: recorded nonfinite event")

    if seen != expected:
        raise ValueError("matrix does not cover the exact expected Cartesian product")
    return len(rows), sha256_file(metrics_path)


def verify_plan(plan: dict[str, Any]) -> None:
    if sha256_file(RUNNER_PATH) != plan["runner_sha256"]:
        raise ValueError("runner SHA-256 differs from the frozen package plan")
    if len(plan["variants"]) != 5:
        raise ValueError("unexpected variant count in package plan")
    for variant in plan["variants"]:
        manifest_path = ROOT / variant["id"] / "crucible_manifest" / "experiment_config.json"
        if sha256_file(manifest_path) != variant["manifest_sha256"]:
            raise ValueError(f"manifest hash mismatch: {variant['id']}")
        manifest = load_json(manifest_path)
        if manifest["meta"]["variant_id"] != variant["id"]:
            raise ValueError(f"variant metadata mismatch: {variant['id']}")
        if int(manifest["hyperparameters"]["num_experts"]) != int(variant["num_experts"]):
            raise ValueError(f"expert-count metadata mismatch: {variant['id']}")
        if int(manifest["hyperparameters"]["top_k"]) != int(variant["top_k"]):
            raise ValueError(f"top-k metadata mismatch: {variant['id']}")
        if int(manifest["hyperparameters"]["d_ff"]) != int(variant["d_ff"]):
            raise ValueError(f"width metadata mismatch: {variant['id']}")


def main() -> int:
    plan = load_json(PLAN_PATH)
    verify_plan(plan)
    FULL_ROOT.mkdir(parents=True, exist_ok=True)

    ledger = {
        "protocol": "crucible_expert_count_ab_full_v1",
        "started_utc": now(),
        "package_plan_sha256": sha256_file(PLAN_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "variants": [],
        "completed": False,
    }
    ledger_path = FULL_ROOT / "run_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for variant in plan["variants"]:
        variant_id = variant["id"]
        variant_root = ROOT / variant_id
        manifest_path = variant_root / "crucible_manifest" / "experiment_config.json"
        manifest = load_json(manifest_path)
        output_dir = FULL_ROOT / variant_id
        output_dir.mkdir(parents=True, exist_ok=True)
        final_metrics = output_dir / "metrics.jsonl"
        temp_metrics = output_dir / "metrics.jsonl.attempt"
        log_path = output_dir / "runner.log"
        metadata_path = output_dir / "run_metadata.json"

        if final_metrics.exists() and metadata_path.exists():
            rows, digest = validate_metrics(final_metrics, manifest, variant)
            prior = load_json(metadata_path)
            if prior.get("metrics_sha256") != digest or prior.get("rows") != rows:
                raise ValueError(f"sealed metadata mismatch: {variant_id}")
            ledger["variants"].append({"id": variant_id, "status": "already_complete"})
            continue

        if temp_metrics.exists():
            temp_metrics.unlink()

        command = [
            sys.executable,
            str(RUNNER_PATH),
            "--config-path",
            str(manifest_path),
            "--config-sha256",
            str(variant["manifest_sha256"]),
            "--output-path",
            str(temp_metrics),
        ]
        started = time.monotonic()
        started_utc = now()
        with log_path.open("w", encoding="utf-8") as log:
            log.write(json.dumps({"started_utc": started_utc, "command": command}, sort_keys=True) + "\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=str(variant_root),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "0"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            raise RuntimeError(
                f"{variant_id} failed with return code {result.returncode}; see {log_path}"
            )
        rows, digest = validate_metrics(temp_metrics, manifest, variant)
        temp_metrics.replace(final_metrics)
        metadata = {
            "variant_id": variant_id,
            "regime": variant["regime"],
            "manifest_sha256": variant["manifest_sha256"],
            "runner_sha256": sha256_file(RUNNER_PATH),
            "metrics_sha256": digest,
            "rows": rows,
            "elapsed_seconds": elapsed,
            "started_utc": started_utc,
            "finished_utc": now(),
            "returncode": result.returncode,
            "command": command,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        ledger["variants"].append(
            {"id": variant_id, "status": "complete", "rows": rows, "metrics_sha256": digest}
        )
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ledger["completed"] = True
    ledger["finished_utc"] = now()
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(ledger, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
