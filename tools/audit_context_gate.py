#!/usr/bin/env python3
"""Audit the collected context-gate metrics against frozen manifests.

This is read-only with respect to source artifacts.  It emits a compact JSON
report that records observed rows, missing seeds/checkpoints and file hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List


CONTEXTS = (128, 256, 512)
REQUIRED_FIELDS = {
    "seed",
    "variant_id",
    "context_window_length",
    "interval_step_end",
    "validation_raw_byte_nll",
    "validation_bits_per_raw_byte",
    "master_config_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def status_hashes(path: Path) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    pattern = re.compile(
        r"^\|\s*(128|256|512)\s*\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*`?([0-9a-f]{64})`?\s*\|$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            context, filename, rows, seeds, status, digest = match.groups()
            result[int(context)] = {
                "filename": filename,
                "rows": int(rows),
                "seeds": seeds.strip(),
                "status": status.strip(),
                "sha256": digest,
            }
    return result


def audit_context(context: int, repo_root: Path, cloud_root: Path) -> Dict[str, Any]:
    config_path = repo_root / "experiments" / "crucible_context_gate_v1_20261018" / f"context_{context}" / "crucible_manifest" / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_seeds = sorted(config["contract"]["seeds"])
    expected_steps = list(config["contract"]["checkpoint_intervals"])
    expected_config_hash = sha256_file(config_path)
    metric_name = "metrics.jsonl" if context == 128 else "metrics_seed42.jsonl"
    metrics_path = cloud_root / f"context_{context}" / metric_name

    result: Dict[str, Any] = {
        "context": context,
        "config_path": str(config_path),
        "config_sha256": expected_config_hash,
        "expected_seeds": expected_seeds,
        "expected_checkpoint_steps": expected_steps,
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path) if metrics_path.exists() else None,
        "observed": {},
        "missing": {},
        "errors": [],
    }
    if not metrics_path.exists():
        result["errors"].append("metrics file missing")
        return result

    rows = load_jsonl(metrics_path)
    seen = set()
    for row in rows:
        missing_fields = sorted(REQUIRED_FIELDS - row.keys())
        if missing_fields:
            result["errors"].append({"row": len(seen), "missing_fields": missing_fields})
            continue
        key = (int(row["seed"]), int(row["interval_step_end"]))
        if key in seen:
            result["errors"].append({"duplicate": list(key)})
        seen.add(key)
        if row["context_window_length"] != context:
            result["errors"].append({"context_mismatch": row["context_window_length"]})
        if row["variant_id"] != f"context_{context}":
            result["errors"].append({"variant_mismatch": row["variant_id"]})
        if row["master_config_sha256"] != expected_config_hash:
            result["errors"].append({"config_hash_mismatch": row["master_config_sha256"]})
        seed = str(row["seed"])
        result["observed"].setdefault(seed, []).append(int(row["interval_step_end"]))

    for seed in expected_seeds:
        observed_steps = sorted(result["observed"].get(str(seed), []))
        result["observed"][str(seed)] = observed_steps
        result["missing"][str(seed)] = sorted(set(expected_steps) - set(observed_steps))
    result["row_count"] = len(rows)
    result["complete"] = (
        not result["errors"]
        and all(not result["missing"][str(seed)] for seed in expected_seeds)
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cloud-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "schema_version": "crucible_context_gate_audit_v1",
        "repo_root": str(args.repo_root),
        "cloud_root": str(args.cloud_root),
        "status_file": str(args.status),
        "status_hashes": status_hashes(args.status),
        "contexts": {
            str(context): audit_context(context, args.repo_root, args.cloud_root)
            for context in CONTEXTS
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    # An incomplete collection is an expected audit finding, not a parser
    # failure.  Return nonzero only when the collected rows are malformed or
    # violate their frozen manifest.
    return 0 if all(not item["errors"] for item in report["contexts"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

