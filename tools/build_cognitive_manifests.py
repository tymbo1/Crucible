#!/usr/bin/env python3
"""Build immutable JSONL manifests for the CPU-only cognitive task suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from cognitive_task_suite import (
    FAMILIES,
    GENERATOR_VERSION,
    GENERATOR_VERSION_V2,
    SCHEMA_VERSION,
    SPLITS,
    canonical_json,
    generate_records,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, records: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--variant", choices=("v1", "v2"), default="v1")
    parser.add_argument("--train-per-family", type=int, default=64)
    parser.add_argument("--validation-per-family", type=int, default=32)
    parser.add_argument("--test-per-family", type=int, default=64)
    args = parser.parse_args()

    generator_version = GENERATOR_VERSION if args.variant == "v1" else GENERATOR_VERSION_V2

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    counts = {
        "train": args.train_per_family,
        "validation": args.validation_per_family,
        "test": args.test_per_family,
    }
    split_files: Dict[str, Dict[str, Any]] = {}
    family_counts: Dict[str, Dict[str, int]] = {family: {} for family in FAMILIES}

    for split in SPLITS:
        records = generate_records(args.seed, split, counts[split], variant=args.variant)
        path = args.output / f"{split}.jsonl"
        write_jsonl(path, records)
        for family in FAMILIES:
            family_counts[family][split] = sum(
                record["family"] == family for record in records
            )
        split_files[split] = {
            "path": path.name,
            "records": len(records),
            "sha256": sha256_file(path),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": generator_version,
        "seed": args.seed,
        "families": list(FAMILIES),
        "splits": split_files,
        "family_counts": family_counts,
        "task_contract": {
            "train_targets_present": True,
            "validation_targets_present": True,
            "test_targets_present": True,
            "test_composition_and_depth_holdouts": True,
            "oracle_policy": "exact deterministic oracle fields are reference-only and must not be supplied as model input",
        },
    }
    if args.variant == "v2":
        manifest["variant"] = "v2"
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "manifest.sha256").write_text(
        sha256_file(manifest_path) + "  manifest.json\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "manifest": manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

