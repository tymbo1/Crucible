#!/usr/bin/env python3
"""Independent read-only verifier for the context-gate package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


CONTEXTS = (128, 256, 512)
SEEDS = (42, 1337)
NUM_STEPS = 3125


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_lengths(path: Path, split: str) -> List[int]:
    lengths: List[int] = []
    seen: set[bytes] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != split:
                raise ValueError(f"split mismatch: {path}:{line_number}")
            encoded = str(record["text"]).encode("utf-8")
            if encoded in seen:
                raise ValueError(f"duplicate text: {path}:{line_number}")
            seen.add(encoded)
            if int(record["example_index"]) != len(lengths):
                raise ValueError(f"index mismatch: {path}:{line_number}")
            if int(record["byte_length"]) != len(encoded):
                raise ValueError(f"length mismatch: {path}:{line_number}")
            if record["text_utf8_sha256"] != hashlib.sha256(encoded).hexdigest():
                raise ValueError(f"text hash mismatch: {path}:{line_number}")
            lengths.append(len(encoded))
    return lengths


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def verify_variant(root: Path, context: int, expected_model_sha: str) -> Dict[str, Any]:
    variant = root / f"context_{context}"
    manifest_dir = variant / "crucible_manifest"
    config_path = manifest_dir / "experiment_config.json"
    recorded_manifest_sha = (manifest_dir / "manifest_sha256.txt").read_text(encoding="ascii").strip()
    actual_manifest_sha = sha256_file(config_path)
    if recorded_manifest_sha != actual_manifest_sha:
        raise ValueError(f"manifest hash mismatch: {variant}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["meta"].get("training_started"):
        raise ValueError(f"training-started flag set: {variant}")
    if config["meta"]["variant_id"] != f"context_{context}":
        raise ValueError(f"variant identity mismatch: {variant}")
    if config["hyperparameters"]["context_window_length"] != context:
        raise ValueError(f"context mismatch: {variant}")
    if config["hyperparameters"]["num_experts"] != 16 or config["hyperparameters"]["top_k"] != 2:
        raise ValueError(f"E16/top-2 mismatch: {variant}")
    if config["hyperparameters"]["expected_parameter_geometry"] != 170233600:
        raise ValueError(f"parameter geometry mismatch: {variant}")
    if config["code_provenance"]["model_core_sha256"] != expected_model_sha:
        raise ValueError(f"model code hash mismatch: {variant}")
    result: Dict[str, Any] = {
        "variant_id": f"context_{context}",
        "manifest_sha256": actual_manifest_sha,
        "training_started": False,
        "files": {},
        "splits": {},
        "schedules": {},
        "anchors": {},
    }
    for path in sorted(variant.rglob("*")):
        if path.is_file() and path.name != "verification.json":
            result["files"][str(path.relative_to(variant))] = sha256_file(path)

    split_lengths: Dict[str, List[int]] = {}
    for split in ("train", "validation"):
        info = config["splits"][split]
        source_path = variant / info["source_jsonl_relative_path"][3:]
        provenance_path = variant / info["provenance_relative_path"][3:]
        binary_path = manifest_dir / info["relative_path"]
        if sha256_file(source_path) != info["source_jsonl_sha256"]:
            raise ValueError(f"source JSONL hash mismatch: {source_path}")
        if sha256_file(provenance_path) != info["provenance_sha256"]:
            raise ValueError(f"provenance hash mismatch: {provenance_path}")
        lengths = read_jsonl_lengths(source_path, split)
        if len(lengths) != int(info["total_constructed_examples"]):
            raise ValueError(f"source count mismatch: {source_path}")
        if count_lines(provenance_path) != int(info["provenance_record_count"]):
            raise ValueError(f"provenance count mismatch: {provenance_path}")
        expected_binary_bytes = len(lengths) * 2 * context * 8
        if binary_path.stat().st_size != expected_binary_bytes:
            raise ValueError(f"binary size mismatch: {binary_path}")
        if sha256_file(binary_path) != info["binary_payload_sha256"]:
            raise ValueError(f"binary hash mismatch: {binary_path}")
        source_bytes = sum(lengths)
        if source_bytes != int(info["source_utf8_bytes"]):
            raise ValueError(f"source-byte total mismatch: {source_path}")
        if max(lengths) > context - 1:
            raise ValueError(f"context truncation risk: {source_path}")
        split_lengths[split] = lengths
        result["splits"][split] = {
            "examples": len(lengths),
            "source_bytes": source_bytes,
            "max_source_bytes": max(lengths),
            "nonpad_target_tokens": source_bytes + len(lengths),
            "binary_sha256": info["binary_payload_sha256"],
            "provenance_records": count_lines(provenance_path),
        }

    train_texts = set()
    train_source_path = variant / config["splits"]["train"]["source_jsonl_relative_path"][3:]
    with train_source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                train_texts.add(json.loads(line)["text"])
    val_source_path = variant / config["splits"]["validation"]["source_jsonl_relative_path"][3:]
    with val_source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and json.loads(line)["text"] in train_texts:
                raise ValueError(f"cross-split duplicate: {variant}")

    for seed in SEEDS:
        anchor_info = config["anchors"][str(seed)]
        anchor_path = manifest_dir / anchor_info["relative_path"]
        if sha256_file(anchor_path) != anchor_info["binary_payload_sha256"]:
            raise ValueError(f"anchor hash mismatch: {anchor_path}")
        result["anchors"][str(seed)] = anchor_info["binary_payload_sha256"]
        schedule_info = config["schedules"][str(seed)]
        schedule_path = manifest_dir / schedule_info["relative_path"]
        if sha256_file(schedule_path) != schedule_info["binary_payload_sha256"]:
            raise ValueError(f"schedule hash mismatch: {schedule_path}")
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        if schedule["schema"] != "crucible_dynamic_source_byte_schedule_v1":
            raise ValueError(f"schedule schema mismatch: {schedule_path}")
        if len(schedule["groups"]) != NUM_STEPS:
            raise ValueError(f"schedule step count mismatch: {schedule_path}")
        flattened: List[int] = []
        for step, group in enumerate(schedule["groups"], start=1):
            if group["step"] != step:
                raise ValueError(f"schedule step index mismatch: {schedule_path}")
            indices = [int(index) for index in group["indices"]]
            if not indices:
                raise ValueError(f"empty schedule group: {schedule_path}")
            actual_bytes = sum(split_lengths["train"][index] for index in indices)
            if actual_bytes != int(group["source_bytes"]):
                raise ValueError(f"schedule byte mismatch: {schedule_path}")
            if int(group["nonpad_target_tokens"]) != actual_bytes + len(indices):
                raise ValueError(f"schedule target mismatch: {schedule_path}")
            flattened.extend(indices)
        if sorted(flattened) != list(range(len(split_lengths["train"]))):
            raise ValueError(f"schedule is not a complete permutation: {schedule_path}")
        result["schedules"][str(seed)] = {
            "steps": len(schedule["groups"]),
            "indices": len(flattened),
            "source_bytes": sum(group["source_bytes"] for group in schedule["groups"]),
            "min_step_bytes": min(group["source_bytes"] for group in schedule["groups"]),
            "max_step_bytes": max(group["source_bytes"] for group in schedule["groups"]),
            "schedule_sha256": schedule_info["binary_payload_sha256"],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.package_root.resolve()
    model_sha = sha256_file(root / "shared" / "model_core.py")
    reports = [verify_variant(root, context, model_sha) for context in CONTEXTS]
    report = {
        "schema": "crucible_context_manifest_verification_v1",
        "status": "PASS",
        "package_root": str(root),
        "model_core_sha256": model_sha,
        "variants": reports,
    }
    output = root / "context_manifest_verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "report": str(output), "report_sha256": sha256_file(output), "variants": reports}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
