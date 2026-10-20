#!/usr/bin/env python3
"""Compile the frozen E16 context-length comparison package.

This compiler creates new, self-contained context variants from the already
audited packed corpora.  It does not modify any prior Crucible artifact and
does not launch training.  Each variant uses a deterministic source-byte
balanced schedule with exactly 3,125 optimizer updates and a full training
example permutation per seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from array import array
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SOURCE_E16 = Path("/home/timbushnell/crucible_sparse_capacity_gate_v1_20261017")
SOURCE_128 = Path("/home/timbushnell/crucible_large_tinystories_v2_packed_v1_20261015")
STAGING = Path("/home/timbushnell/crucible_context_gate_v1_20261017/staging_raw_budget")

CONTEXTS = (128, 256, 512)
SEEDS = (42, 1337)
NUM_STEPS = 3125
LR = 3e-4
MICROBATCH_BY_CONTEXT = {128: 4, 256: 2, 512: 1}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_json(path: Path, value: Any) -> bytes:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    atomic_write(path, payload)
    return payload


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, split: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: set[bytes] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != split:
                raise ValueError(f"split mismatch at {path}:{line_number}")
            text = str(record["text"])
            encoded = text.encode("utf-8")
            if not encoded:
                raise ValueError(f"empty text at {path}:{line_number}")
            if encoded in seen:
                raise ValueError(f"duplicate text at {path}:{line_number}")
            seen.add(encoded)
            if int(record["byte_length"]) != len(encoded):
                raise ValueError(f"byte length mismatch at {path}:{line_number}")
            if record["text_utf8_sha256"] != sha256_bytes(encoded):
                raise ValueError(f"text hash mismatch at {path}:{line_number}")
            records.append(record)
    for index, record in enumerate(records):
        if int(record["example_index"]) != index:
            raise ValueError(f"non-deterministic local id at {path}:{index + 1}")
    return records


def build_binary(records: Sequence[Dict[str, Any]], seq_len: int) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("Crucible manifests require a little-endian host")
    values = array("q")
    for record in records:
        raw = str(record["text"]).encode("utf-8")
        if len(raw) > seq_len - 1:
            raise ValueError("compiler would truncate a context record")
        encoded = [byte + 3 for byte in raw]
        full = [1] + encoded + [2]
        x = full[:-1] + [0] * (seq_len - len(full[:-1]))
        y = full[1:] + [0] * (seq_len - len(full[1:]))
        if len(x) != seq_len or len(y) != seq_len:
            raise AssertionError("invalid binary tensor geometry")
        values.extend(x)
        values.extend(y)
    return values.tobytes()


def make_schedule(
    records: Sequence[Dict[str, Any]],
    seed: int,
    context: int,
) -> Dict[str, Any]:
    """Make a deterministic permutation cut into nearly equal byte groups."""

    if len(records) < NUM_STEPS:
        raise ValueError(f"not enough examples for {NUM_STEPS} steps")
    lengths = [int(record["byte_length"]) for record in records]
    permutation = list(range(len(records)))
    random.Random(seed).shuffle(permutation)
    shuffled_lengths = [lengths[index] for index in permutation]
    cumulative: List[int] = []
    total = 0
    for length in shuffled_lengths:
        total += length
        cumulative.append(total)

    groups: List[Dict[str, Any]] = []
    start = 0
    boundaries: List[int] = []
    for step in range(1, NUM_STEPS):
        desired = (total * step) / NUM_STEPS
        candidate = bisect_left(cumulative, desired) + 1
        candidate = max(candidate, start + 1)
        candidate = min(candidate, len(permutation) - (NUM_STEPS - step))
        boundaries.append(candidate)
        start = candidate
    boundaries.append(len(permutation))

    start = 0
    microbatch_size = MICROBATCH_BY_CONTEXT[context]
    for step, end in enumerate(boundaries, start=1):
        indices = permutation[start:end]
        source_bytes = sum(lengths[index] for index in indices)
        nonpad_targets = source_bytes + len(indices)
        groups.append(
            {
                "step": step,
                "indices": indices,
                "example_count": len(indices),
                "source_bytes": source_bytes,
                "raw_byte_target_tokens": source_bytes,
                "nonpad_target_tokens": nonpad_targets,
                "microbatch_size": microbatch_size,
                "microbatch_count": (len(indices) + microbatch_size - 1) // microbatch_size,
            }
        )
        start = end

    flattened = [index for group in groups for index in group["indices"]]
    if flattened != permutation:
        raise AssertionError("schedule groups are not the shuffled permutation")
    if len(set(flattened)) != len(records):
        raise AssertionError("schedule is not a permutation")
    group_bytes = [int(group["source_bytes"]) for group in groups]
    return {
        "schema": "crucible_dynamic_source_byte_schedule_v1",
        "seed": seed,
        "context_window_length": context,
        "num_steps": NUM_STEPS,
        "total_examples": len(records),
        "total_source_bytes": total,
        "target_source_bytes_per_update": total / NUM_STEPS,
        "microbatch_size": microbatch_size,
        "group_source_bytes_min": min(group_bytes),
        "group_source_bytes_mean": sum(group_bytes) / len(group_bytes),
        "group_source_bytes_max": max(group_bytes),
        "groups": groups,
    }


def copy_checked(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = sha256_file(destination)
    if digest != sha256_file(source):
        raise AssertionError(f"copy hash mismatch: {source}")
    return digest


def source_paths(context: int) -> Tuple[Path, Path, Path | None, Path | None, Path]:
    if context == 128:
        root = SOURCE_128
        return (
            root / "build" / "packed_train.jsonl",
            root / "build" / "packed_validation.jsonl",
            None,
            None,
            root / "packed_source_metadata.json",
        )
    root = STAGING / f"context_{context}"
    return (
        root / "build" / "packed_train.jsonl",
        root / "build" / "packed_validation.jsonl",
        root / "build" / "provenance_train.jsonl",
        root / "build" / "provenance_validation.jsonl",
        root / "source_metadata.json",
    )


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_legacy_provenance_sidecars(
    destination_train: Path,
    destination_validation: Path,
    train: Sequence[Dict[str, Any]],
    validation: Sequence[Dict[str, Any]],
) -> None:
    """Extract the old 128-byte embedded provenance into sidecars.

    The original accepted 128-byte manifest predates the sidecar contract and
    stores provenance inside its JSON manifest.  This conversion preserves
    that data while keeping the new context package compact and explicit.
    """

    legacy_path = SOURCE_128 / "crucible_manifest" / "experiment_config.json"
    legacy = read_json(legacy_path)
    payloads: List[Tuple[Path, str, Sequence[Dict[str, Any]]]] = [
        (destination_train, "train", train),
        (destination_validation, "validation", validation),
    ]
    for destination, split, records in payloads:
        entries = legacy["splits"][split]["provenance"]
        if len(entries) != len(records):
            raise ValueError(f"legacy provenance count mismatch for {split}")
        lines: List[str] = []
        for local_id, (record, entry) in enumerate(zip(records, entries)):
            if int(entry.get("local_split_id", local_id)) != local_id:
                raise ValueError(f"legacy provenance index mismatch for {split}")
            converted = {
                "schema_version": "crucible_context_provenance_v1",
                "local_split_id": local_id,
                "text_utf8_sha256": record["text_utf8_sha256"],
                "source_file": entry.get("source_file", record.get("source_file")),
                "source_record_index": entry["source_record_index"],
                "source_start_byte": entry["source_start_byte"],
                "source_end_byte": entry["source_end_byte"],
                "source_story_sha256": entry["source_story_sha256"],
                "component_unit_ids": entry["component_unit_ids"],
                "component_count": entry["component_count"],
                "components": entry["components"],
                "is_story_final_chunk": entry["is_story_final_chunk"],
                "tail_merge_applied": entry["tail_merge_applied"],
                "unavoidable_final_tail": entry["unavoidable_final_tail"],
                "packing_strategy": entry["packing_strategy"],
            }
            lines.append(json.dumps(converted, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        atomic_write(destination, "".join(lines).encode("utf-8"))


def compile_variant(context: int, source_manifest: Dict[str, Any], source_manifest_sha: str) -> Dict[str, Any]:
    variant_id = f"context_{context}"
    variant_root = ROOT / variant_id
    if variant_root.exists():
        existing_config_path = variant_root / "crucible_manifest" / "experiment_config.json"
        existing_manifest_hash_path = variant_root / "crucible_manifest" / "manifest_sha256.txt"
        if existing_config_path.is_file() and existing_manifest_hash_path.is_file():
            existing_config = read_json(existing_config_path)
            existing_hash = sha256_file(existing_config_path)
            recorded_hash = existing_manifest_hash_path.read_text(encoding="ascii").strip()
            if recorded_hash != existing_hash:
                raise ValueError(f"existing manifest hash mismatch: {variant_root}")
            if existing_config.get("meta", {}).get("variant_id") != variant_id:
                raise ValueError(f"existing variant identity mismatch: {variant_root}")
            if existing_config.get("meta", {}).get("training_started"):
                raise ValueError(f"existing variant is marked training-started: {variant_root}")
            if existing_config.get("code_provenance", {}).get("runner_sha256") != sha256_file(ROOT / "shared" / "context_runner.py"):
                raise ValueError(f"existing runner hash mismatch: {variant_root}")
            return {
                "variant_id": variant_id,
                "context": context,
                "manifest_sha256": existing_hash,
                "train_examples": existing_config["meta"]["train_examples"],
                "validation_examples": existing_config["meta"]["validation_examples"],
                "train_source_bytes": existing_config["meta"]["train_source_utf8_bytes"],
                "validation_source_bytes": existing_config["meta"]["validation_source_utf8_bytes"],
                "manifest_dir": str(variant_root / "crucible_manifest"),
                "schedule_summaries": existing_config.get("schedule_summaries", {}),
                "reused_existing": True,
            }
        existing_files = sorted(path.relative_to(variant_root) for path in variant_root.rglob("*") if path.is_file())
        allowed_partial = context == 128 and existing_files and all(
            str(path) in {"source/packed_train.jsonl", "source/packed_validation.jsonl"}
            for path in existing_files
        )
        if not allowed_partial:
            raise FileExistsError(f"refusing to overwrite {variant_root}")
    manifest_dir = variant_root / "crucible_manifest"
    source_dir = variant_root / "source"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    train_path, val_path, provenance_train, provenance_val, metadata_path = source_paths(context)
    train = read_jsonl(train_path, "train")
    validation = read_jsonl(val_path, "validation")
    if {str(item["text"]) for item in train} & {str(item["text"]) for item in validation}:
        raise ValueError(f"cross-split duplicate in {variant_id}")
    if max(int(item["byte_length"]) for item in train + validation) > context - 1:
        raise ValueError(f"oversized record in {variant_id}")

    source_train_copy = source_dir / "packed_train.jsonl"
    source_val_copy = source_dir / "packed_validation.jsonl"
    source_prov_train_copy = source_dir / "provenance_train.jsonl"
    source_prov_val_copy = source_dir / "provenance_validation.jsonl"
    source_metadata_copy = source_dir / "source_metadata.json"
    for source, destination in (
        (train_path, source_train_copy),
        (val_path, source_val_copy),
        (metadata_path, source_metadata_copy),
    ):
        if not destination.exists():
            copy_checked(source, destination)
        elif sha256_file(destination) != sha256_file(source):
            raise ValueError(f"existing partial source hash mismatch: {destination}")
    if provenance_train is not None and provenance_val is not None:
        for source, destination in (
            (provenance_train, source_prov_train_copy),
            (provenance_val, source_prov_val_copy),
        ):
            copy_checked(source, destination)
    else:
        write_legacy_provenance_sidecars(
            source_prov_train_copy,
            source_prov_val_copy,
            train,
            validation,
        )

    train_binary = build_binary(train, context)
    val_binary = build_binary(validation, context)
    atomic_write(manifest_dir / "stories_train.bin", train_binary)
    atomic_write(manifest_dir / "stories_validation.bin", val_binary)

    e16_manifest_dir = SOURCE_E16 / "matched_active_e16" / "crucible_manifest"
    anchor_info: Dict[str, Any] = {}
    for seed in SEEDS:
        name = f"initial_seed_anchor_{seed}.pt"
        source_anchor = e16_manifest_dir / name
        destination_anchor = manifest_dir / name
        digest = copy_checked(source_anchor, destination_anchor)
        expected = source_manifest["anchors"][str(seed)]["binary_payload_sha256"]
        if digest != expected:
            raise ValueError(f"E16 anchor hash mismatch for seed {seed}")
        anchor_info[str(seed)] = {
            "relative_path": name,
            "binary_payload_sha256": digest,
        }

    schedule_info: Dict[str, Any] = {}
    schedule_summaries: Dict[str, Any] = {}
    for seed in SEEDS:
        schedule = make_schedule(train, seed, context)
        schedule_path = manifest_dir / f"schedule_seed_{seed}.json"
        schedule_bytes = write_json(schedule_path, schedule)
        schedule_info[str(seed)] = {
            "relative_path": schedule_path.name,
            "binary_payload_sha256": sha256_bytes(schedule_bytes),
            "schema": schedule["schema"],
            "num_steps": NUM_STEPS,
            "num_indices": len(train),
            "is_single_train_permutation": True,
            "target_source_bytes_per_update": schedule["target_source_bytes_per_update"],
        }
        schedule_summaries[str(seed)] = {
            "min_source_bytes": schedule["group_source_bytes_min"],
            "mean_source_bytes": schedule["group_source_bytes_mean"],
            "max_source_bytes": schedule["group_source_bytes_max"],
            "min_examples": min(group["example_count"] for group in schedule["groups"]),
            "mean_examples": sum(group["example_count"] for group in schedule["groups"]) / NUM_STEPS,
            "max_examples": max(group["example_count"] for group in schedule["groups"]),
        }

    metadata = read_json(metadata_path)
    model_sha = sha256_file(ROOT / "shared" / "model_core.py")
    runner_sha = sha256_file(ROOT / "shared" / "context_runner.py")
    compiler_sha = sha256_file(Path(__file__).resolve())
    stats_path = (
        SOURCE_128 / "packed_dataset_statistics.json"
        if context == 128
        else STAGING / f"context_{context}" / "context_dataset_statistics.json"
    )
    stats_sha = sha256_file(stats_path)
    train_bytes = sum(int(item["byte_length"]) for item in train)
    val_bytes = sum(int(item["byte_length"]) for item in validation)
    context_hp = dict(source_manifest["hyperparameters"])
    context_hp.update(
        {
            "context_window_length": context,
            "max_source_utf8_bytes": context - 1,
            "num_experts": 16,
            "top_k": 2,
            "d_ff": 3447,
            "expected_parameter_geometry": 170233600,
            "num_steps": NUM_STEPS,
            "warmup_steps": 313,
            "gradient_accumulation_steps": None,
            "microbatch_size": MICROBATCH_BY_CONTEXT[context],
            "batching_mode": "dynamic_source_byte_groups_v1",
            "target_source_bytes_per_update": train_bytes / NUM_STEPS,
            "reference_source_bytes": 9856480,
            "reference_validation_source_bytes": 493539,
            "raw_byte_primary_metric": "validation_raw_byte_nll_eos_excluded",
        }
    )
    config: Dict[str, Any] = {
        "meta": {
            "dataset_name": f"Crucible E16 context gate {context} byte context",
            "experiment_family": "crucible_context_gate_v1",
            "variant_id": variant_id,
            "training_started": False,
            "official_split_preserved": True,
            "source_language": metadata.get("language", "English"),
            "source_license": metadata.get("license", "cdla-sharing-1.0"),
            "source_upstream_repository": metadata.get("upstream_repository", "roneneldan/TinyStories"),
            "source_upstream_revision": metadata.get("upstream_revision", ""),
            "source_files": metadata.get("source_files", []),
            "source_corpus_sha256": sha256_bytes(
                "\n".join(entry["sha256"] for entry in metadata.get("source_files", [])).encode("ascii")
            ),
            "reference_e16_manifest_sha256": source_manifest_sha,
            "reference_e16_model_core_sha256": source_manifest["code_provenance"]["model_core_sha256"],
            "context_dataset_statistics_relative_path": f"../source/context_dataset_statistics.json",
            "context_dataset_statistics_sha256": stats_sha,
            "endianness": sys.byteorder,
            "tensor_data_type": "int64",
            "tokenizer_specification": "LosslessByteTransformerStandard",
            "cross_split_exact_duplicates": 0,
            "truncation_count": 0,
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_source_utf8_bytes": train_bytes,
            "validation_source_utf8_bytes": val_bytes,
            "raw_byte_budget_policy": {
                "train_target_source_bytes": 9856480,
                "validation_reference_source_bytes": 493539,
                "train_actual_source_bytes": train_bytes,
                "validation_actual_source_bytes": val_bytes,
                "optimizer_updates": NUM_STEPS,
                "eos_excluded_primary_metric": True,
            },
        },
        "code_provenance": {
            "model_core_sha256": model_sha,
            "runner_sha256": runner_sha,
            "manifest_compiler_sha256": compiler_sha,
            "source_preprocessor_sha256": (
                sha256_file(
                    SOURCE_128 / "build" / "prepare_packed_corpus.py"
                    if context == 128
                    else STAGING.parent / "build" / "prepare_context_corpus.py"
                )
            ),
            "source_verifier_sha256": (
                sha256_file(
                    SOURCE_128 / "build" / "verify_packed_manifest.py"
                    if context == 128
                    else STAGING.parent / "build" / "verify_context_staging.py"
                )
            ),
        },
        "experiment_variant": {
            "id": variant_id,
            "expert_count": 16,
            "top_k": 2,
            "expert_hidden_width": 3447,
            "regime": "fixed_active_compute",
            "context_length": context,
            "active_compute_reference": "matched_active_e16",
        },
        "contract": {
            "architectural_branches": ["B"],
            "learning_rate_candidates": [LR],
            "seeds": list(SEEDS),
            "num_optimizer_updates": NUM_STEPS,
            "checkpoint_intervals": [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3125],
            "selection_note": (
                "E16 context gate; matched raw UTF-8 source bytes, dynamic byte-balanced "
                "step groups, EOS-excluded raw-byte NLL/BPB primary endpoint"
            ),
        },
        "hyperparameters": context_hp,
        "splits": {},
        "schedules": schedule_info,
        "anchors": anchor_info,
        "schedule_summaries": schedule_summaries,
    }
    for split_name, records, binary, source_jsonl, provenance in (
        ("train", train, train_binary, source_train_copy, source_prov_train_copy),
        ("validation", validation, val_binary, source_val_copy, source_prov_val_copy),
    ):
        provenance_count = count_jsonl(provenance)
        if provenance_count != len(records):
            raise ValueError(f"provenance count mismatch in {variant_id}/{split_name}")
        config["splits"][split_name] = {
            "relative_path": f"stories_{split_name}.bin",
            "binary_payload_sha256": sha256_bytes(binary),
            "total_constructed_examples": len(records),
            "max_source_utf8_bytes": context - 1,
            "truncation_count": 0,
            "source_utf8_bytes": sum(int(item["byte_length"]) for item in records),
            "nonpad_target_tokens": sum(int(item["byte_length"]) + 1 for item in records),
            "source_jsonl_relative_path": f"../source/{source_jsonl.name}",
            "source_jsonl_sha256": sha256_file(source_jsonl),
            "provenance_relative_path": f"../source/{provenance.name}",
            "provenance_sha256": sha256_file(provenance),
            "provenance_record_count": provenance_count,
            "provenance_schema": "crucible_context_provenance_v1",
        }

    config_path = manifest_dir / "experiment_config.json"
    config_bytes = write_json(config_path, config)
    manifest_sha = sha256_bytes(config_bytes)
    atomic_write(manifest_dir / "manifest_sha256.txt", (manifest_sha + "\n").encode("ascii"))
    return {
        "variant_id": variant_id,
        "context": context,
        "manifest_sha256": manifest_sha,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_source_bytes": train_bytes,
        "validation_source_bytes": val_bytes,
        "manifest_dir": str(manifest_dir),
        "schedule_summaries": schedule_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-e16-manifest", type=Path, required=True)
    args = parser.parse_args()
    source_manifest_path = args.source_e16_manifest.resolve()
    source_manifest = read_json(source_manifest_path)
    source_manifest_sha = sha256_file(source_manifest_path)
    if source_manifest["hyperparameters"]["num_experts"] != 16:
        raise ValueError("reference manifest is not E16")
    if source_manifest["hyperparameters"]["expected_parameter_geometry"] != 170233600:
        raise ValueError("reference E16 geometry mismatch")
    if source_manifest["meta"].get("training_started"):
        raise ValueError("reference manifest is marked as training-started")
    reports = [compile_variant(context, source_manifest, source_manifest_sha) for context in CONTEXTS]
    plan = {
        "protocol": "crucible_context_gate_v1",
        "training_started": False,
        "source_e16_manifest_sha256": source_manifest_sha,
        "model_core_sha256": sha256_file(ROOT / "shared" / "model_core.py"),
        "runner_sha256": sha256_file(ROOT / "shared" / "context_runner.py"),
        "compiler_sha256": sha256_file(Path(__file__).resolve()),
        "variants": reports,
    }
    plan_bytes = write_json(ROOT / "context_gate_package_plan.json", plan)
    print(json.dumps({"plan_sha256": sha256_bytes(plan_bytes), "variants": reports}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
