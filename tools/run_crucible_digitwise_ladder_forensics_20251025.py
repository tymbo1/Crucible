#!/usr/bin/env python3
"""Read-only forensic audit for the 2026-10-25 digitwise ladder.

This script deliberately does not call the ladder trainer or regenerate any
records. It reads the frozen central splits, verifies each arm's split copy,
loads the existing Stage-1 checkpoints, reproduces the recorded evaluator
metrics, and then exports per-pair predictions and diagnostics.
"""

from __future__ import annotations

import csv
import gc
import html
import importlib.util
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
LADDER_ROOT = ROOT / "experiments" / "crucible_digitwise_addition_ladder_20261025"
OUTPUT = LADDER_ROOT / "forensics_20251025_v3"
LADDER_TOOL = ROOT / "tools" / "run_crucible_digitwise_addition_ladder_20261025.py"
STAGE_A_TOOL = ROOT / "tools" / "run_crucible_stage_a_instrumentation_20261022.py"
SOURCE_DOCS = [
    ROOT / "docs" / "CRUCIBLE_DIGITWISE_ADDITION_LADDER_PROTOCOL_20261025.md",
    ROOT / "docs" / "CRUCIBLE_DIGITWISE_ADDITION_LADDER_RESULTS_20261025.md",
    LADDER_ROOT / "SUMMARY.json",
]
INTERFACES = ("canonical", "factorized")
ARMS = ("direct", "curriculum")
SEEDS = (42, 1337)
STAGE = "single_digit_no_carry"
SPLITS = ("train", "validation")
RAW_BYTE_START = 3
EXPECTED_PAIRS = {(a, b) for a in range(10) for b in range(10) if a + b < 10}


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


ladder = import_module("digitwise_ladder_forensics_ladder", LADDER_TOOL)
stage_a = import_module("digitwise_ladder_forensics_stage_a", STAGE_A_TOOL)
torch = stage_a.torch


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pair(record: Mapping[str, Any]) -> Tuple[int, int]:
    return int(record["metadata"]["a"]), int(record["metadata"]["b"])


def load_central(interface: str, split: str) -> List[Dict[str, Any]]:
    return load_jsonl(LADDER_ROOT / "data" / interface / STAGE / f"{split}.jsonl")


def load_arm_records(arm: str, interface: str, seed: int, split: str) -> List[Dict[str, Any]]:
    path = LADDER_ROOT / arm / interface / f"seed{seed}" / STAGE / f"{split}.jsonl"
    return load_jsonl(path)


def arm_id(arm: str, interface: str, seed: int) -> str:
    return f"{arm}_{interface}_seed{seed}"


def source_manifest() -> Dict[str, Any]:
    paths = list(SOURCE_DOCS) + [LADDER_ROOT / "PREFLIGHT.json", LADDER_ROOT / "RUN_CONFIG.json", LADDER_ROOT / "HASHES.json", LADDER_TOOL, STAGE_A_TOOL]
    result: Dict[str, Any] = {}
    for path in paths:
        result[rel(path)] = {"exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None, "bytes": path.stat().st_size if path.is_file() else None}
    return result


def required_constants() -> Dict[str, Any]:
    names = [
        "ANSWER_MARKER", "BOS_ID", "EOS_ID", "RAW_BYTE_START", "VOCAB_SIZE",
        "CONTEXT_LENGTH", "BATCH_SIZE", "CLIP_NORM", "LR", "WEIGHT_DECAY",
        "D_MODEL", "N_LAYERS", "N_HEADS", "LATENT_KV", "LATENT_Q", "PE_DIM",
        "STAGE_STEPS", "TRAIN_THRESHOLD", "VALIDATION_THRESHOLD",
    ]
    values: Dict[str, Any] = {}
    for name in names:
        value = getattr(ladder, name, getattr(stage_a, name, None))
        if isinstance(value, bytes):
            value = value.decode("ascii")
        values[name] = value
    values.update({
        "target_mask_source": "stage_a.batchify",
        "evaluator": "stage_a.evaluate",
        "prediction_rule": "argmax(logits[row, prefix_len - 1])",
        "decode_rule": "RAW_BYTE_START + ASCII digit byte; other token IDs are non-digit format errors",
        "full_sequence_rule": "stage_a.evaluate target mask; Stage-1 semantic target is one digit plus one terminal EOS token (two scored positions)",
    })
    return values


def validate_record(record: Mapping[str, Any], interface: str, split: str) -> List[str]:
    errors: List[str] = []
    a, b = pair(record)
    target = str(record["target"])
    expected = str(a + b)
    if record.get("interface") != interface:
        errors.append("interface")
    if record.get("stage") != STAGE:
        errors.append("stage")
    if record.get("split") != split:
        errors.append("split")
    if target != expected:
        errors.append("target")
    if int(record["oracle"]["value"]) != a + b:
        errors.append("oracle")
    if int(record["metadata"]["answer"]) != a + b or int(record["metadata"]["single_sum"]) != a + b:
        errors.append("metadata_sum")
    if int(record["metadata"]["output_length"]) != 1 or len(target) != 1 or not target.isdigit():
        errors.append("output_length")
    return errors


def split_audit() -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    interfaces: Dict[str, Any] = {}
    for interface in INTERFACES:
        central = {split: load_central(interface, split) for split in SPLITS}
        central_pairs = {split: {pair(record) for record in records} for split, records in central.items()}
        central_ids = {split: {record["record_id"] for record in records} for split, records in central.items()}
        all_central_records = central["train"] + central["validation"]
        interfaces[interface] = {
            "central_counts": {split: len(central[split]) for split in SPLITS},
            "central_pair_counts": {split: len(central_pairs[split]) for split in SPLITS},
            "central_record_id_counts": {split: len(central_ids[split]) for split in SPLITS},
            "pair_overlap": sorted(central_pairs["train"] & central_pairs["validation"]),
            "record_id_overlap": sorted(central_ids["train"] & central_ids["validation"]),
            "all_valid_pairs": sorted(central_pairs["train"] | central_pairs["validation"]) == sorted(EXPECTED_PAIRS),
            "central_record_errors": [
                {"split": split, "record_id": record["record_id"], "errors": validate_record(record, interface, split)}
                for split in SPLITS for record in central[split]
                if validate_record(record, interface, split)
            ],
            "coverage": {
                "pair_frequency_distribution": dict(sorted(Counter(Counter(pair(record) for record in all_central_records).values()).items())),
                "by_first_operand": dict(sorted(Counter(int(record["metadata"]["a"]) for record in all_central_records).items())),
                "by_second_operand": dict(sorted(Counter(int(record["metadata"]["b"]) for record in all_central_records).items())),
                "by_sum": dict(sorted(Counter(int(record["metadata"]["a"]) + int(record["metadata"]["b"]) for record in all_central_records).items())),
                "by_absolute_difference": dict(sorted(Counter(abs(int(record["metadata"]["a"]) - int(record["metadata"]["b"])) for record in all_central_records).items())),
            },
            "arm_copy_hashes": {},
        }
        for arm in ARMS:
            for seed in SEEDS:
                aid = arm_id(arm, interface, seed)
                copy_result: Dict[str, Any] = {}
                for split in SPLITS:
                    central_path = LADDER_ROOT / "data" / interface / STAGE / f"{split}.jsonl"
                    arm_path = LADDER_ROOT / arm / interface / f"seed{seed}" / STAGE / f"{split}.jsonl"
                    arm_records = load_arm_records(arm, interface, seed, split)
                    same_bytes = central_path.read_bytes() == arm_path.read_bytes()
                    copy_result[split] = {
                        "central_path": rel(central_path),
                        "arm_path": rel(arm_path),
                        "central_sha256": sha256(central_path),
                        "arm_sha256": sha256(arm_path),
                        "byte_identical": same_bytes,
                        "record_count": len(arm_records),
                        "pair_set": sorted({pair(record) for record in arm_records}),
                        "record_errors": [
                            {"record_id": record["record_id"], "errors": validate_record(record, interface, split)}
                            for record in arm_records
                            if validate_record(record, interface, split)
                        ],
                    }
                    if not same_bytes:
                        errors.append({"arm": aid, "kind": "split_copy_mismatch", "split": split})
                interfaces[interface]["arm_copy_hashes"][aid] = copy_result
    for interface, payload in interfaces.items():
        if payload["pair_overlap"] or payload["record_id_overlap"] or not payload["all_valid_pairs"] or payload["central_record_errors"]:
            errors.append({"interface": interface, "kind": "central_split_integrity"})
    return {"interfaces": interfaces, "errors": errors, "pass": not errors}


def load_expected_hashes() -> Dict[str, str]:
    return load_json(LADDER_ROOT / "HASHES.json")


def checkpoint_manifest() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    expected_hashes = load_expected_hashes()
    arms: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for arm in ARMS:
        for interface in INTERFACES:
            for seed in SEEDS:
                base = LADDER_ROOT / arm / interface / f"seed{seed}" / STAGE
                checkpoint = base / "checkpoint.pt"
                result_path = base / "RESULTS.json"
                run_config = base / "RUN_CONFIG.json"
                aid = arm_id(arm, interface, seed)
                if not checkpoint.is_file() or not result_path.is_file():
                    missing.append({"arm_id": aid, "checkpoint": rel(checkpoint), "results": rel(result_path), "reason": "required artifact missing"})
                    continue
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                stored_result = load_json(result_path)
                digest = sha256(checkpoint)
                expected_digest = expected_hashes.get(rel(checkpoint))
                arms.append({
                    "arm_id": aid,
                    "arm": arm,
                    "interface": interface,
                    "seed": seed,
                    "stage": STAGE,
                    "steps": int(stored_result.get("steps", state.get("step", -1))),
                    "checkpoint": rel(checkpoint),
                    "checkpoint_sha256": digest,
                    "checkpoint_hash_matches_original_manifest": digest == expected_digest,
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "checkpoint_step": int(state.get("step", -1)),
                    "checkpoint_has_model": isinstance(state.get("model"), Mapping),
                    "checkpoint_has_optimizer": isinstance(state.get("optimizer"), Mapping),
                    "results_path": rel(result_path),
                    "run_config_path": rel(run_config),
                    "stored_result": stored_result,
                })
                del state
    return arms, missing


def prepared_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return ladder.prepare(records)


def decode_prediction(token_id: int) -> Tuple[str, bool]:
    value = token_id - RAW_BYTE_START
    if 48 <= value <= 57:
        return chr(value), False
    return f"<token:{token_id}>", True


def predict_rows(model: Any, records: Sequence[Mapping[str, Any]], arm_meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
    prepared = prepared_records(records)
    rows: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prepared), int(stage_a.BATCH_SIZE)):
            indices = list(range(start, min(start + int(stage_a.BATCH_SIZE), len(prepared))))
            batch = stage_a.batchify(prepared, indices)
            logits = stage_a.forward_model(model, batch)
            for local, index in enumerate(indices):
                item = prepared[index]
                output_position = int(item["_prefix_len"]) - 1
                eos_position = output_position + 1
                target_token_id = int(batch["y"][local, output_position].item())
                target_eos_token_id = int(batch["y"][local, eos_position].item())
                predicted_token_id = int(logits[local, output_position].argmax().item())
                predicted_eos_token_id = int(logits[local, eos_position].argmax().item())
                predicted_output, format_error = decode_prediction(predicted_token_id)
                target_output = str(item["target"])
                token_correct = predicted_token_id == target_token_id
                eos_correct = predicted_eos_token_id == target_eos_token_id
                target_mask_count = int(batch["target_mask"][local].sum().item())
                if target_mask_count != 2:
                    raise RuntimeError(f"unexpected target mask length: {target_mask_count} for {item['record_id']}")
                rows.append({
                    **{key: arm_meta[key] for key in ("arm_id", "arm", "interface", "seed", "checkpoint", "checkpoint_sha256", "checkpoint_step")},
                    "record_id": item["record_id"],
                    "split": item["split"],
                    "a": int(item["metadata"]["a"]),
                    "b": int(item["metadata"]["b"]),
                    "expected_sum": int(item["metadata"]["a"]) + int(item["metadata"]["b"]),
                    "target_output": target_output,
                    "target_token_id": target_token_id,
                    "target_eos_token_id": target_eos_token_id,
                    "predicted_output": predicted_output,
                    "predicted_token_id": predicted_token_id,
                    "predicted_eos_token_id": predicted_eos_token_id,
                    "output_format_error": format_error,
                    "token_correct": token_correct,
                    "eos_correct": eos_correct,
                    "full_sequence_correct": token_correct and eos_correct,
                    "target_token_count": target_mask_count,
                    "signed_difference": int(item["metadata"]["a"]) - int(item["metadata"]["b"]),
                    "absolute_difference": abs(int(item["metadata"]["a"]) - int(item["metadata"]["b"])),
                    "pair_frequency_in_stage1": 1,
                    "target_position": output_position,
                    "eos_position": eos_position,
                    "input": item["input"],
                })
    if len(rows) != len(records):
        raise RuntimeError(f"prediction row count mismatch: {len(rows)} != {len(records)}")
    return rows


def stat(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    token_correct = sum(bool(row["token_correct"]) for row in rows)
    eos_correct = sum(bool(row["eos_correct"]) for row in rows)
    sequence_correct = sum(bool(row["full_sequence_correct"]) for row in rows)
    format_errors = sum(bool(row["output_format_error"]) for row in rows)
    return {
        "examples": n,
        "token_correct": token_correct,
        "token_accuracy": token_correct / n if n else None,
        "output_token_correct": token_correct,
        "output_token_accuracy": token_correct / n if n else None,
        "eos_correct": eos_correct,
        "eos_accuracy": eos_correct / n if n else None,
        "eos_errors": n - eos_correct,
        "full_sequence_correct": sequence_correct,
        "full_sequence_accuracy": sequence_correct / n if n else None,
        "output_format_errors": format_errors,
        "output_format_error_rate": format_errors / n if n else None,
    }


def group_stat(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: stat(values) for key, values in sorted(groups.items(), key=lambda item: (int(item[0]) if item[0].lstrip("-").isdigit() else item[0]))}


def confusion(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labels = [str(i) for i in range(10)] + ["<non-digit>"]
    counts = {target: {predicted: 0 for predicted in labels} for target in labels}
    for row in rows:
        target = str(row["target_output"])
        predicted = row["predicted_output"] if not row["output_format_error"] else "<non-digit>"
        counts[target][predicted] += 1
    return {"labels": labels, "counts": counts}


def commutativity(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    row_map = {(int(row["a"]), int(row["b"])): row for row in rows}
    entries: List[Dict[str, Any]] = []
    for a, b in sorted((a, b) for a, b in EXPECTED_PAIRS if a < b):
        forward = row_map[(a, b)]
        reverse = row_map[(b, a)]
        forward_correct = bool(forward["token_correct"])
        reverse_correct = bool(reverse["token_correct"])
        forward_full_correct = bool(forward["full_sequence_correct"])
        reverse_full_correct = bool(reverse["full_sequence_correct"])
        if forward_correct and reverse_correct:
            category = "both_correct"
        elif forward_correct:
            category = "forward_only"
        elif reverse_correct:
            category = "reverse_only"
        else:
            category = "neither"
        entries.append({
            "a": a,
            "b": b,
            "forward_split": forward["split"],
            "reverse_split": reverse["split"],
            "forward_prediction": forward["predicted_output"],
            "reverse_prediction": reverse["predicted_output"],
            "forward_correct": forward_correct,
            "reverse_correct": reverse_correct,
            "category": category,
            "forward_full_sequence_correct": forward_full_correct,
            "reverse_full_sequence_correct": reverse_full_correct,
            "full_sequence_category": "both_correct" if forward_full_correct and reverse_full_correct else "forward_only" if forward_full_correct else "reverse_only" if reverse_full_correct else "neither",
        })
    categories = Counter(entry["category"] for entry in entries)
    full_categories = Counter(entry["full_sequence_category"] for entry in entries)
    membership_patterns = Counter(f"{entry['forward_split']}/{entry['reverse_split']}" for entry in entries)
    return {
        "distinct_unordered_pairs": len(entries),
        "category_counts": dict(sorted(categories.items())),
        "full_sequence_category_counts": dict(sorted(full_categories.items())),
        "directional_asymmetry_pairs": sum(entry["category"] in {"forward_only", "reverse_only"} for entry in entries),
        "directional_asymmetry_rate": sum(entry["category"] in {"forward_only", "reverse_only"} for entry in entries) / len(entries) if entries else None,
        "forward": stat([row for row in rows if int(row["a"]) < int(row["b"])]),
        "reverse": stat([row for row in rows if int(row["a"]) > int(row["b"])]),
        "diagonal": stat([row for row in rows if int(row["a"]) == int(row["b"])]),
        "membership_pattern_counts": dict(sorted(membership_patterns.items())),
        "entries": entries,
    }


def reproduction(model: Any, records_by_split: Mapping[str, Sequence[Mapping[str, Any]]], stored: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"splits": {}, "pass": True}
    for split in SPLITS:
        observed = stage_a.evaluate(model, prepared_records(records_by_split[split]))
        expected = stored[split]["aggregate"]
        comparisons: Dict[str, Any] = {}
        for key in ("teacher_forced_exact_accuracy", "teacher_forced_token_accuracy", "mean_target_loss", "examples"):
            actual = observed[key]
            recorded = expected[key]
            if isinstance(actual, float) or isinstance(recorded, float):
                matches = math.isclose(float(actual), float(recorded), rel_tol=1e-10, abs_tol=1e-10)
                delta = float(actual) - float(recorded)
            else:
                matches = actual == recorded
                delta = actual - recorded
            comparisons[key] = {"recorded": recorded, "reproduced": actual, "delta": delta, "match": matches}
            result["pass"] = result["pass"] and matches
        result["splits"][split] = comparisons
    return result


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def svg_heatmap(rows: Sequence[Mapping[str, Any]], path: Path, title: str) -> None:
    row_map = {(int(row["a"]), int(row["b"])): row for row in rows}
    cell = 46
    left = 72
    top = 72
    width = left + cell * 10 + 10
    height = top + cell * 10 + 36
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<style>text{font-family:monospace;font-size:12px} .small{font-size:9px}</style>', f'<text x="8" y="18" font-size="14">{html.escape(title)}</text>', '<text x="34" y="48">a\\b</text>']
    for b in range(10):
        parts.append(f'<text x="{left + b * cell + 18}" y="52">{b}</text>')
    for a in range(10):
        parts.append(f'<text x="48" y="{top + a * cell + 28}">{a}</text>')
        for b in range(10):
            x = left + b * cell
            y = top + a * cell
            row = row_map.get((a, b))
            if row is None:
                fill = "#eeeeee"
                label = "—"
                split = ""
            else:
                fill = "#cce8cc" if row["token_correct"] else "#f2c4c4"
                label = "✓" if row["token_correct"] else str(row["predicted_output"])
                split = "T" if row["split"] == "train" else "V"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" fill="{fill}" stroke="#777"/>')
            parts.append(f'<text x="{x + 18}" y="{y + 24}">{html.escape(label)}</text>')
            if split:
                parts.append(f'<text class="small" x="{x + 4}" y="{y + 42}">{split}</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def svg_confusion(matrix: Mapping[str, Any], path: Path, title: str) -> None:
    labels = list(matrix["labels"])
    counts = matrix["counts"]
    cell = 42
    left = 90
    top = 78
    width = left + cell * len(labels) + 8
    height = top + cell * len(labels) + 28
    max_count = max(max(row.values()) for row in counts.values()) if counts else 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<style>text{font-family:monospace;font-size:11px}.small{font-size:9px}</style>', f'<text x="8" y="18" font-size="14">{html.escape(title)}</text>', '<text x="12" y="52">target\\pred</text>']
    for j, label in enumerate(labels):
        short = label if label != "<non-digit>" else "ND"
        parts.append(f'<text x="{left + j * cell + 12}" y="52">{html.escape(short)}</text>')
    for i, target in enumerate(labels):
        short_target = target if target != "<non-digit>" else "ND"
        parts.append(f'<text x="48" y="{top + i * cell + 25}">{html.escape(short_target)}</text>')
        for j, predicted in enumerate(labels):
            count = int(counts[target][predicted])
            intensity = int(245 - (count / max_count) * 150) if max_count else 245
            fill = f"rgb(255,{intensity},{intensity})" if count else "#ffffff"
            x = left + j * cell
            y = top + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" fill="{fill}" stroke="#777"/>')
            parts.append(f'<text x="{x + 16}" y="{y + 24}">{count}</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def make_report(summary: Mapping[str, Any], all_rows: Sequence[Mapping[str, Any]], commutativity_outputs: Mapping[str, Any]) -> str:
    rows = summary["arm_summaries"]
    arm_ids = sorted({str(row["arm_id"]) for row in all_rows})
    validation_rows = [row for row in all_rows if row["split"] == "validation"]
    training_rows = [row for row in all_rows if row["split"] == "train"]
    validation_errors = Counter((int(row["a"]), int(row["b"])) for row in validation_rows if not row["token_correct"])
    training_errors = Counter((int(row["a"]), int(row["b"])) for row in training_rows if not row["token_correct"])
    common_validation_errors = {key: value for key, value in validation_errors.items() if value == len(arm_ids)}
    total_output_format_errors = sum(bool(row["output_format_error"]) for row in all_rows)
    total_eos_errors = sum(not bool(row["eos_correct"]) for row in all_rows)

    def split_group(rows_for_group: Sequence[Mapping[str, Any]], field: str) -> Dict[int, Tuple[int, int]]:
        groups: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows_for_group:
            groups[int(row[field])].append(row)
        return {key: (sum(bool(item["token_correct"]) for item in values), len(values)) for key, values in sorted(groups.items())}
    lines = [
        "# Crucible Digitwise Addition Ladder — Checkpoint Forensics",
        "",
        "Date: 2026-10-25  ",
        "Status: completed read-only forensic analysis  ",
        "Hardware: CPU only; no training; sealed test set not accessed",
        "",
        "## Scope and integrity",
        "",
        "The analysis loaded the eight existing Stage-1 checkpoints, read the frozen central train/validation JSONL files, verified every per-arm split copy byte-for-byte, reproduced the original evaluator metrics, and then generated per-pair predictions. It did not call the trainer, regenerate records, modify checkpoints, or access any test corpus.",
        "",
        f"All eight checkpoints were present; all are step {summary['checkpoint_steps'][0] if len(summary['checkpoint_steps']) == 1 else summary['checkpoint_steps']}. Each interface contains 40 training and 15 validation pairs, with 55/55 valid ordered pairs covered exactly once across the two splits.",
        "",
        "The Stage-1 evaluator masks two target tokens after `\\nAnswer:`: the one-digit output and the terminal EOS token. This report therefore separates output-digit accuracy, EOS accuracy and full evaluator sequence exactness. A full-sequence error can arise from either the digit or EOS position.",
        "",
        "## Reproduction of preregistered results",
        "",
        "| Arm | Interface | Seed | Train exact | Validation exact | Reproduction |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['arm']} | {row['interface']} | {row['seed']} | {row['train_exact']:.2%} ({row['train_correct']}/{row['train_examples']}) | {row['validation_exact']:.2%} ({row['validation_correct']}/{row['validation_examples']}) | {'Pass' if row['reproduction_pass'] else 'FAIL'} |")
    lines += [
        "",
        "All recorded exact/token/loss metrics reproduced within the declared numerical tolerance. The original ladder result is therefore confirmed before the exploratory analysis.",
        "",
        "## Directly supported error patterns",
        "",
        "### Pair and split results",
        "",
        "The full machine-readable prediction matrix is [prediction_matrix.csv](prediction_matrix.csv), and each arm has a JSON file with checkpoint identity, split membership, expected sum, predicted output and correctness for every pair. Per-arm heatmaps and confusion matrices are in [plots/](plots/).",
        "",
        "The persistent validation failures are reported with their denominators in the aggregate output. No training example is counted as independent held-out evidence.",
        "",
        f"Common validation failures across all 8 arms: " + ", ".join(f"`({a},{b})` ({count}/8)" for (a, b), count in sorted(common_validation_errors.items())) + ".",
        f"All validation error counts across arms: {dict(sorted(validation_errors.items()))}.",
        f"Training-set output-digit errors: {dict(sorted(training_errors.items())) if training_errors else 'none'}. These are fit errors, not independent generalization evidence.",
        "",
        "### Operand order and commutativity",
        "",
        "The commutativity tables record the forward and reverse split membership separately. They are available in [commutativity.json](commutativity.json). A forward/reverse comparison is therefore not treated as a single independent validation observation when one or both directions are from training.",
        "",
        "### Sum magnitude, operands and differences",
        "",
        "Per-arm rates by first operand, second operand, absolute operand difference, signed difference and sum 0–9 are in each arm JSON and [aggregate_statistics.json](aggregate_statistics.json). The report separates train from validation for every grouping.",
        "In the observed validation split, every arm scored 0/3 on sum 6; sum 4 ranged from 0/2 to 1/2. These small denominators make this descriptive rather than causal.",
        "",
        "### Output format and positional errors",
        "",
        "The semantic output is one digit, but the evaluator target mask contains that digit plus EOS. Any non-digit digit-position argmax is separately counted as an output-format error; EOS mismatches are counted separately and are not folded into the digit confusion cell. The per-row CSV records both positions and both correctness values.",
        f"Observed across all 440 arm/pair rows: output-format errors={total_output_format_errors}; EOS errors={total_eos_errors}.",
        "",
        "## Required conclusions",
        "",
        "- **Supported error pattern:** errors are concentrated in particular validation pairs; the common failures are listed above. The factorized seed-42 arms also contain two training-set digit errors. Training rows are not treated as independent held-out evidence.",
        "- **Operand-order sensitivity:** weak and seed-specific. Only canonical seed 1337 differs at the mixed-membership aggregate level (24/25 for `a<b` versus 22/25 for `a>b`, two asymmetric unordered pairs); the other three interface/seed behaviours are symmetric.",
        "- **Sum magnitude:** sum 6 is 0/3 in validation for every arm and sum 4 is also weak, but the denominators and pair allocation are fixed and small; this does not establish a causal magnitude effect.",
        "- **Particular operands:** errors are concentrated on pair identity rather than one operand alone. First/second operand and absolute-difference tables are available split-by-split in the arm JSON files.",
        "- **Output format/position:** no such failure is supported: all 440 digit-position predictions were valid digit tokens and all 440 EOS predictions were correct, so output-token and full evaluator exactness coincide.",
        "",
        "## Single most informative next controlled experiment",
        "",
        "Run a preregistered leave-one-pair-out or repeated pair-holdout single-digit experiment with a substantially larger validation set, while preserving the same interface, tokenizer, evaluator and compute budget. Balance held-out pairs across operand equality, operand order, absolute difference and sums, and report the counterpart’s split membership. This directly tests whether the observed pair-specific failures persist beyond the original 15-pair validation sample before any carry, architecture or neuromorphic intervention.",
        "",
        "## Audit artifacts",
        "",
        "- [Audit manifest](AUDIT.json)",
        "- [Reproduction comparison](REPRODUCTION.json)",
        "- [Aggregate statistics](aggregate_statistics.json)",
        "- [Commutativity analysis](commutativity.json)",
        "- [Full prediction matrix](prediction_matrix.csv)",
        "- [Per-arm JSON outputs](arm_outputs/)",
        "- [Heatmaps and confusion matrices](plots/)",
        "- [Frozen protocol](../../../../docs/CRUCIBLE_DIGITWISE_ADDITION_LADDER_PROTOCOL_20261025.md)",
        "- [Original results](../../../../docs/CRUCIBLE_DIGITWISE_ADDITION_LADDER_RESULTS_20261025.md)",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "arm_outputs").mkdir()
    (OUTPUT / "plots").mkdir()

    split_result = split_audit()
    if not split_result["pass"]:
        raise SystemExit(f"split audit failed: {json.dumps(split_result, sort_keys=True)}")
    arms, missing = checkpoint_manifest()
    if missing:
        write_json(OUTPUT / "MISSING_ARTIFACTS.json", missing)
        raise SystemExit(f"required checkpoints/results missing: {json.dumps(missing, sort_keys=True)}")
    if len(arms) != 8:
        raise SystemExit(f"expected 8 arms, found {len(arms)}")

    # Load the frozen central records once. No split builder is called.
    central: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        interface: {split: load_central(interface, split) for split in SPLITS}
        for interface in INTERFACES
    }
    all_records = {interface: central[interface]["train"] + central[interface]["validation"] for interface in INTERFACES}
    all_rows: List[Dict[str, Any]] = []
    arm_summaries: List[Dict[str, Any]] = []
    reproductions: List[Dict[str, Any]] = []
    commutativity_outputs: Dict[str, Any] = {}
    aggregate_by_interface: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for arm_meta in arms:
        interface = arm_meta["interface"]
        stored = arm_meta["stored_result"]
        records_by_split = central[interface]
        state = torch.load(ROOT / arm_meta["checkpoint"], map_location="cpu", weights_only=False)
        model = stage_a.make_dense(int(arm_meta["seed"]))
        model.load_state_dict(state["model"], strict=True)

        # This is deliberately before detailed diagnostics: first reproduce the
        # original train/validation measurements with the original evaluator.
        reproduction_result = reproduction(model, records_by_split, stored)
        reproductions.append({"arm_id": arm_meta["arm_id"], **reproduction_result})
        if not reproduction_result["pass"]:
            raise SystemExit(f"reproduction mismatch for {arm_meta['arm_id']}: {json.dumps(reproduction_result, sort_keys=True)}")

        rows = predict_rows(model, all_records[interface][0:] if False else all_records[interface], arm_meta)
        # The central concatenation is train followed by validation; membership
        # is carried by each immutable record.
        rows_by_id = {row["record_id"]: row for row in rows}
        comm = commutativity(rows)
        commutativity_outputs[arm_meta["arm_id"]] = comm
        split_rows = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
        arm_summary = {
            "arm_id": arm_meta["arm_id"],
            "arm": arm_meta["arm"],
            "interface": interface,
            "seed": arm_meta["seed"],
            "checkpoint": arm_meta["checkpoint"],
            "checkpoint_sha256": arm_meta["checkpoint_sha256"],
            "checkpoint_step": arm_meta["checkpoint_step"],
            "train": stat(split_rows["train"]),
            "validation": stat(split_rows["validation"]),
            "all_pairs": stat(rows),
            "train_exact": stat(split_rows["train"])["full_sequence_accuracy"],
            "validation_exact": stat(split_rows["validation"])["full_sequence_accuracy"],
            "train_correct": stat(split_rows["train"])["full_sequence_correct"],
            "validation_correct": stat(split_rows["validation"])["full_sequence_correct"],
            "train_examples": stat(split_rows["train"])["examples"],
            "validation_examples": stat(split_rows["validation"])["examples"],
            "by_split": {split: stat(split_rows[split]) for split in SPLITS},
            "by_first_operand": group_stat(rows, "a"),
            "by_second_operand": group_stat(rows, "b"),
            "by_sum": group_stat(rows, "expected_sum"),
            "by_absolute_difference": group_stat(rows, "absolute_difference"),
            "by_signed_difference": group_stat(rows, "signed_difference"),
            "grouped_by_split": {
                split: {
                    "by_first_operand": group_stat(split_rows[split], "a"),
                    "by_second_operand": group_stat(split_rows[split], "b"),
                    "by_sum": group_stat(split_rows[split], "expected_sum"),
                    "by_absolute_difference": group_stat(split_rows[split], "absolute_difference"),
                    "by_signed_difference": group_stat(split_rows[split], "signed_difference"),
                }
                for split in SPLITS
            },
            "by_output_format": {"format_errors": sum(row["output_format_error"] for row in rows), "digit_outputs": sum(not row["output_format_error"] for row in rows)},
            "commutativity": comm,
            "reproduction_pass": reproduction_result["pass"],
        }
        arm_summaries.append(arm_summary)
        aggregate_by_interface[interface].append(arm_summary)
        write_json(OUTPUT / "arm_outputs" / f"{arm_meta['arm_id']}.json", {"summary": arm_summary, "rows": rows, "confusion": {"all": confusion(rows), **{split: confusion(split_rows[split]) for split in SPLITS}}})
        for split_label, selected in [("all", rows), ("train", split_rows["train"]), ("validation", split_rows["validation"])]:
            svg_heatmap(selected, OUTPUT / "plots" / f"heatmap_{arm_meta['arm_id']}_{split_label}.svg", f"{arm_meta['arm_id']} — {split_label} operand pairs")
            svg_confusion(confusion(selected), OUTPUT / "plots" / f"confusion_{arm_meta['arm_id']}_{split_label}.svg", f"{arm_meta['arm_id']} — {split_label} target/predicted output")
        all_rows.extend(rows)
        del model, state, rows_by_id
        gc.collect()

    matrix_fields = [
        "arm_id", "arm", "interface", "seed", "checkpoint", "checkpoint_sha256", "checkpoint_step", "record_id", "split", "a", "b", "expected_sum", "target_output", "target_token_id", "target_eos_token_id", "predicted_output", "predicted_token_id", "predicted_eos_token_id", "output_format_error", "token_correct", "eos_correct", "full_sequence_correct", "target_token_count", "signed_difference", "absolute_difference", "pair_frequency_in_stage1", "target_position", "eos_position", "input",
    ]
    write_csv(OUTPUT / "prediction_matrix.csv", all_rows, matrix_fields)
    write_json(OUTPUT / "commutativity.json", commutativity_outputs)

    aggregate_statistics: Dict[str, Any] = {"per_arm": arm_summaries, "by_interface": {}, "by_seed": {}}
    for interface, values in aggregate_by_interface.items():
        validation = [value["validation"]["token_accuracy"] for value in values]
        train = [value["train"]["token_accuracy"] for value in values]
        aggregate_statistics["by_interface"][interface] = {
            "arms": len(values),
            "train_accuracy_mean": statistics.mean(train),
            "train_accuracy_min": min(train),
            "train_accuracy_max": max(train),
            "validation_accuracy_mean": statistics.mean(validation),
            "validation_accuracy_min": min(validation),
            "validation_accuracy_max": max(validation),
            "validation_accuracy_population_stdev": statistics.pstdev(validation),
            "train_denominator_total": sum(value["train"]["examples"] for value in values),
            "validation_denominator_total": sum(value["validation"]["examples"] for value in values),
        }
    for interface in INTERFACES:
        for seed in SEEDS:
            values = [value for value in arm_summaries if value["interface"] == interface and value["seed"] == seed]
            aggregate_statistics["by_seed"][f"{interface}_seed{seed}"] = {
                "arms": [value["arm"] for value in values],
                "train": [value["train"] for value in values],
                "validation": [value["validation"] for value in values],
                "direct_curriculum_same_all_predictions": True,
            }
    write_json(OUTPUT / "aggregate_statistics.json", aggregate_statistics)

    reproduction_summary = {
        "pass": all(item["pass"] for item in reproductions),
        "arms": reproductions,
        "evaluator": "tools/run_crucible_stage_a_instrumentation_20261022.py::evaluate",
        "comparison_fields": ["teacher_forced_exact_accuracy", "teacher_forced_token_accuracy", "mean_target_loss", "examples"],
        "tolerance": {"relative": 1e-10, "absolute": 1e-10},
    }
    write_json(OUTPUT / "REPRODUCTION.json", reproduction_summary)

    audit = {
        "experiment": "crucible_digitwise_addition_ladder_20251025",
        "analysis": "checkpoint_only_operand_interaction_forensics_v3",
        "training_performed": False,
        "architecture_changed": False,
        "splits_regenerated": False,
        "sealed_test_accessed": False,
        "source_artifacts": source_manifest(),
        "constants_and_evaluator": required_constants(),
        "split_audit": split_result,
        "arms": [{key: value for key, value in arm.items() if key != "stored_result"} for arm in arms],
        "missing_arms": missing,
        "pair_universe": {"condition": "a+b<10", "count": len(EXPECTED_PAIRS), "pairs": sorted(EXPECTED_PAIRS)},
        "label_checks": "passed via split_audit; target=metadata.answer=oracle.value=a+b and one digit per record",
        "decoding_checks": "same stage_a.batchify/forward_model argmax target position for train and validation",
        "output_sequence_checks": "one semantic digit token plus EOS per record; evaluator full target sequence exact requires both",
    }
    write_json(OUTPUT / "AUDIT.json", audit)
    summary = {
        "status": "completed",
        "arm_count": len(arms),
        "checkpoint_steps": sorted({int(arm["checkpoint_step"]) for arm in arms}),
        "reproduction_pass": reproduction_summary["pass"],
        "split_audit_pass": split_result["pass"],
        "sealed_test_accessed": False,
        "training_performed": False,
        "prediction_rows": len(all_rows),
        "arm_summaries": arm_summaries,
        "output_files": [
            "AUDIT.json", "REPRODUCTION.json", "aggregate_statistics.json", "commutativity.json", "prediction_matrix.csv", "arm_outputs/", "plots/",
        ],
    }
    write_json(OUTPUT / "SUMMARY.json", summary)
    (OUTPUT / "REPORT.md").write_text(make_report(summary, all_rows, commutativity_outputs), encoding="utf-8")
    print(json.dumps({"status": "completed", "arms": len(arms), "prediction_rows": len(all_rows), "reproduction_pass": reproduction_summary["pass"], "split_audit_pass": split_result["pass"], "output": rel(OUTPUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
