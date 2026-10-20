#!/usr/bin/env python3
"""Fail-closed, preregistered Crucible routing-mass evaluator.

This evaluator measures byte-conditioned routing-mass differentiation only.
It does not infer semantic or syntactic specialization and it does not create
token-level permutation p-values from aggregated telemetry.

The evaluator is deliberately strict: required expert-by-vocabulary routing
mass must already be present in every recorded row.  Missing telemetry is an
error, never a reason to synthesize a uniform matrix or fill a metric with
zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


VOCAB_SIZE = 259
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
RAW_BYTE_START = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_parent(path: Path) -> None:
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_parent(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def manifest_binding_matches(stored: Any, full_hash: str) -> bool:
    """Accept the documented full SHA or a stored prefix of at least 16 chars."""
    if not isinstance(stored, str):
        return False
    candidate = stored.strip().lower()
    expected = full_hash.lower()
    return len(candidate) >= 16 and (
        candidate == expected or expected.startswith(candidate)
    )


def entropy_bits(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    positive = values > 0
    if not np.any(positive):
        return 0.0
    return float(-np.sum(values[positive] * np.log2(values[positive])))


def kl_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    left = np.asarray(p, dtype=np.float64)
    right = np.asarray(q, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"KL shape mismatch: {left.shape} vs {right.shape}")
    positive = left > 0
    if np.any(right[positive] <= 0):
        return float("inf")
    return float(np.sum(left[positive] * np.log2(left[positive] / right[positive])))


def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    left = np.asarray(p, dtype=np.float64)
    right = np.asarray(q, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"JSD shape mismatch: {left.shape} vs {right.shape}")
    left_mass = float(left.sum())
    right_mass = float(right.sum())
    if left_mass <= 0 or right_mass <= 0:
        raise ValueError("JSD requires positive-mass distributions")
    left = left / left_mass
    right = right / right_mass
    midpoint = 0.5 * (left + right)
    return 0.5 * kl_divergence_bits(left, midpoint) + 0.5 * kl_divergence_bits(right, midpoint)


def normalize_or_none(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return None
    return (values / total).tolist()


def analyze_affinity_matrix(affinity: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
    """Analyze one [expert, 259] routing-probability-mass matrix."""
    matrix = np.asarray(affinity, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Affinity must be rank 2, got {matrix.shape}")
    experts, vocabulary = matrix.shape
    if vocabulary != VOCAB_SIZE:
        raise ValueError(f"Expected vocab size {VOCAB_SIZE}, got {vocabulary}")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("Affinity contains nonfinite values")
    if np.any(matrix < 0):
        raise ValueError("Affinity contains negative routing mass")

    raw = matrix[:, RAW_BYTE_START:]
    total_mass = float(raw.sum())
    if total_mass <= 0:
        raise ValueError("Raw-byte affinity matrix contains zero mass")

    joint = raw / total_mass
    expert_marginal = joint.sum(axis=1)
    token_marginal = joint.sum(axis=0)
    independence = np.outer(expert_marginal, token_marginal)
    positive = joint > 0
    if np.any(independence[positive] <= 0):
        raise ValueError("Invalid independence model")
    mi_bits = float(np.sum(joint[positive] * np.log2(joint[positive] / independence[positive])))
    expert_entropy = entropy_bits(expert_marginal)

    profiles = np.empty_like(raw, dtype=np.float64)
    for expert in range(experts):
        mass = float(raw[expert].sum())
        if mass <= 0:
            raise ValueError(f"Expert {expert} has zero raw-byte routing mass")
        profiles[expert] = raw[expert] / mass

    jsd_matrix = np.zeros((experts, experts), dtype=np.float64)
    pairwise = []
    for first in range(experts):
        for second in range(first + 1, experts):
            distance = js_divergence_bits(profiles[first], profiles[second])
            jsd_matrix[first, second] = distance
            jsd_matrix[second, first] = distance
            pairwise.append(distance)

    return {
        "analytical_null": {
            "definition": "P0(E,T)=P(E)P(T)",
            "null_mi_bits": 0.0,
            "null_pairwise_jsd_bits": 0.0,
            "note": "Descriptive independence baseline; aggregated telemetry does not support token-level permutation p-values.",
        },
        "raw_byte_mass_total": total_mass,
        "expert_marginal": expert_marginal.tolist(),
        "expert_entropy_bits": expert_entropy,
        "token_entropy_bits": entropy_bits(token_marginal),
        "mi_bits": mi_bits,
        "nmi_by_expert_entropy": mi_bits / expert_entropy if expert_entropy > 0 else 0.0,
        "mean_pairwise_jsd_bits": float(np.mean(pairwise)) if pairwise else 0.0,
        "max_pairwise_jsd_bits": float(np.max(pairwise)) if pairwise else 0.0,
        "pairwise_jsd_matrix_bits": jsd_matrix.tolist(),
        "bos_expert_distribution": normalize_or_none(matrix[:, BOS_ID]),
        "eos_expert_distribution": normalize_or_none(matrix[:, EOS_ID]),
    }, profiles


# The actual Crucible flight uses specialization_telemetry.  The older names
# remain accepted for compatibility, but no affinity alias is invented here.
LAYER_CONTAINER_KEYS = (
    "specialization_telemetry",
    "layer_telemetry",
    "layers",
    "moe_layer_telemetry",
)
AFFINITY_KEYS = (
    "_token_affinity",
    "token_affinity",
    "routing_mass_affinity",
)
LOAD_COUNT_KEYS = (
    "top2_token_count",
    "top2_counts",
    "top_2_counts",
    "hard_assignments",
    "expert_selection_counts",
    "expert_counts",
    "selection_counts",
)


def first_present(mapping: Mapping[str, Any], keys: Sequence[str]):
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    return None, None


def get_layer_payload(row: Mapping[str, Any], layer_index: int) -> Mapping[str, Any]:
    container_key, container = first_present(row, LAYER_CONTAINER_KEYS)
    if container_key is None or not isinstance(container, Mapping):
        raise KeyError(
            "Required per-layer telemetry absent. "
            f"Expected one of {list(LAYER_CONTAINER_KEYS)}; row keys: {sorted(row.keys())}"
        )
    candidates = (f"layer_{layer_index}_moe", f"layer_{layer_index}", str(layer_index), layer_index)
    for key in candidates:
        if key in container:
            payload = container[key]
            if not isinstance(payload, Mapping):
                raise TypeError(f"Layer {layer_index} payload is not a mapping")
            return payload
    raise KeyError(f"Layer {layer_index} telemetry missing. Available: {list(container.keys())}")


def extract_affinity(payload: Mapping[str, Any], num_experts: int, vocab_size: int):
    key, value = first_present(payload, AFFINITY_KEYS)
    if key is None:
        raise KeyError(
            "Missing required expert-by-vocabulary routing-mass matrix. "
            f"Expected one of {list(AFFINITY_KEYS)}; available layer keys: {sorted(payload.keys())}"
        )
    array = np.asarray(value, dtype=np.float64)
    expected = (num_experts, vocab_size)
    if array.shape != expected:
        raise ValueError(f"{key} shape {array.shape}; expected {expected}")
    return key, array


def extract_load_counts(payload: Mapping[str, Any], num_experts: int):
    key, value = first_present(payload, LOAD_COUNT_KEYS)
    if key is None:
        raise KeyError(
            "Missing per-expert top-2 routing count vector. "
            f"Expected one of {list(LOAD_COUNT_KEYS)}; available layer keys: {sorted(payload.keys())}"
        )
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (num_experts,):
        raise ValueError(f"{key} shape {array.shape}; expected {(num_experts,)}")
    if np.any(~np.isfinite(array)) or np.any(array < 0) or array.sum() <= 0:
        raise ValueError(f"{key} contains invalid counts")
    return key, array


def balance_metrics(counts: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(counts, dtype=np.float64)
    shares = values / values.sum()
    mean = float(values.mean())
    minimum = float(values.min())
    return {
        "counts": values.tolist(),
        "shares": shares.tolist(),
        "share_sum": float(shares.sum()),
        "cv": float(values.std(ddof=0) / mean),
        "max_min_ratio": float(values.max() / minimum) if minimum > 0 else float("inf"),
    }


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} is not an object")
            rows.append(value)
    return rows


def analyze_experiment(metrics_path: Path, config_path: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(config_path)
    metrics_sha = sha256_file(metrics_path)
    rows = load_jsonl(metrics_path)
    hp = config["hyperparameters"]
    contract = config["contract"]
    seeds = [int(value) for value in contract["seeds"]]
    branches = [str(value) for value in contract["architectural_branches"]]
    learning_rates = [float(value) for value in contract["learning_rate_candidates"]]
    checkpoints = [int(value) for value in hp["checkpoint_intervals"]]
    num_layers = int(hp["n_layers"])
    num_experts = int(hp["num_experts"])
    vocab_size = int(hp["vocab_size"])
    if vocab_size != VOCAB_SIZE:
        raise ValueError(f"Expected vocab {VOCAB_SIZE}, got {vocab_size}")

    expected = {
        (seed, branch, learning_rate, checkpoint)
        for seed in seeds
        for branch in branches
        for learning_rate in learning_rates
        for checkpoint in checkpoints
    }
    matrix = {}
    scalar_fields = (
        "learning_rate", "interval_train_ce", "validation_nll", "byte_perplexity",
        "grad_norm_mean", "grad_norm_max", "grad_norm_p95", "interval_clipped_fraction",
    )
    for line_number, row in enumerate(rows, start=1):
        key = (int(row["seed"]), str(row["branch"]), float(row["peak_lr"]), int(row["interval_step_end"]))
        if key not in expected:
            raise ValueError(f"Unexpected matrix row at line {line_number}: {key}")
        if key in matrix:
            raise ValueError(f"Duplicate matrix row: {key}")
        if not manifest_binding_matches(row.get("master_config_sha256"), manifest_sha):
            raise ValueError(f"Manifest binding mismatch at JSONL line {line_number}")
        for field in scalar_fields:
            if field not in row or not math.isfinite(float(row[field])):
                raise ValueError(f"Missing or nonfinite {field} at JSONL line {line_number}")
        if int(row.get("interval_nan_inf_count", 0)) != 0:
            raise ValueError(f"Recorded nonfinite training event at line {line_number}")
        matrix[key] = row
    missing = expected - set(matrix)
    if missing:
        raise ValueError(f"Missing {len(missing)} matrix rows; example: {sorted(missing)[:5]}")
    if len(rows) != len(expected):
        raise ValueError(f"Expected {len(expected)} rows, found {len(rows)}")

    profiles = {}
    report = {
        "metadata": {
            "input_metrics_jsonl_sha256": metrics_sha,
            "manifest_sha256": manifest_sha,
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "row_count": len(rows),
            "scope": "byte-conditioned routing-mass differentiation",
            "does_not_claim": ["semantic specialization", "syntactic specialization", "token-level permutation p-values"],
        },
        "contract": {
            "seeds": seeds, "branches": branches, "learning_rates": learning_rates,
            "checkpoints": checkpoints, "num_layers": num_layers, "num_experts": num_experts,
            "vocab_size": vocab_size, "pad_excluded": True, "bos_eos_separate": True,
            "analytical_null": "P0(E,T)=P(E)P(T)",
        },
        "trajectories": {},
        "cross_seed_matching": {},
    }

    for seed in seeds:
        for branch in branches:
            for learning_rate in learning_rates:
                run_key = f"seed_{seed}_branch_{branch}_lr_{learning_rate:.12g}"
                timeline = []
                previous_profiles = {}
                for checkpoint in checkpoints:
                    row = matrix[(seed, branch, learning_rate, checkpoint)]
                    point = {
                        "step": checkpoint,
                        "learning_rate": float(row["learning_rate"]),
                        "train_ce": float(row["interval_train_ce"]),
                        "validation_nll": float(row["validation_nll"]),
                        "byte_perplexity": float(row["byte_perplexity"]),
                        "grad_norm_mean": float(row["grad_norm_mean"]),
                        "grad_norm_max": float(row["grad_norm_max"]),
                        "grad_norm_p95": float(row["grad_norm_p95"]),
                        "clipped_fraction": float(row["interval_clipped_fraction"]),
                        "layers": {},
                    }
                    for layer_index in range(num_layers):
                        payload = get_layer_payload(row, layer_index)
                        affinity_key, affinity = extract_affinity(payload, num_experts, vocab_size)
                        count_key, counts = extract_load_counts(payload, num_experts)
                        info, current_profiles = analyze_affinity_matrix(affinity)
                        profiles[(seed, branch, learning_rate, checkpoint, layer_index)] = current_profiles
                        stability = None
                        stability_per_expert = None
                        if layer_index in previous_profiles:
                            old_profiles = previous_profiles[layer_index]
                            stability_per_expert = [
                                1.0 - js_divergence_bits(old_profiles[expert], current_profiles[expert])
                                for expert in range(num_experts)
                            ]
                            stability = float(np.mean(stability_per_expert))
                        previous_profiles[layer_index] = current_profiles.copy()
                        point["layers"][f"layer_{layer_index}"] = {
                            "affinity_source_key": affinity_key,
                            "load_count_source_key": count_key,
                            "balance": balance_metrics(counts),
                            "information": info,
                            "interval_profile_stability_vs_previous": stability,
                            "interval_profile_stability_per_expert": stability_per_expert,
                        }
                    timeline.append(point)
                report["trajectories"][run_key] = timeline

    final_checkpoint = checkpoints[-1]
    for seed_a, seed_b in combinations(seeds, 2):
        pair_key = f"seed_{seed_a}_vs_{seed_b}"
        report["cross_seed_matching"][pair_key] = {}
        for branch in branches:
            for learning_rate in learning_rates:
                condition_key = f"branch_{branch}_lr_{learning_rate:.12g}"
                layer_results = {}
                for layer_index in range(num_layers):
                    profiles_a = profiles[(seed_a, branch, learning_rate, final_checkpoint, layer_index)]
                    profiles_b = profiles[(seed_b, branch, learning_rate, final_checkpoint, layer_index)]
                    cost = np.zeros((num_experts, num_experts), dtype=np.float64)
                    for first in range(num_experts):
                        for second in range(num_experts):
                            cost[first, second] = js_divergence_bits(profiles_a[first], profiles_b[second])
                    row_ind, col_ind = linear_sum_assignment(cost)
                    assignments = [
                        {"expert_seed_a": int(first), "expert_seed_b": int(second), "jsd_bits": float(cost[first, second])}
                        for first, second in zip(row_ind, col_ind)
                    ]
                    distances = [item["jsd_bits"] for item in assignments]
                    layer_results[f"layer_{layer_index}"] = {
                        "cost_matrix_jsd_bits": cost.tolist(),
                        "assignments": assignments,
                        "mean_assignment_jsd_bits": float(np.mean(distances)),
                        "max_assignment_jsd_bits": float(np.max(distances)),
                    }
                report["cross_seed_matching"][pair_key][condition_key] = {
                    "final_checkpoint": final_checkpoint,
                    "layers": layer_results,
                }
    return report


def flattened_rows(report: Mapping[str, Any]):
    for run_key, timeline in report["trajectories"].items():
        for point in timeline:
            for layer_id, layer in point["layers"].items():
                info = layer["information"]
                balance = layer["balance"]
                yield {
                    "run_key": run_key, "step": point["step"], "learning_rate": point["learning_rate"],
                    "train_ce": point["train_ce"], "validation_nll": point["validation_nll"],
                    "byte_perplexity": point["byte_perplexity"], "grad_norm_mean": point["grad_norm_mean"],
                    "grad_norm_max": point["grad_norm_max"], "grad_norm_p95": point["grad_norm_p95"],
                    "clipped_fraction": point["clipped_fraction"], "layer_id": layer_id,
                    "load_count_source_key": layer["load_count_source_key"],
                    "router_cv": balance["cv"], "router_max_min_ratio": balance["max_min_ratio"],
                    "mi_bits": info["mi_bits"], "nmi_by_expert_entropy": info["nmi_by_expert_entropy"],
                    "mean_pairwise_jsd_bits": info["mean_pairwise_jsd_bits"],
                    "max_pairwise_jsd_bits": info["max_pairwise_jsd_bits"],
                    "temporal_profile_stability": layer["interval_profile_stability_vs_previous"],
                }


def atomic_write_csv(path: Path, rows) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("No rows generated for CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_parent(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed Crucible routing-mass evaluator")
    parser.add_argument("--metrics-path", required=True, type=Path)
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    metrics_path = arguments.metrics_path.resolve()
    config_path = arguments.config_path.resolve()
    output_dir = arguments.output_dir.resolve()
    for path in (metrics_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    report = analyze_experiment(metrics_path, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator_sha = report["metadata"]["evaluator_sha256"]
    manifest_sha = report["metadata"]["manifest_sha256"]
    metrics_sha = report["metadata"]["input_metrics_jsonl_sha256"]
    analysis_config = {
        "protocol": "preregistered_byte_conditioned_routing_mass_v1",
        "input_metrics_jsonl_sha256": metrics_sha,
        "manifest_sha256": manifest_sha,
        "evaluator_sha256": evaluator_sha,
        "primary_scope": "raw byte ids 3..258",
        "pad_excluded": True,
        "bos_eos_reported_separately": True,
        "analytical_null": "P0(E,T)=P(E)P(T)",
        "cross_seed_alignment": "one-to-one Hungarian minimum JSD per layer at final checkpoint",
        "statistical_claim_limit": "No token-level permutation p-values from aggregated telemetry.",
    }
    config_out = output_dir / "analysis_config.json"
    analysis_out = output_dir / "specialization_analysis.json"
    csv_out = output_dir / "specialization_analysis.csv"
    matching_out = output_dir / "cross_seed_assignment.json"
    hashes_out = output_dir / "analysis_output_hashes.json"
    atomic_write_json(config_out, analysis_config)
    atomic_write_json(analysis_out, report)
    atomic_write_json(matching_out, report["cross_seed_matching"])
    atomic_write_csv(csv_out, flattened_rows(report))
    output_hashes = {
        "evaluator_source_sha256": evaluator_sha,
        "input_metrics_jsonl_sha256": metrics_sha,
        "experiment_manifest_sha256": manifest_sha,
        "analysis_config.json": sha256_file(config_out),
        "specialization_analysis.json": sha256_file(analysis_out),
        "specialization_analysis.csv": sha256_file(csv_out),
        "cross_seed_assignment.json": sha256_file(matching_out),
    }
    atomic_write_json(hashes_out, output_hashes)
    print("PREREGISTERED CRUCIBLE EVALUATOR: SUCCESS")
    print("Rows audited:", report["metadata"]["row_count"])
    print("Metrics SHA-256:", metrics_sha)
    print("Manifest SHA-256:", manifest_sha)
    print("Evaluator SHA-256:", evaluator_sha)
    print("Outputs:", output_dir)
    for name, digest in output_hashes.items():
        print(name, digest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
