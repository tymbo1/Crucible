#!/usr/bin/env python3
"""Read-only post-flight analysis for the sealed E4/E8/E16 gate.

The script never edits the full_run outputs.  It validates the exact sealed
matrix, checks the recorded telemetry invariants, and writes only to the new
analysis_v1_20261017 directory.

The specialization quantities are descriptive byte-conditioned routing-mass
statistics.  They do not establish semantic or syntactic specialization.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_v1_20261017"
PLAN_PATH = ROOT / "package_plan.json"
LEDGER_PATH = ROOT / "full_run" / "run_ledger.json"
CHECKPOINTS = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3125]
SEEDS = [42, 1337]
BRANCH = "B"
LR = 0.0003
NUM_LAYERS = 4
VOCAB_SIZE = 259
RAW_START = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return math.isfinite(float(value))


def entropy(values: list[float]) -> float:
    return -sum(value * math.log2(value) for value in values if value > 0.0)


def jsd_bits(left: list[float], right: list[float]) -> float:
    midpoint = [(a + b) * 0.5 for a, b in zip(left, right)]
    result = 0.0
    for probability, mixture in zip(left, midpoint):
        if probability > 0.0:
            result += 0.5 * probability * math.log2(probability / mixture)
    for probability, mixture in zip(right, midpoint):
        if probability > 0.0:
            result += 0.5 * probability * math.log2(probability / mixture)
    return result


def affinity_metrics(affinity: list[list[float]]) -> tuple[dict[str, Any], list[list[float]]]:
    raw = [row[RAW_START:VOCAB_SIZE] for row in affinity]
    total = sum(sum(row) for row in raw)
    if total <= 0.0:
        raise ValueError("raw-byte affinity has zero mass")

    expert_mass = [sum(row) for row in raw]
    token_mass = [sum(raw[expert][token] for expert in range(len(raw))) for token in range(VOCAB_SIZE - RAW_START)]
    p_expert = [mass / total for mass in expert_mass]
    p_token = [mass / total for mass in token_mass]

    mi = 0.0
    for expert, row in enumerate(raw):
        for token, mass in enumerate(row):
            joint = mass / total
            if joint > 0.0:
                independent = p_expert[expert] * p_token[token]
                mi += joint * math.log2(joint / independent)

    expert_entropy = entropy(p_expert)
    profiles = []
    for mass, row in zip(expert_mass, raw):
        if mass <= 0.0:
            raise ValueError("expert has zero raw-byte affinity mass")
        profiles.append([value / mass for value in row])

    pairwise = [
        jsd_bits(profiles[left], profiles[right])
        for left, right in combinations(range(len(profiles)), 2)
    ]

    return {
        "raw_byte_mass_total": total,
        "expert_entropy_bits": expert_entropy,
        "token_entropy_bits": entropy(p_token),
        "mi_bits": mi,
        "nmi_by_expert_entropy": mi / expert_entropy if expert_entropy > 0.0 else 0.0,
        "mean_pairwise_jsd_bits": sum(pairwise) / len(pairwise) if pairwise else 0.0,
        "max_pairwise_jsd_bits": max(pairwise) if pairwise else 0.0,
        "bos_expert_distribution": None,
        "eos_expert_distribution": None,
    }, profiles


def normalized_distribution(values: list[float]) -> list[float] | None:
    total = sum(values)
    if total <= 0.0:
        return None
    return [value / total for value in values]


def analyze_layer(payload: dict[str, Any], num_experts: int) -> tuple[dict[str, Any], list[list[float]], dict[str, float]]:
    required_vectors = (
        "top1_counts",
        "top2_counts",
        "hard_assignments",
        "mean_selected_probability",
        "mean_expert_output_norm",
    )
    affinity = payload.get("_token_affinity")
    selected = payload.get("selected_token_counts")
    if not isinstance(affinity, list) or len(affinity) != num_experts:
        raise ValueError("invalid expert dimension for _token_affinity")
    if not isinstance(selected, list) or len(selected) != num_experts:
        raise ValueError("invalid expert dimension for selected_token_counts")
    if any(not isinstance(row, list) or len(row) != VOCAB_SIZE for row in affinity + selected):
        raise ValueError("invalid telemetry vocabulary dimension")
    for row in affinity + selected:
        if any(not finite(value) or float(value) < 0.0 for value in row):
            raise ValueError("nonfinite or negative affinity/selection value")
    for key in required_vectors:
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != num_experts:
            raise ValueError(f"invalid {key} vector")
        if any(not finite(value) or float(value) < 0.0 for value in values):
            raise ValueError(f"invalid {key} value")

    tokens = int(payload["tokens_routed"])
    top_k = int(payload["top_k"])
    top1 = [float(value) for value in payload["top1_counts"]]
    top2 = [float(value) for value in payload["top2_counts"]]
    hard = [float(value) for value in payload["hard_assignments"]]
    selected_total = sum(sum(row) for row in selected)
    errors = {
        "affinity_mass": abs(float(payload["affinity_mass_total"]) - tokens),
        "top1_count": abs(sum(top1) - tokens),
        "top2_count": abs(sum(top2) - tokens * top_k),
        "hard_count": abs(sum(hard) - tokens * top_k),
        "selected_token_count": abs(selected_total - tokens * top_k),
    }
    conservation_tolerance = max(1e-5, tokens * 1e-6)
    if max(errors.values()) > conservation_tolerance:
        raise ValueError(f"telemetry conservation failure: {errors}")

    info, profiles = affinity_metrics(affinity)
    info["bos_expert_distribution"] = normalized_distribution(
        [row[1] for row in affinity]
    )
    info["eos_expert_distribution"] = normalized_distribution(
        [row[2] for row in affinity]
    )
    load_total = sum(hard)
    shares = [value / load_total for value in hard]
    metrics = {
        "router_cv": float(payload["global_hard_cv"]),
        "router_max_min_ratio": float(payload["max_min_hard_ratio"]),
        "min_top2_share": min(shares),
        "max_top2_share": max(shares),
        "dead_top2_experts": sum(value <= 0.0 for value in hard),
        "affinity_mass_error": errors["affinity_mass"],
        "top1_count_error": errors["top1_count"],
        "top2_count_error": errors["top2_count"],
        "hard_count_error": errors["hard_count"],
        "selected_token_count_error": errors["selected_token_count"],
    }
    return info, profiles, metrics


def load_variant(plan_variant: dict[str, Any]) -> dict[str, Any]:
    variant_id = plan_variant["id"]
    manifest_path = ROOT / variant_id / "crucible_manifest" / "experiment_config.json"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != plan_variant["manifest_sha256"]:
        raise ValueError(f"manifest hash mismatch for {variant_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_path = ROOT / "full_run" / variant_id / "metrics.jsonl"
    metrics_sha = sha256_file(metrics_path)
    rows = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{variant_id}: non-object row {line_no}")
            rows.append(row)

    expected = {
        (seed, BRANCH, LR, checkpoint)
        for seed in SEEDS
        for checkpoint in CHECKPOINTS
    }
    seen = set()
    analyzed = []
    num_experts = int(plan_variant["num_experts"])
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["branch"]),
            float(row["peak_lr"]),
            int(row["interval_step_end"]),
        )
        if key not in expected or key in seen:
            raise ValueError(f"{variant_id}: invalid or duplicate matrix key {key}")
        seen.add(key)
        if row.get("variant_id") != variant_id:
            raise ValueError(f"{variant_id}: variant binding mismatch")
        if str(row.get("master_config_sha256", "")).lower() != manifest_sha[:16].lower():
            raise ValueError(f"{variant_id}: manifest binding mismatch")
        for field in (
            "interval_train_ce",
            "validation_nll",
            "byte_perplexity",
            "router_interval_cv",
            "router_interval_entropy",
            "router_interval_load_ratio",
        ):
            if field not in row or not finite(row[field]):
                raise ValueError(f"{variant_id}: invalid {field}")
        if int(row.get("interval_nan_inf_count", 0)) != 0:
            raise ValueError(f"{variant_id}: recorded nonfinite event")

        layer_results = {}
        for layer_idx in range(NUM_LAYERS):
            layer_key = f"layer_{layer_idx}_moe"
            info, profiles, metrics = analyze_layer(
                row["layer_telemetry"][layer_key], num_experts
            )
            layer_results[layer_key] = {
                **metrics,
                **info,
                "_profiles": profiles,
            }
        analyzed.append({
            "row": row,
            "layers": layer_results,
        })

    if seen != expected or len(rows) != len(expected):
        raise ValueError(f"{variant_id}: exact matrix mismatch")

    analyzed.sort(key=lambda item: (int(item["row"]["seed"]), int(item["row"]["interval_step_end"])))
    trajectories = []
    for seed in SEEDS:
        previous = {}
        for item in [entry for entry in analyzed if int(entry["row"]["seed"]) == seed]:
            row = item["row"]
            layer_summary = []
            for layer_idx in range(NUM_LAYERS):
                layer_key = f"layer_{layer_idx}_moe"
                result = item["layers"][layer_key]
                stability = None
                if layer_idx in previous:
                    stability = sum(
                        1.0 - jsd_bits(old, current)
                        for old, current in zip(previous[layer_idx], result["_profiles"])
                    ) / num_experts
                previous[layer_idx] = result["_profiles"]
                layer_summary.append({
                    key: value
                    for key, value in result.items()
                    if key != "_profiles"
                } | {"profile_stability_vs_previous": stability})
            trajectories.append({
                "seed": seed,
                "step": int(row["interval_step_end"]),
                "validation_nll": float(row["validation_nll"]),
                "byte_perplexity": float(row["byte_perplexity"]),
                "layers": layer_summary,
            })

    final = []
    for item in analyzed:
        row = item["row"]
        if int(row["interval_step_end"]) != CHECKPOINTS[-1]:
            continue
        layers = []
        for layer_idx in range(NUM_LAYERS):
            result = item["layers"][f"layer_{layer_idx}_moe"]
            layers.append({key: value for key, value in result.items() if key != "_profiles"})
        final.append({
            "seed": int(row["seed"]),
            "validation_nll": float(row["validation_nll"]),
            "byte_perplexity": float(row["byte_perplexity"]),
            "train_ce": float(row["interval_train_ce"]),
            "clipped_fraction": float(row["interval_clipped_fraction"]),
            "layers": layers,
        })

    mean_nll = sum(item["validation_nll"] for item in final) / len(final)
    integrity = {
        "rows": len(rows),
        "expected_rows": len(expected),
        "max_affinity_mass_error": max(
            item["layers"][f"layer_{layer_idx}_moe"]["affinity_mass_error"]
            for item in analyzed
            for layer_idx in range(NUM_LAYERS)
        ),
        "max_count_conservation_error": max(
            item["layers"][f"layer_{layer_idx}_moe"][key]
            for item in analyzed
            for layer_idx in range(NUM_LAYERS)
            for key in (
                "top1_count_error",
                "top2_count_error",
                "hard_count_error",
                "selected_token_count_error",
            )
        ),
        "dead_top2_expert_cells": sum(
            item["layers"][f"layer_{layer_idx}_moe"]["dead_top2_experts"]
            for item in analyzed
            for layer_idx in range(NUM_LAYERS)
        ),
        "recorded_nonfinite_events": 0,
    }
    return {
        "id": variant_id,
        "regime": plan_variant["regime"],
        "num_experts": num_experts,
        "top_k": int(plan_variant["top_k"]),
        "d_ff": int(plan_variant["d_ff"]),
        "parameter_count": int(plan_variant["parameter_count"]),
        "manifest_sha256": manifest_sha,
        "metrics_sha256": metrics_sha,
        "final_mean_validation_nll": mean_nll,
        "final_mean_byte_perplexity": sum(item["byte_perplexity"] for item in final) / len(final),
        "final_by_seed": final,
        "integrity": integrity,
        "trajectory": trajectories,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    if not ledger.get("completed"):
        raise ValueError("full-run ledger is not complete")
    if len(plan["variants"]) != 3:
        raise ValueError("unexpected variant count")

    variants = [load_variant(item) for item in plan["variants"]]
    reference_nll = next(item["final_mean_validation_nll"] for item in variants if item["id"] == "reference_e4")
    for item in variants:
        item["nll_improvement_vs_reference"] = reference_nll - item["final_mean_validation_nll"]
        for final in item["final_by_seed"]:
            final["mean_layer_cv"] = sum(layer["router_cv"] for layer in final["layers"]) / NUM_LAYERS
            final["max_layer_cv"] = max(layer["router_cv"] for layer in final["layers"])
            final["mean_layer_load_ratio"] = sum(layer["router_max_min_ratio"] for layer in final["layers"]) / NUM_LAYERS
            final["max_layer_load_ratio"] = max(layer["router_max_min_ratio"] for layer in final["layers"])
            final["min_top2_share"] = min(layer["min_top2_share"] for layer in final["layers"])

    config = {
        "protocol": "crucible_sparse_capacity_gate_postflight_v1",
        "scope": "fixed-active-compute E4/E8/E16 final and trajectory audit",
        "does_not_claim": [
            "semantic specialization",
            "syntactic specialization",
            "causal expert identity",
        ],
        "raw_byte_ids": "3..258; PAD excluded; BOS/EOS reported separately",
        "descriptive_null": "P0(E,T)=P(E)P(T)",
        "checkpoints": CHECKPOINTS,
        "seeds": SEEDS,
        "branch": BRANCH,
        "learning_rate": LR,
        "input_package_plan_sha256": sha256_file(PLAN_PATH),
        "input_ledger_sha256": sha256_file(LEDGER_PATH),
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "telemetry_schema": ledger.get("telemetry_schema"),
    }
    summary = {
        "protocol": config["protocol"],
        "config": config,
        "variants": variants,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    config_path = OUT / "analysis_config.json"
    summary_path = OUT / "sparse_capacity_summary.json"
    csv_path = OUT / "final_seed_layer_summary.csv"
    hashes_path = OUT / "analysis_output_hashes.json"
    write_json(config_path, config)
    write_json(summary_path, summary)

    fields = [
        "variant", "experts", "seed", "validation_nll", "byte_perplexity",
        "mean_layer_cv", "max_layer_cv", "mean_layer_load_ratio",
        "max_layer_load_ratio", "min_top2_share", "layer", "router_cv",
        "router_max_min_ratio", "mi_bits", "nmi_by_expert_entropy",
        "mean_pairwise_jsd_bits", "max_pairwise_jsd_bits",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant in variants:
            for final in variant["final_by_seed"]:
                for layer_index, layer in enumerate(final["layers"]):
                    writer.writerow({
                        "variant": variant["id"],
                        "experts": variant["num_experts"],
                        "seed": final["seed"],
                        "validation_nll": final["validation_nll"],
                        "byte_perplexity": final["byte_perplexity"],
                        "mean_layer_cv": final["mean_layer_cv"],
                        "max_layer_cv": final["max_layer_cv"],
                        "mean_layer_load_ratio": final["mean_layer_load_ratio"],
                        "max_layer_load_ratio": final["max_layer_load_ratio"],
                        "min_top2_share": final["min_top2_share"],
                        "layer": layer_index,
                        "router_cv": layer["router_cv"],
                        "router_max_min_ratio": layer["router_max_min_ratio"],
                        "mi_bits": layer["mi_bits"],
                        "nmi_by_expert_entropy": layer["nmi_by_expert_entropy"],
                        "mean_pairwise_jsd_bits": layer["mean_pairwise_jsd_bits"],
                        "max_pairwise_jsd_bits": layer["max_pairwise_jsd_bits"],
                    })

    artifact_hashes = {
        "analyzer_source_sha256": sha256_file(Path(__file__).resolve()),
        "package_plan_sha256": sha256_file(PLAN_PATH),
        "run_ledger_sha256": sha256_file(LEDGER_PATH),
        "analysis_config.json": sha256_file(config_path),
        "sparse_capacity_summary.json": sha256_file(summary_path),
        "final_seed_layer_summary.csv": sha256_file(csv_path),
        "input_metrics_sha256": {
            variant["id"]: variant["metrics_sha256"] for variant in variants
        },
    }
    write_json(hashes_path, artifact_hashes)
    print(json.dumps({"output_dir": str(OUT), "hashes": artifact_hashes}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
