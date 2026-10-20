"""Freeze Crucible expert-count accounting for E=1,2,4,8,16,32,64.

This is an analytical, read-only design artifact.  It does not import the
model, allocate a GPU, benchmark kernels, create checkpoints, or train.
The formulas intentionally match the validated v1 accounting generator.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parent
V1_ROOT = Path("/home/timbushnell/crucible_expert_count_accounting_v1_20261017")
V1_JSON = V1_ROOT / "artifacts_v2" / "comparison_table.json"
V1_GENERATOR = V1_ROOT / "generate_accounting.py"
PREFLIGHT_ROOT = Path("/home/timbushnell/crucible_topk_preflight_v1_20261017")
PREFLIGHT_RESULTS = PREFLIGHT_ROOT / "preflight_results.json"
GENERALIZED_MODEL = PREFLIGHT_ROOT / "model_core_generalized.py"
OUT = ROOT / "artifacts_v1"
MIB = 1024.0 * 1024.0

EXPERT_COUNTS = (1, 2, 4, 8, 16, 32, 64)
REFERENCE_EXPERTS = 4
REFERENCE_WIDTH = 3447
REFERENCE_TOP_K = 2
MEAN_PACKED_BYTES = 98.565
EFFECTIVE_BATCH = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct_delta(value: float, reference: float) -> float:
    return (value / reference - 1.0) * 100.0


def make_row(
    regime: str,
    experts: int,
    width: int,
    reference: Dict[str, Any],
    q: Dict[str, int],
    assumptions: Dict[str, Any],
) -> Dict[str, Any]:
    d = q["d_model"]
    vocab = q["vocab_size"]
    layers = q["n_layers"]
    heads = q["n_heads"]
    context = assumptions["causal_context_length"]
    microbatch = assumptions["training_microbatch_size"]
    latent_kv = q["latent_dim_kv"]
    pe = q["pe_dim"]
    top_k = min(2, experts)

    router_parameters = layers * d * experts
    expert_parameters_per_expert_per_layer = 3 * d * width
    expert_parameters_total = layers * experts * expert_parameters_per_expert_per_layer
    total_parameters = q["base_no_router_params"] + router_parameters + expert_parameters_total
    nonexpert_parameters = q["base_no_router_params"] + router_parameters
    active_expert_parameters = layers * top_k * expert_parameters_per_expert_per_layer
    active_parameters = nonexpert_parameters + active_expert_parameters

    attention_linear_flops = 2 * q["attention_params_per_layer"]
    average_causal_attention_flops = 4 * d * (context + 1) / 2.0
    decode_attention_flops = 4 * d * context
    norm_flops_per_layer = 2 * 5 * d
    residual_flops_per_layer = 2 * d
    router_gate_flops = 2 * d * experts
    router_elementwise_flops = 8 * experts + 2 * top_k
    router_flops = router_gate_flops + router_elementwise_flops
    expert_flops = top_k * 6 * d * width
    weighted_combine_flops = 2 * top_k * d
    lm_head_flops = 2 * d * vocab
    final_norm_flops = 5 * d

    layer_forward_flops = (
        attention_linear_flops
        + average_causal_attention_flops
        + norm_flops_per_layer
        + residual_flops_per_layer
        + router_flops
        + expert_flops
        + weighted_combine_flops
    )
    layer_decode_flops = (
        attention_linear_flops
        + decode_attention_flops
        + norm_flops_per_layer
        + residual_flops_per_layer
        + router_flops
        + expert_flops
        + weighted_combine_flops
    )
    forward_flops = layers * layer_forward_flops + lm_head_flops + final_norm_flops
    decode_flops = layers * layer_decode_flops + lm_head_flops + final_norm_flops
    training_flops = 3.0 * forward_flops

    dispatch_vector_bytes = 4 * top_k * d
    dispatch_metadata_bytes = top_k * (8 + 4)
    dispatch_lower_bound_bytes = dispatch_vector_bytes + dispatch_metadata_bytes

    b = microbatch
    s = context
    activation_elements_training = (
        layers * (
            2 * top_k * b * s * width
            + 8 * b * s * d
            + b * heads * s * s
        )
        + b * s * d
    )
    activation_bytes_training = 2 * activation_elements_training
    parameter_bytes_fp32 = total_parameters * 4
    training_persistent_bytes = total_parameters * 16
    estimated_training_bytes = training_persistent_bytes + activation_bytes_training

    inference_activation_elements = (
        2 * top_k * s * width
        + 8 * s * d
        + heads * s * s
        + s * d
    )
    inference_activation_bytes = 2 * inference_activation_elements
    estimated_inference_bytes = parameter_bytes_fp32 + inference_activation_bytes
    generation_kv_cache_bytes = layers * s * (latent_kv + pe) * 2

    ideal_full_context_assignments = EFFECTIVE_BATCH * context * top_k / experts
    ideal_mean_length_assignments = EFFECTIVE_BATCH * MEAN_PACKED_BYTES * top_k / experts
    capacity_match = experts * width == REFERENCE_EXPERTS * REFERENCE_WIDTH
    active_expert_compute_match = top_k * width == REFERENCE_TOP_K * REFERENCE_WIDTH

    constraints: List[str] = []
    if regime == "fixed_total_parameters":
        constraints.append("E*d_ff uses integer floor of reference expert capacity")
        constraints.append("router parameter count changes with E")
        if not active_expert_compute_match:
            constraints.append("active expert arithmetic differs from E=4 reference")
    elif regime == "fixed_active_compute":
        constraints.append("top_k*d_ff matches reference expert arithmetic")
        constraints.append("total capacity and router parameters grow with E")
        if experts == 1:
            constraints.append("E=1 is top-1 and is not genuinely sparse")
        if experts == 2:
            constraints.append("E=2/top-2 activates every expert")
    else:
        constraints.append("d_ff and non-expert geometry remain fixed")
        constraints.append("total capacity grows with E")
        if experts == 1:
            constraints.append("E=1 is top-1 and has half reference active expert arithmetic")

    return {
        "regime": regime,
        "expert_count": experts,
        "top_k": top_k,
        "expert_hidden_width": width,
        "total_trainable_parameters": total_parameters,
        "nonexpert_parameters": nonexpert_parameters,
        "router_parameters": router_parameters,
        "expert_parameters_total": expert_parameters_total,
        "expert_parameters_per_expert_per_layer": expert_parameters_per_expert_per_layer,
        "active_expert_parameters": active_expert_parameters,
        "active_parameters": active_parameters,
        "total_parameter_delta_vs_reference": total_parameters - reference["total_parameters"],
        "total_parameter_delta_pct_vs_reference": pct_delta(total_parameters, reference["total_parameters"]),
        "active_parameter_delta_vs_reference": active_parameters - reference["active_parameters"],
        "active_parameter_delta_pct_vs_reference": pct_delta(active_parameters, reference["active_parameters"]),
        "expert_capacity_match": capacity_match,
        "active_expert_arithmetic_match": active_expert_compute_match,
        "total_parameter_match_within_0_01_pct": abs(
            pct_delta(total_parameters, reference["total_parameters"])
        ) <= 0.01,
        "forward_flops_per_token_avg_causal": forward_flops,
        "decode_flops_per_token_at_context_128": decode_flops,
        "training_flops_per_token_3x_forward": training_flops,
        "forward_flop_ratio_vs_reference": forward_flops / reference["forward_flops"],
        "training_flop_ratio_vs_reference": training_flops / reference["training_flops"],
        "expert_flops_per_token": expert_flops,
        "router_gate_flops_per_token": router_gate_flops,
        "router_elementwise_proxy_flops_per_token": router_elementwise_flops,
        "router_total_proxy_flops_per_token": router_flops,
        "dispatch_lower_bound_bytes_bf16_per_token": dispatch_lower_bound_bytes,
        "dispatch_metadata_bytes_per_token": dispatch_metadata_bytes,
        "active_expert_evaluations_per_token": top_k,
        "ideal_selected_assignments_per_expert_per_update_full_context": ideal_full_context_assignments,
        "ideal_selected_assignments_per_expert_per_update_mean_packed_length": ideal_mean_length_assignments,
        "parameter_memory_fp32_mib": parameter_bytes_fp32 / MIB,
        "parameter_memory_bf16_mib_if_cast": total_parameters * 2 / MIB,
        "training_persistent_param_grad_adamw_fp32_mib": training_persistent_bytes / MIB,
        "training_activation_proxy_bf16_mib": activation_bytes_training / MIB,
        "estimated_training_memory_proxy_mib": estimated_training_bytes / MIB,
        "estimated_inference_memory_proxy_mib": estimated_inference_bytes / MIB,
        "generation_kv_cache_proxy_bf16_mib": generation_kv_cache_bytes / MIB,
        "constraints": constraints,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields = [
        "regime", "expert_count", "top_k", "expert_hidden_width",
        "total_trainable_parameters", "active_parameters",
        "expert_flops_per_token", "router_total_proxy_flops_per_token",
        "training_flops_per_token_3x_forward", "training_flop_ratio_vs_reference",
        "ideal_selected_assignments_per_expert_per_update_full_context",
        "ideal_selected_assignments_per_expert_per_update_mean_packed_length",
        "estimated_training_memory_proxy_mib", "estimated_inference_memory_proxy_mib",
        "total_parameter_delta_pct_vs_reference", "active_parameter_delta_pct_vs_reference",
        "expert_capacity_match", "active_expert_arithmetic_match",
        "total_parameter_match_within_0_01_pct", "constraints",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["constraints"] = " | ".join(output["constraints"])
            writer.writerow({field: output.get(field) for field in fields})


def main() -> int:
    source = json.loads(V1_JSON.read_text(encoding="utf-8"))
    reference_geometry = source["reference_geometry"]
    assumptions_v1 = source["assumptions"]
    q = {
        "d_model": reference_geometry["d_model"],
        "vocab_size": reference_geometry["vocab_size"],
        "n_layers": reference_geometry["n_layers"],
        "n_heads": reference_geometry["n_heads"],
        "latent_dim_kv": reference_geometry["latent_dim_kv"],
        "pe_dim": reference_geometry["pe_dim"],
        "attention_params_per_layer": reference_geometry["attention_params_per_layer"],
        "base_no_router_params": reference_geometry["base_no_router_params"],
    }
    reference_row = next(
        row for row in source["comparison_rows"]
        if row["regime"] == "fixed_geometry" and row["expert_count"] == 4
    )
    reference = {
        "total_parameters": reference_geometry["total_trainable_parameters"],
        "active_parameters": reference_geometry["active_parameters"],
        "forward_flops": reference_row["forward_flops_per_token_avg_causal"],
        "training_flops": reference_row["training_flops_per_token_3x_forward"],
    }
    assumptions = {
        "causal_context_length": assumptions_v1["causal_context_length"],
        "training_microbatch_size": assumptions_v1["training_microbatch_size"],
        "effective_batch_size": EFFECTIVE_BATCH,
        "mean_packed_source_bytes_per_example_planning_proxy": MEAN_PACKED_BYTES,
        "runtime_parameter_dtype": assumptions_v1["runtime_parameter_dtype"],
        "autocast_activation_dtype": assumptions_v1["autocast_activation_dtype"],
        "training_flop_multiplier": assumptions_v1["training_flop_multiplier"],
        "capacity_factor": None,
        "token_dropping": False,
        "expert_parallelism": False,
        "dispatch_accounting": "lower-bound vector/metadata bytes, not measured runtime traffic",
        "note": "Mean packed length is a planning proxy; per-expert exposure is ideal balanced exposure, not observed routing.",
    }

    rows: List[Dict[str, Any]] = []
    for experts in EXPERT_COUNTS:
        rows.append(
            make_row(
                "fixed_total_parameters",
                experts,
                (REFERENCE_EXPERTS * REFERENCE_WIDTH) // experts,
                reference,
                q,
                assumptions,
            )
        )
    for experts in EXPERT_COUNTS:
        width = REFERENCE_TOP_K * REFERENCE_WIDTH if experts == 1 else REFERENCE_WIDTH
        rows.append(
            make_row(
                "fixed_active_compute",
                experts,
                width,
                reference,
                q,
                assumptions,
            )
        )
    for experts in EXPERT_COUNTS:
        rows.append(
            make_row(
                "fixed_geometry",
                experts,
                REFERENCE_WIDTH,
                reference,
                q,
                assumptions,
            )
        )

    result = {
        "protocol": "crucible.expert_count_parameter_flop_accounting.v2_extended_1_to_64",
        "status": "PASS",
        "read_only": True,
        "training_started": False,
        "benchmarking_performed": False,
        "reference_geometry": reference_geometry,
        "assumptions": assumptions,
        "formulas": {
            "top_k": "min(2, E)",
            "total_parameters": "base_no_router + L*d_model*E + L*E*3*d_model*d_ff",
            "active_parameters": "base_no_router + L*d_model*E + L*top_k*3*d_model*d_ff",
            "expert_flops_per_token": "top_k*6*d_model*d_ff",
            "router_gate_flops_per_token": "2*d_model*E",
            "router_elementwise_proxy": "8*E + 2*top_k",
            "training_flops": "3*forward_flops",
            "ideal_exposure": "effective_batch*context_length*top_k/E",
        },
        "inputs": {
            "v1_comparison_table_sha256": sha256_file(V1_JSON),
            "v1_generator_sha256": sha256_file(V1_GENERATOR),
            "topk_preflight_results_sha256": sha256_file(PREFLIGHT_RESULTS),
            "generalized_model_core_sha256": sha256_file(GENERALIZED_MODEL),
        },
        "comparison_rows": rows,
        "interpretation": {
            "fixed_total": "Increasing E redistributes fixed expert capacity and reduces active expert width/compute beyond E=4.",
            "fixed_active": "Increasing E preserves expert arithmetic while increasing total parameters and router width.",
            "fixed_geometry": "Increasing E preserves expert width while increasing total capacity; this is not a matched-capacity regime.",
            "next_gate": "Train E=1/2/4 in crossed fixed-total and fixed-active regimes after generalized top-k preflight; treat E=8 as the first expansion.",
            "not_established": "No row establishes learned quality, specialization, throughput, or actual peak memory.",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "comparison_table_extended.json"
    csv_path = OUT / "comparison_table_extended.csv"
    methodology_path = OUT / "accounting_methodology_extended.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    methodology_path.write_text(
        "# Crucible expert-count accounting v2: E=1,2,4,8,16,32,64\n\n"
        "This is an analytical read-only extension of the validated v1 accounting. "
        "It performs no training or benchmarking.\n\n"
        f"Source v1 comparison SHA-256: `{sha256_file(V1_JSON)}`\n\n"
        f"Generator SHA-256: `{sha256_file(Path(__file__).resolve())}`\n\n"
        "The exact formulas, assumptions, 21 comparison rows, integer-width "
        "mismatches, router proxies, exposure proxies, and memory proxies are in "
        "`comparison_table_extended.json`. Exposure values assume ideal balanced "
        "routing and must not be interpreted as measured training utilization.\n",
        encoding="utf-8",
    )
    hashes = {
        "generate_extended_accounting.py": sha256_file(Path(__file__).resolve()),
        "comparison_table_extended.json": sha256_file(json_path),
        "comparison_table_extended.csv": sha256_file(csv_path),
        "accounting_methodology_extended.md": sha256_file(methodology_path),
        "source_v1_comparison_table.json": sha256_file(V1_JSON),
        "source_v1_generator.py": sha256_file(V1_GENERATOR),
        "topk_preflight_results.json": sha256_file(PREFLIGHT_RESULTS),
        "model_core_generalized.py": sha256_file(GENERALIZED_MODEL),
    }
    hashes_path = OUT / "accounting_output_hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "hashes": hashes}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
