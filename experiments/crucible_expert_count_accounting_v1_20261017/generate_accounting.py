#!/usr/bin/env python3
"""Generate a read-only parameter/FLOP/memory accounting for Crucible MoE variants.

This script reads the frozen 100K telemetry-replay manifest and a checkpoint
metadata/hash chain. It does not instantiate, modify, train, benchmark, or
save any model. All figures are analytical estimates under the formulas
recorded in the generated methodology document.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_MANIFEST = Path(
    "/home/timbushnell/crucible_checkpoint_ab_100k_v2_20261016/portable_manifest/experiment_config.json"
)
DEFAULT_CHECKPOINT_METADATA = Path(
    "/home/timbushnell/crucible_checkpoint_ab_100k_B42_20261017/checkpoints/checkpoint_final_step_3125.json"
)
DEFAULT_CHECKPOINT_HASHES = Path(
    "/home/timbushnell/crucible_checkpoint_ab_100k_B42_20261017/checkpoints/checkpoint_final_step_3125.hashes.json"
)
DEFAULT_MODEL_CORE = Path(
    "/home/timbushnell/crucible_checkpoint_ab_100k_v2_20261016/model_core.py"
)
MIB = 1024.0 * 1024.0
GIB = 1024.0 * MIB


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def round_int(value: float) -> int:
    rounded = int(round(value))
    if abs(value - rounded) > 1e-9:
        raise ValueError(f"expected integer-valued quantity, got {value}")
    return rounded


def pct_delta(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference


def model_quantities(hp: Mapping[str, Any]) -> Dict[str, int]:
    d = int(hp["d_model"])
    v = int(hp["vocab_size"])
    layers = int(hp["n_layers"])
    heads = int(hp["n_heads"])
    latent_kv = int(hp["latent_dim_kv"])
    latent_q = int(hp["latent_dim_q"])
    pe = int(hp["pe_dim"])
    if d % heads:
        raise ValueError("d_model is not divisible by n_heads")
    head_dim = d // heads

    attention_params = (
        d * latent_q
        + latent_q * (heads * head_dim)
        + latent_q * (heads * pe)
        + d * latent_kv
        + latent_kv * (heads * head_dim)
        + latent_kv * (heads * head_dim)
        + d * pe
        + (heads * head_dim) * d
    )
    global_params = 2 * v * d + d
    layer_dense_params = attention_params + 2 * d
    base_no_router = global_params + layers * layer_dense_params
    return {
        "d_model": d,
        "vocab_size": v,
        "n_layers": layers,
        "n_heads": heads,
        "latent_dim_kv": latent_kv,
        "latent_dim_q": latent_q,
        "pe_dim": pe,
        "head_dim": head_dim,
        "attention_params_per_layer": attention_params,
        "global_params": global_params,
        "layer_dense_params": layer_dense_params,
        "base_no_router_params": base_no_router,
    }


def variant_row(
    regime: str,
    expert_count: int,
    expert_width: int,
    q: Mapping[str, int],
    reference: Mapping[str, Any],
) -> Dict[str, Any]:
    d = q["d_model"]
    v = q["vocab_size"]
    layers = q["n_layers"]
    heads = q["n_heads"]
    context = int(reference["context_window"])
    microbatch = int(reference["microbatch_size"])
    latent_kv = q["latent_dim_kv"]
    pe = q["pe_dim"]
    k = min(2, expert_count)

    router_params = layers * d * expert_count
    expert_params_per_expert = 3 * d * expert_width
    expert_params_total = layers * expert_count * expert_params_per_expert
    total_params = q["base_no_router_params"] + router_params + expert_params_total
    nonexpert_params = q["base_no_router_params"] + router_params
    active_expert_params = layers * k * expert_params_per_expert
    active_params = nonexpert_params + active_expert_params

    # FLOP convention: one multiply and one add = 2 FLOPs.
    attention_linear_flops = 2 * q["attention_params_per_layer"]
    average_causal_attention_flops = 4 * d * (context + 1) / 2.0
    decode_attention_flops = 4 * d * context
    norm_flops_per_layer = 2 * 5 * d
    residual_flops_per_layer = 2 * d
    router_gate_flops = 2 * d * expert_count
    # Proxy for softmax, top-k selection and top-k renormalization arithmetic.
    router_elementwise_proxy = 8 * expert_count + 2 * k
    router_flops = router_gate_flops + router_elementwise_proxy
    expert_flops = k * 6 * d * expert_width
    weighted_combine_flops = 2 * k * d
    lm_head_flops = 2 * d * v
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

    # Dispatch is a memory movement/metadata cost, not a stable FLOP count.
    # Lower bound: one BF16 vector gather and one BF16 output accumulation
    # vector per selected expert stream, plus int64 expert IDs and fp32 weights.
    dispatch_vector_bytes_bf16 = 4 * k * d
    dispatch_metadata_bytes = k * (8 + 4)
    dispatch_lower_bound_bytes_bf16 = dispatch_vector_bytes_bf16 + dispatch_metadata_bytes

    # Memory planning proxy. This is intentionally labelled analytical, not
    # a measured allocator peak. It assumes FP32 parameters/AdamW state and
    # BF16 saved activations at the current microbatch/context.
    b = microbatch
    s = context
    h = heads
    activation_elements_training = (
        layers * (
            2 * k * b * s * expert_width
            + 8 * b * s * d
            + b * h * s * s
        )
        + b * s * d
    )
    activation_bytes_training_bf16 = 2 * activation_elements_training
    parameter_bytes_fp32 = total_params * 4
    training_persistent_bytes_fp32 = total_params * 16
    estimated_training_bytes = (
        training_persistent_bytes_fp32 + activation_bytes_training_bf16
    )

    inference_batch = 1
    activation_elements_inference = (
        2 * k * inference_batch * s * expert_width
        + 8 * inference_batch * s * d
        + inference_batch * h * s * s
        + inference_batch * s * d
    )
    inference_activation_bytes_bf16 = 2 * activation_elements_inference
    estimated_inference_bytes = parameter_bytes_fp32 + inference_activation_bytes_bf16
    generation_kv_cache_bytes_bf16 = (
        layers * inference_batch * s * (latent_kv + pe) * 2
    )

    result: Dict[str, Any] = {
        "regime": regime,
        "expert_count": expert_count,
        "top_k": k,
        "expert_hidden_width": expert_width,
        "total_trainable_parameters": total_params,
        "nonexpert_parameters": nonexpert_params,
        "expert_parameters_total": expert_params_total,
        "expert_parameters_per_expert_per_layer": expert_params_per_expert,
        "router_parameters": router_params,
        "active_expert_parameters": active_expert_params,
        "active_parameters": active_params,
        "total_parameter_delta_vs_reference": total_params - int(reference["total_parameters"]),
        "total_parameter_delta_pct_vs_reference": pct_delta(
            total_params, float(reference["total_parameters"])
        ),
        "active_parameter_delta_vs_reference": active_params - int(reference["active_parameters"]),
        "active_parameter_delta_pct_vs_reference": pct_delta(
            active_params, float(reference["active_parameters"])
        ),
        "forward_flops_per_token_avg_causal": forward_flops,
        "decode_flops_per_token_at_context_128": decode_flops,
        "training_flops_per_token_3x_forward": training_flops,
        "forward_flop_ratio_vs_reference": forward_flops / float(reference["forward_flops"]),
        "training_flop_ratio_vs_reference": training_flops / float(reference["training_flops"]),
        "expert_flops_per_token": expert_flops,
        "router_gate_flops_per_token": router_gate_flops,
        "router_elementwise_proxy_flops_per_token": router_elementwise_proxy,
        "dispatch_lower_bound_bytes_bf16_per_token": dispatch_lower_bound_bytes_bf16,
        "dispatch_metadata_bytes_per_token": dispatch_metadata_bytes,
        "active_expert_evaluations_per_token": k,
        "parameter_memory_fp32_mib": parameter_bytes_fp32 / MIB,
        "parameter_memory_bf16_mib_if_cast": total_params * 2 / MIB,
        "training_persistent_param_grad_adamw_fp32_mib": training_persistent_bytes_fp32 / MIB,
        "training_activation_proxy_bf16_mib": activation_bytes_training_bf16 / MIB,
        "estimated_training_memory_proxy_mib": estimated_training_bytes / MIB,
        "estimated_inference_memory_proxy_mib": estimated_inference_bytes / MIB,
        "generation_kv_cache_proxy_bf16_mib": generation_kv_cache_bytes_bf16 / MIB,
        "constraints": [],
    }
    return result


def format_num(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.3f}"
    return str(value)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields = [
        "regime",
        "expert_count",
        "top_k",
        "expert_hidden_width",
        "total_trainable_parameters",
        "active_parameters",
        "router_parameters",
        "active_expert_parameters",
        "total_parameter_delta_vs_reference",
        "total_parameter_delta_pct_vs_reference",
        "active_parameter_delta_vs_reference",
        "active_parameter_delta_pct_vs_reference",
        "forward_flops_per_token_avg_causal",
        "training_flops_per_token_3x_forward",
        "forward_flop_ratio_vs_reference",
        "training_flop_ratio_vs_reference",
        "expert_flops_per_token",
        "router_gate_flops_per_token",
        "dispatch_lower_bound_bytes_bf16_per_token",
        "active_expert_evaluations_per_token",
        "parameter_memory_fp32_mib",
        "training_persistent_param_grad_adamw_fp32_mib",
        "training_activation_proxy_bf16_mib",
        "estimated_training_memory_proxy_mib",
        "estimated_inference_memory_proxy_mib",
        "generation_kv_cache_proxy_bf16_mib",
        "constraints",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["constraints"] = "; ".join(item["constraints"])
            writer.writerow({field: item.get(field) for field in fields})


def methodology_markdown(
    manifest_path: Path,
    manifest_sha: str,
    model_core_sha: str,
    checkpoint_metadata_sha: str,
    checkpoint_sha: str,
    q: Mapping[str, int],
    reference: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Crucible expert-count parameter/FLOP accounting v1",
        "",
        "This is a read-only analytical study. It did not instantiate, modify, train, benchmark, or save a model.",
        "",
        "## Frozen reference",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Manifest SHA-256: `{manifest_sha}`",
        f"- Model-core SHA-256: `{model_core_sha}`",
        f"- Reference B/42 checkpoint metadata SHA-256: `{checkpoint_metadata_sha}`",
        f"- Reference B/42 checkpoint SHA-256: `{checkpoint_sha}`",
        f"- Geometry: `{q['n_layers']}` layers, `d_model={q['d_model']}`, `d_ff=3447`, `E=4`, `top_k=2`, `V={q['vocab_size']}`, context `128`.",
        "",
        "The current implementation uses four local experts, top-2 routing, no capacity limit, no token dropping, and no expert parallelism. Branch B adds its balancing loss during training; this accounting does not change or reimplement that loss.",
        "",
        "## Exact parameter formulas",
        "",
        "All linear layers have no bias. One SwiGLU expert has three matrices, so:",
        "",
        "```text",
        "P_expert_total_per_layer = E * 3 * d_model * d_ff",
        "P_router_per_layer = d_model * E",
        "P_attention_per_layer =",
        "    d*latent_q + latent_q*d + latent_q*(n_heads*pe_dim)",
        "  + d*latent_kv + 2*latent_kv*d + d*pe_dim + d*d",
        "P_total = 2*V*d + d_final_norm",
        "        + L*(P_attention + 2*d_norm + P_router + P_expert)",
        "P_active = P_nonexpert + L*top_k*3*d_model*d_ff",
        "```",
        "",
        "`P_active` uses the usual MoE accounting convention: all dense/non-expert parameters plus only the selected expert parameter sets. It is not a claim that every embedding row is touched by one token; the lookup-row nuance is excluded from the headline comparison.",
        "",
        "The reference decomposition is:",
        f"- attention weights per layer: `{q['attention_params_per_layer']:,}` parameters;",
        f"- non-expert base excluding routers: `{q['base_no_router_params']:,}` parameters;",
        f"- reference total: `{int(reference['total_parameters']):,}`;",
        f"- reference active path: `{int(reference['active_parameters']):,}`.",
        "",
        "Reference component breakdown:",
        f"- token embedding: `{2 * q['vocab_size'] * q['d_model'] // 2:,}`;",
        f"- output head: `{q['vocab_size'] * q['d_model']:,}`;",
        f"- final RMSNorm: `{q['d_model']:,}`;",
        f"- attention projections: `{q['n_layers'] * q['attention_params_per_layer']:,}`;",
        f"- per-layer RMSNorms: `{q['n_layers'] * 2 * q['d_model']:,}`;",
        f"- routers: `{q['n_layers'] * q['d_model'] * 4:,}`;",
        f"- experts: `{q['n_layers'] * 4 * 3 * q['d_model'] * 3447:,}`;",
        "",
        "## FLOP formulas and assumptions",
        "",
        "The convention is one multiply plus one add = 2 FLOPs. The reported sequence FLOP uses the average causal context for `S=128`; decode FLOPs use a full context-128 KV-cache step.",
        "",
        "```text",
        "SwiGLU expert evaluation = 6*d_model*d_ff FLOPs",
        "attention projections = 2*P_attention FLOPs",
        "causal attention score/value work = 4*d_model*(S+1)/2 FLOPs per token on average",
        "router gate = 2*d_model*E FLOPs",
        "router softmax/top-k/renormalization = documented 8*E + 2*top_k proxy",
        "expert weighted combine = 2*top_k*d_model FLOPs",
        "LM head = 2*d_model*V FLOPs",
        "training estimate = 3*forward FLOPs",
        "```",
        "",
        "The 3x training multiplier is a planning estimate for forward plus backward, not a measurement. Exact backward cost depends on kernels and autograd saves. RMSNorm and residual arithmetic are included with small fixed proxies. Sin/cos/RoPE, comparisons, indexing, and memory synchronization are not converted into invented FLOPs.",
        "",
        "Dispatch is reported separately because it is memory movement rather than stable arithmetic: a lower bound of `4*top_k*d_model` BF16 bytes per token for input/output vector movement, plus `top_k*(8+4)` bytes for int64 expert IDs and FP32 routing weights. The Python mask/gather implementation may exceed this lower bound.",
        "",
        "## Memory formulas and assumptions",
        "",
        "Runtime code keeps model parameters in FP32 and uses BF16 autocast. The training persistent estimate is `16*P` bytes: FP32 parameters, FP32 gradients, and two FP32 AdamW moments. The activation proxy assumes microbatch 4, context 128, BF16 saved activations, all layers retained for backward, two SwiGLU intermediates per active expert stream, and a conservative attention-score buffer. It is a planning proxy, not an allocator measurement.",
        "",
        "Inference memory is FP32 parameter memory plus a one-batch activation proxy. A separate generation KV-cache proxy uses the implementation's compressed KV and positional-key widths. CUDA allocator reserve, kernel workspaces, fragmentation, and telemetry buffers are excluded.",
        "",
        "## Regime definitions",
        "",
        "### A — fixed total parameters",
        "",
        "Expert width is chosen so `E*d_ff` stays equal to the reference. The only total-parameter mismatch is the router output width: fewer experts mean fewer gate weights. This is below 0.01% here, and is reported rather than hidden.",
        "",
        "### B — fixed active expert compute",
        "",
        "Expert width is chosen so `top_k*d_ff` stays equal to the reference. E=2 with top-2 activates both experts; E=1 necessarily uses top-1. Expert arithmetic is therefore matched, while total capacity and router/dispatch overhead differ.",
        "",
        "### C — fixed model geometry",
        "",
        "The E=4, d_ff=3447, top-2 Branch B model is unchanged. E=1/2 rows retain d_ff=3447 but are not fair capacity-matched controls: they deliberately have less total expert capacity, and E=1 also has half the active expert compute.",
        "",
        "## Implementation feasibility",
        "",
        "The constructor accepts a different expert count and width, but the current forward path hard-codes `torch.topk(..., 2)`. E=2 and E=4 are structurally compatible with top-2; E=1 requires a future generalized top-k path and corresponding telemetry/auxiliary-loss handling. This study does not make that code change.",
        "",
        "Non-expert dimensions can remain unchanged: embeddings, attention, RMSNorms, output head, context, and optimizer schedule. The dimensions that must change are expert count, router output width, expert hidden width, and top-k for E=1. Branch-B balancing semantics must be explicitly generalized before training variants.",
        "",
        "## Table",
        "",
        "See `comparison_table.csv` and `comparison_table.json` for machine-readable values. The table intentionally exposes parameter, active-compute, router, dispatch, and memory mismatches separately.",
        "",
    ]
    return "\n".join(lines)


def feasibility_markdown(rows: Sequence[Mapping[str, Any]], reference: Mapping[str, Any]) -> str:
    widest = max(rows, key=lambda row: row["estimated_training_memory_proxy_mib"])
    cheapest = min(rows, key=lambda row: row["training_flop_ratio_vs_reference"])
    return "\n".join(
        [
            "# Crucible expert-count feasibility and recommended experiment",
            "",
            "## Fairness conclusion",
            "",
            "Fair comparisons are technically feasible as a design study, but the current runner is not yet a drop-in implementation for E=1 because its routing path hard-codes top-2. The future implementation must generalize top-k and Branch-B auxiliary-loss normalization before any variant is trained.",
            "",
            "The strongest matched comparisons are:",
            "",
            "1. Fixed active expert compute: E=1/K=1/d_ff=6894, E=2/K=2/d_ff=3447, and the unchanged E=4/K=2/d_ff=3447 reference. This holds expert arithmetic approximately exactly; total parameter capacity differs by design.",
            "2. Fixed total expert capacity: E=1/K=1/d_ff=13788, E=2/K=2/d_ff=6894, and E=4/K=2/d_ff=3447. This holds total expert parameters exactly; active expert compute is approximately 2x the reference for E=1/2.",
            "3. Fixed geometry: useful as a capacity ablation, not as a fair compute or parameter comparison.",
            "",
            "## Main confounders",
            "",
            "- E=1 cannot use top-2 and therefore changes routing semantics and Branch-B auxiliary-loss behavior.",
            "- E=2/top-2 activates every expert, so it has no sparse expert selection even when its active expert FLOPs are matched to E=4/top-2.",
            "- Router arithmetic, top-k comparisons, dispatch/gather/scatter, and Python mask overhead vary with E and K; they are not captured by expert GEMM FLOPs alone.",
            "- The analytical training FLOPs and activation memory are estimates, not GPU measurements. Kernel choice, allocator fragmentation, and padding alter realized cost.",
            "- Fixed total parameters and fixed active compute deliberately answer different questions; neither alone identifies the full causal effect of expert count.",
            "- The dataset/model may be under-complex for specialization; a TinyStories result does not establish behavior on diverse or harder data.",
            "",
            "## Recommended minimal experiment",
            "",
            "First implement and preflight a generalized top-k model without changing the E=4 reference path. Then run the crossed six-condition design: the three fixed-total rows and three fixed-active-compute rows, sharing the unchanged E=4 reference where identical. Use at least two seeds for an exploratory minimum; three seeds are preferable if the differences are small. Keep the same packed manifest, one corpus pass, optimizer schedule, checkpoint intervals, and validation split.",
            "",
            "Primary metrics:",
            "- validation NLL and bits per raw byte;",
            "- total/active parameter counts and realized peak VRAM;",
            "- measured tokens/sec and optimizer-step time;",
            "- forward/backward wall time separated from validation;",
            "- router CV, load ratio, top-k entropy, and dropped-token count (expected zero);",
            "- gradient norm/clipping and nonfinite events.",
            "",
            "Acceptance criteria should be frozen before training: designated fixed-total rows within 0.01% total parameters; designated fixed-active rows within 0.01% expert arithmetic FLOPs; exact zero capacity drops; identical data/schedule/seed protocol; and conclusions reported as seed-level estimates rather than formal equivalence claims with only two seeds.",
            "",
            "## RTX 3060 Ti assessment",
            "",
            f"The widest analytical training-memory row is `{widest['regime']}/E{widest['expert_count']}` at approximately `{widest['estimated_training_memory_proxy_mib']:.1f}` MiB under the stated proxy. The lowest estimated forward-cost row is `{cheapest['regime']}/E{cheapest['expert_count']}` at `{cheapest['training_flop_ratio_vs_reference']:.3f}x` the reference. These are not measured peaks. Historical Crucible runs used roughly 2.3 GiB on the 8 GiB RTX 3060 Ti, leaving substantial headroom, but each future variant must record actual peak allocation. No benchmarking was performed for this study.",
            "",
            "No training, checkpoint creation, GPU benchmark, or rental is authorized by this accounting artifact.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint-metadata", type=Path, default=DEFAULT_CHECKPOINT_METADATA)
    parser.add_argument("--checkpoint-hashes", type=Path, default=DEFAULT_CHECKPOINT_HASHES)
    parser.add_argument("--model-core", type=Path, default=DEFAULT_MODEL_CORE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to write nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_sha = sha256_file(args.manifest)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_manifest_sha = manifest["meta"].get("manifest_sha256")
    if expected_manifest_sha is not None and expected_manifest_sha != manifest_sha:
        raise RuntimeError("manifest self-hash field does not match file hash")
    hp = manifest["hyperparameters"]
    q = model_quantities(hp)
    ref_e = int(hp["num_experts"])
    ref_f = int(hp["d_ff"])
    ref_k = 2
    ref_total = q["base_no_router_params"] + q["n_layers"] * q["d_model"] * ref_e + q["n_layers"] * ref_e * 3 * q["d_model"] * ref_f
    ref_nonexpert = q["base_no_router_params"] + q["n_layers"] * q["d_model"] * ref_e
    ref_active = ref_nonexpert + q["n_layers"] * ref_k * 3 * q["d_model"] * ref_f
    expected_parameter_geometry = int(hp["expected_parameter_geometry"])
    if ref_total != expected_parameter_geometry:
        raise RuntimeError(f"formula/reference parameter mismatch: {ref_total} != {expected_parameter_geometry}")

    checkpoint_metadata = json.loads(args.checkpoint_metadata.read_text(encoding="utf-8"))
    checkpoint_hashes = json.loads(args.checkpoint_hashes.read_text(encoding="utf-8"))
    checkpoint_sha = checkpoint_hashes["checkpoint_sha256"]
    if checkpoint_metadata["checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("checkpoint metadata/hash-chain mismatch")
    if checkpoint_metadata["manifest_sha256"] != manifest_sha:
        raise RuntimeError("checkpoint/manifest binding mismatch")
    if int(checkpoint_metadata["model_parameter_count"]) != ref_total:
        raise RuntimeError("checkpoint parameter count mismatch")

    # Seed the reference FLOP denominators with nonzero placeholders for the
    # first exact reference-row calculation, then replace them with the
    # calculated values before generating all public rows.
    reference = {
        "total_parameters": ref_total,
        "active_parameters": ref_active,
        "forward_flops": 1.0,
        "training_flops": 1.0,
        "context_window": int(hp["context_window_length"]),
        "microbatch_size": int(hp["microbatch_size"]),
    }
    reference_row = variant_row("fixed_geometry", ref_e, ref_f, q, reference)

    reference_components = {
        "token_embedding_parameters": q["vocab_size"] * q["d_model"],
        "output_head_parameters": q["vocab_size"] * q["d_model"],
        "final_rmsnorm_parameters": q["d_model"],
        "attention_projection_parameters": q["n_layers"] * q["attention_params_per_layer"],
        "per_layer_rmsnorm_parameters": q["n_layers"] * 2 * q["d_model"],
        "router_parameters": q["n_layers"] * q["d_model"] * ref_e,
        "expert_parameters_total": q["n_layers"] * ref_e * 3 * q["d_model"] * ref_f,
    }
    reference_components["total"] = sum(reference_components.values())
    if reference_components["total"] != ref_total:
        raise RuntimeError("reference component breakdown does not sum to total")
    reference_components["active_expert_parameters_top2"] = (
        q["n_layers"] * ref_k * 3 * q["d_model"] * ref_f
    )
    reference_components["active_total_top2"] = (
        ref_total - reference_components["expert_parameters_total"]
        + reference_components["active_expert_parameters_top2"]
    )
    reference["forward_flops"] = reference_row["forward_flops_per_token_avg_causal"]
    reference["training_flops"] = reference_row["training_flops_per_token_3x_forward"]
    reference_row = variant_row("fixed_geometry", ref_e, ref_f, q, reference)

    configs = [
        ("fixed_total_parameters", 1, ref_e * ref_f // 1),
        ("fixed_total_parameters", 2, ref_e * ref_f // 2),
        ("fixed_total_parameters", 4, ref_f),
        ("fixed_active_compute", 1, ref_k * ref_f // 1),
        ("fixed_active_compute", 2, ref_k * ref_f // 2),
        ("fixed_active_compute", 4, ref_f),
        ("fixed_geometry", 1, ref_f),
        ("fixed_geometry", 2, ref_f),
        ("fixed_geometry", 4, ref_f),
    ]
    rows: List[Dict[str, Any]] = []
    for regime, expert_count, width in configs:
        row = variant_row(regime, expert_count, width, q, reference)
        if regime == "fixed_total_parameters":
            row["constraints"].append("E*d_ff equals reference expert capacity; router width changes with E")
            if expert_count in {1, 2}:
                row["constraints"].append("active expert FLOPs are approximately 2x reference because top_k*d_ff doubles")
        elif regime == "fixed_active_compute":
            row["constraints"].append("top_k*d_ff equals reference expert arithmetic")
            row["constraints"].append("E=1 uses top-1; E=2/top-2 activates both experts")
        else:
            row["constraints"].append("d_ff and non-expert geometry unchanged; total capacity varies with E")
            if expert_count == 1:
                row["constraints"].append("E=1 uses top-1 and has half reference active expert arithmetic")
        rows.append(row)

    if rows[2]["total_trainable_parameters"] != ref_total:
        raise RuntimeError("reference row failed exact parameter check")
    for row in rows:
        if row["regime"] == "fixed_total_parameters" and row["expert_count"] in {1, 2, 4}:
            # Router-width mismatch is intentionally reported; expert capacity is exact.
            pass
        if not all(math.isfinite(float(value)) for value in [
            row["forward_flops_per_token_avg_causal"],
            row["training_flops_per_token_3x_forward"],
            row["estimated_training_memory_proxy_mib"],
        ]):
            raise RuntimeError("nonfinite accounting result")

    methodology = methodology_markdown(
        args.manifest,
        manifest_sha,
        sha256_file(args.model_core),
        sha256_file(args.checkpoint_metadata),
        checkpoint_sha,
        q,
        reference,
        rows,
    )
    (args.output_dir / "accounting_methodology.md").write_text(methodology, encoding="utf-8")
    feasibility = feasibility_markdown(rows, reference)
    (args.output_dir / "experiment_feasibility.md").write_text(feasibility, encoding="utf-8")
    write_csv(args.output_dir / "comparison_table.csv", rows)

    report = {
        "protocol": "crucible.expert_count_parameter_flop_accounting.v1",
        "status": "PASS",
        "read_only": True,
        "no_training_or_benchmarking": True,
        "inputs": {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": manifest_sha,
            "model_core_path": str(args.model_core.resolve()),
            "model_core_sha256": sha256_file(args.model_core),
            "reference_checkpoint_metadata_path": str(args.checkpoint_metadata.resolve()),
            "reference_checkpoint_metadata_sha256": sha256_file(args.checkpoint_metadata),
            "reference_checkpoint_path": str(args.checkpoint_metadata.with_suffix(".pt").resolve()),
            "reference_checkpoint_sha256": checkpoint_sha,
            "reference_checkpoint_hash_manifest_path": str(args.checkpoint_hashes.resolve()),
            "reference_checkpoint_hash_manifest_sha256": sha256_file(args.checkpoint_hashes),
        },
        "reference_geometry": {
            **q,
            "num_experts": ref_e,
            "top_k": ref_k,
            "d_ff": ref_f,
            "total_trainable_parameters": ref_total,
            "active_parameters": ref_active,
        },
        "reference_component_breakdown": reference_components,
        "assumptions": {
            "flop_convention": "multiply plus add equals 2 FLOPs",
            "causal_context_length": int(hp["context_window_length"]),
            "training_flop_multiplier": 3.0,
            "runtime_parameter_dtype": "FP32",
            "autocast_activation_dtype": "BF16",
            "training_microbatch_size": int(hp["microbatch_size"]),
            "dispatch_accounting": "lower-bound vector/metadata bytes, not fabricated FLOPs",
            "capacity_factor": None,
            "token_dropping": False,
            "expert_parallelism": False,
        },
        "comparison_rows": rows,
        "feasibility": {
            "technically_feasible_as_design": True,
            "current_runner_drop_in_ready": False,
            "blocker": "ResearchMoELayer.forward hard-codes top-2; E=1 needs generalized top-k and auxiliary-loss/telemetry handling.",
            "recommended_minimum_seeds": 2,
            "preferred_seeds": 3,
            "recommended_design": "crossed fixed-total and fixed-active-compute six-condition experiment with shared unchanged E=4 reference",
            "gpu": "RTX 3060 Ti 8 GiB should fit analytically; actual peak VRAM and throughput must be measured in a separately authorized preflight.",
        },
    }
    atomic_json(args.output_dir / "comparison_table.json", report)
    atomic_json(
        args.output_dir / "reference_component_breakdown.json",
        reference_components,
    )

    artifacts = [
        args.output_dir / "accounting_methodology.md",
        args.output_dir / "experiment_feasibility.md",
        args.output_dir / "comparison_table.csv",
        args.output_dir / "comparison_table.json",
        args.output_dir / "reference_component_breakdown.json",
    ]
    output_hashes = {
        "protocol": report["protocol"],
        "generator_path": str(SCRIPT_PATH),
        "generator_sha256": sha256_file(SCRIPT_PATH),
        "manifest_sha256": manifest_sha,
        "model_core_sha256": sha256_file(args.model_core),
        "reference_checkpoint_sha256": checkpoint_sha,
        "artifacts_excluding_this_file": {
            artifact.name: sha256_file(artifact) for artifact in artifacts
        },
        "self_hash_note": "This file records all generated accounting artifacts except its own self-referential digest.",
    }
    atomic_json(args.output_dir / "accounting_output_hashes.json", output_hashes)

    print(json.dumps({
        "status": "PASS",
        "output_dir": str(args.output_dir.resolve()),
        "manifest_sha256": manifest_sha,
        "generator_sha256": sha256_file(SCRIPT_PATH),
        "reference_parameters": ref_total,
        "row_count": len(rows),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
