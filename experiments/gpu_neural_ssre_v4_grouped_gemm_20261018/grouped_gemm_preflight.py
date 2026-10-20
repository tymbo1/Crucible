#!/usr/bin/env python3
"""GPU Neural SSRE v4 grouped-GEMM execution-engine preflight."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F


V1_DIR = Path("/home/timbushnell/gpu_neural_ssre_v1_grouped_expert_20261018")
V2_DIR = Path("/home/timbushnell/gpu_neural_ssre_v2_strategy_landscape_20261018")
CONTEXT_ROOT = Path("/home/timbushnell/crucible_context_gate_v1_20261018/context_128")
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(V2_DIR))

import dispatch_strategy_landscape as v2  # noqa: E402
import grouped_expert_benchmark as v1  # noqa: E402


SCHEMA_VERSION = "gpu-neural-ssre-v4-grouped-gemm-20261018"
FORWARD_TOL = float(v1.FORWARD_TOL)
GRAD_TOL = float(v1.GRAD_TOL)
VARIANTS = ("cpu_grouped_reference", "gpu_grouped_gemm")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_ns() -> int:
    return time.perf_counter_ns()


def gpu_grouped_assignments(
    indices: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
) -> Dict[str, torch.Tensor]:
    token_count, top_k = indices.shape
    flat_experts = indices.reshape(-1)
    flat_weights = weights.reshape(-1)
    flat_ranks = (
        torch.arange(top_k, device=indices.device)
        .view(1, -1)
        .expand(token_count, -1)
        .reshape(-1)
    )
    flat_tokens = (
        torch.arange(token_count, device=indices.device)
        .view(-1, 1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    pair_ids = flat_experts * top_k + flat_ranks
    order = torch.argsort(
        pair_ids * token_count + flat_tokens,
        stable=True,
    )
    sorted_pair_ids = pair_ids.index_select(0, order)
    sorted_tokens = flat_tokens.index_select(0, order)
    sorted_weights = flat_weights.index_select(0, order)
    pair_counts = torch.bincount(
        pair_ids,
        minlength=num_experts * top_k,
    )
    active_pair_ids = torch.nonzero(pair_counts > 0, as_tuple=False).reshape(-1)
    active_counts = pair_counts.index_select(0, active_pair_ids)
    zero = torch.zeros(1, device=indices.device, dtype=active_counts.dtype)
    active_offsets = torch.cat((zero, torch.cumsum(active_counts, dim=0)))
    return {
        "sorted_pair_ids": sorted_pair_ids,
        "sorted_tokens": sorted_tokens,
        "sorted_weights": sorted_weights,
        "pair_counts": pair_counts,
        "active_pair_ids": active_pair_ids,
        "active_counts": active_counts,
        "active_offsets": active_offsets,
    }


def stack_grouped_weights(
    experts: Sequence[torch.nn.Module],
    top_k: int,
    field: str,
    active_pair_ids: torch.Tensor,
    padded_ff: int,
    device: torch.device,
) -> torch.Tensor:
    base = torch.stack(
        [getattr(expert, field).weight for expert in experts],
        dim=0,
    )
    repeated = base.repeat_interleave(top_k, dim=0)
    selected = repeated.index_select(0, active_pair_ids)
    selected = selected.transpose(1, 2).contiguous()
    if field in ("w1", "w2"):
        pad = padded_ff - selected.shape[-1]
        if pad < 0:
            raise ValueError(f"padded_ff={padded_ff} is smaller than {field} width")
        if pad:
            selected = F.pad(selected, (0, pad))
    else:
        pad = padded_ff - selected.shape[-2]
        if pad < 0:
            raise ValueError(f"padded_ff={padded_ff} is smaller than w3 width")
        if pad:
            selected = F.pad(selected, (0, 0, 0, pad))
    return selected.to(device=device, dtype=torch.bfloat16).contiguous()


def dispatch_gpu_grouped_gemm(
    experts: Sequence[torch.nn.Module],
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, int]]:
    assignments = gpu_grouped_assignments(indices, weights, len(experts))
    sorted_tokens = assignments["sorted_tokens"]
    sorted_weights = assignments["sorted_weights"]
    active_pair_ids = assignments["active_pair_ids"]
    active_offsets = assignments["active_offsets"][1:].to(torch.int32)
    top_k = indices.shape[1]
    sorted_input = x.index_select(0, sorted_tokens).to(torch.bfloat16)
    expert_ff = int(experts[0].w1.out_features)
    padded_ff = ((expert_ff + 7) // 8) * 8

    for expert in experts:
        if any(getattr(expert, field).bias is not None for field in ("w1", "w2", "w3")):
            raise RuntimeError("Grouped-GEMM prototype requires bias-free experts")

    w1 = stack_grouped_weights(
        experts, top_k, "w1", active_pair_ids, padded_ff, x.device
    )
    w2 = stack_grouped_weights(
        experts, top_k, "w2", active_pair_ids, padded_ff, x.device
    )
    w3 = stack_grouped_weights(
        experts, top_k, "w3", active_pair_ids, padded_ff, x.device
    )

    hidden_1 = F.grouped_mm(sorted_input, w1, offs=active_offsets)
    hidden_2 = F.grouped_mm(sorted_input, w2, offs=active_offsets)
    gated = F.silu(hidden_1) * hidden_2
    sorted_out = F.grouped_mm(gated, w3, offs=active_offsets)

    out = torch.zeros_like(x)
    weighted = (
        sorted_out
        * sorted_weights.to(sorted_out.dtype).unsqueeze(1)
    ).to(out.dtype)
    out.index_add_(0, sorted_tokens, weighted)
    return out, {
        "active_groups": int(active_pair_ids.numel()),
        "total_groups": int(len(experts) * top_k),
        "grouped_mm_calls": 3,
        "original_expert_ff": expert_ff,
        "grouped_gemm_ff": padded_ff,
    }


def dispatch_variant(
    variant: str,
    experts: Sequence[torch.nn.Module],
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, int]]:
    if variant == "cpu_grouped_reference":
        return (
            v2.dispatch_grouped(
                experts,
                x,
                weights,
                indices,
                "grouped_rank_preserving",
            ),
            {
                "active_groups": -1,
                "total_groups": 32,
                "grouped_mm_calls": -1,
                "original_expert_ff": -1,
                "grouped_gemm_ff": -1,
            },
        )
    if variant == "gpu_grouped_gemm":
        return dispatch_gpu_grouped_gemm(experts, x, weights, indices)
    raise ValueError(variant)


def gradient_snapshot(experts: Sequence[torch.nn.Module]) -> Dict[str, torch.Tensor]:
    return {
        f"{index}.{name}": parameter.grad.detach().cpu().clone()
        for index, expert in enumerate(experts)
        for name, parameter in expert.named_parameters()
        if parameter.grad is not None
    }


def gradient_delta(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> float:
    if set(reference) != set(candidate):
        return float("inf")
    return max(
        (
            float(
                (reference[name].float() - candidate[name].float())
                .abs()
                .max()
                .item()
            )
            for name in reference
        ),
        default=0.0,
    )


def run_trial(
    variant: str,
    experts: Sequence[torch.nn.Module],
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
) -> Dict[str, Any]:
    for expert in experts:
        for parameter in expert.parameters():
            parameter.grad = None

    torch.cuda.synchronize(x.device)
    host_start = now_ns()
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)

    forward_start.record()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output, engine_info = dispatch_variant(
            variant,
            experts,
            x,
            weights,
            indices,
        )
    forward_end.record()
    forward_end.synchronize()
    forward_host_ns = now_ns() - host_start

    loss = output.square().mean()
    torch.cuda.synchronize(x.device)
    backward_host_start = now_ns()
    backward_start.record()
    loss.backward()
    backward_end.record()
    backward_end.synchronize()
    backward_host_ns = now_ns() - backward_host_start

    return {
        "forward_cuda_ns": int(forward_start.elapsed_time(forward_end) * 1_000_000.0),
        "backward_cuda_ns": int(backward_start.elapsed_time(backward_end) * 1_000_000.0),
        "forward_host_ns": int(forward_host_ns),
        "backward_host_ns": int(backward_host_ns),
        "loss": float(loss.detach().item()),
        "engine_info": engine_info,
        "output": output.detach().float().cpu(),
        "gradients": gradient_snapshot(experts),
    }


def metadata_integrity(
    indices: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
) -> Dict[str, Any]:
    reference = v2.sorted_assignments(indices, weights, num_experts)
    candidate = gpu_grouped_assignments(indices, weights, num_experts)
    candidate_counts = candidate["pair_counts"].detach().cpu().tolist()
    candidate_offsets = candidate["active_offsets"].detach().cpu().tolist()
    active_ids = candidate["active_pair_ids"].detach().cpu().tolist()
    reference_pair_offsets = [int(value) for value in reference["pair_offsets"]]
    reference_pair_counts = [
        reference_pair_offsets[i + 1] - reference_pair_offsets[i]
        for i in range(len(reference_pair_offsets) - 1)
    ]
    reference_active_ids = [
        i for i, count in enumerate(reference_pair_counts) if count > 0
    ]
    reference_active_counts = [
        count for count in reference_pair_counts if count > 0
    ]
    reference_active_offsets = [0]
    for count in reference_active_counts:
        reference_active_offsets.append(reference_active_offsets[-1] + count)
    return {
        "pair_counts_match": [int(x) for x in candidate_counts]
        == reference_pair_counts,
        "active_pair_ids_match": [int(x) for x in active_ids]
        == reference_active_ids,
        "active_offsets_match": [int(x) for x in candidate_offsets]
        == reference_active_offsets,
        "candidate_active_groups": len(active_ids),
        "candidate_pair_counts": [int(x) for x in candidate_counts],
    }


def run_workload(
    model: torch.nn.Module,
    fixture: str,
    token_count: int,
    device: torch.device,
    warmup: int,
    timed: int,
) -> Dict[str, Any]:
    num_experts = 16
    experts = list(model.layers[0].moe.experts[:num_experts])
    torch.manual_seed(20261018 + token_count)
    hidden_dim = int(model.layers[0].moe.experts[0].w1.in_features)
    x = torch.randn(token_count, hidden_dim, device=device)
    weights = (
        torch.tensor([0.6, 0.4], device=device, dtype=torch.float32)
        .view(1, 2)
        .expand(token_count, -1)
        .contiguous()
    )
    indices = v2.fixture_indices(fixture, token_count, num_experts, device)
    metadata = metadata_integrity(indices, weights, num_experts)
    reports: Dict[str, Any] = {}
    reference_output = None
    reference_gradients = None

    for variant in VARIANTS:
        for _ in range(warmup):
            warm = run_trial(variant, experts, x, weights, indices)
            del warm
            torch.cuda.synchronize(device)

        trials = []
        first = None
        for _ in range(timed):
            result = run_trial(variant, experts, x, weights, indices)
            if first is None:
                first = result
            trials.append(result)
            for expert in experts:
                for parameter in expert.parameters():
                    parameter.grad = None

        assert first is not None
        if reference_output is None:
            reference_output = first["output"]
            reference_gradients = first["gradients"]

        output_delta = float(
            (first["output"] - reference_output).abs().max().item()
        )
        grad_delta = gradient_delta(reference_gradients, first["gradients"])
        reports[variant] = {
            "mean_forward_cuda_ns": float(
                sum(item["forward_cuda_ns"] for item in trials) / len(trials)
            ),
            "mean_backward_cuda_ns": float(
                sum(item["backward_cuda_ns"] for item in trials) / len(trials)
            ),
            "mean_forward_host_ns": float(
                sum(item["forward_host_ns"] for item in trials) / len(trials)
            ),
            "mean_backward_host_ns": float(
                sum(item["backward_host_ns"] for item in trials) / len(trials)
            ),
            "engine_info": first["engine_info"],
            "max_output_abs_delta_vs_reference": output_delta,
            "max_gradient_abs_delta_vs_reference": grad_delta,
            "strict_output_pass": output_delta <= FORWARD_TOL,
            "strict_gradient_pass": grad_delta <= GRAD_TOL,
            "strict_equivalence_pass": output_delta <= FORWARD_TOL and grad_delta <= GRAD_TOL,
            "losses_finite": all(math.isfinite(item["loss"]) for item in trials),
        }
        del trials, first
        torch.cuda.empty_cache()

    del experts, x, weights, indices
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "fixture": fixture,
        "expert_count": num_experts,
        "token_count": token_count,
        "metadata_integrity": metadata,
        "variants": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("preflight", "landscape"), default="preflight")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    config_path = CONTEXT_ROOT / "crucible_manifest" / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hp = config["hyperparameters"]
    anchor_path = CONTEXT_ROOT / "crucible_manifest" / config["anchors"]["42"]["relative_path"]
    model = v1.load_model(anchor_path, hp, device)
    model.eval()

    token_counts = (512,) if args.mode == "preflight" else (512, 1024, 2048, 4096)
    warmup = 1
    timed = 2 if args.mode == "preflight" else 3
    reports = []
    for token_count in token_counts:
        for fixture in ("balanced", "high_skew"):
            report = run_workload(
                model,
                fixture,
                token_count,
                device,
                warmup=warmup,
                timed=timed,
            )
            reports.append(report)
            reference = report["variants"]["cpu_grouped_reference"]
            candidate = report["variants"]["gpu_grouped_gemm"]
            print(
                json.dumps(
                    {
                        "fixture": fixture,
                        "token_count": token_count,
                        "reference_total_ms": round(
                            (reference["mean_forward_cuda_ns"] + reference["mean_backward_cuda_ns"]) / 1e6,
                            3,
                        ),
                        "grouped_gemm_total_ms": round(
                            (candidate["mean_forward_cuda_ns"] + candidate["mean_backward_cuda_ns"]) / 1e6,
                            3,
                        ),
                        "output_delta": candidate["max_output_abs_delta_vs_reference"],
                        "gradient_delta": candidate["max_gradient_abs_delta_vs_reference"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        "v4_source_sha256": sha256_file(Path(__file__).resolve()),
        "v2_dispatch_source_sha256": sha256_file(V2_DIR / "dispatch_strategy_landscape.py"),
        "v2_benchmark_source_sha256": sha256_file(V2_DIR / "strategy_landscape_benchmark.py"),
        "v1_source_sha256": sha256_file(V1_DIR / "grouped_expert_benchmark.py"),
        "config_sha256": sha256_file(config_path),
        "anchor_sha256": sha256_file(anchor_path),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "device": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "triton_available": True,
        "triton_version": __import__("triton").__version__,
        "source_hashes": source_hashes,
        "matrix": {
            "expert_counts": [16],
            "token_counts": list(token_counts),
            "fixtures": ["balanced", "high_skew"],
            "variants": list(VARIANTS),
            "warmup": warmup,
            "timed": timed,
        },
        "reports": reports,
        "integrity": {
            "all_metadata_matches": all(
                all(report["metadata_integrity"].values())
                for report in reports
            ),
            "all_strict_equivalence_pass": all(
                report["variants"][variant]["strict_equivalence_pass"]
                for report in reports
                for variant in VARIANTS
            ),
            "all_losses_finite": all(
                report["variants"][variant]["losses_finite"]
                for report in reports
                for variant in VARIANTS
            ),
        },
        "interpretation_limits": [
            "This is a dispatch-only preflight using frozen E16 layer-0 expert weights.",
            "It is not a full-model training result.",
            "Grouped-GEMM backend timing may differ across GPU architectures and matrix shapes.",
            "A candidate that changes numerical behavior beyond tolerance is not eligible for the strict path.",
        ],
    }
    summary_path = output / "grouped_gemm_preflight_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {
        "grouped_gemm_preflight_summary.json": sha256_file(summary_path),
        **source_hashes,
    }
    hashes_path = output / "grouped_gemm_preflight_hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(summary_path),
                "all_metadata_matches": summary["integrity"]["all_metadata_matches"],
                "all_strict_equivalence_pass": summary["integrity"]["all_strict_equivalence_pass"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

