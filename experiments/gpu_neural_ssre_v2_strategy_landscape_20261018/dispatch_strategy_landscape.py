#!/usr/bin/env python3
"""GPU Neural SSRE v2 dispatch-only workload landscape.

This fixture varies expert-count fragmentation, token volume and routing
skew using frozen E16 layer-0 expert weights. It is an execution benchmark,
not a retrained E4/E8/E16 language-model comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F


V1_DIR = Path("/home/timbushnell/gpu_neural_ssre_v1_grouped_expert_20261018")
V2_DIR = Path("/home/timbushnell/gpu_neural_ssre_v2_strategy_landscape_20261018")
V4_DIR = Path("/home/timbushnell/ssre_npa_v4_gpu_step_benchmark_20261018")
CONTEXT_SHARED = Path("/home/timbushnell/crucible_context_gate_v1_20261018/shared")
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(V4_DIR))
sys.path.insert(0, str(CONTEXT_SHARED))

import strategy_landscape_benchmark as v2  # noqa: E402
import grouped_expert_benchmark as v1  # noqa: E402


SCHEMA_VERSION = "gpu-neural-ssre-v2-dispatch-landscape-20261018"
VARIANTS = (
    "original",
    "grouped_rank_preserving",
    "grouped_rank_major",
    "grouped_sorted",
    "batched_padded",
)
EXPERT_COUNTS = (4, 8, 16)
TOKEN_COUNTS = (512, 1024, 2048, 4096)
FIXTURES = ("balanced", "moderate_skew", "high_skew")
FORWARD_TOL = v1.FORWARD_TOL
GRAD_TOL = v1.GRAD_TOL


def now_ns() -> int:
    return time.perf_counter_ns()


def fixture_indices(name: str, token_count: int, num_experts: int, device: torch.device) -> torch.Tensor:
    indices = torch.empty(token_count, 2, dtype=torch.long, device=device)
    positions = torch.arange(token_count, device=device)
    if name == "balanced":
        indices[:, 0] = (2 * positions) % num_experts
        indices[:, 1] = (2 * positions + 1) % num_experts
        return indices
    fraction = 0.50 if name == "moderate_skew" else 0.80
    dominant = int(token_count * fraction)
    indices[:dominant, 0] = 0
    indices[:dominant, 1] = 1
    rest = positions[dominant:] - dominant
    tail = max(1, num_experts - 2)
    indices[dominant:, 0] = 2 + (2 * rest) % tail
    indices[dominant:, 1] = 2 + (2 * rest + 1) % tail
    return indices


def counts_report(indices: torch.Tensor, num_experts: int) -> Dict[str, Any]:
    counts = torch.bincount(indices.reshape(-1), minlength=num_experts).detach().cpu().tolist()
    return {
        "counts": [int(value) for value in counts],
        "min": int(min(counts)),
        "max": int(max(counts)),
        "max_min_ratio": float(max(counts) / max(1, min(counts))),
        "coefficient_of_variation": float(torch.tensor(counts, dtype=torch.float64).std(unbiased=False).item() / max(1e-12, sum(counts) / len(counts))),
    }


def sorted_assignments(indices: torch.Tensor, weights: torch.Tensor, num_experts: int) -> Dict[str, Any]:
    token_count, top_k = indices.shape
    flat_experts = indices.reshape(-1)
    flat_weights = weights.reshape(-1)
    flat_ranks = torch.arange(top_k, device=indices.device).view(1, -1).expand(token_count, -1).reshape(-1)
    flat_tokens = torch.arange(token_count, device=indices.device).view(-1, 1).expand(-1, top_k).reshape(-1)
    key = ((flat_experts * top_k + flat_ranks) * token_count) + flat_tokens
    order = torch.argsort(key, stable=True)
    sorted_experts = flat_experts.index_select(0, order)
    sorted_ranks = flat_ranks.index_select(0, order)
    sorted_tokens = flat_tokens.index_select(0, order)
    sorted_weights = flat_weights.index_select(0, order)
    pair_counts = torch.bincount(flat_experts * top_k + flat_ranks, minlength=num_experts * top_k)
    pair_counts_cpu = [int(value) for value in pair_counts.detach().cpu().tolist()]
    pair_offsets = [0]
    for count in pair_counts_cpu:
        pair_offsets.append(pair_offsets[-1] + count)
    counts = torch.bincount(flat_experts, minlength=num_experts)
    counts_cpu = [int(value) for value in counts.detach().cpu().tolist()]
    offsets = [0]
    for count in counts_cpu:
        offsets.append(offsets[-1] + count)
    return {
        "sorted_experts": sorted_experts,
        "sorted_ranks": sorted_ranks,
        "sorted_tokens": sorted_tokens,
        "sorted_weights": sorted_weights,
        "pair_offsets": pair_offsets,
        "offsets": offsets,
        "counts_cpu": counts_cpu,
        "top_k": top_k,
    }


def dispatch_original(experts: Sequence[torch.nn.Module], x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for expert_index, expert in enumerate(experts):
        for rank in range(indices.shape[1]):
            selected = indices[:, rank] == expert_index
            if selected.any():
                expert_out = expert(x[selected])
                weighted = (expert_out * weights[selected, rank].to(expert_out.dtype).unsqueeze(1)).to(out.dtype)
                out.index_put_((selected,), weighted, accumulate=True)
    return out


def dispatch_grouped(experts: Sequence[torch.nn.Module], x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, variant: str) -> torch.Tensor:
    num_experts = len(experts)
    assignments = sorted_assignments(indices, weights, num_experts)
    sorted_experts = assignments["sorted_experts"]
    sorted_tokens = assignments["sorted_tokens"]
    sorted_weights = assignments["sorted_weights"]
    pair_offsets = assignments["pair_offsets"]
    offsets = assignments["offsets"]
    out = torch.zeros_like(x)
    top_k = indices.shape[1]

    if variant == "grouped_rank_preserving":
        for expert_index in range(num_experts):
            for rank in range(top_k):
                pair_id = expert_index * top_k + rank
                start, end = pair_offsets[pair_id], pair_offsets[pair_id + 1]
                if end <= start:
                    continue
                token_indices = sorted_tokens[start:end]
                expert_out = experts[expert_index](x.index_select(0, token_indices))
                weighted = (expert_out * sorted_weights[start:end].to(expert_out.dtype).unsqueeze(1)).to(out.dtype)
                out.index_put_((token_indices,), weighted, accumulate=True)
        return out

    if variant == "grouped_rank_major":
        rank_major = v2.rank_major_assignments(indices, weights, num_experts)
        rank_major_tokens = rank_major["sorted_tokens"]
        rank_major_weights = rank_major["sorted_weights"]
        rank_major_offsets = rank_major["pair_offsets"]
        for rank in range(top_k):
            for expert_index in range(num_experts):
                pair_id = rank * num_experts + expert_index
                start, end = rank_major_offsets[pair_id], rank_major_offsets[pair_id + 1]
                if end <= start:
                    continue
                token_indices = rank_major_tokens[start:end]
                expert_out = experts[expert_index](x.index_select(0, token_indices))
                weighted = (expert_out * rank_major_weights[start:end].to(expert_out.dtype).unsqueeze(1)).to(out.dtype)
                out.index_put_((token_indices,), weighted, accumulate=True)
        return out

    if variant == "grouped_sorted":
        for expert_index in range(num_experts):
            start, end = offsets[expert_index], offsets[expert_index + 1]
            if end <= start:
                continue
            token_indices = sorted_tokens[start:end]
            expert_out = experts[expert_index](x.index_select(0, token_indices))
            weighted = (expert_out * sorted_weights[start:end].to(expert_out.dtype).unsqueeze(1)).to(out.dtype)
            out.index_add_(0, token_indices, weighted)
        return out

    if variant != "batched_padded":
        raise ValueError(variant)
    counts_cpu = assignments["counts_cpu"]
    max_count = max(counts_cpu)
    positions = torch.arange(sorted_tokens.numel(), device=x.device)
    offset_tensor = torch.tensor(offsets[:-1], device=x.device, dtype=positions.dtype)
    positions = positions - offset_tensor[sorted_experts]
    destinations = sorted_experts * max_count + positions
    padded = x.new_zeros(num_experts * max_count, x.shape[1])
    padded.index_copy_(0, destinations, x.index_select(0, sorted_tokens))
    padded = padded.view(num_experts, max_count, x.shape[1])
    w1 = torch.stack([expert.w1.weight for expert in experts], dim=0)
    w2 = torch.stack([expert.w2.weight for expert in experts], dim=0)
    w3 = torch.stack([expert.w3.weight for expert in experts], dim=0)
    hidden_1 = torch.bmm(padded, w1.transpose(1, 2))
    hidden_2 = torch.bmm(padded, w2.transpose(1, 2))
    padded_out = torch.bmm(F.silu(hidden_1) * hidden_2, w3.transpose(1, 2))
    sorted_out = padded_out.reshape(-1, x.shape[1]).index_select(0, destinations)
    weighted = (sorted_out * sorted_weights.to(sorted_out.dtype).unsqueeze(1)).to(out.dtype)
    out.index_add_(0, sorted_tokens, weighted)
    return out


def timed_dispatch(experts: Sequence[torch.nn.Module], x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, variant: str) -> Dict[str, Any]:
    torch.cuda.synchronize(x.device)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    host_start = now_ns()
    forward_start.record()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = dispatch_original(experts, x, weights, indices) if variant == "original" else dispatch_grouped(experts, x, weights, indices, variant)
    forward_end.record()
    forward_end.synchronize()
    forward_host_ns = now_ns() - host_start
    forward_cuda_ns = int(forward_start.elapsed_time(forward_end) * 1_000_000.0)
    loss = out.square().mean()
    torch.cuda.synchronize(x.device)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    backward_host_start = now_ns()
    backward_start.record()
    loss.backward()
    backward_end.record()
    backward_end.synchronize()
    backward_host_ns = now_ns() - backward_host_start
    backward_cuda_ns = int(backward_start.elapsed_time(backward_end) * 1_000_000.0)
    return {
        "forward_host_ns": forward_host_ns,
        "forward_cuda_ns": forward_cuda_ns,
        "backward_host_ns": backward_host_ns,
        "backward_cuda_ns": backward_cuda_ns,
        "loss": float(loss.detach().item()),
        "output": out.detach(),
    }


def gradient_snapshot(experts: Sequence[torch.nn.Module]) -> Dict[str, torch.Tensor]:
    return {
        f"{index}.{name}": parameter.grad.detach().cpu().clone()
        for index, expert in enumerate(experts)
        for name, parameter in expert.named_parameters()
        if parameter.grad is not None
    }


def gradient_delta(reference: Dict[str, torch.Tensor], candidate: Dict[str, torch.Tensor]) -> float:
    if set(reference) != set(candidate):
        return float("inf")
    return max((float((reference[name].float() - candidate[name].float()).abs().max().item()) for name in reference), default=0.0)


def run_workload(model: torch.nn.Module, num_experts: int, token_count: int, fixture: str, warmup: int, timed: int, device: torch.device) -> Dict[str, Any]:
    experts = list(model.layers[0].moe.experts[:num_experts])
    torch.manual_seed(20261018 + num_experts * 1000 + token_count)
    x = torch.randn(token_count, int(model.layers[0].moe.experts[0].w1.in_features), device=device)
    weights = torch.tensor([0.6, 0.4], device=device, dtype=torch.float32).view(1, 2).expand(token_count, -1).contiguous()
    indices = fixture_indices(fixture, token_count, num_experts, device)
    report: Dict[str, Any] = {
        "expert_count": num_experts,
        "token_count": token_count,
        "fixture": fixture,
        "workload": counts_report(indices, num_experts),
        "variants": {},
    }
    reference_output = None
    reference_grad = None
    for variant in VARIANTS:
        model.zero_grad(set_to_none=True)
        for _ in range(warmup):
            warm = timed_dispatch(experts, x, weights, indices, variant)
            del warm["output"]
            model.zero_grad(set_to_none=True)
        timings = []
        first_output = None
        first_grad = None
        for _ in range(timed):
            result = timed_dispatch(experts, x, weights, indices, variant)
            if first_output is None:
                first_output = result["output"].float().cpu()
                first_grad = gradient_snapshot(experts)
            timings.append(result)
            model.zero_grad(set_to_none=True)
        if reference_output is None:
            reference_output = first_output
            reference_grad = first_grad
        max_output_delta = float((first_output - reference_output).abs().max().item())
        max_grad_delta = gradient_delta(reference_grad, first_grad)
        report["variants"][variant] = {
            "mean_forward_cuda_ns": sum(item["forward_cuda_ns"] for item in timings) / len(timings),
            "mean_backward_cuda_ns": sum(item["backward_cuda_ns"] for item in timings) / len(timings),
            "mean_forward_host_ns": sum(item["forward_host_ns"] for item in timings) / len(timings),
            "mean_backward_host_ns": sum(item["backward_host_ns"] for item in timings) / len(timings),
            "max_forward_abs_delta_vs_original": max_output_delta,
            "max_gradient_abs_delta_vs_original": max_grad_delta,
            "strict_output_pass": max_output_delta <= FORWARD_TOL,
            "strict_gradient_pass": max_grad_delta <= GRAD_TOL,
            "strict_equivalence_pass": max_output_delta <= FORWARD_TOL and max_grad_delta <= GRAD_TOL,
            "losses_finite": all(math.isfinite(item["loss"]) for item in timings),
        }
        del timings, first_output, first_grad
        torch.cuda.empty_cache()
    report["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    report["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    del experts, x, weights, indices
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU Neural SSRE v2 dispatch landscape")
    parser.add_argument("--mode", choices=("preflight", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    root = Path("/home/timbushnell/crucible_context_gate_v1_20261018/context_128")
    config_path = root / "crucible_manifest" / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hp = config["hyperparameters"]
    anchor_path = root / "crucible_manifest" / config["anchors"]["42"]["relative_path"]
    model = v1.load_model(anchor_path, hp, device)
    model.eval()
    if args.mode == "preflight":
        expert_counts, token_counts, fixtures, warmup, timed = (4,), (512,), ("balanced", "high_skew"), 1, 2
    else:
        expert_counts, token_counts, fixtures, warmup, timed = EXPERT_COUNTS, TOKEN_COUNTS, FIXTURES, 2, 8
    reports = []
    for num_experts in expert_counts:
        for token_count in token_counts:
            for fixture in fixtures:
                report = run_workload(model, num_experts, token_count, fixture, warmup, timed, device)
                reports.append(report)
                print(json.dumps({"expert_count": num_experts, "token_count": token_count, "fixture": fixture, "rank_preserving_ms": round(report["variants"]["grouped_rank_preserving"]["mean_forward_cuda_ns"] / 1e6, 3)}, sort_keys=True), flush=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "device": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "config_sha256": v1.sha256_file(config_path),
        "anchor_sha256": v1.sha256_file(anchor_path),
        "v2_source_sha256": v1.sha256_file(Path(__file__).resolve()),
        "matrix": {"expert_counts": list(expert_counts), "token_counts": list(token_counts), "fixtures": list(fixtures), "variants": list(VARIANTS), "warmup": warmup, "timed": timed},
        "reports": reports,
        "integrity": {
            "workload_count": len(reports),
            "all_strict_candidate_fixture_pass": all(
                report["variants"][variant]["strict_equivalence_pass"]
                for report in reports
                for variant in ("grouped_rank_preserving", "grouped_rank_major")
            ),
            "all_losses_finite": all(report["variants"][variant]["losses_finite"] for report in reports for variant in VARIANTS),
        },
        "interpretation_limits": [
            "This is a dispatch-only control using the first N frozen E16 layer-0 expert weights, not retrained E4/E8/E16 language models.",
            "Only rank-preserving candidates are eligible for the strict execution-plan gate.",
            "Shape-changing combined-rank and padded-batched timings are diagnostic when their output or gradient deltas exceed the strict tolerances.",
        ],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    out_path = output / "dispatch_strategy_landscape_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {
        "dispatch_strategy_landscape_summary.json": v1.sha256_file(out_path),
        "v2_source_sha256": summary["v2_source_sha256"],
        "frozen_config_sha256": summary["config_sha256"],
        "frozen_anchor_sha256": summary["anchor_sha256"],
    }
    (output / "dispatch_strategy_landscape_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_path), "mode": args.mode, "workloads": len(reports), "strict_fixture_pass": summary["integrity"]["all_strict_candidate_fixture_pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
