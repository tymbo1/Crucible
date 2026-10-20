#!/usr/bin/env python3
"""Dispatch-only balanced/imbalanced workload control for GPU Neural SSRE v1."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


V1_DIR = Path("/home/timbushnell/gpu_neural_ssre_v1_grouped_expert_20261018")
V4_DIR = Path("/home/timbushnell/ssre_npa_v4_gpu_step_benchmark_20261018")
CONTEXT_SHARED = Path("/home/timbushnell/crucible_context_gate_v1_20261018/shared")
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(V4_DIR))
sys.path.insert(0, str(CONTEXT_SHARED))

from gpu_step_benchmark import load_model, sha256_file  # noqa: E402
from grouped_expert_benchmark import _sorted_assignments  # noqa: E402


VARIANTS = ("original", "grouped_rank_preserving", "grouped_sorted", "batched_padded")
SCHEMA_VERSION = "gpu-neural-ssre-v1-dispatch-fixtures-20261018"


def now_ns() -> int:
    return time.perf_counter_ns()


def dispatch_original(experts: Sequence[torch.nn.Module], x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for expert_index, expert in enumerate(experts):
        for rank in range(indices.shape[1]):
            selected = indices[:, rank] == expert_index
            if selected.any():
                expert_out = expert(x[selected])
                weighted = (
                    expert_out * weights[selected, rank].to(expert_out.dtype).unsqueeze(1)
                ).to(out.dtype)
                out.index_put_((selected,), weighted, accumulate=True)
    return out


def dispatch_grouped(
    experts: Sequence[torch.nn.Module],
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    num_experts = len(experts)
    assignments = _sorted_assignments(indices, weights, num_experts)
    sorted_experts = assignments["sorted_experts"]
    sorted_tokens = assignments["sorted_tokens"]
    sorted_weights = assignments["sorted_weights"]
    offsets = assignments["offsets"]
    pair_offsets = assignments["pair_offsets"]
    counts_cpu = assignments["counts_cpu"]
    out = torch.zeros_like(x)

    if variant == "grouped_rank_preserving":
        for expert_index in range(num_experts):
            for rank in range(indices.shape[1]):
                pair_id = expert_index * indices.shape[1] + rank
                start, end = pair_offsets[pair_id], pair_offsets[pair_id + 1]
                if end <= start:
                    continue
                token_indices = sorted_tokens[start:end]
                expert_out = experts[expert_index](x.index_select(0, token_indices))
                weighted = (
                    expert_out
                    * sorted_weights[start:end].to(expert_out.dtype).unsqueeze(1)
                ).to(out.dtype)
                out.index_put_((token_indices,), weighted, accumulate=True)
        return out

    if variant == "grouped_sorted":
        for expert_index in range(num_experts):
            start, end = offsets[expert_index], offsets[expert_index + 1]
            if end <= start:
                continue
            token_indices = sorted_tokens[start:end]
            expert_out = experts[expert_index](x.index_select(0, token_indices))
            weighted = (
                expert_out
                * sorted_weights[start:end].to(expert_out.dtype).unsqueeze(1)
            ).to(out.dtype)
            out.index_add_(0, token_indices, weighted)
        return out

    if variant != "batched_padded":
        raise ValueError(variant)
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
    padded_out = torch.bmm(torch.nn.functional.silu(hidden_1) * hidden_2, w3.transpose(1, 2))
    sorted_out = padded_out.reshape(-1, x.shape[1]).index_select(0, destinations)
    weighted = (sorted_out * sorted_weights.to(sorted_out.dtype).unsqueeze(1)).to(out.dtype)
    out.index_add_(0, sorted_tokens, weighted)
    return out


def fixture_indices(name: str, token_count: int, device: torch.device) -> torch.Tensor:
    indices = torch.empty(token_count, 2, dtype=torch.long, device=device)
    positions = torch.arange(token_count, device=device)
    if name == "balanced":
        indices[:, 0] = (2 * positions) % 16
        indices[:, 1] = (2 * positions + 1) % 16
        return indices
    if name == "imbalanced":
        dominant = int(token_count * 0.80)
        indices[:dominant, 0] = 0
        indices[:dominant, 1] = 1
        rest = positions[dominant:] - dominant
        indices[dominant:, 0] = 2 + (2 * rest) % 14
        indices[dominant:, 1] = 2 + (2 * rest + 1) % 14
        return indices
    raise ValueError(name)


def counts_report(indices: torch.Tensor) -> Dict[str, Any]:
    counts = torch.bincount(indices.reshape(-1), minlength=16).detach().cpu().tolist()
    return {
        "counts": [int(value) for value in counts],
        "min": int(min(counts)),
        "max": int(max(counts)),
        "max_min_ratio": float(max(counts) / max(1, min(counts))),
    }


def timed_dispatch(
    experts: Sequence[torch.nn.Module],
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    variant: str,
) -> Dict[str, Any]:
    torch.cuda.synchronize(x.device)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    host_start = now_ns()
    forward_start.record()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = (
            dispatch_original(experts, x, weights, indices)
            if variant == "original"
            else dispatch_grouped(experts, x, weights, indices, variant)
        )
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Balanced/imbalanced E16 dispatch fixtures")
    parser.add_argument("--context-root", type=Path, default=Path("/home/timbushnell/crucible_context_gate_v1_20261018/context_128"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-count", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timed", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    device = torch.device("cuda")
    root = args.context_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = root / "crucible_manifest" / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hp = config["hyperparameters"]
    anchor_path = root / "crucible_manifest" / config["anchors"][str(args.seed)]["relative_path"]
    model = load_model(anchor_path, hp, device)
    model.eval()
    experts = model.layers[0].moe.experts
    torch.manual_seed(20261018)
    x = torch.randn(args.token_count, int(hp["d_model"]), device=device)
    weights = torch.tensor([0.6, 0.4], device=device, dtype=torch.float32).view(1, 2).expand(args.token_count, -1).contiguous()
    reports: Dict[str, Any] = {}
    for fixture in ("balanced", "imbalanced"):
        indices = fixture_indices(fixture, args.token_count, device)
        fixture_report: Dict[str, Any] = {"workload": counts_report(indices), "variants": {}}
        reference_output = None
        for variant in VARIANTS:
            model.zero_grad(set_to_none=True)
            for _ in range(args.warmup):
                warm = timed_dispatch(experts, x, weights, indices, variant)
                del warm["output"]
                model.zero_grad(set_to_none=True)
            times = []
            first_output = None
            for _ in range(args.timed):
                result = timed_dispatch(experts, x, weights, indices, variant)
                if first_output is None:
                    first_output = result["output"].float().cpu()
                times.append(result)
                model.zero_grad(set_to_none=True)
            if reference_output is None:
                reference_output = first_output
            max_delta = float((first_output - reference_output).abs().max().item())
            fixture_report["variants"][variant] = {
                "mean_forward_cuda_ns": sum(r["forward_cuda_ns"] for r in times) / len(times),
                "mean_backward_cuda_ns": sum(r["backward_cuda_ns"] for r in times) / len(times),
                "mean_forward_host_ns": sum(r["forward_host_ns"] for r in times) / len(times),
                "mean_backward_host_ns": sum(r["backward_host_ns"] for r in times) / len(times),
                "max_forward_abs_delta_vs_original": max_delta,
                "losses_finite": all(math.isfinite(r["loss"]) for r in times),
            }
            del times, first_output
            torch.cuda.empty_cache()
        reports[fixture] = fixture_report
    summary = {
        "schema_version": SCHEMA_VERSION,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "context_root": str(root),
        "config_sha256": sha256_file(config_path),
        "anchor_sha256": sha256_file(anchor_path),
        "parameters": {"seed": args.seed, "token_count": args.token_count, "d_model": int(hp["d_model"]), "num_experts": 16, "top_k": 2, "warmup": args.warmup, "timed": args.timed},
        "reports": reports,
        "interpretation_limits": [
            "This is a dispatch-only control using real frozen E16 expert weights, not a full transformer training benchmark.",
            "The grouped arms are compared for execution behavior; only rank-preserving arms retain the strict numerical contract.",
        ],
    }
    out_path = output_dir / "dispatch_workload_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {
        "dispatch_workload_summary.json": sha256_file(out_path),
        "benchmark_source_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_config_sha256": summary["config_sha256"],
        "frozen_anchor_sha256": summary["anchor_sha256"],
    }
    (output_dir / "dispatch_output_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_path), "fixtures": list(reports), "device": summary["device"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

