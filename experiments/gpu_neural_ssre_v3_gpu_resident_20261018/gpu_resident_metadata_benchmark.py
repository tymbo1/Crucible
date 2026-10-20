#!/usr/bin/env python3
"""GPU Neural SSRE v3 GPU-resident routing metadata preflight.

This benchmark imports the sealed V2/V1 implementations as read-only
dependencies and compares three dispatch paths on frozen E16 expert weights.
It writes a new report directory and never writes to the source artifacts.
"""

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


V1_DIR = Path("/home/timbushnell/gpu_neural_ssre_v1_grouped_expert_20261018")
V2_DIR = Path("/home/timbushnell/gpu_neural_ssre_v2_strategy_landscape_20261018")
CONTEXT_ROOT = Path("/home/timbushnell/crucible_context_gate_v1_20261018/context_128")
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(V2_DIR))

import dispatch_strategy_landscape as v2  # noqa: E402
import grouped_expert_benchmark as v1  # noqa: E402


SCHEMA_VERSION = "gpu-neural-ssre-v3-gpu-resident-metadata-20261018"
FORWARD_TOL = float(v1.FORWARD_TOL)
GRAD_TOL = float(v1.GRAD_TOL)
VARIANTS = ("cpu_offsets", "gpu_offsets_scalar", "gpu_segment_mask")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_ns() -> int:
    return time.perf_counter_ns()


def gpu_metadata_assignments(
    indices: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
) -> Dict[str, torch.Tensor]:
    """Build sorted assignments and all count/offset metadata on CUDA."""

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
    key = pair_ids * token_count + flat_tokens
    order = torch.argsort(key, stable=True)

    sorted_pair_ids = pair_ids.index_select(0, order)
    sorted_experts = flat_experts.index_select(0, order)
    sorted_tokens = flat_tokens.index_select(0, order)
    sorted_weights = flat_weights.index_select(0, order)

    pair_counts = torch.bincount(
        pair_ids,
        minlength=num_experts * top_k,
    )
    counts = torch.bincount(
        flat_experts,
        minlength=num_experts,
    )
    zero_pair = torch.zeros(1, device=indices.device, dtype=pair_counts.dtype)
    zero_count = torch.zeros(1, device=indices.device, dtype=counts.dtype)
    pair_offsets = torch.cat((zero_pair, torch.cumsum(pair_counts, dim=0)))
    offsets = torch.cat((zero_count, torch.cumsum(counts, dim=0)))

    return {
        "sorted_pair_ids": sorted_pair_ids,
        "sorted_experts": sorted_experts,
        "sorted_tokens": sorted_tokens,
        "sorted_weights": sorted_weights,
        "pair_counts": pair_counts,
        "pair_offsets": pair_offsets,
        "counts": counts,
        "offsets": offsets,
    }


def dispatch_gpu_offsets_scalar(
    experts: Sequence[torch.nn.Module],
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """GPU metadata with explicit scalar reads at the Python boundary."""

    assignments = gpu_metadata_assignments(indices, weights, len(experts))
    out = torch.zeros_like(x)
    top_k = indices.shape[1]
    scalar_reads = 0

    for expert_index, expert in enumerate(experts):
        for rank in range(top_k):
            pair_id = expert_index * top_k + rank
            start = int(assignments["pair_offsets"][pair_id].item())
            end = int(assignments["pair_offsets"][pair_id + 1].item())
            scalar_reads += 2
            if end <= start:
                continue
            token_indices = assignments["sorted_tokens"][start:end]
            expert_out = expert(x.index_select(0, token_indices))
            weighted = (
                expert_out
                * assignments["sorted_weights"][start:end]
                .to(expert_out.dtype)
                .unsqueeze(1)
            ).to(out.dtype)
            out.index_put_((token_indices,), weighted, accumulate=True)

    return out, scalar_reads


def dispatch_gpu_segment_mask(
    experts: Sequence[torch.nn.Module],
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """GPU metadata with no host offset/scalar boundary."""

    assignments = gpu_metadata_assignments(indices, weights, len(experts))
    out = torch.zeros_like(x)
    top_k = indices.shape[1]
    mask_evaluations = 0

    for expert_index, expert in enumerate(experts):
        for rank in range(top_k):
            pair_id = expert_index * top_k + rank
            segment_mask = assignments["sorted_pair_ids"] == pair_id
            mask_evaluations += 1
            token_indices = assignments["sorted_tokens"][segment_mask]
            expert_weights = assignments["sorted_weights"][segment_mask]
            expert_out = expert(x.index_select(0, token_indices))
            weighted = (
                expert_out
                * expert_weights.to(expert_out.dtype).unsqueeze(1)
            ).to(out.dtype)
            out.index_put_((token_indices,), weighted, accumulate=True)

    return out, mask_evaluations


def dispatch_variant(
    variant: str,
    experts: Sequence[torch.nn.Module],
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    if variant == "cpu_offsets":
        return (
            v2.dispatch_grouped(
                experts,
                x,
                weights,
                indices,
                "grouped_rank_preserving",
            ),
            0,
        )
    if variant == "gpu_offsets_scalar":
        return dispatch_gpu_offsets_scalar(experts, x, weights, indices)
    if variant == "gpu_segment_mask":
        return dispatch_gpu_segment_mask(experts, x, weights, indices)
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
        output, boundary_operations = dispatch_variant(
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
        "boundary_operations": int(boundary_operations),
        "output": output.detach().float().cpu(),
        "gradients": gradient_snapshot(experts),
    }


def metadata_integrity(
    indices: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
) -> Dict[str, Any]:
    cpu = v2.sorted_assignments(indices, weights, num_experts)
    gpu = gpu_metadata_assignments(indices, weights, num_experts)

    pair_counts = gpu["pair_counts"].detach().cpu().tolist()
    pair_offsets = gpu["pair_offsets"].detach().cpu().tolist()
    counts = gpu["counts"].detach().cpu().tolist()
    offsets = gpu["offsets"].detach().cpu().tolist()
    cpu_pair_offsets = [int(x) for x in cpu["pair_offsets"]]
    cpu_pair_counts = [
        cpu_pair_offsets[index + 1] - cpu_pair_offsets[index]
        for index in range(len(cpu_pair_offsets) - 1)
    ]

    return {
        "pair_counts_match": [int(x) for x in pair_counts]
        == cpu_pair_counts,
        "pair_offsets_match": [int(x) for x in pair_offsets]
        == cpu_pair_offsets,
        "counts_match": [int(x) for x in counts]
        == [int(x) for x in cpu["counts_cpu"]],
        "offsets_match": [int(x) for x in offsets]
        == [int(x) for x in cpu["offsets"]],
        "cpu_pair_counts": cpu_pair_counts,
        "gpu_pair_counts": [int(x) for x in pair_counts],
    }


def run_workload(
    model: torch.nn.Module,
    num_experts: int,
    token_count: int,
    fixture: str,
    device: torch.device,
    warmup: int,
    timed: int,
) -> Dict[str, Any]:
    experts = list(model.layers[0].moe.experts[:num_experts])
    torch.manual_seed(20261018 + num_experts * 1000 + token_count)
    hidden_dim = int(model.layers[0].moe.experts[0].w1.in_features)
    x = torch.randn(token_count, hidden_dim, device=device)
    weights = (
        torch.tensor([0.6, 0.4], device=device, dtype=torch.float32)
        .view(1, 2)
        .expand(token_count, -1)
        .contiguous()
    )
    indices = v2.fixture_indices(fixture, token_count, num_experts, device)

    integrity = metadata_integrity(indices, weights, num_experts)
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
            "boundary_operations": int(first["boundary_operations"]),
            "max_output_abs_delta_vs_cpu_offsets": output_delta,
            "max_gradient_abs_delta_vs_cpu_offsets": grad_delta,
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
        "expert_count": num_experts,
        "token_count": token_count,
        "fixture": fixture,
        "metadata_integrity": integrity,
        "variants": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("preflight",), default="preflight")
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

    reports = []
    for fixture in ("balanced", "high_skew"):
        report = run_workload(
            model=model,
            num_experts=16,
            token_count=512,
            fixture=fixture,
            device=device,
            warmup=1,
            timed=2,
        )
        reports.append(report)
        print(
            json.dumps(
                {
                    "fixture": fixture,
                    "cpu_forward_ms": round(
                        report["variants"]["cpu_offsets"]["mean_forward_cuda_ns"] / 1e6,
                        3,
                    ),
                    "gpu_scalar_forward_ms": round(
                        report["variants"]["gpu_offsets_scalar"]["mean_forward_cuda_ns"] / 1e6,
                        3,
                    ),
                    "gpu_mask_forward_ms": round(
                        report["variants"]["gpu_segment_mask"]["mean_forward_cuda_ns"] / 1e6,
                        3,
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "v3_source_sha256": sha256_file(Path(__file__).resolve()),
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
        "source_hashes": source_paths,
        "matrix": {
            "expert_counts": [16],
            "token_counts": [512],
            "fixtures": ["balanced", "high_skew"],
            "variants": list(VARIANTS),
            "warmup": 1,
            "timed": 2,
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
            "This is a dispatch-only feasibility preflight using frozen E16 layer-0 expert weights.",
            "GPU metadata construction does not by itself constitute GPU hardware scheduling.",
            "No full-model training, CUDA Graph capture, custom CUDA extension or persistent queue was attempted.",
            "Timed results are preflight evidence only and do not justify a neural scheduler.",
        ],
    }
    summary_path = output / "gpu_resident_metadata_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {
        "gpu_resident_metadata_summary.json": sha256_file(summary_path),
        **source_paths,
    }
    hashes_path = output / "gpu_resident_metadata_hashes.json"
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

