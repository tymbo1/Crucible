#!/usr/bin/env python3
"""GPU Neural SSRE v5 deterministic expert-concurrency preflight."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import torch


V1_DIR = Path("/home/timbushnell/gpu_neural_ssre_v1_grouped_expert_20261018")
V2_DIR = Path("/home/timbushnell/gpu_neural_ssre_v2_strategy_landscape_20261018")
CONTEXT_ROOT = Path("/home/timbushnell/crucible_context_gate_v1_20261018/context_128")
SHARED_ROOT = CONTEXT_ROOT.parent
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(V2_DIR))

import dispatch_strategy_landscape as v2  # noqa: E402
import grouped_expert_benchmark as v1  # noqa: E402


SCHEMA_VERSION = "gpu-neural-ssre-v5-concurrency-20261018"
FORWARD_TOL = float(v1.FORWARD_TOL)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_ns() -> int:
    return time.perf_counter_ns()


def stream_dispatch(
    experts: Sequence[torch.nn.Module],
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    stream_count: int,
) -> tuple[torch.Tensor, Dict[str, int]]:
    assignments = v2.sorted_assignments(indices, weights, len(experts))
    sorted_tokens = assignments["sorted_tokens"]
    sorted_weights = assignments["sorted_weights"]
    pair_offsets = assignments["pair_offsets"]
    top_k = int(indices.shape[1])
    device = x.device
    default_stream = torch.cuda.current_stream(device)
    streams = [torch.cuda.Stream(device=device) for _ in range(stream_count)]
    segment_results: list[tuple[torch.Tensor, torch.Tensor]] = []
    segment_count = 0

    for expert_index, expert in enumerate(experts):
        for rank in range(top_k):
            pair_id = expert_index * top_k + rank
            start = pair_offsets[pair_id]
            end = pair_offsets[pair_id + 1]
            if end <= start:
                continue
            stream = streams[segment_count % stream_count]
            stream.wait_stream(default_stream)
            token_indices = sorted_tokens[start:end]
            segment_weights = sorted_weights[start:end]
            with torch.cuda.stream(stream):
                expert_input = x.index_select(0, token_indices)
                expert_out = expert(expert_input)
                weighted = (
                    expert_out
                    * segment_weights.to(expert_out.dtype).unsqueeze(1)
                ).to(x.dtype)
            segment_results.append((token_indices, weighted))
            segment_count += 1

    for stream in streams:
        default_stream.wait_stream(stream)

    output = torch.zeros_like(x)
    for token_indices, weighted in segment_results:
        output.index_put_((token_indices,), weighted, accumulate=True)

    return output, {
        "stream_count": stream_count,
        "active_segments": segment_count,
    }


def reference_dispatch(
    experts: Sequence[torch.nn.Module],
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, int]]:
    return (
        v2.dispatch_grouped(
            experts,
            x,
            weights,
            indices,
            "grouped_rank_preserving",
        ),
        {"stream_count": 1, "active_segments": -1},
    )


def timed_dispatch(
    variant: str,
    experts: Sequence[torch.nn.Module],
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    stream_count: int,
) -> Dict[str, Any]:
    torch.cuda.synchronize(x.device)
    host_start = now_ns()
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    event_start.record()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if variant == "reference":
            output, engine_info = reference_dispatch(experts, x, weights, indices)
        else:
            output, engine_info = stream_dispatch(
                experts,
                x,
                weights,
                indices,
                stream_count,
            )
    event_end.record()
    event_end.synchronize()
    return {
        "cuda_ns": int(event_start.elapsed_time(event_end) * 1_000_000.0),
        "host_ns": int(now_ns() - host_start),
        "output": output.detach().float().cpu(),
        "engine_info": engine_info,
        "finite": bool(torch.isfinite(output).all().item()),
    }


def run_case(
    model: torch.nn.Module,
    token_count: int,
    fixture: str,
    device: torch.device,
    warmup: int = 1,
    timed: int = 3,
) -> Dict[str, Any]:
    num_experts = 16
    experts = list(model.layers[0].moe.experts[:num_experts])
    torch.manual_seed(20261018 + token_count)
    hidden_dim = int(experts[0].w1.in_features)
    x = torch.randn(token_count, hidden_dim, device=device)
    weights = (
        torch.tensor([0.6, 0.4], device=device, dtype=torch.float32)
        .view(1, 2)
        .expand(token_count, -1)
        .contiguous()
    )
    indices = v2.fixture_indices(fixture, token_count, num_experts, device)
    reports: Dict[str, Any] = {}

    with torch.no_grad():
        for variant, stream_count in [("reference", 1), ("streams", 2), ("streams", 4), ("streams", 8)]:
            key = "reference" if variant == "reference" else f"streams_{stream_count}"
            for _ in range(warmup):
                warm = timed_dispatch(variant, experts, x, weights, indices, stream_count)
                del warm
            trials = [
                timed_dispatch(variant, experts, x, weights, indices, stream_count)
                for _ in range(timed)
            ]
            reference_output = reports.get("_reference_output")
            if reference_output is None:
                reports["_reference_output"] = trials[0]["output"]
                reference_output = trials[0]["output"]
            max_delta = float(
                (trials[0]["output"] - reference_output).abs().max().item()
            )
            reports[key] = {
                "mean_cuda_ns": float(sum(item["cuda_ns"] for item in trials) / len(trials)),
                "mean_host_ns": float(sum(item["host_ns"] for item in trials) / len(trials)),
                "engine_info": trials[0]["engine_info"],
                "max_output_abs_delta_vs_reference": max_delta,
                "forward_tolerance": FORWARD_TOL,
                "strict_output_pass": max_delta <= FORWARD_TOL,
                "all_finite": all(item["finite"] for item in trials),
            }
            del trials
            torch.cuda.empty_cache()

    reports.pop("_reference_output", None)
    del experts, x, weights, indices
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "token_count": token_count,
        "fixture": fixture,
        "variants": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
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
    for token_count in (512, 2048, 4096):
        for fixture in ("balanced", "high_skew"):
            report = run_case(model, token_count, fixture, device)
            reports.append(report)
            print(
                json.dumps(
                    {
                        "token_count": token_count,
                        "fixture": fixture,
                        "reference_ms": round(report["variants"]["reference"]["mean_cuda_ns"] / 1e6, 3),
                        "streams_2_ms": round(report["variants"]["streams_2"]["mean_cuda_ns"] / 1e6, 3),
                        "streams_4_ms": round(report["variants"]["streams_4"]["mean_cuda_ns"] / 1e6, 3),
                        "streams_8_ms": round(report["variants"]["streams_8"]["mean_cuda_ns"] / 1e6, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        "v5_source_sha256": sha256_file(Path(__file__).resolve()),
        "dependency_audit_sha256": sha256_file(Path(__file__).with_name("DEPENDENCY_AUDIT.md")),
        "v2_dispatch_source_sha256": sha256_file(V2_DIR / "dispatch_strategy_landscape.py"),
        "v1_source_sha256": sha256_file(V1_DIR / "grouped_expert_benchmark.py"),
        "model_core_sha256": sha256_file(SHARED_ROOT / "shared" / "model_core.py"),
        "config_sha256": sha256_file(config_path),
        "anchor_sha256": sha256_file(anchor_path),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "source_hashes": source_hashes,
        "matrix": {
            "expert_count": 16,
            "token_counts": [512, 2048, 4096],
            "fixtures": ["balanced", "high_skew"],
            "stream_counts": [2, 4, 8],
            "warmup": 1,
            "timed": 3,
        },
        "reports": reports,
        "integrity": {
            "all_stream_output_pass": all(
                report["variants"][variant]["strict_output_pass"]
                for report in reports
                for variant in ("streams_2", "streams_4", "streams_8")
            ),
            "all_outputs_finite": all(
                report["variants"][variant]["all_finite"]
                for report in reports
                for variant in ("reference", "streams_2", "streams_4", "streams_8")
            ),
        },
        "interpretation_limits": [
            "Forward-only dispatch feasibility; no backward or optimizer equivalence was tested.",
            "CUDA streams are a deterministic concurrency control, not a neural scheduler.",
            "A stream plan is not eligible for full-model integration without a separate training-state gate.",
        ],
    }
    summary_path = output / "concurrency_preflight_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {
        "concurrency_preflight_summary.json": sha256_file(summary_path),
        **source_hashes,
    }
    hashes_path = output / "concurrency_preflight_hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(summary_path),
                "all_stream_output_pass": summary["integrity"]["all_stream_output_pass"],
                "all_outputs_finite": summary["integrity"]["all_outputs_finite"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

