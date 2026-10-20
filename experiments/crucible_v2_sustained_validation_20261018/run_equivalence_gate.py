#!/usr/bin/env python3
"""Same-device 100-step equivalence gate for the V2 Crucible engine.

This is deliberately separate from the sustained validation wrapper.  It runs
the frozen context runner twice on the same device: once unchanged and once
with the V2 rank-preserving grouped MoE patch installed in memory.  It captures
the final model, gradients and optimizer state without changing the frozen
runner, manifest, checkpoints or training artifacts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


CONTEXT_SHARED = Path("/home/timbushnell/crucible_context_gate_v1_20261018/shared")
V1_DIR = Path("/home/timbushnell/gpu_neural_ssre_v1_grouped_expert_20261018")
V2_DIR = Path("/home/timbushnell/gpu_neural_ssre_v2_strategy_landscape_20261018")
PACKAGE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(CONTEXT_SHARED))
sys.path.insert(0, str(V1_DIR))
sys.path.insert(0, str(V2_DIR))

import context_runner  # noqa: E402
import strategy_landscape_benchmark as v2  # noqa: E402
import torch  # noqa: E402


SCHEMA_VERSION = "crucible-v2-same-device-equivalence-gate-20261018"
METRIC_TOLERANCE = float(v2.LOSS_TOL)
GRADIENT_TOLERANCE = float(v2.GRAD_TOL)
TELEMETRY_TOLERANCE = 1e-5

EXACT_METRIC_FIELDS = (
    "seed",
    "branch",
    "peak_lr",
    "context_window_length",
    "d_ff",
    "num_experts",
    "top_k",
    "interval_step_end",
    "interval_train_examples",
    "interval_train_raw_byte_targets",
    "interval_train_source_bytes",
    "interval_nan_inf_count",
    "validation_raw_byte_targets",
    "validation_source_bytes",
    "validation_eos_targets_excluded",
    "model_parameter_count",
    "master_config_sha256",
)

NUMERIC_METRIC_FIELDS = (
    "interval_train_ce",
    "interval_train_raw_byte_nll",
    "interval_branch_b_aux_loss",
    "grad_norm_mean",
    "grad_norm_max",
    "grad_norm_p95",
    "interval_clipped_fraction",
    "validation_nll",
    "validation_raw_byte_nll",
    "validation_bits_per_raw_byte",
    "validation_raw_byte_perplexity",
    "byte_perplexity",
    "router_interval_cv",
    "router_interval_entropy",
    "router_interval_load_ratio",
)


def configure_strict_diagnostic_cuda(warn_only: bool = False) -> None:
    """Use strict deterministic settings for reproducibility diagnostics."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required by the frozen contract")
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gradient_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if parameter.grad is None:
            digest.update(b"NONE\0")
            continue
        value = parameter.grad.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def capture_gradients_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


class NoTelemetry:
    """Diagnostic-only telemetry stub; never used for scientific outputs."""

    @classmethod
    def configure(cls, vocab_size: int) -> None:
        return None

    @classmethod
    def reset_step_buffer(cls) -> None:
        return None

    @classmethod
    def record_routing(cls, *args: Any, **kwargs: Any) -> None:
        return None

    @classmethod
    def record_expert_output(cls, *args: Any, **kwargs: Any) -> None:
        return None

    @classmethod
    def get_summary(cls, branch: str) -> Dict[str, Any]:
        return {}


def invoke_runner(
    config_path: Path,
    config_sha256: str,
    output_path: Path,
    max_steps: int,
    variant: str,
    disable_telemetry: bool = False,
) -> Dict[str, Any]:
    """Invoke the frozen runner while capturing terminal runtime objects."""

    captured_models = []
    captured_optimizers = []
    original_constructor = context_runner.BraidPrototypeLM
    original_adamw = context_runner.torch.optim.AdamW
    original_run_trial = context_runner.run_trial
    original_telemetry = context_runner.ResearchMoETelemetry

    def captured_constructor(*args: Any, **kwargs: Any):
        model = original_constructor(*args, **kwargs)
        if variant == "grouped_rank_preserving":
            v2.patch_model(model, variant)
        captured_models.append(model)
        return model

    def captured_adamw(*args: Any, **kwargs: Any):
        optimizer = original_adamw(*args, **kwargs)
        captured_optimizers.append(optimizer)
        return optimizer

    runtime = []

    def captured_run_trial(seed: int, *args: Any, **kwargs: Any):
        trajectory = original_run_trial(seed, *args, **kwargs)
        if len(captured_models) != 1 or len(captured_optimizers) != 1:
            raise RuntimeError(
                "Expected exactly one model and optimizer per completed trial"
            )
        model = captured_models[0]
        optimizer = captured_optimizers[0]
        runtime.append(
            {
                "seed": int(seed),
                "model_state_sha256": v2.v1.state_digest(model),
                "gradient_sha256": gradient_digest(model),
                "optimizer_state_sha256": v2.optimizer_state_digest(optimizer),
            }
        )
        captured_models.clear()
        captured_optimizers.clear()
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return trajectory

    context_runner.BraidPrototypeLM = captured_constructor
    context_runner.torch.optim.AdamW = captured_adamw
    context_runner.run_trial = captured_run_trial
    if disable_telemetry:
        context_runner.ResearchMoETelemetry = NoTelemetry

    delegated = [
        "context_runner.py",
        "--config-path",
        str(config_path.resolve()),
        "--config-sha256",
        config_sha256.strip().lower(),
        "--output-path",
        str(output_path.resolve()),
        "--max-steps",
        str(max_steps),
    ]

    saved_argv = sys.argv
    try:
        sys.argv = delegated
        return_code = context_runner.main()
    finally:
        sys.argv = saved_argv
        context_runner.BraidPrototypeLM = original_constructor
        context_runner.torch.optim.AdamW = original_adamw
        context_runner.run_trial = original_run_trial
        context_runner.ResearchMoETelemetry = original_telemetry

    if return_code != 0:
        raise RuntimeError(f"Frozen context runner returned {return_code}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    seeds = [int(seed) for seed in config["contract"]["seeds"]]

    if len(runtime) != len(seeds):
        raise RuntimeError(f"Captured {len(runtime)} trials; expected {len(seeds)}")
    if [int(item["seed"]) for item in runtime] != seeds:
        raise RuntimeError("Captured trial seed order does not match manifest")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "variant": variant,
        "runtime": runtime,
        "output_path": str(output_path.resolve()),
    }


def read_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            seed = int(row["seed"])
            if seed in rows:
                raise ValueError(f"Duplicate seed {seed} in {path} line {line_number}")
            rows[seed] = row
    return rows


def close_enough(a: Any, b: Any, tolerance: float) -> bool:
    try:
        left = float(a)
        right = float(b)
    except (TypeError, ValueError):
        return a == b
    if not math.isfinite(left) or not math.isfinite(right):
        return left == right
    return abs(left - right) <= tolerance


def compare_layer_telemetry(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    if set(reference) != set(candidate):
        return {"same_layers": False, "accepted": False}

    count_fields = (
        "top_k",
        "tokens_routed",
        "selected_token_counts",
        "top1_counts",
        "top2_counts",
    )
    float_fields = (
        "affinity_mass_total",
        "selected_probability_count",
        "selected_probability_sum",
        "selected_token_mass_total",
        "activation_norm_count",
        "activation_norm_sum",
        "mean_expert_output_norm",
        "mean_selected_probability",
        "global_hard_cv",
        "max_min_hard_ratio",
        "normalized_entropy",
    )

    layer_reports = {}
    accepted = True
    max_affinity_delta = 0.0
    for layer_name in sorted(reference):
        ref_layer = reference[layer_name]
        cand_layer = candidate[layer_name]
        counts_equal = all(ref_layer.get(key) == cand_layer.get(key) for key in count_fields)
        float_deltas = {}
        for key in float_fields:
            try:
                delta = abs(float(ref_layer[key]) - float(cand_layer[key]))
            except (KeyError, TypeError, ValueError):
                delta = float("inf")
            float_deltas[key] = delta
        try:
            ref_affinity = torch.tensor(ref_layer["_token_affinity"], dtype=torch.float64)
            cand_affinity = torch.tensor(cand_layer["_token_affinity"], dtype=torch.float64)
            affinity_delta = float((ref_affinity - cand_affinity).abs().max().item())
        except (KeyError, RuntimeError, ValueError):
            affinity_delta = float("inf")
        max_affinity_delta = max(max_affinity_delta, affinity_delta)
        layer_ok = counts_equal and affinity_delta <= TELEMETRY_TOLERANCE
        layer_ok = layer_ok and all(
            delta <= TELEMETRY_TOLERANCE for delta in float_deltas.values()
        )
        accepted = accepted and layer_ok
        layer_reports[layer_name] = {
            "count_fields_equal": counts_equal,
            "float_deltas": float_deltas,
            "max_affinity_abs_delta": affinity_delta,
            "accepted": layer_ok,
        }

    return {
        "same_layers": True,
        "max_affinity_abs_delta": max_affinity_delta,
        "tolerance": TELEMETRY_TOLERANCE,
        "layers": layer_reports,
        "accepted": accepted,
    }


def compare_rows(
    reference_rows: Mapping[int, Mapping[str, Any]],
    candidate_rows: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    if set(reference_rows) != set(candidate_rows):
        return {
            "same_seeds": False,
            "accepted": False,
            "reference_seeds": sorted(reference_rows),
            "candidate_seeds": sorted(candidate_rows),
        }

    reports = {}
    accepted = True
    for seed in sorted(reference_rows):
        ref = reference_rows[seed]
        cand = candidate_rows[seed]
        exact_mismatches = {
            key: {"reference": ref.get(key), "candidate": cand.get(key)}
            for key in EXACT_METRIC_FIELDS
            if ref.get(key) != cand.get(key)
        }
        numeric_deltas = {
            key: abs(float(ref[key]) - float(cand[key]))
            for key in NUMERIC_METRIC_FIELDS
        }
        numeric_ok = all(delta <= METRIC_TOLERANCE for delta in numeric_deltas.values())
        telemetry = compare_layer_telemetry(
            ref["layer_telemetry"],
            cand["layer_telemetry"],
        )
        seed_ok = not exact_mismatches and numeric_ok and telemetry["accepted"]
        accepted = accepted and seed_ok
        reports[str(seed)] = {
            "exact_field_mismatches": exact_mismatches,
            "numeric_deltas": numeric_deltas,
            "numeric_tolerance": METRIC_TOLERANCE,
            "numeric_fields_within_tolerance": numeric_ok,
            "telemetry": telemetry,
            "accepted": seed_ok,
        }

    return {"same_seeds": True, "seeds": reports, "accepted": accepted}


def compare_runtime(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    reference_by_seed = {int(item["seed"]): item for item in reference}
    candidate_by_seed = {int(item["seed"]): item for item in candidate}
    if set(reference_by_seed) != set(candidate_by_seed):
        return {"same_seeds": False, "accepted": False}

    reports = {}
    accepted = True
    for seed in sorted(reference_by_seed):
        ref = reference_by_seed[seed]
        cand = candidate_by_seed[seed]
        model_equal = ref["model_state_sha256"] == cand["model_state_sha256"]
        gradient_equal = ref["gradient_sha256"] == cand["gradient_sha256"]
        optimizer_equal = ref["optimizer_state_sha256"] == cand["optimizer_state_sha256"]
        seed_ok = model_equal and gradient_equal and optimizer_equal
        accepted = accepted and seed_ok
        reports[str(seed)] = {
            "model_state_sha256": {
                "reference": ref["model_state_sha256"],
                "candidate": cand["model_state_sha256"],
                "equal": model_equal,
            },
            "gradient_sha256": {
                "reference": ref["gradient_sha256"],
                "candidate": cand["gradient_sha256"],
                "equal": gradient_equal,
            },
            "optimizer_state_sha256": {
                "reference": ref["optimizer_state_sha256"],
                "candidate": cand["optimizer_state_sha256"],
                "equal": optimizer_equal,
            },
            "comparison_contract": "exact SHA-256 equality for terminal model, gradient and optimizer state",
            "accepted": seed_ok,
        }

    return {"same_seeds": True, "seeds": reports, "accepted": accepted}


def scrub_runtime(runtime: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    return [
        {
            key: value
            for key, value in item.items()
            if key not in ("state_cpu", "gradients_cpu")
        }
        for item in runtime
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    config_path = args.config_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_sha256 = sha256_file(config_path)
    if config_sha256 != args.config_sha256.strip().lower():
        raise ValueError("Config SHA-256 does not match --config-sha256")

    configure_strict_diagnostic_cuda()

    original_metrics = output_dir / "original_metrics.jsonl"
    candidate_metrics = output_dir / "grouped_rank_preserving_metrics.jsonl"

    print("[1/2] same-device unchanged reference")
    original = invoke_runner(
        config_path,
        config_sha256,
        original_metrics,
        args.max_steps,
        "original",
    )
    print("[2/2] same-device V2 grouped_rank_preserving")
    candidate = invoke_runner(
        config_path,
        config_sha256,
        candidate_metrics,
        args.max_steps,
        "grouped_rank_preserving",
    )

    metric_report = compare_rows(
        read_rows(original_metrics),
        read_rows(candidate_metrics),
    )
    runtime_report = compare_runtime(original["runtime"], candidate["runtime"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "max_steps": args.max_steps,
        "reference_variant": "original",
        "candidate_variant": "grouped_rank_preserving",
        "metric_report": metric_report,
        "runtime_equivalence_report": runtime_report,
        "accepted": metric_report["accepted"] and runtime_report["accepted"],
        "notes": [
            "Both variants used the frozen context runner and the same local CUDA device.",
            "Timing fields are intentionally excluded from equivalence checks.",
            "The grouped patch was installed only in memory; frozen source and artifacts were not modified.",
        ],
    }
    report_path = output_dir / "equivalence_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    hashes = {
        "run_equivalence_gate.py": sha256_file(Path(__file__).resolve()),
        "original_metrics.jsonl": sha256_file(original_metrics),
        "grouped_rank_preserving_metrics.jsonl": sha256_file(candidate_metrics),
        "equivalence_report.json": sha256_file(report_path),
        "config": config_sha256,
    }
    hashes_path = output_dir / "equivalence_hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "accepted": report["accepted"],
        "original_metrics_sha256": hashes["original_metrics.jsonl"],
        "candidate_metrics_sha256": hashes["grouped_rank_preserving_metrics.jsonl"],
        "report": str(report_path),
    }, sort_keys=True))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
