#!/usr/bin/env python3
"""Read-only telemetry-causality diagnostic for Crucible V2.

This module does not edit the frozen context package.  It reuses the existing
diagnostic wrapper, but replaces the GPU telemetry accumulator in memory with
a CPU reference implementation.  The purpose is to test whether the earlier
training-state divergence follows the telemetry reductions themselves or the
grouped expert implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch


PACKAGE_DIR = Path(__file__).resolve().parent
DIAGNOSTIC_DIR = Path("/home/timbushnell/crucible_v2_sustained_validation_20261018")
CONTEXT_SHARED = Path("/home/timbushnell/crucible_context_gate_v1_20261018/shared")
CONFIG_PATH = CONTEXT_SHARED.parent / "context_128" / "crucible_manifest" / "experiment_config.json"

sys.path.insert(0, str(DIAGNOSTIC_DIR))
sys.path.insert(0, str(CONTEXT_SHARED))
import context_runner as frozen_context_runner  # noqa: E402
import run_equivalence_gate as equivalence  # noqa: E402
import run_prefix_reproducibility as prefix  # noqa: E402


class CPUReferenceTelemetry:
    """Deterministic, detached reference accumulator.

    All reduction inputs are copied to CPU before accumulation.  This is
    intentionally a diagnostic implementation, not a performance backend.
    """

    _vocab_size = 259
    _accumulators: Dict[str, Dict[str, Dict[str, Any]]] = {"A": {}, "B": {}, "C": {}}

    @classmethod
    def configure(cls, vocab_size: int) -> None:
        if int(vocab_size) < 3:
            raise ValueError("Telemetry vocabulary must include PAD/BOS/EOS")
        cls._vocab_size = int(vocab_size)

    @classmethod
    def reset_step_buffer(cls) -> None:
        cls._accumulators = {"A": {}, "B": {}, "C": {}}

    @classmethod
    def _init_layer_buffers(cls, branch: str, layer_id: str, num_experts: int) -> None:
        if layer_id in cls._accumulators[branch]:
            return
        cls._accumulators[branch][layer_id] = {
            "tokens_routed": torch.zeros(1, dtype=torch.long),
            "hard_assignments": torch.zeros(num_experts, dtype=torch.float32),
            "entropy_sum": torch.zeros(1, dtype=torch.float32),
            "token_affinity": torch.zeros(num_experts, cls._vocab_size, dtype=torch.float64),
            "selected_token_counts": torch.zeros(num_experts, cls._vocab_size, dtype=torch.float64),
            "top1_counts": torch.zeros(num_experts, dtype=torch.float64),
            "top2_counts": torch.zeros(num_experts, dtype=torch.float64),
            "selected_probability_sum": torch.zeros(num_experts, dtype=torch.float64),
            "selected_probability_count": torch.zeros(num_experts, dtype=torch.float64),
            "activation_norm_sum": torch.zeros(num_experts, dtype=torch.float64),
            "activation_norm_count": torch.zeros(num_experts, dtype=torch.float64),
        }

    @classmethod
    def record_routing(
        cls,
        branch: str,
        layer_id: str,
        probs: torch.Tensor,
        topk_indices: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> None:
        # The copies are the only CUDA-to-host operations in this reference
        # path.  No returned tensor can participate in model autograd.
        with torch.no_grad():
            probabilities = probs.detach().cpu()
            indices = topk_indices.detach().cpu()
            tokens = valid_tokens.detach().cpu()
            num_tokens, num_experts = probabilities.shape
            if indices.ndim != 2 or indices.shape[0] != num_tokens:
                raise ValueError("routing index shape mismatch")
            if tokens.ndim != 1 or tokens.shape[0] != num_tokens:
                raise ValueError("routing token shape mismatch")
            if int(tokens.min()) < 0 or int(tokens.max()) >= cls._vocab_size:
                raise ValueError("token id outside telemetry vocabulary")
            cls._init_layer_buffers(branch, layer_id, num_experts)
            accumulator = cls._accumulators[branch][layer_id]
            top_k = int(indices.shape[1])
            previous_top_k = accumulator.get("top_k")
            if previous_top_k is not None and previous_top_k != top_k:
                raise ValueError("top_k changed within telemetry interval")
            accumulator["top_k"] = top_k
            accumulator["tokens_routed"] += num_tokens
            flat_indices = indices.reshape(-1)
            accumulator["hard_assignments"] += torch.bincount(
                flat_indices, minlength=num_experts
            ).float()
            accumulator["top1_counts"] += torch.bincount(
                indices[:, 0], minlength=num_experts
            ).double()
            accumulator["top2_counts"] += torch.bincount(
                flat_indices, minlength=num_experts
            ).double()
            selected = torch.gather(probabilities, 1, indices)
            selected = selected / (selected.sum(dim=-1, keepdim=True) + 1e-12)
            accumulator["selected_probability_sum"] += torch.bincount(
                flat_indices,
                weights=selected.reshape(-1).double(),
                minlength=num_experts,
            )
            accumulator["selected_probability_count"] += torch.bincount(
                flat_indices,
                weights=torch.ones(indices.numel(), dtype=torch.float64),
                minlength=num_experts,
            )
            accumulator["token_affinity"].index_add_(
                1, tokens, probabilities.double().transpose(0, 1)
            )
            for expert_index in range(num_experts):
                selected_for_expert = (indices == expert_index).sum(dim=1).double()
                if bool(torch.any(selected_for_expert > 0)):
                    accumulator["selected_token_counts"][expert_index].index_add_(
                        0, tokens, selected_for_expert
                    )
            accumulator["entropy_sum"] += (
                -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1).sum()
            )

    @classmethod
    def record_expert_output(
        cls,
        branch: str,
        layer_id: str,
        expert_index: int,
        expert_output: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            output = expert_output.detach().float().cpu()
            accumulator = cls._accumulators[branch][layer_id]
            norms = output.norm(dim=-1).double()
            accumulator["activation_norm_sum"][expert_index] += norms.sum()
            accumulator["activation_norm_count"][expert_index] += norms.numel()

    @classmethod
    def get_summary(cls, branch: str) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for layer, accumulator in cls._accumulators[branch].items():
            total_tokens = accumulator["tokens_routed"].item()
            if total_tokens == 0:
                continue
            hard = accumulator["hard_assignments"]
            global_cv = 0.0 if hard.numel() == 1 else float(
                (hard.std(unbiased=True) / (hard.mean() + 1e-10)).item()
            )
            affinity = accumulator["token_affinity"]
            affinity_mass = float(affinity.sum().item())
            selected_counts = accumulator["selected_token_counts"]
            selected_total = float(selected_counts.sum().item())
            selected_sum = accumulator["selected_probability_sum"]
            selected_count = accumulator["selected_probability_count"]
            activation_sum = accumulator["activation_norm_sum"]
            activation_count = accumulator["activation_norm_count"]
            summary[layer] = {
                "top_k": accumulator.get("top_k"),
                "tokens_routed": int(total_tokens),
                "_token_affinity": affinity.tolist(),
                "selected_token_counts": selected_counts.tolist(),
                "top1_counts": accumulator["top1_counts"].tolist(),
                "top2_counts": accumulator["top2_counts"].tolist(),
                "hard_assignments": hard.tolist(),
                "selected_probability_sum": selected_sum.tolist(),
                "selected_probability_count": selected_count.tolist(),
                "mean_selected_probability": (selected_sum / (selected_count + 1e-12)).tolist(),
                "activation_norm_sum": activation_sum.tolist(),
                "activation_norm_count": activation_count.tolist(),
                "mean_expert_output_norm": (activation_sum / (activation_count + 1e-12)).tolist(),
                "affinity_mass_total": affinity_mass,
                "selected_token_mass_total": selected_total,
                "global_hard_cv": global_cv,
                "max_min_hard_ratio": float((hard.max() / (hard.min() + 1e-10)).item()),
                "normalized_entropy": float(
                    accumulator["entropy_sum"].item() / total_tokens
                    / (math.log(float(hard.numel())) + 1e-10)
                ),
            }
        return summary


class GPUReferenceWeightedCPU(
    frozen_context_runner.ResearchMoETelemetry
):
    """Original GPU telemetry with only weighted bincounts moved to CPU."""

    @classmethod
    def record_routing(
        cls,
        branch: str,
        layer_id: str,
        probs: torch.Tensor,
        topk_indices: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            probabilities = probs.detach()
            indices = topk_indices.detach()
            num_tokens, num_experts = probabilities.shape
            if indices.ndim != 2 or indices.shape[0] != num_tokens:
                raise ValueError("routing index shape mismatch")
            top_k = int(indices.shape[1])
            cls._init_layer_buffers(branch, layer_id, num_experts, probabilities.device)
            accumulator = cls._accumulators[branch][layer_id]
            previous_top_k = accumulator.get("top_k")
            if previous_top_k is not None and previous_top_k != top_k:
                raise ValueError("top_k changed within telemetry interval")
            accumulator["top_k"] = top_k
            accumulator["tokens_routed"] += num_tokens
            flat_indices = indices.reshape(-1)
            accumulator["hard_assignments"] += torch.bincount(
                flat_indices, minlength=num_experts
            ).float()
            accumulator["top1_counts"] += torch.bincount(
                indices[:, 0], minlength=num_experts
            ).double()
            accumulator["top2_counts"] += torch.bincount(
                flat_indices, minlength=num_experts
            ).double()
            normalized_selected = torch.gather(probabilities, 1, indices)
            normalized_selected = normalized_selected / (
                normalized_selected.sum(dim=-1, keepdim=True) + 1e-12
            )

            # These are the two weighted CUDA bincounts that fail the strict
            # deterministic-algorithm contract in the frozen implementation.
            cpu_indices = flat_indices.cpu()
            cpu_selected = normalized_selected.reshape(-1).double().cpu()
            cpu_ones = torch.ones(cpu_indices.numel(), dtype=torch.float64)
            selected_sum = torch.bincount(
                cpu_indices, weights=cpu_selected, minlength=num_experts
            ).to(device=probabilities.device)
            selected_count = torch.bincount(
                cpu_indices, weights=cpu_ones, minlength=num_experts
            ).to(device=probabilities.device)
            accumulator["selected_probability_sum"] += selected_sum
            accumulator["selected_probability_count"] += selected_count

            accumulator["token_affinity"].index_add_(
                1, valid_tokens, probabilities.double().transpose(0, 1)
            )
            for expert_index in range(num_experts):
                selected_for_expert = (indices == expert_index).sum(dim=1).double()
                if torch.any(selected_for_expert > 0):
                    accumulator["selected_token_counts"][expert_index].index_add_(
                        0, valid_tokens, selected_for_expert
                    )
            accumulator["entropy_sum"] += -torch.sum(
                probabilities * torch.log(probabilities + 1e-10), dim=-1
            ).sum()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, choices=(1, 10, 100), default=10)
    parser.add_argument("--repetitions", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--backend",
        choices=("cpu_reference", "weighted_bincount_cpu"),
        default="cpu_reference",
    )
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "cpu_reference")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_PATH.resolve()
    config_sha = sha256_file(config_path)

    # Match the strict settings used by the accepted no-telemetry diagnostic.
    # The CPU reference avoids the weighted CUDA bincount that otherwise raises
    # under warn_only=False.
    prefix.configure_strict_diagnostic_cuda(warn_only=False)

    # Reuse the existing read-only wrapper, but substitute only its in-memory
    # diagnostic telemetry class.  Frozen source and artifacts are untouched.
    telemetry_backend = {
        "cpu_reference": CPUReferenceTelemetry,
        "weighted_bincount_cpu": GPUReferenceWeightedCPU,
    }[args.backend]
    # invoke_runner is imported by prefix from run_equivalence_gate; its
    # globals therefore resolve NoTelemetry in that module, not prefix.
    equivalence.NoTelemetry = telemetry_backend
    telemetry_backend.reset_step_buffer()

    runs: Dict[str, Any] = {}
    for variant in ("original", "grouped_rank_preserving"):
        runs[variant] = []
        for repetition in range(1, args.repetitions + 1):
            output_path = args.output_dir / variant / f"repeat_{repetition}" / "metrics.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"running {variant} repeat {repetition} with CPU reference telemetry "
                f"through step {args.max_steps}",
                flush=True,
            )
            result = prefix.invoke_runner(
                config_path,
                config_sha,
                output_path,
                args.max_steps,
                variant,
                disable_telemetry=True,
            )
            runs[variant].append({
                "metrics_path": str(output_path.resolve()),
                "metrics_sha256": sha256_file(output_path),
                "runtime": result["runtime"],
            })

    original = {int(item["seed"]): item for item in runs["original"][0]["runtime"]}
    grouped = {int(item["seed"]): item for item in runs["grouped_rank_preserving"][0]["runtime"]}
    cross = {}
    for seed in sorted(original):
        cross[str(seed)] = {
            key: original[seed][key] == grouped[seed][key]
            for key in ("model_state_sha256", "gradient_sha256", "optimizer_state_sha256")
        }
    accepted = all(all(values.values()) for values in cross.values())
    reproducibility = {}
    if args.repetitions == 2:
        for variant in runs:
            first = {int(item["seed"]): item for item in runs[variant][0]["runtime"]}
            second = {int(item["seed"]): item for item in runs[variant][1]["runtime"]}
            reproducibility[variant] = {
                str(seed): {
                    key: first[seed][key] == second[seed][key]
                    for key in ("model_state_sha256", "gradient_sha256", "optimizer_state_sha256")
                }
                for seed in sorted(first)
            }
        accepted = accepted and all(
            all(values.values()) for variant in reproducibility.values() for values in variant.values()
        )

    report = {
        "schema_version": "crucible-v2-telemetry-causality-cpu-reference-20261019",
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "max_steps": args.max_steps,
        "repetitions_per_variant": args.repetitions,
        "telemetry_backend": args.backend,
        "frozen_artifacts_modified": False,
        "runs": runs,
        "within_variant_reproducibility": reproducibility,
        "cross_variant_terminal_hashes": cross,
        "accepted": accepted,
        "interpretation": (
            "CPU-reference telemetry does not change terminal training state across variants"
            if accepted else
            "CPU-reference telemetry still shows a terminal training-state mismatch"
        ),
    }
    report_path = args.output_dir / "cpu_reference_causality_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {
        "run_cpu_reference_telemetry.py": sha256_file(Path(__file__).resolve()),
        "cpu_reference_causality_report.json": sha256_file(report_path),
    }
    for variant in runs:
        for repetition, run in enumerate(runs[variant], start=1):
            hashes[f"{variant}/repeat_{repetition}/metrics.jsonl"] = run["metrics_sha256"]
    hashes_path = args.output_dir / "cpu_reference_hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": accepted, "report": str(report_path)}, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
