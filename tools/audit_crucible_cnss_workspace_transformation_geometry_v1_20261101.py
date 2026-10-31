#!/usr/bin/env python3
"""Read-only geometry audit of canonical workspace transformations A--D.

This measures properties of the frozen encoder/decoder and canonical target
states. It does not load or update learned skill adapters, so the results are
workspace-geometry proxies rather than evidence about adapter optimization.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
import sys
sys.path.insert(0, str(TOOLS))

import run_crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101 as control
import run_crucible_track_a_cnss_partner_generalization_v1_20261101 as partner

OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_workspace_transformation_geometry_audit_v1_20261101"
SEALED_OUTPUT = Path("/home/timbushnell/crucible_artifacts/crucible_cnss_workspace_transformation_geometry_audit_v1_20261101.json")
EXPECTED_WORKSPACE_DIGEST = control.EXPECTED_WORKSPACE_DIGEST
VALUES = list(range(256))
SKILLS = ("A", "B", "C", "D")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def pearson(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if denominator.item() == 0.0:
        return 0.0
    return float((first @ second / denominator).item())


def pairwise_upper(values: torch.Tensor) -> torch.Tensor:
    matrix = torch.cdist(values, values)
    indices = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1)
    return matrix[indices[0], indices[1]]


def rank_energy(singular_values: torch.Tensor, rank: int) -> float:
    total = torch.sum(singular_values.square())
    if total.item() == 0.0:
        return 1.0
    return float((torch.sum(singular_values[:rank].square()) / total).item())


def numerical_rank(singular_values: torch.Tensor) -> int:
    if singular_values.numel() == 0 or singular_values[0].item() == 0.0:
        return 0
    threshold = singular_values[0].item() * 1e-6
    return int((singular_values > threshold).sum().item())


def affine_metrics(input_state: torch.Tensor, target_state: torch.Tensor) -> dict[str, Any]:
    ones = torch.ones((input_state.shape[0], 1), dtype=input_state.dtype)
    augmented = torch.cat([input_state, ones], dim=1)
    solution = torch.linalg.lstsq(augmented, target_state).solution
    fitted = augmented @ solution
    residual = fitted - target_state
    target_centered = target_state - target_state.mean(dim=0, keepdim=True)
    total = torch.sum(target_centered.square())
    residual_sum = torch.sum(residual.square())
    weight = solution[:-1]
    bias = solution[-1]
    singular_values = torch.linalg.svdvals(weight)
    rank_metrics: dict[str, Any] = {}
    for rank in (8, 32, 64, 128, 256):
        if rank > singular_values.numel():
            continue
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        truncated = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
        rank_fitted = input_state @ truncated + bias
        rank_residual = rank_fitted - target_state
        rank_metrics[str(rank)] = {
            "rmse": float(torch.sqrt(torch.mean(rank_residual.square())).item()),
            "relative_frobenius_error": float(torch.linalg.vector_norm(rank_residual).item() / max(torch.linalg.vector_norm(target_state).item(), 1e-12)),
        }
    return {
        "full_affine_rmse": float(torch.sqrt(torch.mean(residual.square())).item()),
        "full_affine_relative_frobenius_error": float(torch.linalg.vector_norm(residual).item() / max(torch.linalg.vector_norm(target_state).item(), 1e-12)),
        "full_affine_r2": float((1.0 - residual_sum / max(total, torch.tensor(1e-12))).item()),
        "coefficient_numerical_rank": numerical_rank(singular_values),
        "coefficient_effective_rank": float((singular_values.square().sum().square() / singular_values.pow(4).sum().clamp_min(1e-12)).item()),
        "coefficient_singular_values_top": [float(value) for value in singular_values[:8].tolist()],
        "rank_truncated_affine": rank_metrics,
    }


def skill_target(skill: str, values: torch.Tensor) -> torch.Tensor:
    return partner.skill_target(skill, values)


def skill_metrics(workspace: partner.runner.Workspace, skill: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = torch.tensor(VALUES, dtype=torch.int64)
    inputs = workspace.encode(partner.runner.bits_from_int(values))
    mapped_values = skill_target(skill, values)
    canonical_targets = workspace.encode(partner.runner.bits_from_int(mapped_values))
    if not finite_tensor(inputs) or not finite_tensor(canonical_targets):
        raise RuntimeError(f"non-finite workspace states for skill {skill}")
    deltas = canonical_targets - inputs
    delta_singular_values = torch.linalg.svdvals(deltas)
    target_logits = workspace.decode_logits(canonical_targets)
    target_bits = partner.runner.bits_from_int(mapped_values)
    exact, _ = partner.runner.exact_bits(target_logits, target_bits)
    signed_margin = torch.where(target_bits >= 0.5, target_logits, -target_logits)
    input_pairwise = pairwise_upper(inputs)
    target_pairwise = pairwise_upper(canonical_targets)
    neighbor_rows: list[dict[str, Any]] = []
    for bit in range(8):
        for value in VALUES:
            if ((value >> bit) & 1) == 0:
                other = value | (1 << bit)
                delta_step = deltas[other] - deltas[value]
                input_step = inputs[other] - inputs[value]
                target_step = canonical_targets[other] - canonical_targets[value]
                neighbor_rows.append({
                    "skill": skill,
                    "value": value,
                    "other": other,
                    "bit": bit,
                    "delta_step_norm": float(torch.linalg.vector_norm(delta_step).item()),
                    "input_step_norm": float(torch.linalg.vector_norm(input_step).item()),
                    "target_step_norm": float(torch.linalg.vector_norm(target_step).item()),
                })
    neighbor_delta_norms = torch.tensor([row["delta_step_norm"] for row in neighbor_rows])
    input_step_norms = torch.tensor([row["input_step_norm"] for row in neighbor_rows])
    target_step_norms = torch.tensor([row["target_step_norm"] for row in neighbor_rows])
    summary = {
        "skill": skill,
        "operation": {
            "A": "rotate-left by 1",
            "B": "xor with 0x59",
            "C": "high/low nibble swap",
            "D": "modular addition by 0x37",
        }[skill],
        "state_count": len(VALUES),
        "canonical_decoder_exact_accuracy": exact,
        "canonical_decoder_mean_signed_margin": float(signed_margin.mean().item()),
        "canonical_decoder_min_signed_margin": float(signed_margin.min().item()),
        "input_state_norm_mean": float(torch.linalg.vector_norm(inputs, dim=1).mean().item()),
        "target_state_norm_mean": float(torch.linalg.vector_norm(canonical_targets, dim=1).mean().item()),
        "canonical_displacement": {
            "mean": float(torch.linalg.vector_norm(deltas, dim=1).mean().item()),
            "std": float(torch.linalg.vector_norm(deltas, dim=1).std(unbiased=False).item()),
            "min": float(torch.linalg.vector_norm(deltas, dim=1).min().item()),
            "max": float(torch.linalg.vector_norm(deltas, dim=1).max().item()),
            "mean_cosine_input_target": float(torch.nn.functional.cosine_similarity(inputs, canonical_targets, dim=1).mean().item()),
        },
        "delta_output_subspace": {
            "numerical_rank": numerical_rank(delta_singular_values),
            "effective_rank": float((delta_singular_values.square().sum().square() / delta_singular_values.pow(4).sum().clamp_min(1e-12)).item()),
            "top_singular_values": [float(value) for value in delta_singular_values[:8].tolist()],
            "energy_at_rank_8": rank_energy(delta_singular_values, 8),
            "energy_at_rank_32": rank_energy(delta_singular_values, 32),
            "energy_at_rank_64": rank_energy(delta_singular_values, 64),
        },
        "pairwise_geometry": {
            "input_target_distance_pearson": pearson(input_pairwise, target_pairwise),
            "input_pairwise_distance_mean": float(input_pairwise.mean().item()),
            "target_pairwise_distance_mean": float(target_pairwise.mean().item()),
            "input_target_pairwise_distance_rmse": float(torch.sqrt(torch.mean((input_pairwise - target_pairwise).square())).item()),
        },
        "affine_accessibility_proxy": affine_metrics(inputs, canonical_targets),
        "one_bit_neighbor_delta": {
            "count": len(neighbor_rows),
            "mean_delta_step_norm": float(neighbor_delta_norms.mean().item()),
            "max_delta_step_norm": float(neighbor_delta_norms.max().item()),
            "mean_input_step_norm": float(input_step_norms.mean().item()),
            "mean_target_step_norm": float(target_step_norms.mean().item()),
            "delta_step_vs_input_step_pearson": pearson(neighbor_delta_norms, input_step_norms),
            "delta_step_vs_target_step_pearson": pearson(neighbor_delta_norms, target_step_norms),
        },
    }
    return summary, neighbor_rows


def public_answer_free(value: Any) -> bool:
    if isinstance(value, dict):
        return all(key not in {"target", "targets", "prediction", "predictions"} and public_answer_free(item) for key, item in value.items())
    if isinstance(value, list):
        return all(public_answer_free(item) for item in value)
    return True


def main() -> int:
    workspace = control.load_workspace()
    if partner.runner.state_dict_digest(workspace) != EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError("workspace digest mismatch")
    if not partner.runner.finite_module(workspace):
        raise RuntimeError("workspace parameters are non-finite")
    summaries = []
    neighbor_rows = []
    for skill in SKILLS:
        summary, rows = skill_metrics(workspace, skill)
        summaries.append(summary)
        neighbor_rows.extend(rows)
    public = {
        "schema_version": "crucible-cnss-workspace-transformation-geometry-audit-v1-20261101",
        "classification": "READ_ONLY_WORKSPACE_TRANSFORMATION_GEOMETRY_AUDIT",
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "workspace_artifact": "experiments/crucible_track_a_cnss_semantic_interface_compatibility_v1_rerun_contractfix_20261101/workspace_artifacts/a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c",
        "skills": list(SKILLS),
        "state_count_per_skill": len(VALUES),
        "training": {"parameter_updates": False, "optimizer_updates": False, "gpu_used": False},
        "summaries": summaries,
        "interpretation_boundary": "These are descriptive geometry and low-rank accessibility proxies of canonical workspace transformations. They do not establish adapter optimization difficulty, a causal workspace limitation, new learning or architecture superiority.",
        "next_step_boundary": "A causal intervention would require a separately registered matched training control; no such intervention is authorized by this audit.",
    }
    sealed = {
        "schema_version": "crucible-cnss-workspace-transformation-geometry-audit-sealed-v1-20261101",
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "public_summary_without_sealed_hash_sha256": digest_json(public),
        "neighbor_rows": neighbor_rows,
    }
    write_json(SEALED_OUTPUT, sealed)
    public["sealed_detail_sha256"] = digest_bytes(SEALED_OUTPUT.read_bytes())
    if not public_answer_free(public):
        raise RuntimeError("public audit contains answer-bearing keys")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "WORKSPACE_TRANSFORMATION_GEOMETRY_AUDIT.json", public)
    write_json(OUTPUT / "WORKSPACE_TRANSFORMATION_GEOMETRY_SEALED_HASH.json", {"sealed_detail_sha256": public["sealed_detail_sha256"], "sealed_path": str(SEALED_OUTPUT)})
    lines = [
        "# CNSS workspace transformation geometry audit",
        "",
        "Classification: `READ_ONLY_WORKSPACE_TRANSFORMATION_GEOMETRY_AUDIT`.",
        "",
        "This compares canonical encoder-space transformations for skills A–D across all 256 states. It uses the exact recovered frozen workspace and performs no adapter or optimizer updates.",
        "",
        "| Skill | Canonical exact | Mean displacement | Delta rank | Rank-32 delta energy | Pairwise distance correlation | Affine rank-32 RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['skill']} | {summary['canonical_decoder_exact_accuracy']:.4f} | "
            f"{summary['canonical_displacement']['mean']:.4f} | "
            f"{summary['delta_output_subspace']['numerical_rank']} | "
            f"{summary['delta_output_subspace']['energy_at_rank_32']:.4f} | "
            f"{summary['pairwise_geometry']['input_target_distance_pearson']:.4f} | "
            f"{summary['affine_accessibility_proxy']['rank_truncated_affine']['32']['rmse']:.4f} |"
        )
    lines.extend([
        "",
        "These measurements are accessibility proxies, not a causal proof that any skill is harder for the registered adapter. D-specific evidence would require a matched training intervention after review.",
        "",
        "Full machine-readable results: `WORKSPACE_TRANSFORMATION_GEOMETRY_AUDIT.json`.",
    ])
    (OUTPUT / "WORKSPACE_TRANSFORMATION_GEOMETRY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": public["classification"],
        "workspace_digest": public["workspace_digest"],
        "summary": [{
            "skill": item["skill"],
            "delta_rank": item["delta_output_subspace"]["numerical_rank"],
            "rank32_delta_energy": item["delta_output_subspace"]["energy_at_rank_32"],
            "pairwise_correlation": item["pairwise_geometry"]["input_target_distance_pearson"],
            "affine_rank32_rmse": item["affine_accessibility_proxy"]["rank_truncated_affine"]["32"]["rmse"],
        } for item in summaries],
        "sealed_detail_sha256": public["sealed_detail_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
