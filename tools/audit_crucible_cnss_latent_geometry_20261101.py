#!/usr/bin/env python3
"""No-training CNSS latent-geometry causal audit.

This audit loads the content-addressed workspace artifact and the existing
final adapter checkpoints. It never calls the workspace trainer, performs no
optimizer update, and does not modify model parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import run_crucible_track_a_cnss_semantic_interface_compatibility_v1_20261101 as runner
from verify_crucible_cnss_workspace_artifact_20261101 import load_bundle


RUN_ROOT = ROOT / "experiments" / "crucible_track_a_cnss_semantic_interface_compatibility_v1_rerun_contractfix_20261101"
OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_latent_geometry_causal_audit_v1_20261101"
ARTIFACT = RUN_ROOT / "workspace_artifacts" / "a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c"
EXPECTED_WORKSPACE_DIGEST = "a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c"
SEEDS = (42, 1337, 2025)
ALPHAS = tuple(index / 20.0 for index in range(21))
PCA_AXES = 8
PROBE_STATES = 256

CONDITIONS = (
    ("lambda_0.1", "B_shared_contract_primary_lambda_0.1"),
    ("lambda_1.0", "B_shared_contract_secondary_lambda_1.0"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_rows(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted = (logits >= 0.0).to(torch.float32)
    return (predicted == target).all(dim=1)


def stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().cpu().to(torch.float64)
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "max": float(values.max().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def adapter_path(condition_dir: str, seed: int, skill: str) -> Path:
    return RUN_ROOT / "checkpoints" / condition_dir / f"seed_{seed}" / f"skill_{skill}" / "checkpoint_2000.pt"


def load_final_adapter(condition_dir: str, seed: int, skill: str, workspace_digest: str) -> tuple[runner.SkillAdapter, dict[str, Any], Path]:
    path = adapter_path(condition_dir, seed, skill)
    if not path.is_file():
        raise RuntimeError(f"missing final adapter checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("workspace_digest") != workspace_digest:
        raise RuntimeError(f"workspace dependency mismatch in {path}")
    adapter = runner.SkillAdapter()
    adapter.load_state_dict(payload["adapter"], strict=True)
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    if not runner.finite_module(adapter):
        raise RuntimeError(f"non-finite adapter checkpoint: {path}")
    return adapter, payload, path


def apply_adapter(workspace: runner.Workspace, adapter: runner.SkillAdapter, state: torch.Tensor, skill: str) -> torch.Tensor:
    return adapter(state, runner.operand_bits(skill, len(state)))


def evaluate_condition(workspace: runner.Workspace, condition: str, condition_dir: str, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adapter_a, payload_a, path_a = load_final_adapter(condition_dir, seed, "A", EXPECTED_WORKSPACE_DIGEST)
    adapter_b, payload_b, path_b = load_final_adapter(condition_dir, seed, "B", EXPECTED_WORKSPACE_DIGEST)
    values = torch.arange(PROBE_STATES, dtype=torch.int64)
    bits = runner.bits_from_int(values)
    target_a_values = runner.skill_target("A", values)
    target_ab_values = runner.skill_target("B", target_a_values)
    target_a_bits = runner.bits_from_int(target_a_values)
    target_ab_bits = runner.bits_from_int(target_ab_values)

    with torch.no_grad():
        base_state = workspace.encode(bits)
        canonical_state = workspace.encode(target_a_bits)
        actual_state = apply_adapter(workspace, adapter_a, base_state, "A")
        error = actual_state - canonical_state
        actual_norm = torch.linalg.vector_norm(error, dim=1)
        actual_state_norm = torch.linalg.vector_norm(actual_state, dim=1)
        canonical_norm = torch.linalg.vector_norm(canonical_state, dim=1)
        cosine = (actual_state * canonical_state).sum(dim=1) / (actual_state_norm.clamp_min(1e-12) * canonical_norm.clamp_min(1e-12))

        a_actual_correct = exact_rows(workspace.decode_logits(actual_state), target_a_bits)
        b_canonical_state = apply_adapter(workspace, adapter_b, canonical_state, "B")
        b_actual_state = apply_adapter(workspace, adapter_b, actual_state, "B")
        b_canonical_correct = exact_rows(workspace.decode_logits(b_canonical_state), target_ab_bits)
        b_actual_correct = exact_rows(workspace.decode_logits(b_actual_state), target_ab_bits)

        interpolation = {}
        interpolation_rows: dict[int, dict[str, bool]] = {int(value): {} for value in values.tolist()}
        for alpha in ALPHAS:
            interpolated = canonical_state + alpha * error
            result = exact_rows(workspace.decode_logits(apply_adapter(workspace, adapter_b, interpolated, "B")), target_ab_bits)
            label = f"{alpha:.2f}"
            interpolation[label] = {
                "exact_accuracy": float(result.to(torch.float32).mean().item()),
                "correct": int(result.sum().item()),
                "count": int(result.numel()),
            }
            for index, value in enumerate(values.tolist()):
                interpolation_rows[value][label] = bool(result[index].item())

        generator = torch.Generator().manual_seed(20261101 + seed + (0 if condition == "lambda_0.1" else 100000))
        gaussian = torch.randn(error.shape, generator=generator, dtype=error.dtype)
        gaussian = gaussian * (actual_norm / torch.linalg.vector_norm(gaussian, dim=1).clamp_min(1e-12)).unsqueeze(1)
        gaussian_correct = exact_rows(workspace.decode_logits(apply_adapter(workspace, adapter_b, canonical_state + gaussian, "B")), target_ab_bits)

        centered_error = error - error.mean(dim=0, keepdim=True)
        _, singular_values, vh = torch.linalg.svd(centered_error, full_matrices=False)
        principal = []
        axis_count = min(PCA_AXES, int(vh.shape[0]))
        for axis_index in range(axis_count):
            axis = vh[axis_index]
            for sign in (1.0, -1.0):
                perturbation = sign * actual_norm.unsqueeze(1) * axis.unsqueeze(0)
                result = exact_rows(workspace.decode_logits(apply_adapter(workspace, adapter_b, canonical_state + perturbation, "B")), target_ab_bits)
                principal.append({
                    "axis": axis_index,
                    "sign": int(sign),
                    "exact_accuracy": float(result.to(torch.float32).mean().item()),
                    "correct": int(result.sum().item()),
                    "count": int(result.numel()),
                })

    rows = []
    for index, value in enumerate(values.tolist()):
        rows.append({
            "condition": condition,
            "seed": seed,
            "state": value,
            "target_a": int(target_a_values[index].item()),
            "target_ab": int(target_ab_values[index].item()),
            "actual_displacement_l2": float(actual_norm[index].item()),
            "actual_canonical_cosine": float(cosine[index].item()),
            "a_actual_correct": bool(a_actual_correct[index].item()),
            "b_actual_correct": bool(b_actual_correct[index].item()),
            "b_canonical_correct": bool(b_canonical_correct[index].item()),
            "canonical_rescue": bool((not b_actual_correct[index].item()) and b_canonical_correct[index].item()),
            "canonical_regression": bool(b_actual_correct[index].item() and not b_canonical_correct[index].item()),
            "interpolation": interpolation_rows[value],
        })

    summary = {
        "condition": condition,
        "seed": seed,
        "lambda": 0.1 if condition == "lambda_0.1" else 1.0,
        "checkpoint_step": 2000,
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "adapter_a_checkpoint": str(path_a.relative_to(ROOT)),
        "adapter_b_checkpoint": str(path_b.relative_to(ROOT)),
        "adapter_a_checkpoint_sha256": sha256_file(path_a),
        "adapter_b_checkpoint_sha256": sha256_file(path_b),
        "adapter_a_digest": runner.state_dict_digest(adapter_a),
        "adapter_b_digest": runner.state_dict_digest(adapter_b),
        "state_count": PROBE_STATES,
        "displacement_l2": stats(actual_norm),
        "canonical_cosine": stats(cosine),
        "a_actual_exact_accuracy": float(a_actual_correct.to(torch.float32).mean().item()),
        "b_actual_exact_accuracy": float(b_actual_correct.to(torch.float32).mean().item()),
        "b_canonical_exact_accuracy": float(b_canonical_correct.to(torch.float32).mean().item()),
        "canonical_rescue_count": int(((~b_actual_correct) & b_canonical_correct).sum().item()),
        "canonical_regression_count": int((b_actual_correct & ~b_canonical_correct).sum().item()),
        "actual_error_count": int((~b_actual_correct).sum().item()),
        "actual_error_injection_exact_accuracy": float(b_actual_correct.to(torch.float32).mean().item()),
        "gaussian_norm_matched_exact_accuracy": float(gaussian_correct.to(torch.float32).mean().item()),
        "interpolation": interpolation,
        "principal_axis_perturbations": principal,
        "principal_axis_singular_values_top": [float(value.item()) for value in singular_values[:PCA_AXES]],
    }
    return summary, rows


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state, architecture, hashes = load_bundle(ARTIFACT, EXPECTED_WORKSPACE_DIGEST)
    workspace = runner.Workspace()
    workspace.load_state_dict(state, strict=True)
    workspace.eval()
    for parameter in workspace.parameters():
        parameter.requires_grad_(False)
    if runner.state_dict_digest(workspace) != EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError("loaded workspace digest mismatch")

    summaries = []
    rows = []
    for condition, condition_dir in CONDITIONS:
        for seed in SEEDS:
            summary, condition_rows = evaluate_condition(workspace, condition, condition_dir, seed)
            summaries.append(summary)
            rows.extend(condition_rows)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        grouped.setdefault(summary["condition"], []).append(summary)
    cross_seed = {}
    for condition, condition_summaries in grouped.items():
        cross_seed[condition] = {
            "b_actual_accuracy_by_seed": {str(item["seed"]): item["b_actual_exact_accuracy"] for item in condition_summaries},
            "b_canonical_accuracy_by_seed": {str(item["seed"]): item["b_canonical_exact_accuracy"] for item in condition_summaries},
            "mean_displacement_l2": sum(item["displacement_l2"]["mean"] for item in condition_summaries) / len(condition_summaries),
            "mean_canonical_cosine": sum(item["canonical_cosine"]["mean"] for item in condition_summaries) / len(condition_summaries),
            "mean_gaussian_norm_matched_accuracy": sum(item["gaussian_norm_matched_exact_accuracy"] for item in condition_summaries) / len(condition_summaries),
            "mean_canonical_rescue_count": sum(item["canonical_rescue_count"] for item in condition_summaries) / len(condition_summaries),
        }

    report = {
        "schema_version": "crucible-cnss-latent-geometry-causal-audit-v1-20261101",
        "classification": "COMPLETED_NO_TRAINING_GEOMETRY_AUDIT",
        "source_scientific_classification_unchanged": "EXPLORATORY_CPU_SCIENTIFIC_RESULT",
        "workspace_artifact": str(ARTIFACT.relative_to(ROOT)),
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "workspace_artifact_file_sha256": hashes["state_file_sha256"],
        "workspace_architecture_schema": architecture["schema_version"],
        "conditions": [condition for condition, _ in CONDITIONS],
        "seeds": list(SEEDS),
        "state_count_per_condition": PROBE_STATES,
        "alpha_values": list(ALPHAS),
        "training": False,
        "parameter_updates": False,
        "workspace_reconstruction": False,
        "gpu_used": False,
        "oracle_repair": False,
        "summaries": summaries,
        "cross_seed": cross_seed,
        "interpretation_boundary": "These are no-training causal diagnostics of existing checkpoints. They do not establish new learning, forward transfer or broad compositional generalization.",
    }
    (OUTPUT / "LATENT_GEOMETRY_AUDIT_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (OUTPUT / "LATENT_GEOMETRY_ROWS.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    lines = [
        "# CNSS Latent Geometry Causal Audit v1 — no-training execution",
        "",
        f"Classification: **{report['classification']}**",
        "",
        f"Workspace digest: `{EXPECTED_WORKSPACE_DIGEST}`",
        f"Conditions: `{', '.join(report['conditions'])}`",
        f"Seeds: `{', '.join(str(seed) for seed in SEEDS)}`",
        f"States evaluated per condition/seed: `{PROBE_STATES}`",
        "",
        "No training, parameter updates, workspace regeneration or GPU work was performed.",
        "",
        "The original CNSS scientific classification remains unchanged. This report is a mechanistic diagnostic of existing checkpoints.",
        "",
        "Machine-readable report: `LATENT_GEOMETRY_AUDIT_REPORT.json`.",
    ]
    (OUTPUT / "LATENT_GEOMETRY_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"classification": report["classification"], "conditions": report["conditions"], "seeds": list(SEEDS), "rows": len(rows), "workspace_digest": EXPECTED_WORKSPACE_DIGEST}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
