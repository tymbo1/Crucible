#!/usr/bin/env python3
"""Read-only CNSS receiver-margin diagnostic.

This diagnostic reuses the verified CNSS workspace artifact and the existing
final A/B adapter checkpoints.  It does not train, reconstruct, or modify any
parameters.  For the registered A->B direction it measures the relationship
between A's latent displacement and B's signed output decision margin.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import audit_crucible_cnss_latent_geometry_20261101 as audit
import run_crucible_track_a_cnss_semantic_interface_compatibility_v1_20261101 as runner
from verify_crucible_cnss_workspace_artifact_20261101 import load_bundle


OUTPUT = audit.OUTPUT
ARTIFACT = audit.ARTIFACT
EXPECTED_WORKSPACE_DIGEST = audit.EXPECTED_WORKSPACE_DIGEST
SEEDS = audit.SEEDS
CONDITIONS = audit.CONDITIONS
PROBE_STATES = audit.PROBE_STATES


def finite_tensor(value: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"non-finite diagnostic tensor: {label}")


def stats(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.detach().cpu().to(torch.float64)
    finite_tensor(values, "stats input")
    if values.numel() == 0:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None, "std": None}
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "max": float(values.max().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.detach().cpu().to(torch.float64)
    y = y.detach().cpu().to(torch.float64)
    if x.numel() < 2 or y.numel() != x.numel():
        return None
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if float(denominator.item()) <= 1e-15:
        return None
    result = (x_centered * y_centered).sum() / denominator
    return float(result.item())


def rank_values(values: torch.Tensor) -> torch.Tensor:
    """Average-rank values without requiring scipy."""
    values = values.detach().cpu().to(torch.float64)
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    index = 0
    while index < values.numel():
        value = values[order[index]]
        end = index + 1
        while end < values.numel() and values[order[end]] == value:
            end += 1
        average_rank = (index + end - 1) / 2.0
        ranks[order[index:end]] = average_rank
        index = end
    return ranks


def spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    return pearson(rank_values(x), rank_values(y))


def group_stats(values: torch.Tensor, correct: torch.Tensor) -> dict[str, Any]:
    correct = correct.detach().cpu().bool()
    values = values.detach().cpu().to(torch.float64)
    return {
        "correct": stats(values[correct]),
        "incorrect": stats(values[~correct]),
    }


def summarize_rows(
    condition: str,
    seed: int | None,
    displacement: torch.Tensor,
    actual_correct: torch.Tensor,
    actual_min_margin: torch.Tensor,
    actual_mean_margin: torch.Tensor,
    canonical_min_margin: torch.Tensor,
    canonical_correct: torch.Tensor,
    checkpoint_a: Path | None = None,
    checkpoint_b: Path | None = None,
) -> dict[str, Any]:
    displacement = displacement.detach().cpu().to(torch.float64)
    actual_correct = actual_correct.detach().cpu().bool()
    actual_min_margin = actual_min_margin.detach().cpu().to(torch.float64)
    actual_mean_margin = actual_mean_margin.detach().cpu().to(torch.float64)
    canonical_min_margin = canonical_min_margin.detach().cpu().to(torch.float64)
    canonical_correct = canonical_correct.detach().cpu().bool()
    finite_tensor(displacement, "displacement")
    finite_tensor(actual_min_margin, "actual min margin")
    finite_tensor(actual_mean_margin, "actual mean margin")
    finite_tensor(canonical_min_margin, "canonical min margin")
    margin_drop = canonical_min_margin - actual_min_margin
    finite_tensor(margin_drop, "margin drop")

    result: dict[str, Any] = {
        "condition": condition,
        "seed": seed,
        "state_count": int(displacement.numel()),
        "actual_exact_accuracy": float(actual_correct.to(torch.float64).mean().item()),
        "canonical_exact_accuracy": float(canonical_correct.to(torch.float64).mean().item()),
        "actual_correct_count": int(actual_correct.sum().item()),
        "actual_incorrect_count": int((~actual_correct).sum().item()),
        "displacement_l2": stats(displacement),
        "actual_min_signed_logit_margin": stats(actual_min_margin),
        "actual_mean_signed_logit_margin": stats(actual_mean_margin),
        "canonical_min_signed_logit_margin": stats(canonical_min_margin),
        "canonical_to_actual_min_margin_drop": stats(margin_drop),
        "displacement_by_actual_result": group_stats(displacement, actual_correct),
        "min_margin_by_actual_result": group_stats(actual_min_margin, actual_correct),
        "mean_margin_by_actual_result": group_stats(actual_mean_margin, actual_correct),
        "margin_drop_by_actual_result": group_stats(margin_drop, actual_correct),
        "correlation": {
            "pearson_displacement_vs_actual_min_margin": pearson(displacement, actual_min_margin),
            "spearman_displacement_vs_actual_min_margin": spearman(displacement, actual_min_margin),
            "pearson_displacement_vs_actual_mean_margin": pearson(displacement, actual_mean_margin),
            "spearman_displacement_vs_actual_mean_margin": spearman(displacement, actual_mean_margin),
        },
        "all_actual_margins_positive_match_exact_accuracy": bool(
            torch.equal(actual_min_margin > 0.0, actual_correct)
        ),
    }
    if checkpoint_a is not None and checkpoint_b is not None:
        result["adapter_a_checkpoint"] = str(checkpoint_a.relative_to(ROOT))
        result["adapter_b_checkpoint"] = str(checkpoint_b.relative_to(ROOT))
        result["adapter_a_checkpoint_sha256"] = audit.sha256_file(checkpoint_a)
        result["adapter_b_checkpoint_sha256"] = audit.sha256_file(checkpoint_b)
    return result


def load_workspace() -> runner.Workspace:
    state, _architecture, _hashes = load_bundle(ARTIFACT, EXPECTED_WORKSPACE_DIGEST)
    workspace = runner.Workspace()
    workspace.load_state_dict(state, strict=True)
    workspace.eval()
    for parameter in workspace.parameters():
        parameter.requires_grad_(False)
    if runner.state_dict_digest(workspace) != EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError("workspace digest mismatch")
    if not runner.finite_module(workspace):
        raise RuntimeError("workspace contains non-finite values")
    return workspace


def evaluate_condition(
    workspace: runner.Workspace,
    condition: str,
    condition_dir: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adapter_a, _payload_a, path_a = audit.load_final_adapter(condition_dir, seed, "A", EXPECTED_WORKSPACE_DIGEST)
    adapter_b, _payload_b, path_b = audit.load_final_adapter(condition_dir, seed, "B", EXPECTED_WORKSPACE_DIGEST)
    values = torch.arange(PROBE_STATES, dtype=torch.int64)
    bits = runner.bits_from_int(values)
    target_a_values = runner.skill_target("A", values)
    target_ab_values = runner.skill_target("B", target_a_values)
    target_a_bits = runner.bits_from_int(target_a_values)
    target_ab_bits = runner.bits_from_int(target_ab_values)

    with torch.no_grad():
        base_state = workspace.encode(bits)
        canonical_state = workspace.encode(target_a_bits)
        actual_state = audit.apply_adapter(workspace, adapter_a, base_state, "A")
        error = actual_state - canonical_state
        displacement = torch.linalg.vector_norm(error, dim=1)
        actual_b_state = audit.apply_adapter(workspace, adapter_b, actual_state, "B")
        canonical_b_state = audit.apply_adapter(workspace, adapter_b, canonical_state, "B")
        actual_logits = workspace.decode_logits(actual_b_state)
        canonical_logits = workspace.decode_logits(canonical_b_state)
        target_sign = target_ab_bits * 2.0 - 1.0
        actual_signed = actual_logits * target_sign
        canonical_signed = canonical_logits * target_sign
        actual_min_margin = actual_signed.min(dim=1).values
        actual_mean_margin = actual_signed.mean(dim=1)
        canonical_min_margin = canonical_signed.min(dim=1).values
        actual_correct = actual_min_margin > 0.0
        canonical_correct = canonical_min_margin > 0.0

    summary = summarize_rows(
        condition,
        seed,
        displacement,
        actual_correct,
        actual_min_margin,
        actual_mean_margin,
        canonical_min_margin,
        canonical_correct,
        path_a,
        path_b,
    )
    rows = []
    for index, value in enumerate(values.tolist()):
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "state": value,
                "actual_displacement_l2": float(displacement[index].item()),
                "actual_min_signed_logit_margin": float(actual_min_margin[index].item()),
                "actual_mean_signed_logit_margin": float(actual_mean_margin[index].item()),
                "canonical_min_signed_logit_margin": float(canonical_min_margin[index].item()),
                "canonical_to_actual_min_margin_drop": float(
                    (canonical_min_margin[index] - actual_min_margin[index]).item()
                ),
                "b_actual_correct": bool(actual_correct[index].item()),
                "b_canonical_correct": bool(canonical_correct[index].item()),
            }
        )
    return summary, rows


def pooled_summary(condition: str, summaries: Iterable[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    condition_summaries = [item for item in summaries if item["condition"] == condition]
    condition_rows = [row for row in rows if row["condition"] == condition]
    displacement = torch.tensor([row["actual_displacement_l2"] for row in condition_rows], dtype=torch.float64)
    actual_min = torch.tensor([row["actual_min_signed_logit_margin"] for row in condition_rows], dtype=torch.float64)
    actual_mean = torch.tensor([row["actual_mean_signed_logit_margin"] for row in condition_rows], dtype=torch.float64)
    canonical_min = torch.tensor([row["canonical_min_signed_logit_margin"] for row in condition_rows], dtype=torch.float64)
    actual_correct = torch.tensor([row["b_actual_correct"] for row in condition_rows], dtype=torch.bool)
    canonical_correct = torch.tensor([row["b_canonical_correct"] for row in condition_rows], dtype=torch.bool)
    result = summarize_rows(condition, None, displacement, actual_correct, actual_min, actual_mean, canonical_min, canonical_correct)
    result["seed_count"] = len(condition_summaries)
    result["source_seeds"] = [item["seed"] for item in condition_summaries]
    return result


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    workspace = load_workspace()
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for condition, condition_dir in CONDITIONS:
        for seed in SEEDS:
            summary, condition_rows = evaluate_condition(workspace, condition, condition_dir, seed)
            summaries.append(summary)
            rows.extend(condition_rows)

    pooled = {
        condition: pooled_summary(condition, summaries, rows)
        for condition, _condition_dir in CONDITIONS
    }
    report = {
        "schema_version": "crucible-cnss-latent-geometry-margin-diagnostic-v1-20261101",
        "classification": "COMPLETED_NO_TRAINING_MARGIN_DIAGNOSTIC",
        "source_scientific_classification_unchanged": "EXPLORATORY_CPU_SCIENTIFIC_RESULT",
        "direction": "A_then_B_only",
        "state_count_per_seed_condition": PROBE_STATES,
        "seeds": list(SEEDS),
        "conditions": [condition for condition, _condition_dir in CONDITIONS],
        "workspace_artifact": str(ARTIFACT.relative_to(ROOT)),
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "training": False,
        "parameter_updates": False,
        "workspace_reconstruction": False,
        "gpu_used": False,
        "margin_definition": "minimum target-signed output logit across the eight B output bits; positive iff every bit is correct",
        "interpretation_boundary": "Descriptive margin/displacement analysis of existing checkpoints; it does not establish causation, new learning, forward transfer, reverse composition, or broad generalization.",
        "summaries": summaries,
        "pooled_by_condition": pooled,
    }
    (OUTPUT / "LATENT_GEOMETRY_MARGIN_DIAGNOSTIC.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (OUTPUT / "LATENT_GEOMETRY_MARGIN_ROWS.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    lines = [
        "# CNSS Latent Geometry Margin Diagnostic v1",
        "",
        "This is a read-only analysis of the existing CNSS checkpoints.",
        "No training, parameter updates, workspace reconstruction, or GPU execution occurred.",
        "The registered direction is A then B only; reverse B then A was not evaluated.",
        "",
        "## Margin definition",
        "",
        "For each B output bit, the target-signed logit is positive when that bit is correct.",
        "The row margin is the minimum target-signed logit across all eight bits.",
        "Therefore, positive row margin and exact-match accuracy are equivalent under this decoder.",
        "",
        "## Pooled results",
        "",
        "| Condition | Exact accuracy | Mean displacement | Mean min margin | Pearson displacement/min margin | Spearman displacement/min margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, values in pooled.items():
        correlation = values["correlation"]
        lines.append(
            f"| {condition} | {values['actual_exact_accuracy']:.4f} | "
            f"{values['displacement_l2']['mean']:.4f} | "
            f"{values['actual_min_signed_logit_margin']['mean']:.4f} | "
            f"{correlation['pearson_displacement_vs_actual_min_margin']:.4f} | "
            f"{correlation['spearman_displacement_vs_actual_min_margin']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This diagnostic tests whether displacement magnitude tracks B's output margin in the saved A->B path.",
            "It does not establish that displacement causes the decision change, and it does not evaluate B->A, longer chains, or new training.",
            "",
        ]
    )
    (OUTPUT / "LATENT_GEOMETRY_MARGIN_DIAGNOSTIC.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "direction": report["direction"],
                "rows": len(rows),
                "output": str((OUTPUT / "LATENT_GEOMETRY_MARGIN_DIAGNOSTIC.json").relative_to(ROOT)),
                "pooled": pooled,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
