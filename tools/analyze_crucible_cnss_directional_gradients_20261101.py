#!/usr/bin/env python3
"""Read-only CNSS directional-gradient diagnostic.

The diagnostic uses the existing A->B checkpoints and measures the local
receiver sensitivity of B at the canonical intermediate state.  It compares
the directional derivative along A's actual representation error with matched
random directions.  No training or parameter updates occur.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import audit_crucible_cnss_latent_geometry_20261101 as audit
import analyze_crucible_cnss_latent_geometry_margins_20261101 as margin
import run_crucible_track_a_cnss_semantic_interface_compatibility_v1_20261101 as runner
from verify_crucible_cnss_workspace_artifact_20261101 import load_bundle


OUTPUT = audit.OUTPUT
ARTIFACT = audit.ARTIFACT
EXPECTED_WORKSPACE_DIGEST = audit.EXPECTED_WORKSPACE_DIGEST
SEEDS = audit.SEEDS
CONDITIONS = audit.CONDITIONS
PROBE_STATES = audit.PROBE_STATES
STATE_WIDTH = runner.STATE_WIDTH
STATE_BITS = runner.STATE_BITS
RANDOM_DIRECTIONS = 32


def finite_tensor(value: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"non-finite diagnostic tensor: {label}")


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


def summarize(
    condition: str,
    seed: int | None,
    displacement: torch.Tensor,
    actual_min_margin: torch.Tensor,
    actual_correct: torch.Tensor,
    harmful_projection: torch.Tensor,
    harmful_cosine: torch.Tensor,
    actual_margin_change: torch.Tensor,
    first_order_residual: torch.Tensor,
    random_harmful_mean: torch.Tensor,
    random_harmful_p95: torch.Tensor,
    actual_harmful_percentile: torch.Tensor,
    gradient_norm: torch.Tensor,
    worst_harmful_projection: torch.Tensor,
    worst_harmful_cosine: torch.Tensor,
    linearized_min_margin: torch.Tensor,
    nonlinear_min_margin_residual: torch.Tensor,
    random_worst_harmful_mean: torch.Tensor,
    random_worst_harmful_p95: torch.Tensor,
    actual_worst_harmful_percentile: torch.Tensor,
    worst_gradient_norm: torch.Tensor,
    checkpoint_a: Path | None = None,
    checkpoint_b: Path | None = None,
) -> dict[str, Any]:
    tensors = {
        "displacement": displacement,
        "actual_min_margin": actual_min_margin,
        "harmful_projection": harmful_projection,
        "harmful_cosine": harmful_cosine,
        "actual_margin_change": actual_margin_change,
        "first_order_residual": first_order_residual,
        "random_harmful_mean": random_harmful_mean,
        "random_harmful_p95": random_harmful_p95,
        "actual_harmful_percentile": actual_harmful_percentile,
        "gradient_norm": gradient_norm,
        "worst_harmful_projection": worst_harmful_projection,
        "worst_harmful_cosine": worst_harmful_cosine,
        "linearized_min_margin": linearized_min_margin,
        "nonlinear_min_margin_residual": nonlinear_min_margin_residual,
        "random_worst_harmful_mean": random_worst_harmful_mean,
        "random_worst_harmful_p95": random_worst_harmful_p95,
        "actual_worst_harmful_percentile": actual_worst_harmful_percentile,
        "worst_gradient_norm": worst_gradient_norm,
    }
    for label, value in tensors.items():
        finite_tensor(value, label)
    actual_correct = actual_correct.detach().cpu().bool()
    linearized_correct = linearized_min_margin.detach().cpu() > 0.0
    result: dict[str, Any] = {
        "condition": condition,
        "seed": seed,
        "state_count": int(displacement.numel()),
        "actual_exact_accuracy": float(actual_correct.to(torch.float64).mean().item()),
        "actual_correct_count": int(actual_correct.sum().item()),
        "actual_incorrect_count": int((~actual_correct).sum().item()),
        "linearized_exact_accuracy": float(linearized_correct.to(torch.float64).mean().item()),
        "linearized_actual_agreement": float((linearized_correct == actual_correct).to(torch.float64).mean().item()),
        "linearized_confusion_vs_actual": {
            "linearized_correct_actual_correct": int((linearized_correct & actual_correct).sum().item()),
            "linearized_correct_actual_incorrect": int((linearized_correct & ~actual_correct).sum().item()),
            "linearized_incorrect_actual_correct": int((~linearized_correct & actual_correct).sum().item()),
            "linearized_incorrect_actual_incorrect": int((~linearized_correct & ~actual_correct).sum().item()),
        },
        "displacement_l2": margin.stats(displacement),
        "actual_min_signed_logit_margin": margin.stats(actual_min_margin),
        "harmful_projection_per_error_norm": margin.stats(harmful_projection),
        "cosine_to_harmful_gradient": margin.stats(harmful_cosine),
        "actual_bottleneck_margin_change": margin.stats(actual_margin_change),
        "first_order_residual": margin.stats(first_order_residual),
        "random_harmful_projection_mean": margin.stats(random_harmful_mean),
        "random_harmful_projection_p95": margin.stats(random_harmful_p95),
        "actual_harmful_percentile_vs_random": margin.stats(actual_harmful_percentile),
        "receiver_gradient_norm": margin.stats(gradient_norm),
        "worst_bit_harmful_projection_per_error_norm": margin.stats(worst_harmful_projection),
        "worst_bit_cosine_to_harmful_gradient": margin.stats(worst_harmful_cosine),
        "linearized_min_signed_logit_margin": margin.stats(linearized_min_margin),
        "nonlinear_min_margin_residual": margin.stats(nonlinear_min_margin_residual),
        "random_worst_harmful_projection_mean": margin.stats(random_worst_harmful_mean),
        "random_worst_harmful_projection_p95": margin.stats(random_worst_harmful_p95),
        "actual_worst_harmful_percentile_vs_random": margin.stats(actual_worst_harmful_percentile),
        "worst_receiver_gradient_norm": margin.stats(worst_gradient_norm),
        "by_actual_result": {
            "harmful_projection": margin.group_stats(harmful_projection, actual_correct),
            "harmful_cosine": margin.group_stats(harmful_cosine, actual_correct),
            "actual_harmful_percentile_vs_random": margin.group_stats(actual_harmful_percentile, actual_correct),
            "actual_bottleneck_margin_change": margin.group_stats(actual_margin_change, actual_correct),
            "first_order_residual": margin.group_stats(first_order_residual, actual_correct),
            "worst_harmful_projection": margin.group_stats(worst_harmful_projection, actual_correct),
            "worst_harmful_cosine": margin.group_stats(worst_harmful_cosine, actual_correct),
            "actual_worst_harmful_percentile_vs_random": margin.group_stats(actual_worst_harmful_percentile, actual_correct),
            "nonlinear_min_margin_residual": margin.group_stats(nonlinear_min_margin_residual, actual_correct),
        },
        "correlation": {
            "pearson_displacement_vs_harmful_projection": margin.pearson(displacement, harmful_projection),
            "spearman_displacement_vs_harmful_projection": margin.spearman(displacement, harmful_projection),
            "pearson_harmful_projection_vs_actual_min_margin": margin.pearson(harmful_projection, actual_min_margin),
            "spearman_harmful_projection_vs_actual_min_margin": margin.spearman(harmful_projection, actual_min_margin),
            "pearson_harmful_percentile_vs_actual_min_margin": margin.pearson(actual_harmful_percentile, actual_min_margin),
            "spearman_harmful_percentile_vs_actual_min_margin": margin.spearman(actual_harmful_percentile, actual_min_margin),
            "pearson_worst_harmful_projection_vs_actual_min_margin": margin.pearson(worst_harmful_projection, actual_min_margin),
            "spearman_worst_harmful_projection_vs_actual_min_margin": margin.spearman(worst_harmful_projection, actual_min_margin),
            "pearson_actual_worst_harmful_percentile_vs_actual_min_margin": margin.pearson(actual_worst_harmful_percentile, actual_min_margin),
            "spearman_actual_worst_harmful_percentile_vs_actual_min_margin": margin.spearman(actual_worst_harmful_percentile, actual_min_margin),
        },
        "fraction_actual_projection_harmful": float((harmful_projection > 0.0).to(torch.float64).mean().item()),
        "fraction_actual_above_random_p95": float((actual_harmful_percentile >= 0.95).to(torch.float64).mean().item()),
        "fraction_worst_projection_harmful": float((worst_harmful_projection > 0.0).to(torch.float64).mean().item()),
        "fraction_worst_above_random_p95": float((actual_worst_harmful_percentile >= 0.95).to(torch.float64).mean().item()),
    }
    if checkpoint_a is not None and checkpoint_b is not None:
        result["adapter_a_checkpoint"] = str(checkpoint_a.relative_to(ROOT))
        result["adapter_b_checkpoint"] = str(checkpoint_b.relative_to(ROOT))
        result["adapter_a_checkpoint_sha256"] = audit.sha256_file(checkpoint_a)
        result["adapter_b_checkpoint_sha256"] = audit.sha256_file(checkpoint_b)
    return result


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
        error_norm = torch.linalg.vector_norm(error, dim=1).clamp_min(1e-12)
        actual_b_state = audit.apply_adapter(workspace, adapter_b, actual_state, "B")
        actual_logits = workspace.decode_logits(actual_b_state)
        target_sign = target_ab_bits * 2.0 - 1.0
        actual_signed = actual_logits * target_sign
        actual_min_margin = actual_signed.min(dim=1).values
        actual_correct = actual_min_margin > 0.0

    # Compute B's local Jacobian at canonical intermediate states.  The
    # selected bit is the canonical bottleneck bit for each row.
    canonical_for_grad = canonical_state.detach().requires_grad_(True)
    canonical_b_state = audit.apply_adapter(workspace, adapter_b, canonical_for_grad, "B")
    canonical_logits = workspace.decode_logits(canonical_b_state)
    canonical_signed = canonical_logits * target_sign
    canonical_min_margin, bottleneck = canonical_signed.min(dim=1)
    gradients = []
    for bit_index in range(STATE_BITS):
        gradient = torch.autograd.grad(
            canonical_signed[:, bit_index].sum(),
            canonical_for_grad,
            retain_graph=bit_index < STATE_BITS - 1,
            create_graph=False,
        )[0]
        gradients.append(gradient.detach())
    jacobian = torch.stack(gradients, dim=1)
    finite_tensor(jacobian, "receiver Jacobian")
    row_index = torch.arange(PROBE_STATES, dtype=torch.long)
    selected_gradient = jacobian[row_index, bottleneck]
    gradient_norm = torch.linalg.vector_norm(selected_gradient, dim=1).clamp_min(1e-12)
    first_order_change = (selected_gradient * error).sum(dim=1)
    harmful_projection = -first_order_change / error_norm
    harmful_cosine = (-selected_gradient * error).sum(dim=1) / (gradient_norm * error_norm)

    actual_margin_change = actual_signed[row_index, bottleneck] - canonical_min_margin
    first_order_residual = actual_margin_change - first_order_change

    # A different output bit can become the actual failure after the error is
    # applied.  Include the worst first-order change across all bits so that
    # the diagnostic does not assume the canonical bottleneck remains the
    # relevant decision boundary.
    directional_changes = torch.einsum("nbd,nd->nb", jacobian, error)
    linearized_min_margin = (canonical_signed + directional_changes).min(dim=1).values
    nonlinear_min_margin_residual = actual_min_margin - linearized_min_margin
    worst_first_order_change, worst_bit = directional_changes.min(dim=1)
    worst_gradient = jacobian[row_index, worst_bit]
    worst_gradient_norm = torch.linalg.vector_norm(worst_gradient, dim=1).clamp_min(1e-12)
    worst_harmful_projection = -worst_first_order_change / error_norm
    worst_harmful_cosine = (-worst_gradient * error).sum(dim=1) / (worst_gradient_norm * error_norm)

    generator = torch.Generator().manual_seed(20261101 + seed + (0 if condition == "lambda_0.1" else 100000))
    random = torch.randn(
        (RANDOM_DIRECTIONS, PROBE_STATES, STATE_WIDTH),
        generator=generator,
        dtype=error.dtype,
    )
    random = random * (
        error_norm.unsqueeze(0).unsqueeze(2)
        / torch.linalg.vector_norm(random, dim=2, keepdim=True).clamp_min(1e-12)
    )
    random_derivative = torch.einsum("nd,rnd->rn", selected_gradient, random) / error_norm.unsqueeze(0)
    random_harmful = -random_derivative
    random_harmful_mean = random_harmful.mean(dim=0)
    random_harmful_p95 = torch.quantile(random_harmful, 0.95, dim=0)
    actual_harmful_percentile = (
        (random_harmful <= harmful_projection.unsqueeze(0)).to(torch.float32).mean(dim=0)
    )
    random_derivative_all = torch.einsum("nbd,rnd->rnb", jacobian, random) / error_norm.unsqueeze(0).unsqueeze(2)
    random_worst_harmful = -random_derivative_all.min(dim=2).values
    random_worst_harmful_mean = random_worst_harmful.mean(dim=0)
    random_worst_harmful_p95 = torch.quantile(random_worst_harmful, 0.95, dim=0)
    actual_worst_harmful_percentile = (
        (random_worst_harmful <= worst_harmful_projection.unsqueeze(0)).to(torch.float32).mean(dim=0)
    )

    summary = summarize(
        condition,
        seed,
        error_norm,
        actual_min_margin,
        actual_correct,
        harmful_projection,
        harmful_cosine,
        actual_margin_change,
        first_order_residual,
        random_harmful_mean,
        random_harmful_p95,
        actual_harmful_percentile,
        gradient_norm,
        worst_harmful_projection,
        worst_harmful_cosine,
        linearized_min_margin,
        nonlinear_min_margin_residual,
        random_worst_harmful_mean,
        random_worst_harmful_p95,
        actual_worst_harmful_percentile,
        worst_gradient_norm,
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
                "canonical_bottleneck_bit": int(bottleneck[index].item()),
                "worst_first_order_bit": int(worst_bit[index].item()),
                "actual_displacement_l2": float(error_norm[index].item()),
                "actual_min_signed_logit_margin": float(actual_min_margin[index].item()),
                "canonical_min_signed_logit_margin": float(canonical_min_margin[index].item()),
                "linearized_min_signed_logit_margin": float(linearized_min_margin[index].item()),
                "receiver_gradient_norm": float(gradient_norm[index].item()),
                "actual_harmful_projection_per_error_norm": float(harmful_projection[index].item()),
                "actual_cosine_to_harmful_gradient": float(harmful_cosine[index].item()),
                "actual_bottleneck_margin_change": float(actual_margin_change[index].item()),
                "first_order_residual": float(first_order_residual[index].item()),
                "worst_bit_harmful_projection_per_error_norm": float(worst_harmful_projection[index].item()),
                "worst_bit_cosine_to_harmful_gradient": float(worst_harmful_cosine[index].item()),
                "nonlinear_min_margin_residual": float(nonlinear_min_margin_residual[index].item()),
                "random_harmful_projection_mean": float(random_harmful_mean[index].item()),
                "random_harmful_projection_p95": float(random_harmful_p95[index].item()),
                "actual_harmful_percentile_vs_random": float(actual_harmful_percentile[index].item()),
                "random_worst_harmful_projection_mean": float(random_worst_harmful_mean[index].item()),
                "random_worst_harmful_projection_p95": float(random_worst_harmful_p95[index].item()),
                "actual_worst_harmful_percentile_vs_random": float(actual_worst_harmful_percentile[index].item()),
                "worst_receiver_gradient_norm": float(worst_gradient_norm[index].item()),
                "b_actual_correct": bool(actual_correct[index].item()),
            }
        )
    return summary, rows


def pooled_summary(condition: str, summaries: Iterable[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    condition_summaries = [item for item in summaries if item["condition"] == condition]
    condition_rows = [row for row in rows if row["condition"] == condition]

    def tensor(key: str) -> torch.Tensor:
        return torch.tensor([row[key] for row in condition_rows], dtype=torch.float64)

    displacement = tensor("actual_displacement_l2")
    actual_min_margin = tensor("actual_min_signed_logit_margin")
    harmful_projection = tensor("actual_harmful_projection_per_error_norm")
    harmful_cosine = tensor("actual_cosine_to_harmful_gradient")
    actual_margin_change = tensor("actual_bottleneck_margin_change")
    first_order_residual = tensor("first_order_residual")
    random_harmful_mean = tensor("random_harmful_projection_mean")
    random_harmful_p95 = tensor("random_harmful_projection_p95")
    actual_harmful_percentile = tensor("actual_harmful_percentile_vs_random")
    gradient_norm = tensor("receiver_gradient_norm")
    worst_harmful_projection = tensor("worst_bit_harmful_projection_per_error_norm")
    worst_harmful_cosine = tensor("worst_bit_cosine_to_harmful_gradient")
    linearized_min_margin = tensor("linearized_min_signed_logit_margin")
    nonlinear_min_margin_residual = tensor("nonlinear_min_margin_residual")
    random_worst_harmful_mean = tensor("random_worst_harmful_projection_mean")
    random_worst_harmful_p95 = tensor("random_worst_harmful_projection_p95")
    actual_worst_harmful_percentile = tensor("actual_worst_harmful_percentile_vs_random")
    worst_gradient_norm = tensor("worst_receiver_gradient_norm")
    actual_correct = torch.tensor([row["b_actual_correct"] for row in condition_rows], dtype=torch.bool)
    result = summarize(
        condition,
        None,
        displacement,
        actual_min_margin,
        actual_correct,
        harmful_projection,
        harmful_cosine,
        actual_margin_change,
        first_order_residual,
        random_harmful_mean,
        random_harmful_p95,
        actual_harmful_percentile,
        gradient_norm,
        worst_harmful_projection,
        worst_harmful_cosine,
        linearized_min_margin,
        nonlinear_min_margin_residual,
        random_worst_harmful_mean,
        random_worst_harmful_p95,
        actual_worst_harmful_percentile,
        worst_gradient_norm,
    )
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
        "schema_version": "crucible-cnss-directional-gradient-diagnostic-v1-20261101",
        "classification": "COMPLETED_NO_TRAINING_DIRECTIONAL_GRADIENT_DIAGNOSTIC",
        "source_scientific_classification_unchanged": "EXPLORATORY_CPU_SCIENTIFIC_RESULT",
        "direction": "A_then_B_only",
        "state_count_per_seed_condition": PROBE_STATES,
        "seeds": list(SEEDS),
        "conditions": [condition for condition, _condition_dir in CONDITIONS],
        "random_directions_per_state": RANDOM_DIRECTIONS,
        "workspace_artifact": str(ARTIFACT.relative_to(ROOT)),
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "training": False,
        "parameter_updates": False,
        "workspace_reconstruction": False,
        "gpu_used": False,
        "gradient_definition": "B output-logit gradients at each canonical intermediate state, selecting the canonical bottleneck output bit",
        "harmful_projection_definition": "negative directional derivative of the selected target-signed logit along A's actual error, normalized by error norm",
        "interpretation_boundary": "Local linear sensitivity analysis of existing checkpoints; it does not establish causation, new learning, forward transfer, reverse composition, longer-chain behavior, or broad generalization. The worst-bit random percentile is an envelope statistic across eight output bits and is not standalone causal evidence.",
        "summaries": summaries,
        "pooled_by_condition": pooled,
    }
    (OUTPUT / "LATENT_GEOMETRY_DIRECTIONAL_GRADIENT_DIAGNOSTIC.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (OUTPUT / "LATENT_GEOMETRY_DIRECTIONAL_GRADIENT_ROWS.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    lines = [
        "# CNSS Directional-Gradient Diagnostic v1",
        "",
        "This is a read-only analysis of the existing CNSS checkpoints.",
        "No training, parameter updates, workspace reconstruction, or GPU execution occurred.",
        "The registered direction is A then B only; reverse B then A was not evaluated.",
        "",
        "The receiver gradient is evaluated at the canonical intermediate state.",
        "The canonical-bottleneck projection uses the canonical minimum-margin bit; the worst-bit projection takes the most margin-reducing first-order direction across all eight output bits.",
        "Matched random directions use the same per-state error norm and 32 independent directions.",
        "",
        "## Pooled results",
        "",
        "| Condition | Exact accuracy | Mean worst-bit harmful projection | Mean worst-bit random percentile | Pearson worst projection/margin | Spearman worst projection/margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, values in pooled.items():
        correlation = values["correlation"]
        lines.append(
            f"| {condition} | {values['actual_exact_accuracy']:.4f} | "
            f"{values['worst_bit_harmful_projection_per_error_norm']['mean']:.4f} | "
            f"{values['actual_worst_harmful_percentile_vs_random']['mean']:.4f} | "
            f"{correlation['pearson_worst_harmful_projection_vs_actual_min_margin']:.4f} | "
            f"{correlation['spearman_worst_harmful_projection_vs_actual_min_margin']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The all-bit local linearization produced these exact-match diagnostics:",
            "",
        ]
    )
    for condition, values in pooled.items():
        lines.append(
            f"- `{condition}`: linearized exact accuracy `{values['linearized_exact_accuracy']:.4f}`, "
            f"agreement with actual outcomes `{values['linearized_actual_agreement']:.4f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A high harmful projection or random-direction percentile indicates local alignment with a margin-reducing direction, not proof that the full nonlinear path is caused by that local derivative.",
            "The worst-bit random percentile is an envelope statistic across eight output bits and is not used as standalone causal evidence.",
            "This diagnostic does not evaluate B->A, longer chains, or new training.",
            "",
        ]
    )
    (OUTPUT / "LATENT_GEOMETRY_DIRECTIONAL_GRADIENT_DIAGNOSTIC.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "direction": report["direction"],
                "rows": len(rows),
                "output": str((OUTPUT / "LATENT_GEOMETRY_DIRECTIONAL_GRADIENT_DIAGNOSTIC.json").relative_to(ROOT)),
                "pooled": pooled,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
