#!/usr/bin/env python3
"""Read-only canonical-state, adapter-geometry and decoder-sensitivity audit."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101 as control
import run_crucible_track_a_cnss_partner_generalization_v1_20261101 as partner
import audit_crucible_track_a_cnss_d_generalization_forensics_v1_20261101 as prior

PROTOCOL = ROOT / "docs" / "CRUCIBLE_TRACK_A_CNSS_D_FAULT_ISOLATION_V1_20261101.json"
SOURCE_RESULT = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101" / "D_SKILL_CORRECTIVE_CONTROLS_RESULT.json"
MANIFEST_PATH = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101" / "MANIFEST_PUBLIC.json"
OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_d_fault_isolation_v1_20261101"
SEALED_ROOT = Path("/home/timbushnell/crucible_artifacts/crucible_cnss_d_fault_isolation_v1_20261101")
CHECKPOINT_ROOT = control.SEALED_ROOT / "checkpoints" / "rank_32"
CONSTANT = 0x37
STATES = tuple(range(256))
LAMBDA_VALUES = (0.1, 0.4, 0.9, 1.0)
SEEDS = (42, 1337, 2025)
UPDATE_STEPS = {2000: (0, 500, 1000, 1500, 2000), 8000: (0, 1000, 2000, 4000, 6000, 8000)}
ERROR_MASKS = (0x80, 0x60, 0x40, 0xC0)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_nested(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
    if isinstance(value, dict):
        return all(finite_nested(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_nested(item) for item in value)
    return True


def answer_free(value: Any) -> bool:
    forbidden = {"target", "targets", "prediction", "predictions", "predicted_values", "expected_values", "displacement_vectors"}
    if isinstance(value, dict):
        return all(key not in forbidden and answer_free(item) for key, item in value.items())
    if isinstance(value, list):
        return all(answer_free(item) for item in value)
    return True


def parse_checkpoint(path: Path) -> tuple[int, int, float, int, int]:
    rank = int(path.parts[-5].split("_")[1])
    updates = int(path.parts[-4].split("_")[1])
    lambda_value = float(path.parts[-3].split("_")[1])
    seed = int(path.parts[-2].split("_")[1])
    step = int(path.stem.split("_")[1])
    return rank, updates, lambda_value, seed, step


def expected_paths() -> list[Path]:
    paths = []
    for updates, steps in UPDATE_STEPS.items():
        for lambda_value in LAMBDA_VALUES:
            for seed in SEEDS:
                for step in steps:
                    paths.append(CHECKPOINT_ROOT / f"updates_{updates}" / f"lambda_{lambda_value:.1f}" / f"seed_{seed}" / f"checkpoint_{step:04d}.pt")
    return paths


def arithmetic_target(value: int) -> int:
    return (value + CONSTANT) & 0xFF


def pctl(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summary(values: Iterable[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0, "mean": None, "std": None, "p10": None, "median": None, "p90": None, "min": None, "max": None}
    mean = sum(numbers) / len(numbers)
    variance = sum((value - mean) ** 2 for value in numbers) / len(numbers)
    return {
        "count": len(numbers),
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "p10": pctl(numbers, 0.10),
        "median": pctl(numbers, 0.50),
        "p90": pctl(numbers, 0.90),
        "min": min(numbers),
        "max": max(numbers),
    }


def average(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return float(sum(x * y for x, y in zip(dx, dy)) / denom) if denom else None


def rank_values(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = rank
        index = end
    return ranks


def correlations(xs: list[float], ys: list[float]) -> dict[str, Any]:
    return {"count": len(xs), "pearson": pearson(xs, ys), "spearman": pearson(rank_values(xs), rank_values(ys))}


def signed_margins(logits: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
    return logits * (target_bits * 2.0 - 1.0)


def decoder_jacobian(workspace: partner.runner.Workspace, states: torch.Tensor) -> torch.Tensor:
    """Return d(logit_bit)/d(state) for every state and output bit."""
    first = workspace.decoder[0](states)
    derivative = 0.5 * (1.0 + torch.erf(first / math.sqrt(2.0))) + first * torch.exp(-0.5 * first.pow(2)) / math.sqrt(2.0 * math.pi)
    first_weight = workspace.decoder[0].weight
    second_weight = workspace.decoder[2].weight
    return torch.einsum("bo,no,oi->nbi", second_weight, derivative, first_weight)


def carry_features(value: int) -> dict[str, Any]:
    carry = 0
    carry_outs = []
    for bit in range(8):
        total = ((value >> bit) & 1) + ((CONSTANT >> bit) & 1) + carry
        carry = int(total >= 2)
        carry_outs.append(carry)
    longest = 0
    current = 0
    for item in carry_outs:
        if item:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "carry_event_count": int(sum(carry_outs)),
        "carry_chain_length": int(longest),
        "wraparound": bool(carry_outs[-1]),
    }


def state_features(value: int) -> tuple[int, int, bool, int]:
    features = carry_features(value)
    return (features["carry_event_count"], features["carry_chain_length"], features["wraparound"], value.bit_count())


def vector_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if left_norm.item() == 0.0 or right_norm.item() == 0.0:
        return 0.0
    return float(torch.dot(left, right).item() / (left_norm.item() * right_norm.item()))


def pca_summary(vectors: list[torch.Tensor]) -> dict[str, Any]:
    if not vectors:
        return {"count": 0, "rank_95": None, "top_explained_variance": [], "mean_vector_norm": None}
    matrix = torch.stack(vectors).to(torch.float64)
    mean_vector = matrix.mean(dim=0)
    centered = matrix - mean_vector
    if len(vectors) < 2 or float(centered.abs().max().item()) == 0.0:
        return {"count": len(vectors), "rank_95": 0, "top_explained_variance": [], "mean_vector_norm": float(torch.linalg.vector_norm(mean_vector).item())}
    singular = torch.linalg.svdvals(centered)
    variance = singular.pow(2)
    total = float(variance.sum().item())
    ratios = (variance / total).tolist() if total else []
    cumulative = 0.0
    rank_95 = None
    for index, ratio in enumerate(ratios, start=1):
        cumulative += ratio
        if cumulative >= 0.95:
            rank_95 = index
            break
    return {
        "count": len(vectors),
        "rank_95": rank_95,
        "top_explained_variance": [float(value) for value in ratios[:5]],
        "mean_vector_norm": float(torch.linalg.vector_norm(mean_vector).item()),
    }


def aggregate_vector_group(samples: list[dict[str, Any]], vector_key: str = "displacement") -> dict[str, Any]:
    vectors = [sample[vector_key] for sample in samples]
    norms = [float(torch.linalg.vector_norm(vector).item()) for vector in vectors]
    cosines = [sample["cosine"] for sample in samples]
    margins = [sample["actual_min_margin"] for sample in samples]
    projections = [sample["wrong_bit_projection_mean"] for sample in samples]
    return {
        "sample_count": len(samples),
        "displacement_norm": summary(norms),
        "canonical_cosine": summary(cosines),
        "actual_min_signed_margin": summary(margins),
        "wrong_bit_margin_projection": summary(projections),
        "pca": pca_summary(vectors),
    }


def evaluate_checkpoint(
    workspace: partner.runner.Workspace,
    adapter: torch.nn.Module,
    base_states: torch.Tensor,
    target_states: torch.Tensor,
    operand_bits: torch.Tensor,
    target_bits: torch.Tensor,
    decoder_gradients: torch.Tensor,
    manifest: dict[str, Any],
    retain_state_records: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    with torch.no_grad():
        actual_states = adapter(base_states, operand_bits)
        actual_logits = workspace.decode_logits(actual_states)
        canonical_logits = workspace.decode_logits(target_states)
        actual_margins = signed_margins(actual_logits, target_bits)
        canonical_margins = signed_margins(canonical_logits, target_bits)
        predicted_bits = (actual_logits >= 0.0).to(torch.float32)
        predicted_values = partner.runner.int_from_bits(predicted_bits).tolist()
        expected_values = [arithmetic_target(value) for value in STATES]
    displacement = actual_states - target_states
    displacement_norm = torch.linalg.vector_norm(displacement, dim=1)
    target_norm = torch.linalg.vector_norm(target_states, dim=1)
    cosine = F.cosine_similarity(actual_states, target_states, dim=1)
    error_masks = [predicted ^ expected for predicted, expected in zip(predicted_values, expected_values)]
    exact_flags = [mask == 0 for mask in error_masks]
    carries = [carry_features(value) for value in STATES]
    signed_gradient = decoder_gradients * (target_bits.unsqueeze(-1) * 2.0 - 1.0)
    gradient_norm = torch.linalg.vector_norm(signed_gradient, dim=2).clamp_min(1e-12)
    unit_gradient = signed_gradient / gradient_norm.unsqueeze(-1)
    projections = (displacement.unsqueeze(1) * unit_gradient).sum(dim=2)
    wrong_projection = []
    for index, mask in enumerate(error_masks):
        wrong_bits = [bit for bit in range(8) if (mask >> bit) & 1]
        wrong_projection.append(float(projections[index, wrong_bits].mean().item()) if wrong_bits else 0.0)
    rows = {
        "exact_accuracy": float(sum(exact_flags) / len(exact_flags)),
        "canonical_exact_accuracy": float(((canonical_margins >= 0.0).all(dim=1)).to(torch.float32).mean().item()),
        "mean_displacement_norm": float(displacement_norm.mean().item()),
        "mean_target_state_norm": float(target_norm.mean().item()),
        "mean_canonical_cosine": float(cosine.mean().item()),
        "mean_actual_min_signed_margin": float(actual_margins.min(dim=1).values.mean().item()),
        "mean_canonical_min_signed_margin": float(canonical_margins.min(dim=1).values.mean().item()),
        "mean_wrong_bit_margin_projection": float(sum(wrong_projection) / len(wrong_projection)),
        "correlations": {
            "carry_count_vs_displacement_norm": correlations([float(item["carry_event_count"]) for item in carries], displacement_norm.tolist()),
            "carry_chain_vs_displacement_norm": correlations([float(item["carry_chain_length"]) for item in carries], displacement_norm.tolist()),
            "carry_count_vs_actual_min_margin": correlations([float(item["carry_event_count"]) for item in carries], actual_margins.min(dim=1).values.tolist()),
            "carry_chain_vs_actual_min_margin": correlations([float(item["carry_chain_length"]) for item in carries], actual_margins.min(dim=1).values.tolist()),
            "displacement_norm_vs_actual_min_margin": correlations(displacement_norm.tolist(), actual_margins.min(dim=1).values.tolist()),
        },
        "carry_buckets": {},
        "chain_buckets": {},
        "wraparound_buckets": {},
        "output_bit_accuracy_lsb_to_msb": [
            float((((actual_logits[:, bit] >= 0.0).to(torch.float32)) == target_bits[:, bit]).to(torch.float32).mean().item())
            for bit in range(8)
        ],
    }
    for field, key in (("carry_buckets", "carry_event_count"), ("chain_buckets", "carry_chain_length"), ("wraparound_buckets", "wraparound")):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(carries):
            groups[str(item[key]).lower() if isinstance(item[key], bool) else str(item[key])].append(index)
        for name, indices in sorted(groups.items(), key=lambda item: item[0]):
            rows[field][name] = {
                "sample_count": len(indices),
                "exact_accuracy": float(sum(exact_flags[index] for index in indices) / len(indices)),
                "displacement_norm": summary([float(displacement_norm[index].item()) for index in indices]),
                "actual_min_signed_margin": summary([float(actual_margins[index].min().item()) for index in indices]),
                "canonical_min_signed_margin": summary([float(canonical_margins[index].min().item()) for index in indices]),
            }
    state_records = None
    if retain_state_records:
        state_records = {
            "predicted_values": predicted_values,
            "expected_values": expected_values,
            "error_masks": error_masks,
            "displacement_norms": displacement_norm.tolist(),
            "canonical_cosines": cosine.tolist(),
            "actual_min_signed_margins": actual_margins.min(dim=1).values.tolist(),
            "canonical_min_signed_margins": canonical_margins.min(dim=1).values.tolist(),
            "wrong_bit_margin_projections": wrong_projection,
            "carry_features": carries,
            "displacements": displacement.tolist(),
            "actual_margins": actual_margins.tolist(),
            "canonical_margins": canonical_margins.tolist(),
            "decoder_margin_projections": projections.tolist(),
        }
    return rows, state_records


def match_controls(samples_by_checkpoint: dict[str, list[dict[str, Any]]], correct_by_checkpoint: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rng = random.Random(20261101)
    output = {}
    for mask in ERROR_MASKS:
        samples = [sample for rows in samples_by_checkpoint.values() for sample in rows if sample["error_mask"] == mask]
        controls = []
        tier_counts = defaultdict(int)
        for sample in samples:
            pool = correct_by_checkpoint[sample["checkpoint"]]
            exact = [candidate for candidate in pool if candidate["features"] == sample["features"]]
            tier = "exact_features"
            if not exact:
                exact = [candidate for candidate in pool if candidate["features"][:3] == sample["features"][:3]]
                tier = "carry_chain_wrap"
            if not exact:
                exact = [candidate for candidate in pool if candidate["features"][0] == sample["features"][0]]
                tier = "carry_count"
            if not exact:
                exact = pool
                tier = "all_correct"
            if not exact:
                continue
            chosen = exact[rng.randrange(len(exact))]
            chosen = dict(chosen)
            wrong_bits = [bit for bit in range(8) if (mask >> bit) & 1]
            chosen["wrong_bit_projection_mean"] = float(sum(chosen["decoder_projections"][bit] for bit in wrong_bits) / len(wrong_bits))
            tier_counts[tier] += 1
            controls.append(chosen)
        group_summary = aggregate_vector_group(samples) if samples else {"sample_count": 0}
        control_summary = aggregate_vector_group(controls) if controls else {"sample_count": 0}
        output[str(mask)] = {
            "error_sample_count": len(samples),
            "matched_control_count": len(controls),
            "matching_tiers": dict(tier_counts),
            "error_group": group_summary,
            "matched_control_group": control_summary,
        }
    return output


def interpolation_summary(final_state_records: dict[str, dict[str, Any]], workspace: partner.runner.Workspace, target_states: torch.Tensor, target_bits: torch.Tensor) -> dict[str, Any]:
    grouped: dict[str, dict[float, list[float]]] = {str(mask): {alpha: [] for alpha in ALPHAS} for mask in ERROR_MASKS}
    overall = {alpha: [] for alpha in ALPHAS}
    for record in final_state_records.values():
        actual = torch.tensor(record["displacements"], dtype=torch.float32) + target_states
        canonical = target_states
        masks = record["error_masks"]
        for alpha in ALPHAS:
            states = canonical + alpha * (actual - canonical)
            with torch.no_grad():
                logits = workspace.decode_logits(states)
                margins = signed_margins(logits, target_bits)
                exact = (margins >= 0.0).all(dim=1).tolist()
            overall[alpha].extend(float(item) for item in exact)
            for mask in ERROR_MASKS:
                grouped[str(mask)][alpha].extend(float(exact[index]) for index, value in enumerate(masks) if value == mask)
    output = {"all_states": {str(alpha): {"count": len(values), "exact_accuracy": average(values)} for alpha, values in overall.items()}}
    output["error_masks"] = {
        mask: {str(alpha): {"count": len(values), "exact_accuracy": average(values)} for alpha, values in alpha_values.items()}
        for mask, alpha_values in grouped.items()
    }
    return output


def main() -> int:
    if OUTPUT.exists() and (OUTPUT / "D_FAULT_ISOLATION_RESULT.json").exists():
        raise RuntimeError("fault-isolation result already exists; refusing to overwrite it")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "REGISTERED_READ_ONLY_DIAGNOSTIC_AUTHORIZED":
        raise RuntimeError("fault-isolation protocol is not authorized")
    if protocol["authorization_boundary"]["training_authorized"] or protocol["authorization_boundary"]["gpu_authorized"]:
        raise RuntimeError("fault-isolation protocol authorizes training or GPU")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source_result = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    workspace = control.load_workspace()
    if partner.runner.state_dict_digest(workspace) != control.EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError("workspace digest mismatch")
    if not partner.runner.finite_module(workspace):
        raise RuntimeError("workspace is non-finite")
    paths = sorted(CHECKPOINT_ROOT.glob("updates_*/lambda_*/seed_*/checkpoint_*.pt"))
    if paths != sorted(expected_paths()):
        raise RuntimeError(f"rank-32 checkpoint set mismatch: {len(paths)} found")
    if manifest["manifest_sha256"] != source_result["manifest_sha256"]:
        raise RuntimeError("manifest/result hash mismatch")
    values = torch.tensor(STATES, dtype=torch.int64)
    input_bits = partner.runner.bits_from_int(values)
    target_values = torch.tensor([arithmetic_target(value) for value in STATES], dtype=torch.int64)
    target_bits = partner.runner.bits_from_int(target_values)
    with torch.no_grad():
        base_states = workspace.encode(input_bits)
        target_states = workspace.encode(target_bits)
        canonical_logits = workspace.decode_logits(target_states)
        input_reconstruction_logits = workspace.decode_logits(base_states)
    input_reconstruction = partner.runner.int_from_bits((input_reconstruction_logits >= 0.0).to(torch.float32)).tolist()
    target_reconstruction = partner.runner.int_from_bits((canonical_logits >= 0.0).to(torch.float32)).tolist()
    input_reconstruction_exact = sum(a == b for a, b in zip(input_reconstruction, STATES)) / len(STATES)
    target_reconstruction_exact = sum(a == b for a, b in zip(target_reconstruction, target_values.tolist())) / len(STATES)
    decoder_gradients = decoder_jacobian(workspace, target_states)

    source_rows = {(row["rank"], row["updates"], row["lambda"], row["seed"]): row for row in source_result["rows"]}
    trajectory = []
    final_state_records: dict[str, dict[str, Any]] = {}
    error_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    correct_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    checkpoint_count = 0
    for path in paths:
        rank, updates, lambda_value, seed, step = parse_checkpoint(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["workspace_digest"] != control.EXPECTED_WORKSPACE_DIGEST or not finite_nested(payload):
            raise RuntimeError(f"invalid checkpoint dependency/state: {path}")
        adapter = control.RankedSkillAdapter(rank)
        adapter.load_state_dict(payload["adapter"], strict=True)
        if not partner.runner.finite_module(adapter):
            raise RuntimeError(f"non-finite adapter: {path}")
        checkpoint_hash = digest_bytes(path.read_bytes())
        source_row = source_rows[(rank, updates, lambda_value, seed)]
        recorded_hash = next(item["sha256"] for item in source_row["checkpoint_hashes"] if item["step"] == step)
        if checkpoint_hash != recorded_hash:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        metrics, state_records = evaluate_checkpoint(
            workspace, adapter, base_states, target_states, partner.operand_bits("D", len(STATES)), target_bits, decoder_gradients, manifest, step == updates
        )
        key = f"updates_{updates}_lambda_{lambda_value:.1f}_seed_{seed}_step_{step:04d}"
        trajectory.append({
            "key": key,
            "rank": rank,
            "updates": updates,
            "lambda": lambda_value,
            "seed": seed,
            "step": step,
            "checkpoint_sha256": checkpoint_hash,
            "metrics": metrics,
        })
        if state_records is not None:
            final_state_records[key] = state_records
            for index, mask in enumerate(state_records["error_masks"]):
                sample = {
                    "checkpoint": key,
                    "features": state_features(STATES[index]),
                    "error_mask": mask,
                    "displacement": torch.tensor(state_records["displacements"][index], dtype=torch.float32),
                    "cosine": state_records["canonical_cosines"][index],
                    "actual_min_margin": state_records["actual_min_signed_margins"][index],
                    "wrong_bit_projection_mean": state_records["wrong_bit_margin_projections"][index],
                    "decoder_projections": state_records["decoder_margin_projections"][index],
                }
                if mask in ERROR_MASKS:
                    error_samples[key].append(sample)
                if mask == 0:
                    correct_samples[key].append(sample)
        checkpoint_count += 1

    final_trajectory = [row for row in trajectory if row["step"] == row["updates"]]
    group_matches = match_controls(error_samples, correct_samples)
    interpolation = interpolation_summary(final_state_records, workspace, target_states, target_bits)
    all_final_metrics = [row["metrics"] for row in final_trajectory]
    result = {
        "schema_version": "crucible-cnss-d-fault-isolation-result-v1-20261101",
        "classification": "COMPLETED_READ_ONLY_FAULT_ISOLATION",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
        "source_result_sha256": digest_bytes(SOURCE_RESULT.read_bytes()),
        "manifest_sha256": manifest["manifest_sha256"],
        "workspace_digest": control.EXPECTED_WORKSPACE_DIGEST,
        "checkpoint_count": checkpoint_count,
        "final_checkpoint_count": len(final_trajectory),
        "training_performed": False,
        "gpu_used": False,
        "checkpoint_hashes_verified": True,
        "canonical_reconstruction": {
            "input_state_exact_accuracy": input_reconstruction_exact,
            "target_state_exact_accuracy": target_reconstruction_exact,
            "input_state_bit_accuracy_lsb_to_msb": [float((((input_reconstruction_logits[:, bit] >= 0.0).to(torch.float32)) == input_bits[:, bit]).to(torch.float32).mean().item()) for bit in range(8)],
            "target_state_bit_accuracy_lsb_to_msb": [float((((canonical_logits[:, bit] >= 0.0).to(torch.float32)) == target_bits[:, bit]).to(torch.float32).mean().item()) for bit in range(8)],
        },
        "final_checkpoint_metrics": final_trajectory,
        "checkpoint_trajectory_metrics": trajectory,
        "high_bit_error_groups": group_matches,
        "canonical_to_actual_interpolation": interpolation,
        "partner_rerun_unlocked": False,
        "scientific_boundary": "This audit localizes observed state, adapter and decoder behavior but does not establish an internal carry algorithm, a unique representation defect, a new capability or permission to rerun partner composition.",
    }
    public_result = {
        key: value for key, value in result.items() if key not in {"final_checkpoint_metrics", "checkpoint_trajectory_metrics", "high_bit_error_groups", "canonical_to_actual_interpolation"}
    }
    public_result["trajectory_summary"] = [
        {
            "rank": row["rank"],
            "updates": row["updates"],
            "lambda": row["lambda"],
            "seed": row["seed"],
            "step": row["step"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "exact_accuracy": row["metrics"]["exact_accuracy"],
            "canonical_exact_accuracy": row["metrics"]["canonical_exact_accuracy"],
            "mean_displacement_norm": row["metrics"]["mean_displacement_norm"],
            "mean_canonical_cosine": row["metrics"]["mean_canonical_cosine"],
            "mean_actual_min_signed_margin": row["metrics"]["mean_actual_min_signed_margin"],
        }
        for row in trajectory
    ]
    public_result["high_bit_group_summary"] = {
        mask: {
            "error_sample_count": data["error_sample_count"],
            "matched_control_count": data["matched_control_count"],
            "matching_tiers": data["matching_tiers"],
            "error_displacement_norm": data["error_group"]["displacement_norm"],
            "control_displacement_norm": data["matched_control_group"]["displacement_norm"],
            "error_margin_projection": data["error_group"]["wrong_bit_margin_projection"],
            "control_margin_projection": data["matched_control_group"]["wrong_bit_margin_projection"],
            "error_pca": data["error_group"]["pca"],
            "control_pca": data["matched_control_group"]["pca"],
        }
        for mask, data in group_matches.items()
    }
    public_result["interpolation_summary"] = interpolation
    if not answer_free(public_result):
        raise RuntimeError("public fault-isolation result contains answer-bearing fields")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SEALED_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "PREFLIGHT_REPORT.json", {
        "protocol_sha256": result["protocol_sha256"],
        "source_result_sha256": result["source_result_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "workspace_digest": result["workspace_digest"],
        "checkpoint_count": checkpoint_count,
        "final_checkpoint_count": len(final_trajectory),
        "training_performed": False,
        "gpu_used": False,
        "partner_rerun_unlocked": False,
    })
    write_json(OUTPUT / "D_FAULT_ISOLATION_RESULT.json", public_result)
    write_json(SEALED_ROOT / "D_FAULT_ISOLATION_RESULT_SEALED.json", result | {"final_state_records": final_state_records})
    lines = [
        "# CNSS Skill D fault-isolation audit — result",
        "",
        "Classification: COMPLETED_READ_ONLY_FAULT_ISOLATION.",
        "",
        f"Audited {checkpoint_count} rank-32 checkpoints ({len(final_trajectory)} final checkpoints) over all 256 states. No training or GPU execution occurred.",
        "",
        f"Canonical input-state reconstruction exact accuracy: {input_reconstruction_exact:.4f}.",
        f"Canonical target-state reconstruction exact accuracy: {target_reconstruction_exact:.4f}.",
        "",
        "The machine-readable report contains carry/chain distributions, correlations, decoder-margin projections, matched high-bit controls and canonical-to-actual interpolation summaries.",
        "",
        "The D competence gate and partner-generalization gate remain closed.",
    ]
    (OUTPUT / "D_FAULT_ISOLATION_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": result["classification"],
        "checkpoint_count": checkpoint_count,
        "final_checkpoint_count": len(final_trajectory),
        "canonical_input_exact_accuracy": input_reconstruction_exact,
        "canonical_target_exact_accuracy": target_reconstruction_exact,
        "partner_rerun_unlocked": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
