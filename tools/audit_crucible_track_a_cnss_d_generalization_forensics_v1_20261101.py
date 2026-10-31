#!/usr/bin/env python3
"""Read-only generalization forensics for saved CNSS rank-32 D checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101 as control
import run_crucible_track_a_cnss_partner_generalization_v1_20261101 as partner

PROTOCOL = ROOT / "docs" / "CRUCIBLE_TRACK_A_CNSS_D_GENERALIZATION_FORENSICS_V1_20261101.json"
SOURCE_RESULT = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101" / "D_SKILL_CORRECTIVE_CONTROLS_RESULT.json"
MANIFEST_PATH = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101" / "MANIFEST_PUBLIC.json"
OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_d_generalization_forensics_v1_20261101"
SEALED_ROOT = Path("/home/timbushnell/crucible_artifacts/crucible_cnss_d_generalization_forensics_v1_20261101")
CHECKPOINT_ROOT = control.SEALED_ROOT / "checkpoints" / "rank_32"
CONSTANT = 0x37
STATES = tuple(range(256))
LAMBDA_VALUES = (0.1, 0.4, 0.9, 1.0)
SEEDS = (42, 1337, 2025)
UPDATE_STEPS = {2000: (0, 500, 1000, 1500, 2000), 8000: (0, 1000, 2000, 4000, 6000, 8000)}


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
    if isinstance(value, dict):
        return all(key not in {"target", "targets", "prediction", "predictions", "predicted_values", "expected_values"} and answer_free(item) for key, item in value.items())
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


def binary_addition_features(value: int) -> dict[str, Any]:
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


def bucket_metrics(values: Iterable[int], predicted: list[int], expected: list[int]) -> dict[str, Any]:
    indices = list(values)
    if not indices:
        return {"count": 0, "exact_count": 0, "exact_accuracy": None}
    exact_flags = [predicted[index] == expected[index] for index in indices]
    return {
        "count": len(indices),
        "exact_count": int(sum(exact_flags)),
        "exact_accuracy": float(sum(exact_flags) / len(indices)),
    }


def feature_strata(indices: Iterable[int], predicted: list[int], expected: list[int]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[int]]] = {
        "carry_event_count": defaultdict(list),
        "carry_chain_length": defaultdict(list),
        "wraparound": defaultdict(list),
        "input_hamming_weight": defaultdict(list),
    }
    for index in indices:
        features = binary_addition_features(index)
        groups["carry_event_count"][str(features["carry_event_count"])].append(index)
        groups["carry_chain_length"][str(features["carry_chain_length"])].append(index)
        groups["wraparound"][str(features["wraparound"]).lower()].append(index)
        groups["input_hamming_weight"][str(index.bit_count())].append(index)
    return {
        name: {key: bucket_metrics(bucket, predicted, expected) for key, bucket in sorted(values.items(), key=lambda item: item[0])}
        for name, values in groups.items()
    }


def split_metrics(indices: list[int], predicted: list[int], expected: list[int]) -> dict[str, Any]:
    exact = [predicted[index] == expected[index] for index in indices]
    bit_errors = [0] * 8
    for index in indices:
        difference = predicted[index] ^ expected[index]
        for bit in range(8):
            bit_errors[bit] += (difference >> bit) & 1
    return {
        "count": len(indices),
        "exact_count": int(sum(exact)),
        "exact_accuracy": float(sum(exact) / len(indices)) if indices else None,
        "output_bit_accuracy_lsb_to_msb": [
            float(sum(((predicted[index] >> bit) & 1) == ((expected[index] >> bit) & 1) for index in indices) / len(indices))
            for bit in range(8)
        ] if indices else [],
        "bit_error_counts_lsb_to_msb": bit_errors,
        "strata": feature_strata(indices, predicted, expected),
    }


def local_transition_metrics(predicted: list[int], expected: list[int]) -> dict[str, Any]:
    per_bit = []
    total_pairs = 0
    total_transition_matches = 0
    total_both_exact = 0
    for bit in range(8):
        pairs = []
        for value in STATES:
            if ((value >> bit) & 1) == 0:
                other = value | (1 << bit)
                pairs.append((value, other))
        transition_matches = sum((predicted[a] ^ predicted[b]) == (expected[a] ^ expected[b]) for a, b in pairs)
        both_exact = sum(predicted[a] == expected[a] and predicted[b] == expected[b] for a, b in pairs)
        per_bit.append({
            "input_bit_lsb_index": bit,
            "pair_count": len(pairs),
            "transition_match_count": int(transition_matches),
            "transition_match_accuracy": float(transition_matches / len(pairs)),
            "both_endpoints_exact_count": int(both_exact),
            "both_endpoints_exact_accuracy": float(both_exact / len(pairs)),
        })
        total_pairs += len(pairs)
        total_transition_matches += transition_matches
        total_both_exact += both_exact
    return {
        "pair_count": total_pairs,
        "transition_match_count": int(total_transition_matches),
        "transition_match_accuracy": float(total_transition_matches / total_pairs),
        "both_endpoints_exact_count": int(total_both_exact),
        "both_endpoints_exact_accuracy": float(total_both_exact / total_pairs),
        "by_input_bit": per_bit,
    }


def evaluate_checkpoint(workspace: partner.runner.Workspace, adapter: torch.nn.Module, manifest: dict[str, Any]) -> dict[str, Any]:
    values = list(STATES)
    tensor_values = torch.tensor(values, dtype=torch.int64)
    with torch.no_grad():
        base_state = workspace.encode(partner.runner.bits_from_int(tensor_values))
        operands = partner.operand_bits("D", len(values))
        transformed = adapter(base_state, operands)
        logits = workspace.decode_logits(transformed)
        bits = (logits >= 0.0).to(torch.float32)
        predicted = partner.runner.int_from_bits(bits).tolist()
    expected = [arithmetic_target(value) for value in values]
    partitions = manifest["partitions"]["D"]
    split_indices = {
        "train": partitions["train"],
        "validation": partitions["validation"],
        "test": partitions["test"],
        "all": values,
    }
    return {
        "splits": {name: split_metrics(indices, predicted, expected) for name, indices in split_indices.items()},
        "local_transitions": local_transition_metrics(predicted, expected),
        "predicted_values": predicted,
        "expected_values": expected,
    }


def public_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "splits": evaluation["splits"],
        "local_transitions": evaluation["local_transitions"],
    }


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def main() -> int:
    if OUTPUT.exists() and (OUTPUT / "D_GENERALIZATION_FORENSICS_RESULT.json").exists():
        raise RuntimeError("forensics result already exists; refusing to overwrite it")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "REGISTERED_READ_ONLY_DIAGNOSTIC_AUTHORIZED":
        raise RuntimeError("forensics protocol is not authorized")
    if protocol["authorization_boundary"]["training_authorized"] or protocol["authorization_boundary"]["gpu_authorized"]:
        raise RuntimeError("forensics protocol authorizes training or GPU")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source_result = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    workspace = control.load_workspace()
    if partner.runner.state_dict_digest(workspace) != control.EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError("workspace digest mismatch")
    if not partner.runner.finite_module(workspace):
        raise RuntimeError("workspace is non-finite")
    if manifest["manifest_sha256"] != source_result["manifest_sha256"]:
        raise RuntimeError("manifest/result hash mismatch")
    if set(manifest["partitions"]["D"]["train"]) & set(manifest["partitions"]["D"]["validation"]):
        raise RuntimeError("D train/validation overlap")
    if set(manifest["partitions"]["D"]["train"]) & set(manifest["partitions"]["D"]["test"]):
        raise RuntimeError("D train/test overlap")
    if set(manifest["partitions"]["D"]["validation"]) & set(manifest["partitions"]["D"]["test"]):
        raise RuntimeError("D validation/test overlap")
    if [arithmetic_target(value) for value in STATES] != [(value + CONSTANT) & 0xFF for value in STATES]:
        raise RuntimeError("arithmetic oracle failed exhaustive check")

    paths = sorted(CHECKPOINT_ROOT.glob("updates_*/lambda_*/seed_*/checkpoint_*.pt"))
    expected = expected_paths()
    if paths != sorted(expected):
        raise RuntimeError(f"rank-32 checkpoint set mismatch: {len(paths)} found, {len(expected)} expected")
    source_rows = {(row["rank"], row["updates"], row["lambda"], row["seed"]): row for row in source_result["rows"]}
    detailed_rows = []
    public_rows = []
    checkpoint_count = 0
    for path in paths:
        rank, updates, lambda_value, seed, step = parse_checkpoint(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["workspace_digest"] != control.EXPECTED_WORKSPACE_DIGEST:
            raise RuntimeError(f"workspace dependency mismatch: {path}")
        if not finite_nested(payload):
            raise RuntimeError(f"non-finite checkpoint: {path}")
        adapter = control.RankedSkillAdapter(rank)
        adapter.load_state_dict(payload["adapter"], strict=True)
        if not partner.runner.finite_module(adapter):
            raise RuntimeError(f"non-finite adapter: {path}")
        checkpoint_hash = digest_bytes(path.read_bytes())
        source_row = source_rows[(rank, updates, lambda_value, seed)]
        recorded_hash = next(item["sha256"] for item in source_row["checkpoint_hashes"] if item["step"] == step)
        if checkpoint_hash != recorded_hash:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        evaluation = evaluate_checkpoint(workspace, adapter, manifest)
        row = {
            "rank": rank,
            "updates": updates,
            "lambda": lambda_value,
            "seed": seed,
            "step": step,
            "checkpoint_sha256": checkpoint_hash,
            "evaluation": evaluation,
        }
        detailed_rows.append(row)
        public_rows.append({
            "rank": rank,
            "updates": updates,
            "lambda": lambda_value,
            "seed": seed,
            "step": step,
            "checkpoint_sha256": checkpoint_hash,
            "evaluation": public_evaluation(evaluation),
        })
        checkpoint_count += 1

    final_rows = [row for row in detailed_rows if row["step"] == row["updates"]]
    summaries = {}
    for updates in UPDATE_STEPS:
        for lambda_value in LAMBDA_VALUES:
            rows = [row for row in final_rows if row["updates"] == updates and row["lambda"] == lambda_value]
            test_values = [row["evaluation"]["splits"]["test"]["exact_accuracy"] for row in rows]
            full_values = [row["evaluation"]["splits"]["all"]["exact_accuracy"] for row in rows]
            transition_values = [row["evaluation"]["local_transitions"]["transition_match_accuracy"] for row in rows]
            summaries[f"updates_{updates}_lambda_{lambda_value:.1f}"] = {
                "updates": updates,
                "lambda": lambda_value,
                "seed_count": len(rows),
                "mean_full_domain_accuracy": mean(full_values),
                "mean_test_accuracy": mean(test_values),
                "minimum_test_accuracy": min(test_values),
                "maximum_test_accuracy": max(test_values),
                "mean_local_transition_accuracy": mean(transition_values),
                "per_seed_test_accuracy": {str(row["seed"]): row["evaluation"]["splits"]["test"]["exact_accuracy"] for row in rows},
            }

    final_all = [row["evaluation"] for row in final_rows]
    aggregate_carry = {}
    for split_name in ("train", "validation", "test", "all"):
        bucket_values = defaultdict(list)
        for evaluation in final_all:
            for bucket, metrics in evaluation["splits"][split_name]["strata"]["carry_event_count"].items():
                bucket_values[bucket].append(metrics["exact_accuracy"])
        aggregate_carry[split_name] = {bucket: {"seed_checkpoint_mean_accuracy": mean(values), "replicate_count": len(values)} for bucket, values in sorted(bucket_values.items())}

    result = {
        "schema_version": "crucible-cnss-d-generalization-forensics-result-v1-20261101",
        "classification": "COMPLETED_READ_ONLY_GENERALIZATION_FORENSICS",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
        "source_result_sha256": digest_bytes(SOURCE_RESULT.read_bytes()),
        "manifest_sha256": manifest["manifest_sha256"],
        "workspace_digest": control.EXPECTED_WORKSPACE_DIGEST,
        "checkpoint_count": checkpoint_count,
        "final_checkpoint_count": len(final_rows),
        "public_answer_free": True,
        "oracle_exhaustive": True,
        "partition_integrity": True,
        "checkpoint_hashes_verified": True,
        "condition_summaries": summaries,
        "aggregate_carry_event_summary": aggregate_carry,
        "maximum_final_test_accuracy": max(row["evaluation"]["splits"]["test"]["exact_accuracy"] for row in final_rows),
        "partner_rerun_unlocked": False,
        "scientific_boundary": "This audit localizes observed rank-32 D behavior but does not establish a new capability, composition, cumulative learning or a causal rank/horizon decomposition.",
    }
    if not answer_free(result):
        raise RuntimeError("public result contains per-example answer-bearing fields")

    sealed_result = result | {"rows": detailed_rows}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SEALED_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "PREFLIGHT_REPORT.json", {
        "protocol_sha256": result["protocol_sha256"],
        "source_result_sha256": result["source_result_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "workspace_digest": result["workspace_digest"],
        "rank": 32,
        "checkpoint_count": checkpoint_count,
        "final_checkpoint_count": len(final_rows),
        "training_performed": False,
        "gpu_used": False,
        "partner_rerun_unlocked": False,
    })
    write_json(OUTPUT / "D_GENERALIZATION_FORENSICS_RESULT.json", result)
    write_json(SEALED_ROOT / "D_GENERALIZATION_FORENSICS_RESULT_SEALED.json", sealed_result)

    lines = [
        "# CNSS D rank-32 generalization forensics — result",
        "",
        "Classification: COMPLETED_READ_ONLY_GENERALIZATION_FORENSICS.",
        "",
        f"Audited {checkpoint_count} rank-32 checkpoints ({len(final_rows)} final checkpoints) across all 256 input states. No parameters were updated and no GPU was used.",
        "",
        "| Updates | Lambda | Mean full-domain | Mean test | Min test | Mean local transition |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(summaries):
        item = summaries[key]
        lines.append(f"| {item['updates']} | {item['lambda']:.1f} | {item['mean_full_domain_accuracy']:.4f} | {item['mean_test_accuracy']:.4f} | {item['minimum_test_accuracy']:.4f} | {item['mean_local_transition_accuracy']:.4f} |")
    lines.extend([
        "",
        "Carry-event stratification is included in `D_GENERALIZATION_FORENSICS_RESULT.json`; detailed per-input traces are retained only in the durable sealed artifact.",
        "",
        "The partner-generalization gate remains closed. This audit does not replace the registered 95% individual-competence requirement.",
    ])
    (OUTPUT / "D_GENERALIZATION_FORENSICS_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": result["classification"],
        "checkpoint_count": checkpoint_count,
        "final_checkpoint_count": len(final_rows),
        "maximum_final_test_accuracy": result["maximum_final_test_accuracy"],
        "partner_rerun_unlocked": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
