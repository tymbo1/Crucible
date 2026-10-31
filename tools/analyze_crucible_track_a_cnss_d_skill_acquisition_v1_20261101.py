#!/usr/bin/env python3
"""Read-only acquisition audit for CNSS skill D."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_crucible_track_a_cnss_partner_generalization_v1_20261101 as partner


PROTOCOL = ROOT / "docs" / "CRUCIBLE_TRACK_A_CNSS_D_SKILL_ACQUISITION_DIAGNOSTIC_V1_20261101.json"
SOURCE_RESULT = ROOT / "experiments" / "crucible_track_a_cnss_partner_generalization_v1_20261101" / "CNSS_PARTNER_GENERALIZATION_RESULT.json"
PARTNER_MANIFEST = ROOT / "experiments" / "crucible_track_a_cnss_partner_generalization_v1_20261101" / "MANIFEST_PUBLIC.json"
OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_acquisition_diagnostic_v1_20261101"
TMP_BACKUP = Path("/tmp/crucible_cnss_partner_generalization_sealed_20261101")
DURABLE_BACKUP = Path("/home/timbushnell/crucible_artifacts/crucible_cnss_partner_generalization_sealed_20261101")
SEALED_OUTPUT = Path("/home/timbushnell/crucible_artifacts/crucible_cnss_d_skill_acquisition_diagnostic_sealed_20261101")
EXPECTED_WORKSPACE_DIGEST = "a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c"
LAMBDA_VALUES = (0.1, 0.4, 0.9, 1.0)
SEEDS = (42, 1337, 2025)
SKILLS = ("A", "B", "C", "D")
CHECKPOINT_STEPS = (0, 500, 1000, 1500, 2000)
CONSTANT = 0x37


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


def exact_accuracy(predictions: list[int], targets: list[int]) -> float:
    if len(predictions) != len(targets):
        raise RuntimeError("prediction/target length mismatch")
    return sum(int(prediction == target) for prediction, target in zip(predictions, targets)) / len(targets)


def bit_accuracy(predictions: list[int], targets: list[int]) -> list[float]:
    counts = [0] * 8
    correct = [0] * 8
    for prediction, target in zip(predictions, targets):
        for bit in range(8):
            counts[bit] += 1
            correct[bit] += int(((prediction >> bit) & 1) == ((target >> bit) & 1))
    return [correct[bit] / counts[bit] for bit in range(8)]


def carry_features(value: int) -> dict[str, Any]:
    carry = 0
    carry_out_positions = []
    active_positions = []
    for bit in range(8):
        input_bit = (value >> bit) & 1
        constant_bit = (CONSTANT >> bit) & 1
        carry_in = carry
        total = input_bit + constant_bit + carry_in
        carry = int(total >= 2)
        if carry:
            carry_out_positions.append(bit)
        if carry_in or carry:
            active_positions.append(bit)
    active_set = set(active_positions)
    longest = 0
    current = 0
    for bit in range(8):
        if bit in active_set:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "carry_count": len(carry_out_positions),
        "carry_chain_length": longest,
        "wraparound": value + CONSTANT >= 256,
        "hamming_weight": value.bit_count(),
        "high_nibble": value >> 4,
        "low_nibble": value & 0x0F,
    }


def verify_backup() -> dict[str, Any]:
    if not DURABLE_BACKUP.is_dir():
        raise RuntimeError(f"durable checkpoint backup is missing: {DURABLE_BACKUP}")

    def inventory(root: Path) -> dict[str, str]:
        return {
            str(path.relative_to(root)): digest_bytes(path.read_bytes())
            for path in root.rglob("*")
            if path.is_file()
        }

    durable = inventory(DURABLE_BACKUP)
    durable_checkpoints = [name for name in durable if name.startswith("checkpoints/") and name.endswith(".pt")]
    if len(durable_checkpoints) != 240:
        raise RuntimeError(f"expected 240 durable checkpoints, found {len(durable_checkpoints)}")
    tmp_present = TMP_BACKUP.is_dir()
    tmp_matches = None
    if tmp_present:
        tmp_matches = inventory(TMP_BACKUP) == durable
        if not tmp_matches:
            raise RuntimeError("durable backup differs from the current /tmp checkpoint tree")
    return {
        "durable_root": str(DURABLE_BACKUP),
        "durable_file_count": len(durable),
        "durable_checkpoint_count": len(durable_checkpoints),
        "tmp_root_present": tmp_present,
        "tmp_and_durable_hashes_match": tmp_matches,
        "durable_backup_verified": True,
    }


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], partner.runner.Workspace, dict[str, Any]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "REGISTERED_READ_ONLY_DIAGNOSTIC_AUTHORIZED":
        raise RuntimeError("diagnostic protocol is not authorized")
    source = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    if source["classification"] != "INCONCLUSIVE_PREREQUISITE_FAILURE":
        raise RuntimeError("source partner result classification was changed")
    manifest = json.loads(PARTNER_MANIFEST.read_text(encoding="utf-8"))
    regenerated = partner.make_manifest()
    if regenerated["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("partner manifest cannot be regenerated exactly")
    sealed = json.loads((DURABLE_BACKUP / "CNSS_PARTNER_GENERALIZATION_RESULT_SEALED.json").read_text(encoding="utf-8"))
    if sealed["workspace_digest"] != EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError("sealed partner result workspace digest mismatch")
    workspace = partner.load_frozen_workspace()
    if partner.runner.state_dict_digest(workspace) != EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError("loaded workspace digest mismatch")
    backup = verify_backup()
    return protocol, source, manifest, workspace, backup | {"sealed": sealed}


def oracle_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    local_targets = [((value + CONSTANT) & 0xFF) for value in range(256)]
    tensor_values = torch.arange(256, dtype=torch.int64)
    partner_targets = partner.skill_target("D", tensor_values).tolist()
    bit_roundtrip = all(
        int(partner.runner.int_from_bits(partner.runner.bits_from_int(torch.tensor([value])))[0].item()) == value
        for value in range(256)
    )
    target_bijection = len(set(local_targets)) == 256
    implementation_match = local_targets == partner_targets
    partitions = manifest["partitions"]["D"]
    partition_disjoint = (
        not set(partitions["train"]).intersection(partitions["validation"])
        and not set(partitions["train"]).intersection(partitions["test"])
        and not set(partitions["validation"]).intersection(partitions["test"])
    )
    return {
        "domain_count": 256,
        "definition": "target = (input + 0x37) mod 256",
        "implementation_matches_independent_oracle": implementation_match,
        "target_bijection": target_bijection,
        "bit_roundtrip_matches_integer_encoding": bit_roundtrip,
        "D_partition_cardinality": {
            "train": len(partitions["train"]) == 128,
            "validation": len(partitions["validation"]) == 64,
            "test": len(partitions["test"]) == 64,
        },
        "D_partition_disjoint": partition_disjoint,
        "all_targets_unique": len(set(local_targets)) == 256,
        "oracle_exact_accuracy": 1.0,
    }


def baseline_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    partitions = manifest["partitions"]["D"]
    targets = {
        split: [((value + CONSTANT) & 0xFF) for value in partitions[split]]
        for split in ("train", "validation", "test")
    }
    train_target_bits = partner.runner.bits_from_int(torch.tensor(targets["train"], dtype=torch.int64))
    majority_bits = (train_target_bits.mean(dim=0) >= 0.5).to(torch.float32)
    majority_value = int(partner.runner.int_from_bits(majority_bits.reshape(1, -1))[0].item())
    rows = {}
    for split in ("train", "validation", "test"):
        inputs = partitions[split]
        target = targets[split]
        rows[split] = {
            "oracle": 1.0,
            "identity": exact_accuracy(inputs, target),
            "no_carry_xor": exact_accuracy([value ^ CONSTANT for value in inputs], target),
            "constant_zero": exact_accuracy([0 for _ in inputs], target),
            "constant_255": exact_accuracy([255 for _ in inputs], target),
            "train_target_bit_majority": exact_accuracy([majority_value for _ in inputs], target),
            "exact_lookup": 1.0 if split == "train" else None,
            "count": len(inputs),
        }
    return {
        "target_bit_majority_value": majority_value,
        "rows": rows,
        "interpretation": "The oracle and lookup are controls, not learned models. The no-carry XOR control tests whether carry-dependent arithmetic is being omitted.",
    }


def load_adapter(path: Path) -> tuple[partner.runner.SkillAdapter, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("workspace_digest") != EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError(f"checkpoint dependency mismatch: {path}")
    if not finite_nested(payload):
        raise RuntimeError(f"non-finite checkpoint payload: {path}")
    adapter = partner.runner.SkillAdapter()
    adapter.load_state_dict(payload["adapter"], strict=True)
    adapter.eval()
    return adapter, payload


def checkpoint_path(skill: str, lambda_value: float, seed: int, step: int) -> Path:
    return (
        DURABLE_BACKUP
        / "checkpoints"
        / f"lambda_{lambda_value:.1f}"
        / f"seed_{seed}"
        / f"skill_{skill}"
        / f"checkpoint_{step:04d}.pt"
    )


def matched_skill_audit(
    workspace: partner.runner.Workspace,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    final_rows = []
    trajectory_rows = []
    gradient_rows = []
    for skill in SKILLS:
        for lambda_value in LAMBDA_VALUES:
            for seed in SEEDS:
                final_adapter, _payload = load_adapter(checkpoint_path(skill, lambda_value, seed, 2000))
                metrics = {}
                for split in ("train", "validation", "test"):
                    metrics[split] = partner.evaluate_single(
                        workspace,
                        final_adapter,
                        manifest["partitions"][skill][split],
                        skill,
                    )
                with torch.enable_grad():
                    probe_adapter, _ = load_adapter(checkpoint_path(skill, lambda_value, seed, 2000))
                    probe_adapter.train()
                    values = torch.tensor(manifest["partitions"][skill]["train"], dtype=torch.int64)
                    output, contract, _transformed, _target = partner.output_and_contract(
                        workspace, probe_adapter, values, skill
                    )
                    loss = output + lambda_value * contract
                    loss.backward()
                    gradient_norm = float(
                        torch.sqrt(
                            sum(
                                parameter.grad.detach().pow(2).sum()
                                for parameter in probe_adapter.parameters()
                                if parameter.grad is not None
                            )
                        ).item()
                    )
                final_rows.append(
                    {
                        "skill": skill,
                        "lambda": lambda_value,
                        "seed": seed,
                        "train_exact_accuracy": metrics["train"]["exact_accuracy"],
                        "validation_exact_accuracy": metrics["validation"]["exact_accuracy"],
                        "test_exact_accuracy": metrics["test"]["exact_accuracy"],
                        "train_bit_accuracy": bit_accuracy(
                            metrics["train"]["predictions"], metrics["train"]["targets"]
                        ),
                        "test_contract_mse": metrics["test"]["contract_mse"],
                        "final_output_loss": float(output.item()),
                        "final_contract_loss": float(contract.item()),
                        "final_gradient_norm": gradient_norm,
                    }
                )
                for step in CHECKPOINT_STEPS:
                    adapter, _ = load_adapter(checkpoint_path(skill, lambda_value, seed, step))
                    step_metrics = {
                        split: partner.evaluate_single(
                            workspace,
                            adapter,
                            manifest["partitions"][skill][split],
                            skill,
                        )["exact_accuracy"]
                        for split in ("train", "validation", "test")
                    }
                    trajectory_rows.append(
                        {
                            "skill": skill,
                            "lambda": lambda_value,
                            "seed": seed,
                            "step": step,
                            **{f"{split}_exact_accuracy": value for split, value in step_metrics.items()},
                        }
                    )
    return final_rows, trajectory_rows, gradient_rows


def aggregate_rows(rows: list[dict[str, Any]], key: str, metric: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(float(row[metric]))
    return {
        group: {
            "count": len(values),
            "mean": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        for group, values in sorted(grouped.items())
    }


def failure_features(value: int) -> dict[str, Any]:
    return {
        "input": value,
        **carry_features(value),
    }


def failure_stratification(
    workspace: partner.runner.Workspace,
    manifest: dict[str, Any],
    sealed: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    sealed_by_condition = {
        (row["skill"], row["lambda"], row["seed"]): row
        for row in sealed["training_rows"]
        if row["skill"] == "D"
    }
    for lambda_value in LAMBDA_VALUES:
        for seed in SEEDS:
            adapter, _ = load_adapter(checkpoint_path("D", lambda_value, seed, 2000))
            values = manifest["partitions"]["D"]["test"]
            evaluated = partner.evaluate_single(workspace, adapter, values, "D")
            stored = sealed_by_condition[("D", lambda_value, seed)]["test"]
            if evaluated["predictions"] != stored["predictions"] or evaluated["targets"] != stored["targets"]:
                raise RuntimeError("durable checkpoint replay differs from recorded D test outputs")
            for value, prediction, target in zip(values, evaluated["predictions"], evaluated["targets"]):
                feature = failure_features(value)
                prediction_bits = [(prediction >> bit) & 1 for bit in range(8)]
                target_bits = [(target >> bit) & 1 for bit in range(8)]
                row = {
                    "lambda": lambda_value,
                    "seed": seed,
                    "prediction": prediction,
                    "target": target,
                    "correct": prediction == target,
                    "bit_error_count": sum(int(a != b) for a, b in zip(prediction_bits, target_bits)),
                    **feature,
                }
                rows.append(row)

    def group(metric: str) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[metric])].append(row)
        return {
            key: {
                "count": len(items),
                "exact_accuracy": sum(int(item["correct"]) for item in items) / len(items),
                "mean_bit_error_count": sum(item["bit_error_count"] for item in items) / len(items),
            }
            for key, items in sorted(grouped.items(), key=lambda item: item[0])
        }

    per_bit_correct = [0] * 8
    per_bit_count = [0] * 8
    for row in rows:
        prediction = row["prediction"]
        target = row["target"]
        for bit in range(8):
            per_bit_count[bit] += 1
            per_bit_correct[bit] += int(((prediction >> bit) & 1) == ((target >> bit) & 1))
    summary = {
        "row_count": len(rows),
        "overall_exact_accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
        "mean_bit_error_count": sum(row["bit_error_count"] for row in rows) / len(rows),
        "by_wraparound": group("wraparound"),
        "by_carry_count": group("carry_count"),
        "by_carry_chain_length": group("carry_chain_length"),
        "by_hamming_weight": group("hamming_weight"),
        "by_high_nibble": group("high_nibble"),
        "by_low_nibble": group("low_nibble"),
        "per_output_bit_lsb_to_msb": [per_bit_correct[bit] / per_bit_count[bit] for bit in range(8)],
        "common_error_patterns": [
            {"prediction": list(key)[0], "target": list(key)[1], "count": count}
            for key, count in Counter((row["prediction"], row["target"]) for row in rows).most_common(20)
        ],
    }
    return summary, rows


def public_skill_summary(final_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for skill in SKILLS:
        selected = [row for row in final_rows if row["skill"] == skill]
        output[skill] = {}
        for metric in ("train_exact_accuracy", "validation_exact_accuracy", "test_exact_accuracy", "final_gradient_norm"):
            values = [float(row[metric]) for row in selected]
            output[skill][metric] = {
                "mean": sum(values) / len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
    return output


def public_trajectory_summary(trajectory_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for skill in SKILLS:
        output[skill] = {}
        for step in CHECKPOINT_STEPS:
            selected = [row for row in trajectory_rows if row["skill"] == skill and row["step"] == step]
            output[skill][str(step)] = {
                metric: {
                    "mean": sum(float(row[metric]) for row in selected) / len(selected),
                    "minimum": min(float(row[metric]) for row in selected),
                    "maximum": max(float(row[metric]) for row in selected),
                }
                for metric in ("train_exact_accuracy", "validation_exact_accuracy", "test_exact_accuracy")
            }
    return output


def main() -> int:
    started = time.perf_counter()
    if OUTPUT.exists() and (OUTPUT / "D_SKILL_ACQUISITION_DIAGNOSTIC_REPORT.json").exists():
        raise RuntimeError("diagnostic result already exists; refusing to overwrite it")
    protocol, source, manifest, workspace, backup = load_inputs()
    oracle = oracle_audit(manifest)
    baselines = baseline_audit(manifest)
    final_rows, trajectory_rows, _gradient_rows = matched_skill_audit(workspace, manifest)
    stratification, sealed_failure_rows = failure_stratification(workspace, manifest, backup["sealed"])

    d_rows = [row for row in final_rows if row["skill"] == "D"]
    d_train_max = max(row["train_exact_accuracy"] for row in d_rows)
    d_test_max = max(row["test_exact_accuracy"] for row in d_rows)
    d_train_mean = sum(row["train_exact_accuracy"] for row in d_rows) / len(d_rows)
    no_specification_defect = all(
        oracle[key]
        for key in (
            "implementation_matches_independent_oracle",
            "target_bijection",
            "bit_roundtrip_matches_integer_encoding",
            "D_partition_disjoint",
        )
    )
    root_cause = {
        "task_oracle_and_encoding_checks_passed": no_specification_defect,
        "D_is_training_competent_at_95_percent": d_train_max >= 0.95,
        "D_is_test_competent_at_95_percent": d_test_max >= 0.95,
        "D_train_accuracy_maximum": d_train_max,
        "D_train_accuracy_mean": d_train_mean,
        "D_test_accuracy_maximum": d_test_max,
        "current_evidence": (
            "D is not training-competent under the existing rank-8 adapter, workspace, "
            "lambda values and 2,000-update budget. The result is therefore not a pure "
            "generalization failure. The read-only evidence is consistent with an "
            "optimization, adapter-capacity or representation mismatch, with no observed "
            "oracle, encoding or partition defect."
        ),
        "not_established": [
            "whether rank 8 is insufficient",
            "whether 2,000 updates are insufficient",
            "whether the frozen workspace discards carry-relevant structure",
            "whether a carry-aware objective would help",
            "whether D would compose after individual competence is restored",
        ],
    }
    public_stratification = dict(stratification)
    public_stratification.pop("common_error_patterns", None)
    public = {
        "schema_version": "crucible-cnss-d-skill-acquisition-diagnostic-result-v1-20261101",
        "classification": "READ_ONLY_DIAGNOSTIC_COMPLETED_PARTNER_GATE_REMAINS_INCONCLUSIVE",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
        "source_partner_result_sha256": digest_bytes(SOURCE_RESULT.read_bytes()),
        "source_partner_commit": "d9fa9fe",
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "read_only": True,
        "optimizer_updates": 0,
        "gpu_used": False,
        "backup_verification": {key: value for key, value in backup.items() if key != "sealed"},
        "oracle_audit": oracle,
        "simple_baselines": baselines,
        "matched_skill_summary": public_skill_summary(final_rows),
        "checkpoint_trajectory_summary": public_trajectory_summary(trajectory_rows),
        "D_failure_stratification": public_stratification,
        "root_cause_assessment": root_cause,
        "corrective_control_protocol": "docs/CRUCIBLE_TRACK_A_CNSS_D_SKILL_CORRECTIVE_CONTROLS_V1_20261101.json",
        "scientific_boundary": (
            "This audit explains the failed individual-skill prerequisite under the tested "
            "conditions. It does not qualify D, change the original partner result, or "
            "establish neural composition."
        ),
        "elapsed_wall_seconds": time.perf_counter() - started,
    }
    # The public report must remain answer-free.
    encoded = json.dumps(public, sort_keys=True)
    if any(token in encoded for token in ('"prediction"', '"predictions"', '"target"', '"targets"')):
        raise RuntimeError("public diagnostic contains answer-bearing keys")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    SEALED_OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUTPUT / "AUTHORIZATION.json",
        {
            "schema_version": "crucible-cnss-d-diagnostic-authorization-v1-20261101",
            "authorization_source": "explicit user instruction: conduct the preregistered acquisition-only diagnostic",
            "read_only": True,
            "training_authorized": False,
            "gpu_authorized": False,
            "source_commit": "d9fa9fe",
        },
    )
    write_json(OUTPUT / "PREFLIGHT_REPORT.json", {
        "protocol_status": protocol["status"],
        "source_classification": source["classification"],
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "backup_verification": {key: value for key, value in backup.items() if key != "sealed"},
        "oracle_audit": oracle,
        "read_only": True,
        "optimizer_updates": 0,
        "gpu_used": False,
    })
    write_json(OUTPUT / "D_SKILL_ACQUISITION_DIAGNOSTIC_REPORT.json", public)
    write_json(
        SEALED_OUTPUT / "D_SKILL_ACQUISITION_DIAGNOSTIC_SEALED.json",
        {
            "public": public,
            "matched_rows": final_rows,
            "trajectory_rows": trajectory_rows,
            "D_failure_rows": sealed_failure_rows,
        },
    )
    markdown = [
        "# CNSS D-skill acquisition diagnostic",
        "",
        "Classification: READ_ONLY_DIAGNOSTIC_COMPLETED_PARTNER_GATE_REMAINS_INCONCLUSIVE",
        "",
        "No optimizer update or GPU operation was performed.",
        "",
        "| Skill | Mean final train accuracy | Maximum final train accuracy | Maximum final test accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for skill, values in public["matched_skill_summary"].items():
        markdown.append(
            f"| {skill} | {values['train_exact_accuracy']['mean']:.4f} | "
            f"{values['train_exact_accuracy']['maximum']:.4f} | "
            f"{values['test_exact_accuracy']['maximum']:.4f} |"
        )
    markdown.extend(
        [
            "",
            f"Oracle/encoding audit passed: {oracle['implementation_matches_independent_oracle'] and oracle['target_bijection'] and oracle['bit_roundtrip_matches_integer_encoding']}.",
            f"D maximum train accuracy: {d_train_max:.4f}; D maximum test accuracy: {d_test_max:.4f}.",
            "",
            "The decisive finding is that D is not training-competent under the existing rank-8 adapter and 2,000-update budget. The partner experiment therefore remains inconclusive; no sender, receiver or pair-specific compatibility conclusion is authorized.",
            "",
            "The proposed rank-versus-update corrective control is registered separately and was not run.",
            "",
            "Machine-readable report: D_SKILL_ACQUISITION_DIAGNOSTIC_REPORT.json.",
        ]
    )
    (OUTPUT / "D_SKILL_ACQUISITION_DIAGNOSTIC_REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": public["classification"],
        "D_train_max": d_train_max,
        "D_test_max": d_test_max,
        "backup_verified": public["backup_verification"]["durable_backup_verified"],
        "output": str((OUTPUT / "D_SKILL_ACQUISITION_DIAGNOSTIC_REPORT.json").relative_to(ROOT)),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
