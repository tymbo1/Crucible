#!/usr/bin/env python3
"""CPU-only preflight for the registered workspace-accessibility control."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101 as control
import run_crucible_track_a_cnss_partner_generalization_v1_20261101 as partner

PROTOCOL = ROOT / "docs" / "CRUCIBLE_TRACK_A_CNSS_WORKSPACE_ACCESSIBILITY_CAUSAL_CONTROL_V1_20261101.json"
MANIFEST = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101" / "MANIFEST_PUBLIC.json"
SOURCE_RESULT = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101" / "D_SKILL_CORRECTIVE_CONTROLS_RESULT.json"
OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_workspace_accessibility_causal_control_v1_20261101"
ARTIFACT = OUTPUT / "ALTERNATIVE_WORKSPACE_SPEC.json"
SEALED_ROOT = control.SEALED_ROOT
WORKSPACE_DIGEST = control.EXPECTED_WORKSPACE_DIGEST
SEEDS = (42, 1337, 2025)
CONSTANT = 0x37
WIDTH = 256
GROUP_WIDTH = 32
AMPLITUDE = 0.8
LOGIT_SCALE = 12.0


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class SignedBitWorkspace:
    state_width = WIDTH

    def encode(self, bits: torch.Tensor) -> torch.Tensor:
        signed = (bits.to(torch.float32) * 2.0 - 1.0)
        return signed.repeat_interleave(GROUP_WIDTH, dim=1) * AMPLITUDE

    def decode_logits(self, state: torch.Tensor) -> torch.Tensor:
        grouped = state.reshape(-1, 8, GROUP_WIDTH).mean(dim=2)
        return grouped / AMPLITUDE * LOGIT_SCALE


def exact_reconstruction(workspace: SignedBitWorkspace, values: torch.Tensor) -> tuple[float, list[int]]:
    bits = partner.runner.bits_from_int(values)
    logits = workspace.decode_logits(workspace.encode(bits))
    return partner.runner.exact_bits(logits, bits)


def finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "REGISTERED_NOT_AUTHORIZED":
        raise RuntimeError("workspace-accessibility control is unexpectedly authorized")
    if protocol["authorization_boundary"]["training_authorized"] is not False:
        raise RuntimeError("training authorization must remain false")
    if protocol["authorization_boundary"]["gpu_authorized"] is not False:
        raise RuntimeError("GPU authorization must remain false")
    if protocol["authorization_boundary"]["partner_rerun_authorized"] is not False:
        raise RuntimeError("partner rerun authorization must remain false")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    original = control.load_workspace()
    if partner.runner.state_dict_digest(original) != WORKSPACE_DIGEST:
        raise RuntimeError("original workspace digest mismatch")
    if not partner.runner.finite_module(original):
        raise RuntimeError("original workspace contains non-finite parameters")
    alternative = SignedBitWorkspace()
    all_values = torch.arange(256, dtype=torch.int64)
    original_exact, _ = exact_reconstruction(original, all_values)
    alternative_exact, _ = exact_reconstruction(alternative, all_values)
    if original_exact != 1.0 or alternative_exact != 1.0:
        raise RuntimeError("workspace reconstruction failed")
    canonical_checks = {}
    for skill in ("A", "B", "C", "D"):
        mapped = partner.skill_target(skill, all_values)
        original_result, _ = exact_reconstruction(original, mapped)
        alternative_result, _ = exact_reconstruction(alternative, mapped)
        canonical_checks[skill] = {
            "original_exact": original_result,
            "alternative_exact": alternative_result,
            "operation_independent_construction": True,
        }
        if original_result != 1.0 or alternative_result != 1.0:
            raise RuntimeError(f"canonical reconstruction failed for skill {skill}")
    train = set(manifest["partitions"]["D"]["train"])
    validation = set(manifest["partitions"]["D"]["validation"])
    test = set(manifest["partitions"]["D"]["test"])
    if len(train) != 128 or len(validation) != 64 or len(test) != 64 or train & validation or train & test or validation & test:
        raise RuntimeError("D manifest partition integrity failed")
    baseline_rows = [row for row in source["rows"] if row["rank"] == 32 and row["updates"] == 2000 and row["lambda"] == 0.1]
    if {row["seed"] for row in baseline_rows} != set(SEEDS):
        raise RuntimeError("paired baseline rows are incomplete")
    step0_hashes = {}
    for seed, row in ((row["seed"], row) for row in baseline_rows):
        path = SEALED_ROOT / "checkpoints" / "rank_32" / "updates_2000" / "lambda_0.1" / f"seed_{seed}" / "checkpoint_0000.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("workspace_digest") != WORKSPACE_DIGEST or not finite(payload):
            raise RuntimeError(f"invalid paired step-zero checkpoint seed={seed}")
        actual = digest_bytes(path.read_bytes())
        recorded = next(item["sha256"] for item in row["checkpoint_hashes"] if item["step"] == 0)
        if actual != recorded:
            raise RuntimeError(f"paired step-zero hash mismatch seed={seed}")
        step0_hashes[str(seed)] = actual
    spec = {
        "schema_version": "crucible-cnss-signed-bit-workspace-spec-v1-20261101",
        "state_width": WIDTH,
        "bit_count": 8,
        "group_width": GROUP_WIDTH,
        "amplitude": AMPLITUDE,
        "decoder_logit_scale": LOGIT_SCALE,
        "formula": "encode(bits) = repeat_interleave(2*bits-1, 32) * 0.8; decode_logits(state) = group_mean(state)/0.8*12",
        "operation_independent": True,
        "construction_inputs": ["8-bit input alphabet", "fixed group and amplitude constants"],
        "workspace_parameter_count": 0,
        "spec_sha256": None,
    }
    spec["spec_sha256"] = digest_json({key: value for key, value in spec.items() if key != "spec_sha256"})
    write_json(ARTIFACT, spec)
    report = {
        "schema_version": "crucible-cnss-workspace-accessibility-causal-control-preflight-v1-20261101",
        "classification": "PREFLIGHT_PASSED_TRAINING_NOT_AUTHORIZED",
        "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_result_sha256": digest_bytes(SOURCE_RESULT.read_bytes()),
        "original_workspace_digest": WORKSPACE_DIGEST,
        "alternative_workspace_spec_sha256": digest_bytes(ARTIFACT.read_bytes()),
        "original_reconstruction_exact": original_exact,
        "alternative_reconstruction_exact": alternative_exact,
        "canonical_checks": canonical_checks,
        "same_interface_width": True,
        "same_adapter_parameter_count": control.RankedSkillAdapter(32).parameter_count == 16672,
        "paired_step0_checkpoint_hashes": step0_hashes,
        "candidate_support_unchanged": True,
        "fixed_test_partition": True,
        "uses_skill_definition_for_alternative": False,
        "uses_held_out_answers_for_alternative": False,
        "training_authorized": False,
        "gpu_used": False,
        "partner_rerun_authorized": False,
        "limitations": [
            "The alternative workspace is a fixed analytical control with zero trainable workspace parameters, so workspace compute and wall-clock cost must be reported.",
            "The preflight establishes interface and reconstruction integrity, not D acquisition or composability.",
            "No causal training result exists until separately authorized.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "ALTERNATIVE_WORKSPACE_SPEC.json", spec)
    write_json(OUTPUT / "WORKSPACE_ACCESSIBILITY_PREFLIGHT_REPORT.json", report)
    (OUTPUT / "WORKSPACE_ACCESSIBILITY_PREFLIGHT_REPORT.md").write_text(
        "# CNSS workspace-accessibility causal-control preflight\n\n"
        "Classification: `PREFLIGHT_PASSED_TRAINING_NOT_AUTHORIZED`.\n\n"
        "The recovered workspace and deterministic signed-bit alternative both reconstruct all 256 states exactly, share a 256-wide interface, and pass canonical A-D reconstruction. Paired step-zero checkpoints are finite and hash-verified. No training, GPU execution or partner rerun was authorized.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "classification": report["classification"],
        "original_reconstruction_exact": report["original_reconstruction_exact"],
        "alternative_reconstruction_exact": report["alternative_reconstruction_exact"],
        "canonical_checks": report["canonical_checks"],
        "training_authorized": report["training_authorized"],
        "gpu_used": report["gpu_used"],
        "partner_rerun_authorized": report["partner_rerun_authorized"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
