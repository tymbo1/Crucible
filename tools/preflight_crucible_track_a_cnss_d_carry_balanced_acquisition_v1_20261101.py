#!/usr/bin/env python3
"""CPU-only preflight for the registered D carry-balanced acquisition control."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101 as control
import run_crucible_track_a_cnss_partner_generalization_v1_20261101 as partner

PROTOCOL = ROOT / "docs" / "CRUCIBLE_TRACK_A_CNSS_D_CARRY_BALANCED_ACQUISITION_V1_20261101.json"
MANIFEST = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101" / "MANIFEST_PUBLIC.json"
SOURCE_RESULT = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101" / "D_SKILL_CORRECTIVE_CONTROLS_RESULT.json"
RAW_ROOT = control.SEALED_ROOT
OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_d_carry_balanced_acquisition_v1_20261101"
QUOTAS = {index: 15 if index < 2 else 14 for index in range(9)}
SEEDS = (42, 1337, 2025)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def features(value: int) -> tuple[int, int, bool]:
    carry = 0
    carry_outs = []
    for bit in range(8):
        total = ((value >> bit) & 1) + ((0x37 >> bit) & 1) + carry
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
    return sum(carry_outs), longest, bool(carry_outs[-1])


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
        raise RuntimeError("carry-balanced protocol is unexpectedly authorized")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    workspace = control.load_workspace()
    if partner.runner.state_dict_digest(workspace) != control.EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError("workspace digest mismatch")
    train = list(manifest["partitions"]["D"]["train"])
    validation = set(manifest["partitions"]["D"]["validation"])
    test = set(manifest["partitions"]["D"]["test"])
    if len(train) != 128 or len(set(train)) != 128:
        raise RuntimeError("D training support is not 128 unique states")
    if set(train) & validation or set(train) & test or validation & test:
        raise RuntimeError("D partition overlap")
    strata = {str(index): sorted(value for value in train if features(value)[0] == index) for index in range(9)}
    if any(not values for values in strata.values()):
        raise RuntimeError("existing training support lacks a carry stratum")
    balanced_pool = []
    for index in range(9):
        values = strata[str(index)]
        for offset in range(QUOTAS[index]):
            balanced_pool.append(values[offset % len(values)])
    if len(balanced_pool) != 128 or not set(balanced_pool).issubset(set(train)):
        raise RuntimeError("balanced epoch changed size or escaped candidate support")
    balanced_counts = Counter(features(value)[0] for value in balanced_pool)
    if dict(sorted(balanced_counts.items())) != QUOTAS:
        raise RuntimeError("balanced quotas are not exact")
    baseline_rows = [row for row in source["rows"] if row["rank"] == 32 and row["updates"] == 2000 and row["lambda"] == 0.1]
    if {row["seed"] for row in baseline_rows} != set(SEEDS):
        raise RuntimeError("baseline rank-32 lambda-0.1 rows are incomplete")
    step0_hashes = {}
    for seed in SEEDS:
        path = RAW_ROOT / "checkpoints" / "rank_32" / "updates_2000" / "lambda_0.1" / f"seed_{seed}" / "checkpoint_0000.pt"
        if not path.exists():
            raise RuntimeError(f"missing paired step-0 checkpoint: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["workspace_digest"] != control.EXPECTED_WORKSPACE_DIGEST or not finite(payload):
            raise RuntimeError(f"invalid paired step-0 checkpoint: {path}")
        row = next(item for item in baseline_rows if item["seed"] == seed)
        recorded = next(item["sha256"] for item in row["checkpoint_hashes"] if item["step"] == 0)
        actual = digest_bytes(path.read_bytes())
        if actual != recorded:
            raise RuntimeError(f"step-0 hash mismatch: {path}")
        step0_hashes[str(seed)] = actual
    result = {
        "schema_version": "crucible-cnss-d-carry-balanced-acquisition-preflight-v1-20261101",
        "classification": "PREFLIGHT_PASSED_TRAINING_NOT_AUTHORIZED",
        "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_result_sha256": digest_bytes(SOURCE_RESULT.read_bytes()),
        "workspace_digest": control.EXPECTED_WORKSPACE_DIGEST,
        "training_authorized": False,
        "gpu_used": False,
        "unique_train_states": 128,
        "balanced_pool_size": len(balanced_pool),
        "existing_train_carry_counts": {str(k): len(v) for k, v in strata.items()},
        "balanced_carry_counts": {str(k): balanced_counts[k] for k in sorted(balanced_counts)},
        "balanced_chain_counts": dict(sorted(Counter(features(value)[1] for value in balanced_pool).items())),
        "balanced_wrap_counts": {str(k).lower(): v for k, v in sorted(Counter(features(value)[2] for value in balanced_pool).items(), key=lambda item: str(item[0]))},
        "train_validation_test_overlap": False,
        "candidate_support_unchanged": True,
        "initial_preflight_correction": "The first preflight implementation incorrectly required one balanced epoch to contain every candidate state; exact quotas and unequal source strata make that impossible. The corrected gate checks candidate-support containment instead. No training occurred during the failed preflight.",
        "fixed_test_manifest": True,
        "paired_step0_checkpoint_hashes": step0_hashes,
        "partner_rerun_unlocked": False
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "CARRY_BALANCED_PREFLIGHT_REPORT.json", result)
    (OUTPUT / "CARRY_BALANCED_PREFLIGHT_REPORT.md").write_text(
        "# D carry-balanced acquisition preflight\n\n"
        "Classification: `PREFLIGHT_PASSED_TRAINING_NOT_AUTHORIZED`.\n\n"
        "The balanced epoch draws only from the same 128-state D training candidate support, with exact carry-event quotas and no validation/test overlap. Paired step-0 checkpoints were finite and hash-verified for all three seeds. No training or GPU execution occurred.\n",
        encoding="utf-8",
    )
    print(json.dumps({"classification": result["classification"], "balanced_counts": result["balanced_carry_counts"], "training_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
