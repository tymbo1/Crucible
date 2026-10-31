#!/usr/bin/env python3
"""Create the content-addressed CNSS workspace artifact.

The original run did not serialize its frozen workspace.  This tool rebuilds
the workspace through the same deterministic path used by the original run,
requires the recorded state digest to match, and saves the complete state and
its provenance.  It does not retrain adapters or rerun the CNSS experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_crucible_track_a_cnss_semantic_interface_compatibility_v1_20261101 as runner


EXPECTED_DIGEST = "a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c"
RUN_ROOT = ROOT / "experiments" / "crucible_track_a_cnss_semantic_interface_compatibility_v1_rerun_contractfix_20261101"
ARTIFACT_ROOT = RUN_ROOT / "workspace_artifacts" / EXPECTED_DIGEST
STATE_PATH = ARTIFACT_ROOT / "state.pt"
ARCHITECTURE_PATH = ARTIFACT_ROOT / "architecture.json"
SERIALIZATION_PATH = ARTIFACT_ROOT / "serialization.json"
HASHES_PATH = ARTIFACT_ROOT / "HASHES.json"
REPORT_PATH = RUN_ROOT / "WORKSPACE_RECONSTRUCTION_REPORT.json"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def probe_digest(workspace: runner.Workspace) -> str:
    probe = torch.tensor([0, 1, 2, 3, 17, 31, 64, 127, 128, 199, 231, 255], dtype=torch.int64)
    with torch.no_grad():
        logits = workspace.decode_logits(workspace.encode(runner.bits_from_int(probe)))
    return sha256_bytes(logits.detach().cpu().contiguous().numpy().tobytes())


def main() -> int:
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty artifact directory: {ARTIFACT_ROOT}")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = runner.make_manifest()
    workspace_result = runner.train_workspace(manifest)
    workspace = workspace_result.workspace
    workspace_digest = runner.state_dict_digest(workspace)
    if workspace_digest != EXPECTED_DIGEST:
        raise RuntimeError(f"deterministic reconstruction digest mismatch: {workspace_digest} != {EXPECTED_DIGEST}")

    state = {name: value.detach().cpu().clone() for name, value in workspace.state_dict().items()}
    tensor_hashes = {name: tensor_digest(value) for name, value in sorted(state.items())}
    state_payload = {
        "schema_version": "crucible-cnss-frozen-workspace-state-v1-20261101",
        "workspace_digest": workspace_digest,
        "state_dict": state,
    }
    temporary_state = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    torch.save(state_payload, temporary_state)
    os.replace(temporary_state, STATE_PATH)

    source_path = ROOT / "tools" / "run_crucible_track_a_cnss_semantic_interface_compatibility_v1_20261101.py"
    protocol_path = ROOT / "docs" / "CRUCIBLE_TRACK_A_CNSS_SEMANTIC_INTERFACE_COMPATIBILITY_V1_20261101.json"
    architecture = {
        "schema_version": "crucible-cnss-frozen-workspace-architecture-v1-20261101",
        "module": "Workspace",
        "state_bits": runner.STATE_BITS,
        "state_width": runner.STATE_WIDTH,
        "encoder": [
            {"type": "Linear", "in_features": runner.STATE_BITS, "out_features": 64, "bias": True},
            {"type": "GELU"},
            {"type": "Linear", "in_features": 64, "out_features": runner.STATE_WIDTH, "bias": True},
        ],
        "decoder": [
            {"type": "Linear", "in_features": runner.STATE_WIDTH, "out_features": 64, "bias": True},
            {"type": "GELU"},
            {"type": "Linear", "in_features": 64, "out_features": runner.STATE_BITS, "bias": True},
        ],
        "state_schema": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype), "requires_grad": False}
            for name, value in sorted(state.items())
        },
    }
    serialization = {
        "schema_version": "crucible-cnss-frozen-workspace-serialization-v1-20261101",
        "format": "torch.save",
        "loader": "torch.load(path, map_location='cpu', weights_only=False)['state_dict']",
        "map_location": "cpu",
        "state_file_schema": state_payload["schema_version"],
        "tensor_layout": "CPU contiguous tensors; state_dict keys and tensor dtypes are fixed by architecture.json",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    atomic_json(ARCHITECTURE_PATH, architecture)
    atomic_json(SERIALIZATION_PATH, serialization)

    hashes = {
        "schema_version": "crucible-cnss-frozen-workspace-hashes-v1-20261101",
        "workspace_digest": workspace_digest,
        "state_file_sha256": sha256_file(STATE_PATH),
        "architecture_file_sha256": sha256_file(ARCHITECTURE_PATH),
        "serialization_file_sha256": sha256_file(SERIALIZATION_PATH),
        "tensor_sha256": tensor_hashes,
        "source_runner_sha256": sha256_file(source_path),
        "protocol_sha256": sha256_file(protocol_path),
    }
    atomic_json(HASHES_PATH, hashes)

    report = {
        "schema_version": "crucible-cnss-workspace-reconstruction-report-v1-20261101",
        "classification": "DETERMINISTIC_RECONSTRUCTION_ARTIFACT_CREATED",
        "artifact_path": str(ARTIFACT_ROOT.relative_to(ROOT)),
        "workspace_digest": workspace_digest,
        "recorded_digest_match": workspace_digest == EXPECTED_DIGEST,
        "workspace_validation_exact": workspace_result.validation_exact,
        "workspace_test_exact": workspace_result.test_exact,
        "reconstruction_optimizer_steps": runner.WORKSPACE_STEPS,
        "scientific_adapter_training_performed": False,
        "cnss_replication_performed": False,
        "provenance_note": "The original workspace bytes were not persisted. This bundle is a deterministic reconstruction through the original runner, verified by the recorded full state digest; it is not claimed to be the original serialized file.",
        "files": {
            "state": str(STATE_PATH.relative_to(ROOT)),
            "architecture": str(ARCHITECTURE_PATH.relative_to(ROOT)),
            "serialization": str(SERIALIZATION_PATH.relative_to(ROOT)),
            "hashes": str(HASHES_PATH.relative_to(ROOT)),
        },
        "probe_digest": probe_digest(workspace),
    }
    atomic_json(REPORT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
