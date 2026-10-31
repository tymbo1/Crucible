#!/usr/bin/env python3
"""Fresh-process integrity and dependency verification for a CNSS workspace."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "experiments" / "crucible_track_a_cnss_semantic_interface_compatibility_v1_rerun_contractfix_20261101"
EXPECTED_DIGEST = "a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c"
ARTIFACT = RUN_ROOT / "workspace_artifacts" / EXPECTED_DIGEST
REPLAY_REPORT = RUN_ROOT / "CHECKPOINT_REPLAY_REPORT.json"


class ArtifactIntegrityError(RuntimeError):
    pass


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


def state_digest(state: dict[str, torch.Tensor]) -> str:
    per_tensor = {name: tensor_digest(value) for name, value in sorted(state.items())}
    return sha256_bytes(canonical(per_tensor).encode("utf-8"))


def probe_digest(workspace: torch.nn.Module) -> str:
    probe = torch.tensor([0, 1, 2, 3, 17, 31, 64, 127, 128, 199, 231, 255], dtype=torch.int64)
    values = probe.reshape(-1, 1)
    shifts = torch.arange(7, -1, -1, dtype=torch.int64).reshape(1, -1)
    bits = ((values >> shifts) & 1).to(torch.float32)
    with torch.no_grad():
        logits = workspace.decode_logits(workspace.encode(bits))
    return sha256_bytes(logits.detach().cpu().contiguous().numpy().tobytes())


def load_bundle(root: Path, expected_digest: str = EXPECTED_DIGEST):
    required = [root / "state.pt", root / "architecture.json", root / "serialization.json", root / "HASHES.json"]
    if not all(path.is_file() for path in required):
        raise ArtifactIntegrityError("required workspace artifact file is missing")
    hashes = json.loads((root / "HASHES.json").read_text(encoding="utf-8"))
    architecture = json.loads((root / "architecture.json").read_text(encoding="utf-8"))
    serialization = json.loads((root / "serialization.json").read_text(encoding="utf-8"))
    if hashes.get("workspace_digest") != expected_digest:
        raise ArtifactIntegrityError("workspace digest identity mismatch")
    if sha256_file(root / "state.pt") != hashes.get("state_file_sha256"):
        raise ArtifactIntegrityError("state file hash mismatch")
    if sha256_file(root / "architecture.json") != hashes.get("architecture_file_sha256"):
        raise ArtifactIntegrityError("architecture file hash mismatch")
    if sha256_file(root / "serialization.json") != hashes.get("serialization_file_sha256"):
        raise ArtifactIntegrityError("serialization file hash mismatch")
    if serialization.get("state_file_schema") != "crucible-cnss-frozen-workspace-state-v1-20261101":
        raise ArtifactIntegrityError("unsupported state serialization schema")
    payload = torch.load(root / "state.pt", map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
        raise ArtifactIntegrityError("state file does not contain a tensor state_dict")
    if payload.get("workspace_digest") != expected_digest:
        raise ArtifactIntegrityError("state payload digest mismatch")
    schema = architecture.get("state_schema", {})
    if set(schema) != set(state):
        raise ArtifactIntegrityError("architecture/state key mismatch")
    for name, value in state.items():
        expected = schema[name]
        if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
            raise ArtifactIntegrityError(f"architecture/state tensor mismatch for {name}")
        if tensor_digest(value) != hashes["tensor_sha256"].get(name):
            raise ArtifactIntegrityError(f"tensor hash mismatch for {name}")
    if state_digest(state) != expected_digest:
        raise ArtifactIntegrityError("reconstructed state digest mismatch")
    if not all(bool(torch.isfinite(value).all().item()) for value in state.values()):
        raise ArtifactIntegrityError("workspace contains non-finite values")
    return state, architecture, hashes


def copy_bundle(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("state.pt", "architecture.json", "serialization.json", "HASHES.json"):
        shutil.copy2(ARTIFACT / name, destination / name)


def assert_rejects(label: str, root: Path) -> str:
    try:
        load_bundle(root)
    except ArtifactIntegrityError as exc:
        return f"{label}: PASS ({exc})"
    raise RuntimeError(f"negative test did not fail closed: {label}")


def main() -> int:
    state, architecture, hashes = load_bundle(ARTIFACT)

    # Load through the production Workspace class in this independent process.
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import run_crucible_track_a_cnss_semantic_interface_compatibility_v1_20261101 as runner

    workspace = runner.Workspace()
    workspace.load_state_dict(state, strict=True)
    if runner.state_dict_digest(workspace) != EXPECTED_DIGEST:
        raise RuntimeError("production Workspace loader produced a digest mismatch")
    recorded_probe = json.loads((RUN_ROOT / "WORKSPACE_RECONSTRUCTION_REPORT.json").read_text(encoding="utf-8"))["probe_digest"]
    if probe_digest(workspace) != recorded_probe:
        raise RuntimeError("workspace probe output mismatch")

    checkpoint_paths = sorted((RUN_ROOT / "checkpoints").rglob("checkpoint_*.pt"))
    replay_rows = json.loads(REPLAY_REPORT.read_text(encoding="utf-8"))["rows"]
    replay_by_path = {row["path"]: row for row in replay_rows}
    if len(checkpoint_paths) != 210 or len(replay_by_path) != 210:
        raise RuntimeError("dependent checkpoint inventory is not the registered 210-checkpoint set")
    probe = torch.tensor([0, 1, 2, 3, 17, 31, 64, 127, 128, 199, 231, 255], dtype=torch.int64)
    for checkpoint in checkpoint_paths:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("workspace_digest") != hashes["workspace_digest"]:
            raise ArtifactIntegrityError(f"dependent checkpoint references a different workspace: {checkpoint}")
        replay_row = replay_by_path.get(str(checkpoint.relative_to(ROOT)))
        if replay_row is None:
            raise RuntimeError(f"checkpoint is absent from replay report: {checkpoint}")
        adapter = runner.SkillAdapter()
        adapter.load_state_dict(payload["adapter"], strict=True)
        with torch.no_grad():
            state_probe = workspace.encode(runner.bits_from_int(probe))
            transformed = adapter(state_probe, runner.operand_bits(str(payload["skill"]), len(probe)))
            logits = workspace.decode_logits(transformed)
        output_digest = sha256_bytes(logits.detach().cpu().contiguous().numpy().tobytes())
        if output_digest != replay_row["output_digest"]:
            raise RuntimeError(f"dependent checkpoint output replay mismatch: {checkpoint}")

    with tempfile.TemporaryDirectory(prefix="crucible-cnss-artifact-negative-") as temporary:
        temporary_root = Path(temporary)
        missing = temporary_root / "missing"
        missing.mkdir()
        missing_result = assert_rejects("missing workspace", missing)

        corrupted = temporary_root / "corrupted"
        copy_bundle(corrupted)
        raw = bytearray((corrupted / "state.pt").read_bytes())
        raw[-1] ^= 1
        (corrupted / "state.pt").write_bytes(raw)
        corrupted_result = assert_rejects("corrupted workspace", corrupted)

        incompatible = temporary_root / "incompatible"
        copy_bundle(incompatible)
        altered_architecture = json.loads((incompatible / "architecture.json").read_text(encoding="utf-8"))
        first_key = sorted(altered_architecture["state_schema"])[0]
        altered_architecture["state_schema"][first_key]["shape"][0] += 1
        (incompatible / "architecture.json").write_text(json.dumps(altered_architecture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        incompatible_hashes = json.loads((incompatible / "HASHES.json").read_text(encoding="utf-8"))
        incompatible_hashes["architecture_file_sha256"] = sha256_file(incompatible / "architecture.json")
        (incompatible / "HASHES.json").write_text(json.dumps(incompatible_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        incompatible_result = assert_rejects("incompatible workspace", incompatible)

    report = {
        "schema_version": "crucible-cnss-workspace-artifact-verification-v1-20261101",
        "classification": "WORKSPACE_ARTIFACT_VERIFY_PASS",
        "workspace_digest": EXPECTED_DIGEST,
        "state_tensor_count": len(state),
        "fresh_process_load": True,
        "dependent_checkpoints_verified": len(checkpoint_paths),
        "dependent_outputs_reproduced": len(checkpoint_paths),
        "negative_tests": {
            "missing": missing_result,
            "corrupted": corrupted_result,
            "incompatible": incompatible_result,
        },
        "scientific_training_performed": False,
        "gpu_used": False,
    }
    output = RUN_ROOT / "WORKSPACE_ARTIFACT_VERIFICATION_REPORT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
