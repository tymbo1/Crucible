#!/usr/bin/env python3
"""Fail-closed preflight for the no-training CNSS latent-geometry audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "experiments" / "crucible_track_a_cnss_semantic_interface_compatibility_v1_rerun_contractfix_20261101"
OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_latent_geometry_causal_audit_v1_20261101"
EXPECTED_WORKSPACE_DIGEST = "a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c"
WORKSPACE_ARTIFACT = RUN_ROOT / "workspace_artifacts" / EXPECTED_WORKSPACE_DIGEST

sys.path.insert(0, str(ROOT / "tools"))
from verify_crucible_cnss_workspace_artifact_20261101 import load_bundle


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    adapter_checkpoints = sorted((RUN_ROOT / "checkpoints").rglob("checkpoint_*.pt"))
    workspace_candidates = sorted(path for path in (WORKSPACE_ARTIFACT / "state.pt",) if path.is_file())
    artifact_error = None
    artifact_verified = False
    if workspace_candidates:
        try:
            load_bundle(WORKSPACE_ARTIFACT, EXPECTED_WORKSPACE_DIGEST)
            artifact_verified = True
        except Exception as exc:
            artifact_error = f"{type(exc).__name__}: {exc}"
    ready = len(adapter_checkpoints) == 210 and artifact_verified
    report = {
        "schema_version": "crucible-cnss-latent-geometry-audit-preflight-v1-20261101",
        "classification": "READY_ARTIFACT_VERIFIED" if ready else "BLOCKED_MISSING_OR_INVALID_FROZEN_WORKSPACE_ARTIFACT",
        "adapter_checkpoint_count": len(adapter_checkpoints),
        "workspace_candidates": [str(path.relative_to(ROOT)) for path in workspace_candidates],
        "workspace_artifact": str(WORKSPACE_ARTIFACT.relative_to(ROOT)),
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "workspace_artifact_verified": artifact_verified,
        "workspace_artifact_error": artifact_error,
        "training_started": False,
        "parameter_updates": False,
        "workspace_reconstruction": False,
        "gpu_used": False,
        "reason": "Verified content-addressed workspace artifact is present." if ready else "A complete, verifiable workspace artifact is required for no-training geometry analysis.",
    }
    (OUTPUT / "AUDIT_PREFLIGHT_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# CNSS latent geometry causal audit preflight",
        "",
        f"Classification: **{report['classification']}**",
        "",
        f"Adapter checkpoints found: `{report['adapter_checkpoint_count']}`",
        f"Workspace artifact candidates: `{len(workspace_candidates)}`",
        "",
        report["reason"],
        "",
        f"Workspace artifact verified: `{artifact_verified}`",
        "",
        "No training, parameter update, workspace reconstruction or GPU work was performed.",
    ]
    (OUTPUT / "AUDIT_PREFLIGHT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"classification": report["classification"], "adapter_checkpoint_count": len(adapter_checkpoints), "workspace_candidates": len(workspace_candidates)}, sort_keys=True))
    return 0 if report["classification"] == "READY_ARTIFACT_VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

