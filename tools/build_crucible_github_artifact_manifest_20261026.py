#!/usr/bin/env python3
"""Build the manifest for Crucible artifacts intentionally kept outside GitHub."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUTPUT = ROOT / "docs" / "CRUCIBLE_GITHUB_ARTIFACT_MANIFEST_20261026.json"
LATEST = {
    "experiments/crucible_d2_component_ablation_20261026_v1",
    "experiments/crucible_compositional_authority_transfer_20261026_v1",
    "experiments/crucible_shared_candidate_scoring_cardinality_20261026_v1",
    "experiments/crucible_cardinality_competence_diagnostic_20261026_v1",
}
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def curated(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    if relative.startswith("docs/") or relative.startswith("tools/"):
        return True
    if not relative.startswith("experiments/"):
        return False
    if any(relative.startswith(prefix + "/") for prefix in LATEST):
        return path.suffix not in {".pt", ".pth", ".safetensors", ".bin", ".tar", ".gz"} and "TRAINING_TRACE" not in path.name
    name = path.name.lower()
    result_file = (
        name in {"report_final.md", "readme.md"}
        or name.startswith("summary")
        or name.startswith("result")
        or name.startswith("finalization")
        or name.startswith("prediction")
        or "manifest" in name
        or "hash" in name
        or "audit" in name
        or name.endswith(".sha256")
        or name.endswith("results.md")
    )
    return result_file and path.stat().st_size <= MAX_GITHUB_FILE_BYTES


def main() -> int:
    included = []
    excluded = []
    for path in sorted(p for p in EXPERIMENTS.rglob("*") if p.is_file()):
        relative = path.relative_to(ROOT).as_posix()
        entry = {"path": relative, "bytes": path.stat().st_size}
        if curated(path):
            included.append(entry)
        else:
            if path.suffix in {".pt", ".pth", ".safetensors", ".bin", ".tar", ".gz"}:
                reason = "checkpoint_or_binary_artifact"
            elif path.stat().st_size > MAX_GITHUB_FILE_BYTES:
                reason = "github_file_limit"
            else:
                reason = "raw_trace_or_intermediate_artifact"
            entry["reason"] = reason
            entry["sha256"] = sha256(path)
            excluded.append(entry)
    manifest = {
        "schema_version": "crucible_github_artifact_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "tymbo1/Crucible",
        "policy": {
            "curated_commit": "docs, tools, reports, summaries/results, predictions, manifests, audits, hashes and complete metadata for the latest accepted experiments",
            "excluded": "checkpoints/binaries, raw traces/intermediate files and files over 100 MiB",
            "large_artifact_storage": "Keep excluded files in the local/archive store; use sha256 values below to verify recovery.",
        },
        "included_summary": {"files": len(included), "bytes": sum(item["bytes"] for item in included)},
        "excluded_summary": {"files": len(excluded), "bytes": sum(item["bytes"] for item in excluded)},
        "excluded_files": excluded,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "included": manifest["included_summary"], "excluded": manifest["excluded_summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
