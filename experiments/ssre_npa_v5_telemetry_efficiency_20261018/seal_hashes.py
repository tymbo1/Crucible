#!/usr/bin/env python3
"""Seal SHA-256 hashes for the v5 telemetry-efficiency package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifact_hashes.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    files = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUT or "__pycache__" in path.parts:
            continue
        files[str(path.relative_to(ROOT))] = sha256(path)
    OUT.write_text(json.dumps({
        "schema_version": "ssre-npa-v5-artifact-hashes",
        "algorithm": "sha256",
        "self_hash_excluded": True,
        "files": files,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"file_count": len(files), "output": str(OUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
