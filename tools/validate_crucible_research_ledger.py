#!/usr/bin/env python3
"""Validate the Crucible Research Ledger schema and artifact links."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "CRUCIBLE_RESEARCH_LEDGER.json"
STATUSES = {"ESTABLISHED", "FAILED", "INCONCLUSIVE", "UNTESTED"}
REQUIRED = {"id", "status", "title", "hypothesis", "mechanisms", "result", "limits", "unresolved", "artifacts", "novelty_terms"}


def main() -> int:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    records = ledger.get("records", [])
    errors = []
    ids = set()
    for index, record in enumerate(records):
        missing = sorted(REQUIRED - set(record))
        if missing:
            errors.append(f"record {index} missing {missing}")
        if record.get("id") in ids:
            errors.append(f"duplicate id {record.get('id')}")
        ids.add(record.get("id"))
        if record.get("status") not in STATUSES:
            errors.append(f"{record.get('id')}: invalid status {record.get('status')}")
        for artifact in record.get("artifacts", []):
            if not (ROOT / artifact).exists():
                errors.append(f"{record.get('id')}: missing artifact {artifact}")
    print(json.dumps({"records": len(records), "errors": errors, "valid": not errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
