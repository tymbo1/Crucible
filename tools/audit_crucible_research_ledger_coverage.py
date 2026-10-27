#!/usr/bin/env python3
"""Find experiment reports that are not linked from the Crucible ledger."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "CRUCIBLE_RESEARCH_LEDGER.json"


def main() -> int:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    indexed = {artifact for record in ledger.get("records", []) for artifact in record.get("artifacts", [])}
    report_patterns = ("REPORT_FINAL.md", "REPORT.md")
    reports = sorted({
        path.relative_to(ROOT).as_posix()
        for pattern in report_patterns
        for path in (ROOT / "experiments").glob(f"**/{pattern}")
    })
    indexed_reports = [path for path in reports if path in indexed]
    unindexed_reports = [path for path in reports if path not in indexed]
    result = {
        "report_patterns": list(report_patterns),
        "total_reports": len(reports),
        "indexed_reports": len(indexed_reports),
        "unindexed_reports": len(unindexed_reports),
        "unindexed": unindexed_reports,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
