#!/usr/bin/env python3
"""Search the repository-backed Crucible Research Ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "CRUCIBLE_RESEARCH_LEDGER.json"


def haystack(record: Dict[str, Any]) -> str:
    parts = [record.get("id", ""), record.get("title", ""), record.get("status", ""), record.get("hypothesis", ""), record.get("result", "")]
    for key in ("mechanisms", "interventions", "controls", "limits", "unresolved", "novelty_terms"):
        parts.extend(str(value) for value in record.get(key, []))
    return " ".join(parts).lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="free-text search across IDs, mechanisms, results and questions")
    parser.add_argument("--status", choices=("ESTABLISHED", "FAILED", "INCONCLUSIVE", "UNTESTED"))
    parser.add_argument("--mechanism")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    records: Iterable[Dict[str, Any]] = ledger["records"]
    if args.status:
        records = [record for record in records if record.get("status") == args.status]
    if args.mechanism:
        needle = args.mechanism.lower()
        records = [record for record in records if needle in {str(value).lower() for value in record.get("mechanisms", [])}]
    if args.query:
        needle = args.query.lower()
        records = [record for record in records if needle in haystack(record)]
    result = list(records)
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for record in result:
            print(f"{record['status']:13} {record['id']}: {record['title']}")
            print(f"  result: {record['result']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
