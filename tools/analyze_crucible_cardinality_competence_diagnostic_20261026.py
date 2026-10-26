#!/usr/bin/env python3
"""Read-only error breakdown for cardinality competence diagnostic v1."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "crucible_cardinality_competence_diagnostic_20261026_v1"
OUTPUT = EXPERIMENT / "ERROR_BREAKDOWN.json"


def grouped(rows: List[Dict[str, str]], field: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        key: {
            "correct": sum(row["pointer_correct"] == "True" for row in values),
            "rows": len(values),
            "accuracy": sum(row["pointer_correct"] == "True" for row in values) / len(values),
        }
        for key, values in sorted(groups.items())
    }


def main() -> int:
    result: Dict[str, Any] = {
        "experiment": "crucible_cardinality_competence_diagnostic_20261026_v1",
        "read_only": True,
        "cells": {},
    }
    for path in sorted(EXPERIMENT.glob("D*_*/PREDICTIONS.csv")):
        cell = path.parent.name
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        pair_groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row["counterfactual_pair_id"]:
                pair_groups[row["counterfactual_pair_id"]].append(row)
        complete = [values for values in pair_groups.values() if len(values) == 2]
        result["cells"][cell] = {
            "by_candidate_count": grouped(rows, "candidate_count"),
            "by_target_slot": grouped(rows, "target_slot"),
            "present_vs_absent": {
                "present": grouped([row for row in rows if row["target_slot"] != "8"], "target_slot"),
                "absent": grouped([row for row in rows if row["target_slot"] == "8"], "target_slot"),
            },
            "paired": {
                "groups": len(complete),
                "correct_both": sum(all(row["pointer_correct"] == "True" for row in values) for values in complete),
                "pointer_flips": sum(values[0]["predicted_slot"] != values[1]["predicted_slot"] for values in complete),
            },
        }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "cells": len(result["cells"]), "read_only": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
