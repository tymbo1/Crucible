#!/usr/bin/env python3
"""Safe parsing helpers for repeated Crucible checkpoint metric rows.

The sustained gate emits multiple rows for each seed.  Rows are therefore
identified by (seed, interval_step_end), not by seed alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

RowKey = Tuple[int, int]


def read_rows(path: Path) -> Dict[RowKey, dict]:
    """Read JSONL rows and reject only duplicate seed/checkpoint pairs."""
    rows: Dict[RowKey, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                key = (int(row["seed"]), int(row["interval_step_end"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Missing valid seed/interval_step_end in {path} line {line_number}"
                ) from exc
            if key in rows:
                raise ValueError(
                    f"Duplicate seed/checkpoint {key} in {path} line {line_number}"
                )
            rows[key] = row
    return rows


def final_rows(rows: Mapping[RowKey, Mapping]) -> Dict[int, Mapping]:
    """Return the highest-step row for each seed."""
    result: Dict[int, Mapping] = {}
    for (seed, step), row in rows.items():
        previous = result.get(seed)
        if previous is None or step > int(previous["interval_step_end"]):
            result[seed] = row
    return result


def matching_keys(
    reference: Mapping[RowKey, Mapping], candidate: Mapping[RowKey, Mapping]
) -> tuple[set[RowKey], set[RowKey], set[RowKey]]:
    """Return shared, reference-only and candidate-only row keys."""
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    return (
        reference_keys & candidate_keys,
        reference_keys - candidate_keys,
        candidate_keys - reference_keys,
    )
