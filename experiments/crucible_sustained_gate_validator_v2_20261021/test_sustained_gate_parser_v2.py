#!/usr/bin/env python3
"""Regression tests for the sustained-gate row-key correction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sustained_gate_parser_v2 import final_rows, matching_keys, read_rows


ROOT = Path("/home/timbushnell/runpod_equivalence_3125_results_20261021_v1")
EXPECTED_STEPS = {100, 250, 500, 750, 1000, 1500, 2000, 2500, 3125}


class SustainedGateParserTests(unittest.TestCase):
    def test_captured_logs_allow_repeated_seed_rows(self) -> None:
        original = read_rows(ROOT / "original_metrics.jsonl")
        grouped = read_rows(ROOT / "grouped_rank_preserving_metrics.jsonl")
        self.assertEqual(len(original), 18)
        self.assertEqual(len(grouped), 18)
        self.assertEqual({seed for seed, _ in original}, {42, 1337})
        self.assertEqual({step for _, step in original}, EXPECTED_STEPS)
        self.assertEqual(matching_keys(original, grouped)[1:], (set(), set()))
        self.assertEqual(
            {seed: int(row["interval_step_end"]) for seed, row in final_rows(original).items()},
            {42: 3125, 1337: 3125},
        )

    def test_duplicate_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.jsonl"
            row = {"seed": 42, "interval_step_end": 100}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate seed/checkpoint"):
                read_rows(path)

    def test_same_seed_at_different_checkpoints_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "repeated_seed.jsonl"
            rows = [
                {"seed": 42, "interval_step_end": 100},
                {"seed": 42, "interval_step_end": 3125},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            parsed = read_rows(path)
            self.assertEqual(set(parsed), {(42, 100), (42, 3125)})


if __name__ == "__main__":
    unittest.main()
