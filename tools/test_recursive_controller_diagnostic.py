#!/usr/bin/env python3
"""CPU checks for the read-only C1 diagnostic analysis."""

from __future__ import annotations

import unittest
from pathlib import Path

from analyze_recursive_controller_c1 import DEFAULT_INPUT, build_diagnostic


class RecursiveControllerDiagnosticTests(unittest.TestCase):
    def test_sealed_trace_analysis_is_reproducible_and_explanatory(self) -> None:
        report = build_diagnostic(Path(DEFAULT_INPUT))
        self.assertEqual(report["source"]["trace_sha256"], report["source"]["sealed_hash_index_entry"])
        self.assertEqual(report["aggregates"]["verification_only"]["accuracy"], 1.0)
        self.assertEqual(report["aggregates"]["adaptive"]["by_category"]["shortcut_probe"]["unresolved"], 51)
        self.assertEqual(report["aggregates"]["adaptive"]["verification_by_category"].get("shortcut_probe", 0), 0)


if __name__ == "__main__":
    unittest.main()
