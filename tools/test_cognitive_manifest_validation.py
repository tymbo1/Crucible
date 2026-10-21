#!/usr/bin/env python3
"""Tests for independent cognitive-manifest validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_cognitive_manifests import main as build_main
from cognitive_task_suite import generate_records
from validate_cognitive_manifests import validate


class CognitiveManifestValidationTests(unittest.TestCase):
    def test_independent_oracles_and_holdouts(self) -> None:
        records = {
            split: generate_records(20261021, split, 16)
            for split in ("train", "validation", "test")
        }
        report = validate(records)
        self.assertEqual(report["oracle_checks"], 16 * 4 * 3)
        self.assertEqual(report["input_leakage"], [])
        self.assertEqual(report["relational_path_lengths"], {"train": [2], "validation": [3], "test": [4]})
        self.assertEqual(report["logic_rule_depths"], {"train": [2], "validation": [3], "test": [4]})

    def test_checked_in_manifest_is_not_needed_for_generation(self) -> None:
        # Keep this test CPU-only and verify the validator works on a small
        # independently generated collection without touching repository files.
        records = {split: generate_records(7, split, 2) for split in ("train", "validation", "test")}
        report = validate(records)
        self.assertEqual(report["oracle_checks"], 24)


if __name__ == "__main__":
    unittest.main()

