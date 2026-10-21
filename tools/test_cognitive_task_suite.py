#!/usr/bin/env python3
"""CPU-only integrity tests for the deterministic cognitive task suite."""

from __future__ import annotations

import json
import unittest

from cognitive_task_suite import FAMILIES, generate_records


class CognitiveTaskSuiteTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        first = generate_records(20261021, "test", 8)
        second = generate_records(20261021, "test", 8)
        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, sort_keys=True, separators=(",", ":")),
            json.dumps(second, sort_keys=True, separators=(",", ":")),
        )

    def test_all_families_and_unique_ids(self) -> None:
        records = generate_records(20261021, "train", 16)
        self.assertEqual({record["family"] for record in records}, set(FAMILIES))
        self.assertEqual(len(records), len({record["record_id"] for record in records}))
        self.assertTrue(all(record["target"] for record in records))

    def test_holdouts_are_declared(self) -> None:
        validation = generate_records(20261021, "validation", 8)
        test = generate_records(20261021, "test", 8)
        for record in validation + test:
            if record["family"] in {"math", "relational_binding", "logic", "error_correction"}:
                self.assertTrue(
                    any(value is True for value in record["metadata"].values()),
                    record,
                )

    def test_exact_oracles_have_expected_shapes(self) -> None:
        records = generate_records(20261021, "test", 12)
        for record in records:
            kind = record["oracle"]["kind"]
            if kind == "integer_exact":
                self.assertEqual(record["target"], str(record["oracle"]["value"]))
            elif kind == "candidate_verification":
                self.assertEqual(
                    record["target"],
                    f"{record['oracle']['status']}|{record['oracle']['expected']}",
                )
            elif kind in {"directed_transitive_relation", "forward_chain_three_way"}:
                self.assertEqual(record["target"], record["oracle"]["answer"])
            else:
                self.fail(f"unknown oracle kind: {kind}")

    def test_v2_correction_labels_are_balanced_within_holdout_compositions(self) -> None:
        for split in ("validation", "test"):
            records = [
                record
                for record in generate_records(20261021, split, 32, variant="v2")
                if record["family"] == "error_correction"
            ]
            by_composition = {}
            for record in records:
                composition = tuple(record["metadata"]["source_composition"])
                by_composition.setdefault(composition, []).append(
                    record["metadata"]["candidate_is_correct"]
                )
            self.assertTrue(by_composition)
            for statuses in by_composition.values():
                self.assertEqual(statuses.count(True), statuses.count(False))


if __name__ == "__main__":
    unittest.main()

