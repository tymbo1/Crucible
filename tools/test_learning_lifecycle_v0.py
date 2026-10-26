#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from crucible_learning_lifecycle_v0 import (
    DOMAIN,
    Observation,
    PersistentStore,
    ProceduralLearner,
    RawExampleRetriever,
    apply_rule,
    compatible_rules,
)


class LearningLifecycleTests(unittest.TestCase):
    def oracle(self, a, b):
        return apply_rule("ADD_MOD10", a, b)

    def demos(self):
        return [
            Observation.make(1, 2, 3, "demo", {"task": "t"}),
            Observation.make(4, 5, 9, "demo", {"task": "t"}),
        ]

    def test_candidate_filter_and_probe_learning(self):
        candidates = compatible_rules(self.demos())
        self.assertIn("ADD_MOD10", candidates)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store.json"
            store = PersistentStore(path)
            learner = ProceduralLearner(store, probe_budget=8)
            result = learner.learn("glyph", self.demos(), [(7, 8), (9, 1)], self.oracle, DOMAIN, {"test": True})
            self.assertEqual(result["status"], "validated")
            self.assertEqual(learner.execute("glyph", 6, 7), 3)
            store2 = PersistentStore(path)
            learner2 = ProceduralLearner(store2)
            learner2.reset_context()
            self.assertEqual(learner2.execute("glyph", 6, 7), 3)

    def test_ambiguous_without_probes_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PersistentStore(Path(directory) / "store.json")
            learner = ProceduralLearner(store, probe_budget=0)
            result = learner.learn("ambiguous", [Observation.make(1, 2, 3, "demo", {})], [], self.oracle, DOMAIN, {})
            self.assertEqual(result["status"], "uncertain")
            self.assertIsNone(learner.execute("ambiguous", 8, 9))

    def test_contradiction_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store.json"
            store = PersistentStore(path)
            learner = ProceduralLearner(store, probe_budget=8)
            result = learner.learn("glyph", self.demos(), [(7, 8)], self.oracle, DOMAIN, {})
            self.assertEqual(result["status"], "validated")
            self.assertTrue(learner.invalidate("glyph", 6, 7, 9, "contradictory_observation"))
            reloaded = PersistentStore(path)
            self.assertIsNone(reloaded.active("glyph"))
            versions = reloaded.procedures["glyph"]
            self.assertEqual(versions[0]["status"], "invalidated")
            self.assertEqual(len(versions[0]["revision_history"]), 1)

    def test_raw_examples_do_not_execute_novel_inputs(self):
        retriever = RawExampleRetriever(self.demos())
        self.assertEqual(retriever.execute(1, 2), 3)
        self.assertIsNone(retriever.execute(8, 9))


if __name__ == "__main__":
    unittest.main()
