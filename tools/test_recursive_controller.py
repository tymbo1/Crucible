#!/usr/bin/env python3
"""Independent CPU tests for the C0/C1 controller contract."""

from __future__ import annotations

import unittest

from recursive_controller import (
    BeliefState,
    ControllerEngine,
    ContractError,
    MemoryProposal,
    MemoryStore,
    ReasoningBudget,
    Scenario,
    initial_state,
    run_c0_checks,
    run_policy,
)


class RecursiveControllerTests(unittest.TestCase):
    def test_c0_suite_passes(self) -> None:
        result = run_c0_checks()
        self.assertTrue(result["passed"], result)

    def test_budget_cannot_increase(self) -> None:
        budget = ReasoningBudget(actions=1)
        with self.assertRaises(ContractError):
            budget.child(actions=2)

    def test_uncalibrated_entropy_is_not_available(self) -> None:
        belief = BeliefState.from_mapping({"a": 0.5, "b": 0.5}, calibrated=False, calibration_source="none")
        self.assertIsNone(belief.entropy())

    def test_cycle_rejection_is_explicit(self) -> None:
        scenario = Scenario("cycle", "math", "", "1", "2", "1", 0.5, "clean_correct")
        engine = ControllerEngine()
        state = initial_state(scenario)
        state, first = engine.apply(state, scenario, "answer", policy="one_pass", sequence=0)
        self.assertTrue(first.accepted)
        state, first = engine.apply(state, scenario, "critique", policy="one_pass", sequence=1)
        self.assertTrue(first.accepted)
        _, second = engine.apply(state, scenario, "critique", policy="one_pass", sequence=2)
        self.assertFalse(second.accepted)
        self.assertEqual(second.reason, "cycle_detected")

    def test_invalid_memory_write_is_quarantined(self) -> None:
        store = MemoryStore()
        proposal = MemoryProposal("p", "c", "claim", "", (), (), True)
        with self.assertRaises(ContractError):
            store.commit(proposal)
        self.assertEqual(store.records, [])

    def test_verification_policy_corrects_wrong_candidate(self) -> None:
        scenario = Scenario("wrong", "math", "", "gold", "wrong", "wrong", 0.2, "detectable_wrong")
        run = run_policy(scenario, "verification_only")
        self.assertTrue(run.final_correct)
        self.assertEqual(run.false_correction, False)
        self.assertEqual(run.valid_memory_commits, 1)

    def test_contradiction_does_not_commit_unverified_memory(self) -> None:
        scenario = Scenario("contradiction", "math", "", "gold", "wrong", "gold", 0.9, "contradiction")
        run = run_policy(scenario, "fixed_depth_2")
        self.assertEqual(run.valid_memory_commits, 0)
        self.assertGreaterEqual(run.unresolved, 0)


if __name__ == "__main__":
    unittest.main()
