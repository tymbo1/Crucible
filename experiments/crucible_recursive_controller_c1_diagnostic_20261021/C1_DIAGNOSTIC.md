# C1 diagnostic analysis

Date: 2026-10-21
Scope: read-only analysis of the sealed C0/C1 traces; no existing artifact was changed.

## Conclusion

Verification-only is the current fixture winner because exact verification is cheap and always available. Adaptive policy advantage is not established; the result primarily diagnoses a missing condition, not the recursive equations.

The existing C1 policies had identical budget ceilings and access to the same action types, but they did not incur equal realized costs. Therefore the result is not a matched-cost efficiency comparison.

## Policy comparison

| Policy | Accuracy | Mean cost | Verify calls | Unresolved | Valid commits | Invalid memory attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adaptive` | 0.8008 | 6.646 | 157 | 51 | 103 | 54 |
| `confidence_stop` | 0.4023 | 1.773 | 0 | 153 | 0 | 0 |
| `fixed_depth_2` | 0.7891 | 4.500 | 0 | 54 | 0 | 0 |
| `one_pass` | 0.4023 | 1.000 | 0 | 153 | 0 | 0 |
| `verification_only` | 1.0000 | 5.000 | 256 | 0 | 256 | 0 |

## Findings

### perfect_verifier_available

The fixture supplies an exact, always-available independent verifier at declared cost; this strongly favors verification-first behavior.

Evidence: `{"verification_only_calls": 256, "verification_only_records": 256, "verification_signals": {"verified_correct": 103, "verified_incorrect": 153}}`

### adaptive_skips_verification_on_shortcut_probe

The current low-confidence adaptive branch uses critique/counterexample/revision but never escalates to verification, so it misses every shortcut-probe case.

Evidence: `{"accuracy": 0.0, "records": 51, "unresolved": 51, "verify_calls": 0}`

### adaptive_spends_more_and_resolves_less

Under this fixture, adaptive is neither cost-dominant nor accuracy-dominant; it has higher mean cost and lower accuracy.

Evidence: `{"adaptive_accuracy": 0.80078125, "adaptive_mean_cost": 6.646484375, "verification_only_accuracy": 1.0, "verification_only_mean_cost": 5.0}`

### confidence_is_not_calibrated

The supplied confidence values are diagnostic synthetic signals, not a calibrated posterior. They cannot justify entropy or information-gain optimization.

Evidence: `{"brier": 0.319644140625, "ece": 0.45}`

### budgets_are_matched_as_ceilings_not_as_realized_cost

Policies receive the same maximum budget and action/tool types, but actual cost is not equalized. Efficiency claims therefore require a separate matched-cost analysis.

Evidence: `{"adaptive_mean_cost": 6.646484375, "shared_budget": {"actions": 8, "cost": 12.0, "counterexample_calls": 2, "depth": 8, "evidence_calls": 2, "memory_writes": 2}, "verification_only_mean_cost": 5.0}`

### memory_gate_exposes_contradiction_risk

Adaptive evidence paths create contradiction-linked proposals that are correctly quarantined rather than committed.

Evidence: `{"adaptive_invalid_memory_attempts": 54, "adaptive_valid_memory_commits": 103}`

## What C1 did and did not test

- It tested a perfect, immediately available verifier, not noisy, delayed, incomplete or unavailable verification.
- It tested deterministic synthetic response categories; it did not test a learned candidate generator.
- It tested contract safety and policy traces, not future competence or model cognition.
- The exact verifier is an allowed action, not a hidden privilege, but its reliability/cost regime creates an easy verification-first environment.
- The adaptive miss is localized: 51 shortcut-probe cases received no verification and remained unresolved.

## Recommendation

Retain the sealed C1 result. Do not retune the existing policy against it. Review the separately versioned C1b protocol before executing any new replay.

Sealed trace SHA-256: `b325c142f734523d088e1ab8533f45cc17fcd7fbce10d31b40921a48a8d36e98`
