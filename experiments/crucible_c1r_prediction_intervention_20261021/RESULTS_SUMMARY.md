# C1r prediction and intervention testing

Date: 2026-10-21
Execution: CPU-only; hand-coded symbolic consequence-testing experiment.

## Aggregate results

| Policy | Accuracy | Coverage | Conditional accuracy | Mean cost | Harmful/all | Harmful/accepted | Truthful resolution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `corroboration_only` | 0.0977 | 0.7999 | 0.1222 | 0.554 | 0.7022 | 0.8778 | 0.1984 |
| `prediction_testing` | 0.3479 | 0.3973 | 0.8758 | 0.799 | 0.0493 | 0.1242 | 0.6062 |
| `intervention_testing` | 0.3283 | 0.3743 | 0.8770 | 0.904 | 0.0460 | 0.1230 | 0.5699 |

Harmful/all is harmful transfer divided by all tasks. Harmful/accepted is the conditional rate among accepted transfers.
Truthful resolution is correct resolution divided by tasks with a truthful discriminating test available.

## Results by condition

| Condition | Policy | Accuracy | Coverage | Harmful/all | Truthful resolution | Test attempts | Mean cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `correlated_misinformation` | `corroboration_only` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.600 |
| `correlated_misinformation` | `intervention_testing` | 0.9413 | 0.9413 | 0.0000 | 0.9413 | 3609 | 1.100 |
| `correlated_misinformation` | `prediction_testing` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 3609 | 0.950 |
| `inconclusive_test` | `corroboration_only` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.600 |
| `inconclusive_test` | `intervention_testing` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3630 | 1.100 |
| `inconclusive_test` | `prediction_testing` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3630 | 0.950 |
| `intervention_only` | `corroboration_only` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.600 |
| `intervention_only` | `intervention_testing` | 0.9367 | 0.9367 | 0.0000 | 1.0000 | 3650 | 1.100 |
| `intervention_only` | `prediction_testing` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3650 | 0.950 |
| `misleading_intervention` | `corroboration_only` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.600 |
| `misleading_intervention` | `intervention_testing` | 0.4822 | 0.9477 | 0.4656 | 0.0000 | 3559 | 1.100 |
| `misleading_intervention` | `prediction_testing` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3559 | 0.950 |
| `misleading_prediction` | `corroboration_only` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.600 |
| `misleading_prediction` | `intervention_testing` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3556 | 1.100 |
| `misleading_prediction` | `prediction_testing` | 0.5006 | 1.0000 | 0.4994 | 0.0000 | 3556 | 0.950 |
| `prediction_only` | `corroboration_only` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.600 |
| `prediction_only` | `intervention_testing` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3619 | 1.100 |
| `prediction_only` | `prediction_testing` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 3619 | 0.950 |
| `role_break` | `corroboration_only` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `role_break` | `intervention_testing` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `role_break` | `prediction_testing` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `source_conflict` | `corroboration_only` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.600 |
| `source_conflict` | `intervention_testing` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.600 |
| `source_conflict` | `prediction_testing` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.600 |
| `truthful_consensus` | `corroboration_only` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.600 |
| `truthful_consensus` | `intervention_testing` | 0.9341 | 0.9341 | 0.0000 | 0.9341 | 3518 | 1.100 |
| `truthful_consensus` | `prediction_testing` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 3518 | 0.950 |
| `unavailable_tests` | `corroboration_only` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.600 |
| `unavailable_tests` | `intervention_testing` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.600 |
| `unavailable_tests` | `prediction_testing` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.600 |

## Results by phase

| Phase | Policy | Accuracy | Coverage | Harmful/all | Truthful resolution | Mean cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `discovery` | `corroboration_only` | 0.0970 | 0.7996 | 0.7026 | 0.1966 | 0.554 |
| `discovery` | `intervention_testing` | 0.3350 | 0.3818 | 0.0468 | 0.5718 | 0.904 |
| `discovery` | `prediction_testing` | 0.3449 | 0.3956 | 0.0507 | 0.6047 | 0.799 |
| `heldout` | `corroboration_only` | 0.0980 | 0.8000 | 0.7020 | 0.1991 | 0.555 |
| `heldout` | `intervention_testing` | 0.3260 | 0.3718 | 0.0458 | 0.5693 | 0.904 |
| `heldout` | `prediction_testing` | 0.3490 | 0.3979 | 0.0489 | 0.6068 | 0.799 |

## Testing diagnostics

| Policy | Test attempts | Test matches | Inconclusive abstentions | Memory contamination |
| --- | ---: | ---: | ---: | ---: |
| `corroboration_only` | 0 | 0 | 0 | 25278 |
| `prediction_testing` | 25141 | 12526 | 10839 | 1776 |
| `intervention_testing` | 25141 | 11818 | 11666 | 1657 |

## Gates

- `oracle_and_manifest_reproducible`: `True`.
- `prediction_reduces_correlated_harm`: `True`.
- `intervention_reduces_correlated_harm`: `True`.
- `prediction_resolves_prediction_only`: `True`.
- `intervention_resolves_intervention_only`: `True`.
- `testing_abstains_inconclusive_and_unavailable`: `True`.
- `misleading_test_failure_visible`: `True`.
- `hand_coded_not_learned`: `True`.

This is an explicit symbolic consequence-testing result, not evidence that a neural model learned causal reasoning.

## Harmful-transfer traces

Stored harmful-transfer traces: `28711`.

## Source hashes

- `experiments/crucible_c1p_active_information_20261021/HASHES.json`: `10a6b5108be7c40e3fe1b144974a95320625674eba6850639e9220fb79af84ba`
- `experiments/crucible_c1p_active_information_20261021/RESULTS.json`: `11cf3127890501355b6fb63e14d0a3c0b55a1110ba82f90d077ca90d0794158a`
- `experiments/crucible_c1q_evidence_corroboration_20261021/HASHES.json`: `0abc9f3cf711876f0cebbe9a89e8559f1b57f30b35a564b065a01271010a04e7`
- `experiments/crucible_c1q_evidence_corroboration_20261021/RESULTS.json`: `ab26a8873ee26e3af6599aad38ad8f75555bc718fbfbc3e9bff480ba2420fd23`
- `experiments/crucible_c1r_prediction_intervention_20261021/PROTOCOL.md`: `7deb28fd094e41629d039cade0661dbc20dddb3fd446caf86581cba1686bf377`
- `tools/run_crucible_c1r_prediction_intervention_20261021.py`: `b23d7ec015c9ff69dd95e3b1b7904da1809f4882ab4dffad8959c9a62ce7506f`
