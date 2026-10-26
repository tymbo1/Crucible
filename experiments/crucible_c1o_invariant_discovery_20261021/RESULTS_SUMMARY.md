# C1o invariant discovery and counterexample-guided refinement

Date: 2026-10-21
Execution: CPU-only; hand-coded symbolic mechanism test.

## Aggregate results

| Policy | Accuracy | Coverage | Conditional accuracy | Mean cost | Harmful/all | Harmful/accepted | Valid recall | Invalid rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `frozen_relational` | 0.4990 | 0.8744 | 0.5707 | 0.150 | 0.3754 | 0.4293 | 1.0000 | 0.2506 |
| `counterexample_refinement` | 0.4990 | 0.7484 | 0.6667 | 0.369 | 0.2494 | 0.3333 | 1.0000 | 0.5021 |
| `active_evidence` | 0.4990 | 0.6256 | 0.7977 | 0.369 | 0.1265 | 0.2023 | 1.0000 | 0.7474 |

Harmful/all is harmful transfer divided by all tasks. Harmful/accepted is the conditional rate among accepted transfers.

## Results by condition

| Condition | Policy | Accuracy | Coverage | Harmful/all | Valid recall | Invalid rejection | Mean cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `depth_extension` | `active_evidence` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.500 |
| `depth_extension` | `counterexample_refinement` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.500 |
| `depth_extension` | `frozen_relational` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.150 |
| `exact_role` | `active_evidence` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.500 |
| `exact_role` | `counterexample_refinement` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.500 |
| `exact_role` | `frozen_relational` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.150 |
| `marker_guard_invalid` | `active_evidence` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.150 |
| `marker_guard_invalid` | `counterexample_refinement` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.150 |
| `marker_guard_invalid` | `frozen_relational` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.150 |
| `marker_guard_valid` | `active_evidence` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.500 |
| `marker_guard_valid` | `counterexample_refinement` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.500 |
| `marker_guard_valid` | `frozen_relational` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.150 |
| `misleading_evidence` | `active_evidence` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.500 |
| `misleading_evidence` | `counterexample_refinement` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.500 |
| `misleading_evidence` | `frozen_relational` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.150 |
| `role_break` | `active_evidence` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.150 |
| `role_break` | `counterexample_refinement` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.150 |
| `role_break` | `frozen_relational` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.150 |
| `role_permutation` | `active_evidence` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.500 |
| `role_permutation` | `counterexample_refinement` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.500 |
| `role_permutation` | `frozen_relational` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.150 |
| `unobservable_ambiguity` | `active_evidence` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.150 |
| `unobservable_ambiguity` | `counterexample_refinement` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.150 |
| `unobservable_ambiguity` | `frozen_relational` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.150 |

## Results by phase

| Phase | Policy | Accuracy | Coverage | Harmful/all | Valid recall | Mean cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `discovery` | `active_evidence` | 0.4870 | 0.6159 | 0.1289 | 1.0000 | 0.366 |
| `discovery` | `counterexample_refinement` | 0.4870 | 0.7376 | 0.2506 | 1.0000 | 0.366 |
| `discovery` | `frozen_relational` | 0.4870 | 0.8720 | 0.3850 | 1.0000 | 0.150 |
| `heldout` | `active_evidence` | 0.5030 | 0.6288 | 0.1257 | 1.0000 | 0.370 |
| `heldout` | `counterexample_refinement` | 0.5030 | 0.7521 | 0.2490 | 1.0000 | 0.370 |
| `heldout` | `frozen_relational` | 0.5030 | 0.8753 | 0.3722 | 1.0000 | 0.150 |

## Refinement diagnostics

| Policy | Counterexamples | Refinements | Probe attempts | Unresolvable abstentions | Final guards |
| --- | ---: | ---: | ---: | ---: | --- |
| `frozen_relational` | 0 | 0 | 0 | 0 | `{}` |
| `counterexample_refinement` | 3 | 3 | 22523 | 0 | `{"anchor_marker": 0}` |
| `active_evidence` | 3 | 3 | 22523 | 4424 | `{"anchor_marker": 0}` |

## Gates

- `oracle_and_manifest_reproducible`: `True`.
- `refinement_discoverable_recall_or_tradeoff`: `True`.
- `refinement_reduces_discoverable_harm`: `True`.
- `active_reduces_unobservable_harm`: `True`.
- `active_not_trivial_always_abstain`: `True`.
- `misleading_evidence_visible`: `True`.
- `hand_coded_not_learned`: `True`.

This is an explicit symbolic invariant-refinement result, not evidence that a neural model learned relational reasoning.

## Harmful-transfer traces

Stored harmful-transfer traces: `27049`.

## Source hashes

- `experiments/crucible_c1m_exact_reuse_adaptation_20261021/HASHES.json`: `cf002946d15b4fb437f946b03c6ec217e3a5e2bb172d2008997554639f4aa7b8`
- `experiments/crucible_c1m_exact_reuse_adaptation_20261021/RESULTS.json`: `d2946e8c5859b45f1a2342d0f4999a7f49e0cf1d9f2b76563ead21e77343f162`
- `experiments/crucible_c1n_relational_transfer_20261021/HASHES.json`: `a6c9ac1aa1d000403ee6aee02d59b29a1aa8a8299f233774ebcebc8725fbe8ac`
- `experiments/crucible_c1n_relational_transfer_20261021/RESULTS.json`: `fb2d82f4ebdacf2762ec120deceb5f6f879132e2ac57bef7795c1bfdf826c699`
- `experiments/crucible_c1o_invariant_discovery_20261021/PROTOCOL.md`: `c27efd1f40421be5b3818601057bf549e987da021489ee343030b8deae275dba`
- `tools/run_crucible_c1o_invariant_discovery_20261021.py`: `06393b680fd71015197b4fe42d06a7b3cdd0ca12992ebdec7404d8d15ec76a0c`
