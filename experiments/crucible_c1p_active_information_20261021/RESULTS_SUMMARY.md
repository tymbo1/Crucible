# C1p active information acquisition

Date: 2026-10-21
Execution: CPU-only; hand-coded symbolic probe-selection test.

## Aggregate results

| Policy | Accuracy | Coverage | Conditional accuracy | Mean cost | Harmful/all | Harmful/accepted | Resolution rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `passive_abstention` | 0.0000 | 0.0000 | 0.0000 | 0.150 | 0.0000 | 0.0000 | 0.0000 |
| `fixed_probe` | 0.0717 | 0.1460 | 0.4915 | 0.230 | 0.0742 | 0.5085 | 0.0000 |
| `targeted_probe` | 0.4999 | 0.5742 | 0.8707 | 0.515 | 0.0742 | 0.1293 | 1.0000 |
| `no_information_control` | 0.4316 | 0.8576 | 0.5032 | 0.150 | 0.4261 | 0.4968 | 0.5064 |

Harmful/all is harmful transfer divided by all tasks. Harmful/accepted is the conditional rate among accepted transfers.
Resolution rate is correct resolution divided by tasks with a truthful informative probe available.

## Results by condition

| Condition | Policy | Accuracy | Coverage | Harmful/all | Resolution rate | Probe attempts | Mean cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ambiguous_distractor` | `fixed_probe` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5141 | 0.200 |
| `ambiguous_distractor` | `no_information_control` | 0.5145 | 1.0000 | 0.4855 | 0.5145 | 0 | 0.150 |
| `ambiguous_distractor` | `passive_abstention` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `ambiguous_distractor` | `targeted_probe` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 5141 | 0.500 |
| `ambiguous_expensive` | `fixed_probe` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5123 | 0.200 |
| `ambiguous_expensive` | `no_information_control` | 0.4962 | 1.0000 | 0.5038 | 0.4962 | 0 | 0.150 |
| `ambiguous_expensive` | `passive_abstention` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `ambiguous_expensive` | `targeted_probe` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 5123 | 1.650 |
| `ambiguous_multi` | `fixed_probe` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5151 | 0.200 |
| `ambiguous_multi` | `no_information_control` | 0.5083 | 1.0000 | 0.4917 | 0.5083 | 0 | 0.150 |
| `ambiguous_multi` | `passive_abstention` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `ambiguous_multi` | `targeted_probe` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 5151 | 0.500 |
| `misleading_evidence` | `fixed_probe` | 0.4915 | 1.0000 | 0.5085 | 0.0000 | 5255 | 0.500 |
| `misleading_evidence` | `no_information_control` | 0.4915 | 1.0000 | 0.5085 | 0.0000 | 0 | 0.150 |
| `misleading_evidence` | `passive_abstention` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `misleading_evidence` | `targeted_probe` | 0.4915 | 1.0000 | 0.5085 | 0.0000 | 5255 | 0.500 |
| `no_information` | `fixed_probe` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5069 | 0.200 |
| `no_information` | `no_information_control` | 0.5058 | 1.0000 | 0.4942 | 0.0000 | 0 | 0.150 |
| `no_information` | `passive_abstention` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `no_information` | `targeted_probe` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `role_break` | `fixed_probe` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `role_break` | `no_information_control` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `role_break` | `passive_abstention` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `role_break` | `targeted_probe` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `unavailable_evidence` | `fixed_probe` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `unavailable_evidence` | `no_information_control` | 0.5032 | 1.0000 | 0.4968 | 0.0000 | 0 | 0.150 |
| `unavailable_evidence` | `passive_abstention` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `unavailable_evidence` | `targeted_probe` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |

## Results by phase

| Phase | Policy | Accuracy | Coverage | Harmful/all | Resolution rate | Mean cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `discovery` | `fixed_probe` | 0.0727 | 0.1433 | 0.0707 | 0.0000 | 0.229 |
| `discovery` | `no_information_control` | 0.4308 | 0.8611 | 0.4303 | 0.5046 | 0.150 |
| `discovery` | `passive_abstention` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.150 |
| `discovery` | `targeted_probe` | 0.4981 | 0.5688 | 0.0707 | 1.0000 | 0.512 |
| `heldout` | `fixed_probe` | 0.0714 | 0.1469 | 0.0754 | 0.0000 | 0.230 |
| `heldout` | `no_information_control` | 0.4318 | 0.8564 | 0.4246 | 0.5069 | 0.150 |
| `heldout` | `passive_abstention` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.150 |
| `heldout` | `targeted_probe` | 0.5006 | 0.5760 | 0.0754 | 1.0000 | 0.516 |

## Investigation diagnostics

| Policy | Probe attempts | Informative probes | No-information abstentions | Misleading harmful cases |
| --- | ---: | ---: | ---: | ---: |
| `passive_abstention` | 0 | 0 | 10204 | 0 |
| `fixed_probe` | 25739 | 5255 | 10204 | 2672 |
| `targeted_probe` | 20670 | 20670 | 10204 | 2672 |
| `no_information_control` | 0 | 0 | 0 | 2672 |

## Gates

- `oracle_and_manifest_reproducible`: `True`.
- `targeted_resolves_more_than_passive`: `True`.
- `targeted_harm_below_unverified_control`: `True`.
- `targeted_abstains_without_information`: `True`.
- `targeted_selects_more_informative_than_fixed`: `True`.
- `misleading_evidence_visible`: `True`.
- `hand_coded_not_learned`: `True`.

This is an explicit symbolic probe-selection result, not evidence that a neural model learned active reasoning.

## Harmful-transfer traces

Stored harmful-transfer traces: `20682`.

## Source hashes

- `experiments/crucible_c1n_relational_transfer_20261021/HASHES.json`: `a6c9ac1aa1d000403ee6aee02d59b29a1aa8a8299f233774ebcebc8725fbe8ac`
- `experiments/crucible_c1n_relational_transfer_20261021/RESULTS.json`: `fb2d82f4ebdacf2762ec120deceb5f6f879132e2ac57bef7795c1bfdf826c699`
- `experiments/crucible_c1o_invariant_discovery_20261021/HASHES.json`: `c3db2a87e525f92662eb69308c6d985388b86c38d82b75fc827e0e470b689be2`
- `experiments/crucible_c1o_invariant_discovery_20261021/RESULTS.json`: `3b3161586b59ae9057625f83c3e3ab3a80eba9e9b32ff01ef5cb9b372e8aa636`
- `experiments/crucible_c1p_active_information_20261021/PROTOCOL.md`: `e5e8132c41cda3578e9f988c1600eddbd5629fbb68401533ec0efeff791120d2`
- `tools/run_crucible_c1p_active_information_20261021.py`: `32e631c4109bc5fe55ea681a2d86c7b832a680ea221d24904766d9c9734efc34`
