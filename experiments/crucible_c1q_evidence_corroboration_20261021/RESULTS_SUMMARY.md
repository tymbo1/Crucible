# C1q evidence reliability and corroboration

Date: 2026-10-21
Execution: CPU-only; hand-coded symbolic evidence-handling test.

## Aggregate results

| Policy | Accuracy | Coverage | Conditional accuracy | Mean cost | Harmful/all | Harmful/accepted | Trustworthy resolution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_targeted` | 0.3733 | 0.7500 | 0.4978 | 0.300 | 0.3767 | 0.5022 | 0.5963 |
| `naive_corroboration` | 0.2497 | 0.4991 | 0.5003 | 0.558 | 0.2494 | 0.4997 | 0.3988 |
| `independence_aware` | 0.1241 | 0.2480 | 0.5003 | 0.721 | 0.1239 | 0.4997 | 0.1982 |
| `reliability_aware` | 0.2477 | 0.3717 | 0.6665 | 0.847 | 0.1239 | 0.3335 | 0.3957 |

Harmful/all is harmful transfer divided by all tasks. Harmful/accepted is the conditional rate among accepted transfers.
Trustworthy resolution is correct resolution divided by tasks with trustworthy evidence available.

## Results by condition

| Condition | Policy | Accuracy | Coverage | Harmful/all | Trustworthy resolution | Contradictions | Mean cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `correlated_misinformation` | `independence_aware` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4516 | 1.350 |
| `correlated_misinformation` | `naive_corroboration` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.600 |
| `correlated_misinformation` | `reliability_aware` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4516 | 1.350 |
| `correlated_misinformation` | `single_targeted` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.350 |
| `correlated_truth` | `independence_aware` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4522 | 1.150 |
| `correlated_truth` | `naive_corroboration` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.600 |
| `correlated_truth` | `reliability_aware` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4522 | 1.150 |
| `correlated_truth` | `single_targeted` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.350 |
| `independent_contradiction` | `independence_aware` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4451 | 0.600 |
| `independent_contradiction` | `naive_corroboration` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4451 | 0.600 |
| `independent_contradiction` | `reliability_aware` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 2980 | 1.317 |
| `independent_contradiction` | `single_targeted` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.350 |
| `no_trustworthy_evidence` | `independence_aware` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.600 |
| `no_trustworthy_evidence` | `naive_corroboration` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.600 |
| `no_trustworthy_evidence` | `reliability_aware` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.600 |
| `no_trustworthy_evidence` | `single_targeted` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.350 |
| `role_break` | `independence_aware` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `role_break` | `naive_corroboration` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `role_break` | `reliability_aware` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `role_break` | `single_targeted` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `single_misleading` | `independence_aware` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4583 | 1.150 |
| `single_misleading` | `naive_corroboration` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4583 | 1.150 |
| `single_misleading` | `reliability_aware` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4583 | 1.150 |
| `single_misleading` | `single_targeted` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0 | 0.350 |
| `truthful_agreement` | `independence_aware` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.600 |
| `truthful_agreement` | `naive_corroboration` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.600 |
| `truthful_agreement` | `reliability_aware` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1510 | 0.904 |
| `truthful_agreement` | `single_targeted` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0 | 0.350 |
| `unavailable_sources` | `independence_aware` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `unavailable_sources` | `naive_corroboration` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `unavailable_sources` | `reliability_aware` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |
| `unavailable_sources` | `single_targeted` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.150 |

## Results by phase

| Phase | Policy | Accuracy | Coverage | Harmful/all | Trustworthy resolution | Mean cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `discovery` | `independence_aware` | 0.1280 | 0.2487 | 0.1207 | 0.2016 | 0.723 |
| `discovery` | `naive_corroboration` | 0.2461 | 0.4952 | 0.2491 | 0.3876 | 0.561 |
| `discovery` | `reliability_aware` | 0.2586 | 0.3792 | 0.1207 | 0.4072 | 0.857 |
| `discovery` | `single_targeted` | 0.3767 | 0.7557 | 0.3790 | 0.5932 | 0.301 |
| `heldout` | `independence_aware` | 0.1228 | 0.2478 | 0.1250 | 0.1970 | 0.720 |
| `heldout` | `naive_corroboration` | 0.2509 | 0.5004 | 0.2495 | 0.4026 | 0.556 |
| `heldout` | `reliability_aware` | 0.2441 | 0.3691 | 0.1250 | 0.3918 | 0.844 |
| `heldout` | `single_targeted` | 0.3722 | 0.7481 | 0.3759 | 0.5974 | 0.300 |

## Corroboration diagnostics

| Policy | Contradictions | Third-source attempts | No-trustworthy harmful | Unavailable abstentions | Consensus updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| `single_targeted` | 0 | 0 | 4462 | 4451 | 0 |
| `naive_corroboration` | 9034 | 0 | 4462 | 4451 | 0 |
| `independence_aware` | 18072 | 0 | 4462 | 4451 | 8929 |
| `reliability_aware` | 18111 | 4490 | 4462 | 4451 | 13380 |

## Gates

- `oracle_and_manifest_reproducible`: `True`.
- `independence_reduces_correlated_harm`: `True`.
- `reliability_resolves_third_source_contradictions`: `True`.
- `safe_policies_abstain_unavailable`: `True`.
- `not_always_abstain`: `True`.
- `no_trustworthy_failure_visible`: `True`.
- `hand_coded_not_learned`: `True`.

This is an explicit symbolic evidence-handling result, not evidence that a neural model learned source reliability.

## Harmful-transfer traces

Stored harmful-transfer traces: `31463`.

## Source hashes

- `experiments/crucible_c1n_relational_transfer_20261021/HASHES.json`: `a6c9ac1aa1d000403ee6aee02d59b29a1aa8a8299f233774ebcebc8725fbe8ac`
- `experiments/crucible_c1n_relational_transfer_20261021/RESULTS.json`: `fb2d82f4ebdacf2762ec120deceb5f6f879132e2ac57bef7795c1bfdf826c699`
- `experiments/crucible_c1o_invariant_discovery_20261021/HASHES.json`: `c3db2a87e525f92662eb69308c6d985388b86c38d82b75fc827e0e470b689be2`
- `experiments/crucible_c1o_invariant_discovery_20261021/RESULTS.json`: `3b3161586b59ae9057625f83c3e3ab3a80eba9e9b32ff01ef5cb9b372e8aa636`
- `experiments/crucible_c1p_active_information_20261021/HASHES.json`: `10a6b5108be7c40e3fe1b144974a95320625674eba6850639e9220fb79af84ba`
- `experiments/crucible_c1p_active_information_20261021/RESULTS.json`: `11cf3127890501355b6fb63e14d0a3c0b55a1110ba82f90d077ca90d0794158a`
- `experiments/crucible_c1q_evidence_corroboration_20261021/PROTOCOL.md`: `8f010a04f47efbd01ade54ce1e43b2440610238d877dbeef4c94764de1b69d03`
- `tools/run_crucible_c1q_evidence_corroboration_20261021.py`: `74a2ffd62c596bdbb15062d6bdf5bddc1402564a8972c544ca40aa0803202b24`
