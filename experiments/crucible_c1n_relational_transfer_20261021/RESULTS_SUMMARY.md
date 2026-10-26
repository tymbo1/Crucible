# C1n relational transfer validity

Date: 2026-10-21
Execution: CPU-only; hand-coded symbolic mechanism test.

## Aggregate results

| Policy | Accuracy | Coverage | Conditional accuracy | Mean cost | Harmful transfer | Valid-transfer recall | Invalid rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `relational_matching` | 0.5167 | 0.8729 | 0.5920 | 0.150 | 0.4080 | 1.0000 | 0.2529 |
| `relational_verified` | 0.3901 | 0.5166 | 0.7552 | 0.411 | 0.2448 | 0.7454 | 0.7097 |
| `surface_association` | 0.0000 | 0.6364 | 0.0000 | 0.050 | 1.0000 | 0.0000 | 0.2293 |

## Results by condition

| Condition | Policy | Accuracy | Coverage | Harmful transfer | Valid recall | Invalid rejection |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `isomorphic_relabel` | `relational_matching` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| `isomorphic_relabel` | `relational_verified` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| `isomorphic_relabel` | `surface_association` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `role_permutation` | `relational_matching` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| `role_permutation` | `relational_verified` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| `role_permutation` | `surface_association` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `depth_extension` | `relational_matching` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| `depth_extension` | `relational_verified` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| `depth_extension` | `surface_association` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `role_break` | `relational_matching` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `role_break` | `relational_verified` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `role_break` | `surface_association` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hidden_invariant_change` | `relational_matching` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hidden_invariant_change` | `relational_verified` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `hidden_invariant_change` | `surface_association` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `surface_decoy` | `relational_matching` | 0.1593 | 1.0000 | 0.8407 | 0.0000 | 0.0000 |
| `surface_decoy` | `relational_verified` | 0.1593 | 0.1593 | 0.0000 | 0.0000 | 0.8407 |
| `surface_decoy` | `surface_association` | 0.0000 | 0.0587 | 1.0000 | 0.0000 | 0.9413 |
| `no_probe` | `relational_matching` | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| `no_probe` | `relational_verified` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `no_probe` | `surface_association` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `misleading_probe` | `relational_matching` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `misleading_probe` | `relational_verified` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| `misleading_probe` | `surface_association` | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## Results by template

| Template | Policy | Accuracy | Coverage | Harmful transfer | Mean cost |
| --- | --- | ---: | ---: | ---: | ---: |
| `balanced` | `relational_matching` | 0.5343 | 0.8715 | 0.3870 | 0.150 |
| `balanced` | `relational_verified` | 0.4114 | 0.5334 | 0.2285 | 0.412 |
| `balanced` | `surface_association` | 0.0000 | 0.6353 | 1.0000 | 0.050 |
| `asymmetric` | `relational_matching` | 0.4969 | 0.8771 | 0.4334 | 0.150 |
| `asymmetric` | `relational_verified` | 0.3659 | 0.4979 | 0.2650 | 0.411 |
| `asymmetric` | `surface_association` | 0.0000 | 0.6358 | 1.0000 | 0.050 |
| `deep` | `relational_matching` | 0.5238 | 0.8698 | 0.3978 | 0.150 |
| `deep` | `relational_verified` | 0.4015 | 0.5238 | 0.2335 | 0.412 |
| `deep` | `surface_association` | 0.0000 | 0.6411 | 1.0000 | 0.050 |
| `wide` | `relational_matching` | 0.5124 | 0.8730 | 0.4131 | 0.150 |
| `wide` | `relational_verified` | 0.3823 | 0.5116 | 0.2528 | 0.410 |
| `wide` | `surface_association` | 0.0000 | 0.6333 | 1.0000 | 0.050 |

## Gates

- `oracle_and_manifest_reproducible`: `True`.
- `relational_valid_recall_above_surface`: `True`.
- `relational_decoy_harm_below_surface`: `True`.
- `verification_reduces_relational_harm`: `True`.
- `hand_coded_not_learned`: `True`.

This is an explicit symbolic mechanism result, not evidence that a neural model learned relational reasoning.

## Source hashes

- `experiments/crucible_c1l_certificate_amortization_20261021/CERTIFICATE_CATALOGUE.json`: `d9cb1548a54f491072427e85729dd5174367a5c343165fb1ec20986c38061977`
- `experiments/crucible_c1m_exact_reuse_adaptation_20261021/HASHES.json`: `cf002946d15b4fb437f946b03c6ec217e3a5e2bb172d2008997554639f4aa7b8`
- `experiments/crucible_c1m_exact_reuse_adaptation_20261021/RESULTS.json`: `d2946e8c5859b45f1a2342d0f4999a7f49e0cf1d9f2b76563ead21e77343f162`
- `experiments/crucible_c1n_relational_transfer_20261021/PROTOCOL.md`: `b66a277b58b19542a89395580671f699c11c55f44712b5db8b80096b13000940`
- `tools/run_crucible_c1j_cpu_stress_20261021.py`: `c816f03c3dcea376c43743f70f947e4c33860f6a9faea50f81171f7165e3366f`
- `tools/run_crucible_c1n_relational_transfer_20261021.py`: `165e0926b867f7928d5255a4289400cec268f9599ef955456dfa32d19a153472`
