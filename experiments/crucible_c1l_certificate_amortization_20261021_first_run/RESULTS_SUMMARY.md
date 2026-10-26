# C1l verification amortization and certificate invalidation

Date: 2026-10-21
Execution: CPU-only; no GPU, cloud or Crucible neural training.

## Aggregate paired results

| Policy | Accuracy | Mean task cost | Mean cost incl. shared issuance | Harmful transfer | Valid certificate uses | Invalidations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `certificate_amortized` | 0.8737 | 1.100 | 1.102 | 0.0104 | 6921 | 23079 |
| `certificate_corrupted` | 0.8737 | 1.136 | 1.138 | 0.0104 | 4740 | 25260 |
| `deterministic_reference` | 0.8761 | 1.145 | 1.145 | 0.0000 | 0 | 0 |
| `learned_constrained_reference` | 0.8761 | 1.066 | 1.066 | 0.0000 | 0 | 0 |
| `stale_certificate_control` | 0.6107 | 0.066 | 0.068 | 0.3983 | 0 | 0 |
| `stateless_reference` | 0.8761 | 1.210 | 1.210 | 0.0000 | 0 | 0 |

## Certificate catalogue

- Independent seed candidates: `50000`.
- Catalogue entries: `16` / `16`.
- Catalogue complete: `True`.
- Test-task overlap: `0`.

## Cost decomposition

- Certificate lookup cost: `0.02`.
- Certificate validity cost: `0.04`.
- Issuance cost per certificate: `1.98`.
- Shared catalogue issuance cost: `63.36`.

## Invalidation reasons

- `contradictory_record`: `165`.
- `insufficient_evidence_quality`: `228`.
- `invalid_provenance`: `3369`.
- `no_matching_certificate`: `12153`.
- `probe_disagreement`: `7164`.

## Results by condition

| Condition | Policy | Accuracy | Mean cost | Harmful transfer |
| --- | --- | ---: | ---: | ---: |
| `applicable` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `applicable` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `applicable` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `applicable` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `applicable` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `applicable` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |
| `adaptation` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `adaptation` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `adaptation` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `adaptation` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `adaptation` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `adaptation` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |
| `invalidated` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `invalidated` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `invalidated` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `invalidated` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `invalidated` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `invalidated` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |
| `contradictory` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `contradictory` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `contradictory` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `contradictory` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `contradictory` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `contradictory` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |
| `misleading_similarity` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `misleading_similarity` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `misleading_similarity` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `misleading_similarity` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `misleading_similarity` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `misleading_similarity` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |
| `noisy_evidence` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `noisy_evidence` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `noisy_evidence` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `noisy_evidence` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `noisy_evidence` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `noisy_evidence` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |
| `provenance_corrupt` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `provenance_corrupt` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `provenance_corrupt` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `provenance_corrupt` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `provenance_corrupt` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `provenance_corrupt` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |
| `unresolvable` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `unresolvable` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `unresolvable` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `unresolvable` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `unresolvable` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `unresolvable` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |

## Results by held-out family

| Family | Policy | Accuracy | Mean cost | Harmful transfer |
| --- | --- | ---: | ---: | ---: |
| `interval` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `interval` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `interval` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `interval` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `interval` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `interval` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |
| `multiset` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `multiset` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `multiset` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `multiset` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `multiset` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `multiset` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |
| `tree` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `tree` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `tree` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `tree` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `tree` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `tree` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |
| `matrix` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `matrix` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `matrix` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `matrix` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `matrix` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `matrix` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |

## Results by memory size

| Memory size | Policy | Accuracy | Mean cost | Harmful transfer |
| ---: | --- | ---: | ---: | ---: |
| `10` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `10` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `10` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `10` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `10` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `10` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |
| `100` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `100` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `100` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `100` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `100` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `100` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |
| `1000` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `1000` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `1000` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `1000` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `1000` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `1000` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |
| `10000` | `certificate_amortized` | 0.0000 | 0.000 | 0.0000 |
| `10000` | `certificate_corrupted` | 0.0000 | 0.000 | 0.0000 |
| `10000` | `deterministic_reference` | 0.0000 | 0.000 | 0.0000 |
| `10000` | `learned_constrained_reference` | 0.0000 | 0.000 | 0.0000 |
| `10000` | `stale_certificate_control` | 0.0000 | 0.000 | 0.0000 |
| `10000` | `stateless_reference` | 0.0000 | 0.000 | 0.0000 |

## Gates

- `catalogue_completeness`: `True`.
- `test_separation`: `True`.
- `primary_safety`: `False`.
- `primary_efficiency_10_percent`: `False`.
- `invalidation_safety`: `False`.
- `corruption_detection`: `True`.
- `stale_control_exposed_failure`: `True`.
- `frozen_reference_exact`: `True`.

Failed gates remain valid results. These synthetic outcomes do not establish neural reasoning or CMS2 training value.

## Source hashes

- `experiments/crucible_c1j_cpu_stress_20261021/MODELS.json`: `27797d8094a33a7aa8b022e81058b514cd1fe67eae1d1e853e1255eab76c2ea4`
- `experiments/crucible_c1j_cpu_stress_20261021/RESULTS.json`: `d5be55d0d723b40026147f73f6c47d50568b0a0acb6684ba41c4121eb8752d2d`
- `experiments/crucible_c1j_cpu_stress_20261021/SPLIT_MANIFEST.json`: `bac5739b28d7b2d85d08c41f6dca7d829b255aa023dce679009f901a20b6e641`
- `experiments/crucible_c1k_constrained_procedure_20261021/RESULTS.json`: `e8d2cbdf24fbe7b083a7eefd6bfbb5e6c0afe282b51b0c090005634a79febd1f`
- `experiments/crucible_c1k_repeat_constrained_procedure_20261021/REPEAT_COMPARISON.json`: `6b42d392038a706dab7b3d497a64267a32c99f936b6b2c365425dcba5fdb89d4`
- `experiments/crucible_c1l_certificate_amortization_20261021/PROTOCOL.md`: `b65c9c688a8790fc0c6d7ae7ea3dc084a693a9167525835c5ae0d1580adbe80d`
- `tools/run_crucible_c1j_cpu_stress_20261021.py`: `c816f03c3dcea376c43743f70f947e4c33860f6a9faea50f81171f7165e3366f`
- `tools/run_crucible_c1k_constrained_procedure_20261021.py`: `1cbe59a4783127ee75d8d37568cbc38f3dcf376388411d20307059cd6bc494e2`
- `tools/run_crucible_c1l_certificate_amortization_20261021.py`: `c375d3f447aea9754f7087412b2d7447cb9ecd39eb8c918a60543ab0bb7e31e6`
