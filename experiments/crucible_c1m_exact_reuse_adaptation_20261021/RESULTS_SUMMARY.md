# C1m exact reuse versus adaptation

Date: 2026-10-21
Execution: CPU-only; no GPU, cloud or Crucible neural training.

## Aggregate paired results

| Policy | Accuracy | Mean task cost | Mean cost incl. issuance | Harmful transfer | Valid cert uses | Cert-authorized adaptations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `broad_certificate_control` | 0.8737 | 1.100 | 1.101 | 0.0104 | 6921 | 3636 |
| `deterministic_reference` | 0.8761 | 1.145 | 1.145 | 0.0000 | 0 | 0 |
| `exact_certificate_only` | 0.8761 | 1.153 | 1.154 | 0.0000 | 3285 | 0 |
| `learned_constrained_reference` | 0.8761 | 1.066 | 1.066 | 0.0000 | 0 | 0 |
| `stateless_reference` | 0.8761 | 1.210 | 1.210 | 0.0000 | 0 | 0 |

## Scope-category results

| Scope | Policy | Accuracy | Mean cost | Harmful transfer | Independent adaptation verifications |
| --- | --- | ---: | ---: | ---: | ---: |
| `exact_core_reuse` | `broad_certificate_control` | 1.0000 | 0.219 | 0.0000 | 0 |
| `exact_core_reuse` | `deterministic_reference` | 1.0000 | 0.320 | 0.0000 | 0 |
| `exact_core_reuse` | `exact_certificate_only` | 1.0000 | 0.216 | 0.0000 | 0 |
| `exact_core_reuse` | `learned_constrained_reference` | 1.0000 | 0.240 | 0.0000 | 0 |
| `exact_core_reuse` | `stateless_reference` | 1.0000 | 1.300 | 0.0000 | 0 |
| `adaptation_transfer` | `broad_certificate_control` | 0.9819 | 0.192 | 0.0198 | 0 |
| `adaptation_transfer` | `deterministic_reference` | 1.0000 | 0.771 | 0.0000 | 3984 |
| `adaptation_transfer` | `exact_certificate_only` | 1.0000 | 0.771 | 0.0000 | 3984 |
| `adaptation_transfer` | `learned_constrained_reference` | 1.0000 | 0.700 | 0.0000 | 3984 |
| `adaptation_transfer` | `stateless_reference` | 1.0000 | 1.366 | 0.0000 | 3984 |
| `changed_or_untrusted` | `broad_certificate_control` | 0.8336 | 1.408 | 0.0000 | 0 |
| `changed_or_untrusted` | `deterministic_reference` | 0.8336 | 1.348 | 0.0000 | 0 |
| `changed_or_untrusted` | `exact_certificate_only` | 0.8336 | 1.376 | 0.0000 | 0 |
| `changed_or_untrusted` | `learned_constrained_reference` | 0.8336 | 1.268 | 0.0000 | 0 |
| `changed_or_untrusted` | `stateless_reference` | 0.8336 | 1.168 | 0.0000 | 0 |

## Broad-control failure attribution

- Broad-control harmful events: `72`.
- By family: `{'tree': 72}`.
- By scope: `{'adaptation_transfer': 72}`.

## Gates

- `frozen_input_audit`: `True`.
- `frozen_reference_exact`: `True`.
- `test_separation`: `True`.
- `exact_safety`: `True`.
- `no_certificate_authorized_adaptations`: `True`.
- `broad_failure_signature_reproduced`: `True`.

C1m does not establish neural reasoning or CMS2 training value. Failed gates remain valid results.

## Source hashes

- `experiments/crucible_c1j_cpu_stress_20261021/MODELS.json`: `27797d8094a33a7aa8b022e81058b514cd1fe67eae1d1e853e1255eab76c2ea4`
- `experiments/crucible_c1j_cpu_stress_20261021/RESULTS.json`: `d5be55d0d723b40026147f73f6c47d50568b0a0acb6684ba41c4121eb8752d2d`
- `experiments/crucible_c1j_cpu_stress_20261021/SPLIT_MANIFEST.json`: `bac5739b28d7b2d85d08c41f6dca7d829b255aa023dce679009f901a20b6e641`
- `experiments/crucible_c1k_constrained_procedure_20261021/RESULTS.json`: `e8d2cbdf24fbe7b083a7eefd6bfbb5e6c0afe282b51b0c090005634a79febd1f`
- `experiments/crucible_c1k_repeat_constrained_procedure_20261021/REPEAT_COMPARISON.json`: `6b42d392038a706dab7b3d497a64267a32c99f936b6b2c365425dcba5fdb89d4`
- `experiments/crucible_c1l_certificate_amortization_20261021/CERTIFICATE_CATALOGUE.json`: `d9cb1548a54f491072427e85729dd5174367a5c343165fb1ec20986c38061977`
- `experiments/crucible_c1l_certificate_amortization_20261021/RESULTS.json`: `a6a39f3d6e3c08cfff696d55c8bb62aea1c2a04d3acfc351cbf3f0af154ea044`
- `experiments/crucible_c1m_exact_reuse_adaptation_20261021/PROTOCOL.md`: `1e6783121ab57d9982ac37cfd95b0c596cff7c7fe051ac53f41d459cb4be91d8`
- `tools/run_crucible_c1j_cpu_stress_20261021.py`: `c816f03c3dcea376c43743f70f947e4c33860f6a9faea50f81171f7165e3366f`
- `tools/run_crucible_c1k_constrained_procedure_20261021.py`: `1cbe59a4783127ee75d8d37568cbc38f3dcf376388411d20307059cd6bc494e2`
- `tools/run_crucible_c1l_certificate_amortization_20261021.py`: `dcb39d7c43fc05ee9dea0b569bde5e0036aded693c789544dedbd90290bef9dd`
- `tools/run_crucible_c1m_exact_reuse_adaptation_20261021.py`: `baefee9050b6240667f28605c4c8c275085ef7683c1cd128bf19d74683177f83`
