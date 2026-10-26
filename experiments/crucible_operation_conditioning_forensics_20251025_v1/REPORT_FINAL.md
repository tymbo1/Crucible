# Crucible Operation-Conditioning Mechanism Investigation — Final Report

CPU-only, read-only forensics on the frozen S3 joint-operation checkpoints. No new training, architecture change, GPU execution or sealed-test access.

## Phase 1 — Reproduction

S3 arm reproduction: PASS; token-identifier control reproduction: PASS.

| Interface | Seed | S3 train exact | S3 validation exact | Checkpoint hash |
|---|---:|---:|---:|---|
| canonical | 42 | 60.00% | 60.00% | match |
| canonical | 1337 | 61.25% | 55.00% | match |
| factorized | 42 | 77.50% | 60.00% | match |
| factorized | 1337 | 60.00% | 50.00% | match |

## Phase 2 — Representation evidence

The operation-identity probes are fitted on training rows and evaluated on disjoint validation rows. Probe decodability is evidence that information is recoverable, not evidence that the forward computation uses it causally.

| Arm | State/location with highest validation ridge accuracy | Ridge validation | Nearest-centroid validation |
|---|---|---:|---:|
| S3_JOINT_OPERATION_CONDITIONED_canonical_seed42 | embedding/operation_last | 100.00% | 100.00% |
| S3_JOINT_OPERATION_CONDITIONED_canonical_seed1337 | embedding/operation_last | 100.00% | 100.00% |
| S3_JOINT_OPERATION_CONDITIONED_factorized_seed42 | embedding/operation_last | 100.00% | 100.00% |
| S3_JOINT_OPERATION_CONDITIONED_factorized_seed1337 | embedding/operation_last | 100.00% | 100.00% |

| Arm | Mean output logit L2 change | Mean output logit cosine | Pairwise argmax changes |
|---|---:|---:|---:|
| S3_JOINT_OPERATION_CONDITIONED_canonical_seed42 | 0.4382 | 1.0000 | 0/8 |
| S3_JOINT_OPERATION_CONDITIONED_canonical_seed1337 | 0.6629 | 1.0000 | 0/8 |
| S3_JOINT_OPERATION_CONDITIONED_factorized_seed42 | 20.1585 | 0.9990 | 0/8 |
| S3_JOINT_OPERATION_CONDITIONED_factorized_seed1337 | 0.1957 | 1.0000 | 0/8 |

## Phase 3 — Gradient diagnostics

| Arm | First gradient norm | Second gradient norm | Cross-operation cosine |
|---|---:|---:|---:|
| S3_JOINT_OPERATION_CONDITIONED_canonical_seed42 | 0.3814 | 0.3887 | 0.8791 |
| S3_JOINT_OPERATION_CONDITIONED_canonical_seed1337 | 0.5785 | 0.5782 | 0.9968 |
| S3_JOINT_OPERATION_CONDITIONED_factorized_seed42 | 0.5031 | 0.8143 | 0.7256 |
| S3_JOINT_OPERATION_CONDITIONED_factorized_seed1337 | 0.6850 | 0.6883 | 0.9681 |

## Phase 4 — Activation patching

Patching used validation unequal pairs, layers 0, 2 and 3, operation-token/output-position locations, and same-operation/random controls. A patch changing a prediction or margin is reported as an intervention effect for that tested location; it is not treated as proof of a unique mechanism.

| Arm | Control | Rows | Prediction changes | Patched outputs equal donor value | Mean margin movement |
|---|---|---:|---:|---:|---:|
| S3_JOINT_OPERATION_CONDITIONED_canonical_seed1337 | matched_opposite | 96 | 0 | 48 | 0.0004 |
| S3_JOINT_OPERATION_CONDITIONED_canonical_seed1337 | same_operation_other_pair | 96 | 36 | 30 | 0.0024 |
| S3_JOINT_OPERATION_CONDITIONED_canonical_seed1337 | random_noise | 96 | 47 | 24 | 0.0793 |
| S3_JOINT_OPERATION_CONDITIONED_canonical_seed42 | matched_opposite | 96 | 0 | 48 | -0.0118 |
| S3_JOINT_OPERATION_CONDITIONED_canonical_seed42 | same_operation_other_pair | 96 | 40 | 30 | -0.0081 |
| S3_JOINT_OPERATION_CONDITIONED_canonical_seed42 | random_noise | 96 | 47 | 27 | -0.0298 |
| S3_JOINT_OPERATION_CONDITIONED_factorized_seed1337 | matched_opposite | 96 | 0 | 48 | -0.0008 |
| S3_JOINT_OPERATION_CONDITIONED_factorized_seed1337 | same_operation_other_pair | 96 | 40 | 29 | -0.0022 |
| S3_JOINT_OPERATION_CONDITIONED_factorized_seed1337 | random_noise | 96 | 46 | 25 | 0.0184 |
| S3_JOINT_OPERATION_CONDITIONED_factorized_seed42 | matched_opposite | 96 | 0 | 48 | -0.2183 |
| S3_JOINT_OPERATION_CONDITIONED_factorized_seed42 | same_operation_other_pair | 96 | 29 | 33 | -0.0297 |
| S3_JOINT_OPERATION_CONDITIONED_factorized_seed42 | random_noise | 96 | 47 | 26 | -0.2233 |

## Supported conclusions

- The S3 metrics and checkpoint hashes were reproduced without opening the sealed test set.
- The separate operation-token identification control establishes that the model can classify the requested operation under the same broad task interface.
- The S3 failure therefore cannot be reduced to a basic inability to recognize the operation string.
- Operation identity was 100% validation-decodable at the operation-token position in all four S3 arms, although that source-position result is partly lexical because the two operation strings contain different byte tokens. Answer-position decodability was only 50–60% at the final layer in three arms; one factorized arm reached 100%, so propagation failure is not universal.
- Changing only the operation identifier changed none of the 32 matched unequal validation argmax outputs. Matched opposite-operation patches changed 0/384 intervention predictions, while same-operation and random controls did change predictions.
- Endpoint operation-specific gradient cosine was positive in every arm (0.726–0.997), so the final checkpoint does not support a simple opposing-gradient explanation.

## Ranked hypothesis interpretation

1. **Most supported — H2: conditional routing/propagation.** Operation identity is represented locally but is not reliably available to the answer computation.
2. **Behaviorally supported — H5: operation-blind shortcut.** The output behaves as though it selects an operand without conditioning on the operation identifier.
3. **Unresolved — H4: role binding/conditional execution.** The model may recognize the instruction without binding it to the correct procedure.
4. **Not supported by the endpoint check — H3: optimization interference.** Gradients were aligned rather than opposed at the final checkpoints, although transient interference is not ruled out.
5. **Rejected as a sole explanation — H1: no operation representation.** The operation token is perfectly decodable at its source position.

These rankings are behavioral and intervention-bounded. They do not establish a unique neural mechanism. The machine-readable details are in [HYPOTHESIS_RANKING.json](HYPOTHESIS_RANKING.json).

## Attention limitation

The dense model routes attention through the current scaled-dot-product-attention API without exposing attention probability matrices. Native attention-pattern results are therefore unavailable in this run; no substitute was mislabeled as native attention telemetry.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_OPERATION_CONDITIONING_FORENSICS_PROTOCOL_20251025.md)
- [Machine-readable summary](SUMMARY.json)
- [Reproduced metrics](REPRODUCED_METRICS.json)
- [Predictions](PREDICTIONS.csv)
- [Hidden states](hidden_states/)
- [Operation probes](OPERATION_PROBES.json)
- [Hypothesis ranking](HYPOTHESIS_RANKING.json)
- [Gradient diagnostics](GRADIENT_DIAGNOSTICS.json)
- [Activation patch matrix](ACTIVATION_PATCHING.csv)
- [Training curves](CURVE_SUMMARY.json)
- [Finalization](FINALIZATION.json)
