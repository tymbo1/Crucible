# Crucible Authority Representation-versus-Use Diagnostic v1

Read-only hidden-state probe and activation-patching analysis of the verified A3 checkpoints.

## Scope

No weights were changed, no training was run, and no sealed test set was accessed. The clean counterfactual baseline and familiar v2 validation predictions were reproduced before analysis.

## Baseline reproduction

All four A3 checkpoint/interface/seed combinations matched the accepted pointer predictions on the clean unseen counterfactual export and on familiar validation layouts.

## Authority-cue probe

The complete machine-readable probe table is in `AUTHORITY_PROBE_RESULTS.json`. The probe was trained on frozen authority-training representations; probe accuracy is not evidence of causal use.

| Domain | Hidden location | Model rows | Accuracy range |
|---|---|---:|---:|
| AUTHORITY_NEAR | final_norm / authority token | 80/4 model evaluations | 100.0%–100.0% |
| AUTHORITY_EARLY | final_norm / authority token | 80/4 model evaluations | 100.0%–100.0% |
| AUTHORITY_MIDDLE | final_norm / authority token | 80/4 model evaluations | 100.0%–100.0% |
| unseen_reversed_authority | final_norm / authority token | 320/4 model evaluations | 100.0%–100.0% |

The operation-position probe is the more diagnostic comparison because the pointer head scores these positions:

| Domain | Location | Per-model accuracy range |
|---|---|---:|
| AUTHORITY_NEAR | final_norm / op_first | 7/20–12/20 |
| AUTHORITY_NEAR | final_norm / op_second | 20/20–20/20 |
| AUTHORITY_EARLY | final_norm / op_first | 20/20–20/20 |
| AUTHORITY_EARLY | final_norm / op_second | 20/20–20/20 |
| AUTHORITY_MIDDLE | final_norm / op_first | 10/20–10/20 |
| AUTHORITY_MIDDLE | final_norm / op_second | 20/20–20/20 |
| unseen_reversed_authority | final_norm / op_first | 40/80–40/80 |
| unseen_reversed_authority | final_norm / op_second | 40/80–40/80 |

## Activation patching

Opposite-authority patches replace the authority-position hidden vector with its paired counterfactual counterpart. In this clean layout the authority token is later than both operation candidates, so the causal-access audit is the primary interpretation; the patch is a bounded negative control. Self-patching is the same-input numerical control.

| Model | Site | Opposite pointer changes | Opposite patched correct | Self max logit delta | Self control |
|---|---|---:|---:|---:|---|
| canonical_seed42 | embedding | 0/80 | 40/80 | 0.000e+00 | PASS |
| canonical_seed42 | layer_0 | 0/80 | 40/80 | 0.000e+00 | PASS |
| canonical_seed42 | layer_1 | 0/80 | 40/80 | 0.000e+00 | PASS |
| canonical_seed42 | layer_2 | 0/80 | 40/80 | 0.000e+00 | PASS |
| canonical_seed42 | layer_3 | 0/80 | 40/80 | 0.000e+00 | PASS |
| canonical_seed42 | final_norm | 0/80 | 40/80 | 0.000e+00 | PASS |
| canonical_seed1337 | embedding | 0/80 | 40/80 | 0.000e+00 | PASS |
| canonical_seed1337 | layer_0 | 0/80 | 40/80 | 0.000e+00 | PASS |
| canonical_seed1337 | layer_1 | 0/80 | 40/80 | 0.000e+00 | PASS |
| canonical_seed1337 | layer_2 | 0/80 | 40/80 | 0.000e+00 | PASS |
| canonical_seed1337 | layer_3 | 0/80 | 40/80 | 0.000e+00 | PASS |
| canonical_seed1337 | final_norm | 0/80 | 40/80 | 0.000e+00 | PASS |
| factorized_seed42 | embedding | 0/80 | 40/80 | 7.153e-07 | PASS |
| factorized_seed42 | layer_0 | 0/80 | 40/80 | 7.153e-07 | PASS |
| factorized_seed42 | layer_1 | 0/80 | 40/80 | 7.153e-07 | PASS |
| factorized_seed42 | layer_2 | 0/80 | 40/80 | 7.153e-07 | PASS |
| factorized_seed42 | layer_3 | 0/80 | 40/80 | 7.153e-07 | PASS |
| factorized_seed42 | final_norm | 0/80 | 40/80 | 7.153e-07 | PASS |
| factorized_seed1337 | embedding | 0/80 | 40/80 | 1.431e-06 | PASS |
| factorized_seed1337 | layer_0 | 0/80 | 40/80 | 1.431e-06 | PASS |
| factorized_seed1337 | layer_1 | 0/80 | 40/80 | 1.431e-06 | PASS |
| factorized_seed1337 | layer_2 | 0/80 | 40/80 | 1.431e-06 | PASS |
| factorized_seed1337 | layer_3 | 0/80 | 40/80 | 1.431e-06 | PASS |
| factorized_seed1337 | final_norm | 0/80 | 40/80 | 1.431e-06 | PASS |

## Causal-access audit

| Model | Pairs | Max pointer-logit delta at operation positions | Max hidden delta at operation positions |
|---|---:|---:|---:|
| canonical_seed42 | 40 | 0.000e+00 | 0.000e+00 |
| canonical_seed1337 | 40 | 0.000e+00 | 0.000e+00 |
| factorized_seed42 | 40 | 0.000e+00 | 0.000e+00 |
| factorized_seed1337 | 40 | 0.000e+00 | 0.000e+00 |

## Interpretation boundary

A decodable authority cue would show representation, not use. A patch effect would show sensitivity under the intervention, not a unique mechanism. Familiar/unseen differences remain conditional on these layouts and cannot be generalized to real-world instruction hierarchy.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_AUTHORITY_REPRESENTATION_USE_PROTOCOL_20261026.md)
- [Probe results](AUTHORITY_PROBE_RESULTS.json)
- [Probe predictions](AUTHORITY_PROBE_PREDICTIONS.csv)
- [Activation patch results](ACTIVATION_PATCH_RESULTS.json)
- [Activation patch rows](ACTIVATION_PATCH_ROWS.csv)
- [Self-patch controls](SELF_PATCH_ROWS.csv)
- [Causal-access audit](CAUSAL_ACCESS.json)
- [Fidelity and invariants](FIDELITY.json)
- [Analysis](ANALYSIS.json)
