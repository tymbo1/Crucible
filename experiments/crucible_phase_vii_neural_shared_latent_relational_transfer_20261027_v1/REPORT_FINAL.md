# Crucible Phase VII: learned shared-latent relational transfer

N1 and N2 were trained on the frozen topology-control records. N0 is the read-only 2,000-step dense reference from the audited duration extension.

The executed matrix contains eight trained arms: two learned variants × two
graph conditions × two seeds. The preregistered cost paragraph says “four new
N1/N2 arms”; that wording is an accounting inconsistency in the protocol, not
a change to the data, optimizer, or decision rules. The complete executed
matrix is reported below.

| Variant | Condition | Seed | Train | Gate | Validation | Paired |
|---|---|---:|---:|---|---:|---:|
| N1_SHARED_LATENT | P12_PRISM | 42 | 140/144 (0.972) | PASS | 34/96 (0.354) | 17/48 |
| N1_SHARED_LATENT | P12_PRISM | 1337 | 142/144 (0.986) | PASS | 34/96 (0.354) | 17/48 |
| N1_SHARED_LATENT | R12_DIAMETER5 | 42 | 144/144 (1.000) | PASS | 28/96 (0.292) | 14/48 |
| N1_SHARED_LATENT | R12_DIAMETER5 | 1337 | 142/144 (0.986) | PASS | 22/96 (0.229) | 11/48 |
| N2_SHARED_LATENT_CYCLE | P12_PRISM | 42 | 144/144 (1.000) | PASS | 20/96 (0.208) | 10/48 |
| N2_SHARED_LATENT_CYCLE | P12_PRISM | 1337 | 144/144 (1.000) | PASS | 30/96 (0.312) | 15/48 |
| N2_SHARED_LATENT_CYCLE | R12_DIAMETER5 | 42 | 142/144 (0.986) | PASS | 34/96 (0.354) | 17/48 |
| N2_SHARED_LATENT_CYCLE | R12_DIAMETER5 | 1337 | 144/144 (1.000) | PASS | 22/96 (0.229) | 11/48 |

## Result against the preregistered criteria

All eight trained arms passed the 95% training-competence gate. However, no
N1 or N2 arm exceeded the matched N0 paired-transfer baseline:

| Comparison | N0 paired | N1 paired | N2 paired |
|---|---:|---:|---:|
| P12_PRISM, seed 42 | 29/48 (0.604) | 17/48 (0.354) | 10/48 (0.208) |
| P12_PRISM, seed 1337 | 23/48 (0.479) | 17/48 (0.354) | 15/48 (0.312) |
| R12_DIAMETER5, seed 42 | 18/48 (0.375) | 14/48 (0.292) | 17/48 (0.354) |
| R12_DIAMETER5, seed 1337 | 16/48 (0.333) | 11/48 (0.229) | 11/48 (0.229) |
| Pooled | 86/192 (0.448) | 59/192 (0.307) | 53/192 (0.276) |

Therefore the preregistered constructive-transfer improvement criterion was
not met, and the cycle-consistency benefit criterion was not met. This is a
negative result for these tested Phase VII configurations, not evidence that
shared latent relational models or cycle consistency are impossible in
general.

## Interpretation boundary

Held-out transfer is interpreted only for arms that pass the training gate. N1/N2 use a structured parsed representation, so any advantage over N0 is a combined representation-and-architecture result; it is not evidence that the original dense scorer discovered the same latent algorithm. Identifier-renaming and entirely new-identifier evaluations are separate follow-up axes and are not silently substituted for the fixed known-identifier pair transfer.

The primary result is limited to known-identifier held-out pairs in the two
committed graph families. Identifier-renaming and entirely unseen-identifier
transfer were not executed in this phase. The independent post-run audit
verified all eight traces, CSV metrics, checkpoint hashes, source references,
and the absence of sealed-test access.

See `SUMMARY.json`, per-arm predictions and traces for exact counts.
