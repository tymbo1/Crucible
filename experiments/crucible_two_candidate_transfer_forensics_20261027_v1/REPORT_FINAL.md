# Crucible Two-Candidate Transfer Forensics v1

Checkpoint-only 2x2 diagnostic crossing candidate-pair familiarity with input serialization. No training was performed.

## Archived reproduction

The original `T_COMPACT` training predictions and `V_PADDED` validation predictions reproduced exactly for both passing cyclic checkpoints.

| Seed | T_COMPACT | T_PADDED | V_COMPACT | V_PADDED |
|---:|---:|---:|---:|---:|
| 42 | 144/144 (1.000) | 144/144 (1.000) | 75/96 (0.781) | 70/96 (0.729) |
| 1337 | 144/144 (1.000) | 144/144 (1.000) | 80/96 (0.833) | 81/96 (0.844) |

## Paired counterfactual correctness

| Seed | T_COMPACT | T_PADDED | V_COMPACT | V_PADDED |
|---:|---:|---:|---:|---:|
| 42 | 48/48 | 48/48 | 11/32 | 9/32 |
| 1337 | 48/48 | 48/48 | 16/32 | 17/32 |

## Direct findings

- Familiar candidate sets remained at 144/144 row accuracy and 48/48 paired groups for both seeds after padding was substituted, so the serialization change alone did not explain the training fit.
- Recombined candidate sets remained impaired under compact serialization: seed 42 was 75/96 with 11/32 paired groups, and seed 1337 was 80/96 with 16/32 paired groups. This is close to the corresponding padded validation results.
- The source populations are not fully matched: training-source two-candidate rows use six `(a,b)` contexts, while validation-source rows use four different contexts. The matched-context probe evaluates both candidate regimes on each context stratum; its results are required before attributing the gap specifically to candidate-set recombination.
- Counterfactual target flips are 32/32 in each recombined validation condition, but predicted flips remain incomplete (seed 42: 11/32 compact and 9/32 padded; seed 1337: 16/32 compact and 17/32 padded).

The matched-context probe is the targeted follow-up for the remaining confound. It is exploratory and generated from source context populations, not an independent validation benchmark.

## Interpretation boundary

The derived conditions preserve labels, IDs and candidate metadata while changing only serialization. The four conditions use different record populations, so their differences are diagnostic evidence rather than independent validation samples. They do not establish a causal mechanism or justify variable-cardinality claims.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_TWO_CANDIDATE_TRANSFER_FORENSICS_PROTOCOL_20261027.md)
- [Novelty audit](../../docs/CRUCIBLE_TWO_CANDIDATE_TRANSFER_FORENSICS_NOVELTY_AUDIT_20261027.json)
- [Summary](SUMMARY.json)
- [Data audit](DATA_AUDIT.json)
- [Error breakdown](ERROR_BREAKDOWN.json)
- Per-seed prediction matrices in `PREDICTIONS_SEED42.csv` and `PREDICTIONS_SEED1337.csv`
