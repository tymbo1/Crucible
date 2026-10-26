# Crucible Batch-Composition and Update-Ordering Control — Final Report

CPU-only, 500-update, compute-matched comparison. The sealed test set was not accessed.

## Experimental invariants

Schedule audit: **PASS**. Input audit: **PASS**.
Each arm used 12 records/update, 500 updates and 6,000 record presentations; each expanded training record was presented exactly 25 times.

## Training fit and gradient trace

| Arm | Interface | Seed | Train exact | Mean adjacent gradient cosine | Mean unique pair groups/batch |
|---|---|---:|---:|---:|---:|
| B0_DIVERSE_SHUFFLED | canonical | 42 | 74.58% | 0.1009 | 12.0 |
| B0_DIVERSE_SHUFFLED | canonical | 1337 | 65.42% | 0.1069 | 12.0 |
| B0_DIVERSE_SHUFFLED | factorized | 42 | 72.08% | 0.0832 | 12.0 |
| B0_DIVERSE_SHUFFLED | factorized | 1337 | 62.92% | 0.0913 | 12.0 |
| B1_DIVERSE_GROUPED | canonical | 42 | 80.83% | -0.1117 | 12.0 |
| B1_DIVERSE_GROUPED | canonical | 1337 | 85.42% | -0.1940 | 12.0 |
| B1_DIVERSE_GROUPED | factorized | 42 | 75.00% | -0.1761 | 12.0 |
| B1_DIVERSE_GROUPED | factorized | 1337 | 75.42% | -0.2172 | 12.0 |
| B2_SAME_PAIR_SHUFFLED | canonical | 42 | 65.83% | 0.0409 | 2.0 |
| B2_SAME_PAIR_SHUFFLED | canonical | 1337 | 57.08% | 0.0708 | 2.0 |
| B2_SAME_PAIR_SHUFFLED | factorized | 42 | 57.92% | 0.0477 | 2.0 |
| B2_SAME_PAIR_SHUFFLED | factorized | 1337 | 55.00% | 0.0645 | 2.0 |
| B3_SAME_PAIR_GROUPED | canonical | 42 | 57.92% | 0.4056 | 2.0 |
| B3_SAME_PAIR_GROUPED | canonical | 1337 | 51.67% | 0.3791 | 2.0 |
| B3_SAME_PAIR_GROUPED | factorized | 42 | 52.08% | 0.4148 | 2.0 |
| B3_SAME_PAIR_GROUPED | factorized | 1337 | 48.33% | 0.3906 | 2.0 |

## Validation paired conditional correctness

Values are output-token paired correctness; denominators are operand-pair groups, averaged across the two seeds and interfaces where shown.

| Arm | Fixed-near | Early | Middle | Unseen-between | Unseen-early-context |
|---|---:|---:|---:|---:|---:|
| B0_DIVERSE_SHUFFLED | 7/32 | 1/32 | 6/32 | 0/32 | 1/32 |
| B1_DIVERSE_GROUPED | 13/32 | 0/32 | 16/32 | 3/32 | 1/32 |
| B2_SAME_PAIR_SHUFFLED | 1/32 | 0/32 | 1/32 | 1/32 | 0/32 |
| B3_SAME_PAIR_GROUPED | 1/32 | 0/32 | 1/32 | 0/32 | 0/32 |

## Batch-composition and ordering contrasts

The new arms use the same batch multiset within B0/B1 and within B2/B3. Therefore B0 versus B1 and B2 versus B3 isolate batch ordering within their respective compositions, subject to stochastic optimizer dynamics. Diverse versus same-pair contrasts test batch composition with the scheduled order held in the corresponding grouped or shuffled family.

- **B0_DIVERSE_SHUFFLED versus B1_DIVERSE_GROUPED:**
  - `FIXED_NEAR_PADDED`: B0_DIVERSE_SHUFFLED 7/32; B1_DIVERSE_GROUPED 13/32.
  - `EARLY_PADDED`: B0_DIVERSE_SHUFFLED 1/32; B1_DIVERSE_GROUPED 0/32.
  - `MIDDLE_POSITION`: B0_DIVERSE_SHUFFLED 6/32; B1_DIVERSE_GROUPED 16/32.
  - `UNSEEN_BETWEEN_OPERANDS`: B0_DIVERSE_SHUFFLED 0/32; B1_DIVERSE_GROUPED 3/32.
  - `UNSEEN_EARLY_CONTEXT_FIRST`: B0_DIVERSE_SHUFFLED 1/32; B1_DIVERSE_GROUPED 1/32.
- **B2_SAME_PAIR_SHUFFLED versus B3_SAME_PAIR_GROUPED:**
  - `FIXED_NEAR_PADDED`: B2_SAME_PAIR_SHUFFLED 1/32; B3_SAME_PAIR_GROUPED 1/32.
  - `EARLY_PADDED`: B2_SAME_PAIR_SHUFFLED 0/32; B3_SAME_PAIR_GROUPED 0/32.
  - `MIDDLE_POSITION`: B2_SAME_PAIR_SHUFFLED 1/32; B3_SAME_PAIR_GROUPED 1/32.
  - `UNSEEN_BETWEEN_OPERANDS`: B2_SAME_PAIR_SHUFFLED 1/32; B3_SAME_PAIR_GROUPED 0/32.
  - `UNSEEN_EARLY_CONTEXT_FIRST`: B2_SAME_PAIR_SHUFFLED 0/32; B3_SAME_PAIR_GROUPED 0/32.

## Historical A0 reference

The prior A0 checkpoints were evaluated read-only and are not part of the new training arms.

| Interface | Seed | Fixed-near validation paired | Early validation paired | Middle validation paired |
|---|---:|---:|---:|---:|
| canonical | 42 | 2/8 | 2/8 | 2/8 |
| canonical | 1337 | 6/8 | 8/8 | 6/8 |
| factorized | 42 | 4/8 | 3/8 | 4/8 |
| factorized | 1337 | 7/8 | 5/8 | 5/8 |

## Cross-layout consistency

The full machine-readable cross-layout table is in `CROSS_LAYOUT.json`. It separates familiar operands in new layouts from unseen validation operands in trained layouts.

## Interpretation boundary

Gradient cosine is a diagnostic of update similarity, not a causal proof of gradient interference. High training fit with low validation transfer remains a memorization/transfer result. The experiment does not establish a position-independent representation or an architectural ceiling.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_BATCH_COMPOSITION_ORDER_PROTOCOL_20251025.md)
- [Summary](SUMMARY.json)
- [Predictions](PREDICTIONS.csv)
- [Training traces](TRAINING_TRACE.jsonl)
- [Cross-layout analysis](CROSS_LAYOUT.json)
- [Scientific interpretation](SCIENTIFIC_INTERPRETATION.md)
- [Schedule audit](SCHEDULE_AUDIT.json)
- [Finalization](FINALIZATION.json)
