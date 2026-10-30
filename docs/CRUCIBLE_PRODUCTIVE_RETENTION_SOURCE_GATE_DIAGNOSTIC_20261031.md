# Source-gate CPU diagnostic

Date: 2026-10-31  
Classification: **CPU diagnostic only; no training or target evaluation**

This diagnostic audited the source curriculum and replayed the two saved
update-1,000 checkpoints on CPU. It did not retrain either model and did not
load target data or sealed answers.

## Task identifiability

The registered source grammar contains 429 distinct truth tables for four
Boolean inputs, using `and`, `or`, `xor` and `not` at depth at most two.

| Split | Rows | Query-identifiable | Ambiguous |
|---|---:|---:|---:|
| Seed 42 source train | 32 | 10 | 22 |
| Seed 42 source validation | 16 | 7 | 9 |
| Seed 1337 source train | 32 | 13 | 19 |
| Seed 1337 source validation | 16 | 8 | 8 |
| Combined validation | 32 | 15 | **17** |

All validation queries use an input vector not present among that task's eight
demonstrations. The ambiguous cases have multiple grammar-consistent
functions that agree with every demonstration but disagree on the query.
This does not make probabilistic inference impossible, but it prevents every
validation answer from being uniquely justified by the supplied evidence.

Both validation splits have a majority-label constant baseline of 10/16
(62.5%).

## Checkpoint fit diagnostic

The final source-gate checkpoints were replayed on CPU in FP32. This replay is
diagnostic only; the canonical gate values remain the BF16 GPU evaluations
recorded during the authorized run.

| Seed | CPU source-train | CPU source-validation | Canonical BF16 GPU validation |
|---:|---:|---:|---:|
| 42 | 30/32 (93.75%) | 8/16 (50.0%) | 6/16 (37.5%) |
| 1337 | 29/32 (90.625%) | 11/16 (68.75%) | 11/16 (68.75%) |

The high training fit and weak validation performance are compatible with
memorization or weak transfer. The CPU/GPU difference for seed 42 is retained
as an evaluation-precision discrepancy, not silently merged into the
scientific result.

## Interpretation

Established by this diagnostic:

- The source curriculum has substantial non-identifiability under its own
  grammar.
- The model fits the repeated source presentations more strongly than it
  generalizes to held-out source procedures.
- The original two-seed source gate remains failed: 6/16 and 11/16 under the
  canonical BF16 GPU evaluation.

Not established:

- A causal explanation for the source-gate failure.
- Productive retention or target transfer.
- That a revised curriculum or longer training would succeed.

The correct status remains **INVALID/INCONCLUSIVE prerequisite failure** for
the productive-transfer experiment. No C0--C3 target training should start
from this diagnostic.
