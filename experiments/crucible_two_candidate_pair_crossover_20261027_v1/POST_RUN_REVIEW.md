# Post-run scientific review

## Verification

This crossover combines the four immutable cyclic cells from
`crucible_two_candidate_pair_coverage_20261027_v1` with the 16 newly trained
arms in this directory. The prior cells were not retrained or modified.

The new run used the same dense `SHARED_SCORER`, cyclic schedule, 500 updates,
batch size 12, compact serialization, contexts, tokenizer, evaluator and 95%
training gate. All 16 new arms passed the gate. The combined analysis contains
24 arms, three training panels for each held-out panel and two seeds per
ordered panel combination.

The finalization record reports no sealed-test access and no prior-checkpoint
modification. The 126 files listed in the crossover hash manifest were
recomputed with zero mismatches, including the verified 66-file hash manifest
from the prior experiment.

## Combined results

| Measure | Result |
|---|---:|
| Training rows | 3,456/3,456 (100%) |
| Training paired groups | 1,152/1,152 (100%) |
| Validation rows | 2,031/2,304 (88.15%) |
| Candidate-selection rows | 1,263/1,536 (82.23%) |
| Validation paired groups | 495/768 (64.45%) |
| Validation abstention | 768/768 (100%) |

By seed, row accuracy was 1,002/1,152 for seed 42 and 1,029/1,152 for seed
1337. Paired correctness was 234/384 and 261/384 respectively. The seed
difference exists, but the panel-combination effect is larger in several
matched comparisons.

## Training-partner effects

Paired validation aggregated over the two seeds for each fixed held-out panel
and training partner:

| Held-out panel | Train M0 | Train M1 | Train M2 | Train M3 |
|---|---:|---:|---:|---:|
| M0 | — | 47/64 | 0/64 | 47/64 |
| M1 | 32/64 | — | 64/64 | 34/64 |
| M2 | 47/64 | 33/64 | — | 48/64 |
| M3 | 47/64 | 64/64 | 32/64 | — |

The M0 row is the clearest controlled contrast: the same held-out panel is
0/64 paired after M2 training but 47/64 after either M1 or M3 training, with
the zero result replicated by both seeds. This supports a training-panel or
coverage-graph dependence under the tested protocol.

## Pair-level crossover

Each unordered pair was evaluated in six held-out conditions. The pooled
paired results ranged from 16/48 for `(4,5)` to 41/48 for `(4,7)`. No pair was
zero across all six evaluations. The apparent failures therefore do not
support a persistent single-pair or identifier-family defect by themselves;
the same pair can succeed or fail as the training partner panel and seed
change.

The most informative negative condition is M2→M0: all four M0 pairs had
0/8 paired groups for both seeds. That common panel-level failure is more
consistent with a coverage arrangement or training-trajectory effect than
with one isolated pair, although it does not identify the internal cause.

## Conclusions supported

1. The dense cyclic model can fit every arm's training task under this
   two-candidate protocol.
2. Perfect training fit does not guarantee transferable matching: paired
   validation remains 495/768.
3. Abstention is not the bottleneck: all 768 abstention rows are correct.
4. Held-out performance changes materially with the training panel, including
   a replicated 0/64 versus 47/64 contrast for M0. Behavioral coverage-graph
   dependence is supported under the tested conditions.
5. Seed and training-partner effects interact. This is not evidence for a
   specific neural mechanism or an architectural impossibility.

## Conclusions not established

The crossover does not establish a general identifier-matching algorithm,
variable-cardinality transfer, or a unique optimization/representation
mechanism. The four perfect matchings cover only 16 of the 28 possible
unordered pairs among eight identifiers. Richer partner coverage, additional
seeds and a separate evaluation of all 28 pairs would be needed before
generalizing beyond this graph family.

## Next controlled step

Freeze this crossover as the behavioral baseline. The next experiment should
vary training-partner coverage while holding the held-out pairs and exposure
constant, then add the missing unordered pairs in a separate, preregistered
extension. Do not change the architecture or add auxiliary objectives until
the coverage effect is characterized further.
