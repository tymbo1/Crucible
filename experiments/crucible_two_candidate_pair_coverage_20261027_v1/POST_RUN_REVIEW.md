# Post-run scientific review

## Scope and verification

This review interprets the completed `crucible_two_candidate_pair_coverage_20261027_v1`
run. It does not alter the generated report, checkpoints, predictions, splits or
hash manifest.

The run used the unchanged dense `SHARED_SCORER`, cyclic schedule, 500-update
budget, batch size 12, compact serialization and two seeds (42 and 1337). Four
partitions rotated four disjoint perfect-matching panels so that each panel was
held out once. Training and validation unordered-pair sets were disjoint in
every partition; the contexts used for validation were a subset of the training
context pool.

The finalization record reports 8/8 training gates, no prior-checkpoint
modification and no sealed-test access. The 66 files listed in `HASHES.json`
were recomputed successfully with zero mismatches.

## Reproduced aggregate results

| Measure | Result |
|---|---:|
| Training rows | 1,152/1,152 (100%) |
| Training paired groups | 384/384 (100%) |
| Validation rows | 656/768 (85.42%) |
| Validation paired groups | 144/256 (56.25%) |
| Validation abstention | 256/256 (100%) |
| Validation slot 0 | 207/256 (80.86%) |
| Validation slot 1 | 193/256 (75.39%) |

The two seeds had the same aggregate validation totals: 328/384 rows and
72/128 paired groups each. This equality hides a strong partition-by-seed
interaction rather than demonstrating seed invariance.

## Held-out panel results

| Held-out panel | Seed 42 | Seed 1337 | Pooled |
|---|---:|---:|---:|
| M0 | 87/96, 23/32 paired | 88/96, 24/32 paired | 175/192, 47/64 paired |
| M1 | 80/96, 16/32 paired | 80/96, 16/32 paired | 160/192, 32/64 paired |
| M2 | 65/96, 1/32 paired | 96/96, 32/32 paired | 161/192, 33/64 paired |
| M3 | 96/96, 32/32 paired | 64/96, 0/32 paired | 160/192, 32/64 paired |

The P1/M2 and P2/M3 results are the clearest seed interaction: the stronger
seed in one partition is the weaker seed in the other. All four panels were
therefore evaluated, but the design does not provide repeated independent
held-out evaluations of the same unordered pair.

## Conclusions supported by the data

1. The dense cyclic configuration can fit this balanced two-candidate task:
   all eight arms passed the 95% training gate.
2. Training competence does not imply reliable pair-transfer competence under
   these splits: paired validation was 144/256, despite 100% training fit.
3. Abstention is not the bottleneck in this run; all 256 abstention rows were
   correct.
4. The validation failures are compatible with a candidate-pair coverage gap,
   and the large seed-by-panel interaction shows that optimization or
   initialization effects remain important.

## Conclusions not established

This run does not establish a persistent identifier-family defect, exact-pair
memorization as the mechanism, or a neural explanation. Each of the 16 tested
unordered pairs was assigned to one held-out panel only. When a pair was in a
training panel it was fit perfectly, but there was no independent validation
measurement of that same pair under a trained condition. Therefore the current
design cannot distinguish pair-specific difficulty from the effect of the
particular panel and seed combination.

## Next controlled experiment

Run repeated balanced pair holdouts in which each tested unordered pair is
held out in multiple independently generated partitions, with the same
architecture, cyclic schedule, update budget, competence gate and compact
interface. Report pair-level held-out accuracy and paired counterfactual
correctness across partitions and seeds. A pair-specific claim should require
the same pair to remain weak across different training contexts and holdout
compositions; a coverage explanation is supported if performance moves with
the holdout while trained-pair fit remains high.

This is a behavioral replication design, not an architectural intervention.
