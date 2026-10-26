# Crucible Two-Candidate Pair-Coverage Crossover v1

Date: 2026-10-27
Status: frozen before execution
Branch: `research/two-candidate-forensics-20261027`

## Objective

Complete the ordered training-panel/held-out-panel crossover for the four
disjoint candidate-pair matchings used in the pair-coverage experiment. The
prior run evaluated four of the twelve possible ordered panel combinations.
This run trains only the remaining eight combinations, yielding 24 total arms
when combined with the eight immutable prior arms.

The purpose is to distinguish pair-specific weakness, training-pair coverage
or graph effects, and seed/optimization instability. This is a behavioral
experiment; it does not claim to identify a neural mechanism.

## Panels and crossover matrix

The four fixed perfect matchings are:

- `M0`: `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`
- `M1`: `(0,2)`, `(1,3)`, `(4,6)`, `(5,7)`
- `M2`: `(0,3)`, `(1,2)`, `(4,7)`, `(5,6)`
- `M3`: `(0,4)`, `(1,5)`, `(2,6)`, `(3,7)`

Every ordered combination with different train and validation panels is valid.
The already completed cells are retained by reference and are not retrained:

| Train panel | Held-out panel | Prior cell |
|---|---|---|
| M0 | M1 | P0 |
| M1 | M2 | P1 |
| M2 | M3 | P2 |
| M3 | M0 | P3 |

The eight new ordered combinations are:

| Train panel | Held-out panel |
|---|---|
| M0 | M2 |
| M0 | M3 |
| M1 | M0 |
| M1 | M3 |
| M2 | M0 |
| M2 | M1 |
| M3 | M1 |
| M3 | M2 |

Each combination uses seeds `42` and `1337`, for 16 new arms and 24 total
arms in the combined analysis. Thus each of the 16 panel pairs is held out
under three different training-panel arrangements and two seeds.

## Fixed task and controls

Preserve the prior pair-coverage settings exactly:

- dense `SHARED_SCORER`, approximately 22.1M parameters;
- cyclic schedule, not shuffled;
- 500 updates, batch size 12;
- seeds 42 and 1337;
- canonical `TRAIN_COMPACT` serialization;
- six training contexts `(1,6)`, `(1,8)`, `(2,7)`, `(2,9)`, `(3,8)`, `(4,9)`;
- validation contexts `(1,6)`, `(2,7)`, `(3,8)`, `(4,9)`;
- 144 balanced training rows and 96 balanced validation rows per arm;
- target classes 0, 1 and abstention 8 balanced within each split;
- 95% row-level training competence gate;
- deterministic oracle, existing tokenizer and evaluator;
- no sealed-test access and no modification of prior checkpoints.

The new cells regenerate their records deterministically under the same frozen
panel/context rule. Training and validation unordered-pair sets must be
disjoint in every cell; exact input and record-ID overlap must be zero.

## Required measurements

For every new arm, and for the combined 24-arm analysis, report:

- training and validation row accuracy;
- training-gate status;
- slot-0, slot-1 and abstention accuracy;
- paired counterfactual correctness;
- predicted counterfactual flips;
- accuracy by unordered pair, held-out panel and context;
- seed-level and train-panel/held-out-panel variation.

Transfer claims are permitted only for arms passing the training gate.

## Decision criteria

- If the same pair remains weak across different training panels and seeds,
  pair-specific or identifier-specific difficulty is supported.
- If performance changes with the training partner panel while the held-out
  pair is fixed, coverage-graph dependence is supported.
- If results differ primarily by seed under the same panel combination,
  optimization or initialization instability is supported.
- If all crossover cells transfer comparably, the earlier gap was likely tied
  to the original split or training trajectory.

These criteria are descriptive and do not establish an internal mechanism.

## Stop conditions

Stop before new training if the prior hash audit, panel construction, class
balance, context controls, oracle labels, candidate spans or cross-split
contamination checks fail. Do not silently change architecture, optimizer,
tokenizer, evaluator, schedule or update budget to repair an invariant.
