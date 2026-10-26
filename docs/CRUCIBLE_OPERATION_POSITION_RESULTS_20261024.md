# Crucible operation-position gate results

Date: 2026-10-24

## Executive result

The six CPU arms completed successfully, but the isolated B-only control did not achieve primitive generalization. All conditions fit their training records substantially better than their validation or held-out records.

Final 2,000-update free-running exact accuracy:

| Condition | Seed 42 train | Seed 42 validation | Seed 42 held-out | Seed 1337 train | Seed 1337 validation | Seed 1337 held-out |
|---|---:|---:|---:|---:|---:|---:|
| A→B standard | 86.61% | 3.57% | 1.56% | 97.32% | 5.36% | 4.69% |
| B-only | 100.00% | 12.50% | 0.00% | 100.00% | 4.17% | 8.33% |
| B→A reversed | 95.54% | 8.93% | 6.25% | 94.64% | 8.93% | 0.00% |

The B-only arm therefore does not support the proposed interpretation that the second operation is easy in isolation but fails only when composed. It learned the isolated training examples, but not the underlying operation reliably on new examples under this task interface and budget.

## Frozen protocol

- Three conditions: `A_to_B`, `B_only`, `B_to_A`.
- Seeds 42 and 1337.
- Dense control with 21,968,640 parameters.
- 2,000 CPU updates per arm; checkpoints at 0, 500, 1,000, 1,500 and 2,000.
- A→B and B→A: 112 train, 56 validation and 64 held-out records.
- B-only: 48 train, 24 validation and 24 held-out records.
- Training pairs covered seven ordered operation pairs; held-out base pairs were `add+sub` and `mul+sub`.
- Every operation appeared in both positions in the composition training pair set.
- The diagnostic used new versioned prompts and records; the original benchmark and sealed test split were not modified or evaluated.

The B-only prompt supplies the correct intermediate and asks for one final digit. It is intentionally an isolated control, but its prompt format differs from the two-step composition prompt; its result therefore measures isolated-task learnability plus interface transfer, not a perfectly context-matched primitive oracle.

## Secondary metrics

At the final checkpoint, validation results were:

| Condition | Seed | Teacher-forced final | Teacher-forced intermediate | Free final | Free end-to-end |
|---|---:|---:|---:|---:|---:|
| A→B | 42 | 16.07% | 17.86% | 8.93% | 3.57% |
| A→B | 1337 | 16.07% | 35.71% | 10.71% | 5.36% |
| B-only | 42 | 12.50% | — | 12.50% | 12.50% |
| B-only | 1337 | 4.17% | — | 4.17% | 4.17% |
| B→A | 42 | 21.43% | 33.93% | 21.43% | 8.93% |
| B→A | 1337 | 19.64% | 37.50% | 8.93% | 8.93% |

The higher B→A final-token result for seed 42 does not translate into strong end-to-end performance. Across both seeds, neither order demonstrates reliable held-out composition.

## Interpretation

The experiment establishes:

1. The dense control can fit broader operation-position training sets.
2. Longer training does not by itself produce transfer on the new validation and held-out distributions.
3. B-only is not yet a high-performing primitive control; the 90% interpretive marker was not approached on validation or held-out data.
4. The current result cannot distinguish operation difficulty from prompt/interface generalization, because B-only uses a different prompt form and all arms have sparse task coverage relative to the possible input combinations.

The apparent difference between A→B and B→A is therefore descriptive, not evidence that one order is intrinsically superior. The two seeds disagree in held-out behaviour, and all held-out accuracies remain low.

## Integrity and resource boundary

All six arm hash manifests matched with zero mismatches. Train, validation and held-out inputs were disjoint within every arm. No original benchmark artifacts were modified. No GPU, cloud resource or pretrained checkpoint was used.

## Recommended next gate

Do not change the neural architecture yet. First repair the control design before drawing a position conclusion:

- create a context-matched B-only prompt using the same composition template and supplied intermediate;
- increase primitive-task diversity while keeping operation and operand coverage balanced;
- use a small positive-control set to verify B-only can generalize before treating it as a competence ceiling;
- retain held-out operation combinations and free-running evaluation;
- report operation-level and target-value strata with uncertainty intervals.

Only if the repaired B-only control generalizes strongly while A→B remains weak would the data support a specific composition/intermediate-integration bottleneck.
