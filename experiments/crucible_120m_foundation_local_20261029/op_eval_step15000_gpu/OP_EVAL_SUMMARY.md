# OP evaluation at the 120M step-15,000 checkpoint

Classification: `post_foundation_checkpoint_op_comparison`

This is a read-only, single-checkpoint evaluation of the preserved 120M
Crucible model after 25,270,547 cumulative training tokens. It is exploratory
and is not a preregistered architecture comparison.

## Results

| Policy | Verified correct | Accuracy | Mean generated tokens | Mean passes |
|---|---:|---:|---:|---:|
| One pass | 52/304 | 17.11% | 9.47 | 1.00 |
| Fixed depth 2 | 9/304 | 2.96% | 23.33 | 2.00 |
| External OP controller | 57/304 | 18.75% | 52.40 | 1.83 |

Relative to one pass, the external controller added 5 correct records
(`+1.64` percentage points) while producing approximately 5.5 times as many
model tokens. Fixed-depth reconsideration substantially degraded accuracy.

## Domain breakdown

| Domain | Records | One pass | Fixed depth 2 | External OP |
|---|---:|---:|---:|---:|
| BBH | 64 | 17 | 1 | 20 |
| Big-Bench Lite | 64 | 0 | 0 | 0 |
| ETHICS | 64 | 33 | 7 | 35 |
| MATH | 64 | 1 | 1 | 1 |
| SciBench | 48 | 1 | 0 | 1 |

The apparent OP gain is concentrated in BBH and ETHICS. Math, SciBench and
Big-Bench Lite remain essentially unsolved in this evaluation.

## Interpretation boundary

The external OP controller produced a small positive difference at this
checkpoint, but this does not establish that OP improves reasoning generally.
The controller is external and bounded, the evaluation uses one checkpoint
and seed, and the test set is a capped pilot. Gold answers were used only
after generation for verification; they were not supplied during proposal
generation. The model's OP mechanism remains uninternalized.

The result supports running a properly matched OP ablation only if its
additional compute and task-selection rules are fixed in advance. It does not
justify claiming cumulative learning, broad reasoning competence, or solved
recursive reasoning.
