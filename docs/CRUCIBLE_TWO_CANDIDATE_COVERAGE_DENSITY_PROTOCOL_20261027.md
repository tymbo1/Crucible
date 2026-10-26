# Crucible Two-Candidate Coverage-Density Experiment v1

Date: 2026-10-27  
Status: frozen before execution  
Branch: `research/two-candidate-forensics-20261027`

## Objective

Test whether broader training-pair coverage improves transfer to one fixed,
unseen candidate panel when the model, optimizer, update budget, context pool,
serialization and evaluator are held constant.

This is a coverage-density experiment, not a causal topology experiment. The
conditions are nested and therefore change graph degree, connectivity and
finite path structure together with the number of distinct training pairs.
The results must not be interpreted as isolating graph diameter or proving
relational composition.

## Fixed validation panel

All conditions are evaluated on `M2`, which is never present in training:

`M2 = (0,3), (1,2), (4,7), (5,6)`.

The exact validation records from the completed `T_M0_V_M2` crossover cell
are reused by reference. This keeps the held-out rows identical across
conditions and avoids regenerating the validation split.

## Coverage conditions

The training graph uses identifiers 0–7 and the following unordered edges.
Every edge is represented in both candidate orders and with authority equal to
each candidate plus authority `9` for abstention.

| Condition | Training edges | Edges | Contexts per edge | Rows |
|---|---|---:|---:|---:|
| `D4` | `M0` | 4 | 6 | 144 |
| `D8` | `M0 ∪ M1` | 8 | 3 | 144 |
| `D12` | `M0 ∪ M1 ∪ M3` | 12 | 2 | 144 |

The six training contexts are fixed as `(1,6)`, `(1,8)`, `(2,7)`, `(2,9)`,
`(3,8)`, `(4,9)`. Context assignments are deterministic and round-robin
balanced so that every condition has 24 edge-context assignments and each
context is used four times. The number of rows, optimizer updates and total
token exposure are therefore matched; repetition per distinct edge is not.

`D4` is not retrained. Its two arms are the immutable `M0`-train/`M2`-held
out checkpoints from `crucible_two_candidate_pair_crossover_20261027_v1`.
Only the four `D8`/`D12` arms (two conditions × two seeds) are new training.

## Model and training controls

Use the unchanged dense `SHARED_SCORER` path:

- seeds: `42`, `1337`;
- deterministic cyclic schedule;
- 500 updates;
- batch size 12;
- optimizer, learning rate, clipping, tokenizer, model and evaluator unchanged;
- training competence gate: at least 95% row-level accuracy on the 144 training rows.

The fixed validation set has 96 rows: 32 target-slot 0, 32 target-slot 1 and
32 abstentions, with both candidate orders and four contexts.

At steps 1, 50, 100, 200, 300, 400 and 500, record training and fixed-panel
validation metrics. The validation trace is diagnostic monitoring only; the
pre-registered endpoint is the final step-500 checkpoint.

## Required measurements

Report, with denominators:

- training and validation row accuracy;
- candidate-selection accuracy excluding abstentions;
- slot accuracy for targets 0, 1 and 8;
- paired counterfactual correctness and predicted flips;
- accuracy by held-out unordered pair and candidate order;
- training-graph components, degree sequence, finite diameter and shortest
  paths from each held-out pair to document structural changes;
- seed-level variation and validation trajectories.

The shortest-path fields are descriptive. Because this task directly labels
identifier matching rather than requiring a transitive relation through graph
intermediates, they do not constitute a path-composition test.

## Decision criteria

- Consistent improvement from `D4` to `D8` to `D12` across seeds supports a
  coverage-density contribution under this task.
- Improvement that depends strongly on the seed or on the specific added
  edges supports training-context or optimization sensitivity.
- No improvement after the training gate passes argues against simple pair
  coverage as a sufficient explanation, but does not identify a mechanism.
- No claim about graph topology, path-length composition or variable-cardinality
  transfer is permitted from this experiment alone.

## Invariants and stop conditions

Stop before training if any of the following fail: exact fixed validation
identity, M2 exclusion from every training condition, 144-row training budget,
balanced target counts, balanced context assignment, unique inputs/record IDs,
oracle labels, source-context membership, checkpoint hash verification or
train/validation contamination audit.

Do not alter the frozen architecture, tokenizer, optimizer, schedule, evaluator
or sealed-test boundary. Do not modify prior checkpoints. If an invariant cannot
be maintained, report the failure instead of silently changing the design.
