# Digitwise Addition Competence Ladder — Results

Date: 2026-10-25  
Protocol: `CRUCIBLE_DIGITWISE_ADDITION_LADDER_PROTOCOL_20261025.md`  
Run directory: `experiments/crucible_digitwise_addition_ladder_20261025/`

## Outcome

The ladder stopped at Stage 1. All eight Stage-1 arms reached the 95% training-fit threshold, but none reached the 90% validation advancement threshold. Stage 2 was therefore not run.

## Stage 1: single-digit addition without carry

| Arm | Interface | Seed | Train exact | Validation exact | Fit | Advance |
|---|---|---:|---:|---:|---|---|
| Direct | Canonical | 42 | 100.00% | 73.33% | Pass | Fail |
| Curriculum | Canonical | 42 | 100.00% | 73.33% | Pass | Fail |
| Direct | Canonical | 1337 | 100.00% | 60.00% | Pass | Fail |
| Curriculum | Canonical | 1337 | 100.00% | 60.00% | Pass | Fail |
| Direct | Factorized | 42 | 95.00% | 60.00% | Pass | Fail |
| Curriculum | Factorized | 42 | 95.00% | 60.00% | Pass | Fail |
| Direct | Factorized | 1337 | 100.00% | 73.33% | Pass | Fail |
| Curriculum | Factorized | 1337 | 100.00% | 73.33% | Pass | Fail |

The validation set contains 15 held-out operand pairs. Training covered every input digit and every sum `0..9`, while complete operand pairs remained disjoint.

## Controls and integrity

- Eight Stage-1 arms: two interfaces × two seeds × direct/curriculum controls.
- 500 updates per arm.
- Symbolic oracle: 100%.
- Zero oracle errors and zero train/validation pair overlap.
- Coverage audit passed for all required digits and sums.
- Output length was fixed at one digit.
- No original sealed test set was opened.
- GPU use was disabled.
- Stage 2 was correctly skipped after the Stage-1 advancement gate failed.

The initial launch produced five large checkpoints before stopping under resource pressure. The three missing arms were completed with the same protocol using a checkpoint-light repair runner; those repair artifacts are included under `experiments/.../repair/` and did not alter the measured conditions.

## Interpretation

Crucible can fit single-digit addition examples, but it did not reliably transfer the operation to held-out operand pairs even when every individual digit and every possible no-carry sum appeared in training. This is a more informative failure than the earlier mixed addition result, because training competence was established first.

The result does not yet diagnose two-digit positional composition or carry propagation. Those stages were not run. It does indicate that pairwise operand combination remains an unresolved transfer problem under this interface, objective and update budget.

## Decision

Keep the architecture frozen and do not introduce carry mechanisms or neuromorphic components yet. The next focused control should enlarge the single-digit validation set or use a systematic leave-one-pair-out design, then test whether explicit intermediate supervision or a symbolic pairwise operation improves transfer. Two-digit addition should remain blocked until this single-digit gate passes.
