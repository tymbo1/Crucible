# Crucible frozen source-competence diagnostic

Date: 2026-10-31  
Status: **Authorized diagnostic; frozen inference only**  
This is a separate experiment from the blocked two-arm comparison and does not
open C0--C3.

## Question

How does the existing step-15,000 Crucible checkpoint perform on the repaired
source curriculum when execution, query-identifiable induction and
function-identifiable induction are measured separately under the corrected
one-step decoder?

This diagnostic does not claim neural learning, architectural superiority,
productive retention or cumulative learning. It does not modify the checkpoint.

## Frozen inputs

- Model checkpoint: `/home/timbushnell/crucible_120m_checkpoints_20261029/crucible_120m_foundation_step_015000.pt`
- Checkpoint SHA-256: `e49faa226aaa96bc26f4abbed700ecf313ad59cbf68869cfc0327c61f5902d5a`
- Curriculum manifest SHA-256: `699c7a2f82a5ff630ee59df689916131ec4eabd5de1513cdf4665f15141e055c`
- Manifest seeds: `42`, `1337`.
- Evaluated splits: source/query validation, function validation, execution validation, source test and execution test.
- Private source oracles are read-only evaluator inputs and are never supplied to the model.
- No target data or sealed target answers are loaded.

## Decoder and metrics

The evaluator selects logits at the fixed position predicting the first byte
after `ANSWER=`. Only `0` and `1` are answers. Ties and non-answer tokens are
abstentions; non-finite logits are a hard failure.

For each seed, split and category it records exact accuracy, selective accuracy,
coverage, abstention rate, margins, per-task predictions and the initial
protected-test result. Since this is frozen inference, checkpoint transition
counts are explicitly `not_applicable`; no retention claim is made.

## Positive control

The same decoder is exercised with synthetic finite logits whose correct answer
is known. This validates evaluator plumbing only and is not neural performance.

## Training boundary

No source-task training is included. If frozen inference shows that the
checkpoint is not competent, a separate source-training diagnostic must be
registered with its own update budget, checkpoint schedule, transition rules and
authorization. It must not be appended to this run.

