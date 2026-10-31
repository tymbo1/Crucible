# Crucible source-training diagnostic

Date: 2026-10-31
Status: **registered diagnostic; training not yet authorized**
Protocol schema: `crucible-source-training-diagnostic-v1-20261031`

## Purpose

The frozen source diagnostic found no valid primary answers from the step-15,000
checkpoint and only chance-level forced-binary performance. This separate,
bounded diagnostic asks whether the existing 120M Crucible model can acquire the
repaired source capabilities when trained on public source examples.

It is not a comparison with a conventional dense baseline, does not test
productive retention or C0--C3, and does not reopen the closed Track-B
experiment. A positive result establishes only source-task acquisition under
this prompt, objective, budget and decoder.

## Frozen inputs and boundaries

- Base checkpoint: `crucible_120m_foundation_step_015000.pt`.
- Public curriculum: `experiments/crucible_source_curriculum_repair_20261031/`.
- Private source oracles: local-only files outside Git, used only for
  validation/test scoring.
- Seeds: `42` and `1337`.
- No target manifests, target inputs or sealed target answers may be opened.
- The base checkpoint, public manifest and private oracle hashes are recorded
  before training.
- The original frozen-inference diagnostic and old source-gate run remain
  unchanged.

## Two diagnostic phases

Each phase starts independently from the same frozen base weights. The source
training optimizer is fresh AdamW; the base foundation optimizer state is not
reused. This prevents foundation momentum from silently becoming a phase
intervention.

### Phase E — explicit-procedure positive control

Train only on `execution_train` (16 public labelled examples). The canonical
AST is supplied in the prompt, so this phase isolates execution from procedure
induction. It runs 500 optimizer updates, with checkpoints at updates
`0, 50, 100, 250, 500`.

### Phase I — source induction diagnostic

Train only on `source_train` (32 public labelled examples). The AST is hidden;
the prompt contains demonstrations and a query. It runs 1,000 optimizer
updates, with checkpoints at updates `0, 100, 250, 500, 750, 1000`.

Phase I is run only after Phase E has been completed for both seeds and the
execution-validation qualification threshold is met for both seeds. If Phase E
fails, Phase I is not interpreted as evidence about induction.

## Training configuration

| Setting | Frozen value |
|---|---:|
| Model | Existing 120M Crucible latent-attention + SwiGLU |
| Microbatch | 4 examples |
| Gradient accumulation | 2 microbatches |
| Effective examples/update | 8 |
| Optimizer | fresh AdamW, betas `(0.9, 0.95)`, eps `1e-8` |
| Learning rate | `3e-4` |
| Weight decay | `0.01` through the existing parameter groups |
| Context | 512 byte-level positions |
| CUDA precision | BF16 autocast only when supported; otherwise training refuses CUDA |
| CPU precision | FP32 |
| Gradient clipping | max norm `1.0`, non-finite gradients hard-stop |
| Answer objective | cross-entropy only at the fixed first-answer-byte position |
| Data order | deterministic per seed and phase; schedule digest recorded |

The decoder used during training and evaluation is identical: the answer
position is fixed after `ANSWER=`, and only byte tokens for `0` and `1` are
eligible. Non-answer top tokens and ties are abstentions. Free-running
generation is not used for the primary gate.

## Metrics and gates

Every checkpoint reports training and held-out results separately for:

- explicit procedure execution;
- query-identifiable induction;
- function-identifiable induction.

For every split it records exact accuracy over all eligible rows, selective
accuracy, coverage, abstention, per-task logits/predictions, label balance,
constant-majority baselines and transition tables. Protected replay compares
each checkpoint with both the previous checkpoint and the phase's initial
checkpoint. Correct-to-wrong, correct-to-abstain and all other answer-state
transitions are retained; no aggregate score can hide them.

Qualification thresholds are applied separately to each seed and only to the
validation splits:

| Category | Validation rows/seed | Minimum exact correct | Minimum coverage |
|---|---:|---:|---:|
| Explicit execution | 16 | 14/16 | 15/16 |
| Query-identifiable induction | 32 | 26/32 | 29/32 |
| Function-identifiable induction | 16 | 13/16 | 14/16 |

Test splits are sealed from training and checkpoint selection. They are scored
only at the fixed checkpoints and reported descriptively. The 16-row function
validation result is pilot evidence and requires a larger replication before a
broad function-induction claim.

Balanced-random (`0.5`) and constant-label baselines are descriptive controls,
not substitutes for exact accuracy. Abstention cannot improve the exact gate.

## Fail-closed and stopping rules

The runner stops immediately and refuses to mark a checkpoint recoverable on
any non-finite loss, gradient, model tensor, optimizer tensor, output logit or
checkpoint payload; an OOM/resource failure; a hash mismatch; or a public/private
row mismatch. Checkpoints are written atomically, reloaded and rechecked before
being reported.

There is no validation-based early checkpoint selection. The fixed final update
is the primary result. A checkpoint with protected regressions is recorded but
not promoted as an accepted capability checkpoint. Phase E failure blocks
interpretation of Phase I; it does not falsify neural induction generally.

## Scientific classifications

- **Established under tested conditions:** the corresponding source capability
  passes its per-seed validation gate with valid coverage and no evaluator
  integrity failure.
- **Not established:** the capability misses its gate or the run stops before
  qualification.
- **Invalid:** leakage, malformed splits, non-finite state, checkpoint failure,
  decoder mismatch or another protocol-integrity failure prevents interpretation.
- **Inconclusive prerequisite failure:** Phase E fails, so induction evidence is
  not interpreted as a test of productive retention.

No outcome from this diagnostic authorizes C0--C3, claims cumulative learning,
or establishes architectural superiority.
