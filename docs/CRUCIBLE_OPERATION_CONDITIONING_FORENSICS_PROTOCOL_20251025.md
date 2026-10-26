# Crucible Operation-Conditioning Mechanism Investigation

Date: 2025-10-25  
Status: preregistered read-only checkpoint forensics  
Hardware: CPU only  

## Objective

Identify which observable mechanism is consistent with the existing S3 failure:
the dense 22M model can learn fixed `SELECT_FIRST` and `SELECT_SECOND` tasks,
but did not reliably condition operand selection on the operation identifier
when both operations were trained jointly.

This study is diagnostic. It does not change the architecture, retrain the
original S3 checkpoints, open the sealed test set, or claim that a probe alone
establishes a causal mechanism.

## Frozen sources

- Checkpoints and immutable records:
  `experiments/crucible_role_selection_competence_20251025_v5/`
- Four S3 arms:
  canonical/factorized interfaces × seeds 42/1337.
- Existing dense model, tokenizer, evaluator and 500-update training recipe.
- The completed operation-token identification control:
  `experiments/crucible_operation_token_identification_20251025_v1/`.

## Phase 1 — Reproduction and checkpoint audit

For every S3 arm, reload the checkpoint and reproduce the recorded training and
validation metrics. Verify checkpoint hashes, record hashes, split disjointness,
labels, output positions and output decoding. Any mismatch is reported rather
than silently repaired.

## Phase 2 — Read-only representation and sensitivity analysis

For matched unequal operand pairs under both operation identifiers:

- Capture token-embedding, per-layer block and final-normalization states.
- Test operation-identity decodability with a train-fitted ridge probe and a
  nearest-centroid control, evaluating validation rows separately.
- Measure operation-conditioned hidden-state differences.
- Measure output-logit sensitivity when only the operation identifier changes.
- Record expert-routing/attention diagnostics only where exposed by the loaded
  model. The dense checkpoints do not expose attention probability matrices
  through the current scaled-dot-product-attention interface; this is reported
  explicitly rather than reconstructed as if it were native telemetry.

## Phase 3 — Gradient and trajectory diagnostics

Using the frozen S3 checkpoints, compute operation-specific loss gradients on
matched examples and report cosine similarity, norms and per-parameter effects.
Also reproduce the saved training curves and compare the two operation losses.
These are observational diagnostics of the trained objective, not proof that
gradient conflict caused the result.

## Phase 4 — Activation patching

For a preregistered subset of matched validation pairs and layers, patch the
operation-token representation or output-position representation from the
opposite operation into the target run. Include same-operation cross-pair and
deterministic unrelated-pair controls. Report whether the patched prediction
moves toward the donor operation's expected operand, while preserving the
unpatched baseline.

Patch results are interpreted as causal evidence only for the intervention
tested. They do not establish that the patched feature is the unique mechanism.

## Hypothesis interpretation

The report distinguishes:

1. operation information not represented;
2. operation information represented but lost across layers;
3. operation information retained but not causally used at the output;
4. optimization or gradient interference;
5. an operation-blind output shortcut.

If the evidence cannot distinguish these hypotheses, the conclusion remains
unresolved.

## Exclusions

No new training, architecture changes, FiLM, recurrence, memory, symbolic
execution assistance, neural selector, CMS2 integration, GPU execution or
sealed-test access is permitted in this run.

## Required artifacts

- Reproduced metrics and checkpoint/hash audit.
- Per-example predictions and logits summary.
- Layerwise probe and activation statistics.
- Gradient diagnostics and training-curve comparison.
- Activation-patching intervention matrix with controls.
- Machine-readable results, protocol manifest and final report.
