# Crucible D2 Readout Competence Ladder v1

Date: 2026-10-26  
Status: preregistered CPU diagnostic

## Objective

Determine whether the unresolved D2 result is caused by the final readout
implementation, by the frozen final decision representation, or by the
authority-conditioned selection problem itself.

The ladder progresses from an explicitly encoded synthetic selection signal to
the actual late-authority input. Training competence is required before any
held-out transfer result is interpreted.

## Frozen controls and boundaries

- Preserve the existing 22M dense architecture and tokenizer.
- Use the accepted D1 early-authority checkpoints as frozen backbone controls;
  their hashes are recorded in the output manifest.
- Use the same canonical and factorized operand-pair splits as the accepted
  causal-authority experiment.
- Use seeds 42 and 1337.
- Do not access the sealed test split.
- Do not modify the accepted D0/D1/D2 checkpoints.
- New linear readout heads are diagnostic components, not replacements for the
  frozen baseline architecture.

## Ladder stages

### S0 — Synthetic readout sanity

Train a two-class linear head on a two-dimensional feature in which the target
candidate slot is explicitly encoded. This tests the head, target convention,
optimizer and evaluation code without involving the backbone.

### S1 — Single candidate

Use one operation candidate and a final decision token. The only valid pointer
is candidate slot 0. This is intentionally a simple final-readout competence
check with no authority conflict.

### S2 — Two candidates with direct target

Use two operation candidates and add a late `select_position=E/L` cue whose
meaning is directly the candidate slot to select. This tests whether a readout
can use a causally available explicit selection signal.

### S3 — Two candidates with late authority

Use the actual late `authority_code=E/L` cue from the D2 interface. The target
candidate slot is determined by the authority code, while operation order is
varied between `FS` and `SF`. This is the first authority-conditioned ladder
stage.

### S4 — End-to-end late-authority readout

Train a fresh decision-pointer head jointly with the dense backbone using only
the decision loss on S3. This is a controlled end-to-end follow-up to the
frozen-state ladder, not a modification of the accepted checkpoints.

### S5 — Held-out transfer

After S3 or S4 passes training competence, evaluate on the disjoint validation
pairs. Also evaluate the selected model on an unseen late-layout variant in
which the candidate block is moved after the operands and neutral padding is
inserted before authority. These are exploratory transfer diagnostics and are
reported separately from the standard validation split.

## Metrics and gates

For each stage and arm report:

- training and validation rows;
- candidate-pointer accuracy;
- paired counterfactual flips for S2/S3/S4;
- paired two-way correctness;
- target and prediction positions;
- loss traces and checkpoint hashes where applicable.

The competence gate is at least 95% training pointer accuracy. The validation
threshold is 90%, but no validation interpretation is made for a stage that
fails the training gate.

For S3/S4, authority sensitivity requires the selected candidate slot to flip
when only `authority_code=E` versus `authority_code=L` changes.

## Interpretation rules

- S0 failure indicates an implementation or optimizer problem in the readout
  control.
- S0/S1 pass but S2 fails indicates a problem using an explicit late cue or
  the frozen representation.
- S2 passes but S3 fails indicates that direct target reading works while
  authority-to-candidate binding remains unresolved.
- S3 frozen failure followed by S4 end-to-end success implicates frozen
  representation mismatch or insufficient backbone adaptation.
- S4 failure before training competence is not evidence that a causally valid
  final decision state cannot use late authority.

No stage establishes general instruction hierarchy or autonomous reasoning.
