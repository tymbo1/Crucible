# Crucible Authority Counterfactual Diagnostic v1

Date: 2026-10-26  
Status: preregistered read-only checkpoint diagnostic

## Objective

Determine whether the verified v2 parser checkpoints use the authority
designation to select between two otherwise identical operation cues.

This is a counterfactual evaluation only. No model is retrained, no
architecture or tokenizer is changed, no new split is generated, and the
sealed test set is not accessed.

## Frozen source and checkpoints

Use the frozen validation records, data audit, checkpoints and evaluator from
`experiments/crucible_authority_aware_role_binding_20251025_v2`.

Evaluate A0, A1, A2, A3 and the E3 deterministic oracle across canonical and
factorized interfaces and seeds 42 and 1337. A0 uses the frozen v3 pointer
checkpoint; A1--A3 use the completed v2 checkpoints. Checkpoint hashes must be
verified before inference.

## Counterfactual construction

For every v2 validation operand record, construct two operation orders:

- `FS`: `SELECT_FIRST` is the first cue and `SELECT_SECOND` is the second;
- `SF`: `SELECT_SECOND` is the first cue and `SELECT_FIRST` is the second.

For each operation order, construct two inputs with the same operands, both
operation tokens, token positions, distractor values, layout and context:

- `authority=EARLIER`, whose target is the first operation cue;
- `authority=LATER`, whose target is the second operation cue.

The `LATER` label is padded with two neutral spaces so both authority labels
have equal byte length. The parser's authority matcher still recognizes the
label, and the diagnostic verifies that both operation positions and operand
positions are identical within every counterfactual pair. The only semantic
change is the authority designation and therefore the target operation.

Use all seven v2 layouts:

`AUTHORITY_NEAR`, `AUTHORITY_EARLY`, `AUTHORITY_MIDDLE`,
`UNSEEN_REVERSED_AUTHORITY`, `UNSEEN_DIGIT_DISTRACTOR`,
`UNSEEN_NEAREST_DECOY`, and `UNSEEN_CONFLICTING_CUES`.

The nearest-decoy layout is included as a counterfactual here even though the
original v2 stress evaluation fixed it to earlier authority. Its paired
counterfactual result must be reported separately from the original v2
nearest-decoy result.

## Measurements

For each row report classifier and pointer outputs, selected operation, cue
index, output, correctness, operation-pointer correctness and all target
positions. For each counterfactual pair report:

- whether the selected operation changed;
- whether the selected cue position changed;
- whether the change was correct in both directions;
- whether the final output changed;
- whether both final outputs were correct, with unequal operands scored as the
  discriminating subset;
- first-cue versus second-cue preference;
- `SELECT_FIRST` versus `SELECT_SECOND` preference;
- breakdown by layout, operation order, authority side and interface/seed.

The primary selection metric is the pointer-selected operation for A0, A2 and
A3. A1 is reported as a tuple/classifier negative control because its pointer
loss was intentionally inactive. The oracle is reported as a solvability
control.

## Integrity requirements

- Validation source record IDs must remain disjoint from the frozen v2
  training IDs.
- Every counterfactual pair must have identical operand values, operation
  positions, operand positions, operation order and non-authority content.
- Counterfactual target operations must reverse exactly when the authority
  label reverses.
- No prediction row may be duplicated or omitted.
- The deterministic executor must agree with all valid selected outputs.
- Checkpoint hashes and the v2 finalization state must verify.

## Interpretation rules

- A pointer that never changes when authority changes supports an
  authority-insensitive selection shortcut.
- A pointer that changes but is wrong supports authority sensitivity without
  reliable referent binding.
- Consistent correct two-way flips on unseen layouts support transferable
  authority binding within this synthetic task.
- Position or operation preferences are descriptive diagnostics, not proof of
  an internal mechanism.
- Equal operands cannot establish output flips because both operations return
  the same digit; pointer metrics remain valid for them.

