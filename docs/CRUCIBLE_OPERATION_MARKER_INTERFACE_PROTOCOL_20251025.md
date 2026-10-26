# Crucible Single-Token Operation-Marker Experiment

Date: 2025-10-25  
Status: preregistered A0/A1 CPU comparison  

## Objective

Test whether replacing the multi-byte operation words with single-byte
operation markers improves instruction-conditioned operand selection, without
changing the model architecture, task split, optimizer, training budget or
absolute positions of the operands and answer marker.

## Conditions

| Arm | Operation encoding | Position |
|---|---|---|
| A0 | Existing `SELECT_FIRST` / `SELECT_SECOND` words | Existing S3 prompt |
| A1 | One-byte `F` / `S` markers with neutral `_` padding | Same operation field and length as the corresponding A0 prompt |

The A0 results are the frozen S3 checkpoints from
`crucible_role_selection_competence_20251025_v5`; they are not retrained.
A1 uses the same canonical and factorized interfaces, seeds 42 and 1337,
500 updates, dense 22M model and existing S3 train/validation pair split.

For A1, the first prompt uses `F` plus eleven underscores and the second uses
`S` plus twelve underscores. These lengths match the original operation-word
lengths (12 and 13 bytes respectively), preserving the subsequent operand and
answer positions. The marker itself is one raw byte under the existing byte
serializer. The padding is deliberately neutral but remains a possible
lexical cue; the result is therefore an encoding intervention, not a proof
that all lexical differences have been removed.

## Primary metrics

Report:

- training and validation exact accuracy;
- separate `SELECT_FIRST` and `SELECT_SECOND` accuracy;
- unequal-pair accuracy;
- paired conditional correctness: both operations correct for the same pair;
- paired prediction switching: whether the two operation prompts produce
  different outputs for the same unequal pair;
- paired correct-and-switch accuracy.

The existing competence gates remain unchanged: at least 95% training exact
accuracy, 90% overall validation exact accuracy and 90% unequal validation
accuracy.

## Controls and invariants

- A0 source checkpoint hashes and prediction hashes are recorded.
- A1 uses the same frozen train/validation pairs and no sealed test data.
- Initial dense-model state digests and training example order are recorded.
- A1 prompt lengths and operand/answer-marker byte positions are checked
  against the corresponding A0 prompts.
- The deterministic symbolic role-selection oracle remains unchanged.
- No FiLM, recurrence, extra experts, memory, recursion or neural selector is
  introduced.

## Interpretation boundary

Improved paired conditional correctness would support an interface-level
explanation for the S3 failure. It would not establish a new architecture or
general instruction-following capability. A null result would motivate the
pre-registered positional matrix or a more direct conditional-routing
intervention, not an immediate architecture expansion.
