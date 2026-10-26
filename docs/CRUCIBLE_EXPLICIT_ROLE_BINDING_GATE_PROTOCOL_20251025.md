# Crucible Explicit Role-Binding Gate

Date: 2026-10-25  
Status: preregistered CPU-only decomposition study

## Research question

Is the conditional-execution failure primarily caused by extracting operand
roles and operation identity from variable layouts, or by the downstream
executor after those roles are available?

## Conditions

| Condition | Parser/binding | Executor | Source |
|---|---|---|---|
| C0 | Oracle metadata parser | Frozen successful A3 dense executor | A3 checkpoints |
| C1 | Learned neural parser | Same frozen A3 dense executor | New parser checkpoints + A3 |
| C2 | Learned neural parser | Deterministic role executor | New parser checkpoints |
| C3 | No explicit parser | Frozen raw end-to-end dense control | B0 diverse/shuffled checkpoints |

C0 canonicalizes every raw layout into the exact successful A3 prompt format
before execution. C1 uses the learned parser's predicted operand digits and
operation to construct that same A3 prompt. C2 executes the learned parser's
predicted tuple directly. C3 consumes the raw layout, using the existing B0
end-to-end checkpoints without retraining.

## Learned parser

The parser uses the existing 22M dense backbone with three lightweight heads
for operand A, operand B and operation identity. It is trained only on the
immutable expanded training records (three trained layouts per pair), using
the existing B0 diverse/shuffled 500-update schedule, effective batch size 12,
and seeds 42 and 1337. The parser target is the structured tuple
`(operand_a, operand_b, operation)`; it is not given the selected output as an
auxiliary target.

Parser competence is reported separately for each field and for the complete
three-field tuple. No validation labels are used for training or persistent
state.

## Evaluation

Evaluate all conditions on:

- training pairs in each trained layout;
- validation pairs in each trained layout;
- validation pairs in the two previously unseen layouts;
- parser field accuracy, complete tuple accuracy, output-token accuracy and
  paired conditional correctness.

The primary decomposition is:

- C0 success establishes that canonicalization is sufficient for the frozen
  executor.
- C1 versus C0 measures the cost of learned parsing while holding execution
  fixed.
- C2 versus C1 separates parser errors from executor errors.
- C3 is the raw end-to-end reference.

## Controls and boundaries

- Existing A3 and B0 checkpoints are loaded read-only.
- Only the learned parser is newly trained; no Crucible executor is retrained.
- No architecture search, memory, recursion, symbolic critique or sealed-test
  access.
- Oracle parsing and oracle execution are diagnostic controls, not evidence of
  autonomous neural role binding.
- A learned parser passing does not establish position-independent reasoning;
  it establishes structured tuple extraction under this task distribution.
