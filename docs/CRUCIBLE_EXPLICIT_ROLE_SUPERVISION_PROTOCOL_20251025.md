# Crucible Explicit Role-Supervision Gate

Date: 2026-10-25  
Status: preregistered CPU-only intervention

## Research question

Does explicit supervision of operand and operation locations improve transfer
of role binding beyond tuple-only classification?

## Conditions

| Condition | Binding mechanism | Execution |
|---|---|---|
| E0 | Frozen tuple-only parser from the prior gate | Deterministic role executor |
| E1 | New parser with tuple heads plus explicit A/B/operation pointer losses; classifier tuple output | Deterministic role executor |
| E2 | Same role-supervised parser; pointer locations decoded into the bound tuple | Deterministic role executor |
| E3 | Oracle binding | Deterministic role executor |
| E4 | Frozen raw B0 end-to-end dense control | Dense model output |

E1 and E2 share the same newly trained checkpoints. E1 uses the parser's
10-way operand and 2-way operation heads. E2 uses the predicted positions of
the A digit, B digit and authoritative operation marker, decoding the values
at those positions. Pointer accuracy and tuple accuracy are reported
separately.

## Training

The parser backbone and tuple heads are the same as the prior role-binding
parser. The only intervention is three token-position supervision losses for
the A digit, B digit and authoritative operation marker, weighted equally with
the tuple loss. Training uses the immutable three-layout training records,
the existing B0 diverse/shuffled 500-update schedule, effective batch size 12,
and seeds 42 and 1337 for both interfaces.

## Evaluation controls

Evaluate training and validation pairs on the three trained layouts and the
two prior unseen layouts. Add two diagnostic layouts withheld from training:

- an irrelevant-digit/neutral-context distractor;
- an early competing operation cue followed by the authoritative operation.

Validation includes reversed operand pairs from the existing split. Report:

- A-position, B-position and operation-position accuracy;
- classifier tuple and pointer tuple accuracy;
- final output accuracy and paired conditional correctness;
- results by layout and split.

## Boundaries

- E0, E3 and E4 are frozen controls; no executor is retrained.
- No sealed-test access, persistent memory, recursion or architecture search.
- Oracle binding establishes task sufficiency, not autonomous neural binding.
- A positive result establishes benefit from this supervision under this task;
  it does not prove general position-independent reasoning.
