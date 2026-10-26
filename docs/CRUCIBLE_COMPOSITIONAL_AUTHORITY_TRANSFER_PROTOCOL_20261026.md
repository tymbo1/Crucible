# Crucible Compositional Authority-Transfer Gate v1

Date: 2026-10-26  
Status: preregistered CPU gate

## Objective

Test whether the causally valid final-decision mechanism demonstrated by S4 can
select among more than two candidate operations when authority is expressed as
a relation to candidate content, rather than only as an earlier/later cue.

The gate uses a dynamic candidate-pointer readout. It is trained on three
candidates and evaluated both on held-out three-candidate layouts and on a
four-candidate transfer condition. The readout has a separate abstain outcome
for cases where the authority rule has no matching candidate.

## Task

Candidate operation markers are:

- `F`: return operand A;
- `S`: return operand B;
- `A`: return `(A+B) mod 10`;
- `D`: return `(A-B) mod 10`.

Authority rules are `EARLIEST`, `LATEST`, `MATCH_A`, `MATCH_D` and `NONE`.
`MATCH_A` and `MATCH_D` select the candidate carrying the matching operation
marker; if that marker is absent, the correct result is abstention. `NONE`
always requires abstention. The primary target is the candidate pointer, not
the numeric output, so distinct candidate positions remain distinguishable
even when two operations happen to produce the same digit.

## Data partitions

- Training: three candidates, compact layout, deterministic operand pairs and
  all candidate-subset permutations.
- Validation-3: three candidates with held-out operand pairs and padded or
  reordered layouts.
- Transfer-4: four candidates containing all operation markers, held-out
  operand pairs and a transfer layout. This candidate count is absent from
  training.

Counterfactual pairs keep operands, candidate order and layout fixed while
  changing only the authority rule. Position pairs use `EARLIEST`/`LATEST`;
  relation pairs use `MATCH_A`/`MATCH_D` when both candidates are present.

## Frozen controls and limits

- Existing 22M dense backbone family and tokenizer remain unchanged.
- The new component is a dynamic final decision pointer with shared candidate
  scoring and one abstain logit; this is the intended compositional readout,
  not a claim that the original two-class S4 head scales automatically.
- Canonical and factorized operand interfaces, seeds 42 and 1337, 500 updates,
  effective batch size 12, existing optimizer and CPU execution are used.
- No sealed test set, persistent memory, symbolic execution assistance or
  autonomous rule induction is used.

## Metrics and gates

Report training selection accuracy, validation-3 selection accuracy,
four-candidate transfer accuracy, abstention accuracy, output accuracy and
counterfactual flips/two-way correctness with denominators.

Require at least 95% training pointer accuracy before treating validation-3 or
transfer results as clean evidence. Transfer results are reported separately
from same-count validation. A successful transfer must include correct
counterfactual behavior, not only marginal row accuracy.

## Interpretation boundaries

Success would demonstrate a bounded dynamic-pointer composition result under
this synthetic authority task. It would not establish arbitrary instruction
hierarchy, general reasoning, autonomous learning or persistent procedure
selection. Failure on the four-candidate condition would be a candidate-count
transfer failure, not evidence that the two-candidate S4 mechanism is invalid.
