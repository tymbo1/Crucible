# Crucible Two-Candidate Matched Context Probe v1

Date: 2026-10-27
Status: frozen before execution
Branch: `research/two-candidate-forensics-20261027`

## Objective

Separate candidate-pair-regime effects from source-context effects in the
accepted two-candidate matcher. This is a checkpoint-only, exploratory probe.
It does not train, modify or replace any checkpoint and it is not an
independent sealed validation benchmark.

## Motivation

The preceding 2x2 checkpoint audit showed that compact serialization did not
restore the recombined validation regime. However, the original training and
validation populations differ in both candidate sets and `(a,b)` operand
contexts. This probe evaluates both candidate regimes on the same contexts so
those two factors are not changed together.

## Frozen inputs

Use the two accepted cyclic checkpoints from the 2026-10-27 competence gate:

- `DENSE_CYCLIC_SEED42`
- `DENSE_CYCLIC_SEED1337`

Use the ten two-candidate source contexts, without changing their values:

- Training-source contexts: `(1,6)`, `(2,7)`, `(3,8)`, `(4,9)`, `(1,8)`,
  `(2,9)`.
- Validation-source contexts: `(0,5)`, `(1,7)`, `(2,8)`, `(3,9)`.

Evaluate both regimes on every context:

- Familiar candidate sets: `(0,1)`, `(1,0)`, `(2,3)`, `(3,2)`, `(4,5)`,
  `(5,4)`, `(6,7)`, `(7,6)`.
- Recombined candidate sets: `(0,3)`, `(3,0)`, `(1,4)`, `(4,1)`, `(2,5)`,
  `(5,2)`, `(3,6)`, `(6,3)`.

For each context and candidate set, evaluate authorities equal to the first
candidate, the second candidate and `9` (the abstention case). Serialize all
rows with the fixed `TRAIN_COMPACT` layout. The oracle target is deterministic:
the matching candidate slot, or slot `8` when authority `9` is absent.

This produces 240 rows total per regime (144 on training-source contexts and
96 on validation-source contexts), and 480 rows per checkpoint, with 80 paired
counterfactual groups per regime. The generated rows are diagnostic probes;
they must not be described as fresh held-out validation examples.

## Controls and invariants

- Load the archived checkpoints without optimizer state or parameter writes.
- Use the existing dense `SHARED_SCORER` evaluator and tokenizer.
- Verify every generated target against the identifier-match oracle.
- Verify candidate spans and authority spans after serialization.
- Preserve the original source context values and candidate-set definitions.
- Do not access the sealed test set.
- Perform no training, gradient calculation or checkpoint modification.
- Record source-context membership so results can be stratified by training- or
  validation-source context.

## Primary comparisons

For each seed and source-context stratum, report:

1. Familiar-regime versus recombined-regime row accuracy.
2. Paired counterfactual correctness and predicted authority flips.
3. Abstention accuracy.
4. Accuracy by candidate set, authority and context.
5. Error concentration by regime and context.

The matched regime contrast is descriptive. It does not establish that the
model learned a general identifier-comparison algorithm, because the
checkpoint was trained only on the original task distribution.

## Interpretation rule

- If familiar pairs outperform recombined pairs within both context strata,
  candidate-pair familiarity is implicated independently of source context.
- If both regimes degrade mainly on validation-source contexts, context shift
  is implicated.
- If the effect differs by context stratum, report an interaction rather than
  assigning a single cause.
- If counterfactual flips remain below target flips, the model is not reliably
  using authority to select between both candidates, even where row accuracy is
  higher.

These are evidence labels for this probe only; they are not architectural
claims.

## Stop boundary

Stop if checkpoint loading, source-context audit, oracle verification or parser
span validation fails. Do not substitute invented predictions or launch a
training run to repair the probe.
