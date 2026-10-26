# Crucible composition-interface diagnostic results

Date: 2026-10-22  
Status: CPU-only oracle/alignment audit and positive-control overfit completed.

## Question

The learnability ladder showed that the dense control mastered the atomic task
but failed two-operation composition even on its training split. This bounded
diagnostic separated an interface or target defect from an optimization or
coverage problem without changing the neural architecture.

The diagnostic performed four checks:

1. Independently parse every composition prompt and recompute the modulo-ten
   answer in left-to-right order.
2. Check target labels, operation ordering and byte-token alignment.
3. Train a fresh dense control on eight fixed composition examples for 500
   updates.
4. Record gradient and answer-prediction diagnostics during the overfit run.

The model was the same 21,968,640-parameter dense active-compute control used
in the ladder. No E4 arm was run. The test split was inspected only by the
independent generator/oracle audit; it was not evaluated by the model or used
for selection.

## Oracle and serialization audit

The audit covered 320 composition records per seed across train, validation,
held-out and sealed test splits: 640 records total. It independently parsed
the textual `start`, `step1`, and `step2` fields and checked:

- left-to-right operation order;
- modulo-ten arithmetic;
- recorded target versus independently computed target;
- operation metadata consistency;
- one answer byte followed by EOS in the serialized sequence;
- unique record identifiers within each split.

Result: **0 audit errors** for both seeds. The audit confirms that the
composition generator, target construction and byte alignment are internally
consistent. It does not establish that the task is easy to learn at scale.

## Tiny-set overfit result

The positive control used eight fixed training examples and 500 optimizer
updates. Exact accuracy includes the answer byte and EOS under the existing
teacher-forced evaluation contract.

| Seed | Step 50 exact | Step 100 exact | Step 500 exact | Final target loss | Final answer probability | Final gradient norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 75.0% | 100.0% | 100.0% | 6.56e-7 | 0.999999 | 6.58e-6 |
| 1337 | 37.5% | 100.0% | 100.0% | 5.74e-7 | 0.999999 | 7.16e-6 |

The target token became top-ranked for all eight examples at step 100 and
remained so through step 500. Both seeds passed the configured positive-control
criterion of at least 99% exact accuracy.

## Interpretation

This rules out a basic target-construction, serialization or total inability to
fit composition explanation for the earlier ladder failure. The current model
and loss can memorize a small set of two-step examples, and the gradient path
is active.

It does **not** show that the model has learned a compositional procedure. The
positive control is deliberately tiny and contains no held-out generalization
claim. The unresolved possibilities now include:

- insufficient updates for the 128-example composition set;
- optimization interference across the operation combinations;
- weak representation of intermediate state when many examples are mixed;
- inadequate training coverage or batch ordering; or
- failure to generalize operation composition despite memorization.

The original full composition ladder result remains frozen and unchanged. No
full composition rerun was performed by this diagnostic.

## Decision and next gate

The adapter is sufficiently functional to continue composition research, but
the full composition learnability gate remains failed. The next bounded CPU
experiment should vary one factor at a time: task-set size, update budget and
operation-family coverage, with validation kept separate and the test split
sealed. An optional intermediate-state diagnostic may be used to localize the
optimization problem, but it must not be treated as the final composition
benchmark.

Do not begin E4 comparison, neural recursive-reasoning evaluation, GPU
training or cloud execution until a dense composition control passes the
original train/validation learnability criterion.

## Frozen artifact locations

- Seed 42 accepted run: `experiments/crucible_cognitive_composition_diagnostic_20261022_seed42_v3/`
- Seed 1337 accepted run: `experiments/crucible_cognitive_composition_diagnostic_20261022_seed1337_v3/`
- Diagnostic runner: `tools/run_crucible_composition_diagnostic_20261022.py`
- Prior ladder result: `docs/CRUCIBLE_COGNITIVE_LEARNABILITY_LADDER_RESULTS_20261022.md`

Both accepted runs contain seven artifact hashes; all were independently
recomputed with zero mismatches.
