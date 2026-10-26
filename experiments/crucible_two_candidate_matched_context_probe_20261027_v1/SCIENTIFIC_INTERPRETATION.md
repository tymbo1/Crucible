# Scientific Interpretation

This checkpoint-only probe evaluates the same source operand contexts with
both the familiar and recombined candidate-set regimes. It therefore tests
whether the candidate-regime effect survives when the original `(a,b)` context
shift is removed within each comparison.

The probe did not train, modify checkpoints or access sealed test data. It is
exploratory and generated from the canonical source context populations, not a
new independent validation benchmark.

## Result summary

- Seed 42, TRAIN_SOURCE: familiar 144/144 versus recombined 111/144 (row-accuracy delta -0.229).
- Seed 42, VALIDATION_SOURCE: familiar 96/96 versus recombined 75/96 (row-accuracy delta -0.219).
- Seed 1337, TRAIN_SOURCE: familiar 144/144 versus recombined 121/144 (row-accuracy delta -0.160).
- Seed 1337, VALIDATION_SOURCE: familiar 96/96 versus recombined 80/96 (row-accuracy delta -0.167).

## Interpretation rule

If recombined performance is lower within both context strata, the result is
consistent with a candidate-pair-regime effect independent of the original
source-context shift. If both regimes degrade mainly on validation-source
contexts, context shift is implicated. If the pattern differs by stratum, the
correct description is an interaction. These are behavioral descriptions, not
mechanistic or architectural conclusions.

See `SUMMARY.json` for denominators, paired correctness and counterfactual
flip counts; see `DATA_AUDIT.json` for the source-context construction.
