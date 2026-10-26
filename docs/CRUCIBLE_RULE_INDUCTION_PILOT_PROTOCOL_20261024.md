# Crucible Rule-Induction CPU Pilot Protocol

Date: 2026-10-24  
Benchmark: `crucible_rule_induction_benchmark_20261024_v3`  
Status: frozen before training

## Arms

- Query-only baseline: the model receives the task framing and query, but no demonstrations.
- Mixed-count evidence model: each training task receives exactly one demonstration count selected by a fixed balanced schedule over 1, 2, 4 and 8 examples. Validation is evaluated at all four counts.

Both arms use the existing dense control, optimizer, tokenizer and raw-byte answer interface. Each seed runs exactly 2,000 updates with metric checkpoints at 0, 500, 1,000, 1,500 and 2,000. Only initial and final checkpoints are retained.

Seeds: `42`, `1337`.

## Validation measurements

For each count and seed, report:

- raw query exact accuracy;
- accuracy on tasks whose current demonstrations make the query identifiable;
- candidate-consistent accuracy, meaning the prediction is produced by at least one candidate rule still compatible with the supplied demonstrations;
- task count, candidate count and per-family results;
- mixed-model accuracy with demonstrations replaced by unrelated validation demonstrations.

The query-only baseline is evaluated on the same task/query records and grouped by the identifiability of the valid demonstration condition. It does not receive those demonstrations at inference.

## Frozen validation gates

Using the final 2,000-step checkpoints and aggregate results over both seeds:

1. Evidence response: mixed-model raw accuracy at 8 demonstrations must exceed its 1-demonstration accuracy by at least 5 percentage points.
2. Demonstration advantage: mixed-model identifiable-task accuracy at 8 demonstrations must exceed the query-only baseline on the same identifiable tasks by at least 5 percentage points.
3. Evidence dependence: replacing valid 8-demonstration evidence with unrelated evidence must reduce raw accuracy by at least 5 percentage points.

The adjacent-count curve and all per-seed results are reported even if a gate fails. No test result is used for checkpoint selection or gate tuning.

## Test policy

The 33 test rule instances remain sealed during training and validation. They are evaluated once, at the final checkpoints, only if all three aggregate validation gates pass. Otherwise the test split remains unopened and the pilot is recorded as a validation failure.

## Interpretation

A passing pilot supports evidence-dependent in-context rule induction under this finite benchmark. It does not establish weight updates, meta-learning, causal reasoning or autonomous experiment design. A failed pilot does not invalidate the benchmark; it directs the next diagnosis toward prompt use, representation, optimization or rule-library difficulty.
