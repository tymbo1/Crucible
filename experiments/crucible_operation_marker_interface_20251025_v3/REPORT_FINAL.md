# Crucible Single-Token Operation-Marker Experiment — Final Report

CPU-only A0/A1 encoding comparison. A0 is the frozen S3 baseline; A1 is retrained with length-preserving one-byte operation markers. No sealed-test access.

## Results

| Interface | Seed | A0 validation | A1 validation | A1 unequal validation | A1 gate | A1 paired both correct | A1 correct-and-switched |
|---|---:|---:|---:|---:|---|---:|---:|
| canonical | 42 | 60.00% | 60.00% | 50.00% | FAIL | 0/8 | 0/8 |
| canonical | 1337 | 55.00% | 60.00% | 50.00% | FAIL | 0/8 | 0/8 |
| factorized | 42 | 60.00% | 60.00% | 50.00% | FAIL | 0/8 | 0/8 |
| factorized | 1337 | 50.00% | 50.00% | 50.00% | FAIL | 0/8 | 0/8 |

## Interpretation

The primary result is paired conditional correctness on the same unequal operand pair, not aggregate accuracy alone. A1 changes only the operation encoding while preserving the corresponding prompt length and operand/answer positions. The neutral padding remains a possible lexical cue, so this is an encoding intervention rather than a complete tokenizer ablation.

The original S3 baseline remains preserved in its source experiment. This experiment does not run the positional A2/A3 matrix; that is the follow-up if A1 does not resolve the conditional-execution failure.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_OPERATION_MARKER_INTERFACE_PROTOCOL_20251025.md)
- [Summary](SUMMARY.json)
- [A1 predictions](PREDICTIONS.csv)
- [A0/A1 comparison](COMPARISON.csv)
- [Machine-readable metrics](RESULTS.json)
- [Finalization](FINALIZATION.json)
