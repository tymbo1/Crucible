# CNSS Directional-Gradient Diagnostic v1

This is a read-only analysis of the existing CNSS checkpoints.
No training, parameter updates, workspace reconstruction, or GPU execution occurred.
The registered direction is A then B only; reverse B then A was not evaluated.

The receiver gradient is evaluated at the canonical intermediate state.
The canonical-bottleneck projection uses the canonical minimum-margin bit; the worst-bit projection takes the most margin-reducing first-order direction across all eight output bits.
Matched random directions use the same per-state error norm and 32 independent directions.

## Pooled results

| Condition | Exact accuracy | Mean worst-bit harmful projection | Mean worst-bit random percentile | Pearson worst projection/margin | Spearman worst projection/margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| lambda_0.1 | 0.8242 | 1.4695 | 0.9855 | -0.5754 | -0.5857 |
| lambda_1.0 | 0.9935 | 1.7189 | 0.9812 | -0.6236 | -0.6518 |

The all-bit local linearization produced these exact-match diagnostics:

- `lambda_0.1`: linearized exact accuracy `0.8216`, agreement with actual outcomes `0.8984`.
- `lambda_1.0`: linearized exact accuracy `0.9948`, agreement with actual outcomes `0.9935`.

## Interpretation boundary

A high harmful projection or random-direction percentile indicates local alignment with a margin-reducing direction, not proof that the full nonlinear path is caused by that local derivative.
The worst-bit random percentile is an envelope statistic across eight output bits and is not used as standalone causal evidence.
This diagnostic does not evaluate B->A, longer chains, or new training.
