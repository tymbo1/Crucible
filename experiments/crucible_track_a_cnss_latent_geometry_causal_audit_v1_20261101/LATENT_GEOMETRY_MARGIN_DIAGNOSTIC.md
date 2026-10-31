# CNSS Latent Geometry Margin Diagnostic v1

This is a read-only analysis of the existing CNSS checkpoints.
No training, parameter updates, workspace reconstruction, or GPU execution occurred.
The registered direction is A then B only; reverse B then A was not evaluated.

## Margin definition

For each B output bit, the target-signed logit is positive when that bit is correct.
The row margin is the minimum target-signed logit across all eight bits.
Therefore, positive row margin and exact-match accuracy are equivalent under this decoder.

## Pooled results

| Condition | Exact accuracy | Mean displacement | Mean min margin | Pearson displacement/min margin | Spearman displacement/min margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| lambda_0.1 | 0.8242 | 5.1385 | 3.0552 | -0.4102 | -0.3616 |
| lambda_1.0 | 0.9935 | 2.7798 | 7.0095 | -0.6177 | -0.6365 |

## Interpretation boundary

This diagnostic tests whether displacement magnitude tracks B's output margin in the saved A->B path.
It does not establish that displacement causes the decision change, and it does not evaluate B->A, longer chains, or new training.
