# Crucible topology symbolic-hypothesis audit

Read-only independent replication of the symbolic graph mechanisms and checkpoint-level falsification tests.

## Findings

- The structural claims replicate: the prism has diameter 4, λ₂ 1.000, no bridges and mean held-out resistance 0.933; the diameter-5 graph has diameter 5, λ₂ 0.181, bridge (6,8) and mean held-out resistance 2.228.
- The diffusion ratio at time 4 is approximately 18.7× in favour of the prism's faster mixing.
- Exact modular path composition succeeds on both graphs for moduli 5, 11 and 97: 8,000/8,000 trials per graph/modulus.
- Effective-resistance predictions reproduce the global least-squares noise simulations. This validates the linear reconstruction mathematics, not a neural mechanism.
- The random 500-graph control reproduces the reported diameter distribution and shows diameter-5 graphs can be bridged or non-bridged.

## Neural checkpoint tests

The eight held-out pairs are too few to distinguish shortest-path distance from effective resistance: the predictors remain correlated. Pair-level error correlations are seed-dependent and do not establish H1 or H2. Seven of eight diameter-5 pairs cross the bridge, but the observed checkpoint errors do not consistently increase for those pairs. Pair 0–9 is better under the diameter-5 graph in both seeds, consistent with its lower distance and resistance there; pair 0–8 is slightly worse despite a shorter D5 path, consistent with its higher D5 resistance. Both are exploratory observations, not causal tests.

The original topology run had only one training-competent arm. Therefore all checkpoint comparisons remain diagnostic, not validated transfer evidence.

See `RESULTS.json` for exact values and `CHECKPOINT_PAIR_ANALYSIS.csv` for denominators.
