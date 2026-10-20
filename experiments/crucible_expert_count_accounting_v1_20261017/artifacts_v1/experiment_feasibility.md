# Crucible expert-count feasibility and recommended experiment

## Fairness conclusion

Fair comparisons are technically feasible as a design study, but the current runner is not yet a drop-in implementation for E=1 because its routing path hard-codes top-2. The future implementation must generalize top-k and Branch-B auxiliary-loss normalization before any variant is trained.

The strongest matched comparisons are:

1. Fixed active expert compute: E=1/K=1/d_ff=6894, E=2/K=2/d_ff=3447, and the unchanged E=4/K=2/d_ff=3447 reference. This holds expert arithmetic approximately exactly; total parameter capacity differs by design.
2. Fixed total expert capacity: E=1/K=1/d_ff=13788, E=2/K=2/d_ff=6894, and E=4/K=2/d_ff=3447. This holds total expert parameters exactly; active expert compute is approximately 2x the reference for E=1/2.
3. Fixed geometry: useful as a capacity ablation, not as a fair compute or parameter comparison.

## Main confounders

- E=1 cannot use top-2 and therefore changes routing semantics and Branch-B auxiliary-loss behavior.
- E=2/top-2 activates every expert, so it has no sparse expert selection even when its active expert FLOPs are matched to E=4/top-2.
- Router arithmetic, top-k comparisons, dispatch/gather/scatter, and Python mask overhead vary with E and K; they are not captured by expert GEMM FLOPs alone.
- The analytical training FLOPs and activation memory are estimates, not GPU measurements. Kernel choice, allocator fragmentation, and padding alter realized cost.
- Fixed total parameters and fixed active compute deliberately answer different questions; neither alone identifies the full causal effect of expert count.
- The dataset/model may be under-complex for specialization; a TinyStories result does not establish behavior on diverse or harder data.

## Recommended minimal experiment

First implement and preflight a generalized top-k model without changing the E=4 reference path. Then run the crossed six-condition design: the three fixed-total rows and three fixed-active-compute rows, sharing the unchanged E=4 reference where identical. Use at least two seeds for an exploratory minimum; three seeds are preferable if the differences are small. Keep the same packed manifest, one corpus pass, optimizer schedule, checkpoint intervals, and validation split.

Primary metrics:
- validation NLL and bits per raw byte;
- total/active parameter counts and realized peak VRAM;
- measured tokens/sec and optimizer-step time;
- forward/backward wall time separated from validation;
- router CV, load ratio, top-k entropy, and dropped-token count (expected zero);
- gradient norm/clipping and nonfinite events.

Acceptance criteria should be frozen before training: designated fixed-total rows within 0.01% total parameters; designated fixed-active rows within 0.01% expert arithmetic FLOPs; exact zero capacity drops; identical data/schedule/seed protocol; and conclusions reported as seed-level estimates rather than formal equivalence claims with only two seeds.

## RTX 3060 Ti assessment

The widest analytical training-memory row is `fixed_total_parameters/E2` at approximately `776.4` MiB under the stated proxy. The lowest estimated forward-cost row is `fixed_geometry/E1` at `0.519x` the reference. These are not measured peaks. Historical Crucible runs used roughly 2.3 GiB on the 8 GiB RTX 3060 Ti, leaving substantial headroom, but each future variant must record actual peak allocation. No benchmarking was performed for this study.

No training, checkpoint creation, GPU benchmark, or rental is authorized by this accounting artifact.
