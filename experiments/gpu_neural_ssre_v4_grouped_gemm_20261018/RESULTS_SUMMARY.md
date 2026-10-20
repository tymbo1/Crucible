# GPU Neural SSRE v4 — grouped-GEMM execution-engine result

Date: 2026-10-18  
Device: NVIDIA GeForce RTX 3060 Ti  
Scope: dispatch-only E16 layer-0 execution fixture

## Outcome

The grouped-GEMM prototype passed the integrity gate across all eight tested
workloads:

- 16 experts and top-2 routing;
- 512, 1,024, 2,048 and 4,096 routed tokens;
- balanced and highly skewed assignments;
- matching GPU routing metadata and active-group offsets;
- finite losses;
- forward output and expert-gradient errors within the declared tolerances.

It did not pass the performance gate. The grouped-GEMM engine was slower than
the accepted V2 CPU-offset grouped reference in every workload.

## Timed results

Mean forward-plus-backward CUDA time:

| Tokens | Fixture | Reference | Grouped GEMM | Difference |
|---:|---|---:|---:|---:|
| 512 | balanced | 10.109 ms | 17.431 ms | +72.4% |
| 512 | high skew | 10.052 ms | 17.589 ms | +75.0% |
| 1,024 | balanced | 11.452 ms | 19.167 ms | +67.4% |
| 1,024 | high skew | 11.015 ms | 17.929 ms | +62.8% |
| 2,048 | balanced | 10.456 ms | 20.467 ms | +95.7% |
| 2,048 | high skew | 11.825 ms | 19.415 ms | +64.2% |
| 4,096 | balanced | 12.209 ms | 22.917 ms | +87.7% |
| 4,096 | high skew | 13.463 ms | 22.624 ms | +68.1% |

Mean slowdown was 74.15%; the range was 62.77–95.74%.

## Interpretation

This prototype removes the Python loop over expert matrix operations, but it
also pays for BF16-compatible padding from expert width 3,447 to 3,448,
per-forward packed weight construction, grouped-kernel launch overhead and
grouped backward costs. On this Ampere GPU and these E16 shapes, those costs
outweigh the reduction in expert launches.

The maximum output drift was below `3.5e-3`, and gradient drift was below
`1.5e-7`, so the candidate is numerically within the current diagnostic
tolerance. That does not make it an exact mathematical identity: the backend
uses a different BF16 execution path and padded matrix shapes.

## Decision

Do not integrate this grouped-GEMM implementation into full Crucible training,
and do not train a neural selector over it. V2's deterministic grouped engine
remains the faster reference for this workload.

The next GPU-engine work, if resumed, should target a genuinely different
implementation—such as cached packed weights with separately measured
inference cost, a fused SwiGLU/grouped kernel, or a custom persistent
device-side queue. Any such work must pass the same equivalence and paired
timing gates.

