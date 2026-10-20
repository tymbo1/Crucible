# GPU Neural SSRE v5 — concurrency feasibility result

Date: 2026-10-18

## Decision

The tested CUDA-stream fan-out design is rejected as an optimization for this
workload. It was slower than the accepted V2 sequential rank-preserving grouped
reference in every fixture and at every stream count.

This closes the tested stream-concurrency route. It does not establish that all
GPU concurrency or GPU-resident work engines are unhelpful.

## Tested system

- Hardware: NVIDIA GeForce RTX 3060 Ti, compute capability 8.6
- PyTorch CUDA runtime: 13.0
- Expert count: 16
- Top-2-shaped routing assignments
- Token counts: 512, 2,048 and 4,096
- Fixtures: balanced and highly skewed
- Variants: V2 sequential grouped reference; CUDA-stream fan-out with 2, 4
  and 8 streams
- Timing: one warm-up and three timed forward trials per case

The stream implementation used independent temporary outputs for each active
expert/rank segment and ordered accumulation on the default stream. It avoided
output races and preserved the same deterministic routing computation.

## CUDA-event timings

Times are mean forward CUDA-event milliseconds. Percentages are relative to the
reference in the same fixture.

| Tokens | Fixture | Reference | 2 streams | 4 streams | 8 streams | Best stream slowdown |
|---:|---|---:|---:|---:|---:|---:|
| 512 | balanced | 5.726 | 6.681 (+16.7%) | 7.087 (+23.8%) | 6.587 (+15.0%) | +15.0% |
| 512 | high-skew | 5.685 | 6.258 (+10.1%) | 6.437 (+13.2%) | 6.569 (+15.5%) | +10.1% |
| 2,048 | balanced | 5.591 | 7.115 (+27.3%) | 6.617 (+18.3%) | 7.416 (+32.6%) | +18.3% |
| 2,048 | high-skew | 5.605 | 6.857 (+22.3%) | 7.083 (+26.4%) | 7.132 (+27.2%) | +22.3% |
| 4,096 | balanced | 6.253 | 7.169 (+14.6%) | 7.134 (+14.1%) | 8.781 (+40.4%) | +14.1% |
| 4,096 | high-skew | 5.603 | 6.768 (+20.8%) | 7.349 (+31.2%) | 7.825 (+39.7%) | +20.8% |

Across all 18 stream observations, the mean slowdowns were 18.64% for two
streams, 21.17% for four streams and 28.43% for eight streams. The best
individual stream observation was still 10.08% slower than its reference.

## Integrity

- All 18 stream trials produced finite outputs.
- All stream outputs passed the declared `5e-3` forward tolerance.
- The recorded maximum output absolute delta versus the reference was `0.0`
  in every reported stream case.
- No backward, optimizer or full-model equivalence test was run because the
  forward feasibility gate was negative.
- No frozen Crucible artifact, checkpoint, manifest or RunPod job was modified.

## Interpretation

The independent expert/rank FFN calls expose a legal dependency-level
concurrency opportunity, but on this RTX 3060 Ti the tested stream fan-out adds
stream management, temporary-output and ordered-accumulation overhead that
outweighs any overlap. Increasing the stream count worsened the average result.

The result does not justify training a neural scheduler for this decision
space. A learned selector cannot recover a benefit that was not present in the
tested candidate execution plans. Any future GPU Neural SSRE work would need a
different, genuinely competitive execution engine and a new preregistered
feasibility gate.

## Artifact references

The raw result and hash index are in `preflight_20261018/`:

- `concurrency_preflight_summary.json`
- `concurrency_preflight_hashes.json`

