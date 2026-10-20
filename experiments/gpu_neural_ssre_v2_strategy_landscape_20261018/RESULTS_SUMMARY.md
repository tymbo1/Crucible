# GPU Neural SSRE v2 — strategy landscape result

Date: 2026-10-18  
Device: NVIDIA GeForce RTX 3060 Ti  
Runtime: PyTorch 2.12.0+cu130 / CUDA 13.0

## Outcome

V2 passed its execution and integrity gates. It found a large, stable
deterministic improvement from rank-preserving GPU grouping, but it did not
find a strong enough strategy crossover to justify a neural scheduler yet.

The correct current default is still the deterministic rank-preserving grouped
engine from V1.

## Full-model matrix

The matrix contained six frozen cells:

* E16, top-2, Branch B;
* context 128, 256 and 512;
* seeds 42 and 1337;
* two warm-up steps, eight timed optimizer steps, two repetitions;
* original, rank-preserving, largest-first and rank-major execution.

Every strict candidate passed logits, loss, gradient, post-step model-state
and post-step optimizer-state checks. Optimizer state was independently hashed
from its serialized slot tensors and optimizer-group metadata.

| Context | Seed | Original wall | Rank-preserving | Largest-first | Rank-major |
|---:|---:|---:|---:|---:|---:|
| 128 | 42 | 12.173 s | 7.645 s | 7.643 s | 7.644 s |
| 128 | 1337 | 12.205 s | 7.681 s | 7.675 s | 7.666 s |
| 256 | 42 | 11.478 s | 7.227 s | 7.202 s | 7.232 s |
| 256 | 1337 | 11.539 s | 7.262 s | 7.269 s | 7.266 s |
| 512 | 42 | 11.657 s | 7.309 s | 7.345 s | 7.331 s |
| 512 | 1337 | 12.638 s | 7.937 s | 7.974 s | 7.956 s |

Across the six cells, mean wall time was:

| Plan | Mean wall | Relative to original |
|---|---:|---:|
| Original | 11.948 s | — |
| Rank-preserving | 7.510 s | 37.14% lower |
| Largest-first | 7.518 s | 36.99% lower |
| Rank-major | 7.516 s | 37.09% lower |

The three grouped layouts are effectively tied at full-model scale. The
full-model cell winner counts were rank-preserving 3, largest-first 2 and
rank-major 1; the differences are small and do not form a stable context or
seed-dependent rule.

Integrity results:

* 6/6 input prefixes exact;
* 6/6 strict equivalence probes passed;
* 6/6 final model-state comparisons passed;
* 6/6 final optimizer-state comparisons passed;
* all timed losses finite;
* no frozen Crucible artifact modified;
* no scientific training run or checkpoint created.

## Dispatch landscape

The dispatch fixture contained 36 workload shapes, repeated independently,
for 72 paired observations. It varied E4/E8/E16-shaped fragmentation, 512,
1024, 2048 and 4096 tokens, and balanced, moderate-skew and high-skew
assignments.

The fixture uses the first N frozen layer-0 expert weights for N in 4, 8 and
16. It is therefore an execution-shape study, not a quality comparison of
retrained E4 or E8 language models.

Among the strict candidates:

* rank-preserving won 61/72 paired observations;
* rank-major won 11/72;
* only three workload patterns favored rank-major in both independent runs:
  E16/1024 moderate-skew, E16/2048 high-skew and E16/4096 high-skew;
* five workload patterns changed winner between repeats, indicating timing
  sensitivity;
* the largest repeatable rank-major improvement was approximately 3.4% in one
  E16/2048 high-skew repeat, not a broad dominance pattern.

This is enough to justify a small deterministic selector follow-up if desired,
but not enough to train a neural selector. The selector would have to learn a
very small and noisy choice at present.

## Shape-changing diagnostics

Padded batched GEMM was faster than strict grouping in 9/36 workload patterns
in both dispatch repeats, especially for balanced small workloads. It also
became dramatically worse under high skew because padding every expert to the
dominant group size wastes work. The combined-rank and padded paths remain
diagnostic: the full-model V1 equivalence gate rejected their changed BF16
matrix shapes.

They are not valid scientific speed baselines until a numerical contract is
defined and passed.

## Decision

The V2 decision gate is **not yet sufficient for a neural scheduler**.

Recommended next step, if more work is wanted:

1. Run a targeted deterministic replication on the three E16 high-skew
   crossover candidates with more independent repetitions and routing patterns
   taken from actual full-model traces.
2. Compare a fixed rank-preserving plan against that small deterministic rule.
3. Only introduce a neural layer-plan selector if the deterministic rule has a
   repeatable net gain larger than its measurement and dispatch overhead.

Until then, use rank-preserving grouping as the default deterministic engine.
