# GPU Neural SSRE v5 — workload-concurrency feasibility

V5 follows the sealed V1–V4 GPU execution studies. It does not train a
neural scheduler and does not modify any Crucible artifact.

The model audit found one genuine forward-time concurrency opportunity:
independent expert/rank FFN calls inside one MoE layer. Attention projections
and transformer layers remain dependency-constrained, and the third SwiGLU
projection depends on the first two projections.

This preflight compares the accepted V2 sequential rank-preserving grouped
engine with deterministic CUDA-stream fan-out using 2, 4 and 8 streams. Each
active expert/rank segment computes into its own temporary output, then the
default stream performs the same ordered accumulation. This avoids output
write races and keeps the mathematical routing order inspectable.

It is a forward-only feasibility test. A promising result would require a
separate backward and full-model equivalence gate. A negative result closes
this concurrency route without introducing a neural controller.

