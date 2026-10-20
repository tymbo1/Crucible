# GPU Neural SSRE v5 measurement protocol

## Matrix

- Frozen E16 layer-0 expert weights;
- 16 experts, top-2-shaped assignments;
- 512, 2,048 and 4,096 routed tokens;
- balanced and highly skewed routing;
- V2 sequential grouped reference;
- CUDA stream fan-out with 2, 4 and 8 streams;
- one warm-up and three timed forward trials per case.

## Measurements

Record CUDA event time, host wall time, output error versus the V2 reference,
finite outputs, active segment count and peak memory. Profiling is disabled
during timing.

## Decision rule

A stream plan is potentially useful only if it is faster in repeated cases and
within the V2 forward tolerance of `5e-3`. This preflight does not establish
training equivalence. Any promising plan requires a separate backward,
optimizer-state and full-model gate before adoption.

