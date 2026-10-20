# GPU Neural SSRE v3 — preflight result

Date: 2026-10-18  
Device: NVIDIA GeForce RTX 3060 Ti  
Scope: dispatch-only GPU-resident routing-metadata feasibility

## Outcome

The metadata feasibility preflight passed all integrity checks:

- CPU and GPU pair counts matched;
- CPU and GPU offsets matched;
- both GPU-resident candidates matched the accepted CPU-offset path;
- output and expert-gradient tolerances passed for both fixtures;
- all losses were finite;
- no model, checkpoint, manifest or V2 artifact was modified.

The performance result is negative for this narrow change. Moving metadata
construction to CUDA did not remove the Python expert loop or its dispatch
boundary, and was slower in both tested fixtures.

## Timed CUDA results

Mean forward-plus-backward device time, milliseconds:

| Fixture | CPU offsets | GPU offsets with scalar reads | GPU segment masks |
|---|---:|---:|---:|
| Balanced, 512 tokens | 10.524 | 11.225 (+6.66%) | 16.449 (+56.30%) |
| High skew, 512 tokens | 10.716 | 11.208 (+4.60%) | 17.396 (+62.35%) |

The scalar-offset path performed 64 host scalar boundary reads per dispatch
(16 experts × top-2 × two boundaries). The segment-mask path avoided those
reads but performed 32 device-side mask evaluations and repeated scans.

## Decision

Do not promote either metadata-only candidate to a full-model benchmark. The
preflight does not justify a neural scheduler or additional cloud spending.

The result identifies the remaining boundary precisely: GPU-resident counts
and prefix sums are straightforward, but the Python loop still needs a way to
launch variable-sized expert work without host scalar dispatch. The next
meaningful V3 work, if authorized, is a separate feasibility probe for a
compiled/fused or persistent device-side execution primitive—not another
metadata relocation.

This negative result is limited to the tested implementation and workload; it
does not establish that all GPU-resident execution engines are ineffective.

