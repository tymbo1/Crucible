# CNSS Latent Geometry Causal Audit v1 — interpretation

Classification: **COMPLETED_NO_TRAINING_GEOMETRY_AUDIT**
Original CNSS classification: **unchanged; exploratory and inconclusive**

## Scope

This is a no-training diagnostic of the existing final adapter checkpoints.
It evaluates all 256 eight-bit states for seeds 42, 1337 and 2025 under the
valid `lambda=0.1` and exploratory `lambda=1.0` contract conditions. The
content-addressed workspace artifact was loaded from disk; no workspace
optimization, adapter update or GPU work was performed.

## Main measurements

| Condition | Mean A-state displacement | Mean state cosine | B on actual A output | B on canonical state | Mean canonical rescues |
| --- | ---: | ---: | ---: | ---: | ---: |
| λ=0.1 | 5.139 | 0.9451 | 82.42% | 100% | 45.0 / 256 |
| λ=1.0 | 2.780 | 0.9833 | 99.35% | 100% | 1.67 / 256 |

The stronger contract reduced mean displacement by approximately 46% and
raised the mean cosine similarity by 0.0383. Adapter A remained individually
competent at approximately 98.0–100% across these final checkpoints, while B
remained perfect on canonical intermediate states.

## Directional sensitivity

Norm-matched Gaussian perturbations of canonical states produced 100% B
accuracy for every seed and both contract strengths. The actual structured
errors produced by A did not: λ=0.1 produced 26–61 canonical rescues per seed,
whereas λ=1.0 produced 0–3.

This rules out a simple magnitude-only explanation under this diagnostic. B
can tolerate arbitrary tested perturbations with the same per-state norm, but
some structured directions associated with A's output are harmful.

Principal-axis perturbations also show directional fragility. The minimum
accuracy among the tested signed principal-axis interventions was 30.86% for
λ=0.1 and 87.89% for λ=1.0. These axes are condition-specific, so this is
evidence of directional sensitivity, not a direct comparison of identical
directions across conditions.

## Causal interpretation

The evidence supports an interaction between two mechanisms:

1. **State production:** stronger canonical alignment makes A's output more
   like the canonical representation and greatly reduces observed composition
   failures.
2. **State consumption:** B is sensitive to particular structured deviations,
   even though it tolerates the norm-matched Gaussian controls.

Canonical-state replacement repairs every observed λ=0.1 failure on the
tested rows and introduces no canonical regressions. That establishes the
interface state as causally involved. It does not prove that A alone is at
fault: B is independently trained under each contract condition, and its
directional tolerance also changes in the diagnostics.

## What this does not establish

The audit does not demonstrate lifelong learning, positive forward transfer,
broad compositional generalization or superiority of CNSS. It is a bounded
mechanistic explanation of the existing eight-bit result. The original
scientific classification remains unchanged.

Machine-readable details: `LATENT_GEOMETRY_AUDIT_REPORT.json`
Per-state rows: `LATENT_GEOMETRY_ROWS.jsonl`
