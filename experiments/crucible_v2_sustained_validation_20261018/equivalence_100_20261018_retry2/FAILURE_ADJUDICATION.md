# V2 sustained-validation preflight: failed equivalence gate

Date: 2026-10-18

This directory is the authoritative completed same-device 100-step gate. The
unchanged context runner and the in-memory `grouped_rank_preserving` V2 patch
were run separately on the same RTX 3060 Ti using the frozen context-128
manifest.

## Decision

The gate failed. The 3,125-step V2 validation was **not launched**.

Both seeds failed terminal model-state, gradient-state and optimizer-state
SHA-256 equality. The recorded routing counters and affinity telemetry also
diverged by step 100. The result is therefore not accepted as a sustained
equivalent implementation.

Absolute metric differences at step 100 were small for loss, but that does not
override the state and telemetry failures:

| Seed | validation NLL delta | max affinity absolute delta |
| ---: | ---: | ---: |
| 42 | 0.000773313822 | 113.6933035068 |
| 1337 | 0.000978784324 | 71.2158090174 |

## Interpretation boundary

The failure does not establish that the grouped engine is mathematically
wrong. It establishes that the short V2 equivalence result does not extend to
100 optimizer steps under this frozen runner and must not be used to justify a
long validation run.

The likely investigation target is changed accumulation/dispatch ordering in
the grouped implementation: the original path uses expert/rank masked updates,
while the grouped path uses sorted segments and `index_put_(accumulate=True)`.
This is a hypothesis, not a concluded root cause.

## Preserved artifacts

- `original_metrics.jsonl`
- `grouped_rank_preserving_metrics.jsonl`
- `equivalence_report.json`
- `equivalence_hashes.json`
- `run_equivalence_gate.py` hash recorded in `equivalence_hashes.json`

Two earlier disposable harness attempts remain in sibling directories and are
not scientific results: one retained both seed models and hit CUDA OOM; the
next capture-heavy version was interrupted after becoming impractically slow.

## Required next step

Audit and repair the grouped accumulation/equivalence contract, then repeat the
100-step same-device gate in a new versioned directory. Do not run the
3,125-step validation until that gate passes.
