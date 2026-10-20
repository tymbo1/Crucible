# Sustained-equivalence gate reuse audit

**Date:** 2026-10-20

The existing Crucible production-validation package already contains a tested
same-device equivalence harness. Reusing it is preferable to creating a second
independent comparator.

## Existing evidence

Package:
`/home/timbushnell/crucible_v2_production_telemetry_v1_20261019/`

- `run_equivalence_gate.py` compiles successfully.
- Its corrected 100-step report is accepted for seeds 42 and 1337.
- The runtime comparison uses exact model, gradient and optimizer-state
  SHA-256 equality.
- The metric comparison covers both seeds and all four MoE-layer telemetry
  records.
- The separate `sustained_3125` artifact completed 3,125 grouped-only updates,
  but has no original-arm counterpart and therefore cannot establish paired
  long-horizon equivalence.

The 100-step corrected report records:

- `accepted: true`
- `max_steps: 100`
- runtime equivalence: accepted
- metric/telemetry equivalence: accepted
- seeds: 42 and 1337

## Cross-GPU source identity

The core source and manifest hashes in the tested harness match the sealed RTX
4090 package:

| Component | SHA-256 |
|---|---|
| `shared/context_runner.py` | `948fe1ba2b466b8ddbd55d67e32890314407ba6b3d413b250c2faa756f9a335b` |
| `shared/model_core.py` | `febc61bc38d23b32593c5e1682ea3ef44b272913878d52ad8e4576c732112415` |
| grouped expert implementation | `223aa756a36235df02858ea8171a7516c2e9afbceacf4005a9c8b7b138df6b90` |
| E16 manifest | `3d7b4e4e3ce976004df88e78b8588456c974b40a38e57f8cba79583f1f3695d0` |

This establishes that the earlier harness design can be applied to the 4090
artifact without changing the model or manifest. The harness itself remains a
separate source artifact and must be included in the next run's hash index.

## Remaining gap

The existing comparator captures terminal model, gradient and optimizer state,
but not checkpoint-by-checkpoint RNG, sampler or scheduler hashes. Before the
3,125-step 4090 run, add those fields to a new versioned wrapper or explicitly
record why the staged gate uses terminal exact state plus checkpoint metrics.

The exact 4090 execution sequence is therefore:

1. Run the existing equivalence harness against the sealed 4090 context for a
   10-step smoke test.
2. Run it for 100 steps on both seeds and preserve the exact state report.
3. Only after that passes, run the paired 3,125-step original/grouped gate.
4. Compare the complete state and metrics offline before stopping the pod.

No GPU or cloud execution was performed during this reuse audit.
