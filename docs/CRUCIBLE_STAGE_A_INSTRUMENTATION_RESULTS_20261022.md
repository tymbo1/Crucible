# Crucible Stage A instrumentation results

Date: 2026-10-22  
Status: CPU-only pilot complete; C2 neural recursive-reasoning gate remains closed.

## Scope

This experiment tested checkpointing, reload, continuation, routing telemetry and
expert interventions for an E4 Crucible control against a dense active-compute
reference. It did not test recursive cognition, causal reasoning or general
intelligence.

The sealed v2 cognitive manifests were used after the independent oracle and
shortcut audits. The test split was loaded for manifest integrity and length
checks but was not evaluated or used for model selection.

## Runs

The accepted runs used 500 optimizer steps, seeds 42 and 1337, and the complete
diagnostic schedule at steps 25, 50, 75, 100, 200, 300, 400 and 500. Complete
checkpoints were saved at steps 0, 100 and 500.

The E4 arm has 43,151,104 total parameters, four experts, top-2 routing and
expert width 3,447. The dense reference has 21,968,640 parameters and a single
FFN width of 6,894, matching E4's nominal two-expert active FFN width. This is
an instrumentation and task-learnability control, not a matched-total-parameter
or matched-wall-clock architectural comparison.

## Final validation results

Metrics are teacher-forced over target bytes; they are not autoregressive
generation scores.

| Seed | Arm | Target loss | Token accuracy | Exact accuracy |
| ---: | --- | ---: | ---: | ---: |
| 42 | E4 | 0.838318 | 78.842% | 29.688% |
| 42 | Dense active-compute | 0.905428 | 77.579% | 24.219% |
| 1337 | E4 | 0.976338 | 76.842% | 17.188% |
| 1337 | Dense active-compute | 0.859922 | 79.053% | 13.281% |

The ordering is mixed across seeds. Averaged over the two seeds, E4 had target
loss 0.907328, token accuracy 77.842% and exact accuracy 23.438%; dense had
target loss 0.882675, token accuracy 78.316% and exact accuracy 18.750%.
These small-run differences do not establish architectural superiority.

## Mechanical instrumentation gate

Passed for both arms and both seeds:

- all eight scheduled diagnostic records were present;
- checkpoints at steps 0, 100 and 500 reloaded with matching fixed-probe logits;
- step-100 model, optimizer and RNG state was restored twice and produced the
  same resumed update, optimizer state and next-sample fingerprint;
- E4 fixed-probe routing and expert-intervention outputs reproduced after
  reload at steps 100 and 500;
- intervention probes changed the probe output while preserving the original
  model parameter digest;
- no NaN/Inf or runtime failure occurred;
- each run's nine recorded artifact hashes were independently rechecked with
  zero mismatches.

Each two-arm run occupies approximately 1.82 GB, including raw checkpoints.
No GPU, cloud resource or checkpoint download was used.

## Learnability gate

Not passed and not interpreted as an architectural failure. The preregistered
reference criterion required at least 90% training and 70% validation exact
accuracy for each task family. At 500 updates, neither arm met that criterion
across the four families. In particular, exact validation accuracy for math and
error correction remained zero, while relational binding remained at 50%.

This indicates that the current short byte-level adapter and training budget do
not yet provide a competent cognitive-task reference. It prevents a meaningful
E4-versus-dense conclusion about reasoning ability.

For additional context, final exact validation accuracy averaged over the two
seeds was 0% for error correction, 0% for math, 50% for relational binding,
43.75% for logic in E4, and 0%, 0%, 50% and 25% respectively in the dense
control. These are adapter learnability diagnostics, not claims about the
intrinsic difficulty of the task families.

## Frozen artifact locations

- Seed 42 rigorous run: `experiments/crucible_stage_a_instrumentation_pilot_500step_seed42_rigorous_20261022/`
- Seed 1337 rigorous run: `experiments/crucible_stage_a_instrumentation_pilot_500step_seed1337_rigorous_20261022/`
- Runner: `tools/run_crucible_stage_a_instrumentation_20261022.py`
- Stage A protocol: `docs/CRUCIBLE_COGNITIVE_INSTRUMENTATION_PILOT_20261021.md`
- C2 readiness audit: `docs/CRUCIBLE_C2_READINESS_AUDIT_20261022.md`

The earlier seed-42 and seed-1337 runs without step-100 continuation checks are
preserved as exploratory artifacts and are not the accepted protocol record.

## Decision

Stage A instrumentation is validated. Stage A cognitive-task learnability is
not yet established. C2 remains **not ready** because the model/adapter pair
has not demonstrated sufficient baseline task competence. The next review gate
should choose between a bounded CPU adapter/curriculum learnability calibration
and a separately approved competent frozen model; no neural recursive-reasoning
claim follows from this pilot.
