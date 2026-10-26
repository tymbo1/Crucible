# Crucible cognitive learnability ladder results

Date: 2026-10-22  
Status: CPU-only dense-control diagnostic complete; ladder stopped at the first failed learnability gate.

## Scope

This was a bounded learnability calibration for the current byte-level task
adapter. It used a fresh dense active-compute control for each stage, 300
optimizer updates per stage, and seeds 42 and 1337. The ladder was ordered as
atomic tasks, two-operation composition, relational chains and logical
forward-chaining. A stage had to reach at least 90% training exact accuracy and
70% validation exact accuracy before the next stage could run.

The E4 model was not run. No test split was evaluated or used for model
selection. No GPU, cloud resource or checkpoint download was used. This is a
learnability diagnostic, not evidence for or against recursive cognition,
relational reasoning in the full model, or E4-versus-dense architectural
superiority.

## Results

Metrics are teacher-forced target-byte metrics.

| Stage | Seed | Train exact | Validation exact | Held-out exact | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Atomic | 42 | 100.0% | 100.0% | 100.0% | Pass |
| Atomic | 1337 | 100.0% | 100.0% | 100.0% | Pass |
| Composition | 42 | 18.75% | 10.94% | 10.94% | Fail |
| Composition | 1337 | 20.31% | 6.25% | 9.38% | Fail |

The atomic task reached perfect accuracy on its 64-example validation and
held-out splits at both seeds. The two-operation composition task did not reach
the gate even on its training split, and validation performance remained close
to chance-level exact matching. The same boundary appeared at both seeds.

Because the ladder stops after the first failed stage, the relational and logic
stages were not trained or evaluated. Their manifests were generated and
hashed, but their results must be reported as **not run**, not as failures.

## Interpretation

The current dense control can learn the simplest atomic mapping within 300
updates. It does not yet provide a competent reference for two-step
composition under the current adapter, task encoding, optimizer and budget.
This is a stable diagnostic boundary across the two tested seeds, not a proof
that the architecture cannot learn composition.

The result leaves several hypotheses open:

- the byte-level interface may not expose the intermediate computation clearly;
- 300 updates may be insufficient for the compositional mapping;
- the current objective may not provide a useful signal for decomposing the two
  operations; or
- the generated composition distribution may need a simpler calibration task
  before held-out operation combinations are attempted.

Increasing model size or comparing E4 against dense before resolving this
boundary would confound architecture with task learnability. C2 neural
recursive-reasoning evaluation therefore remains blocked.

## Artifact and integrity checks

Both runs completed without runtime errors. Each output contains the generated
ladder manifest, split manifests, step-0 and final checkpoints for the atomic
and composition stages, configuration, results and run log. The 24 recorded
artifact hashes per run were independently recomputed with zero mismatches.

The test split was not evaluated, and `RESULTS.json` records
`test_evaluated: false` for both seeds.

## Frozen locations

- Seed 42: `experiments/crucible_cognitive_learnability_ladder_20261022_seed42/`
- Seed 1337: `experiments/crucible_cognitive_learnability_ladder_20261022_seed1337/`
- Runner: `tools/run_crucible_cognitive_learnability_ladder_20261022.py`

## Next review gate

Do not launch E4 or neural recursive-reasoning training yet. The smallest
decisive follow-up is a separately approved composition calibration that keeps
the same dense control and disjoint splits while testing one change at a time:

1. expose or supervise the intermediate result without revealing held-out
   answers;
2. verify that operation tokens and input formatting are unambiguous; and
3. only then test a larger update budget if the interface is demonstrably
   learnable.

The original ladder outputs and the failed composition gate remain frozen as
evidence. No later stage should be run until the composition control passes the
same preregistered learnability criteria.
