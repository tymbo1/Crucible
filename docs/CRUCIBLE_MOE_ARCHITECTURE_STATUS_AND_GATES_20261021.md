# Crucible MoE architecture status and dependency-aware gates

Date: 2026-10-21  
Status: audit and planning only; no training, checkpoint download, cloud rental or frozen-artifact modification performed.

## Executive verdict

Crucible has a substantially validated execution optimization, but its
architectural superiority is not established. The grouped backend is evidence
about how the computation is executed. It is not evidence that E16 learns
better representations than E4, that experts acquire transferable functions,
or that the architecture generalizes better than a simpler model.

Current classification:

- **Engineering:** grouped execution is a useful opt-in optimization, with
  short-run exact state checks and sustained matched logged metrics.
- **MoE routing:** Branch B provides strong load-balance evidence.
- **Expert-count quality:** a preliminary positive fixed-active-compute trend
  exists through E16, but it is not a general scaling law.
- **Functional specialization:** unresolved; current analysis deliberately
  does not claim semantic, syntactic or causal expert specialization.
- **Generalization and relational memory:** not decisively tested.
- **Recursive cognition, symbolic reasoning, CMS2 and autonomous goals:**
  future research tracks, not substitutes for the unfinished MoE gates.

## Evidence audit

### Completed or substantially completed

| Area | Evidence | Correct interpretation |
|---|---|---|
| Analytical accounting | `crucible_expert_count_accounting_v1_20261017/artifacts_v2/` and v2 extended accounting | Parameter/FLOP/memory accounting for E1/E2/E4 and extended configurations; analytical, not a benchmark |
| Small expert-count controls | `crucible_expert_count_ab_v1_20261017/full_run/run_ledger.json` | Fixed-total and fixed-active E1/E2 plus reference E4 completed for two seeds and 18 records per variant |
| Sparse-capacity gate | `crucible_sparse_capacity_gate_v1_20261017/analysis_v1_20261017/sparse_capacity_summary.json` | E4/E8/E16 fixed-active-compute comparison completed for two seeds |
| Routing balance | Sparse-capacity telemetry and Branch B records | Load balance and conservation are measured; balance is not functional specialization |
| Execution optimization | `docs/CRUCIBLE_SUSTAINED_METRIC_VALIDATION_20261021.md` and grouped-execution reports | Engineering speed/equivalence evidence; not architectural quality evidence |
| Parser integrity | `crucible_sustained_gate_validator_v2_20261021/` | Repeated seed checkpoints are handled by `(seed, checkpoint)` identity; regression tests passed |

### Inconclusive or limited

#### Expert-count quality

The fixed-active E4/E8/E16 postflight reports these mean validation NLLs:

| Variant | Total parameters | Mean validation NLL | Seeds | Interpretation |
|---|---:|---:|---:|---|
| E4 | 43,151,104 | 0.792905 | 42, 1337 | Reference |
| E8 | 85,511,936 | 0.761320 | 42, 1337 | Positive preliminary result |
| E16 | 170,233,600 | 0.749846 | 42, 1337 | Positive preliminary result |

The active expert geometry was held approximately constant while dormant
capacity increased. All reported cells had 18 records, no dead top-2 expert
cells and no recorded non-finite events. This supports the narrower statement
that additional sparse capacity helped in this small matched-active-compute
experiment.

It does not yet establish a transferable scaling law because the evidence uses
two seeds, a TinyStories-style byte task, short context and one main training
protocol. The fixed-total E1/E2/E4 results are a different control regime and
must not be pooled with this result as though they tested the same hypothesis.

#### Functional specialization

`crucible_telemetry_ab_100k_analysis_v1_20261016/specialization_analysis.json`
measures byte-conditioned routing-mass differentiation and explicitly states
that it does not claim semantic specialization, syntactic specialization or
causal expert identity. Cross-seed expert matching therefore cannot be treated
as evidence that Expert 3, for example, learned a stable transferable role.

The earlier expert-swap findings remain inconclusive. A decisive test must
intervene on expert computation and evaluate held-out task performance, not
only routing entropy, load CV, mutual information or affinity distributions.

#### Generalization and context

The current context-gate collection is incomplete:

- context 128: two-seed records complete;
- context 256: seed 42 captured, seed 1337 pending;
- context 512: seed 42 captured, seed 1337 pending.

The language metrics are therefore not evidence of unfamiliar-task
generalization. No Crucible experiment currently closes long-distance
relational binding, compositional key-value retrieval or held-out task-family
transfer.

#### Sustained execution equivalence

The 100-step grouped/original gate has exact model, gradient and optimizer
state hashes. The 3,125-step comparison has matching logged losses, routing
telemetry and validation metrics with an 18.58% recorded interval reduction,
but does not persist full state hashes at every matched checkpoint. Grouped
execution remains opt-in until that stronger engineering gate is completed.

## Dependency-aware research sequence

### Gate 0 — Evidence and implementation freeze

Already complete for planning purposes.

- Keep the original backend as the scientific reference.
- Keep grouped execution opt-in.
- Do not rewrite frozen artifacts.
- Keep the private Crucible repository and `Crucible-artifacts` archive
  separate from raw corpora and checkpoints.

### Gate 1 — Close the execution-equivalence limitation

Design, but do not launch yet, a paired original/grouped sustained run with
canonical hashes at predetermined checkpoints for:

- model parameters and buffers;
- gradients;
- optimizer state;
- scheduler state, if present;
- RNG and sampler state;
- routing and auxiliary-loss records;
- sample-order fingerprints.

This gate blocks promotion of the grouped backend to the unqualified default,
but it does not block architectural comparisons when the original backend is
used for both arms.

### Gate 2 — Finish expert-count quality accounting and scaling

First audit the already completed E1/E2/E4 and E4/E8/E16 records against the
frozen manifests. Do not rerun sealed controls automatically.

If the audit finds no coverage gap, the smallest additional scientific check
is a preregistered third-seed E4/E8/E16 replication at matched active expert
compute. Do not expand to E32 until this gate is adjudicated.

Required measurements:

- EOS-excluded byte NLL/BPB and validation loss;
- raw-byte exposure and optimizer updates;
- active and total parameters;
- estimated and measured active FLOPs;
- throughput and peak memory;
- per-expert token exposure, routing CV, load ratio and dead experts;
- trajectory, not only final checkpoint, comparisons.

Suggested promotion rule:

- E8/E16 improve mean BPB over E4 by a preregistered effect size, with no
  seed-specific regression beyond the declared tolerance;
- routing remains healthy and no expert is dead;
- the result survives a held-out validation split;
- the gain is reported as sparse-capacity benefit, not as proof of
  specialization.

### Gate 3 — Causal functional specialization

Use frozen matched checkpoints where available; if final checkpoints are not
available, specify the minimum new training required before running the
interventions.

Construct held-out task families with distinct required transformations, such
as delayed retrieval, entity tracking, byte-pattern completion, arithmetic
structure and compositional binding. Then compare:

1. normal routing;
2. leave-one-expert-out ablation;
3. counterfactual expert replacement or rerouting;
4. matched random expert permutations;
5. cross-seed expert matching under the same task intervention.

Evidence for specialization requires consistent expert-specific causal effects
on held-out tasks, greater than a preregistered permutation null, across seeds
and preferably across task variants. Routing imbalance or byte-conditioned
mutual information alone does not pass this gate.

### Gate 4 — Relational memory and generalization

Before adding recursive cognition, isolate the neural memory question with a
small synthetic suite:

- long-distance key-value retrieval;
- distractor-resistant associative recall;
- multi-hop relation composition;
- interference and overwrite tests;
- length extrapolation beyond the training context;
- held-out combinations of familiar relations.

Compare the current Crucible attention/memory path with one architectural
change at a time. Match active compute and training bytes. Record exact
generalization, not merely in-distribution language loss. Lessons from
BraidFormer can motivate hypotheses, but they do not establish that a
BraidFormer mechanism transfers to Crucible.

### Gate 5 — Recursive and neuro-symbolic cognition

Keep the previously proposed external scaffold active as a future track, but
do not let it replace Gates 2–4. The first narrow cognitive test remains:

> Does frozen neural generation plus symbolic critique/retrieval outperform
> additional neural recursion alone at equal or measured compute cost?

Only after that result is measured should CMS2 persistence, adaptive compute,
goal generation or neural internalization be introduced.

### Gate 6 — Autonomous goals and continual learning

This is downstream of verification and persistent-state gates. Measure goal
generation separately from goal pursuit, and memory retention separately from
weight updates. Require interruption, rollback, false-progress and
catastrophic-forgetting tests before making claims about autonomous learning.

## Immediate recommendation

No cloud run or new training run is justified by the current audit alone.

The next action is CPU-only: verify the existing expert-count package,
reconcile the E1/E2/E4 and E4/E8/E16 regimes, inventory available final
checkpoints, and finalize the preregistered specialization intervention suite.

The next paid or substantial GPU experiment, if later authorized, should be
the smallest missing Gate 2 cell—not E32, not recursive cognition and not a
pretrained initialization run.

Pretrained initialization remains an optional enabling ablation. It must not
replace the from-scratch MoE controls or be used to declare the architecture
successful.
