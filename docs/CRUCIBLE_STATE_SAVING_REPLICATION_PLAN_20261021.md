# Crucible state-saving replication plan

Date: 2026-10-21  
Status: design-only; no training or cloud execution authorized.

## Objective

Close the missing-state gap in the E4/E8/E16 sparse-capacity evidence and
create the minimum artifacts needed to test causal expert specialization.
The campaign must also include a dense control if its geometry can be defined
without adding a second confound.

The existing E4/E8/E16 results remain frozen. This is a new state-saving
replication, not a rewrite or extension of the old runs.

This campaign is a small-scale mechanism and specialization gate. It is not
intended to decide whether a 130–170M model can demonstrate advanced cognition;
those claims require task-specific training and later capacity tiers.

## Recommended staged campaign

### Stage 1 — sparse state-saving replication

Run E4, E8 and E16 under the matched-active-compute protocol with seeds 42 and
1337. Keep the same dataset, byte exposure, optimizer schedule, context and
checkpoint intervals as the sealed sparse-capacity gate. Add state persistence
and fixed-probe routing capture.

This is the smallest campaign that can answer whether the existing language-loss
trend is reproducible while making specialization interventions possible.

### Stage 2 — dense control decision

Only if Stage 1 reproduces the sparse trend should the dense control be run.
The dense control must be defined from the same active-compute and parameter
accounting contract before launch. If no geometry satisfies both constraints,
report the control as a separate explicitly labelled regime rather than
pretending it is matched.

The preferred minimum is two seeds for the dense control. A third seed is a
follow-up replication for a small effect, not a prerequisite for the first
state-saving gate.

## Required state at each checkpoint

Use checkpoints at steps `100, 500, 1500, 2500, 3125`. The final checkpoint is
mandatory even if an earlier save fails.

Capture canonical, tensor-by-tensor records for:

1. model parameters and buffers, including names, shapes, dtype and SHA-256;
2. optimizer state tensors and parameter-group configuration;
3. scheduler state, if present;
4. Python, NumPy, CPU Torch and CUDA RNG states;
5. sampler state and an exact sample-order fingerprint;
6. loss, gradient norm, clipping and auxiliary-loss metrics;
7. routing counts, selected probabilities and load statistics;
8. a fixed-probe routing trace for every layer;
9. manifest, code, environment and configuration hashes.

The canonical state digest must hash sorted tensor names and raw contiguous
bytes with a declared dtype/shape header. It must not depend on pickle order,
filesystem ordering or device-specific serialization metadata.

The default lean profile saves full model snapshots at every scheduled
checkpoint, stores compact optimizer/RNG/sampler digests at every checkpoint,
and saves the full resumable optimizer/RNG/sampler state at the final step.
The full-resume profile saves those full training states at every checkpoint.
The lean profile is sufficient for specialization-development analysis and a
final restart test; it is not a claim that arbitrary intermediate checkpoints
are resumable.

## Fixed routing probe

Create a separate immutable probe schedule before training. It should contain
the same held-out byte sequences for every expert-count arm, with a fixed
sample order and no optimizer updates.

At each checkpoint record, per layer and token:

- top-1 and top-2 expert identifiers;
- routing probabilities;
- selected expert output norms if already available;
- optional pre- and post-expert activation digests.

The probe is for comparison and intervention analysis, not training. It must
not alter RNG consumption or optimizer state.

## Intervention hooks

The evaluation runner should support, without changing the training run:

- normal routing;
- leave-one-expert-out execution with a declared replacement policy;
- counterfactual expert replacement;
- fixed random expert permutation controls;
- cross-seed matched expert comparison;
- layer-specific intervention and no-intervention controls.

Each intervention must run on a frozen checkpoint and a held-out task suite.
Routing frequency, load balance or byte-conditioned affinity alone are not
specialization evidence. A specialization claim requires a reproducible,
expert-specific causal effect greater than the permutation null.

## Acceptance criteria

### State-saving integrity

- every scheduled checkpoint has a complete state manifest;
- model, optimizer, RNG and sample-order hashes verify after save and reload;
- interrupted save is detected and cannot be accepted as a checkpoint;
- a resumed dry-run reproduces the next scheduled sample fingerprint;
- no non-finite model, gradient or optimizer tensors occur.

### Sparse-capacity replication

- E4/E8/E16 use identical task exposure and optimizer updates;
- both seeds complete all five state checkpoints;
- primary endpoint remains EOS-excluded validation BPB/NLL;
- per-seed results are reported, not only means;
- routing remains healthy with no dead top-2 expert;
- any improvement is labelled sparse-capacity evidence, not specialization.

### Dense control

- active compute, total parameters, data exposure and evaluation budget are
  declared before execution;
- dense geometry is not silently changed to improve its score;
- if no fair match exists, the comparison is marked exploratory and cannot
  decide architectural superiority.

## Storage estimate

The estimate below is deliberately conservative. It assumes approximately 16
bytes per parameter per full model-plus-optimizer checkpoint: BF16 model
weights, two FP32 Adam moments, a master-weight allowance and metadata.
Actual files must be measured during a preflight.

| State profile | Checkpoints | Two seeds | Conservative storage |
|---|---:|---:|---:|
| Lean: four model-only snapshots plus one full final state, E4 + E8 + E16 | 5 each | 2 | approximately 19.1 GB / 17.8 GiB |
| Lean dense upper-bound at E16 parameter count | 5 | 2 | approximately 10.9 GB / 10.1 GiB |
| Lean sparse plus dense upper bound | — | — | approximately 30.0 GB / 28.0 GiB |
| Full-resume: five full states, E4 + E8 + E16 | 5 each | 2 | approximately 47.8 GB / 44.5 GiB |
| Full-resume dense upper-bound at E16 parameter count | 5 | 2 | approximately 27.2 GB / 25.4 GiB |
| Full-resume sparse plus dense upper bound | — | — | approximately 75.0 GB / 69.9 GiB |

The lean profile needs approximately 25% headroom for routing traces, manifests,
temporary files and failed checkpoint cleanup: a 50 GB volume is sufficient for
the staged campaign. The full-resume profile needs approximately 25% headroom,
so a 100 GB volume is the minimum sensible target; 150 GB is safer if the
dense control proceeds or optional diagnostics are retained.
The private Git repository should receive manifests, hashes and compact
metrics, not raw checkpoints.

Raw gradients, full activation tensors and unrestricted per-token traces are
optional diagnostics and are excluded from these estimates. Compact gradient
and optimizer digests remain required for equivalence checks.

## Hardware and cost estimate

The previous secure RTX 4090 paired run recorded approximately 1,574 seconds
for an E16 grouped 3,125-step arm and 1,933 seconds for its original arm,
including the final interval. That suggests roughly 30–40 minutes per E16 arm
before state-save overhead, but it is not a promise for E4/E8, the dense
control or a new environment.

Planning envelope:

- Stage 1, six sparse arms: approximately 4–6 GPU-hours sequentially;
- Stage 2, two dense arms: approximately 1–2 additional GPU-hours;
- preflight, reload checks and storage overhead: reserve 1–2 hours;
- total planning cap: 6–10 GPU-hours on one 4090-class instance.

Using the previously observed secure RTX 4090 planning rate of US$1.19 per
GPU-hour, compute alone would be approximately US$7.14–US$11.90. This is an
estimate, not a current price quote; persistent-volume, transfer and provider
charges are excluded. A paid launch requires a fresh availability and price
check, an explicit spending cap and an artifact-collection test first.

The current local E16 peak CUDA allocation in the captured context metrics is
approximately 3.47 GB. The state-saving design should fit the 8 GB local GPU
for a mechanical preflight, but the dense geometry and checkpoint staging must
be verified before any training claim.

## Local work before authorization

Permitted CPU-only work:

- validate canonical state-digest code on synthetic tensors;
- validate save/reload manifests and interrupted-write detection;
- generate fixed-probe schedules and sample-order fingerprints;
- run the deterministic cognitive task generators and exact oracles;
- audit the context collection and existing sparse metrics.

Not authorized by this plan:

- new model training;
- checkpoint downloads;
- changes to frozen runners or historical artifacts;
- cloud deployment or paid GPU use.

## Decision gate

Return the following for approval before Stage 1:

1. exact model and dense-control geometry;
2. preflight storage measurement;
3. final command lines and environment hashes;
4. checkpoint and probe file schemas;
5. expected runtime and cost with a hard cap;
6. stop conditions and result-collection verification.

