# Crucible cognitive instrumentation and learnability pilot

Date: 2026-10-21
Status: proposal only; do not execute without explicit approval.

## Purpose

Establish that the corrected cognitive tasks are learnable, that Crucible can
save and reload the required state, and that expert interventions are
reproducible.
This pilot is deliberately separate from the full E4/E8/E16-versus-dense
architectural campaign and cannot establish architectural superiority.

Version 1 of the task manifests is preserved as a historical validation
artifact. Its error-correction generator made candidate status predictable
from arithmetic composition on the validation split; the CPU shortcut audit
measured a 75% prompt-length-bucket baseline there. No GPU run is permitted
with v1. The pilot uses the corrected v2 manifests only.

## Stage A — small state-saving pilot

### Models

Use one small Crucible model, preferably the existing E4 Branch B geometry
(approximately 43M parameters), plus one simple dense learnability control of
similar active compute. The dense control is a task-learnability reference,
not the full architectural comparison. Do not include E8 or E16 in this
instrumentation pilot.

### Data and task protocol

- Use the sealed v2 manifests in
  `experiments/crucible_cognitive_eval_v2_20261021/manifests/`.
- Keep oracle fields out of model input.
- Train on the train split, select only with validation, and use test only at
  the final report.
- Measure exact-answer accuracy for arithmetic, relational binding, logic and
  correction, alongside byte/token loss where the adapter supports it.
- Report training-distribution accuracy separately from held-out composition,
  relation-depth and logic-depth accuracy.
- Run the cheap majority and prompt-length baselines first. The v2 audit must
  be attached to the pilot record and any newly discovered shortcut blocks
  GPU training until the affected family is redesigned.

### Training schedule

1. CPU-only manifest and shortcut audits.
2. A disposable 20-step capture preflight.
3. A limited 500-optimizer-step learnability pilot.
4. Only if the preflight passes, repeat the 500-step pilot for a second seed.

The 20-step run tests plumbing. The 500-step run tests whether the control can
learn the supplied task curriculum. Neither is a cognitive capability claim.

## Checkpoint retention policy

### Lightweight diagnostics

At steps 25, 50, 75, 100, 200, 300, 400 and 500, record:

- loss and exact task accuracy;
- model-state digest;
- gradient and optimizer-state digests;
- RNG and next-sample fingerprints;
- fixed-probe routing counts and top-k assignments;
- peak VRAM, host RAM and wall time.

These records are compact and support trajectory inspection.

### Full resumable artifacts

Save complete model, optimizer, scheduler, RNG and sampler state at steps 0,
100 and 500. Verify reload and continuation from step 100 before accepting the
pilot. Store manifests and hashes immutably; retain raw checkpoints outside
Git.

The pilot should fit comfortably within the lean storage profile. A 20 GB
local or cloud volume is a safe initial allocation; measure actual files before
the 500-step run.

## Expert intervention test

At the final checkpoint and one intermediate checkpoint, replay the fixed probe
under:

1. normal routing;
2. one-expert ablation with a declared replacement policy;
3. a matched random expert permutation;
4. the same intervention repeated after save/reload.

The intervention passes its instrumentation gate only when the routing trace,
output digest and task metrics reproduce after save/reload. A performance
difference is not yet functional specialization evidence; the pilot only proves
that the hook is operational and reproducible.

## Success criteria

### Mechanical pass

- all scheduled lightweight records are present;
- full checkpoints reload with matching canonical state digests;
- the resumed step has the expected sample-order fingerprint;
- fixed-probe routing and intervention outputs reproduce within declared
  exact/tolerance contracts;
- no NaN/Inf, missing tensor, provenance mismatch or unbounded storage growth.

### Learnability pass

The dense reference must reach at least 90% train accuracy and 70% validation
accuracy on each task family, or the family is marked under-specified rather
than used to reject Crucible. Held-out composition/depth results are reported
separately; the pilot does not require them to be high.

The Crucible E4 pilot must beat the corresponding cheap baseline on validation
and show a positive held-out signal on at least one declared task family. This
is a readiness signal, not evidence that E4 is architecturally superior.

## Failure and early-stop criteria

Stop immediately and preserve artifacts if:

- any checkpoint hash or RNG/sample-order reload check fails;
- any intervention mutates training state or changes the fixed probe on repeat;
- a non-finite tensor, routing count or loss appears;
- storage exceeds 80% of the allocated volume;
- the task shortcut audit finds a majority or length-only baseline that makes
  the intended task trivial;
- the dense control cannot learn a family within the limited budget.

The last condition is a task/capacity diagnosis, not evidence that recursive or
MoE mechanisms fail.

## Local RTX 3060 Ti estimate

The current captured E16 context metrics use approximately 3.47 GB peak CUDA
memory. E4 is smaller, so the pilot should be locally feasible, subject to a
fresh mechanical preflight. The planning envelope is:

- 20-step preflight: approximately 5–15 minutes;
- one 500-step E4/control pilot: approximately 15–45 minutes;
- two-seed Stage A including reload and intervention checks: approximately
  1–2 hours;
- storage: 10 GB expected, 20 GB allocated for safety.

These are planning estimates, not measured runtime promises. The first 20-step
preflight must establish actual throughput before any longer run.

## Cloud alternative

If the local preflight fails for memory or stability, use one private secure
4090-class instance only after approval. Using the previous US$1.19/hour rate
as a planning assumption, the pilot would likely cost roughly US$1–US$4 of
compute plus storage. This is not a current price quote and excludes provider
storage/transfer charges. No cloud deployment is authorized by this document.

## Stage B — recursive-supervision experiment

Only after Stage A passes should the three matched training conditions be
considered:

| Condition | Supervision |
|---|---|
| A — baseline | ordinary answer supervision |
| B — recursive | answer, critique, counterexample and revision supervision |
| C — verified recursive | B plus independently checked feedback and correction targets |

Use the same underlying task instances, architecture, model size, seeds and
evaluation manifests. Account for additional recursive target tokens and the
cost of producing verified supervision. Keep test answers and final test
selection independent of training.

The recursive objective is a hypothesis, not an established algorithm. Its
operational outcomes should include answer accuracy, correction success,
contradiction rate, unsupported-claim rate, unnecessary-pass rate and compute
per solved item. A later adaptive controller may choose whether to answer,
critique, seek evidence, create a counterexample or stop, but it is not part of
Stage A.

## Decision

Do not launch Stage A until the proposal is reviewed. If approved, run the
local 20-step preflight first. The full E4/E8/E16 versus dense campaign remains
a separate later gate and should not be inferred from this pilot.
