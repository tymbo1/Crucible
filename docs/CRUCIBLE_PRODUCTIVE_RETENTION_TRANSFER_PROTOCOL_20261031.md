# Crucible productive-retention transfer protocol

Date: 2026-10-31  
Status: **UNTESTED — protocol and CPU manifest preflight only; no neural training or sealed evaluation approved**

## Research question

Can a verified skill acquired earlier help Crucible learn a genuinely unfamiliar target procedure more quickly and reliably, rather than merely preserving old answers or retrieving old examples?

The primary outcome is productive transfer:

```text
earlier verified skill → faster/more reliable acquisition of a disjoint target skill
```

The previous retention factorial measured correct-answer preservation. It did not expose new target procedures or measure target sample efficiency. This protocol is therefore not a relabelled retention run.

## Closest prior work and novelty audit

| Prior record | Established | Why this protocol is different |
|---|---|---|
| Commit `836de1e`, retention factorial | A3 reduced one-pass regressions from 26 to 10; no arm passed zero-regression preservation | Introduces a sequential source-skill phase and functionally novel target phase; measures transfer and learning speed |
| Track B B0–B3, frozen negative baseline | B3 fit training targets but acquired 0/48 novel procedures | B3 was one-shot proposal; this protocol tests whether prior learned skill changes later target acquisition |
| C1 verified operator-library experiments | Engineered CPU policy reuse and motif transfer | No neural checkpoint, no source-skill target dependency and no target sample-efficiency curve |
| OP and checkpoint-transition diagnostics | Additional inference passes can preserve or damage answers | Does not test acquisition of a new skill after a source skill is learned |
| Symbolic relational baselines | Explicit solvers can compose held-out relations | No learned source phase or neural productive-transfer measurement |

The proposal is invalid if it reuses the B3 manifest, counts direct source-procedure retrieval as target learning, or lacks a source competence gate.

## Hypothesis and predictions

### H1 — productive retention

A model that has learned and verified the source skill, then receives retention protection during target learning, will acquire target procedures requiring reuse of that skill with a higher target-validation area-under-curve and fewer target examples/updates than the source-reset and ordinary-continuation controls.

### H2 — not simple memory retrieval

The productive arm will outperform the memory-only arm on target procedures whose canonical AST and exhaustive truth table are absent from source memory. Any target hash hit is a leakage/invalidity event, not a success.

### H3 — specificity

The benefit will be concentrated on the dependent target family that reuses the source primitive. There should be no comparable advantage on the independent target family, which does not require the source skill.

The hypothesis fails under the tested conditions if the productive arm does not improve target sample efficiency over the prespecified controls, or if its apparent gain disappears after target-retrieval and source/target-overlap audits.

## Internal state and update boundary

The sequential learner has these explicit state components:

```text
theta_0       common step-15,000 120M Crucible initialization
theta_S       source-skilled checkpoint after source competence gate
M_S           optional exact source-example memory, memory-only arm only
C_S           verified source-skill certificate and provenance
R_S           source validation replay used for retention anchoring
theta_T       target-learning checkpoint
E_T           transient target episode state, cleared on reset
```

The productive arm uses the existing A3-style output-logit and final-state representation anchoring against `R_S`. It does not receive raw source demonstrations during target learning. The verifier and executor remain external and authoritative; the model cannot promote its own output.

The update rule for the productive arm is:

```text
loss = target_loss
     + 0.05 * KL(student_source_logits || frozen_source_logits)
     + 0.05 * MSE(norm(student_source_state), norm(frozen_source_state))
```

The source certificate is created only after independent source-validation success. It records the source family, validation manifest hash, competence interval and checkpoint hash. It is not a target answer and is not supplied as an oracle at inference.

Persistence stores `C_S`, `R_S` references and checkpoint provenance separately from `M_S`. A reset clears `E_T`; it does not silently reload sealed answers. Post-reset target execution is required.

## Task families

The CPU manifest generator creates a new few-shot Boolean-function curriculum, not the prior modular-offset graph, digitwise AST, or B3 procedure manifest.

### Source family S: shallow Boolean composition

- Four Boolean inputs.
- Operators: `and`, `or`, `xor`, `not`.
- Expression depth at most two.
- The model receives demonstrations of input vectors and outputs, then must answer a held-out query.
- Source train and validation functions are disjoint by canonical AST hash and exhaustive 16-row truth-table hash.

### Dependent target family T-dep: novel deeper compositions

- Same source primitives, but depth-three compositions and variable permutations not present in source.
- Every target procedure is structurally and functionally disjoint from source procedures.
- Correct acquisition requires reusing the learned Boolean composition skill on a new composition, not retrieving a source answer.

### Independent target family T-ind: negative transfer control

- A disjoint target grammar using `nand`/`nor`/`xnor` compositions and novel structures. `xnor` is explicitly defined as boolean equivalence (`1 - (a XOR b)`).
- It is not expected to benefit from the source operator certificate; it is a negative-transfer control for a different primitive vocabulary. The extra `xnor` primitive is required because four-input depth-three `nand`/`nor`/`not` alone has only 162 distinct truth tables, fewer than the frozen 320 independent targets across both seeds.
- It controls for a general increase in training quality or task familiarity.

Each episode includes demonstrations and a query. Training and validation query labels are available only in their assigned splits. Sealed target query labels remain outside the repository and are opened only after all checkpoints and sample-efficiency curves are fixed.

## Frozen manifest and split requirements

The CPU preflight must generate and hash, for seeds 42 and 1337:

- 32 source-train procedures and 16 source-validation procedures;
- 64 T-dep train, 32 T-dep validation and 64 sealed T-dep procedures;
- 64 T-ind train, 32 T-ind validation and 64 sealed T-ind procedures;
- exposure buckets of 8, 16, 32 and 64 target demonstrations/examples;
- canonical AST hashes, exhaustive truth-table hashes and split membership;
- a public target manifest without sealed query labels;
- a private sealed-answer file outside Git with its SHA-256 recorded in the manifest.

No source procedure may share a canonical AST or truth-table hash with any target procedure. No target test procedure may appear in target training or validation. The generator must fail closed if it cannot satisfy the requested counts.

## Experimental conditions

All conditions use the same 120M Crucible architecture, source/target tokens, fixed update schedule, seeds, microbatch, optimizer, target exposure order and verifier budget.

### C0 — source-reset / skill ablation control

Run the source phase for matched compute, then restore `theta_0` and its optimizer/RNG state before target learning. This selectively removes the learned source skill while retaining the same source exposure budget.

### C1 — ordinary sequential learner

Continue from `theta_S` into target learning without source replay protection and without raw source-memory retrieval.

### C2 — memory-only control

Run and verify the source phase, discard source weight updates, and retain only a fixed-size source-example memory. During target learning, retrieve the registered source context into matched token slots. No target procedure or target label may be retrieved.

### C3 — productive retention

Continue from `theta_S` with A3-style source output and representation anchoring. No raw source-example memory is supplied during target learning. This is the primary productive-transfer condition.

The exact source-reset, memory-only and productive conditions are reported separately. C3 is not credited for simply retaining old answers.

## Training and evaluation schedule

The initial bounded run uses the validated 120M BF16/microbatch-8 path:

- source phase: fixed 1,000 updates;
- target phase: fixed 3,000 updates;
- target validation checkpoints: 250, 500, 1,000, 2,000 and 3,000 updates;
- no validation-driven checkpoint selection;
- source competence must reach at least 80% exact validation accuracy in both seeds before target results are interpretable;
- final sealed evaluation is performed only at the fixed 3,000-update checkpoint;
- post-reset execution uses a fresh process and the persisted condition state.

If the source gate fails, the run is classified **INVALID/INCONCLUSIVE for productive transfer**, not as evidence that productive retention failed.

## Primary and secondary metrics

Primary:

1. T-dep validation accuracy AUC over exposure/update checkpoints.
2. T-dep sealed accuracy at the fixed final checkpoint.
3. Target updates/examples to the preregistered 80% validation criterion.

Secondary:

- T-ind accuracy and harmful-transfer rate;
- paired target transitions between C0–C3;
- post-reset target execution;
- verifier acceptance and abstention;
- source-skill retention after target learning;
- exact target-retrieval/hash hits;
- training tokens, wall time, verifier calls and GPU/RAM/VRAM cost.

### Interpretation limitation of T-ind

The independent arm is a primitive-vocabulary negative control, not an
algebraically independent Boolean world. `nand`, `nor` and `xnor` can be
constructed from familiar Boolean operations when negation and composition
are available. Therefore a T-ind gain could reflect legitimate transfer of a
generic Boolean-composition skill, not only a nonspecific training benefit.
The registered claim is consequently limited: T-ind must not show a
comparable unexplained gain if C3 is to support source-specific productive
transfer. The arm does not prove that transfer between the two Boolean bases
is impossible.

The primary pooled unit is a target-procedure-by-seed pair. A paired bootstrap stratified by family and seed reports the confidence interval. The four exposure curves are not treated as independent samples.

## Success and failure criteria

### Success requires all of the following

- source competence gate passes for both seeds;
- C3 exceeds the best of C0–C2 on T-dep validation AUC by at least 0.10 absolute, with a paired 95% bootstrap interval excluding zero;
- C3 reaches the 80% T-dep validation criterion at least 25% earlier in examples or updates than the best control, or reaches it when every control fails;
- the direction holds in both seeds and both target-generation batches;
- no target hash/retrieval leakage, lifecycle failure or budget mismatch;
- T-ind performance does not show a comparable unexplained gain or harmful transfer;
- post-reset execution succeeds at the registered threshold.

### Failure classifications

- source gate failure: **invalid/inconclusive**;
- target overlap, answer leakage or budget mismatch: **invalid**;
- no C3 advantage over controls: **falsified under tested conditions**;
- C2 memory-only matches C3 while C3 does not improve sample efficiency: **memory transfer, productive retention not established**;
- gain only in one seed/family: **supported but inconclusive**;
- C3 preserves source skill but does not improve target learning: **retention without productive transfer**.

## Compute and risks

Planning estimate from the measured retention throughput is approximately 6–9 local GPU-hours for four conditions × two seeds, including checkpointing and validation. No cloud compute is justified before CPU manifest and source-gate preflight. Checkpoints remain outside Git; reports and hashes are committed.

Major risks are source incompetence, prompt-format learning, direct source-memory contamination, target-family overlap, retention loss being mistaken for transfer, and a target task that does not causally require the source skill. The preflight records each risk and must fail closed on overlap or missing labels.

## Recommended next implementation step

Implement and run only the CPU manifest/leakage preflight. It must emit the public manifests, private sealed-answer hash, truth-table overlap audit, exposure schedule, source/target family counts and a machine-readable `scientific_authorization: false` readiness report. No neural training begins from this document alone.

