# Crucible training bug ledger

Status: operational working record, created 2026-10-30

This ledger is part of the training infrastructure. It is not a retrospective summary to read only after a run fails. Before changing a training runner, dataset manifest, checkpoint format, evaluator, or long-running command, read this file and run the source audit described below.

The ledger distinguishes:

- **Confirmed engineering bug:** an implementation or measurement failure with a reproduced symptom.
- **Confirmed scientific regression:** the system executed as designed but lost a verified capability or violated a research invariant.
- **Risk/control:** a failure mode not yet observed in the current runner but covered because an earlier experiment exposed it.
- **Suspected mechanism:** an interpretation that must not be promoted to a root cause without an intervention.

## Mandatory pre-training sequence

Every future build or training continuation must leave these artifacts before a long run begins:

1. Source audit against this ledger and the runner's required safety markers.
2. Known-bug checks, including data-split and checkpoint-path checks.
3. Numerical smoke test with finite loss, gradients, parameters and optimizer state.
4. Checkpoint round-trip: save, reload, validate progress, optimizer state, scheduler/loss-scale state and CPU/CUDA RNG state.
5. Evaluation sanity check: fixed manifest, answer normalization, EOS/termination accounting and a paired transition report.
6. Resource smoke test: actual per-run token delta, step timing, allocated/reserved VRAM and fail-closed OOM handling.
7. Only then authorize a long continuation. A clean smoke test is not scientific evidence.

Run the machine-readable audit with:

```bash
python3 tools/crucible_training_preflight_20261030.py \
  --ledger docs/CRUCIBLE_TRAINING_BUG_LEDGER.json \
  --runner tools/run_crucible_foundation_training_20261029.py \
  --checkpoint /path/to/checkpoint.pt \
  --output /path/to/preflight.json
```

The command is deliberately a gate, not a replacement for scientific review. It must fail closed when a required marker or checkpoint field is absent.

## Incident index

### CT-001 — Compatibility preflight was mistaken for a complete experiment runner

- **Class:** confirmed process/infrastructure bug; fixed.
- **Observed:** the Track B compatibility machinery passed while the complete B0–B3 scientific runner, lifecycle harness, sealed-target isolation and matched accounting were still missing.
- **Impact:** execution approval was proposed one gate too early.
- **Fix:** separate compatibility, runner-validation and scientific-execution statuses; require the complete runner and machine-readable readiness report before target evaluation. Relevant records: commits `9663702`, `6b2a8ad`.
- **Regression test:** runner-validation artifacts must explicitly report implementation readiness separately from scientific authorization; sealed targets must remain unopened during synthetic validation.

### CT-002 — Protocol requirements were treated as frozen before all decisions were specified

- **Class:** confirmed protocol-integrity bug; fixed.
- **Observed:** search budgets, checkpoint selection, accounting and other decisions remained unresolved after the nominal protocol was called frozen.
- **Impact:** post-hoc choices could have changed the scientific result.
- **Fix:** record a dated amendment with machine-readable values and source hashes before execution (`6b2a8ad`).
- **Regression test:** preflight requires complete protocol fields and refuses execution when an unresolved requirement remains.

### CT-003 — Operator records stored root signatures instead of executable motifs

- **Class:** confirmed implementation bug; fixed in the cumulative-adaptation branch.
- **Observed:** verified source experience could not construct a target candidate; records only affected root-name ranking.
- **Impact:** the claimed source → validation → promotion → future-proposal path was absent.
- **Fix:** canonical typed-hole AST motifs, operator-specific validation, executable candidate construction and proposal-order instrumentation (`a507c41` and subsequent exploratory transfer work).
- **Regression test:** a synthetic verified motif must activate and change candidate order; negative and deterministic-replay fixtures must also pass.

### CT-004 — Training competence was conflated with unfamiliar-procedure acquisition

- **Class:** confirmed scientific interpretation bug; fixed in reporting.
- **Observed:** B3 covered 64/64 training targets but acquired 0/48 functionally novel targets.
- **Impact:** fitting known targets was at risk of being reported as procedural learning.
- **Fix:** report training coverage, functional novelty, abstention, post-reset execution and baseline comparison separately; freeze the negative result.
- **Regression test:** every induction report must include disjoint functional truth-table checks and a target denominator.

### CT-005 — FP16 continuation produced non-finite state and a corrupted final checkpoint

- **Class:** confirmed numerical/infrastructure bug; fixed for the BF16 continuation.
- **Observed:** values became non-finite after the intact region near steps 47,000–48,000, while monitoring continued to describe the run as healthy.
- **Impact:** the terminal artifact was not recoverable and nine hours of work were placed at risk.
- **Fix:** finite checks for logits, losses, gradients, parameters and optimizer states; loss-scale retry budget; atomic checkpoint writes; reject non-finite checkpoints; resume from verified step 45,000 and use BF16. Evidence: `OPTIMIZATION_AND_RECOVERY_REPORT_20261030.md` and `run_crucible_foundation_training_20261029.py`.
- **Regression test:** injected non-finite forward values must produce `NONFINITE_ABORT.json`; a corrupted checkpoint must be rejected before resume.

### CT-006 — Checkpoint writes and recovery were not initially fail-closed

- **Class:** confirmed checkpoint-integrity bug; fixed.
- **Observed:** checkpoint validity was inferred from the existence of a file rather than finite model/optimizer tensors and complete progress state.
- **Impact:** a damaged artifact could be selected as the next training source.
- **Fix:** validate before commit, write through a temporary file and atomic replace, persist training-token count, loss-scale state, optimizer state and CPU/CUDA RNG state.
- **Regression test:** save/reload/validate must check every floating model and optimizer tensor and required state fields.

### CT-007 — Short benchmark throughput used a cumulative token counter

- **Class:** confirmed measurement bug; fixed.
- **Observed:** a resumed checkpoint's cumulative token count was divided by a short trial duration, inflating tokens/second.
- **Impact:** optimization decisions and ETA estimates were biased.
- **Fix:** record `training_tokens_start`, `training_tokens_this_run`, `steps_this_run` and calculate throughput from the per-run delta. Evidence: `OPTIMIZATION_AND_RECOVERY_REPORT_20261030.md`.
- **Regression test:** benchmark reports must satisfy `training_tokens_this_run == end - start` and use that delta for throughput.

### CT-008 — VRAM boundary and resource failure were not represented consistently

- **Class:** confirmed resource/accounting risk; partly fixed.
- **Observed:** microbatch 11/12 OOM trials demonstrated a hard memory boundary; allocated and reserved VRAM were easy to confuse, and a runner could terminate without a structured resource artifact.
- **Impact:** unsafe configurations could be retried blindly and memory estimates could be overstated.
- **Fix:** record allocated and reserved peaks separately; use bounded microbatch trials; write `RESOURCE_ABORT.json` and stop on CUDA OOM.
- **Regression test:** an intentional OOM fixture must produce a structured abort and no “successful” checkpoint.

### CT-009 — Retention runner held unnecessary teacher and activation graphs on the GPU

- **Class:** confirmed implementation bug discovered during this run; fixed in the working runner.
- **Observed:** the first A0 attempt kept student, teacher, optimizer and simultaneous train/replay autograd graphs; it OOMed on the 8GB card before a scientific update.
- **Impact:** the authorized retention experiment could not start and left only an empty curve artifact.
- **Fix:** precompute frozen teacher logits/hidden states into a CPU BF16 cache, release the teacher, and backpropagate ordinary and replay losses sequentially before one optimizer step. Add structured OOM abort handling.
- **Regression test:** A0 completed the full 5,000-update continuation without OOM; the cache path was deterministic and replay targets matched the frozen teacher outputs.

### CT-010 — Retention runner discarded its target tensor before accounting valid tokens

- **Class:** confirmed bookkeeping bug discovered during this run; fixed in the working runner.
- **Observed:** after the sequential-backward memory fix, the target tensor was released before `valid_tokens` was counted.
- **Impact:** the run stopped with an `UnboundLocalError`; no scientific result was produced.
- **Fix:** count valid targets before releasing the tensor.
- **Regression test:** all four arms emitted finite curve records with positive valid-token counts and completed to step 20,000.

### CT-011 — Evaluation aggregates hid answer-level regressions

- **Class:** confirmed evaluation-design bug; fixed in diagnostics.
- **Observed:** aggregate accuracy alone concealed 33 correct-to-wrong and 19 wrong-to-correct transitions between the step-15,000 and step-59,300 checkpoints.
- **Impact:** a lower score could be misread as mere noise or a higher score as broad improvement.
- **Fix:** use the identical 304-record manifest, preserve generated text/token IDs and termination reasons, and report paired transitions by domain. Evidence: `REASONING_TRAJECTORY_COMPARISON_15000_59300.md`.
- **Regression test:** every checkpoint comparison must emit per-record hashes, target, extracted answer, termination reason and transition counts.

### CT-012 — Fixed-depth recursion was destructive and external OP was over-interpretable

- **Class:** confirmed scientific behavior; not a software defect.
- **Observed:** fixed depth 2 collapsed from 9/304 to 6/304 at later training and caused many correct-to-wrong transitions; external OP made only small net gains.
- **Impact:** “more reasoning steps” could be mistaken for safer reasoning.
- **Fix:** do not promote recursion based on aggregate accuracy; require preservation/repair transition analysis and a conservative acceptance rule.
- **Regression test:** OP evaluations must report correct-preserved, correct-lost, wrong-repaired and wrong-unchanged cells.

### CT-013 — Continued foundation training caused capability regression despite lower language loss

- **Class:** confirmed scientific regression; mechanism partly localized.
- **Observed:** one-pass reasoning declined from 52/304 at step 15,000 to 38/304 at step 59,300. Activation patching implicated output mapping for much of BBH and late representation drift for ETHICS.
- **Impact:** language-loss improvement was not sufficient evidence of cumulative competence.
- **Fix:** introduce a protected-competency retention factorial (A0 frozen continuation, A1 output anchoring, A2 representation anchoring, A3 both) with locked evaluation and zero-regression reporting.
- **Regression test:** every promoted checkpoint must pass the protected-set gate and report new-learning performance separately; a checkpoint that preserves examples only by refusing all updates is not a success.

### CT-014 — Tiny pilots were used to make claims about substantive reasoning

- **Class:** confirmed scientific interpretation risk; fixed in planning.
- **Observed:** 542k parameters and 128 updates exercised the pipeline but scored 0/64; the OP controller terms were external or disabled.
- **Impact:** pipeline compatibility could be mistaken for a reasoning result.
- **Fix:** label smoke tests as compatibility only; require language competence, held-out reasoning and enabled-controller accounting before interpreting OP.
- **Regression test:** reports must include model size, update count, processed tokens and enabled/disabled mechanism terms.

### CT-015 — Token counts were not comparable without tokenizer context

- **Class:** confirmed measurement interpretation risk.
- **Observed:** the 120M model used a 259-token byte-like vocabulary; raw token counts and losses were compared with conventional subword-model expectations.
- **Impact:** estimates of data exposure and model quality could be misleading.
- **Fix:** report tokenizer vocabulary, sequence construction, valid-token counts and source-byte coverage with every training result.
- **Regression test:** manifest and report must hash tokenizer/configuration and distinguish padded from valid tokens.

### CT-016 — Checkpoint reload restored state but CUDA updates were not bit-identical

- **Class:** confirmed reproducibility limitation.
- **Observed:** progress, optimizer/RNG state and validation outputs restored, but final tensors differed by about 0.00274 under CUDA nondeterminism.
- **Impact:** exact bitwise replay cannot be assumed from state restoration alone.
- **Fix:** report state restoration and numerical/output equivalence separately; preserve deterministic flags and tolerances.
- **Regression test:** checkpoint round-trip must record both exact state fields and an explicit tensor/output tolerance result.

### CT-017 — Sealed-target isolation must be enforced, not merely documented

- **Class:** persistent evaluation-integrity risk.
- **Observed:** earlier protocols repeatedly required explicit checks because scientific runners and preflights were not the same artifact.
- **Impact:** target labels could leak through debugging or checkpoint selection.
- **Fix:** training runners may use source and validation data only; sealed evaluation is a separate read-only process after the final checkpoint exists.
- **Regression test:** manifest hashes and file-open audit must show no sealed target access during training; synthetic leakage fixtures must fail.

### CT-018 — Research-ledger coverage excluded relevant report names

- **Class:** confirmed bookkeeping/process bug; fixed in the ledger audit workflow.
- **Observed:** report discovery rules omitted some reports named `REPORT.md`, making the nominal 40-report ledger incomplete.
- **Impact:** new proposals risked repeating prior work.
- **Fix:** repair coverage discovery and record 47/47 coverage before Track B execution; require closest-experiment novelty checks.
- **Regression test:** coverage audit must enumerate all supported report naming patterns and fail on unindexed artifacts.

### CT-019 — Host GPU availability differed from the sandbox execution environment

- **Class:** confirmed environment boundary.
- **Observed:** the host's RTX 3060 Ti and NVIDIA driver were healthy, while an earlier sandbox process could not see the driver.
- **Impact:** a “GPU unavailable” report could be mistaken for a hardware or driver failure.
- **Fix:** run an explicit host capability check (`nvidia-smi`, `/dev/nvidia*`, framework CUDA query) before changing drivers; do not make disruptive system changes as a benchmark workaround.
- **Regression test:** record host-vs-sandbox device visibility in the run metadata.

### CT-020 — Preflight source audit did not follow a local checkpoint-helper dependency

- **Class:** confirmed preflight-tool bug discovered during this run; fixed.
- **Observed:** the first ledger preflight falsely failed the retention runner because `os.replace` lived in its imported local foundation helper rather than in the retention file itself.
- **Impact:** a safe runner could be rejected, while a naive marker-only audit would encourage weakening the atomic-write requirement.
- **Fix:** the preflight follows the known local helper import and audits the helper source as part of the runner dependency closure.
- **Regression test:** both the foundation runner and dependent retention runner pass the same source audit while sealed-test access remains absent.

## Open items that must not be silently marked fixed

- The retention factorial is exploratory and currently running; no arm has yet demonstrated preserved competence.
- A representation or output-head intervention that protects the locked test set may still overfit the replay set or suppress new learning.
- The exact numerical cause of the FP16 failure is not fully isolated; BF16 is a demonstrated safer configuration under the tested workload, not a universal proof.
- Capability preservation on a finite locked set is not proof of preservation on all inputs.

## Change protocol

When a new bug is found, append an ID with: first observed date, exact command/configuration, smallest reproduction, affected artifact, root-cause status, fix, regression test, source commit and whether any scientific result must be invalidated. Never overwrite an earlier entry. A fix is not “known” until its regression test and source hash are recorded.

