# Crucible source-competence qualification protocol

Date: 2026-10-31  
Status: **Design frozen for review; execution not authorized**  
Protocol schema: `crucible-source-competence-qualification-v1-20261031`

## Purpose

This is a bounded source-only qualification run for the repaired curriculum. It
does not open C0--C3 and does not test productive retention. Its purpose is to
establish whether the model can execute an explicit procedure, infer a query
answer from sufficient demonstrations, and identify an underlying function.

The 32 function-identifiable procedures are a pilot-sized diagnostic. Passing
this protocol cannot by itself establish general function induction; results
must be reported per seed, per procedure, with uncertainty and transition
tables.

## Frozen inputs and boundaries

- Public manifest: `experiments/crucible_source_curriculum_repair_20261031/MANIFEST.json`
- Manifest SHA-256: `699c7a2f82a5ff630ee59df689916131ec4eabd5de1513cdf4665f15141e055c`
- Public hash manifest SHA-256: `c5450b6a50574a24a146a557d611564954edb22451c6877f9ce6c96662608f0c`
- Seeds: `42`, `1337`.
- Training input: only `source_train.jsonl` and `execution_train.jsonl`.
- Validation: `source_validation.jsonl`, `source_function_validation.jsonl` and `execution_validation.jsonl`.
- Novel held-out evaluation: `source_test.jsonl` and `execution_test.jsonl`.
- Private source oracles remain outside Git and are never supplied to training.
- No target manifests, target answers or C0--C3 data may be loaded.

The public model-facing rows contain no validation/test answer labels or
compatibility diagnostics. AST and truth-table hashes are audit metadata, not
model inputs.

## Three capability categories

### Explicit-procedure execution

The canonical AST is supplied. The model must execute it on a new query. This
isolates execution from induction.

### Query-identifiable induction

The AST is hidden. Demonstrations are selected so every grammar-consistent
function agrees on the held-out query. This tests answer induction, not
necessarily recovery of the complete function.

### Function-identifiable induction

The AST is hidden. Fifteen of sixteen truth-table rows are supplied and the
remaining query is held out. Exactly one function in the registered grammar is
compatible with the demonstrations. This tests procedure identification, but
the small 16-task-per-seed sample requires replication before a broad claim.

All held-out procedures are structurally and functionally disjoint from
training procedures. Query vectors are absent from their own demonstrations.

## Controlled decoder

The source gate uses one fixed answer logit position: the position predicting the
first byte after `ANSWER=`. Only byte tokens `0` and `1` are eligible. Any
non-answer top token, tie, missing position or non-finite logit is an abstention
or hard evaluator failure as appropriate. Free-running generation and
first-digit extraction are not used for the primary gate.

## Metrics

For every category, seed and checkpoint report:

- exact accuracy = correct / all eligible tasks;
- selective accuracy = correct / answered tasks;
- coverage = answered / eligible tasks;
- abstention rate;
- per-task predictions, margins and answer transitions;
- per-seed and pooled uncertainty intervals.

Abstentions count as incorrect for exact accuracy. Selective accuracy is
descriptive and cannot replace exact accuracy. A high selective score with low
coverage is not a pass.

Transitions against the immediately preceding checkpoint must include:

`correct→correct`, `correct→wrong`, `correct→abstain`, `wrong→correct`,
`wrong→abstain`, `abstain→correct`, `abstain→wrong`,
`wrong→wrong`, and `abstain→abstain`.

## Qualification thresholds

Thresholds are applied separately for each seed; pooled accuracy may not rescue a
failed seed.

| Category | Validation size / seed | Minimum exact correct | Minimum coverage |
|---|---:|---:|---:|
| Explicit execution | 16 | 14/16 | 15/16 |
| Query-identifiable induction | 32 | 26/32 | 29/32 |
| Function-identifiable induction | 16 | 13/16 | 14/16 |

The function-identification result is classified **pilot qualification** even if
both seeds pass. It becomes an established capability only after a separately
registered, larger replication.

No capability category passes if its transition table shows unexplained output
formatting or decoder mismatch. Correctness is judged against private oracle
answers, while the model never receives those answers.

## Protected-instance accounting

The held-out source-test rows are evaluated at the initial checkpoint and every
scheduled checkpoint. A row that is correct at the initial checkpoint becomes a
protected instance for transition accounting; this does not turn one correct
answer into a verified general capability.

Any protected `correct→wrong` or `correct→abstain` transition marks that
checkpoint **not promotion-eligible**. The run may continue for diagnosis, but
the last eligible checkpoint remains the accepted checkpoint. No checkpoint may
be selected retrospectively using sealed/test performance.

Capability claims require repeated family-level success, not isolated protected
instances.

## Training and checkpoint rules

- Same source examples, ordering, seeds, optimizer, learning-rate schedule,
  effective batch and update budget for every model arm.
- Evaluation checkpoints: updates `0, 250, 500, 750, 1000`.
- Non-finite loss, gradients, parameters, optimizer state or checkpoint tensors
  causes immediate termination and invalidates the affected checkpoint.
- Checkpoints use atomic writes and round-trip validation, including model,
  optimizer, scheduler and RNG state where present.
- Validation/test results cannot select checkpoints or alter training.
- The exact dense-baseline architecture manifest must be hashed before execution
  and parameter-matched to the Crucible candidate within the registered budget.
  If that manifest is missing, execution remains blocked rather than being
  improvised.

## Classification rules

- **Established under tested conditions:** all category gates pass, transition
  logs and integrity checks pass, and uncertainty is reported.
- **Supported but pilot-limited:** gates pass but the function-identification
  sample is too small for a general claim.
- **Inconclusive prerequisite failure:** a source gate fails, or coverage is too
  low to interpret; C0--C3 remain blocked.
- **Invalid:** leakage, decoder inconsistency, oracle access, protocol change,
  non-finite state or checkpoint-integrity failure.

No source-gate failure is interpreted as falsification of productive retention.

