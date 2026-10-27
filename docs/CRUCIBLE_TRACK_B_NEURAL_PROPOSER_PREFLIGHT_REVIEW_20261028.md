# Track B neural proposer preflight review

Date: 2026-10-28
Status: **NOT APPROVED FOR EXECUTION**

## Review target

`CRUCIBLE_TRACK_B_NEURAL_PROPOSER_PROTOCOL_20261028.md` and its machine-readable
manifest, plus the CPU-only preflight package generated from the registered
protocol. This review resolves design concerns before implementation or GPU
allocation; it does not execute a preflight training run.

## Required resolutions

| Concern | Resolution | Status |
|---|---|---|
| B3 might only rerank an engineered catalogue | B3 primary must emit grammar-constrained ASTs directly; catalogue reranking is B3-R and cannot support an independent-proposal claim | PASS |
| Non-identifiable tasks could disappear from the denominator | Training exclusions are counted; validation/test unresolvable tasks remain and receive separate abstention metrics | PASS |
| Candidate budget could be mistaken for compute equality | Candidate evaluations, verifier/oracle calls, model passes, decoder tokens, CPU/GPU time, wall time and peak memory are separate required metrics | PASS |

## CPU-only preparation result

The registered preparation scripts generated and independently validated the
frozen package at
`experiments/crucible_track_b_neural_proposer_preflight_20261028/`.

| Check | Result |
|---|---|
| Grammar schema, operator semantics and test vectors | PASS |
| Task manifest and split counts | PASS: 56 tasks; 32 train; 12 validation; 12 test |
| Structural/function hash separation | PASS: 24 held-out novel tasks; zero train/eval overlap |
| Proposer-input leakage boundary | PASS: no target AST or truth-table hash |
| Retained unresolvable evaluation denominator | PASS: 4 tasks retained with abstention status |
| Pair partition audit | PASS: demonstrations, verifier pairs and queries disjoint |
| Direct grammar-constrained B3 implementation | PASS: emits `neural_generated` ASTs; no catalogue input |
| Runtime instrumentation | PASS: independent counter validator passed |
| 50-update compatibility preflight | PASS: CPU-only, training-only, deterministic replay |
| Neural scientific training, checkpoint selection, sealed-test access | NOT RUN |

The independent validator is
`tools/validate_crucible_track_b_neural_proposer_preflight_20261028.py`.
The compatibility validator is
`tools/validate_crucible_track_b_neural_proposer_compatibility_preflight_20261028.py`.

## Compatibility result

The implementation-only preflight is recorded at
`experiments/crucible_track_b_neural_proposer_compatibility_preflight_20261028/`.
It ran 50 optimizer updates on 32 training tasks using CPU only. It emitted
128 direct AST candidates, all independently grammar-valid, with
`candidate_source=neural_generated`. Same-seed replay matched the state,
training trace and proposal outputs. The lifecycle smoke fixture also passed
schema validation, persistence reload, post-reset execution, externally
triggered invalidation and successor verification.

These are compatibility results only. They do not establish procedural
acquisition, generalisation, lifecycle success for B3, or an advantage over
B1.

## Four-arm runner validation

The validation-first runner is recorded at
`experiments/crucible_track_b_scientific_runner_validation_20261028/`.
On a synthetic training-only fixture it exercised B0, B1, B2 and B3, including
proposal provenance, grammar validation, candidate/verifier counters,
persistence reload, post-reset execution, externally triggered invalidation,
successor verification and deterministic replay. The independent runner
validator passed.

The runner's scientific mode refuses to start while the protocol audit is
incomplete. It reports these fields as not frozen in the machine-readable
protocol: B1 candidate/verifier budgets, B2/B3 architecture details, optimizer
settings, contradiction and interference fixtures, sealed-test source, and
checkpoint-selection rule. The synthetic runner's defaults are not scientific
defaults.

## Remaining approval boundary

The three registered implementation blockers and the four-arm runner
validation are complete. Scientific execution remains withheld because the
protocol is not yet exact enough to reproduce the registered comparison and
the protocol's `run_approved` flag remains false. The missing fields must be
frozen without changing the hypotheses, manifest, success criteria or resource
limits before any GPU allocation.

The compatibility run used training-only verifier fixtures after proposal
emission. It did not select a scientific checkpoint, read held-out oracle
outputs or access the sealed test set.

## Decision

The CPU-only preparation, implementation-compatibility and runner-validation
gates are complete and passing. The protocol remains `UNTESTED` and is not
approved for neural training, GPU allocation or scientific execution. The next
permitted step is to freeze the missing machine-readable protocol fields and
perform a separate approval review.
