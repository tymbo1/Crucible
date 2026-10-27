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
| Instrumentation plan | PASS as a plan; runtime implementation pending |
| Neural training, checkpoint selection, sealed-test access | NOT RUN |

The independent validator is
`tools/validate_crucible_track_b_neural_proposer_preflight_20261028.py`.

## Remaining execution blockers

The following items must be completed and reviewed before approval:

1. A direct-generation B3 implementation and a separately labelled B3-R, if
   the reranking diagnostic is retained.
2. Runtime instrumentation that independently verifies all candidate and
   device-compute counters.
3. A leakage audit proving that target ASTs, held-out truth tables and verifier
   outputs do not enter proposer training or checkpoint selection.
4. A 50-update compatibility report showing deterministic replay and lifecycle
   interface compatibility, without selecting scientific checkpoints.

## Decision

The CPU-only preparation gate is complete and passing. The protocol remains
`UNTESTED` and is not approved for neural training, GPU allocation or scientific
execution. The next permitted step is implementation review of direct B3,
runtime instrumentation and the 50-update compatibility preflight.
