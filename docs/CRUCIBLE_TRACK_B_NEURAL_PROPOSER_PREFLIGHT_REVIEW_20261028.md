# Track B neural proposer preflight review

Date: 2026-10-28
Status: **NOT APPROVED FOR EXECUTION**

## Review target

`CRUCIBLE_TRACK_B_NEURAL_PROPOSER_PROTOCOL_20261028.md` and its machine-readable
manifest. This review resolves design concerns before implementation or GPU
allocation; it does not execute a preflight training run.

## Required resolutions

| Concern | Resolution | Status |
|---|---|---|
| B3 might only rerank an engineered catalogue | B3 primary must emit grammar-constrained ASTs directly; catalogue reranking is B3-R and cannot support an independent-proposal claim | PASS |
| Non-identifiable tasks could disappear from the denominator | Training exclusions are counted; validation/test unresolvable tasks remain and receive separate abstention metrics | PASS |
| Candidate budget could be mistaken for compute equality | Candidate evaluations, verifier/oracle calls, model passes, decoder tokens, CPU/GPU time, wall time and peak memory are separate required metrics | PASS |

## Remaining execution blockers

The following artifacts must exist before approval:

1. Frozen grammar limits and primitive test vectors.
2. Hashed train/validation/test manifests with AST and truth-table partitions.
3. A leakage audit proving that target ASTs, held-out truth tables and verifier
   outputs do not enter proposer training or checkpoint selection.
4. A direct-generation B3 implementation and a separately labelled B3-R, if
   the reranking diagnostic is retained.
5. Instrumentation that independently verifies all candidate and device-compute
   counters.
6. A 50-update preflight report showing deterministic replay and lifecycle
   interface compatibility, without selecting scientific checkpoints.

## Decision

The protocol is coherent enough for implementation review. It is not approved
for neural training, GPU allocation or scientific execution. The next permitted
step is producing the frozen manifest, grammar test vectors, leakage audit and
instrumentation plan.
