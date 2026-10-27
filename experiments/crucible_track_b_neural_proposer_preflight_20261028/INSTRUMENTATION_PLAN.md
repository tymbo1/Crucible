# Track B proposer instrumentation plan

Status: plan complete; implementation pending.

Every task must record candidate source/counts, verifier and oracle calls, proposer passes, decoder tokens, CPU/GPU/wall time and peak memory. Lifecycle events are recorded separately and aggregated per task, split, stratum, seed and arm. The 50-update compatibility preflight may not select a scientific checkpoint or consume held-out oracle outputs.
