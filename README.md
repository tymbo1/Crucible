# Crucible

Private source and reproducibility repository for the Crucible sparse-expert
architecture experiments.

## Scope

This repository contains the reproducible layer of the work:

- experiment and benchmark source code;
- protocols, design notes and reports;
- compact metrics, manifests and integrity hashes;
- SSRE-NPA and GPU Neural SSRE engineering experiments.
- CPU-only cognitive task generators, exact oracles and evaluation manifests.

Large raw corpora, packed datasets, model checkpoints and local credentials
remain outside Git. They are referenced by the local experiment records and
must be restored separately when reproducing a run.

## Current architecture reference

The main Crucible reference is the E16 Branch B model: 16 top-2 SwiGLU
experts per MoE layer, byte-level vocabulary, custom latent attention and
170,233,600 trainable parameters.

The grouped-expert execution backend is preserved as an opt-in engineering
optimization. Its sustained validation is documented separately from claims
about architectural capability or long-horizon full-state equivalence.

## Evidence policy

Frozen result artifacts are preserved as generated. Reports distinguish:

1. measured results;
2. artifact-verified claims;
3. plans and hypotheses;
4. results that still require independent reproduction.

No credential, API key or private environment file belongs in this repository.
