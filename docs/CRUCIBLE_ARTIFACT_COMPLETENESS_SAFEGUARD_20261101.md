# Crucible artifact-completeness safeguard

Date: 2026-11-01
Status: **CNSS IMPLEMENTED; GENERAL PIPELINE ADOPTION PENDING**

## Requirement

Every frozen dependency required for a forward pass must be saved as an immutable, content-addressed artifact before dependent training begins. For CNSS this includes the exact encoder, decoder, normalization and schema state, not only a digest recorded inside adapter checkpoints.

## Required artifact bundle

```text
workspace/<sha256>/state.pt
workspace/<sha256>/architecture.json
workspace/<sha256>/serialization.json
workspace/<sha256>/HASHES.json
```

`state.pt` contains the complete frozen state dictionary. `architecture.json` fixes module names, shapes, dtypes, state schema and framework version. `serialization.json` fixes the loader and tensor layout. `HASHES.json` records the file digest and per-tensor digests.

Every dependent checkpoint must reference the workspace content address, schema version and full digest. A checkpoint without a resolvable exact dependency is not recoverable.

## Acceptance gate

Before an experiment is marked complete:

1. A fresh process loads the workspace artifact.
2. It verifies file, tensor and architecture hashes.
3. It reproduces the recorded workspace digest.
4. It executes a fixed probe and compares output hashes.
5. It loads one dependent checkpoint and verifies the dependency reference.
6. The bundle is copied or archived independently of the dependent checkpoints.

Failure is an artifact-integrity failure, not a scientific result. A missing
workspace must never be silently regenerated or substituted.

## Deterministic reconstruction policy

A deterministic reconstruction may be accepted as an explicit recovery path
only when all of the following are independently verified:

1. The complete reconstructed state digest exactly matches the recorded
   content identity.
2. A fresh process loads the reconstructed artifact exclusively from disk.
3. The fixed probe outputs and every registered dependent-checkpoint output
   reproduce exactly.
4. Missing, corrupted and incompatible artifact tests fail closed.

Such a bundle must be labelled **deterministic reconstruction**, not recovery
of the original serialized bytes. The distinction must remain in the research
record.
