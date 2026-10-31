# CNSS artifact-completeness root cause

Date: 2026-11-01
Classification: **ARTIFACT_COMPLETENESS_FAILURE_CONFIRMED; DETERMINISTIC RECONSTRUCTION VERIFIED**

## Root cause

The CNSS runner created and trained the frozen `Workspace` in memory, but its
checkpoint payload persisted only `workspace_digest`. It persisted the learned
adapter, optimizer state and RNG state, but not the encoder/decoder
`state_dict`.

The checkpoint integrity check compared the saved digest with the still-live
workspace. It did not test whether the workspace could be loaded from disk.

## Why the 210 replay checks passed

The replay audit called `train_workspace()` at the start of a fresh process.
That deterministically rebuilt the workspace using the same source code, seeds,
training schedule and CPU runtime. It then loaded each adapter checkpoint and
compared it against that regenerated workspace.

Therefore the old result established adapter replay against a regenerated
workspace. It did not establish complete checkpoint recovery from persisted
artifacts. The original CNSS scientific result and classification are not
changed.

## Local recovery outcome

The original serialized workspace bytes were not found. The deterministic
reconstruction nevertheless produced the recorded full state digest:

`a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c`

This is recorded as a reconstruction, not falsely labelled as recovery of the
original file.

## Artifact created

The complete frozen workspace is now stored under:

`experiments/crucible_track_a_cnss_semantic_interface_compatibility_v1_rerun_contractfix_20261101/workspace_artifacts/a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c/`

The bundle contains `state.pt`, `architecture.json`, `serialization.json` and
`HASHES.json`. The broad `*.pt` ignore rule now explicitly allows these
content-addressed workspace artifacts to be versioned.

## Verification

The independent verifier loaded the bundle in a new process, checked file,
architecture and tensor hashes, loaded all **210** dependent adapter
checkpoints, reproduced all **210** recorded probe outputs, and passed negative
tests for missing, corrupted and incompatible workspace artifacts.

Verification report:

`experiments/crucible_track_a_cnss_semantic_interface_compatibility_v1_rerun_contractfix_20261101/WORKSPACE_ARTIFACT_VERIFICATION_REPORT.json`

No adapter training, CNSS replication or GPU work was performed during this
repair.

## Remaining limitation

The artifact is now independently loadable and cryptographically equivalent by
the recorded state digest, but the original workspace serialization event was
not preserved. Future experiments must save the workspace bundle before any
dependent training begins and must perform fresh-process replay without
regenerating it.
