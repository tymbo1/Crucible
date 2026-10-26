# Crucible operation-position and isolated-second-step protocol

Protocol frozen: 2026-10-24

## Question

Is the weak second step caused by primitive operation learning, operation position, or integration of a generated intermediate state?

## Conditions

Each condition uses the same dense control architecture, seed, optimizer, update budget and newly generated diagnostic task families.

1. **A→B standard composition**

   Two operations are presented in order. The target contains the intermediate and final digits. Free-running evaluation must generate the intermediate before the final.

2. **B-only isolated second operation**

   The correct intermediate digit is supplied in the prompt. The model applies only operation B and predicts the final digit. This is a primitive/conditional second-operation control, not an autonomous two-step task.

3. **B→A reversed composition**

   The same operation families are executed in the reverse order. The target contains the reversed intermediate and final digits.

## Data design

Operations are `add`, `sub` and `mul`, with operands 1–4 and modulo-ten arithmetic.

Training base pairs contain seven of the nine ordered pairs:

`add+add`, `add+mul`, `mul+add`, `mul+mul`, `sub+add`, `sub+mul`, `sub+sub`.

Held-out base pairs are:

`add+sub`, `mul+sub`.

Every operation appears in both positions in the training pair set. The held-out pairs are not present in the original Crucible composition train, validation, held-out or test pair lists. All diagnostic prompts use a separate versioned prompt family and newly generated records; original benchmark artifacts remain untouched.

For A→B and B→A:

- 16 records per training pair: 112 training records.
- 8 records per training pair: 56 validation records.
- 32 records per held-out pair: 64 held-out records.

For B-only:

- 16 records per operation: 48 training records.
- 8 records per operation: 24 validation records.
- 8 records per operation: 24 held-out records.

Train, validation and held-out inputs are disjoint within each arm. The held-out split is diagnostic only and is not the original sealed test split.

## Training and evaluation

- Seeds: 42 and 1337.
- Dense control: 21,968,640 parameters.
- CPU only.
- 2,000 continuous updates per arm.
- Checkpoints: 0, 500, 1,000, 1,500 and 2,000.
- Fresh model per condition and seed; paired initialization procedure.
- No GPU, cloud, pretrained checkpoints or architecture changes.
- No checkpoint selection using held-out outcomes.

Report separately:

- teacher-forced sequence and token accuracy;
- free-running intermediate, final and end-to-end accuracy;
- B-only final accuracy;
- results by operation pair and operation type;
- validation versus held-out performance;
- training exposure and runtime.

The 90% B-only figure is an interpretive marker for strong primitive competence, not a claim of success unless replicated across both seeds and splits. The primary scientific comparison is the gap between B-only conditional performance and free-running A→B/B→A composition.

## Safeguards

- Independent modulo-ten oracle checks every generated record.
- Immutable train/validation/held-out manifests and hashes.
- No reuse of the original sealed test records.
- No evaluation-set tuning during execution.
- Preserve failed hypotheses and all checkpoint trajectories.
