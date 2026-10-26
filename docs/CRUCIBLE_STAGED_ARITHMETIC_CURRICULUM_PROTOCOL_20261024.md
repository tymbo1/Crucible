# Staged Arithmetic Curriculum versus Direct Training

Date: 2026-10-24  
Status: frozen before training  
Hardware: CPU only

## Objective

Test whether a staged arithmetic curriculum helps the current dense model learn two-digit addition compared with the same total number of updates on the final mixed task from the beginning.

The factorized digit representation is held fixed because it was the stronger interface in the preceding carry diagnostic:

`a_tens, a_ones, b_tens, b_ones -> decimal sum`

## Curriculum arm

Each stage receives 1,000 updates and retains earlier examples:

1. No-carry addition.
2. No-carry plus ones-place carry.
3. No-carry plus ones-place carry plus cascading carry.

Stage promotion is recorded against a preregistered gate of at least 95% training exact accuracy and at least 90% validation exact accuracy. If a gate fails, later stages remain diagnostic and are not treated as established curriculum advancement.

## Direct arm

The direct control receives the full three-regime training distribution from step 1 for 3,000 updates. It uses the same architecture, optimizer, seed set, total update count, factorized representation and final validation distribution as the curriculum arm.

## Controls

- Seeds `42` and `1337`.
- Checkpoints at each stage boundary and final direct steps.
- Deterministic addition oracle and split audit before training.
- No pair overlap between training and validation.
- All three regimes represented at every relevant stage.
- Legal digit coverage audited in aggregate training data.
- Original sealed test set is not opened.
- The successful successor checkpoint from the preceding operand-representation experiment is preserved and hash-referenced, not modified or used to initialize this run.
- No GPU, E4 routing, cloud resources or architecture changes.

## Measurements

Report training and validation exact/token accuracy by stage, carry regime and output length; gate status at each promotion point; total updates, elapsed CPU time and checkpoint hashes.

## Interpretation

- High train and validation accuracy for curriculum but not direct supports a curriculum benefit.
- Similar performance supports no detected curriculum benefit at this budget.
- Low training accuracy in both arms means the arithmetic task remains underfit and does not support a transfer conclusion.
- A stage-specific improvement must be interpreted alongside its training-fit gate; validation alone is insufficient.

This experiment does not test general intelligence, recursive reasoning, meta-learning or neuromorphic learning.
