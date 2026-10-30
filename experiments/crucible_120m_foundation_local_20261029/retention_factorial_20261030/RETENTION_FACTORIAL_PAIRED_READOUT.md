# Crucible 120M retention factorial — paired readout

Classification: exploratory; all arms started from the untouched step-15,000 checkpoint and trained to step 20,000. The sealed 304-record manifest was opened only after training.

## Final policy scores

| Condition | One pass | Fixed depth 2 | External OP | Tokens/s | Peak allocated VRAM | Safety |
|---|---:|---:|---:|---:|---:|---|
| A0 | 39/304 | 16/304 | 39/304 | 2634.5 | 5.19 GB | PASS |
| A1 | 43/304 | 3/304 | 43/304 | 2635.3 | 5.19 GB | PASS |
| A2 | 42/304 | 1/304 | 42/304 | 2630.7 | 5.19 GB | PASS |
| A3 | 45/304 | 2/304 | 50/304 | 2642.1 | 5.19 GB | PASS |

## Paired one-pass transitions versus step 15,000

| Condition | Correct→Wrong | Wrong→Correct | Correct→Correct | Wrong→Wrong | Net change |
|---|---:|---:|---:|---:|---:|
| A0 | 26 | 13 | 26 | 239 | -13 |
| A1 | 12 | 3 | 40 | 249 | -9 |
| A2 | 13 | 3 | 39 | 249 | -10 |
| A3 | 10 | 3 | 42 | 249 | -7 |

## Interpretation

- The protected one-pass gate is zero correct→wrong transitions relative to step 15,000; passing it requires preserving the tested competency, not merely improving aggregate accuracy.
- Aggregate score changes are not treated as causal evidence until paired transitions and domain distributions are considered.
- A3 external OP is an inference-time result and does not establish that the retention loss improved the learned model.
- This run tests a five-thousand-update continuation, one seed and a finite protected manifest; it is not evidence of lifelong retention or universal capability preservation.
