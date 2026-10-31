# Crucible source-training diagnostic — final report

Status: **completed; execution positive-control gate failed**
Classification: **INCONCLUSIVE PREREQUISITE FAILURE**

## Result

Both seeds fit the 16 public execution-training procedures, but neither passed
the frozen held-out execution gate of 14/16 correct with at least 15/16
coverage.

| Seed | Execution train | Execution validation | Coverage | Majority baseline | Gate |
|---:|---:|---:|---:|---:|---|
| 42 | 16/16 | 7/16 | 16/16 | 10/16 | FAIL |
| 1337 | 16/16 | 9/16 | 16/16 | 11/16 | FAIL |

The runner therefore stopped before the induction phase. No source-induction
training, C0--C3 training, Track-B execution or target evaluation occurred.

## Category measurements at the fixed update-500 checkpoint

| Seed | Source validation | Function validation | Execution test | Source test |
|---:|---:|---:|---:|---:|
| 42 | 17/32 | 11/16 | 8/16 | 20/32 |
| 1337 | 10/32 | 11/16 | 10/16 | 16/32 |

These are descriptive replay measurements from the execution-trained model;
they are not induction results. Primary and forced-binary accuracies were
identical because every final row produced a valid binary token.

## Retention and transitions

The frozen checkpoint abstained on every primary-decoder row. Consequently the
reported zero correct-to-wrong regressions are vacuous: there were no initially
correct protected instances. Final transitions were abstain-to-correct versus
abstain-to-wrong:

| Seed | Execution validation | Execution test | Source validation | Function validation | Source test |
|---:|---:|---:|---:|---:|---:|
| 42 | 7 / 9 | 8 / 8 | 17 / 15 | 11 / 5 | 20 / 12 |
| 1337 | 9 / 7 | 10 / 6 | 10 / 22 | 11 / 5 | 16 / 16 |

## Integrity

- Device: RTX 3060 Ti, BF16 autocast.
- Execution budgets: 500 updates per seed; checkpoints at 0, 50, 100, 250
  and 500.
- Eight saved checkpoints loaded successfully and all model/optimizer tensors
  were finite.
- No numerical or resource failure occurred.
- Target and sealed-answer access remained false.

## Scientific conclusion

Explicit-procedure execution was **not established** under the registered
configuration. Query-identifiable induction, function-identifiable induction
and productive retention remain **untested** because the positive-control gate
failed. This result does not falsify neural procedure learning generally; it
shows that this checkpoint and answer-position fine-tuning budget did not
produce a transferable execution skill.
