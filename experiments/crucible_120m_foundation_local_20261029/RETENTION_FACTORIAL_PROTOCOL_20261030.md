# Crucible retention factorial: frozen protocol

Status: exploratory run authorized; target evaluation remains sealed until each arm finishes.

## Starting point and budget

- Starting checkpoint: Crucible 120M step 15,000 checkpoint.
- End step: 20,000, exactly 5,000 continuation updates per arm.
- Precision: BF16 on the RTX 3060 Ti, identical across arms.
- Microbatch: 8; gradient accumulation: 1; effective batch: 8.
- Seed, data order, learning rate, reasoning cadence and optimizer state are identical across arms.
- Training data: the existing FineWeb-Edu training pack and reasoning training split.
- Replay data: the existing reasoning validation split, used only for retention anchoring; the sealed 304-record test set is excluded from training.
- Replay compute is executed in every arm. A0 computes the same replay path with zero retention weights.

## Arms

| Arm | Head weight | Representation weight | Description |
| --- | ---: | ---: | --- |
| A0 | 0.00 | 0.00 | Ordinary continuation with matched replay compute |
| A1 | 0.05 | 0.00 | Output-logit distillation to the step-15,000 teacher |
| A2 | 0.00 | 0.05 | Final-state representation anchoring to the step-15,000 teacher |
| A3 | 0.05 | 0.05 | Both retention mechanisms |

Head anchoring is temperature-2 KL distillation at non-padding replay positions. Representation anchoring is cosine-normalized final-state MSE at the same positions. Weights are frozen before execution.

## Evaluation

After all arms finish, run the existing sealed-test evaluator on each final checkpoint and compare:

- one-pass accuracy and domain accuracy;
- regressions relative to the step-15,000 checkpoint;
- new improvements relative to the step-15,000 checkpoint;
- fixed-depth and external OP behavior;
- language/reasoning validation loss;
- training throughput, VRAM and replay-anchor losses.

The protected-set gate is zero observed regressions on the locked test manifest, but that gate does not by itself establish generalization. New-learning performance and validation behavior must be reported alongside retention.
