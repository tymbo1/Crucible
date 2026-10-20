# GPU Neural SSRE v1 protocol

## Frozen contract

- Crucible context-128 package;
- Branch B, E16, top-2, four MoE layers;
- seed 42 for the initial gate;
- exact frozen schedule and input assembly;
- BF16 autocast, same optimizer and learning rate as the frozen contract;
- no telemetry in the timing path, so v5 telemetry overhead is excluded;
- v4-matched deterministic CUDA controls, with aggregate routing `bincount`
  allowed to warn because PyTorch has no deterministic CUDA implementation;
- all variants start each trial from the same immutable anchor.

## Execution variants

`original` is imported from the frozen model implementation.

`grouped_rank_preserving` computes the same routing probabilities and top-k
indices, forms one GPU assignment list sorted by `(expert, rank, token)`, and
keeps the frozen per-expert/per-rank input shapes. It uses GPU-resident token
indices and one segment-length synchronization for the deterministic Python
loop.

`grouped_rank_largest_first` uses the same rank-preserving representation but
orders expert groups by descending selected-token count. It tests whether a
deterministic workload order changes execution time without changing the
mathematical route.

`grouped_sorted` combines both routing ranks into one call per expert. It is
included as an exploratory arm, but changing the GEMM shape can change BF16
kernel tiling; it must pass the equivalence gate before it can be treated as a
valid baseline.

`batched_padded` stacks the expert weights and evaluates all expert slots with
batched matrix multiplies, padding each expert to the largest group in the
microbatch. Padding waste, stacking and final scatter are measured explicitly.

## Equivalence gate

For each grouped variant, a disposable one-step probe compares:

- per-microbatch logits;
- scalar losses;
- accumulated parameter gradients;
- model parameters after one identical optimizer step.

Declared tolerances:

- logits: `5e-3` absolute;
- loss: `1e-5` absolute;
- gradients: `5e-4` absolute;
- post-step parameters: `5e-4` absolute.

An implementation that exceeds these limits is reported as numerically
non-equivalent and cannot be accepted as a speed baseline.

## Performance gate

Each arm uses warmed repeated optimizer steps and records CPU assembly, H2D,
forward host/CUDA, backward host/CUDA, optimizer, wall time, tokens/sec and
peak VRAM. No profiler is active during timing. The benchmark is repeated on
the same real frozen workload; a separate synthetic imbalance fixture is used
only to exercise the grouping code and is never mixed into the scientific
timing result.

The timing result must distinguish deterministic speedup from numerical drift.
No neural scheduler or cloud experiment is authorized by this package.

