# SSRE-NPA v0 profiling and experiment plan

## Gate 0 — static contract checks

- Verify manifest and code hashes before constructing packets.
- Confirm deterministic packet IDs, sequence indices, and sample ordering.
- Confirm every queue has a finite capacity.
- Confirm malformed packets and duplicate/missing sequence IDs fail closed.
- Confirm a scheduler decision cannot modify model, optimizer, seed, or data order.

## Gate 1 — CPU-only synthetic benchmark

Run `synthetic_benchmark.py` with repeated seeds and queue sizes. Required checks:

- all packets consumed exactly once;
- output sequence exactly equals the input sequence;
- output digest is stable across repeated runs;
- bounded queue high-water marks never exceed capacity;
- backpressure is observable when the consumer is deliberately slowed;
- worker completion order can differ without changing output order;
- no GPU, model, network, or paid service is touched.

The benchmark is mechanical validation only. Its throughput is not a neural-training performance claim.

## Gate 2 — current-run trace replay, read-only

After the active experiment is sealed, replay representative packet metadata from a copy of its schedule. Do not launch training. Measure:

- source lookup time;
- tokenization/packing time;
- target/mask assembly time;
- serialization/deserialization time;
- queue wait time;
- host-to-device copy time if a disposable harness is used;
- CUDA synchronization points;
- forward/backward/optimizer time;
- GPU active versus idle intervals.

Use CUDA events or an approved profiler with explicit synchronization. A Python process at 100% CPU is not enough evidence that data preparation is the bottleneck.

## Gate 3 — five-arm engineering comparison

Keep the neural workload fixed and compare:

| Arm | Description |
|---|---|
| A | unchanged runner |
| B | precomputed/indexed/memory-mapped input |
| C | persistent deterministic CPU reservoir |
| D | SSRE-inspired heuristic scheduling |
| E | D plus offline-trained neural controller |

Use short repeated trials, warm-up, randomized arm order, and a separate disposable output directory. Do not use the active scientific run as a benchmark sandbox.

Primary endpoint: valid end-to-end training tokens per second.  
Secondary endpoints: GPU wait fraction, update wall time, CPU contention, H2D time, queue stalls, host/VRAM peak, controller overhead, and reproducibility.

## GPU coexistence safeguards

- Reserve one or more CPU cores for the GPU coordinator path where profiling supports it.
- Do not saturate every logical CPU by default.
- Add an explicit CPU-load versus GPU-throughput arm.
- Record GPU utilization and clocks, but do not infer causality from utilization alone.
- Randomize trial order and include settling time to avoid the historical warm-up/order bias.

## Adoption rule

Adopt E only if it beats D reproducibly after controller cost, maintains exact packet ordering and integrity, and does not increase GPU wait or memory pressure beyond preregistered limits. Otherwise retain D or B as the simpler implementation.
