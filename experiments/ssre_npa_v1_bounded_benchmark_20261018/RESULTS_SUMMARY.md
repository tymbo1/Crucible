# SSRE-NPA v1 bounded-resource benchmark results

Run date: 2026-10-18 (Australia/Brisbane)

The full CPU-only suite used 192 deterministic packets, four workers, queue
capacity four, and a global in-flight window of 16 packets / 4 MiB estimated
payload. It exercised uniform, mixed-cost, straggler, and large-payload
workloads in serial, fixed-worker, and adaptive deterministic modes.

## Integrity result

All five required checks passed:

- serial/fixed/adaptive result digests matched;
- global in-flight packet bounds held;
- reorder buffers stayed within the global window;
- transient fault injection was observed and recovered by retry;
- cancellation plus recovery covered every sequence exactly once.

The synthetic workload manifest digest was:

`5804127a7d7a469cd92556f51fd29aa9344b2269b10e44400a2d35d99c16bfb2`

Peak process RSS for the complete run was 22,048 KiB. The benchmark used no
CUDA, PyTorch, Crucible model, training, cloud instance, or credentials.

## Standard workload timings

Throughput is committed packets/second. `reorder` is the maximum completed
out-of-order packet count, and `window` is the maximum dispatched-but-
unconsumed packet count.

| workload | fixed time (s) | fixed pkt/s | adaptive time (s) | adaptive pkt/s | fixed reorder/window | adaptive reorder/window |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 0.006149 | 31,223 | 0.006549 | 29,317 | 1 / 9 | 1 / 9 |
| mixed | 0.204112 | 941 | 0.204469 | 939 | 12 / 16 | 12 / 16 |
| stragglers | 0.208448 | 921 | 0.208564 | 921 | 12 / 16 | 16 / 16 |
| large | 0.206644 | 929 | 0.206516 | 930 | 11 / 16 | 15 / 16 |

Blocking was condition-variable based, not busy-spin retry. For example, the
mixed fixed-worker case recorded 175 global-window wait episodes totaling
0.183962 seconds; the corresponding adaptive case recorded 175 episodes
totaling 0.184376 seconds. Queue-level wait counts and p50/p95 durations are
stored in `benchmark_results.json`.

The current CPU utilization metric is process CPU seconds and
process-CPU/core-equivalent. Per-logical-core attribution was not collected in
this gate and remains a separate profiling extension.

## Fault, cancellation, and recovery

The injected transient failure was observed once, retried once, and produced
the exact serial digest with no missing or duplicate packets. Cancellation
committed 48 of 192 packets; the recovery run committed the remaining 144,
and combined sequence coverage was complete with no duplicates.

## Bug found and corrected

The initial v1 retry path could deadlock: the assembler blocked putting a retry
into a full input queue while workers were blocked delivering completion
records. Retries now enter a nonblocking assembler backlog and are drained
when input capacity is available, so the assembler continues consuming
completion events. The corrected implementation passed both the 48-packet
smoke run and the 192-packet full run.

## Interpretation

The global resource-bound and deterministic recovery contract passed. This
benchmark does not demonstrate neural-training acceleration. The fixed and
adaptive scheduler timings are nearly identical on these small synthetic
workloads; a neural controller has not been implemented or claimed.
