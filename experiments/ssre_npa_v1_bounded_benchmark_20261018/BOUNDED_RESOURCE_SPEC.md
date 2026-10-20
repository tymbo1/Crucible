# SSRE-NPA v1 bounded-resource specification

## Problem found in v0

The v0 benchmark bounded three individual queues, but it did not bound the sum of:

```text
input queue + completed queue + reorder buffer + staging queue + consumer lag
```

Consequently, a queue capacity of four could coexist with a reorder buffer of 63. The v1 contract treats every dispatched-but-unconsumed packet as one global in-flight unit.

## Global bound

For a configured window `W`:

```text
0 <= dispatched_packets - consumed_packets <= W
```

The producer must acquire one permit before dispatching a packet. The consumer releases that permit only after the packet has passed ordered consumption and integrity checks. Retries retain the original permit; they do not create additional in-flight work.

The same accounting tracks estimated payload bytes:

```text
0 <= in_flight_payload_bytes <= configured_payload_window
```

The synthetic benchmark reports the observed maxima and asserts the count bound. Real adapters should also enforce a byte budget before allocating large prepared payloads.

## Blocking semantics

The benchmark uses condition variables. A full queue or exhausted window calls `Condition.wait()` and records the duration of the wait episode. It does not repeatedly call `put_nowait()` or sleep-and-retry in a busy loop.

Reported fields:

- wait-event count;
- total blocked duration;
- maximum blocked duration;
- p50 and p95 blocked duration;
- queue and global-window high-water marks.

## Deterministic modes

- **serial:** one worker and no concurrency, reference digest;
- **fixed:** all configured workers may prepare packets concurrently;
- **adaptive:** the same maximum worker pool is available, but a deterministic condition-based gate changes the active worker limit from queue telemetry. It cannot change packet order.

The adaptive policy is deliberately heuristic. It is not a neural controller.

## Failure model

The benchmark injects a transient preparation failure for a chosen packet. The worker emits a failure event; the assembler retries the same packet identity with a bounded attempt count. The packet's global in-flight permit is retained across the retry. Exhausted retries fail closed.

Cancellation closes all channels and wakes blocked producers, workers, assemblers, and consumers. A recovery run processes every uncommitted packet. The combined committed result digest must equal the uninterrupted serial reference, with no duplicate sequence IDs.

## Interpretation boundary

This benchmark establishes scheduler mechanics and resource accounting only. It does not establish a CPU speedup for neural training, a CUDA performance improvement, or a benefit from a neural scheduler.
