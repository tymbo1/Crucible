# SSRE-NPA v2 telemetry schema

Schema version: `ssre-npa-telemetry-v2`

## Event record

Each JSONL line is one event and must contain:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Exact schema identifier. |
| `source_kind` | string | `deterministic_virtual_workload_proxy` for this package; a live adapter must use a distinct measured label. |
| `run_id` | string | Stable run identifier. |
| `scenario_id` | string | Workload scenario from `WORKLOAD_MATRIX.json`. |
| `scheduler_arm` | string | Scheduler configuration used for the trace. |
| `event_id` | integer | Monotonic event sequence within the trace. |
| `packet_seq` | integer or null | Ordered packet identifier; null for scheduler-level events. |
| `worker_id` | integer or null | CPU worker, when applicable. |
| `phase` | string | `dispatch`, `prepare`, `queue_wait`, `stage`, `h2d`, `gpu_compute`, `commit`, `backpressure`, or `scheduler_decision`. |
| `start_ms` | number | Virtual monotonic start time. |
| `duration_ms` | number | Non-negative virtual duration. |
| `payload_bytes` | integer | Prepared payload associated with the packet/event. |
| `token_count` | integer | Logical token/byte work represented by the packet. |
| `cache_state` | string | `hot`, `warm`, `cold`, or `not_applicable`. |
| `queue_depth` | integer | Queue depth observed at event emission. |
| `inflight_packets` | integer | Dispatched but not committed packets. |
| `inflight_bytes` | integer | Dispatched but not committed payload bytes. |
| `worker_count` | integer | Active worker configuration. |
| `window_packets` | integer | Global in-flight packet bound. |
| `window_bytes` | integer | Global in-flight payload bound. |
| `outcome` | string | `ok`, `blocked`, `retry`, `cancelled`, or `error`. |

Optional fields include `reason`, `predicted_queue_empty_ms`,
`observed_queue_empty_ms`, `memory_rss_bytes`, `pinned_bytes`,
`gpu_util_percent`, `gpu_wait_ms`, `packet_digest`, and `error_class`.

## Required invariants

1. `event_id` is unique and strictly increasing in file order.
2. `packet_seq` is within the declared workload range when present.
3. Packet commits are exactly once and occur in ascending `packet_seq` order.
4. `duration_ms`, payload, token and queue values are finite and non-negative.
5. `inflight_packets` never exceeds `window_packets`.
6. `inflight_bytes` never exceeds `window_bytes` when the byte window is
   enabled.
7. A packet's `commit` event cannot precede its `dispatch` event.
8. A live adapter must record blocked duration, not only backpressure count.
9. Cancellation and worker failure are represented explicitly; they must not
   be silently converted into successful commits.
10. The event stream must identify whether timings are measured or virtual.

## Derived metrics

For each `(run_id, scenario_id, scheduler_arm)`:

- committed packet count and payload bytes;
- virtual/measured wall time;
- packet/sec and payload MB/sec;
- preparation CPU-ms and GPU-compute ms;
- queue-wait and GPU-wait p50/p95;
- backpressure event count and blocked duration;
- maximum packet/byte in-flight window;
- maximum queue depth;
- retry, cancellation and error counts;
- host RSS/pinned-memory maxima when measured;
- GPU active/idle time when measured;
- scheduler-decision count and controller overhead when applicable.

The first v2 output intentionally reports virtual timings only. It is a
coverage and replay artifact, not a claim about RTX 3060 Ti or RunPod speed.

