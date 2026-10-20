# SSRE-NPA v2 results summary

Date: 2026-10-18 (Brisbane)

## Execution boundary

This gate did not launch training, import Torch, use CUDA, contact RunPod, or
modify any Crucible/v0/v1 artifact. The active remote context jobs were left
untouched.

## Static audit of the current E16 path

Audited source files:

| File | SHA-256 |
|---|---|
| `crucible_context_gate_v1_20261018/shared/context_runner.py` | `948fe1ba2b466b8ddbd55d67e32890314407ba6b3d413b250c2faa756f9a335b` |
| `crucible_context_gate_v1_20261018/shared/model_core.py` | `271cfde17e073077947a3540c1d41fdcbe382309fa1021d89d42307a9a5cd6ab` |

The audit found the following real candidate boundaries:

- Every training microbatch is assembled through Python `dataset[index]`
  calls, two per-example tensor clones, and `torch.stack`.
- Training and validation `.to(device)` calls do not explicitly request
  nonblocking copies or identify pinned staging.
- Validation uses `DataLoader` without explicit `num_workers`, `pin_memory` or
  `persistent_workers` settings; the effective default is serial host loading.
- Each trial re-reads and validates source JSONL/provenance before training.
  That is appropriate for launch integrity but must not be counted as steady
  state throughput.
- Explicit CUDA synchronization surrounds checkpoint/validation timing, and
  metric rows are durably flushed with `os.fsync`.
- Each MoE layer loops over experts and top-k ranks, applies boolean token
  selection, evaluates selected experts, and records routing/output telemetry.
  `bincount` and `index_add_` accumulation are therefore separate possible
  overheads from ordinary model computation.

These observations identify where a live adapter should place timers. They do
not prove which boundary dominates on the active GPU.

## Deterministic workload framework

The matrix contains 6 workload scenarios × 4 scheduler arms × 96 packets:

- 24 virtual runs;
- 17,560 event records;
- exact ordered commits for every run;
- all schema and resource-window invariants passed;
- global bound enforced at 16 in-flight packets and 4 MiB payload;
- no missing or duplicate commits;
- no neural-controller training.

The four arms are `serial_reference`,
`indexed_precomputed_proxy`, `fixed_reservoir` and `heuristic_reservoir`.
The scenarios cover steady work, variable lengths, hot/warm/cold cache paths,
long-tail stragglers, a slow consumer and telemetry-heavy preparation.

The timings in `telemetry_summary.json` are virtual milliseconds. They are
coverage tests only. For example, the proxy predicts larger gains from
precomputed preparation in variable-length and cold-cache scenarios, while a
slow consumer remains consumer-bound. Those patterns validate that the matrix
contains meaningful trade-offs; they are not hardware performance results.

## Decision

The v2 schema/framework gate passes structurally. The next required artifact
is a live, read-only adapter around the deterministic SSRE-NPA scheduler that
emits the same fields with measured timings and CPU/GPU/resource counters.
Only after those traces exist should an offline controller dataset be split
into training and validation partitions. No GRU/controller was trained in v2.

