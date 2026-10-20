# SSRE V2 — recovered material for GPT review

**Prepared:** 2026-10-18 (Australia/Brisbane)  
**Purpose:** Shareable recovery brief for the historical SSRE V2 / WSE / CPU-parallel work.  
**Scope boundary:** This is separate from Crucible/Braidformer. No Crucible run, checkpoint, manifest, or source file was modified while preparing it.

## Executive summary

The historical records contain several related things that were being called “SSRE V2”:

1. **SuperServer V2 / SSRE V2 hardware revision:** a Threadripper PRO 7995WX workstation with faster memory/storage and an optional multi-FPGA SSRE implementation. The local benchmark document labels its numbers **“Expected Results (Hardware-Optimized Predictions)”**, so those figures must not be treated as independently captured hardware measurements.
2. **A measured phone SSRE tuner v2 baseline:** six fast cores on the S24 Ultra, ARM NEON, equal partitioning, with cache and streaming results recorded in `claudecode.db`.
3. **The reusable SSRE/WSE CPU-parallel execution pattern:** work split into pods, thread/process-local engines and caches, deterministic chunking, asynchronous completion, adaptive scheduling, NUMA-style locality penalties, and persistent worker pools.
4. **A missing older C-worker lineage:** a later design note refers to `ssre_wse` / `ssre_runtime_v4_1`, but that exact source is not present in the accessible workspace.

The historical evidence supports an SSRE-inspired CPU preparation/execution reservoir for a future Crucible data pipeline, but it does **not** show that SSRE has already been integrated with neural training.

## Search result and naming clarification

Exact literal searches were performed over the accessible `claudecode.db`, ProjectBrain Markdown/source trees, raw ProjectBrain conversation fragments, and recovered GPT exports.

- Exact `SSREV2`: **not found**.
- Exact `SSRE V2` / `SSRE_v2` / `SSRE-V2`: found in the recovered GPT export and in historical SuperServer documentation.
- `ssre.phone.v2.cache` and `ssre.phone.v2.stream`: present as database run IDs.
- `ssre_runtime_v4_1`: **not present** as an accessible source file; it is only mentioned in `SELYRION_1000AGENT_CONCURRENCY_EXPERIMENT_DESIGN.md` as legacy code from an unrelated GPT thread that was not shown in that session.

Therefore “SSRE V2” should not be treated as one single immutable software artifact. The sections below preserve the separate lineages instead of merging them into an unsupported story.

## 1. SuperServer V2 / SSRE V2 hardware lineage

### Hardware comparison recorded locally

Source: `Documents/Projectbrain/SUPERSERVER_TESTBENCH_RESULTS.md`.

| Component | SuperServer V1 | SuperServer V2 |
|---|---:|---:|
| CPU | Threadripper PRO 5995WX, 64 cores / 128 threads | Threadripper PRO 7995WX, 96 cores / 192 threads |
| CPU frequency | 2.7 GHz base, 4.5 GHz boost | 2.5 GHz base, 5.1 GHz boost |
| RAM | 512 GB DDR4-3200 ECC, 204 GB/s | 1 TB DDR5-6400 ECC, 307 GB/s |
| NVMe | 3 × 2 TB Gen4 | 4 × 8 TB Gen5 |
| PCIe | Gen4 x16, 32 GB/s bidirectional | Gen5 x16, 64 GB/s bidirectional |
| Board | ASUS Pro WS WRX80E-SAGE SE | ASUS Pro WS WRX90E-SAGE SE |
| Optional accelerator | — | Xilinx Versal AI Edge ACAP |

### Documented V1/V2 benchmark table

The local document explicitly marks these as **expected, hardware-optimized predictions**, not a verified capture from installed V1/V2 machines.

| Test | V1 | V2 | Reported change |
|---|---:|---:|---:|
| Single-query minimum | 8.0 µs | 5.0 µs | 37% faster |
| Single-query average | 12.0 µs | 8.5 µs | 29% faster |
| Single-query P50 | 11.0 µs | 7.8 µs | 29% faster |
| Single-query P95 | 18.0 µs | 12.0 µs | 33% faster |
| Single-query P99 | 25.0 µs | 18.0 µs | 28% faster |
| Single-query maximum | 50.0 µs | 35.0 µs | 30% faster |
| Sequential QPS | 1,487 | 2,497 | 68% higher |
| Maximum parallel QPS | 65,432 at 128 threads | 234,567 at 192 threads | 258% higher |
| NVMe round-trip P50 | 85 µs | 42 µs | 51% faster |
| NVMe round-trip P95 | 180 µs | 88 µs | 51% faster |
| NVMe round-trip P99 | 350 µs | 165 µs | 53% faster |
| Driver GET_STATS P50 | 380 ns | 280 ns | 26% faster |
| Driver GET_STATS P95 | 650 ns | 480 ns | 26% faster |

Parallel QPS rows recorded by the same document:

| Threads | V1 QPS | V2 QPS |
|---:|---:|---:|
| 1 | 1,487 | 2,497 |
| 2 | 2,856 | 4,834 |
| 4 | 5,523 | 9,456 |
| 8 | 10,234 | 18,123 |
| 16 | 18,123 | 34,567 |
| 32 | 31,456 | 65,234 |
| 64 | 49,234 | 118,456 |
| 96 | — | 156,789 |
| 128 | 65,432 | 187,234 |
| 192 | — | 234,567 |

The document attributes the V2 improvement mainly to DDR5 bandwidth, PCIe Gen5, Zen 4, 50% more CPU cores, and Gen5 NVMe. It reports a V2 system cost of approximately AUD $14,000–18,000 in one section and a larger approximately $38,000 seven-FPGA projection elsewhere; these are different configurations and should not be conflated.

### Seven-layer / multi-FPGA projection

Recovered GPT material describes an expanded **SSRE V2 (7-layer)** configuration:

| Field | Historical V2 projection |
|---|---:|
| FPGA count | 7 |
| Throughput | approximately 1.77M QPS |
| Latency | approximately 12–130 µs depending on path |
| CPU/FPGAs | 1 CPU plus 7 FPGAs |
| Memory bandwidth | 307 GB/s |
| On-chip BRAM | approximately 630 MB |
| Power | approximately 500–600 W |
| System cost | approximately AUD $38K |
| NVMe latency | approximately 88 µs |
| PCIe | Gen5 |

The recovered GPT conversation also gives a five-to-seven-FPGA projection of 1M–1.6M QPS at 8–12 µs for parallel mode, 200K–350K QPS at 40–80 µs for cascaded mode, and approximately 600K QPS at 20–40 µs for a hierarchical mixed mode. These are projections, not independent measurements.

### Claimed FPGA/software inventory

The recovered GPT export contains a “complete SSRE FPGA implementation” summary claiming:

- six Verilog modules: spectral sketch, resonance matcher, PCIe interface, identity validator, timing controller, and top-level integration;
- a roughly 2,500-line Linux driver;
- test applications, Verilog testbenches, and a benchmark suite;
- a Xilinx Kintex UltraScale+ XCKU5P target, with Versal/Zynq alternatives;
- PCIe DMA, NVMe interaction, capsule matching, SHA-256 identity validation, and IOCTL statistics.

The named generated benchmark files (`ssre_bench.c`, `ssre_benchmark.py`, and `run_benchmarks.sh`) are referenced in the documentation, but the exact files were not found in the current accessible tree. Treat this inventory as recovered conversation/documentation claims, not as proof that the complete hardware stack is currently buildable.

## 2. Measured SSRE tuner v2 records in `claudecode.db`

These are distinct from the SuperServer V2 projection. They are real database rows for an S24 Ultra (`SM-S928B`, ARM64, NEON):

| Run ID | Threads / cores | Workload | Throughput | Input rate | Notes |
|---|---|---|---:|---:|---|
| `ssre.phone.v2.cache` | 6 / `6-fast` | cache | 3.024B comparisons/s | 45.05 GiB/s | cool-start peak |
| `ssre.phone.v2.stream` | 6 / `6-fast` | stream | 2.446B comparisons/s | 36.45 GiB/s | cool-start peak |

The database fields record `tuner_version=v2`, `partition_strategy=equal`, `kernel=NEON`, and `phase=peak`. The source database also records a later v3.1 revalidation:

- cache peak 3.010B comparisons/s, within roughly 1% of the v2 baseline;
- stream peak 2.427B comparisons/s, within roughly 1% of v2;
- thermally sustained cache 2.612B/s at approximately 84°C;
- thermally sustained stream 2.114B/s at approximately 85°C;
- six fast cores with equal partitioning remained the preferred configuration.

The v3 regression was attributed to measurement faults: repeated ARM core IDs causing false SMT labels and thermal-zone selection drift. The later record also says PC RAPL power readings of 0.00 W are unavailable and must not be treated as measurements.

### GPU contention correction

The database contains an earlier “contention v2” probe that reported a counterintuitive small decode improvement under sustained CPU load. A later randomized four-condition probe explicitly retracted that conclusion as warm-up/order bias.

The corrected result was:

- no-load decode: 41.44 ± 0.38 tok/s;
- CPU-load decode: 18.25 ± 0.20 tok/s;
- GPU utilization collapsed to approximately 18–19% while the CPU was saturated;
- CPU-saturating SSRE work must be gated or throttled during GPU-LLM decode windows;
- reserve coordinator cores, use NUMA-aware placement, or reduce SSRE parallelism when GPU decode is active.

This is directly relevant to a future Crucible data-preparation pipeline: CPU parallelism must not be judged only by its own throughput; it must also be measured for GPU queue starvation and end-to-end throughput.

## 3. Recoverable CPU/WSE parallel execution design

The exact old `ssre_runtime_v4_1` C worker source was not recovered, but the current tree contains several concrete ancestors of the design.

### Frozen database invariant: SSRE/WSE parallelism pattern

`claudecode.db`, invariant `inv.795fcdee`, records a proven 3.2× speedup in parallel SSRE multipass work:

- use `ThreadPoolExecutor(max_workers=N)` where N is the pod count;
- each pod owns its SQLite connection in WAL mode and its own local cache;
- split `game_ids` into N deterministic chunks;
- dispatch chunks and collect with `as_completed()`;
- sum counters in the main thread;
- use `INSERT OR IGNORE` for safe database-level deduplication across pods.

Discovery `disc.59beb886` records the same pattern added to `chess_position_analyze.py`: `--pods N`, `_process_chunk()`, one connection/cache per pod, deterministic chunk splitting, returned counters, four pods recommended on PC and two on Termux.

### `MLBUILD/ssre/wse_substrate.py`

Actual source structure:

- `SubstrateASIC`: one logical worker with a thread-local `ActivationEngine` and SQLite connection;
- `SubstratePod`: a `ThreadPoolExecutor`, even round-robin concept slicing, and asynchronous completion through `as_completed()`;
- `SubstrateWSE`: adaptive pod weights, smoothed times, confidence, epsilon exploration, optional active-region/NUMA penalty, and throughput logging;
- a warm-up phase loads the engine in every worker before normal work.

This is a usable scheduling skeleton for CPU-side preparation, but it currently executes symbolic concept queries rather than tokenization or neural batches.

### `MLBUILD/ssre/wse_substrate_mp.py`

Actual process-based variant:

- `ProcessPoolExecutor` with an initializer that constructs one engine per process;
- each worker owns its own SQLite connection and warm cache;
- batches are submitted to persistent workers;
- the same adaptive weighting/confidence logic is used;
- an optional NUMA-style remote penalty is included in scheduling metrics.

This is the stronger starting point when Python thread contention or SQLite thread affinity makes a process pool preferable.

### `Downloads/phone_plans/phone_docs/MlLang/ssre_wse_full_sim.py`

This simulation explicitly separates:

1. wave-scheduled proposal generation;
2. deterministic parsing and verification;
3. bulk WSE execution in large batches;
4. authoritative JSONL accumulation.

Its header explicitly says **no training, no learning, no inference**. It is a control/execution-plane simulation, not a neural data loader.

### Other concrete ancestors

- `Downloads/phone_tests/ssre_research/test_work_stealing.py`: a simple multiprocessing `Manager().Queue()` / `Pool` work-stealing-like benchmark over 300,000 tasks. Useful as historical evidence, but the Manager queue has avoidable IPC overhead and is not a production reservoir.
- `ssre_native_autotune_v2.sh`: a native C benchmark with a persistent pthread worker pool, condition variables, generation counters, CPU affinity, kernel selection, cache/stream working sets, and repeatable throughput logging.
- `MLBUILD/ssre_tr_runtime.py`: a simpler Threadripper runtime mapping pods to worker groups.
- `MLBUILD/SSRE_PACKAGE/SSRE/tests/test_tr_gpu_hybrid.py`: a toy CPU/GPU split emulator; it is not a neural training benchmark.

## 4. SSRE/WSE control-plane versus execution-plane model

`SSRE_WSE_PERFORMANCE_MODEL.md` establishes the historical doctrine:

- control plane: proposal generation, parsing, verification, and authority decisions; branch-heavy, latency-sensitive, deterministic;
- execution plane: bulk work, constraint evaluation, and fan-out; parallel/vectorizable and eligible for CPU/GPU/ASIC acceleration;
- epistemic authority remains in the control plane;
- latency model: `L(B) = L_control + B / T_exec`;
- small batches are control-limited, large batches are throughput-limited;
- the model is explicitly **not an LLM benchmark**, not FLOP-based, and not evidence that a GPU should perform epistemic control.

This maps cleanly onto the proposed Crucible adaptation:

```text
CPU SSRE reservoir
  -> deterministic packet preparation
  -> bounded pinned-memory queue
  -> nonblocking host-to-device transfer
  -> GPU forward/backward/update
```

Candidate CPU-side work:

- decode and byte-tokenize once;
- validate provenance/integrity once during dataset preparation;
- construct packed sequences and length-aware batches;
- generate the deterministic shuffle/schedule;
- assemble targets/masks;
- stage pinned memory and issue nonblocking copies;
- aggregate telemetry and write artifacts outside the critical path.

The GPU should retain matrix-heavy forward/backward work, optimizer updates, and fused kernels. The CPU pipeline must use bounded queues and backpressure rather than unbounded prefetch.

## 5. Recommended future Crucible benchmark, without changing the current run

After the current context experiment is sealed, compare four separate implementations while holding model, data order, seed, precision, optimizer, and targets fixed:

| Arm | Implementation |
|---|---|
| A | Current runner, unchanged reference |
| B | Pre-tokenized, validated, indexed or memory-mapped dataset |
| C | Persistent CPU workers with deterministic bounded batch queue and prefetch |
| D | SSRE/WSE scheduler additions: work packets, locality-aware placement, dependency-aware preparation, adaptive queue depth, and controlled GPU handoff |

Measure:

- end-to-end tokens/second;
- GPU active time versus queue-idle time;
- CPU utilization by core;
- preparation, serialization, and host-to-device-copy time;
- queue depth, stalls, and backpressure events;
- peak host RAM and VRAM;
- wall time and cost per fixed training budget;
- numerical/training-state equivalence;
- impact on GPU inference/LLM coordinator threads when co-scheduled.

The success criterion should be end-to-end throughput and reduced GPU waiting, not CPU utilization alone.

## 6. Important negative findings / unresolved items

- No exact `SSREV2` package or exact `ssre_runtime_v4_1` source was found.
- No evidence was found that the historical SSRE V2 FPGA claims are currently buildable from the present workspace.
- The local SuperServer V2 benchmark file says its figures are expected predictions.
- The recovered GPT export includes enthusiastic claims such as “production ready” and market comparisons; those are historical generated text, not independent validation.
- The existing SSRE/WSE code is symbolic execution code, not Crucible neural-training code.
- The CPU/GPU hybrid toy test is not evidence that moving neural matrix work to CPU is beneficial.
- The later randomized contention test shows that saturating CPU cores can starve a GPU workload; coordinator-core reservation and explicit coexistence scheduling are mandatory for any combined benchmark.

## Source inventory and hashes

Hashes are included so the material can be rechecked against the local archive.

| Source | SHA-256 |
|---|---|
| `Documents/Projectbrain/SUPERSERVER_TESTBENCH_RESULTS.md` | `484eaf5b6d4bae257cde2c550aafe2d2575345220e0ed1cae6ed9849a48fc5d3` |
| `Documents/Projectbrain/SESSION_SUMMARY_COMPLETE.md` | `8e52838eb7ed4c7303c516a7eeb5a30c414a1ffb64f2b74829f6a02dc4944d15` |
| `Downloads/phone_plans/phone_docs/SSRE_WSE_PERFORMANCE_MODEL.md` | `24f57c24b58ebdc25bf659349e3d32f5befd3f0ba7fd984d046c79c8a4f07f4e` |
| `MLBUILD/SSRE_PACKAGE/SSRE/SSRE_EXECUTION_CONTRACT.md` | `21faca358219f147585b2856605c1166b162945f11f2ac96cfea63aeff676f10` |
| `MLBUILD/ssre/wse_substrate.py` | `176edcb733cafa5ea00639ea7781b12b2947dfca6e416cb8554452b3d41b13cd` |
| `MLBUILD/ssre/wse_substrate_mp.py` | `2be3ec10911e315f93e017d336482d21c397c38856119728d36d39e474e89588` |
| `MLBUILD/ssre_tr_runtime.py` | `4a81f1a5cf91f0ab47221882ad635b26a3354dd7abe79616f8e1cab8531bf5f9` |
| `MLBUILD/ssre_gpu_emulator.py` | `d641a69704481dd6a26a2bc78bbc768dd9c857715e3e6840e0dace69d63e6fc8` |
| `Downloads/phone_plans/phone_docs/MlLang/ssre_wse_full_sim.py` | `f7ef8543194e5fd185e8db02de7c49b4103a2077824c6411944cfe688dc4a542` |
| `ssre_native_autotune_v2.sh` | `0eb43f6c2c559324fc40ad8149f75bc7cbec06d3cf91c88e7fbf496a85641c02` |
| `claudecode.db` | `c829327ae932d5aa4da5d31cc5bcd7abcedf15a56987cf8f4955816f1997e883` |
| `projectbrain_dev/recovered_gpt_export_2026_04/conversations-004.json` | `9944a153716ec2790f17131abedeaf55bd8531fd04c50fa90f1046c569401710` |
| `projectbrain_dev/SELYRION_1000AGENT_CONCURRENCY_EXPERIMENT_DESIGN.md` | `a007bac88405301458f81b592016b4389209c140458e1bc59e3963fe844d74ac` |

## Suggested prompt for GPT

> Treat this as a historical evidence packet, not as proof that every claim is experimentally validated. Separate: (a) measured `claudecode.db` phone tuner rows, (b) local documents explicitly marked expected/predicted, (c) generated GPT conversation claims, and (d) actual reusable source code. Explain how the SSRE/WSE work-reservoir, pod, queue, backpressure, locality, and CPU/GPU coexistence ideas can be adapted to Crucible data preparation without moving matrix-heavy neural training to CPU. Preserve the negative finding that CPU saturation can starve GPU inference. Identify what must be reimplemented and profiled before claiming a speedup.
