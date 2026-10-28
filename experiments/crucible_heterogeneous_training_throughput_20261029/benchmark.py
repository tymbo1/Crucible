#!/usr/bin/env python3
"""Heterogeneous Crucible training-throughput benchmark.

This is an engineering benchmark, not a scientific Crucible experiment.  It
uses the frozen context-128 byte corpus and schedule as a repeatable workload,
but trains a fresh, smaller dense model with the same latent-attention and
SwiGLU building blocks.  It compares four execution configurations:

  baseline_gpu              direct Python batch assembly + AdamW
  ssre_v1_v3_delivery       bounded indexed prefetch + AdamW
  cpu_adamw_offload         direct assembly + synchronized CPU AdamW
  ssre_offload_verification bounded prefetch + four process verification
                             workers + synchronized CPU AdamW

CUDA is optional.  With --device auto, CPU execution validates the data path,
model, optimizer and equivalence instrumentation when CUDA is unavailable;
GPU throughput fields are then explicitly marked unmeasured.

The benchmark deliberately keeps CPU optimizer updates synchronous.  SSRE
workers prepare and verify work; they do not concurrently mutate optimizer
state or model parameters.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
V3_DIR = ROOT / "experiments" / "ssre_npa_v3_measured_batch_replay_20261018"
SHARED_DIR = ROOT / "experiments" / "crucible_v2_production_telemetry_v1_20261019" / "shared"
if str(V3_DIR) not in sys.path:
    sys.path.insert(0, str(V3_DIR))
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from cpu_replay import (  # noqa: E402
    Batch,
    assemble_current,
    assemble_indexed,
    batch_fingerprint,
    load_binary,
    load_manifest,
    load_schedule,
    load_source_records,
    sha256_bytes,
    sha256_file,
)
from model_core import MultiHeadLatentAttention, SwiGLU  # noqa: E402


PAD_ID = 0
RAW_BYTE_START = 3
SCHEMA_VERSION = "crucible-heterogeneous-training-throughput-v2-20261029"
LOSS_SCALE = 128.0
DEFAULT_CONTEXT_ROOT = Path(os.environ.get("CRUCIBLE_CONTEXT_ROOT", "context_128"))


def now_ns() -> int:
    return time.perf_counter_ns()


def rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def sha256_file_or_none(path: Path) -> Optional[str]:
    return sha256_file(path) if path.is_file() else None


def redact_external_value(value: Any) -> Any:
    """Remove local paths from artifacts intended for external review."""
    if isinstance(value, dict):
        return {key: redact_external_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_external_value(item) for item in value]
    if isinstance(value, str) and (
        value.startswith("/home/")
        or value.startswith("/mnt/")
        or value.startswith("/tmp/")
    ):
        return "<local-path-redacted>"
    return value


def state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def model_config(name: str) -> Dict[str, int]:
    if name == "tiny":
        return {
            "vocab_size": 259,
            "n_layers": 2,
            "d_model": 128,
            "n_heads": 4,
            "latent_dim_kv": 32,
            "latent_dim_q": 32,
            "pe_dim": 16,
            "d_ff": 512,
        }
    if name == "small":
        return {
            "vocab_size": 259,
            "n_layers": 4,
            "d_model": 256,
            "n_heads": 4,
            "latent_dim_kv": 64,
            "latent_dim_q": 64,
            "pe_dim": 32,
            "d_ff": 1024,
        }
    raise ValueError(f"unknown model size: {name}")


class DenseCrucibleBlock(nn.Module):
    """Dense block reusing Crucible's latent attention and SwiGLU primitives."""

    def __init__(self, cfg: Mapping[str, int]):
        super().__init__()
        d_model = int(cfg["d_model"])
        self.attn_norm = nn.RMSNorm(d_model)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.attn = MultiHeadLatentAttention(
            d_model,
            int(cfg["n_heads"]),
            int(cfg["latent_dim_kv"]),
            int(cfg["latent_dim_q"]),
            int(cfg["pe_dim"]),
        )
        self.ffn = SwiGLU(d_model, int(cfg["d_ff"]))

    def forward(self, x: torch.Tensor, seq_idx: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), seq_idx)
        return x + self.ffn(self.ffn_norm(x))


class DenseCrucibleLM(nn.Module):
    """Small dense byte LM using the same core primitives as Crucible."""

    def __init__(self, cfg: Mapping[str, int]):
        super().__init__()
        self.cfg = dict(cfg)
        self.token_emb = nn.Embedding(int(cfg["vocab_size"]), int(cfg["d_model"]))
        self.layers = nn.ModuleList([DenseCrucibleBlock(cfg) for _ in range(int(cfg["n_layers"]))])
        self.final_norm = nn.RMSNorm(int(cfg["d_model"]))
        self.lm_head = nn.Linear(int(cfg["d_model"]), int(cfg["vocab_size"]), bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_idx = torch.arange(tokens.size(1), device=tokens.device, dtype=torch.long)
        seq_idx = seq_idx.unsqueeze(0).expand(tokens.size(0), -1)
        x = self.token_emb(tokens)
        for layer in self.layers:
            x = layer(x, seq_idx)
        return self.lm_head(self.final_norm(x))


class CPUAdamWOffload(torch.optim.Optimizer):
    """Synchronous CPU-master AdamW for a model whose parameters may be on GPU.

    The model parameters remain on their execution device.  FP32 master
    parameters and Adam moments live on CPU RAM.  A complete gradient is
    transferred to the CPU for each optimizer step, the update is applied by
    one owner, and updated parameters are copied back.  This is intentionally
    conservative and measurable rather than asynchronous or sharded.
    """

    def __init__(self, params: Iterable[nn.Parameter], lr: float, betas: Tuple[float, float], eps: float, weight_decay: float):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.cpu_state_bytes = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                self.state[parameter]["step"] = 0
                self.state[parameter]["master"] = parameter.detach().float().cpu().clone()
                self.state[parameter]["exp_avg"] = torch.zeros_like(self.state[parameter]["master"])
                self.state[parameter]["exp_avg_sq"] = torch.zeros_like(self.state[parameter]["master"])
                self.cpu_state_bytes += sum(
                    int(self.state[parameter][key].numel() * self.state[parameter][key].element_size())
                    for key in ("master", "exp_avg", "exp_avg_sq")
                )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Optional[torch.Tensor]:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                state["step"] += 1
                grad = parameter.grad.detach().float().cpu()
                master = state["master"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if weight_decay:
                    master.mul_(1.0 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                denom.add_(eps)
                master.addcdiv_(exp_avg, denom, value=-step_size)
                parameter.copy_(master.to(device=parameter.device, dtype=parameter.dtype))
        return loss


@dataclass(frozen=True)
class PreparedBatch:
    x: torch.Tensor
    y: torch.Tensor
    queue_wait_ns: int
    verification_ns: int
    verification_digest: Optional[str]


def verify_source_batch(payload: Tuple[Tuple[bytes, ...], Tuple[int, ...]]) -> Tuple[str, int, int, int]:
    """Independent CPU fixture for generation/verification workers."""
    started = now_ns()
    records, indices = payload
    digest = hashlib.sha256()
    total_bytes = 0
    for index, value in zip(indices, records):
        encoded = bytes(value)
        digest.update(str(int(index)).encode("ascii"))
        digest.update(encoded)
        total_bytes += len(encoded)
        if len(encoded) == 0:
            raise ValueError("empty source record")
    return digest.hexdigest(), total_bytes, len(records), now_ns() - started


class BoundedIndexedPrefetcher:
    """SSRE-NPA v1-style bounded window over the v3 indexed path."""

    def __init__(
        self,
        raw: torch.Tensor,
        batches: Sequence[Batch],
        index_tensors: Sequence[torch.Tensor],
        source_records: Sequence[bytes],
        window: int,
        workers: int,
        process_verification: bool,
        pin_memory: bool,
    ):
        self.raw = raw
        self.batches = batches
        self.index_tensors = index_tensors
        self.source_records = source_records
        self.window = max(1, int(window))
        self.workers = max(1, int(workers))
        self.pin_memory = bool(pin_memory)
        self.process_verification = bool(process_verification)
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="ssre-v3-indexed")
        self.verify_pool = (
            ProcessPoolExecutor(max_workers=self.workers)
            if process_verification
            else None
        )
        self.pending: Dict[int, Tuple[Future[Any], int]] = {}
        self.next_submit = 0
        self.next_commit = 0

    def _prepare(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[str], int]:
        started = now_ns()
        selected = self.raw.index_select(0, self.index_tensors[index])
        x = selected[:, 0].contiguous()
        y = selected[:, 1].contiguous()
        verification_digest = None
        verification_ns = 0
        if self.verify_pool is not None:
            batch = self.batches[index]
            records = tuple(self.source_records[item] for item in batch.indices)
            verification_digest, total_bytes, count, verification_ns = self.verify_pool.submit(
                verify_source_batch, (records, tuple(batch.indices))
            ).result()
            if total_bytes != batch.source_bytes or count != len(batch.indices):
                raise ValueError("verification worker accounting mismatch")
        if self.pin_memory and torch.cuda.is_available():
            x = x.pin_memory()
            y = y.pin_memory()
        return x, y, now_ns() - started, verification_digest, verification_ns

    def _fill(self) -> None:
        while self.next_submit < len(self.batches) and len(self.pending) < self.window:
            submitted = now_ns()
            self.pending[self.next_submit] = (self.pool.submit(self._prepare, self.next_submit), submitted)
            self.next_submit += 1

    def get(self, index: int) -> PreparedBatch:
        if index != self.next_commit:
            raise RuntimeError(f"prefetch commit order violation: expected {self.next_commit}, got {index}")
        self._fill()
        future, submitted = self.pending.pop(index)
        x, y, _prep_ns, verification_digest, verification_ns = future.result()
        committed = now_ns()
        self.next_commit += 1
        self._fill()
        return PreparedBatch(
            x=x,
            y=y,
            queue_wait_ns=max(0, committed - submitted),
            verification_ns=verification_ns,
            verification_digest=verification_digest,
        )

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=False)
        if self.verify_pool is not None:
            self.verify_pool.shutdown(wait=True, cancel_futures=False)


def build_batches(context_root: Path, max_steps: int) -> Tuple[Dict[str, Any], str, List[bytes], torch.Tensor, List[Batch], Dict[str, Any]]:
    config, config_sha = load_manifest(context_root)
    manifest_root = context_root / "crucible_manifest"
    train_info = config["splits"]["train"]
    records, source_report = load_source_records(manifest_root, train_info)
    raw, binary_report = load_binary(
        manifest_root,
        train_info,
        len(records),
        int(config["hyperparameters"]["context_window_length"]),
    )
    schedule = config["schedules"]["42"]
    all_batches, schedule_report = load_schedule(
        manifest_root, schedule, 42, [len(item) for item in records]
    )
    batches = [batch for batch in all_batches if batch.step <= max_steps]
    if not batches:
        raise ValueError("empty benchmark schedule prefix")
    return config, config_sha, records, raw, batches, {
        "source": source_report,
        "binary": binary_report,
        "schedule": schedule_report,
    }


def validate_input_equivalence(raw: torch.Tensor, batches: Sequence[Batch], index_tensors: Sequence[torch.Tensor], count: int) -> Dict[str, Any]:
    reference: List[str] = []
    candidate: List[str] = []
    for batch, index_tensor in zip(batches[:count], index_tensors[:count]):
        x_a, y_a, _, _ = assemble_current(raw, batch)
        x_b, y_b, _, _ = assemble_indexed(raw, batch, index_tensor)
        fp_a = batch_fingerprint(x_a, y_a)
        fp_b = batch_fingerprint(x_b, y_b)
        if fp_a != fp_b:
            raise ValueError(f"indexed input mismatch at step {batch.step}/{batch.microbatch_index}")
        reference.append(fp_a["fingerprint_sha256"])
        candidate.append(fp_b["fingerprint_sha256"])
    return {
        "checked_batches": count,
        "exact_equivalence": reference == candidate,
        "reference_sha256": sha256_bytes("|".join(reference).encode("ascii")),
        "indexed_sha256": sha256_bytes("|".join(candidate).encode("ascii")),
    }


def parameter_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    no_decay_types = (nn.RMSNorm, nn.LayerNorm, nn.Embedding)
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if isinstance(module, no_decay_types) or name.endswith("bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def loss_sum(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, int]:
    values = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    valid = targets.reshape(-1) != PAD_ID
    return values[valid].sum(), int(valid.sum().item())


def select_device(mode: str) -> Tuple[torch.device, str, Optional[str]]:
    if mode == "cpu":
        return torch.device("cpu"), "cpu", "forced_cpu"
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda"), "cuda", None
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", None
    return torch.device("cpu"), "cpu", "cuda_unavailable"


def make_model(cfg: Mapping[str, int], device: torch.device, seed: int) -> DenseCrucibleLM:
    torch.manual_seed(seed)
    model = DenseCrucibleLM(cfg).float().to(device)
    model.train()
    return model


def bytes_for_tensors(x: torch.Tensor, y: torch.Tensor) -> int:
    return int((x.numel() + y.numel()) * x.element_size())


def unscale_grads(model: nn.Module, scale: float) -> bool:
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        parameter.grad.div_(scale)
        finite = finite and bool(torch.isfinite(parameter.grad).all().item())
    return finite


def cuda_event_pair(device: torch.device) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def run_trial(
    *,
    arm: str,
    cfg: Mapping[str, int],
    device: torch.device,
    device_kind: str,
    seed: int,
    batches: Sequence[Batch],
    raw: torch.Tensor,
    records: Sequence[bytes],
    index_tensors: Sequence[torch.Tensor],
    warmup_steps: int,
    timed_steps: int,
    window: int,
    workers: int,
    learning_rate: float,
    weight_decay: float,
) -> Dict[str, Any]:
    offload = arm in {"cpu_adamw_offload", "ssre_offload_verification"}
    use_ssre = arm in {"ssre_v1_v3_delivery", "ssre_offload_verification"}
    use_verification = arm == "ssre_offload_verification"
    model = make_model(cfg, device, seed)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial_state = state_digest(model)
    if offload:
        optimizer: torch.optim.Optimizer = CPUAdamWOffload(
            parameter_groups(model, weight_decay),
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            parameter_groups(model, weight_decay),
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    provider: Optional[BoundedIndexedPrefetcher] = None
    if use_ssre:
        provider = BoundedIndexedPrefetcher(
            raw,
            batches,
            index_tensors,
            records,
            window=window,
            workers=workers,
            process_verification=use_verification,
            pin_memory=device_kind == "cuda",
        )
    batch_by_step: Dict[int, List[Tuple[int, Batch]]] = {}
    for index, batch in enumerate(batches):
        batch_by_step.setdefault(batch.step, []).append((index, batch))
    max_step = warmup_steps + timed_steps
    if max(batch_by_step) < max_step:
        raise ValueError(f"schedule prefix has no step {max_step}")
    autocast_enabled = device_kind == "cuda"
    autocast_dtype = torch.float16 if autocast_enabled else torch.float32
    records_out: List[Dict[str, Any]] = []
    total_start = now_ns()
    cpu_start = time.process_time_ns()
    if device_kind == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for step in range(1, max_step + 1):
            group = batch_by_step[step]
            total_valid = sum(item.nonpad_target_tokens for _, item in group)
            optimizer.zero_grad(set_to_none=True)
            step_start = now_ns()
            record: Dict[str, Any] = {
                "step": step,
                "phase": "warmup" if step <= warmup_steps else "timed",
                "microbatches": len(group),
                "valid_targets": 0,
                "source_bytes": 0,
                "cpu_batch_prepare_ns": 0,
                "batch_wait_ns": 0,
                "verification_ns": 0,
                "h2d_ns": 0,
                "h2d_bytes": 0,
                "forward_host_ns": 0,
                "forward_cuda_ns": 0,
                "backward_host_ns": 0,
                "backward_cuda_ns": 0,
                "optimizer_ns": 0,
                "loss_sum": 0.0,
                "grad_norm": None,
                "finite_gradients": True,
            }
            for batch_index, batch in group:
                prepare_start = now_ns()
                if provider is None:
                    x_cpu, y_cpu, _, _ = assemble_current(raw, batch)
                    prepared = PreparedBatch(x_cpu, y_cpu, 0, 0, None)
                else:
                    prepared = provider.get(batch_index)
                record["cpu_batch_prepare_ns"] += now_ns() - prepare_start
                record["batch_wait_ns"] += prepared.queue_wait_ns
                record["verification_ns"] += prepared.verification_ns
                x_cpu, y_cpu = prepared.x, prepared.y
                record["source_bytes"] += batch.source_bytes
                record["valid_targets"] += batch.nonpad_target_tokens
                record["h2d_bytes"] += bytes_for_tensors(x_cpu, y_cpu)
                if device_kind == "cuda":
                    torch.cuda.synchronize(device)
                    h2d_start = now_ns()
                    x_gpu = x_cpu.to(device, non_blocking=bool(x_cpu.is_pinned()))
                    y_gpu = y_cpu.to(device, non_blocking=bool(y_cpu.is_pinned()))
                    torch.cuda.synchronize(device)
                    record["h2d_ns"] += now_ns() - h2d_start
                else:
                    x_gpu, y_gpu = x_cpu, y_cpu

                forward_start = now_ns()
                if device_kind == "cuda":
                    f_start, f_end = cuda_event_pair(device)
                    f_start.record()
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                    logits = model(x_gpu)
                    ce_sum, valid_count = loss_sum(logits, y_gpu)
                    loss = ce_sum / float(total_valid)
                if device_kind == "cuda":
                    f_end.record()
                    f_end.synchronize()
                    record["forward_cuda_ns"] += int(f_start.elapsed_time(f_end) * 1_000_000.0)
                record["forward_host_ns"] += now_ns() - forward_start
                record["loss_sum"] += float(ce_sum.detach().float().item())
                backward_start = now_ns()
                if device_kind == "cuda":
                    b_start, b_end = cuda_event_pair(device)
                    b_start.record()
                backward_loss = loss * (LOSS_SCALE if autocast_enabled else 1.0)
                backward_loss.backward()
                if device_kind == "cuda":
                    b_end.record()
                    b_end.synchronize()
                    record["backward_cuda_ns"] += int(b_start.elapsed_time(b_end) * 1_000_000.0)
                record["backward_host_ns"] += now_ns() - backward_start
                del logits, loss, backward_loss, x_gpu, y_gpu

            if autocast_enabled:
                record["finite_gradients"] = unscale_grads(model, LOSS_SCALE)
            else:
                record["finite_gradients"] = all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                    for parameter in model.parameters()
                )
            clip_start = now_ns()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            clip_ns = now_ns() - clip_start
            optimizer_start = now_ns()
            if device_kind == "cuda":
                torch.cuda.synchronize(device)
            optimizer.step()
            if device_kind == "cuda":
                torch.cuda.synchronize(device)
            record["optimizer_ns"] = now_ns() - optimizer_start + clip_ns
            record["grad_norm"] = grad_norm
            record["loss"] = record["loss_sum"] / max(1, record["valid_targets"])
            record["wall_ns"] = now_ns() - step_start
            record["gpu_idle_ns_proxy"] = (
                max(0, record["wall_ns"] - record["forward_cuda_ns"] - record["backward_cuda_ns"])
                if device_kind == "cuda"
                else None
            )
            if step > warmup_steps:
                records_out.append(record)
    finally:
        if provider is not None:
            provider.close()

    total_ns = now_ns() - total_start
    cpu_ns = time.process_time_ns() - cpu_start
    timed_tokens = sum(record["valid_targets"] for record in records_out)
    timed_wall = sum(record["wall_ns"] for record in records_out)
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "device": device_kind,
        "seed": seed,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "workers": workers if use_ssre else 0,
        "window": window if use_ssre else 0,
        "parameter_count": parameter_count,
        "model_config": dict(cfg),
        "initial_state_sha256": initial_state,
        "final_state_sha256": state_digest(model),
        "total_wall_ns": total_ns,
        "process_cpu_ns": cpu_ns,
        "timed_tokens": timed_tokens,
        "timed_wall_ns": timed_wall,
        "tokens_per_second": timed_tokens / max(1e-9, timed_wall / 1e9),
        "peak_rss_bytes": rss_bytes(),
        "cpu_optimizer_state_bytes": int(getattr(optimizer, "cpu_state_bytes", 0)),
        "timed_records": records_out,
        "numerical": {
            "all_finite_gradients": all(bool(record["finite_gradients"]) for record in records_out),
            "all_finite_losses": all(math.isfinite(float(record["loss"])) for record in records_out),
        },
        "gpu_metrics": {
            "measured": device_kind == "cuda",
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device_kind == "cuda" else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device_kind == "cuda" else None,
            "forward_cuda_ns": sum(int(record["forward_cuda_ns"]) for record in records_out),
            "backward_cuda_ns": sum(int(record["backward_cuda_ns"]) for record in records_out),
            "h2d_ns": sum(int(record["h2d_ns"]) for record in records_out),
            "h2d_bytes": sum(int(record["h2d_bytes"]) for record in records_out),
            "gpu_idle_ns_proxy": (
                sum(int(record["gpu_idle_ns_proxy"]) for record in records_out)
                if device_kind == "cuda"
                else None
            ),
        },
    }
    return result


def run_update_equivalence_probe(
    *,
    cfg: Mapping[str, int],
    device: torch.device,
    device_kind: str,
    seed: int,
    raw: torch.Tensor,
    batch: Batch,
    index_tensor: torch.Tensor,
    learning_rate: float,
    weight_decay: float,
) -> Dict[str, Any]:
    """Compare one regular AdamW update with the synchronized CPU update."""
    x_cpu, y_cpu, _, _ = assemble_current(raw, batch)
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    regular = make_model(cfg, device, seed)
    offloaded = make_model(cfg, device, seed)
    regular_optimizer = torch.optim.AdamW(
        parameter_groups(regular, weight_decay),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    offloaded_optimizer = CPUAdamWOffload(
        parameter_groups(offloaded, weight_decay),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    def update(model: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        ce_sum, valid_count = loss_sum(logits, y)
        loss = ce_sum / float(valid_count)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device_kind == "cuda":
            torch.cuda.synchronize(device)
        return float(loss.detach().item())

    regular_loss = update(regular, regular_optimizer)
    offloaded_loss = update(offloaded, offloaded_optimizer)
    max_abs = 0.0
    max_rel = 0.0
    for name, regular_tensor in regular.state_dict().items():
        candidate = offloaded.state_dict()[name]
        difference = (regular_tensor.detach().float().cpu() - candidate.detach().float().cpu()).abs()
        denominator = regular_tensor.detach().float().cpu().abs().clamp_min(1e-12)
        max_abs = max(max_abs, float(difference.max().item()))
        max_rel = max(max_rel, float((difference / denominator).max().item()))
    tolerance = 1e-5
    result = {
        "device": device_kind,
        "batch_step": batch.step,
        "batch_microbatch_index": batch.microbatch_index,
        "initial_state_match": state_digest(regular) == state_digest(offloaded),
        "regular_loss": regular_loss,
        "offloaded_loss": offloaded_loss,
        "loss_abs_delta": abs(regular_loss - offloaded_loss),
        "max_parameter_abs_delta": max_abs,
        "max_parameter_relative_delta": max_rel,
        "tolerance": tolerance,
        "within_tolerance": max_abs <= tolerance,
        "regular_final_state_sha256": state_digest(regular),
        "offloaded_final_state_sha256": state_digest(offloaded),
        "exact_final_state_match": state_digest(regular) == state_digest(offloaded),
        "cpu_optimizer_state_bytes": int(offloaded_optimizer.cpu_state_bytes),
    }
    del regular, offloaded, regular_optimizer, offloaded_optimizer, x, y
    if device_kind == "cuda":
        torch.cuda.empty_cache()
    return result


def compare_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Mapping[str, Any]]] = {}
    for result in results:
        by_arm.setdefault(str(result["arm"]), []).append(result)
    summary: Dict[str, Any] = {}
    for arm, rows in by_arm.items():
        summary[arm] = {
            "runs": len(rows),
            "mean_tokens_per_second": sum(float(row["tokens_per_second"]) for row in rows) / len(rows),
            "mean_timed_wall_ns": sum(int(row["timed_wall_ns"]) for row in rows) / len(rows),
            "max_peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in rows),
            "gpu_measured": any(bool(row["gpu_metrics"]["measured"]) for row in rows),
            "cpu_optimizer_state_bytes": max(int(row["cpu_optimizer_state_bytes"]) for row in rows),
        }
    equivalence: Dict[str, Any] = {}
    baseline = [row for row in results if row["arm"] == "baseline_gpu"]
    ssre = [row for row in results if row["arm"] == "ssre_v1_v3_delivery"]
    if baseline and ssre:
        equivalence["baseline_vs_ssre_initial_match"] = baseline[0]["initial_state_sha256"] == ssre[0]["initial_state_sha256"]
        equivalence["baseline_vs_ssre_final_match"] = baseline[0]["final_state_sha256"] == ssre[0]["final_state_sha256"]
    return {"arms": summary, "equivalence": equivalence}


def main() -> int:
    parser = argparse.ArgumentParser(description="Heterogeneous Crucible throughput benchmark")
    parser.add_argument("--context-root", type=Path, default=DEFAULT_CONTEXT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-size", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--timed-steps", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-machine-metadata",
        action="store_true",
        help="Include local paths and host metadata in artifacts; keep disabled for external review.",
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device, device_kind, device_note = select_device(args.device)
    max_steps = args.warmup_steps + args.timed_steps
    config, config_sha, records, raw, batches, data_report = build_batches(args.context_root.resolve(), max_steps)
    index_tensors = [torch.tensor(batch.indices, dtype=torch.long) for batch in batches]
    input_report = validate_input_equivalence(raw, batches, index_tensors, min(16, len(batches)))
    cfg = model_config(args.model_size)
    model_probe = make_model(cfg, torch.device("cpu"), args.seed)
    parameter_count = sum(parameter.numel() for parameter in model_probe.parameters())
    del model_probe
    update_equivalence = run_update_equivalence_probe(
        cfg=cfg,
        device=device,
        device_kind=device_kind,
        seed=args.seed,
        raw=raw,
        batch=batches[0],
        index_tensor=index_tensors[0],
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    arms = (
        "baseline_gpu",
        "ssre_v1_v3_delivery",
        "cpu_adamw_offload",
        "ssre_offload_verification",
    )
    results: List[Dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        for arm in arms:
            result = run_trial(
                arm=arm,
                cfg=cfg,
                device=device,
                device_kind=device_kind,
                seed=args.seed,
                batches=batches,
                raw=raw,
                records=records,
                index_tensors=index_tensors,
                warmup_steps=args.warmup_steps,
                timed_steps=args.timed_steps,
                window=args.window,
                workers=args.workers,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            result["repetition"] = repetition
            results.append(result)

    environment = {
        "python": sys.version,
        "platform": platform.platform() if args.include_machine_metadata else "redacted",
        "processor": platform.processor() if args.include_machine_metadata else "redacted",
        "cpu_count": os.cpu_count() if args.include_machine_metadata else None,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() and args.include_machine_metadata
            else None
        ),
        "device_selected": device_kind,
        "device_note": device_note,
    }
    public_data_report = data_report if args.include_machine_metadata else redact_external_value(data_report)
    report = {
        "schema_version": SCHEMA_VERSION,
        "classification": "engineering_benchmark_cpu_validated_gpu_unmeasured" if device_kind == "cpu" else "engineering_benchmark_gpu_measured",
        "scientific_status": "not_a_scientific_training_run",
        "data_manifest_sha256": config_sha,
        "data_report": public_data_report,
        "input_equivalence": input_report,
        "update_equivalence": update_equivalence,
        "model": {"family": "Crucible latent-attention + SwiGLU dense", "size": args.model_size, "parameter_count": parameter_count, "config": cfg},
        "benchmark_config": {
            "arms": arms,
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "repetitions": args.repetitions,
            "window": args.window,
            "workers": args.workers,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "precision": "float16 autocast + static loss scale on CUDA; float32 on CPU",
        },
        "environment": environment,
        "results": results,
        "summary": compare_results(results),
        "limitations": [
            "The CPU-only result does not measure GPU throughput or PCIe transfer time.",
            "CPU AdamW offload is synchronized and intentionally not an asynchronous optimizer design.",
            "The dense benchmark model is a smaller representative Crucible-family model, not the future 500M model.",
            "The verification workers use deterministic source-record validation as a CPU workload proxy; they are not a full curriculum generator.",
        ],
    }
    write_json(output / "BENCHMARK_REPORT.json", report)
    write_json(output / "ARM_SUMMARY.json", report["summary"])
    write_json(output / "ENVIRONMENT.json", environment)
    write_json(output / "INPUT_EQUIVALENCE.json", input_report)
    write_json(output / "UPDATE_EQUIVALENCE.json", update_equivalence)
    (output / "BENCHMARK_REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    hashes = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "HASHES.json":
            hashes[path.name] = sha256_file(path)
    write_json(output / "HASHES.json", {"schema_version": SCHEMA_VERSION, "files": hashes})
    return 0


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Crucible heterogeneous training-throughput benchmark",
        "",
        f"Classification: `{report['classification']}`",
        "",
        "This is an engineering benchmark, not a scientific learning claim.",
        "",
        "## Model",
        "",
        f"- Family: {report['model']['family']}",
        f"- Size: {report['model']['size']}",
        f"- Parameters: {report['model']['parameter_count']:,}",
        "",
        "## Arm summary",
        "",
        "| Arm | Runs | Mean tokens/s | Mean timed wall | GPU measured | CPU optimizer state |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for arm, row in report["summary"]["arms"].items():
        lines.append(
            f"| {arm} | {row['runs']} | {row['mean_tokens_per_second']:.2f} | "
            f"{row['mean_timed_wall_ns'] / 1e9:.4f}s | {row['gpu_measured']} | "
            f"{row['cpu_optimizer_state_bytes'] / (1024 ** 2):.2f} MiB |"
        )
    lines += [
        "",
        "## Integrity",
        "",
        f"- Indexed input exact equivalence: `{report['input_equivalence']['exact_equivalence']}`",
        f"- One-update CPU-offload equivalence: `{report['update_equivalence']['within_tolerance']}`",
        f"- One-update maximum parameter delta: `{report['update_equivalence']['max_parameter_abs_delta']:.3g}`",
        f"- Baseline/SSRE initial-state match: `{report['summary']['equivalence'].get('baseline_vs_ssre_initial_match')}`",
        f"- Baseline/SSRE final-state match: `{report['summary']['equivalence'].get('baseline_vs_ssre_final_match')}`",
        f"- Selected device: `{report['environment']['device_selected']}`",
        f"- CUDA available: `{report['environment']['cuda_available']}`",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
