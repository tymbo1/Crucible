#!/usr/bin/env python3
"""Crucible E16 context-gate runner.

This runner is isolated from prior Crucible runners. It uses the frozen E16
model and anchors, a deterministic source-byte-balanced schedule, and one
Branch-B trial per seed. The primary validation metric is EOS-excluded
negative log likelihood per raw UTF-8 byte (and its bits/byte form); the
historical all-non-PAD NLL is retained for continuity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model_core import BraidPrototypeLM, ResearchMoETelemetry


PAD_ID = 0
RAW_BYTE_START = 3


class ImmutableBinaryDataset(Dataset):
    def __init__(
        self,
        full_path: str,
        seq_len: int,
        expected_hash: str,
        expected_count: int,
    ):
        with open(full_path, "rb") as handle:
            raw_bytes = handle.read()
        actual_hash = hashlib.sha256(raw_bytes).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"binary integrity failure: {full_path}")
        stride = 2 * seq_len * 8
        if len(raw_bytes) % stride:
            raise ValueError(f"binary geometry failure: {full_path}")
        self.seq_len = seq_len
        self.total_samples = len(raw_bytes) // stride
        if self.total_samples != expected_count:
            raise ValueError(f"binary count failure: {full_path}")
        self.raw_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.int64)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        offset = idx * (2 * self.seq_len)
        return {
            "x": self.raw_tensor[offset : offset + self.seq_len].clone(),
            "y": self.raw_tensor[offset + self.seq_len : offset + 2 * self.seq_len].clone(),
        }


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_file_sha256_integrity(path: str, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def read_jsonl_source_lengths(base_dir: str, split_info: Dict[str, Any]) -> List[int]:
    path = os.path.join(base_dir, split_info["source_jsonl_relative_path"])
    assert_file_sha256_integrity(path, split_info["source_jsonl_sha256"])
    lengths: List[int] = []
    seen: set[bytes] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            encoded = str(record["text"]).encode("utf-8")
            if encoded in seen:
                raise ValueError(f"duplicate source text at {path}:{line_number}")
            seen.add(encoded)
            if int(record["example_index"]) != len(lengths):
                raise ValueError(f"source index mismatch at {path}:{line_number}")
            if int(record["byte_length"]) != len(encoded):
                raise ValueError(f"source byte length mismatch at {path}:{line_number}")
            if record["text_utf8_sha256"] != hashlib.sha256(encoded).hexdigest():
                raise ValueError(f"source text hash mismatch at {path}:{line_number}")
            lengths.append(len(encoded))
    if len(lengths) != int(split_info["total_constructed_examples"]):
        raise ValueError(f"source record count mismatch: {path}")
    provenance_path = os.path.join(base_dir, split_info["provenance_relative_path"])
    assert_file_sha256_integrity(provenance_path, split_info["provenance_sha256"])
    with open(provenance_path, "r", encoding="utf-8") as handle:
        provenance_count = sum(1 for line in handle if line.strip())
    if provenance_count != int(split_info["provenance_record_count"]):
        raise ValueError(f"provenance record count mismatch: {provenance_path}")
    if provenance_count != len(lengths):
        raise ValueError(f"source/provenance count mismatch: {provenance_path}")
    return lengths


def load_schedule(
    base_dir: str,
    schedule_info: Dict[str, Any],
    seed: int,
    context: int,
    lengths: Sequence[int],
) -> Dict[str, Any]:
    path = os.path.join(base_dir, schedule_info["relative_path"])
    assert_file_sha256_integrity(path, schedule_info["binary_payload_sha256"])
    with open(path, "r", encoding="utf-8") as handle:
        schedule = json.load(handle)
    if schedule.get("schema") != "crucible_dynamic_source_byte_schedule_v1":
        raise ValueError("unsupported dynamic schedule schema")
    if int(schedule["seed"]) != seed or int(schedule["context_window_length"]) != context:
        raise ValueError("schedule identity mismatch")
    groups = schedule.get("groups")
    if not isinstance(groups, list) or len(groups) != int(schedule_info["num_steps"]):
        raise ValueError("schedule step count mismatch")
    flattened: List[int] = []
    for expected_step, group in enumerate(groups, start=1):
        if int(group["step"]) != expected_step:
            raise ValueError("schedule step index mismatch")
        indices = [int(index) for index in group["indices"]]
        if not indices:
            raise ValueError(f"empty schedule group at step {expected_step}")
        if any(index < 0 or index >= len(lengths) for index in indices):
            raise ValueError(f"schedule index out of range at step {expected_step}")
        source_bytes = sum(int(lengths[index]) for index in indices)
        if source_bytes != int(group["source_bytes"]):
            raise ValueError(f"schedule byte total mismatch at step {expected_step}")
        if int(group["raw_byte_target_tokens"]) != source_bytes:
            raise ValueError(f"schedule raw-byte target mismatch at step {expected_step}")
        if int(group["nonpad_target_tokens"]) != source_bytes + len(indices):
            raise ValueError(f"schedule non-PAD target mismatch at step {expected_step}")
        if int(group["example_count"]) != len(indices):
            raise ValueError(f"schedule example count mismatch at step {expected_step}")
        flattened.extend(indices)
    if len(flattened) != len(lengths) or len(set(flattened)) != len(lengths):
        raise ValueError("schedule is not a complete train permutation")
    if sorted(flattened) != list(range(len(lengths))):
        raise ValueError("schedule permutation has missing indices")
    if int(schedule["total_source_bytes"]) != sum(lengths):
        raise ValueError("schedule corpus byte total mismatch")
    return schedule


def generate_frozen_lr_vector(peak_lr: float, warmup_steps: int, total_steps: int) -> torch.Tensor:
    if warmup_steps <= 0 or total_steps <= warmup_steps:
        raise ValueError("invalid warmup/total step contract")
    values: List[float] = []
    for step in range(warmup_steps):
        values.append(peak_lr * (step + 1) / warmup_steps)
    decay_steps = total_steps - warmup_steps
    for step in range(warmup_steps, total_steps):
        progress = (step - warmup_steps + 1) / decay_steps
        values.append(peak_lr * (0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))))
    return torch.tensor(values, dtype=torch.float32)


def loss_sums(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    values = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )
    flat_targets = targets.reshape(-1)
    valid = flat_targets != PAD_ID
    raw = flat_targets >= RAW_BYTE_START
    return values[valid].sum(), values[raw].sum(), int(valid.sum().item()), int(raw.sum().item())


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_trial(
    seed: int,
    peak_lr: float,
    anchor_info: Dict[str, Any],
    schedule_info: Dict[str, Any],
    manifest: Dict[str, Any],
    config_sha: str,
    device: torch.device,
    base_dir: str,
    max_steps: int | None,
) -> List[Dict[str, Any]]:
    hp = manifest["hyperparameters"]
    context = int(hp["context_window_length"])
    top_k = int(hp["top_k"])
    anchor_path = os.path.join(base_dir, anchor_info["relative_path"])
    assert_file_sha256_integrity(anchor_path, anchor_info["binary_payload_sha256"])
    train_info = manifest["splits"]["train"]
    val_info = manifest["splits"]["validation"]
    train_lengths = read_jsonl_source_lengths(base_dir, train_info)
    val_lengths = read_jsonl_source_lengths(base_dir, val_info)
    train_dataset = ImmutableBinaryDataset(
        os.path.join(base_dir, train_info["relative_path"]), context,
        train_info["binary_payload_sha256"], int(train_info["total_constructed_examples"]),
    )
    validation_dataset = ImmutableBinaryDataset(
        os.path.join(base_dir, val_info["relative_path"]), context,
        val_info["binary_payload_sha256"], int(val_info["total_constructed_examples"]),
    )
    if len(train_lengths) != len(train_dataset) or len(val_lengths) != len(validation_dataset):
        raise ValueError("dataset/source length mismatch")
    schedule = load_schedule(base_dir, schedule_info, seed, context, train_lengths)
    steps_to_run = int(hp["num_steps"])
    if max_steps is not None:
        if max_steps < 1 or max_steps > steps_to_run:
            raise ValueError("max_steps outside manifest range")
        steps_to_run = max_steps

    ResearchMoETelemetry.configure(int(hp["vocab_size"]))
    model = BraidPrototypeLM(
        int(hp["vocab_size"]), int(hp["n_layers"]), int(hp["d_model"]),
        int(hp["n_heads"]), int(hp["latent_dim_kv"]), int(hp["latent_dim_q"]),
        int(hp["pe_dim"]), int(hp["num_experts"]), int(hp["d_ff"]), top_k=top_k,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(hp["expected_parameter_geometry"]):
        raise ValueError(f"model geometry mismatch: {parameter_count}")
    model.load_state_dict(torch.load(anchor_path, map_location=device, weights_only=True), strict=True)

    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    no_decay_types = (nn.RMSNorm, nn.LayerNorm, nn.Embedding)
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            (no_decay_params if isinstance(module, no_decay_types) or name.endswith("bias") else decay_params).append(parameter)
    decay_ids = {id(parameter) for parameter in decay_params}
    no_decay_ids = {id(parameter) for parameter in no_decay_params}
    all_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if decay_ids & no_decay_ids or decay_ids | no_decay_ids != all_ids:
        raise ValueError("optimizer parameter partition mismatch")
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": float(hp["weight_decay"]), "lr": peak_lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": peak_lr},
        ],
        betas=(float(hp["adamw_beta1"]), float(hp["adamw_beta2"])), eps=float(hp["adamw_eps"]),
    )
    lr_vector = generate_frozen_lr_vector(peak_lr, int(hp["warmup_steps"]), int(hp["num_steps"]))
    autocast_context = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    interval_train_ce_num = 0.0
    interval_train_raw_ce_num = 0.0
    interval_train_tokens = 0
    interval_train_raw_tokens = 0
    interval_aux_loss_num = 0.0
    interval_aux_tokens = 0
    interval_grad_norms: List[float] = []
    interval_total_steps = 0
    interval_clipped_steps = 0
    interval_nan_inf_steps = 0
    interval_source_bytes = 0
    interval_examples = 0
    trajectory: List[Dict[str, Any]] = []
    interval_started = time.perf_counter()
    ResearchMoETelemetry.reset_step_buffer()
    model.train()

    for step in range(steps_to_run):
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = lr_vector[step].item()
        optimizer.zero_grad()
        group = schedule["groups"][step]
        indices = [int(index) for index in group["indices"]]
        microbatch_size = int(group["microbatch_size"])
        total_valid_step = int(group["nonpad_target_tokens"])
        if total_valid_step <= 0:
            raise ValueError("empty optimization target group")
        interval_source_bytes += int(group["source_bytes"])
        interval_examples += len(indices)

        for microbatch_index, start in enumerate(range(0, len(indices), microbatch_size)):
            batch_indices = indices[start : start + microbatch_size]
            x_batch = torch.stack([train_dataset[index]["x"] for index in batch_indices]).to(device)
            y_batch = torch.stack([train_dataset[index]["y"] for index in batch_indices]).to(device)
            seq_idx = torch.arange(x_batch.size(1), device=device).unsqueeze(0).expand(x_batch.size(0), -1)
            current_sequence_ids = torch.arange(x_batch.size(0), device=device) + (microbatch_index * microbatch_size)
            with autocast_context:
                logits = model(
                    x_batch, seq_idx, current_branch="B",
                    current_sequence_ids=current_sequence_ids,
                    telemetry_class=ResearchMoETelemetry,
                )
                ce_sum, raw_ce_sum, valid_count, raw_count = loss_sums(logits, y_batch)
                auxiliary_loss = torch.stack([layer.moe.last_aux_loss for layer in model.layers]).mean()
                loss = (ce_sum + float(hp["branch_b_aux_coef"]) * auxiliary_loss * valid_count) / total_valid_step
            loss.backward()
            interval_train_ce_num += float(ce_sum.item())
            interval_train_raw_ce_num += float(raw_ce_sum.item())
            interval_train_tokens += valid_count
            interval_train_raw_tokens += raw_count
            interval_aux_loss_num += float(auxiliary_loss.item()) * valid_count
            interval_aux_tokens += valid_count

        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), float(hp["max_norm_clip"])).item())
        interval_total_steps += 1
        if not math.isfinite(grad_norm):
            interval_nan_inf_steps += 1
            optimizer.zero_grad()
        else:
            interval_grad_norms.append(grad_norm)
            if grad_norm > float(hp["max_norm_clip"]):
                interval_clipped_steps += 1
            optimizer.step()

        current_step = step + 1
        if current_step not in hp["checkpoint_intervals"]:
            continue
        synchronize(device)
        train_wall = time.perf_counter() - interval_started
        train_summary = ResearchMoETelemetry.get_summary("B")
        ResearchMoETelemetry.reset_step_buffer()
        model.eval()
        validation_started = time.perf_counter()
        total_val_ce = 0.0
        total_val_raw_ce = 0.0
        total_val_tokens = 0
        total_val_raw_tokens = 0
        val_loader = DataLoader(validation_dataset, batch_size=int(hp["microbatch_size"]), shuffle=False)
        with torch.no_grad():
            for validation_batch in val_loader:
                x_val = validation_batch["x"].to(device)
                y_val = validation_batch["y"].to(device)
                seq_idx_val = torch.arange(x_val.size(1), device=device).unsqueeze(0).expand(x_val.size(0), -1)
                current_sequence_ids_val = torch.arange(x_val.size(0), device=device)
                with autocast_context:
                    logits_val = model(
                        x_val, seq_idx_val, current_branch="B",
                        current_sequence_ids=current_sequence_ids_val,
                        telemetry_class=None,
                    )
                    val_ce, val_raw_ce, val_tokens, val_raw_tokens = loss_sums(logits_val, y_val)
                total_val_ce += float(val_ce.item())
                total_val_raw_ce += float(val_raw_ce.item())
                total_val_tokens += val_tokens
                total_val_raw_tokens += val_raw_tokens
        synchronize(device)
        validation_wall = time.perf_counter() - validation_started
        validation_nll = total_val_ce / max(1, total_val_tokens)
        validation_raw_nll = total_val_raw_ce / max(1, total_val_raw_tokens)
        grad_tensor = torch.tensor(interval_grad_norms) if interval_grad_norms else torch.tensor([0.0])
        layer_zero = train_summary.get("layer_0_moe", {})
        peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        trajectory.append(
            {
                "interval_step_end": current_step,
                "learning_rate": lr_vector[step].item(),
                "model_parameter_count": parameter_count,
                "context_window_length": context,
                "interval_train_ce": interval_train_ce_num / max(1, interval_train_tokens),
                "interval_train_raw_byte_nll": interval_train_raw_ce_num / max(1, interval_train_raw_tokens),
                "interval_train_raw_byte_targets": interval_train_raw_tokens,
                "interval_train_source_bytes": interval_source_bytes,
                "interval_train_examples": interval_examples,
                "validation_nll": validation_nll,
                "validation_raw_byte_nll": validation_raw_nll,
                "validation_bits_per_raw_byte": validation_raw_nll / math.log(2.0),
                "validation_raw_byte_perplexity": math.exp(min(20.0, validation_raw_nll)),
                "validation_raw_byte_targets": total_val_raw_tokens,
                "validation_source_bytes": sum(val_lengths),
                "validation_eos_targets_excluded": True,
                "byte_perplexity": math.exp(min(20.0, validation_nll)),
                "grad_norm_mean": float(grad_tensor.mean().item()),
                "grad_norm_max": float(grad_tensor.max().item()),
                "grad_norm_p95": float(torch.quantile(grad_tensor, 0.95).item()),
                "interval_clipped_fraction": interval_clipped_steps / max(1, interval_total_steps),
                "interval_nan_inf_count": interval_nan_inf_steps,
                "router_interval_cv": float(layer_zero.get("global_hard_cv", 0.0)),
                "router_interval_entropy": float(layer_zero.get("normalized_entropy", 1.0)),
                "router_interval_load_ratio": float(layer_zero.get("max_min_hard_ratio", 1.0)),
                "interval_branch_b_aux_loss": interval_aux_loss_num / max(1, interval_aux_tokens),
                "interval_wall_seconds": train_wall,
                "validation_wall_seconds": validation_wall,
                "interval_source_bytes_per_second": interval_source_bytes / max(train_wall, 1e-12),
                "interval_optimizer_updates_per_second": interval_total_steps / max(train_wall, 1e-12),
                "peak_cuda_memory_bytes": peak_memory,
                "master_config_sha256": config_sha,
                "layer_telemetry": train_summary,
            }
        )
        interval_train_ce_num = 0.0
        interval_train_raw_ce_num = 0.0
        interval_train_tokens = 0
        interval_train_raw_tokens = 0
        interval_aux_loss_num = 0.0
        interval_aux_tokens = 0
        interval_grad_norms = []
        interval_total_steps = 0
        interval_clipped_steps = 0
        interval_nan_inf_steps = 0
        interval_source_bytes = 0
        interval_examples = 0
        ResearchMoETelemetry.reset_step_buffer()
        interval_started = time.perf_counter()
        model.train()
    return trajectory


def main() -> int:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser(description="Crucible E16 context gate")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    config_path = os.path.abspath(args.config_path)
    assert_file_sha256_integrity(config_path, args.config_sha256)
    base_dir = os.path.dirname(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["meta"].get("training_started"):
        raise ValueError("manifest is already marked training-started")
    if manifest["contract"]["architectural_branches"] != ["B"]:
        raise ValueError("context gate must run Branch B only")
    if manifest["hyperparameters"]["num_experts"] != 16 or manifest["hyperparameters"]["top_k"] != 2:
        raise ValueError("context gate geometry mismatch")
    model_core_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_core.py")
    assert_file_sha256_integrity(model_core_path, manifest["code_provenance"]["model_core_sha256"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required by this frozen contract")

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output:
        for seed in manifest["contract"]["seeds"]:
            for learning_rate in manifest["contract"]["learning_rate_candidates"]:
                trajectory = run_trial(
                    int(seed), float(learning_rate), manifest["anchors"][str(seed)],
                    manifest["schedules"][str(seed)], manifest, args.config_sha256,
                    device, base_dir, args.max_steps,
                )
                for point in trajectory:
                    row = {
                        "seed": int(seed),
                        "branch": "B",
                        "peak_lr": float(learning_rate),
                        "variant_id": manifest["meta"]["variant_id"],
                        "num_experts": int(manifest["hyperparameters"]["num_experts"]),
                        "top_k": int(manifest["hyperparameters"]["top_k"]),
                        "d_ff": int(manifest["hyperparameters"]["d_ff"]),
                        **point,
                    }
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                output.flush()
                os.fsync(output.fileno())
    print(f"metrics_sha256={sha256_file(output_path)}")
    with open(output_path, encoding="utf-8") as handle:
        print(f"rows={sum(1 for line in handle if line.strip())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
