import argparse
import hashlib
import json
import math
import os
import sys
from contextlib import nullcontext
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model_core import BraidPrototypeLM, ResearchMoETelemetry


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
        assert actual_hash == expected_hash, (
            f"Cryptographic Verification Failed for: {full_path}"
        )

        self.seq_len = seq_len
        self.stride = 2 * seq_len * 8
        assert len(raw_bytes) % self.stride == 0
        self.total_samples = len(raw_bytes) // self.stride
        assert self.total_samples == expected_count, (
            "Dataset file entry count inequality desync."
        )
        self.raw_tensor = torch.frombuffer(
            bytearray(raw_bytes), dtype=torch.int64
        )

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        offset = idx * (2 * self.seq_len)
        return {
            "x": self.raw_tensor[offset : offset + self.seq_len].clone(),
            "y": self.raw_tensor[
                offset + self.seq_len : offset + 2 * self.seq_len
            ].clone(),
        }


def assert_file_sha256_integrity(file_path: str, expected_hash: str) -> None:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    assert hasher.hexdigest() == expected_hash, (
        f"Integrity signature verification failed: {file_path}"
    )


def generate_frozen_lr_vector(
    peak_lr: float, warmup_steps: int, total_steps: int
) -> torch.Tensor:
    if warmup_steps <= 0 or total_steps <= warmup_steps:
        raise ValueError("total_steps must be greater than warmup_steps > 0")
    learning_rates = []
    for step in range(warmup_steps):
        learning_rates.append(
            peak_lr * (float(step + 1) / float(warmup_steps))
        )
    decay_steps = total_steps - warmup_steps
    for step in range(warmup_steps, total_steps):
        progress = float(step - warmup_steps + 1) / float(decay_steps)
        learning_rates.append(
            peak_lr
            * (
                0.1
                + 0.9
                * (0.5 * (1.0 + math.cos(math.pi * progress)))
            )
        )
    return torch.tensor(learning_rates, dtype=torch.float32)


def run_smoke_crucible_trial(
    seed: int,
    branch: str,
    peak_lr: float,
    anchor_info: Dict[str, str],
    schedule_info: Dict[str, str],
    manifest: Dict[str, Any],
    device: torch.device,
    base_dir: str,
) -> List[Dict[str, Any]]:
    del seed
    anchor_path = os.path.join(base_dir, anchor_info["relative_path"])
    schedule_path = os.path.join(base_dir, schedule_info["relative_path"])
    assert_file_sha256_integrity(
        anchor_path, anchor_info["binary_payload_sha256"]
    )
    assert_file_sha256_integrity(
        schedule_path, schedule_info["binary_payload_sha256"]
    )

    hyperparameters = manifest["hyperparameters"]
    top_k = int(
        hyperparameters.get(
            "top_k",
            min(2, int(hyperparameters["num_experts"])),
        )
    )
    if top_k < 1 or top_k > int(hyperparameters["num_experts"]):
        raise ValueError(
            "Manifest top_k must satisfy 1 <= top_k <= num_experts"
        )
    train_split = manifest["splits"]["train"]
    validation_split = manifest["splits"]["validation"]
    train_dataset = ImmutableBinaryDataset(
        os.path.join(base_dir, train_split["relative_path"]),
        hyperparameters["context_window_length"],
        train_split["binary_payload_sha256"],
        train_split["total_constructed_examples"],
    )
    validation_dataset = ImmutableBinaryDataset(
        os.path.join(base_dir, validation_split["relative_path"]),
        hyperparameters["context_window_length"],
        validation_split["binary_payload_sha256"],
        validation_split["total_constructed_examples"],
    )
    assert len(train_dataset) > 0 and len(validation_dataset) > 0

    model = BraidPrototypeLM(
        hyperparameters["vocab_size"],
        hyperparameters["n_layers"],
        hyperparameters["d_model"],
        hyperparameters["n_heads"],
        hyperparameters["latent_dim_kv"],
        hyperparameters["latent_dim_q"],
        hyperparameters["pe_dim"],
        hyperparameters["num_experts"],
        hyperparameters["d_ff"],
        top_k=top_k,
    ).to(device)
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == hyperparameters["expected_parameter_geometry"]
    )
    model.load_state_dict(
        torch.load(anchor_path, map_location=device, weights_only=True),
        strict=True,
    )

    decay_params = []
    no_decay_params = []
    no_decay_types = (nn.RMSNorm, nn.LayerNorm, nn.Embedding)
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if isinstance(module, no_decay_types) or name.endswith("bias"):
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

    decay_ids = {id(parameter) for parameter in decay_params}
    no_decay_ids = {id(parameter) for parameter in no_decay_params}
    all_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    assert decay_ids.isdisjoint(no_decay_ids)
    assert decay_ids | no_decay_ids == all_ids
    assert sum(parameter.numel() for parameter in no_decay_params) == 68608
    assert sum(parameter.numel() for parameter in decay_params) == 43082496

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": hyperparameters["weight_decay"],
                "lr": peak_lr,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "lr": peak_lr,
            },
        ],
        betas=(
            hyperparameters["adamw_beta1"],
            hyperparameters["adamw_beta2"],
        ),
        eps=hyperparameters["adamw_eps"],
    )

    lr_vector = generate_frozen_lr_vector(
        peak_lr,
        hyperparameters["warmup_steps"],
        hyperparameters["num_steps"],
    )
    with open(schedule_path, "rb") as handle:
        schedule_indices = torch.frombuffer(
            bytearray(handle.read()), dtype=torch.int64
        ).clone()

    required_indices_total = (
        hyperparameters["num_steps"]
        * hyperparameters["gradient_accumulation_steps"]
        * hyperparameters["microbatch_size"]
    )
    assert schedule_indices.numel() == required_indices_total
    assert schedule_indices.min().item() >= 0
    assert schedule_indices.max().item() < len(train_dataset)

    schedule_ptr = 0
    trajectory: List[Dict[str, Any]] = []
    interval_train_ce_num = 0.0
    interval_train_tokens_den = 0
    interval_aux_loss_num = 0.0
    interval_aux_tokens_den = 0
    interval_grad_norms: List[float] = []
    interval_total_steps = 0
    interval_clipped_steps = 0
    interval_nan_inf_steps = 0
    bias_deltas_tracked: List[float] = []

    autocast_context = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    model.train()
    ResearchMoETelemetry.reset_step_buffer()

    for step in range(hyperparameters["num_steps"]):
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = lr_vector[step].item()
        optimizer.zero_grad()

        micro_batches = []
        total_valid_tokens_step = 0
        for microbatch_index in range(
            hyperparameters["gradient_accumulation_steps"]
        ):
            indices = schedule_indices[
                schedule_ptr : schedule_ptr
                + hyperparameters["microbatch_size"]
            ]
            schedule_ptr += hyperparameters["microbatch_size"]
            x_batch = torch.stack(
                [train_dataset[index.item()]["x"] for index in indices]
            )
            y_batch = torch.stack(
                [train_dataset[index.item()]["y"] for index in indices]
            )
            valid_token_count = int((y_batch != 0).sum())
            total_valid_tokens_step += valid_token_count
            micro_batches.append((x_batch, y_batch, valid_token_count))

        if total_valid_tokens_step == 0:
            total_valid_tokens_step = 1
        step_contains_nan_inf = False

        for microbatch_index, (x_batch, y_batch, valid_token_count) in enumerate(
            micro_batches
        ):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            seq_idx = torch.arange(x_batch.size(1), device=device).unsqueeze(0)
            seq_idx = seq_idx.expand(x_batch.size(0), -1)
            current_sequence_ids = torch.arange(
                x_batch.size(0), device=device
            ) + (microbatch_index * hyperparameters["microbatch_size"])

            with autocast_context:
                logits = model(
                    x_batch,
                    seq_idx,
                    current_branch=branch,
                    current_sequence_ids=current_sequence_ids,
                    telemetry_class=ResearchMoETelemetry,
                )
                ce_sum = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y_batch.reshape(-1),
                    ignore_index=0,
                    reduction="sum",
                )
                auxiliary_loss = torch.stack(
                    [layer.moe.last_aux_loss for layer in model.layers]
                ).mean()
                loss = (
                    ce_sum
                    + hyperparameters["branch_b_aux_coef"]
                    * auxiliary_loss
                    * valid_token_count
                ) / total_valid_tokens_step

            loss.backward()
            interval_train_ce_num += ce_sum.item()
            interval_train_tokens_den += valid_token_count
            interval_aux_loss_num += auxiliary_loss.item() * valid_token_count
            interval_aux_tokens_den += valid_token_count

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), hyperparameters["max_norm_clip"]
        ).item()
        interval_total_steps += 1
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            interval_nan_inf_steps += 1
            optimizer.zero_grad()
            step_contains_nan_inf = True
            if branch == "C":
                for layer in model.layers:
                    layer.moe.accumulated_expert_counts.zero_()
        else:
            interval_grad_norms.append(grad_norm)
            if grad_norm > hyperparameters["max_norm_clip"]:
                interval_clipped_steps += 1
            optimizer.step()

        if branch == "C" and not step_contains_nan_inf:
            for layer in model.layers:
                bias_delta, _ = layer.moe.execute_global_bias_step_update_(
                    step_size=hyperparameters["branch_c_bias_step"]
                )
                bias_deltas_tracked.append(bias_delta)

        current_step = step + 1
        if current_step not in hyperparameters["checkpoint_intervals"]:
            continue

        train_moe_summary = ResearchMoETelemetry.get_summary(branch)
        ResearchMoETelemetry.reset_step_buffer()
        model.eval()
        total_val_ce_sum = 0.0
        total_val_tokens = 0
        val_loader = DataLoader(
            validation_dataset,
            batch_size=hyperparameters["microbatch_size"],
            shuffle=False,
        )
        with torch.no_grad():
            for validation_batch in val_loader:
                x_val = validation_batch["x"].to(device)
                y_val = validation_batch["y"].to(device)
                seq_idx_val = torch.arange(
                    x_val.size(1), device=device
                ).unsqueeze(0)
                seq_idx_val = seq_idx_val.expand(x_val.size(0), -1)
                current_sequence_ids_val = torch.arange(
                    x_val.size(0), device=device
                )
                with autocast_context:
                    logits_val = model(
                        x_val,
                        seq_idx_val,
                        current_branch=branch,
                        current_sequence_ids=current_sequence_ids_val,
                        telemetry_class=None,
                    )
                    ce_sum_val = F.cross_entropy(
                        logits_val.reshape(-1, logits_val.size(-1)),
                        y_val.reshape(-1),
                        ignore_index=0,
                        reduction="sum",
                    )
                total_val_ce_sum += ce_sum_val.item()
                total_val_tokens += (y_val != 0).sum().item()

        validation_nll = total_val_ce_sum / max(1, total_val_tokens)
        moe_metrics = train_moe_summary.get(
            "layer_0_moe",
            {
                "global_hard_cv": 0.0,
                "normalized_entropy": 1.0,
                "max_min_hard_ratio": 1.0,
            },
        )
        grad_norm_tensor = (
            torch.tensor(interval_grad_norms)
            if interval_grad_norms
            else torch.tensor([0.0])
        )
        trajectory.append(
            {
                "interval_step_end": current_step,
                "learning_rate": lr_vector[step].item(),
                "interval_train_ce": interval_train_ce_num
                / max(1, interval_train_tokens_den),
                "validation_nll": validation_nll,
                "byte_perplexity": math.exp(min(20.0, validation_nll)),
                "grad_norm_mean": grad_norm_tensor.mean().item(),
                "grad_norm_max": grad_norm_tensor.max().item(),
                "grad_norm_p95": torch.quantile(
                    grad_norm_tensor, 0.95
                ).item(),
                "interval_clipped_fraction": interval_clipped_steps
                / max(1, interval_total_steps),
                "interval_nan_inf_count": interval_nan_inf_steps,
                "router_interval_cv": moe_metrics["global_hard_cv"],
                "router_interval_entropy": moe_metrics[
                    "normalized_entropy"
                ],
                "router_interval_load_ratio": moe_metrics[
                    "max_min_hard_ratio"
                ],
                "interval_branch_b_aux_loss": interval_aux_loss_num
                / max(1, interval_aux_tokens_den),
                "interval_branch_c_avg_bias_delta": sum(
                    bias_deltas_tracked
                )
                / max(1, len(bias_deltas_tracked)),
            }
        )
        interval_train_ce_num = 0.0
        interval_train_tokens_den = 0
        interval_aux_loss_num = 0.0
        interval_aux_tokens_den = 0
        interval_total_steps = 0
        interval_clipped_steps = 0
        interval_nan_inf_steps = 0
        interval_grad_norms = []
        bias_deltas_tracked = []
        ResearchMoETelemetry.reset_step_buffer()
        model.train()

    return trajectory


def main() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description="Stateless Crucible Flight Engine")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to master experiment_config.json file asset.",
    )
    parser.add_argument(
        "--config-sha256",
        type=str,
        required=True,
        help="Expected cryptographic SHA signature.",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config_path)
    assert_file_sha256_integrity(config_path, args.config_sha256)
    base_dir = os.path.dirname(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert manifest["meta"]["endianness"] == sys.byteorder
    assert manifest["meta"]["tensor_data_type"] == "int64"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_core_path = os.path.join(script_dir, "model_core.py")
    assert os.path.isfile(model_core_path)
    assert_file_sha256_integrity(
        model_core_path,
        manifest["code_provenance"]["model_core_sha256"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        assert torch.cuda.is_bf16_supported()

    contract = manifest["contract"]
    jsonl_output_path = os.path.abspath("./crucible_matrix_metrics.jsonl")
    with open(jsonl_output_path, "w", encoding="utf-8") as output_file:
        for seed in contract["seeds"]:
            for learning_rate in contract["learning_rate_candidates"]:
                for branch in contract["architectural_branches"]:
                    print(
                        " -> Launching Run Loop: "
                        f"Seed={seed} | Branch={branch} | "
                        f"Max LR={learning_rate:.0e}..."
                    )
                    trajectory = run_smoke_crucible_trial(
                        seed,
                        branch,
                        learning_rate,
                        manifest["anchors"][str(seed)],
                        manifest["schedules"][str(seed)],
                        manifest,
                        device,
                        base_dir,
                    )
                    for point in trajectory:
                        output_file.write(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "branch": branch,
                                    "peak_lr": learning_rate,
                                    "master_config_sha256": args.config_sha256[
                                        :16
                                    ],
                                    **point,
                                }
                            )
                            + "\n"
                        )
                    output_file.flush()
                    os.fsync(output_file.fileno())

    with open(jsonl_output_path, "rb") as handle:
        final_hash = hashlib.sha256(handle.read()).hexdigest()
    print(
        "\n================================================================================"
        f"\n [RUN SUCCESSFUL] Matrix output file SHA-256 Authority Lock : {final_hash}"
        "\n================================================================================"
    )


if __name__ == "__main__":
    main()
