#!/usr/bin/env python3
"""Run the authorized CNSS D-skill rank-versus-horizon control."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_crucible_track_a_cnss_partner_generalization_v1_20261101 as partner
from verify_crucible_cnss_workspace_artifact_20261101 import load_bundle

os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROTOCOL = ROOT / "docs" / "CRUCIBLE_TRACK_A_CNSS_D_SKILL_CORRECTIVE_CONTROLS_V1_20261101.json"
MANIFEST_PATH = ROOT / "experiments" / "crucible_track_a_cnss_partner_generalization_v1_20261101" / "MANIFEST_PUBLIC.json"
OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101"
SEALED_ROOT = Path("/home/timbushnell/crucible_artifacts/crucible_cnss_d_skill_corrective_controls_v1_20261101")
WORKSPACE_ARTIFACT = ROOT / "experiments" / "crucible_track_a_cnss_semantic_interface_compatibility_v1_rerun_contractfix_20261101" / "workspace_artifacts" / "a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c"
EXPECTED_WORKSPACE_DIGEST = "a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c"
LAMBDA_VALUES = (0.1, 0.4, 0.9, 1.0)
SEEDS = (42, 1337, 2025)
RANKS = (8, 32)
UPDATE_VALUES = (2000, 8000)
SKILL = "D"
CONSTANT = 0x37
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
CHECKPOINTS = {
    2000: (0, 500, 1000, 1500, 2000),
    8000: (0, 1000, 2000, 4000, 6000, 8000),
}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_nested(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
    if isinstance(value, dict):
        return all(finite_nested(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_nested(item) for item in value)
    return True


class RankedSkillAdapter(nn.Module):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(partner.runner.STATE_WIDTH, rank, bias=False)
        self.up = nn.Linear(rank, partner.runner.STATE_WIDTH, bias=False)
        self.operand = nn.Linear(partner.runner.STATE_BITS, rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(rank))
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.operand.weight, mean=0.0, std=0.02)

    def forward(self, state: torch.Tensor, operand: torch.Tensor) -> torch.Tensor:
        update = self.up(torch.tanh(self.down(state) + self.operand(operand) + self.bias))
        return state + update

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def load_workspace() -> partner.runner.Workspace:
    state, _architecture, _hashes = load_bundle(WORKSPACE_ARTIFACT, EXPECTED_WORKSPACE_DIGEST)
    workspace = partner.runner.Workspace()
    workspace.load_state_dict(state, strict=True)
    workspace.eval()
    for parameter in workspace.parameters():
        parameter.requires_grad_(False)
    if partner.runner.state_dict_digest(workspace) != EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError("workspace digest mismatch")
    if not partner.runner.finite_module(workspace):
        raise RuntimeError("workspace has non-finite parameters")
    return workspace


def load_protocol_and_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "REGISTERED_SCIENTIFIC_TRAINING_AUTHORIZED":
        raise RuntimeError(f"corrective protocol is not authorized: {protocol['status']}")
    if protocol["authorization_boundary"]["gpu_authorized"] is not False:
        raise RuntimeError("GPU boundary is not false")
    if protocol["factors"]["adapter_rank"] != list(RANKS):
        raise RuntimeError("rank matrix differs from protocol")
    if protocol["factors"]["updates"] != list(UPDATE_VALUES):
        raise RuntimeError("update matrix differs from protocol")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    regenerated = partner.make_manifest()
    if regenerated["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("partner manifest changed")
    if not all(regenerated["checks"].values()):
        raise RuntimeError("partner manifest checks failed")
    return protocol, manifest


def batch_values(values: list[int], step: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(values), generator=generator).tolist()
    start = ((step - 1) * BATCH_SIZE) % len(values)
    indices = [order[(start + offset) % len(values)] for offset in range(BATCH_SIZE)]
    return torch.tensor([values[index] for index in indices], dtype=torch.int64)


def output_and_contract(
    workspace: partner.runner.Workspace,
    adapter: RankedSkillAdapter,
    values: torch.Tensor,
    lambda_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_values = partner.skill_target(SKILL, values)
    with torch.no_grad():
        base_state = workspace.encode(partner.runner.bits_from_int(values))
        canonical_state = workspace.encode(partner.runner.bits_from_int(target_values))
    transformed = adapter(base_state, partner.operand_bits(SKILL, len(values)))
    logits = workspace.decode_logits(transformed)
    output = F.binary_cross_entropy_with_logits(logits, partner.runner.bits_from_int(target_values))
    contract = ((transformed - canonical_state).pow(2).sum(dim=1) / partner.runner.STATE_WIDTH).mean()
    return output, contract, transformed, target_values


def evaluate(
    workspace: partner.runner.Workspace,
    adapter: RankedSkillAdapter,
    values: list[int],
) -> dict[str, Any]:
    tensor_values = torch.tensor(values, dtype=torch.int64)
    with torch.no_grad():
        output, contract, transformed, targets = output_and_contract(workspace, adapter, tensor_values, 0.0)
        logits = workspace.decode_logits(transformed)
        exact, predictions = partner.runner.exact_bits(logits, partner.runner.bits_from_int(targets))
    return {
        "count": len(values),
        "exact_accuracy": exact,
        "loss": float(output.item()),
        "contract_mse": float(contract.item()),
        "predictions": predictions,
        "targets": [int(value) for value in targets.tolist()],
    }


def checkpoint_payload(
    workspace: partner.runner.Workspace,
    adapter: RankedSkillAdapter,
    optimizer: torch.optim.Optimizer,
    seed: int,
    rank: int,
    updates: int,
    lambda_value: float,
    step: int,
) -> dict[str, Any]:
    return {
        "schema_version": "crucible-cnss-d-corrective-checkpoint-v1-20261101",
        "skill": SKILL,
        "seed": seed,
        "rank": rank,
        "updates": updates,
        "lambda": lambda_value,
        "step": step,
        "parameter_count": adapter.parameter_count,
        "workspace_digest": partner.runner.state_dict_digest(workspace),
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_random_state": random.getstate(),
    }


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def verify_checkpoint(
    path: Path,
    workspace: partner.runner.Workspace,
    adapter: RankedSkillAdapter,
    optimizer: torch.optim.Optimizer,
    rank: int,
    updates: int,
    step: int,
) -> str:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema_version": "crucible-cnss-d-corrective-checkpoint-v1-20261101",
        "skill": SKILL,
        "rank": rank,
        "updates": updates,
        "step": step,
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
    }
    for key, value in expected.items():
        if loaded.get(key) != value:
            raise RuntimeError(f"checkpoint field mismatch {key}: {path}")
    if not finite_nested(loaded):
        raise RuntimeError(f"non-finite checkpoint payload: {path}")
    reloaded = RankedSkillAdapter(rank)
    reloaded.load_state_dict(loaded["adapter"], strict=True)
    if partner.runner.state_dict_digest(reloaded) != partner.runner.state_dict_digest(adapter):
        raise RuntimeError(f"adapter checkpoint round-trip mismatch: {path}")
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    reloaded_optimizer.load_state_dict(loaded["optimizer"])
    if not partner.runner.check_finite_optimizer(reloaded_optimizer):
        raise RuntimeError(f"optimizer checkpoint state is non-finite: {path}")
    if not partner.runner.finite_module(reloaded):
        raise RuntimeError(f"reloaded adapter is non-finite: {path}")
    if partner.runner.state_dict_digest(workspace) != EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError(f"workspace changed during checkpoint validation: {path}")
    return digest_bytes(path.read_bytes())


def train_condition(
    workspace: partner.runner.Workspace,
    manifest: dict[str, Any],
    seed: int,
    rank: int,
    updates: int,
    lambda_value: float,
) -> tuple[RankedSkillAdapter, dict[str, Any]]:
    skill_index = 3
    derived_seed = seed + 100000 * skill_index + int(lambda_value * 10000) + 910000 + rank * 17 + updates
    partner.runner.set_seed(derived_seed)
    adapter = RankedSkillAdapter(rank)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_values = manifest["partitions"][SKILL]["train"]
    validation_values = manifest["partitions"][SKILL]["validation"]
    checkpoint_dir = SEALED_ROOT / "checkpoints" / f"rank_{rank}" / f"updates_{updates}" / f"lambda_{lambda_value:.1f}" / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records = []
    loss_tail = []
    started = time.perf_counter()

    for step in range(updates + 1):
        if step in CHECKPOINTS[updates]:
            path = checkpoint_dir / f"checkpoint_{step:04d}.pt"
            atomic_save(checkpoint_payload(workspace, adapter, optimizer, seed, rank, updates, lambda_value, step), path)
            checkpoint_hash = verify_checkpoint(path, workspace, adapter, optimizer, rank, updates, step)
            records.append({"step": step, "path": str(path), "sha256": checkpoint_hash})
        if step == updates:
            break
        values = batch_values(train_values, step + 1, derived_seed + step * 1009 + len(train_values))
        optimizer.zero_grad(set_to_none=True)
        output, contract, _transformed, _targets = output_and_contract(workspace, adapter, values, lambda_value)
        loss = output + lambda_value * contract
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at rank={rank}, updates={updates}, lambda={lambda_value}, seed={seed}, step={step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), GRAD_CLIP)
        if not partner.runner.check_finite_optimizer(optimizer):
            raise RuntimeError(f"non-finite gradient/optimizer at rank={rank}, updates={updates}, lambda={lambda_value}, seed={seed}, step={step + 1}")
        optimizer.step()
        if not partner.runner.finite_module(adapter):
            raise RuntimeError(f"non-finite adapter at rank={rank}, updates={updates}, lambda={lambda_value}, seed={seed}, step={step + 1}")
        if step >= updates - 10:
            loss_tail.append({"step": step + 1, "total": float(loss.item()), "output": float(output.item()), "contract": float(contract.item())})

    train_eval = evaluate(workspace, adapter, train_values)
    validation_eval = evaluate(workspace, adapter, validation_values)
    test_eval = evaluate(workspace, adapter, manifest["partitions"][SKILL]["test"])
    final_payload = torch.load(records[-1]["path"], map_location="cpu", weights_only=False)
    replay = RankedSkillAdapter(rank)
    replay.load_state_dict(final_payload["adapter"], strict=True)
    replay_eval = evaluate(workspace, replay, manifest["partitions"][SKILL]["test"])
    if replay_eval["predictions"] != test_eval["predictions"]:
        raise RuntimeError("final checkpoint replay changed D outputs")
    return adapter, {
        "skill": SKILL,
        "seed": seed,
        "rank": rank,
        "parameter_count": adapter.parameter_count,
        "updates": updates,
        "lambda": lambda_value,
        "train": train_eval,
        "validation": validation_eval,
        "test": test_eval,
        "replay_test": replay_eval,
        "checkpoint_count": len(records),
        "checkpoints": records,
        "checkpoint_integrity": True,
        "loss_tail": loss_tail,
        "wall_seconds": time.perf_counter() - started,
    }


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"rank_{row['rank']}_updates_{row['updates']}"
        grouped.setdefault(key, []).append(row)
    summary = {}
    for key, items in grouped.items():
        summary[key] = {
            "rank": items[0]["rank"],
            "updates": items[0]["updates"],
            "parameter_count": items[0]["parameter_count"],
            "seed_count": len(items),
            "lambda_count": len(set(item["lambda"] for item in items)),
            "train_accuracy_mean": sum(item["train"]["exact_accuracy"] for item in items) / len(items),
            "train_accuracy_min": min(item["train"]["exact_accuracy"] for item in items),
            "validation_accuracy_mean": sum(item["validation"]["exact_accuracy"] for item in items) / len(items),
            "validation_accuracy_min": min(item["validation"]["exact_accuracy"] for item in items),
            "test_accuracy_mean": sum(item["test"]["exact_accuracy"] for item in items) / len(items),
            "test_accuracy_min": min(item["test"]["exact_accuracy"] for item in items),
            "replay_accuracy_min": min(item["replay_test"]["exact_accuracy"] for item in items),
            "all_seed_lambda_rows_test_at_least_95": all(item["test"]["exact_accuracy"] >= 0.95 for item in items),
            "all_checkpoints_integrity": all(item["checkpoint_integrity"] for item in items),
            "parameter_count_explicit": True,
            "wall_seconds_total": sum(item["wall_seconds"] for item in items),
        }
    return summary


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"train", "validation", "test", "replay_test", "checkpoints"}
    } | {
        "train_exact_accuracy": row["train"]["exact_accuracy"],
        "validation_exact_accuracy": row["validation"]["exact_accuracy"],
        "test_exact_accuracy": row["test"]["exact_accuracy"],
        "replay_test_exact_accuracy": row["replay_test"]["exact_accuracy"],
        "checkpoint_hashes": [{"step": item["step"], "sha256": item["sha256"]} for item in row["checkpoints"]],
    }


def public_answer_free(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"target", "targets", "prediction", "predictions"}:
                return False
            if not public_answer_free(item):
                return False
    elif isinstance(value, list):
        return all(public_answer_free(item) for item in value)
    return True


def main() -> int:
    started = time.perf_counter()
    if OUTPUT.exists() and (OUTPUT / "D_SKILL_CORRECTIVE_CONTROLS_RESULT.json").exists():
        raise RuntimeError("corrective-control result already exists; refusing to overwrite it")
    protocol, manifest = load_protocol_and_manifest()
    workspace = load_workspace()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SEALED_ROOT.mkdir(parents=True, exist_ok=True)
    (SEALED_ROOT / "runs").mkdir(parents=True, exist_ok=True)
    write_json(
        OUTPUT / "AUTHORIZATION.json",
        {
            "schema_version": "crucible-cnss-d-corrective-controls-authorization-v1-20261101",
            "authorization_source": protocol["authorization_boundary"]["authorized_by"],
            "training_authorized": True,
            "gpu_authorized": False,
            "partner_rerun_authorized": False,
            "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
            "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        },
    )
    write_json(
        OUTPUT / "PREFLIGHT_REPORT.json",
        {
            "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
            "protocol_status": protocol["status"],
            "manifest_sha256": manifest["manifest_sha256"],
            "workspace_digest": partner.runner.state_dict_digest(workspace),
            "workspace_finite": partner.runner.finite_module(workspace),
            "gpu_used": False,
            "ranks": list(RANKS),
            "updates": list(UPDATE_VALUES),
            "lambdas": list(LAMBDA_VALUES),
            "seeds": list(SEEDS),
            "checkpoints": {str(key): list(value) for key, value in CHECKPOINTS.items()},
            "partner_rerun_authorized": False,
        },
    )
    write_json(OUTPUT / "MANIFEST_PUBLIC.json", manifest)
    write_json(SEALED_ROOT / "MANIFEST_SEALED.json", manifest)

    rows = []
    for rank in RANKS:
        for updates in UPDATE_VALUES:
            for lambda_value in LAMBDA_VALUES:
                for seed in SEEDS:
                    _adapter, result = train_condition(workspace, manifest, seed, rank, updates, lambda_value)
                    rows.append(result)
                    write_json(
                        SEALED_ROOT / "runs" / f"rank_{rank}_updates_{updates}_lambda_{lambda_value:.1f}_seed_{seed}.json",
                        result,
                    )
                    write_json(
                        OUTPUT / "runs" / f"rank_{rank}_updates_{updates}_lambda_{lambda_value:.1f}_seed_{seed}.json",
                        public_row(result),
                    )

    summary = {
        "schema_version": "crucible-cnss-d-skill-corrective-controls-result-v1-20261101",
        "classification": "COMPLETED_AUTHORIZED_D_ACQUISITION_CONTROL",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
        "manifest_sha256": manifest["manifest_sha256"],
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "skill": SKILL,
        "operation": "x + 0x37 modulo 256",
        "gpu_used": False,
        "ranks": list(RANKS),
        "updates": list(UPDATE_VALUES),
        "lambdas": list(LAMBDA_VALUES),
        "seeds": list(SEEDS),
        "row_count": len(rows),
        "condition_summary": condition_summary(rows),
        "rows": [public_row(row) for row in rows],
        "original_partner_rerun_authorized": False,
        "competence_gate": {
            "required_test_accuracy": 0.95,
            "passed_any_condition": any(
                all(row["test"]["exact_accuracy"] >= 0.95 for row in rows if row["rank"] == rank and row["updates"] == updates)
                for rank in RANKS
                for updates in UPDATE_VALUES
            ),
            "partner_rerun_unlocked": False,
            "rule": "D must pass the individual gate across every registered lambda and seed in a selected condition before partner rerun review.",
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "source_runner_sha256": digest_bytes(Path(__file__).read_bytes()),
        "scientific_boundary": "This is an acquisition control for one 8-bit operation. It does not establish composition, cumulative learning or architecture superiority.",
    }
    if not public_answer_free(summary):
        raise RuntimeError("public corrective result contains answer-bearing keys")
    write_json(OUTPUT / "D_SKILL_CORRECTIVE_CONTROLS_RESULT.json", summary)
    write_json(
        SEALED_ROOT / "D_SKILL_CORRECTIVE_CONTROLS_RESULT_SEALED.json",
        summary | {"raw_rows": rows},
    )

    lines = [
        "# CNSS D-skill corrective controls - CPU result",
        "",
        "Classification: COMPLETED_AUTHORIZED_D_ACQUISITION_CONTROL",
        "",
        "No GPU was used. The original partner-generalization result remains unchanged and partner rerun remains unauthorized.",
        "",
        "| Condition | Parameters | Mean train | Min train | Mean test | Min test | All rows pass 95%? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, value in summary["condition_summary"].items():
        lines.append(
            f"| {key} | {value['parameter_count']} | {value['train_accuracy_mean']:.4f} | "
            f"{value['train_accuracy_min']:.4f} | {value['test_accuracy_mean']:.4f} | "
            f"{value['test_accuracy_min']:.4f} | {value['all_seed_lambda_rows_test_at_least_95']} |"
        )
    lines.extend(
        [
            "",
            f"Rows: {len(rows)}. Every checkpoint was finite, hash-validated and replay-checked during training.",
            "",
            "The 95% individual gate is required before any partner-generalization rerun.",
            "",
            "Machine-readable details: D_SKILL_CORRECTIVE_CONTROLS_RESULT.json.",
        ]
    )
    (OUTPUT / "D_SKILL_CORRECTIVE_CONTROLS_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": summary["classification"],
        "rows": len(rows),
        "conditions": summary["condition_summary"],
        "passed_any_condition": summary["competence_gate"]["passed_any_condition"],
        "partner_rerun_unlocked": False,
        "elapsed_wall_seconds": summary["elapsed_wall_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
