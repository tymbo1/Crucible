#!/usr/bin/env python3
"""Run the registered CNSS partner-generalization pilot.

This CPU-only follow-up trains four independent neural skills at the
registered lambda values, selects policies only on development partnerships,
and evaluates those policies on held-out directed partnerships.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_crucible_track_a_cnss_semantic_interface_compatibility_v1_20261101 as runner
from verify_crucible_cnss_workspace_artifact_20261101 import load_bundle

PROTOCOL = ROOT / "docs" / "CRUCIBLE_TRACK_A_CNSS_PARTNER_GENERALIZATION_V1_20261101.json"
OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_partner_generalization_v1_20261101"
SEALED_ROOT = Path("/tmp/crucible_cnss_partner_generalization_sealed_20261101")
ORIGINAL_ROOT = ROOT / "experiments" / "crucible_track_a_cnss_semantic_interface_compatibility_v1_rerun_contractfix_20261101"
WORKSPACE_ARTIFACT = ORIGINAL_ROOT / "workspace_artifacts" / "a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c"
EXPECTED_WORKSPACE_DIGEST = "a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c"

SKILLS = ("A", "B", "C", "D")
LAMBDA_VALUES = (0.1, 0.4, 0.9, 1.0)
SEEDS = (42, 1337, 2025)
DEV_PAIRS = (("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C"), ("B", "D"))
TEST_PAIRS = (("B", "A"), ("C", "B"), ("D", "C"), ("A", "D"), ("C", "A"), ("D", "B"))
ALL_PAIRS = tuple((sender, receiver) for sender in SKILLS for receiver in SKILLS if sender != receiver)
UPDATES = 2000
BATCH_SIZE = 32
CHECKPOINT_STEPS = (0, 500, 1000, 1500, 2000)
LR = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
MIN_COLLISION_FREE = 8
MANIFEST_SEARCH_START = 20261101
MANIFEST_SEARCH_LIMIT = 10000
OPERANDS = {"A": 1, "B": 0x59, "C": 0, "D": 0x37}
SKILL_DESCRIPTIONS = {
    "A": "8-bit rotate-left by 1",
    "B": "8-bit XOR with 0x59",
    "C": "8-bit high/low nibble swap",
    "D": "8-bit modular addition by 0x37",
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(canonical(value).encode("utf-8"))


def finite_nested(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
    if isinstance(value, dict):
        return all(finite_nested(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_nested(item) for item in value)
    return True


def skill_target(skill: str, values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.int64)
    if skill == "A":
        return ((values << 1) & 0xFF) | (values >> 7)
    if skill == "B":
        return values ^ 0x59
    if skill == "C":
        return ((values << 4) & 0xF0) | (values >> 4)
    if skill == "D":
        return (values + 0x37) & 0xFF
    raise ValueError(f"unknown skill: {skill}")


def operand_bits(skill: str, count: int) -> torch.Tensor:
    return runner.bits_from_int(torch.full((count,), OPERANDS[skill], dtype=torch.int64))


def hamming_weight(value: int) -> int:
    return int(value).bit_count()


def stratified_order(seed: int) -> list[int]:
    buckets: dict[int, list[int]] = {weight: [] for weight in range(9)}
    for value in range(256):
        buckets[hamming_weight(value)].append(value)
    for weight, values in buckets.items():
        random.Random(seed + 1009 * (weight + 1)).shuffle(values)
    result: list[int] = []
    while any(buckets.values()):
        for weight in range(9):
            if buckets[weight]:
                result.append(buckets[weight].pop())
    return result


def make_partitions(seed: int) -> dict[str, dict[str, list[int]]]:
    partitions: dict[str, dict[str, list[int]]] = {}
    for index, skill in enumerate(SKILLS):
        order = stratified_order(seed + 7919 * index)
        partitions[skill] = {
            "train": sorted(order[:128]),
            "validation": sorted(order[128:192]),
            "test": sorted(order[192:256]),
        }
    return partitions


def collision_free_inputs(
    partitions: dict[str, dict[str, list[int]]], sender: str, receiver: str
) -> list[int]:
    receiver_seen = set(partitions[receiver]["train"]) | set(partitions[receiver]["validation"])
    sender_test = partitions[sender]["test"]
    values = torch.tensor(sender_test, dtype=torch.int64)
    intermediate = skill_target(sender, values).tolist()
    return [value for value, middle in zip(sender_test, intermediate) if middle not in receiver_seen]


def make_manifest() -> dict[str, Any]:
    selected_seed = None
    selected_partitions = None
    selected_counts = None
    attempts = 0
    for candidate in range(MANIFEST_SEARCH_START, MANIFEST_SEARCH_START + MANIFEST_SEARCH_LIMIT):
        attempts += 1
        partitions = make_partitions(candidate)
        counts = {
            f"{sender}_to_{receiver}": len(collision_free_inputs(partitions, sender, receiver))
            for sender, receiver in ALL_PAIRS
        }
        if min(counts.values()) >= MIN_COLLISION_FREE:
            selected_seed = candidate
            selected_partitions = partitions
            selected_counts = counts
            break
    if selected_seed is None or selected_partitions is None or selected_counts is None:
        raise RuntimeError("no manifest seed met the collision-free threshold")

    pairs = []
    for sender, receiver in ALL_PAIRS:
        full_inputs = selected_partitions[sender]["test"]
        collision_free = collision_free_inputs(selected_partitions, sender, receiver)
        pairs.append(
            {
                "sender": sender,
                "receiver": receiver,
                "split": "development" if (sender, receiver) in DEV_PAIRS else "test",
                "full_inputs": full_inputs,
                "collision_free_inputs": collision_free,
                "full_count": len(full_inputs),
                "collision_free_count": len(collision_free),
                "collision_count": len(full_inputs) - len(collision_free),
            }
        )

    partition_checks = {}
    for skill in SKILLS:
        train = set(selected_partitions[skill]["train"])
        validation = set(selected_partitions[skill]["validation"])
        test = set(selected_partitions[skill]["test"])
        partition_checks[skill] = {
            "train_cardinality": len(train) == 128,
            "validation_cardinality": len(validation) == 64,
            "test_cardinality": len(test) == 64,
            "train_validation_disjoint": not train.intersection(validation),
            "train_test_disjoint": not train.intersection(test),
            "validation_test_disjoint": not validation.intersection(test),
        }
    checks = {
        "all_skills_present": set(selected_partitions) == set(SKILLS),
        "partition_checks": partition_checks,
        "all_directed_pairs_present": {
            (row["sender"], row["receiver"]) for row in pairs
        } == set(ALL_PAIRS),
        "development_test_partnerships_disjoint": not set(DEV_PAIRS).intersection(TEST_PAIRS),
        "development_partnerships_registered": set(DEV_PAIRS) == {
            (row["sender"], row["receiver"]) for row in pairs if row["split"] == "development"
        },
        "test_partnerships_registered": set(TEST_PAIRS) == {
            (row["sender"], row["receiver"]) for row in pairs if row["split"] == "test"
        },
        "minimum_collision_free_inputs": min(selected_counts.values()) >= MIN_COLLISION_FREE,
        "no_composition_training_or_validation": True,
        "answers_absent": all("target" not in row and "prediction" not in row for row in pairs),
    }
    required_checks = (
        "all_skills_present",
        "all_directed_pairs_present",
        "development_test_partnerships_disjoint",
        "development_partnerships_registered",
        "test_partnerships_registered",
        "minimum_collision_free_inputs",
        "no_composition_training_or_validation",
        "answers_absent",
    )
    if not all(checks[key] for key in required_checks):
        raise RuntimeError(f"partner manifest checks failed: {checks}")
    if not all(all(values.values()) for values in partition_checks.values()):
        raise RuntimeError(f"partition checks failed: {partition_checks}")

    manifest: dict[str, Any] = {
        "schema_version": "crucible-cnss-partner-generalization-manifest-v1-20261101",
        "manifest_seed": selected_seed,
        "candidate_search": {
            "start": MANIFEST_SEARCH_START,
            "limit": MANIFEST_SEARCH_LIMIT,
            "attempts": attempts,
            "selection_rule": f"first seed with every directed pair having at least {MIN_COLLISION_FREE} collision-free inputs",
        },
        "skills": SKILL_DESCRIPTIONS,
        "partitions": selected_partitions,
        "development_partnerships": [list(pair) for pair in DEV_PAIRS],
        "test_partnerships": [list(pair) for pair in TEST_PAIRS],
        "pairs": pairs,
        "checks": checks,
        "collision_free_counts": selected_counts,
    }
    manifest["manifest_sha256"] = digest_json(manifest)
    return manifest


def load_protocol() -> tuple[dict[str, Any], str]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_hash = digest_bytes(PROTOCOL.read_bytes())
    if protocol.get("status") != "REGISTERED_SCIENTIFIC_TRAINING_AUTHORIZED":
        raise RuntimeError(f"protocol is not authorized: {protocol.get('status')}")
    if tuple(protocol.get("lambda_candidates", [])) != LAMBDA_VALUES:
        raise RuntimeError("lambda candidates differ from the protocol")
    if tuple(protocol.get("seeds", [])) != SEEDS:
        raise RuntimeError("seeds differ from the protocol")
    if protocol.get("training", {}).get("updates_per_skill") != UPDATES:
        raise RuntimeError("training budget differs from the protocol")
    if protocol.get("training", {}).get("composition_examples_in_training") is not False:
        raise RuntimeError("composition training leakage boundary is not false")
    if protocol.get("training", {}).get("composition_examples_in_validation") is not False:
        raise RuntimeError("composition validation leakage boundary is not false")
    return protocol, protocol_hash


def load_frozen_workspace() -> runner.Workspace:
    state, _architecture, _hashes = load_bundle(WORKSPACE_ARTIFACT, EXPECTED_WORKSPACE_DIGEST)
    workspace = runner.Workspace()
    workspace.load_state_dict(state, strict=True)
    workspace.eval()
    for parameter in workspace.parameters():
        parameter.requires_grad_(False)
    if runner.state_dict_digest(workspace) != EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError("frozen workspace digest mismatch")
    if not runner.finite_module(workspace):
        raise RuntimeError("frozen workspace contains non-finite values")
    return workspace


def batch_values(values: list[int], step: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(values), generator=generator).tolist()
    start = ((step - 1) * BATCH_SIZE) % len(values)
    indices = [order[(start + offset) % len(values)] for offset in range(BATCH_SIZE)]
    return torch.tensor([values[index] for index in indices], dtype=torch.int64)


def output_and_contract(
    workspace: runner.Workspace,
    adapter: runner.SkillAdapter,
    values: torch.Tensor,
    skill: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_bits = runner.bits_from_int(values)
    target_values = skill_target(skill, values)
    with torch.no_grad():
        base_state = workspace.encode(input_bits)
        canonical_state = workspace.encode(runner.bits_from_int(target_values))
    transformed = adapter(base_state, operand_bits(skill, len(values)))
    logits = workspace.decode_logits(transformed)
    output = F.binary_cross_entropy_with_logits(logits, runner.bits_from_int(target_values))
    contract = ((transformed - canonical_state).pow(2).sum(dim=1) / runner.STATE_WIDTH).mean()
    return output, contract, transformed, target_values


def evaluate_single(
    workspace: runner.Workspace,
    adapter: runner.SkillAdapter,
    values: list[int],
    skill: str,
) -> dict[str, Any]:
    tensor_values = torch.tensor(values, dtype=torch.int64)
    with torch.no_grad():
        output, contract, transformed, target_values = output_and_contract(workspace, adapter, tensor_values, skill)
        logits = workspace.decode_logits(transformed)
        accuracy, predictions = runner.exact_bits(logits, runner.bits_from_int(target_values))
    return {
        "count": len(values),
        "exact_accuracy": accuracy,
        "loss": float(output.item()),
        "contract_mse": float(contract.item()),
        "predictions": predictions,
        "targets": [int(value) for value in target_values.tolist()],
    }


def checkpoint_payload(
    workspace: runner.Workspace,
    adapter: runner.SkillAdapter,
    optimizer: torch.optim.Optimizer,
    seed: int,
    skill: str,
    lambda_value: float,
    step: int,
) -> dict[str, Any]:
    return {
        "schema_version": "crucible-cnss-partner-skill-checkpoint-v1-20261101",
        "seed": seed,
        "skill": skill,
        "lambda": lambda_value,
        "step": step,
        "workspace_digest": runner.state_dict_digest(workspace),
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_random_state": random.getstate(),
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def verify_checkpoint(
    path: Path,
    workspace: runner.Workspace,
    adapter: runner.SkillAdapter,
    optimizer: torch.optim.Optimizer,
    expected_step: int,
) -> str:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if loaded.get("schema_version") != "crucible-cnss-partner-skill-checkpoint-v1-20261101":
        raise RuntimeError(f"checkpoint schema mismatch: {path}")
    if loaded.get("workspace_digest") != EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError(f"checkpoint workspace dependency mismatch: {path}")
    if loaded.get("step") != expected_step:
        raise RuntimeError(f"checkpoint step mismatch: {path}")
    if not finite_nested(loaded):
        raise RuntimeError(f"checkpoint contains non-finite state: {path}")
    reloaded_adapter = runner.SkillAdapter()
    reloaded_adapter.load_state_dict(loaded["adapter"], strict=True)
    if runner.state_dict_digest(reloaded_adapter) != runner.state_dict_digest(adapter):
        raise RuntimeError(f"adapter round-trip mismatch: {path}")
    reloaded_optimizer = torch.optim.AdamW(reloaded_adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    reloaded_optimizer.load_state_dict(loaded["optimizer"])
    if not runner.check_finite_optimizer(reloaded_optimizer):
        raise RuntimeError(f"optimizer round-trip contains non-finite state: {path}")
    if not runner.finite_module(reloaded_adapter):
        raise RuntimeError(f"adapter round-trip contains non-finite parameters: {path}")
    if runner.state_dict_digest(workspace) != EXPECTED_WORKSPACE_DIGEST:
        raise RuntimeError(f"workspace changed during checkpoint verification: {path}")
    return digest_bytes(path.read_bytes())


def train_adapter(
    workspace: runner.Workspace,
    manifest: dict[str, Any],
    seed: int,
    skill: str,
    lambda_value: float,
) -> tuple[runner.SkillAdapter, dict[str, Any]]:
    skill_index = SKILLS.index(skill)
    derived_seed = seed + 100000 * skill_index + int(lambda_value * 10000) + 910000
    runner.set_seed(derived_seed)
    adapter = runner.SkillAdapter()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_values = manifest["partitions"][skill]["train"]
    validation_values = manifest["partitions"][skill]["validation"]
    checkpoint_dir = SEALED_ROOT / "checkpoints" / f"lambda_{lambda_value:.1f}" / f"seed_{seed}" / f"skill_{skill}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_records = []
    loss_tail = []
    started = time.perf_counter()

    for step in range(UPDATES + 1):
        if step in CHECKPOINT_STEPS:
            path = checkpoint_dir / f"checkpoint_{step:04d}.pt"
            payload = checkpoint_payload(workspace, adapter, optimizer, seed, skill, lambda_value, step)
            atomic_torch_save(payload, path)
            checkpoint_hash = verify_checkpoint(path, workspace, adapter, optimizer, step)
            checkpoint_records.append({"step": step, "path": str(path), "sha256": checkpoint_hash})
        if step == UPDATES:
            break
        values = batch_values(train_values, step + 1, derived_seed + step * 1009 + len(train_values))
        optimizer.zero_grad(set_to_none=True)
        output, contract, _transformed, _targets = output_and_contract(workspace, adapter, values, skill)
        loss = output + lambda_value * contract
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at {skill}, lambda={lambda_value}, seed={seed}, step={step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), GRAD_CLIP)
        if not runner.check_finite_optimizer(optimizer):
            raise RuntimeError(f"non-finite gradient or optimizer state at {skill}, lambda={lambda_value}, seed={seed}, step={step + 1}")
        optimizer.step()
        if not runner.finite_module(adapter):
            raise RuntimeError(f"non-finite adapter parameters at {skill}, lambda={lambda_value}, seed={seed}, step={step + 1}")
        if step >= UPDATES - 10:
            loss_tail.append(
                {
                    "step": step + 1,
                    "output": float(output.item()),
                    "contract": float(contract.item()),
                    "total": float(loss.item()),
                }
            )

    validation = evaluate_single(workspace, adapter, validation_values, skill)
    test = evaluate_single(workspace, adapter, manifest["partitions"][skill]["test"], skill)
    final_payload = torch.load(checkpoint_records[-1]["path"], map_location="cpu", weights_only=False)
    replay_adapter = runner.SkillAdapter()
    replay_adapter.load_state_dict(final_payload["adapter"], strict=True)
    replay_test = evaluate_single(workspace, replay_adapter, manifest["partitions"][skill]["test"], skill)
    if replay_test["predictions"] != test["predictions"]:
        raise RuntimeError(f"fresh adapter checkpoint replay changed outputs for {skill}, lambda={lambda_value}, seed={seed}")
    result = {
        "seed": seed,
        "skill": skill,
        "lambda": lambda_value,
        "updates": UPDATES,
        "train_count": len(train_values),
        "validation": validation,
        "test": test,
        "replay_test": replay_test,
        "retention_drop_after_reload": test["exact_accuracy"] - replay_test["exact_accuracy"],
        "adapter_digest": runner.state_dict_digest(adapter),
        "checkpoints": checkpoint_records,
        "checkpoint_count": len(checkpoint_records),
        "checkpoint_integrity": True,
        "loss_tail": loss_tail,
        "wall_seconds": time.perf_counter() - started,
    }
    return adapter, result


def latent_metrics(actual: torch.Tensor, canonical_state: torch.Tensor) -> dict[str, float]:
    error = actual - canonical_state
    displacement = torch.linalg.vector_norm(error, dim=1)
    actual_norm = torch.linalg.vector_norm(actual, dim=1).clamp_min(1e-12)
    canonical_norm = torch.linalg.vector_norm(canonical_state, dim=1).clamp_min(1e-12)
    cosine = (actual * canonical_state).sum(dim=1) / (actual_norm * canonical_norm)
    if not torch.isfinite(displacement).all() or not torch.isfinite(cosine).all():
        raise RuntimeError("non-finite composition latent metrics")
    return {
        "mean_displacement_l2": float(displacement.mean().item()),
        "max_displacement_l2": float(displacement.max().item()),
        "mean_canonical_cosine": float(cosine.mean().item()),
        "min_canonical_cosine": float(cosine.min().item()),
    }


def evaluate_pair(
    workspace: runner.Workspace,
    sender_adapter: runner.SkillAdapter,
    receiver_adapter: runner.SkillAdapter,
    sender: str,
    receiver: str,
    values: list[int],
    split_label: str,
) -> dict[str, Any]:
    tensor_values = torch.tensor(values, dtype=torch.int64)
    with torch.no_grad():
        base_state = workspace.encode(runner.bits_from_int(tensor_values))
        sender_target = skill_target(sender, tensor_values)
        final_target = skill_target(receiver, sender_target)
        sender_state = sender_adapter(base_state, operand_bits(sender, len(values)))
        final_state = receiver_adapter(sender_state, operand_bits(receiver, len(values)))
        final_logits = workspace.decode_logits(final_state)
        accuracy, predictions = runner.exact_bits(final_logits, runner.bits_from_int(final_target))
        sender_logits = workspace.decode_logits(sender_state)
        sender_accuracy, _ = runner.exact_bits(sender_logits, runner.bits_from_int(sender_target))
        canonical_sender_state = workspace.encode(runner.bits_from_int(sender_target))
        canonical_final_state = receiver_adapter(canonical_sender_state, operand_bits(receiver, len(values)))
        canonical_logits = workspace.decode_logits(canonical_final_state)
        canonical_accuracy, _ = runner.exact_bits(canonical_logits, runner.bits_from_int(final_target))
        metrics = latent_metrics(sender_state, canonical_sender_state)
        contract = ((sender_state - canonical_sender_state).pow(2).sum(dim=1) / runner.STATE_WIDTH).mean()
    metrics.update(
        {
            "split": split_label,
            "count": len(values),
            "exact_accuracy": accuracy,
            "predictions": predictions,
            "targets": [int(value) for value in final_target.tolist()],
            "first_state_decode_accuracy": sender_accuracy,
            "receiver_on_canonical_state_accuracy": canonical_accuracy,
            "first_intermediate_contract_mse": float(contract.item()),
        }
    )
    return metrics


def make_public(value: Any) -> Any:
    return runner.redact_answers(value)


def pair_score(row: dict[str, Any]) -> float:
    return float(row["collision_free"]["exact_accuracy"])


def select_best(
    rows: Iterable[dict[str, Any]],
    candidates: Iterable[float],
    selector: Any,
) -> tuple[float, dict[str, Any]]:
    rows = list(rows)
    scored = []
    for candidate in candidates:
        selected = [row for row in rows if selector(row, candidate)]
        if not selected:
            raise RuntimeError("lambda selection had no development rows")
        score = sum(pair_score(row) for row in selected) / len(selected)
        scored.append({"lambda": candidate, "score": score, "count": len(selected)})
    best = max(scored, key=lambda item: (item["score"], -item["lambda"]))
    return float(best["lambda"]), {"selected": best, "candidates": scored}


def select_policies(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    dev_rows = [row for row in pair_results if row["partnership_split"] == "development"]
    global_lambda, global_detail = select_best(
        dev_rows,
        LAMBDA_VALUES,
        lambda row, candidate: row["sender_lambda"] == candidate and row["receiver_lambda"] == candidate,
    )

    sender_lambdas: dict[str, float] = {}
    sender_details: dict[str, Any] = {}
    for sender in SKILLS:
        sender_lambdas[sender], sender_details[sender] = select_best(
            dev_rows,
            LAMBDA_VALUES,
            lambda row, candidate, sender=sender: (
                row["sender"] == sender
                and row["sender_lambda"] == candidate
                and row["receiver_lambda"] == global_lambda
            ),
        )

    receiver_lambdas: dict[str, float] = {}
    receiver_details: dict[str, Any] = {}
    for receiver in SKILLS:
        receiver_lambdas[receiver], receiver_details[receiver] = select_best(
            dev_rows,
            LAMBDA_VALUES,
            lambda row, candidate, receiver=receiver: (
                row["receiver"] == receiver
                and row["sender_lambda"] == global_lambda
                and row["receiver_lambda"] == candidate
            ),
        )

    skill_lambdas: dict[str, float] = {}
    skill_details: dict[str, Any] = {}
    for skill in SKILLS:
        skill_lambdas[skill], skill_details[skill] = select_best(
            dev_rows,
            LAMBDA_VALUES,
            lambda row, candidate, skill=skill: (
                (row["sender"] == skill and row["sender_lambda"] == candidate and row["receiver_lambda"] == global_lambda)
                or (row["receiver"] == skill and row["sender_lambda"] == global_lambda and row["receiver_lambda"] == candidate)
            ),
        )

    return {
        "global": {
            "lambda_by_skill_role": {"sender": global_lambda, "receiver": global_lambda},
            "selection": global_detail,
        },
        "sender_only": {
            "sender_lambdas": sender_lambdas,
            "receiver_lambda": global_lambda,
            "selection": sender_details,
        },
        "receiver_only": {
            "sender_lambda": global_lambda,
            "receiver_lambdas": receiver_lambdas,
            "selection": receiver_details,
        },
        "skill_shared": {
            "lambda_by_skill": skill_lambdas,
            "selection": skill_details,
        },
    }


def policy_lambdas(policy_name: str, policy: dict[str, Any], sender: str, receiver: str) -> tuple[float, float]:
    if policy_name == "global":
        return float(policy["lambda_by_skill_role"]["sender"]), float(policy["lambda_by_skill_role"]["receiver"])
    if policy_name == "sender_only":
        return float(policy["sender_lambdas"][sender]), float(policy["receiver_lambda"])
    if policy_name == "receiver_only":
        return float(policy["sender_lambda"]), float(policy["receiver_lambdas"][receiver])
    if policy_name == "skill_shared":
        return float(policy["lambda_by_skill"][sender]), float(policy["lambda_by_skill"][receiver])
    raise ValueError(policy_name)


def summarize_policy_test(pair_results: list[dict[str, Any]], policies: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for policy_name, policy in policies.items():
        rows = []
        for seed in SEEDS:
            for sender, receiver in TEST_PAIRS:
                sender_lambda, receiver_lambda = policy_lambdas(policy_name, policy, sender, receiver)
                matching = [
                    row for row in pair_results
                    if row["seed"] == seed
                    and row["sender"] == sender
                    and row["receiver"] == receiver
                    and row["sender_lambda"] == sender_lambda
                    and row["receiver_lambda"] == receiver_lambda
                ]
                if len(matching) != 1:
                    raise RuntimeError(f"policy lookup failed for {policy_name}, {seed}, {sender}->{receiver}")
                row = matching[0]
                rows.append(
                    {
                        "seed": seed,
                        "sender": sender,
                        "receiver": receiver,
                        "sender_lambda": sender_lambda,
                        "receiver_lambda": receiver_lambda,
                        "collision_free": row["collision_free"],
                        "full": row["full"],
                    }
                )
        pair_means = {}
        for sender, receiver in TEST_PAIRS:
            values = [
                row["collision_free"]["exact_accuracy"]
                for row in rows
                if row["sender"] == sender and row["receiver"] == receiver
            ]
            pair_means[f"{sender}_to_{receiver}"] = sum(values) / len(values)
        seed_means = {}
        for seed in SEEDS:
            values = [row["collision_free"]["exact_accuracy"] for row in rows if row["seed"] == seed]
            seed_means[str(seed)] = sum(values) / len(values)
        accuracies = [row["collision_free"]["exact_accuracy"] for row in rows]
        output[policy_name] = {
            "test_row_count": len(rows),
            "collision_free_accuracy_mean": sum(accuracies) / len(accuracies),
            "collision_free_accuracy_min_row": min(accuracies),
            "weakest_test_partnership_mean": min(pair_means.values()),
            "pair_means": pair_means,
            "seed_means": seed_means,
            "selected_lambdas_by_row": [
                {
                    "seed": row["seed"],
                    "sender": row["sender"],
                    "receiver": row["receiver"],
                    "sender_lambda": row["sender_lambda"],
                    "receiver_lambda": row["receiver_lambda"],
                }
                for row in rows
            ],
            "rows": rows,
        }
    return output


def summarize_tensor(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for sender, receiver in ALL_PAIRS:
        rows = [row for row in pair_results if row["sender"] == sender and row["receiver"] == receiver]
        matrix: dict[str, Any] = {}
        for sender_lambda in LAMBDA_VALUES:
            matrix[f"{sender_lambda:.1f}"] = {}
            for receiver_lambda in LAMBDA_VALUES:
                values = [
                    row["collision_free"]["exact_accuracy"]
                    for row in rows
                    if row["sender_lambda"] == sender_lambda and row["receiver_lambda"] == receiver_lambda
                ]
                matrix[f"{sender_lambda:.1f}"][f"{receiver_lambda:.1f}"] = {
                    "mean": sum(values) / len(values),
                    "min_seed": min(values),
                    "seed_count": len(values),
                }
        output[f"{sender}_to_{receiver}"] = matrix
    return output


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preflight() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol, protocol_hash = load_protocol()
    manifest = make_manifest()
    workspace = load_frozen_workspace()
    if OUTPUT.exists() and (OUTPUT / "CNSS_PARTNER_GENERALIZATION_RESULT.json").exists():
        raise RuntimeError("completed partner result exists; refusing to overwrite it")
    gpu_status = "visible_to_torch_but_forbidden_by_cpu_protocol" if torch.cuda.is_available() else "not_available"
    preflight_report = {
        "schema_version": "crucible-cnss-partner-generalization-preflight-v1-20261101",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": protocol_hash,
        "protocol_status": protocol["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_seed": manifest["manifest_seed"],
        "manifest_checks": manifest["checks"],
        "collision_free_counts": manifest["collision_free_counts"],
        "workspace_artifact": str(WORKSPACE_ARTIFACT.relative_to(ROOT)),
        "workspace_digest_expected": EXPECTED_WORKSPACE_DIGEST,
        "workspace_digest_loaded": runner.state_dict_digest(workspace),
        "workspace_finite": runner.finite_module(workspace),
        "gpu_status": gpu_status,
        "gpu_used": False,
        "lambda_values": list(LAMBDA_VALUES),
        "seeds": list(SEEDS),
        "skills": list(SKILLS),
        "development_partnerships": [list(pair) for pair in DEV_PAIRS],
        "test_partnerships": [list(pair) for pair in TEST_PAIRS],
        "updates_per_skill": UPDATES,
        "checkpoints": list(CHECKPOINT_STEPS),
        "raw_checkpoints_outside_git": True,
        "answers_not_persisted_in_public_artifacts": True,
        "original_cnss_artifacts_immutable": True,
    }
    return protocol, manifest, preflight_report


def run_experiment(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    preflight_report: dict[str, Any],
    workspace: runner.Workspace,
) -> dict[str, Any]:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SEALED_ROOT.mkdir(parents=True, exist_ok=True)
    (SEALED_ROOT / "runs").mkdir(parents=True, exist_ok=True)
    protocol_hash = digest_bytes(PROTOCOL.read_bytes())
    write_json(
        OUTPUT / "AUTHORIZATION.json",
        {
            "schema_version": "crucible-cnss-partner-generalization-authorization-v1-20261101",
            "authorization_source": "explicit user instruction: proceed with the authorized partner-generalization experiment",
            "protocol_sha256": protocol_hash,
            "scientific_training": True,
            "target_evaluation": True,
            "gpu": False,
            "original_artifacts_preserved": True,
            "test_partnerships_used_for_lambda_selection": False,
            "authorized_at_epoch": time.time(),
        },
    )
    write_json(OUTPUT / "MANIFEST_PUBLIC.json", manifest)
    write_json(SEALED_ROOT / "MANIFEST_SEALED.json", manifest)
    write_json(OUTPUT / "PREFLIGHT_REPORT.json", preflight_report)

    adapters: dict[tuple[int, str, float], runner.SkillAdapter] = {}
    training_rows: list[dict[str, Any]] = []
    for lambda_value in LAMBDA_VALUES:
        for seed in SEEDS:
            for skill in SKILLS:
                adapter, result = train_adapter(workspace, manifest, seed, skill, lambda_value)
                adapters[(seed, skill, lambda_value)] = adapter
                training_rows.append(result)
                write_json(
                    SEALED_ROOT / "runs" / f"skill_{skill}_lambda_{lambda_value:.1f}_seed_{seed}.json",
                    result,
                )
                write_json(
                    OUTPUT / "runs" / f"skill_{skill}_lambda_{lambda_value:.1f}_seed_{seed}.json",
                    make_public(result),
                )

    pair_manifest = {(row["sender"], row["receiver"]): row for row in manifest["pairs"]}
    pair_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        for sender, receiver in ALL_PAIRS:
            pair_row = pair_manifest[(sender, receiver)]
            for sender_lambda in LAMBDA_VALUES:
                for receiver_lambda in LAMBDA_VALUES:
                    full = evaluate_pair(
                        workspace,
                        adapters[(seed, sender, sender_lambda)],
                        adapters[(seed, receiver, receiver_lambda)],
                        sender,
                        receiver,
                        pair_row["full_inputs"],
                        pair_row["split"],
                    )
                    collision_free = evaluate_pair(
                        workspace,
                        adapters[(seed, sender, sender_lambda)],
                        adapters[(seed, receiver, receiver_lambda)],
                        sender,
                        receiver,
                        pair_row["collision_free_inputs"],
                        pair_row["split"],
                    )
                    pair_results.append(
                        {
                            "seed": seed,
                            "sender": sender,
                            "receiver": receiver,
                            "partnership_split": pair_row["split"],
                            "sender_lambda": sender_lambda,
                            "receiver_lambda": receiver_lambda,
                            "full": full,
                            "collision_free": collision_free,
                            "full_count": pair_row["full_count"],
                            "collision_free_count": pair_row["collision_free_count"],
                            "collision_count": pair_row["collision_count"],
                        }
                    )

    policies = select_policies(pair_results)
    policy_test = summarize_policy_test(pair_results, policies)
    tensor = summarize_tensor(pair_results)
    individual_rows = [row for row in training_rows]
    individual_gate = all(row["test"]["exact_accuracy"] >= 0.95 for row in individual_rows)
    max_by_skill = {
        skill: max(row["test"]["exact_accuracy"] for row in individual_rows if row["skill"] == skill)
        for skill in SKILLS
    }
    failing_skills = [skill for skill, value in max_by_skill.items() if value < 0.95]
    summary = {
        "schema_version": "crucible-cnss-partner-generalization-result-v1-20261101",
        "classification": (
            "COMPLETED_EXPLORATORY_PARTNER_GENERALIZATION_CPU_PILOT"
            if individual_gate
            else "INCONCLUSIVE_PREREQUISITE_FAILURE"
        ),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": protocol_hash,
        "manifest_sha256": manifest["manifest_sha256"],
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "gpu_used": False,
        "skills": SKILL_DESCRIPTIONS,
        "lambda_values": list(LAMBDA_VALUES),
        "seeds": list(SEEDS),
        "development_partnerships": [list(pair) for pair in DEV_PAIRS],
        "test_partnerships": [list(pair) for pair in TEST_PAIRS],
        "training_rows_count": len(training_rows),
        "pair_rows_count": len(pair_results),
        "prerequisite_gate": {
            "required_individual_test_accuracy": 0.95,
            "passed": individual_gate,
            "max_test_accuracy_by_skill": max_by_skill,
            "failing_skills": failing_skills,
            "interpretation": (
                "Held-out partnership results are descriptive only until every independently trained skill qualifies."
                if not individual_gate
                else "All independently trained skills met the registered individual-competence gate."
            ),
        },
        "training_rows": make_public(training_rows),
        "selection": make_public(policies),
        "held_out_test_policy_results": make_public(policy_test),
        "descriptive_pair_tensor_collision_free": make_public(tensor),
        "all_pair_rows": make_public(pair_results),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "scientific_boundary": (
            "This is a bounded 8-bit mechanism probe. It does not establish broad reasoning, "
            "lifelong learning, forward transfer, scalability, or architectural superiority. "
            "Independent new seeds, larger state spaces and longer chains remain required."
        ),
        "selection_boundary": (
            "All lambda and policy selections used development partnerships only. "
            "Held-out test partnerships were not used for selection."
        ),
        "source_runner_sha256": digest_bytes(Path(__file__).read_bytes()),
        "base_runner_sha256": digest_bytes(
            (TOOLS / "run_crucible_track_a_cnss_semantic_interface_compatibility_v1_20261101.py").read_bytes()
        ),
    }
    write_json(OUTPUT / "CNSS_PARTNER_GENERALIZATION_RESULT.json", summary)
    write_json(
        SEALED_ROOT / "CNSS_PARTNER_GENERALIZATION_RESULT_SEALED.json",
        summary | {"training_rows": training_rows, "all_pair_rows": pair_results},
    )

    lines = [
        "# CNSS Partner Generalization v1 - CPU result",
        "",
        f"Classification: {summary['classification']}",
        "",
        f"Skills: {', '.join(SKILLS)}; seeds: {', '.join(str(seed) for seed in SEEDS)}; "
        f"lambda values: {', '.join(str(value) for value in LAMBDA_VALUES)}.",
        "",
        "Development partnerships: " + ", ".join(f"{a}->{b}" for a, b in DEV_PAIRS) + ".",
        "Held-out test partnerships: " + ", ".join(f"{a}->{b}" for a, b in TEST_PAIRS) + ".",
        "",
        "| Policy | Held-out collision-free mean | Weakest held-out partnership mean |",
        "| --- | ---: | ---: |",
    ]
    for policy_name, result in policy_test.items():
        lines.append(
            f"| {policy_name} | {result['collision_free_accuracy_mean']:.4f} | "
            f"{result['weakest_test_partnership_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Trained skill adapters: {len(training_rows)}. Pair/lambda/seed evaluations: {len(pair_results)}.",
            "",
            "The complete sender-lambda/receiver-lambda tensor is descriptive. "
            "No held-out partnership was used to select a deployable policy.",
            "",
            "This result is limited to the registered 8-bit mechanism probe and does not establish "
            "broad reasoning, lifelong learning, or scalability.",
            "",
            "Machine-readable details: CNSS_PARTNER_GENERALIZATION_RESULT.json.",
            "",
        ]
    )
    (OUTPUT / "CNSS_PARTNER_GENERALIZATION_RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    protocol, manifest, preflight_report = preflight()
    if "--preflight-only" in sys.argv:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_json(OUTPUT / "PREFLIGHT_REPORT.json", preflight_report)
        write_json(OUTPUT / "MANIFEST_PUBLIC.json", manifest)
        print(
            json.dumps(
                {
                    "classification": "PREFLIGHT_PASS",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "manifest_seed": manifest["manifest_seed"],
                    "collision_free_counts": manifest["collision_free_counts"],
                },
                sort_keys=True,
            )
        )
        return 0
    workspace = load_frozen_workspace()
    summary = run_experiment(protocol, manifest, preflight_report, workspace)
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "training_rows": summary["training_rows_count"],
                "pair_rows": summary["pair_rows_count"],
                "policy_results": {
                    name: {
                        "collision_free_accuracy_mean": result["collision_free_accuracy_mean"],
                        "weakest_test_partnership_mean": result["weakest_test_partnership_mean"],
                    }
                    for name, result in summary["held_out_test_policy_results"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
