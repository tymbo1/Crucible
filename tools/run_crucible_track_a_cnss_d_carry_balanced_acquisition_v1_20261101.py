#!/usr/bin/env python3
"""Run the authorized CPU-only carry-balanced acquisition control for CNSS Skill D.

The existing rank-32/lambda-0.1/2,000-update arm is the paired control.  This
runner loads each control's persisted step-zero adapter and optimizer state and
changes only the deterministic exposure schedule over the existing 128-state
training candidate support.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101 as control
import run_crucible_track_a_cnss_partner_generalization_v1_20261101 as partner

PROTOCOL = ROOT / "docs" / "CRUCIBLE_TRACK_A_CNSS_D_CARRY_BALANCED_ACQUISITION_V1_20261101.json"
MANIFEST = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101" / "MANIFEST_PUBLIC.json"
SOURCE_RESULT = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101" / "D_SKILL_CORRECTIVE_CONTROLS_RESULT.json"
PREFLIGHT = ROOT / "experiments" / "crucible_track_a_cnss_d_carry_balanced_acquisition_v1_20261101" / "CARRY_BALANCED_PREFLIGHT_REPORT.json"
OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_d_carry_balanced_acquisition_v1_20261101"
SEALED_ROOT = Path("/home/timbushnell/crucible_artifacts/crucible_cnss_d_carry_balanced_acquisition_v1_20261101")
BASELINE_ROOT = control.SEALED_ROOT
WORKSPACE_DIGEST = control.EXPECTED_WORKSPACE_DIGEST

SKILL = "D"
RANK = 32
LAMBDA = 0.1
UPDATES = 2000
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
SEEDS = (42, 1337, 2025)
CHECKPOINT_STEPS = (0, 500, 1000, 1500, 2000)
QUOTAS = {bucket: 15 if bucket < 2 else 14 for bucket in range(9)}
SCHEMA = "crucible-cnss-d-carry-balanced-checkpoint-v1-20261101"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_json(value: Any) -> str:
    return digest_bytes(canonical(value).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def carry_features(value: int) -> tuple[int, int, bool]:
    carry = 0
    carry_outs: list[int] = []
    for bit in range(8):
        total = ((value >> bit) & 1) + ((0x37 >> bit) & 1) + carry
        carry = int(total >= 2)
        carry_outs.append(carry)
    longest = 0
    current = 0
    for carry_out in carry_outs:
        if carry_out:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return sum(carry_outs), longest, bool(carry_outs[-1])


def load_protocol() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "REGISTERED_SCIENTIFIC_TRAINING_AUTHORIZED":
        raise RuntimeError(f"carry-balanced protocol is not authorized: {protocol.get('status')}")
    boundary = protocol.get("authorization_boundary", {})
    if boundary.get("training_authorized") is not True:
        raise RuntimeError("training authorization is not true")
    if boundary.get("gpu_authorized") is not False:
        raise RuntimeError("GPU authorization is not false")
    if boundary.get("partner_rerun_authorized") is not False:
        raise RuntimeError("partner rerun authorization is not false")
    design = protocol.get("design", {})
    expected = {
        "rank": RANK,
        "lambda": LAMBDA,
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRAD_CLIP,
        "device": "cpu",
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise RuntimeError(f"protocol factor differs for {key}: {design.get(key)!r} != {value!r}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    regenerated = partner.make_manifest()
    # The public D manifest is the exact fixed partition used by the corrective
    # controls, not the partner manifest.  Validate its structure directly.
    train = list(manifest["partitions"][SKILL]["train"])
    validation = list(manifest["partitions"][SKILL]["validation"])
    test = list(manifest["partitions"][SKILL]["test"])
    if len(train) != 128 or len(set(train)) != 128:
        raise RuntimeError("D training support is not 128 unique states")
    if len(validation) != 64 or len(test) != 64:
        raise RuntimeError("D validation/test cardinality changed")
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise RuntimeError("D partition overlap detected")
    if not all(0 <= value < 256 for value in train + validation + test):
        raise RuntimeError("D partition contains an out-of-range state")
    if not finite(manifest):
        raise RuntimeError("manifest is non-finite")
    # Rebuilding the partner manifest is a source-module integrity check.  It
    # must still satisfy its own frozen checks, but it is not substituted for
    # the D corrective-control manifest.
    if regenerated.get("manifest_sha256") != json.loads(
        (ROOT / "experiments" / "crucible_track_a_cnss_partner_generalization_v1_20261101" / "MANIFEST_PUBLIC.json").read_text(encoding="utf-8")
    ).get("manifest_sha256"):
        raise RuntimeError("partner manifest generation changed")
    if not all(regenerated["checks"].values()):
        raise RuntimeError("partner manifest checks failed")
    source = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    baseline_rows = [
        row for row in source["rows"]
        if row["rank"] == RANK and row["updates"] == UPDATES and row["lambda"] == LAMBDA
    ]
    if {row["seed"] for row in baseline_rows} != set(SEEDS):
        raise RuntimeError("paired baseline rows are incomplete")
    return protocol, manifest, {"source": source, "baseline_rows": baseline_rows}


def load_workspace() -> partner.runner.Workspace:
    workspace = control.load_workspace()
    if partner.runner.state_dict_digest(workspace) != WORKSPACE_DIGEST:
        raise RuntimeError("workspace digest mismatch")
    if not partner.runner.finite_module(workspace):
        raise RuntimeError("workspace contains non-finite parameters")
    return workspace


def strata(train_values: list[int]) -> dict[int, list[int]]:
    grouped = {bucket: [] for bucket in range(9)}
    for value in train_values:
        grouped[carry_features(value)[0]].append(value)
    if any(not values for values in grouped.values()):
        raise RuntimeError("training candidate support lacks a carry stratum")
    return {bucket: sorted(values) for bucket, values in grouped.items()}


def balanced_epoch(grouped: dict[int, list[int]], derived_seed: int, epoch: int) -> list[int]:
    values: list[int] = []
    for bucket in range(9):
        rng = random.Random(derived_seed + epoch * 1_000_003 + bucket * 10_007)
        candidates = grouped[bucket]
        for _ in range(QUOTAS[bucket]):
            values.append(candidates[rng.randrange(len(candidates))])
    random.Random(derived_seed + epoch * 1_000_003 + 9_000_007).shuffle(values)
    if len(values) != 128:
        raise RuntimeError("balanced epoch cardinality changed")
    if dict(sorted(Counter(carry_features(value)[0] for value in values).items())) != QUOTAS:
        raise RuntimeError("balanced epoch quota mismatch")
    return values


def build_schedule(train_values: list[int], derived_seed: int) -> tuple[list[list[int]], str, dict[str, Any]]:
    grouped = strata(train_values)
    batches: list[list[int]] = []
    epoch_summaries: list[dict[str, Any]] = []
    for step in range(1, UPDATES + 1):
        epoch = (step - 1) // 4
        batch_index = (step - 1) % 4
        while len(epoch_summaries) <= epoch:
            pool = balanced_epoch(grouped, derived_seed, len(epoch_summaries))
            epoch_summaries.append({
                "epoch": len(epoch_summaries),
                "carry_counts": {str(k): v for k, v in sorted(Counter(carry_features(value)[0] for value in pool).items())},
                "pool_digest": digest_json(pool),
            })
        # Reconstruct from the deterministic pool; the digest above makes the
        # exact schedule auditable without storing answer-bearing records.
        pool = balanced_epoch(grouped, derived_seed, epoch)
        batches.append(pool[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE])
    schedule_hash = digest_json(batches)
    return batches, schedule_hash, {
        "candidate_support_size": len(train_values),
        "strata_sizes": {str(k): len(v) for k, v in grouped.items()},
        "quotas": {str(k): QUOTAS[k] for k in range(9)},
        "epoch_size": 128,
        "batches_per_epoch": 4,
        "sampling_with_replacement": True,
        "seed_rule": "derived_seed + epoch*1000003 + bucket*10007 for bucket draws; derived_seed + epoch*1000003 + 9000007 for pool shuffle",
        "epoch_count": (UPDATES + 3) // 4,
        "epoch_summaries": epoch_summaries,
        "schedule_sha256": schedule_hash,
    }


def checkpoint_path(seed: int, step: int) -> Path:
    return SEALED_ROOT / "checkpoints" / "rank_32" / "updates_2000" / "lambda_0.1" / f"seed_{seed}" / f"checkpoint_{step:04d}.pt"


def load_paired_step0(seed: int) -> tuple[control.RankedSkillAdapter, torch.optim.Optimizer, dict[str, Any], Path]:
    path = BASELINE_ROOT / "checkpoints" / "rank_32" / "updates_2000" / "lambda_0.1" / f"seed_{seed}" / "checkpoint_0000.pt"
    if not path.exists():
        raise RuntimeError(f"missing paired baseline step-0 checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema_version": "crucible-cnss-d-corrective-checkpoint-v1-20261101",
        "skill": SKILL,
        "seed": seed,
        "rank": RANK,
        "updates": UPDATES,
        "lambda": LAMBDA,
        "step": 0,
        "workspace_digest": WORKSPACE_DIGEST,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"paired step-0 field mismatch {key}: {path}")
    if not finite(payload):
        raise RuntimeError(f"paired step-0 checkpoint is non-finite: {path}")
    adapter = control.RankedSkillAdapter(RANK)
    adapter.load_state_dict(payload["adapter"], strict=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(payload["optimizer"])
    if not partner.runner.finite_module(adapter) or not partner.runner.check_finite_optimizer(optimizer):
        raise RuntimeError(f"paired step-0 adapter/optimizer is non-finite: {path}")
    return adapter, optimizer, payload, path


def finite_gradients(adapter: torch.nn.Module) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()) for parameter in adapter.parameters())


def save_checkpoint(
    path: Path,
    workspace: partner.runner.Workspace,
    adapter: control.RankedSkillAdapter,
    optimizer: torch.optim.Optimizer,
    seed: int,
    step: int,
    derived_seed: int,
    schedule_hash: str,
) -> str:
    payload = {
        "schema_version": SCHEMA,
        "skill": SKILL,
        "seed": seed,
        "rank": RANK,
        "updates": UPDATES,
        "lambda": LAMBDA,
        "step": step,
        "parameter_count": adapter.parameter_count,
        "workspace_digest": partner.runner.state_dict_digest(workspace),
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_random_state": random.getstate(),
        "derived_schedule_seed": derived_seed,
        "schedule_sha256": schedule_hash,
    }
    if not finite(payload):
        raise RuntimeError(f"refusing non-finite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("schema_version", "skill", "seed", "rank", "updates", "lambda", "step", "workspace_digest", "schedule_sha256"):
        if loaded.get(key) != payload.get(key):
            raise RuntimeError(f"checkpoint round-trip field mismatch {key}: {path}")
    if not finite(loaded):
        raise RuntimeError(f"checkpoint round-trip is non-finite: {path}")
    reloaded = control.RankedSkillAdapter(RANK)
    reloaded.load_state_dict(loaded["adapter"], strict=True)
    replay_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    replay_optimizer.load_state_dict(loaded["optimizer"])
    if partner.runner.state_dict_digest(reloaded) != partner.runner.state_dict_digest(adapter):
        raise RuntimeError(f"adapter checkpoint digest mismatch: {path}")
    if not partner.runner.finite_module(reloaded) or not partner.runner.check_finite_optimizer(replay_optimizer):
        raise RuntimeError(f"checkpoint adapter/optimizer validation failed: {path}")
    if partner.runner.state_dict_digest(workspace) != WORKSPACE_DIGEST:
        raise RuntimeError("workspace changed during checkpoint validation")
    return digest_bytes(path.read_bytes())


def evaluate(workspace: partner.runner.Workspace, adapter: control.RankedSkillAdapter, values: list[int]) -> dict[str, Any]:
    tensor_values = torch.tensor(values, dtype=torch.int64)
    with torch.no_grad():
        output, contract, transformed, targets = control.output_and_contract(workspace, adapter, tensor_values, LAMBDA)
        target_bits = partner.runner.bits_from_int(targets)
        logits = workspace.decode_logits(transformed)
        exact, predictions = partner.runner.exact_bits(logits, target_bits)
        canonical_state = workspace.encode(target_bits)
        displacement = torch.linalg.vector_norm(transformed - canonical_state, dim=1)
        cosine = F.cosine_similarity(transformed, canonical_state, dim=1, eps=1e-8)
        signed_margin = torch.where(target_bits >= 0.5, logits, -logits)
        bit_correct = ((logits >= 0.0).to(torch.float32) == target_bits).to(torch.float32)
    per_carry: dict[str, Any] = {}
    for bucket in range(9):
        indices = [index for index, value in enumerate(values) if carry_features(value)[0] == bucket]
        if indices:
            per_carry[str(bucket)] = {
                "count": len(indices),
                "exact_accuracy": float(sum(predictions[i] == int(targets[i].item()) for i in indices) / len(indices)),
                "mean_displacement": float(displacement[indices].mean().item()),
                "mean_signed_margin": float(signed_margin[indices].mean().item()),
            }
    return {
        "count": len(values),
        "exact_accuracy": exact,
        "loss": float(output.item()),
        "contract_mse": float(contract.item()),
        "mean_displacement": float(displacement.mean().item()),
        "mean_cosine": float(cosine.mean().item()),
        "mean_signed_margin": float(signed_margin.mean().item()),
        "min_signed_margin": float(signed_margin.min().item()),
        "bit_accuracy": float(bit_correct.mean().item()),
        "per_carry": per_carry,
        "predictions": predictions,
        "targets": [int(value) for value in targets.tolist()],
    }


def train_seed(
    workspace: partner.runner.Workspace,
    manifest: dict[str, Any],
    seed: int,
    baseline_row: dict[str, Any],
) -> dict[str, Any]:
    adapter, optimizer, step0_payload, baseline_step0 = load_paired_step0(seed)
    train_values = list(manifest["partitions"][SKILL]["train"])
    derived_seed = seed + 100000 * 3 + int(LAMBDA * 10000) + 910000 + RANK * 17 + UPDATES
    batches, schedule_hash, schedule_meta = build_schedule(train_values, derived_seed)
    schedule_path = SEALED_ROOT / "schedules" / f"seed_{seed}.json"
    write_json(schedule_path, {"seed": seed, "derived_seed": derived_seed, "schedule": batches, "metadata": schedule_meta})
    checkpoint_hashes: list[dict[str, Any]] = []
    start = time.perf_counter()
    step_zero = checkpoint_path(seed, 0)
    step_zero_hash = save_checkpoint(step_zero, workspace, adapter, optimizer, seed, 0, derived_seed, schedule_hash)
    checkpoint_hashes.append({"step": 0, "sha256": step_zero_hash})
    loss_tail: list[dict[str, Any]] = []
    checkpoint_set = set(CHECKPOINT_STEPS)
    for step in range(1, UPDATES + 1):
        values = torch.tensor(batches[step - 1], dtype=torch.int64)
        optimizer.zero_grad(set_to_none=True)
        output, contract, _transformed, _targets = control.output_and_contract(workspace, adapter, values, LAMBDA)
        loss = output + LAMBDA * contract
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss seed={seed} step={step}")
        loss.backward()
        if not finite_gradients(adapter):
            raise RuntimeError(f"non-finite gradients seed={seed} step={step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), GRAD_CLIP)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm seed={seed} step={step}")
        if not partner.runner.check_finite_optimizer(optimizer):
            raise RuntimeError(f"non-finite optimizer state before step seed={seed} step={step}")
        optimizer.step()
        if not partner.runner.finite_module(adapter) or not partner.runner.check_finite_optimizer(optimizer):
            raise RuntimeError(f"non-finite adapter/optimizer after step seed={seed} step={step}")
        if partner.runner.state_dict_digest(workspace) != WORKSPACE_DIGEST:
            raise RuntimeError(f"workspace changed seed={seed} step={step}")
        if step > UPDATES - 10:
            loss_tail.append({"step": step, "total": float(loss.item()), "output": float(output.item()), "contract": float(contract.item()), "grad_norm": float(grad_norm.item())})
        if step in checkpoint_set:
            path = checkpoint_path(seed, step)
            checkpoint_hashes.append({"step": step, "sha256": save_checkpoint(path, workspace, adapter, optimizer, seed, step, derived_seed, schedule_hash)})
    train_eval = evaluate(workspace, adapter, train_values)
    validation_eval = evaluate(workspace, adapter, list(manifest["partitions"][SKILL]["validation"]))
    test_eval = evaluate(workspace, adapter, list(manifest["partitions"][SKILL]["test"]))
    final_payload = torch.load(checkpoint_path(seed, UPDATES), map_location="cpu", weights_only=False)
    replay = control.RankedSkillAdapter(RANK)
    replay.load_state_dict(final_payload["adapter"], strict=True)
    replay_eval = evaluate(workspace, replay, list(manifest["partitions"][SKILL]["test"]))
    if replay_eval["predictions"] != test_eval["predictions"]:
        raise RuntimeError(f"final checkpoint replay changed outputs seed={seed}")
    result = {
        "schema_version": "crucible-cnss-d-carry-balanced-acquisition-run-v1-20261101",
        "skill": SKILL,
        "seed": seed,
        "rank": RANK,
        "updates": UPDATES,
        "lambda": LAMBDA,
        "parameter_count": adapter.parameter_count,
        "derived_schedule_seed": derived_seed,
        "schedule_sha256": schedule_hash,
        "schedule_metadata": schedule_meta,
        "paired_baseline_step0_sha256": digest_bytes(baseline_step0.read_bytes()),
        "paired_baseline_result_row": {
            "train_exact_accuracy": baseline_row["train_exact_accuracy"],
            "validation_exact_accuracy": baseline_row["validation_exact_accuracy"],
            "test_exact_accuracy": baseline_row["test_exact_accuracy"],
        },
        "train": train_eval,
        "validation": validation_eval,
        "test": test_eval,
        "replay_test": replay_eval,
        "checkpoint_hashes": checkpoint_hashes,
        "checkpoint_integrity": True,
        "loss_tail": loss_tail,
        "wall_seconds": time.perf_counter() - start,
    }
    write_json(SEALED_ROOT / "runs" / f"seed_{seed}.json", result)
    return result


def public_run(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": result["schema_version"],
        "skill": result["skill"],
        "seed": result["seed"],
        "rank": result["rank"],
        "updates": result["updates"],
        "lambda": result["lambda"],
        "parameter_count": result["parameter_count"],
        "schedule_sha256": result["schedule_sha256"],
        "paired_baseline_step0_sha256": result["paired_baseline_step0_sha256"],
        "paired_baseline_result_row": result["paired_baseline_result_row"],
        "train_exact_accuracy": result["train"]["exact_accuracy"],
        "validation_exact_accuracy": result["validation"]["exact_accuracy"],
        "test_exact_accuracy": result["test"]["exact_accuracy"],
        "replay_test_exact_accuracy": result["replay_test"]["exact_accuracy"],
        "train_mean_displacement": result["train"]["mean_displacement"],
        "validation_mean_displacement": result["validation"]["mean_displacement"],
        "test_mean_displacement": result["test"]["mean_displacement"],
        "test_mean_cosine": result["test"]["mean_cosine"],
        "test_mean_signed_margin": result["test"]["mean_signed_margin"],
        "test_bit_accuracy": result["test"]["bit_accuracy"],
        "checkpoint_hashes": result["checkpoint_hashes"],
        "checkpoint_integrity": result["checkpoint_integrity"],
        "wall_seconds": result["wall_seconds"],
    }


def public_answer_free(value: Any) -> bool:
    if isinstance(value, dict):
        return all(key not in {"target", "targets", "prediction", "predictions"} and public_answer_free(item) for key, item in value.items())
    if isinstance(value, list):
        return all(public_answer_free(item) for item in value)
    return True


def preflight_report(protocol: dict[str, Any], manifest: dict[str, Any], source: dict[str, Any], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train = list(manifest["partitions"][SKILL]["train"])
    grouped = strata(train)
    sample_pool = balanced_epoch(grouped, 123456789, 0)
    step0_hashes: dict[str, str] = {}
    for seed, row in ((row["seed"], row) for row in baseline_rows):
        path = BASELINE_ROOT / "checkpoints" / "rank_32" / "updates_2000" / "lambda_0.1" / f"seed_{seed}" / "checkpoint_0000.pt"
        actual = digest_bytes(path.read_bytes())
        recorded = next(item["sha256"] for item in row["checkpoint_hashes"] if item["step"] == 0)
        if actual != recorded:
            raise RuntimeError(f"paired step-0 hash mismatch: {path}")
        step0_hashes[str(seed)] = actual
    report = {
        "schema_version": "crucible-cnss-d-carry-balanced-acquisition-authorized-preflight-v1-20261101",
        "classification": "PREFLIGHT_PASSED_TRAINING_AUTHORIZED",
        "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_result_sha256": digest_bytes(SOURCE_RESULT.read_bytes()),
        "workspace_digest": WORKSPACE_DIGEST,
        "training_authorized": True,
        "gpu_used": False,
        "partner_rerun_authorized": False,
        "candidate_support_size": len(train),
        "existing_train_carry_counts": {str(k): len(v) for k, v in grouped.items()},
        "balanced_epoch_size": len(sample_pool),
        "balanced_carry_counts": {str(k): v for k, v in sorted(Counter(carry_features(value)[0] for value in sample_pool).items())},
        "quotas": {str(k): QUOTAS[k] for k in range(9)},
        "candidate_support_unchanged": set(sample_pool).issubset(set(train)),
        "train_validation_test_disjoint": True,
        "fixed_test_manifest": True,
        "paired_step0_checkpoint_hashes": step0_hashes,
        "baseline_rows_retained": len(baseline_rows) == len(SEEDS),
        "authorization_source": protocol["authorization_boundary"]["authorized_by"],
        "source_control_result_sha256": digest_bytes(SOURCE_RESULT.read_bytes()),
        "preflight_correction_preserved": True,
    }
    if report["balanced_carry_counts"] != {str(k): QUOTAS[k] for k in range(9)}:
        raise RuntimeError("authorized balanced preflight quota check failed")
    return report


def replay_only() -> int:
    protocol, manifest, context = load_protocol()
    workspace = load_workspace()
    for seed in SEEDS:
        run_path = SEALED_ROOT / "runs" / f"seed_{seed}.json"
        if not run_path.exists():
            raise RuntimeError(f"missing sealed run for replay: {run_path}")
        result = json.loads(run_path.read_text(encoding="utf-8"))
        payload = torch.load(checkpoint_path(seed, UPDATES), map_location="cpu", weights_only=False)
        if payload.get("workspace_digest") != WORKSPACE_DIGEST or payload.get("schedule_sha256") != result["schedule_sha256"]:
            raise RuntimeError(f"sealed final dependency mismatch seed={seed}")
        adapter = control.RankedSkillAdapter(RANK)
        adapter.load_state_dict(payload["adapter"], strict=True)
        replay_eval = evaluate(workspace, adapter, list(manifest["partitions"][SKILL]["test"]))
        if replay_eval["predictions"] != result["test"]["predictions"]:
            raise RuntimeError(f"fresh-process replay changed predictions seed={seed}")
        if replay_eval["exact_accuracy"] != result["test"]["exact_accuracy"]:
            raise RuntimeError(f"fresh-process replay changed accuracy seed={seed}")
        if not finite(payload) or not partner.runner.finite_module(adapter):
            raise RuntimeError(f"fresh-process replay non-finite seed={seed}")
    print(json.dumps({"classification": "FRESH_PROCESS_REPLAY_PASSED", "seeds": list(SEEDS), "gpu_used": False}, sort_keys=True))
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--replay-only":
        return replay_only()
    if (OUTPUT / "CARRY_BALANCED_ACQUISITION_RESULT.json").exists():
        raise RuntimeError("carry-balanced result already exists; refusing to overwrite")
    started = time.perf_counter()
    protocol, manifest, context = load_protocol()
    workspace = load_workspace()
    preflight = preflight_report(protocol, manifest, context["source"], context["baseline_rows"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SEALED_ROOT.mkdir(parents=True, exist_ok=True)
    (SEALED_ROOT / "runs").mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "AUTHORIZATION.json", {
        "schema_version": "crucible-cnss-d-carry-balanced-authorization-v1-20261101",
        "authorization_source": protocol["authorization_boundary"]["authorized_by"],
        "training_authorized": True,
        "gpu_authorized": False,
        "partner_rerun_authorized": False,
        "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
        "workspace_digest": WORKSPACE_DIGEST,
    })
    write_json(OUTPUT / "CARRY_BALANCED_AUTHORIZED_PREFLIGHT_REPORT.json", preflight)
    (OUTPUT / "CARRY_BALANCED_AUTHORIZED_PREFLIGHT_REPORT.md").write_text(
        "# D carry-balanced acquisition authorized preflight\n\n"
        "Classification: `PREFLIGHT_PASSED_TRAINING_AUTHORIZED`.\n\n"
        "The paired step-zero checkpoints, exact workspace, fixed partitions and carry quotas passed validation. This run is CPU-only; GPU and partner rerun authorization remain false.\n",
        encoding="utf-8",
    )
    write_json(OUTPUT / "MANIFEST_PUBLIC.json", manifest)
    write_json(SEALED_ROOT / "MANIFEST_SEALED.json", manifest)
    baseline_by_seed = {row["seed"]: row for row in context["baseline_rows"]}
    rows = []
    for seed in SEEDS:
        print(json.dumps({"event": "seed_start", "seed": seed}, sort_keys=True), flush=True)
        result = train_seed(workspace, manifest, seed, baseline_by_seed[seed])
        rows.append(result)
        write_json(OUTPUT / "runs" / f"seed_{seed}.json", public_run(result))
        print(json.dumps({"event": "seed_complete", "seed": seed, "train": result["train"]["exact_accuracy"], "validation": result["validation"]["exact_accuracy"], "test": result["test"]["exact_accuracy"]}, sort_keys=True), flush=True)
    replay = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay-only"], cwd=str(ROOT), check=False, capture_output=True, text=True)
    if replay.returncode != 0:
        raise RuntimeError(f"fresh-process replay failed: {replay.stdout}\n{replay.stderr}")
    replay_report = json.loads(replay.stdout.strip().splitlines()[-1])
    public_rows = [public_run(row) for row in rows]
    mean_balanced = sum(row["test"]["exact_accuracy"] for row in rows) / len(rows)
    mean_control = sum(row["paired_baseline_result_row"]["test_exact_accuracy"] for row in rows) / len(rows)
    summary = {
        "schema_version": "crucible-cnss-d-carry-balanced-acquisition-result-v1-20261101",
        "classification": "COMPLETED_AUTHORIZED_CARRY_BALANCED_ACQUISITION_CONTROL",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_result_sha256": digest_bytes(SOURCE_RESULT.read_bytes()),
        "workspace_digest": WORKSPACE_DIGEST,
        "skill": SKILL,
        "operation": "x + 0x37 modulo 256",
        "gpu_used": False,
        "seeds": list(SEEDS),
        "rank": RANK,
        "lambda": LAMBDA,
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "balanced_quotas": {str(k): QUOTAS[k] for k in range(9)},
        "rows": public_rows,
        "paired_control": {
            "source_result": str(SOURCE_RESULT.relative_to(ROOT)),
            "condition": "rank_32_updates_2000_lambda_0.1",
            "mean_test_accuracy": mean_control,
            "balanced_mean_test_accuracy": mean_balanced,
            "mean_delta_balanced_minus_control": mean_balanced - mean_control,
        },
        "fresh_process_replay": replay_report,
        "checkpoint_integrity": all(row["checkpoint_integrity"] for row in rows),
        "competence_gate": {
            "required_test_accuracy": 0.95,
            "balanced_all_seed_pass": all(row["test"]["exact_accuracy"] >= 0.95 for row in rows),
            "partner_rerun_unlocked": False,
            "rule": "Skill D must independently pass the 95% held-out gate before partner-generalization review; this run does not unlock it.",
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "source_runner_sha256": digest_bytes(Path(__file__).read_bytes()),
        "scientific_boundary": "This is a paired CPU acquisition-frequency control for one 8-bit operation. It does not establish a carry algorithm, composition, cumulative learning or architecture superiority.",
    }
    if not public_answer_free(summary):
        raise RuntimeError("public carry-balanced result contains answer-bearing keys")
    write_json(OUTPUT / "CARRY_BALANCED_ACQUISITION_RESULT.json", summary)
    write_json(SEALED_ROOT / "CARRY_BALANCED_ACQUISITION_RESULT_SEALED.json", {"summary": summary, "raw_rows": rows})
    lines = [
        "# CNSS Skill D carry-balanced acquisition control - CPU result",
        "",
        "Classification: `COMPLETED_AUTHORIZED_CARRY_BALANCED_ACQUISITION_CONTROL`.",
        "",
        "The balanced arm used the existing 128-state candidate support, exact carry quotas and paired persisted step-zero states. No GPU was used; partner rerun remains unauthorized.",
        "",
        "| Seed | Train | Validation | Test | Paired control test | Delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in public_rows:
        control_test = row["paired_baseline_result_row"]["test_exact_accuracy"]
        lines.append(f"| {row['seed']} | {row['train_exact_accuracy']:.4f} | {row['validation_exact_accuracy']:.4f} | {row['test_exact_accuracy']:.4f} | {control_test:.4f} | {row['test_exact_accuracy'] - control_test:+.4f} |")
    lines.extend([
        "",
        f"Mean balanced test accuracy: `{mean_balanced:.6f}`; mean paired-control test accuracy: `{mean_control:.6f}`; mean delta: `{mean_balanced - mean_control:+.6f}`.",
        "",
        "All checkpoints were finite, atomically saved, hash-validated, optimizer-validated and replay-checked in a fresh process.",
        "",
        "The 95% individual D gate remains unchanged and partner rerun remains locked.",
    ])
    (OUTPUT / "CARRY_BALANCED_ACQUISITION_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"classification": summary["classification"], "rows": len(rows), "mean_test": mean_balanced, "mean_control_test": mean_control, "mean_delta": mean_balanced - mean_control, "d_gate": summary["competence_gate"]["balanced_all_seed_pass"], "partner_rerun_unlocked": False, "elapsed_wall_seconds": summary["elapsed_wall_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
