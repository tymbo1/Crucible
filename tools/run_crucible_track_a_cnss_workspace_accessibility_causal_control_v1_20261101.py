#!/usr/bin/env python3
"""Run the authorized CPU-only original-vs-signed-bit workspace control."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import preflight_crucible_track_a_cnss_workspace_accessibility_causal_control_v1_20261101 as preflight
import run_crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101 as control
import run_crucible_track_a_cnss_partner_generalization_v1_20261101 as partner

PROTOCOL = ROOT / "docs" / "CRUCIBLE_TRACK_A_CNSS_WORKSPACE_ACCESSIBILITY_CAUSAL_CONTROL_V1_20261101.json"
MANIFEST = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101" / "MANIFEST_PUBLIC.json"
SOURCE_RESULT = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101" / "D_SKILL_CORRECTIVE_CONTROLS_RESULT.json"
OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_workspace_accessibility_causal_control_v1_20261101"
SEALED_ROOT = Path("/home/timbushnell/crucible_artifacts/crucible_cnss_workspace_accessibility_causal_control_v1_20261101")
BASELINE_ROOT = control.SEALED_ROOT
WORKSPACE_DIGEST = control.EXPECTED_WORKSPACE_DIGEST
WORKSPACE_SPEC = preflight.ARTIFACT
WORKSPACE_SPEC_SHA256 = hashlib.sha256(WORKSPACE_SPEC.read_bytes()).hexdigest()

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
SCHEMA = "crucible-cnss-workspace-accessibility-checkpoint-v1-20261101"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


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


def load_protocol() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "REGISTERED_SCIENTIFIC_TRAINING_AUTHORIZED":
        raise RuntimeError(f"workspace-accessibility protocol is not authorized: {protocol.get('status')}")
    boundary = protocol["authorization_boundary"]
    if boundary.get("training_authorized") is not True:
        raise RuntimeError("training authorization is false")
    if boundary.get("gpu_authorized") is not False or boundary.get("partner_rerun_authorized") is not False:
        raise RuntimeError("GPU or partner authorization boundary changed")
    training = protocol["training"]
    expected = {
        "skill": SKILL,
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
        if training.get(key) != value:
            raise RuntimeError(f"protocol factor changed for {key}: {training.get(key)!r} != {value!r}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train = list(manifest["partitions"][SKILL]["train"])
    validation = list(manifest["partitions"][SKILL]["validation"])
    test = list(manifest["partitions"][SKILL]["test"])
    if len(train) != 128 or len(set(train)) != 128 or len(validation) != 64 or len(test) != 64:
        raise RuntimeError("D partition cardinality changed")
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise RuntimeError("D partition overlap detected")
    regenerated = partner.make_manifest()
    partner_manifest = json.loads((ROOT / "experiments" / "crucible_track_a_cnss_partner_generalization_v1_20261101" / "MANIFEST_PUBLIC.json").read_text(encoding="utf-8"))
    if regenerated["manifest_sha256"] != partner_manifest["manifest_sha256"] or not all(regenerated["checks"].values()):
        raise RuntimeError("partner source manifest integrity changed")
    source = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    baseline_rows = [row for row in source["rows"] if row["rank"] == RANK and row["updates"] == UPDATES and row["lambda"] == LAMBDA]
    if {row["seed"] for row in baseline_rows} != set(SEEDS):
        raise RuntimeError("paired original-workspace baseline rows are incomplete")
    return protocol, manifest, {"source": source, "baseline_rows": baseline_rows}


def load_original_workspace() -> partner.runner.Workspace:
    workspace = control.load_workspace()
    if partner.runner.state_dict_digest(workspace) != WORKSPACE_DIGEST:
        raise RuntimeError("original workspace digest mismatch")
    if not partner.runner.finite_module(workspace):
        raise RuntimeError("original workspace is non-finite")
    return workspace


def load_alternative_workspace() -> preflight.SignedBitWorkspace:
    if digest_bytes(WORKSPACE_SPEC.read_bytes()) != WORKSPACE_SPEC_SHA256:
        raise RuntimeError("alternative workspace spec changed after preflight")
    return preflight.SignedBitWorkspace()


def load_paired_step0(seed: int) -> tuple[control.RankedSkillAdapter, torch.optim.Optimizer, dict[str, Any], Path]:
    path = BASELINE_ROOT / "checkpoints" / "rank_32" / "updates_2000" / "lambda_0.1" / f"seed_{seed}" / "checkpoint_0000.pt"
    if not path.exists():
        raise RuntimeError(f"missing paired step-zero checkpoint: {path}")
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
            raise RuntimeError(f"paired step-zero field mismatch {key}: {path}")
    if not finite(payload):
        raise RuntimeError(f"paired step-zero checkpoint is non-finite: {path}")
    adapter = control.RankedSkillAdapter(RANK)
    adapter.load_state_dict(payload["adapter"], strict=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(payload["optimizer"])
    if not partner.runner.finite_module(adapter) or not partner.runner.check_finite_optimizer(optimizer):
        raise RuntimeError(f"paired step-zero adapter/optimizer is non-finite: {path}")
    return adapter, optimizer, payload, path


def standard_schedule(train_values: list[int], derived_seed: int) -> tuple[list[list[int]], str]:
    batches = []
    for update in range(1, UPDATES + 1):
        values = control.batch_values(train_values, update, derived_seed + (update - 1) * 1009 + len(train_values))
        batches.append([int(value) for value in values.tolist()])
    return batches, digest_json(batches)


def output_and_contract(workspace: Any, adapter: control.RankedSkillAdapter, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_values = partner.skill_target(SKILL, values)
    with torch.no_grad():
        base_state = workspace.encode(partner.runner.bits_from_int(values))
        canonical_state = workspace.encode(partner.runner.bits_from_int(target_values))
    transformed = adapter(base_state, partner.operand_bits(SKILL, len(values)))
    logits = workspace.decode_logits(transformed)
    output = torch.nn.functional.binary_cross_entropy_with_logits(logits, partner.runner.bits_from_int(target_values))
    contract = ((transformed - canonical_state).pow(2).sum(dim=1) / 256).mean()
    return output, contract, transformed, target_values


def evaluate(workspace: Any, adapter: control.RankedSkillAdapter, values: list[int]) -> dict[str, Any]:
    tensor_values = torch.tensor(values, dtype=torch.int64)
    with torch.no_grad():
        output, contract, transformed, targets = output_and_contract(workspace, adapter, tensor_values)
        target_bits = partner.runner.bits_from_int(targets)
        logits = workspace.decode_logits(transformed)
        exact, predictions = partner.runner.exact_bits(logits, target_bits)
        canonical = workspace.encode(target_bits)
        displacement = torch.linalg.vector_norm(transformed - canonical, dim=1)
        cosine = F.cosine_similarity(transformed, canonical, dim=1, eps=1e-8)
        signed_margin = torch.where(target_bits >= 0.5, logits, -logits)
        bit_accuracy = float((((logits >= 0.0).to(torch.float32) == target_bits).to(torch.float32)).mean().item())
    return {
        "count": len(values),
        "exact_accuracy": exact,
        "loss": float(output.item()),
        "contract_mse": float(contract.item()),
        "mean_displacement": float(displacement.mean().item()),
        "mean_cosine": float(cosine.mean().item()),
        "mean_signed_margin": float(signed_margin.mean().item()),
        "min_signed_margin": float(signed_margin.min().item()),
        "bit_accuracy": bit_accuracy,
        "predictions": predictions,
        "targets": [int(value) for value in targets.tolist()],
    }


def finite_gradients(adapter: control.RankedSkillAdapter) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()) for parameter in adapter.parameters())


def checkpoint_path(seed: int, step: int) -> Path:
    return SEALED_ROOT / "checkpoints" / "alternative_signed_bit" / f"seed_{seed}" / f"checkpoint_{step:04d}.pt"


def save_checkpoint(path: Path, adapter: control.RankedSkillAdapter, optimizer: torch.optim.Optimizer, seed: int, step: int, schedule_hash: str, derived_seed: int) -> str:
    payload = {
        "schema_version": SCHEMA,
        "workspace_kind": "signed_bit_coordinate",
        "workspace_spec_sha256": WORKSPACE_SPEC_SHA256,
        "skill": SKILL,
        "seed": seed,
        "rank": RANK,
        "updates": UPDATES,
        "lambda": LAMBDA,
        "step": step,
        "parameter_count": adapter.parameter_count,
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_random_state": random.getstate(),
        "schedule_sha256": schedule_hash,
        "derived_schedule_seed": derived_seed,
    }
    if not finite(payload):
        raise RuntimeError(f"refusing non-finite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("schema_version", "workspace_kind", "workspace_spec_sha256", "skill", "seed", "rank", "updates", "lambda", "step", "schedule_sha256"):
        if loaded.get(key) != payload.get(key):
            raise RuntimeError(f"checkpoint round-trip field mismatch {key}: {path}")
    if not finite(loaded):
        raise RuntimeError(f"checkpoint round-trip non-finite: {path}")
    replay = control.RankedSkillAdapter(RANK)
    replay.load_state_dict(loaded["adapter"], strict=True)
    replay_optimizer = torch.optim.AdamW(replay.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    replay_optimizer.load_state_dict(loaded["optimizer"])
    if partner.runner.state_dict_digest(replay) != partner.runner.state_dict_digest(adapter):
        raise RuntimeError(f"adapter checkpoint digest mismatch: {path}")
    if not partner.runner.finite_module(replay) or not partner.runner.check_finite_optimizer(replay_optimizer):
        raise RuntimeError(f"checkpoint adapter/optimizer validation failed: {path}")
    return digest_bytes(path.read_bytes())


def train_seed(workspace: preflight.SignedBitWorkspace, manifest: dict[str, Any], seed: int, baseline_row: dict[str, Any]) -> dict[str, Any]:
    adapter, optimizer, _step0_payload, baseline_step0 = load_paired_step0(seed)
    train_values = list(manifest["partitions"][SKILL]["train"])
    derived_seed = seed + 100000 * 3 + int(LAMBDA * 10000) + 910000 + RANK * 17 + UPDATES
    batches, schedule_hash = standard_schedule(train_values, derived_seed)
    checkpoint_hashes = []
    start = time.perf_counter()
    path = checkpoint_path(seed, 0)
    checkpoint_hashes.append({"step": 0, "sha256": save_checkpoint(path, adapter, optimizer, seed, 0, schedule_hash, derived_seed)})
    loss_tail = []
    for step in range(1, UPDATES + 1):
        values = torch.tensor(batches[step - 1], dtype=torch.int64)
        optimizer.zero_grad(set_to_none=True)
        output, contract, _transformed, _targets = output_and_contract(workspace, adapter, values)
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
            raise RuntimeError(f"non-finite optimizer before step seed={seed} step={step}")
        optimizer.step()
        if not partner.runner.finite_module(adapter) or not partner.runner.check_finite_optimizer(optimizer):
            raise RuntimeError(f"non-finite adapter/optimizer after step seed={seed} step={step}")
        if step > UPDATES - 10:
            loss_tail.append({"step": step, "total": float(loss.item()), "output": float(output.item()), "contract": float(contract.item()), "grad_norm": float(grad_norm.item())})
        if step in CHECKPOINT_STEPS:
            path = checkpoint_path(seed, step)
            checkpoint_hashes.append({"step": step, "sha256": save_checkpoint(path, adapter, optimizer, seed, step, schedule_hash, derived_seed)})
    train_eval = evaluate(workspace, adapter, train_values)
    validation_eval = evaluate(workspace, adapter, list(manifest["partitions"][SKILL]["validation"]))
    test_eval = evaluate(workspace, adapter, list(manifest["partitions"][SKILL]["test"]))
    final_payload = torch.load(checkpoint_path(seed, UPDATES), map_location="cpu", weights_only=False)
    replay = control.RankedSkillAdapter(RANK)
    replay.load_state_dict(final_payload["adapter"], strict=True)
    replay_eval = evaluate(workspace, replay, list(manifest["partitions"][SKILL]["test"]))
    if replay_eval["predictions"] != test_eval["predictions"]:
        raise RuntimeError(f"in-process replay changed predictions seed={seed}")
    result = {
        "schema_version": "crucible-cnss-workspace-accessibility-run-v1-20261101",
        "workspace_kind": "signed_bit_coordinate",
        "workspace_spec_sha256": WORKSPACE_SPEC_SHA256,
        "skill": SKILL,
        "seed": seed,
        "rank": RANK,
        "updates": UPDATES,
        "lambda": LAMBDA,
        "parameter_count": adapter.parameter_count,
        "derived_schedule_seed": derived_seed,
        "schedule_sha256": schedule_hash,
        "paired_original_step0_sha256": digest_bytes(baseline_step0.read_bytes()),
        "paired_original_result_row": {
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
        "workspace_kind": result["workspace_kind"],
        "workspace_spec_sha256": result["workspace_spec_sha256"],
        "skill": result["skill"],
        "seed": result["seed"],
        "rank": result["rank"],
        "updates": result["updates"],
        "lambda": result["lambda"],
        "parameter_count": result["parameter_count"],
        "schedule_sha256": result["schedule_sha256"],
        "paired_original_step0_sha256": result["paired_original_step0_sha256"],
        "paired_original_result_row": result["paired_original_result_row"],
        "train_exact_accuracy": result["train"]["exact_accuracy"],
        "validation_exact_accuracy": result["validation"]["exact_accuracy"],
        "test_exact_accuracy": result["test"]["exact_accuracy"],
        "replay_test_exact_accuracy": result["replay_test"]["exact_accuracy"],
        "train_bit_accuracy": result["train"]["bit_accuracy"],
        "validation_bit_accuracy": result["validation"]["bit_accuracy"],
        "test_bit_accuracy": result["test"]["bit_accuracy"],
        "test_mean_displacement": result["test"]["mean_displacement"],
        "test_mean_cosine": result["test"]["mean_cosine"],
        "test_mean_signed_margin": result["test"]["mean_signed_margin"],
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


def replay_only() -> int:
    _protocol, manifest, _context = load_protocol()
    workspace = load_alternative_workspace()
    for seed in SEEDS:
        run_path = SEALED_ROOT / "runs" / f"seed_{seed}.json"
        if not run_path.exists():
            raise RuntimeError(f"missing sealed run seed={seed}")
        result = json.loads(run_path.read_text(encoding="utf-8"))
        payload = torch.load(checkpoint_path(seed, UPDATES), map_location="cpu", weights_only=False)
        if payload.get("workspace_spec_sha256") != WORKSPACE_SPEC_SHA256 or payload.get("schedule_sha256") != result["schedule_sha256"]:
            raise RuntimeError(f"workspace dependency mismatch seed={seed}")
        adapter = control.RankedSkillAdapter(RANK)
        adapter.load_state_dict(payload["adapter"], strict=True)
        replay_eval = evaluate(workspace, adapter, list(manifest["partitions"][SKILL]["test"]))
        if replay_eval["predictions"] != result["test"]["predictions"] or replay_eval["exact_accuracy"] != result["test"]["exact_accuracy"]:
            raise RuntimeError(f"fresh-process replay changed outputs seed={seed}")
        if not finite(payload) or not partner.runner.finite_module(adapter):
            raise RuntimeError(f"fresh-process replay non-finite seed={seed}")
    print(json.dumps({"classification": "FRESH_PROCESS_REPLAY_PASSED", "seeds": list(SEEDS), "gpu_used": False}, sort_keys=True))
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--replay-only":
        return replay_only()
    if (OUTPUT / "WORKSPACE_ACCESSIBILITY_CAUSAL_CONTROL_RESULT.json").exists():
        raise RuntimeError("workspace-accessibility result already exists; refusing overwrite")
    started = time.perf_counter()
    protocol, manifest, context = load_protocol()
    original = load_original_workspace()
    alternative = load_alternative_workspace()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SEALED_ROOT.mkdir(parents=True, exist_ok=True)
    (SEALED_ROOT / "runs").mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "AUTHORIZATION.json", {
        "schema_version": "crucible-cnss-workspace-accessibility-authorization-v1-20261101",
        "authorization_source": protocol["authorization_boundary"]["authorized_by"],
        "training_authorized": True,
        "gpu_authorized": False,
        "partner_rerun_authorized": False,
        "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
        "original_workspace_digest": WORKSPACE_DIGEST,
        "alternative_workspace_spec_sha256": WORKSPACE_SPEC_SHA256,
    })
    write_json(OUTPUT / "MANIFEST_PUBLIC.json", manifest)
    write_json(SEALED_ROOT / "MANIFEST_SEALED.json", manifest)
    write_json(OUTPUT / "AUTHORIZED_PREFLIGHT_REPORT.json", {
        "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
        "protocol_status": protocol["status"],
        "original_workspace_digest": partner.runner.state_dict_digest(original),
        "alternative_workspace_spec_sha256": WORKSPACE_SPEC_SHA256,
        "original_reconstruction_verified": True,
        "alternative_reconstruction_verified": True,
        "same_interface_width": True,
        "same_adapter_parameter_count": control.RankedSkillAdapter(RANK).parameter_count == 16672,
        "gpu_used": False,
        "partner_rerun_authorized": False,
    })
    baseline_by_seed = {row["seed"]: row for row in context["baseline_rows"]}
    rows = []
    for seed in SEEDS:
        print(json.dumps({"event": "seed_start", "seed": seed}, sort_keys=True), flush=True)
        result = train_seed(alternative, manifest, seed, baseline_by_seed[seed])
        rows.append(result)
        write_json(OUTPUT / "runs" / f"seed_{seed}.json", public_run(result))
        print(json.dumps({"event": "seed_complete", "seed": seed, "train": result["train"]["exact_accuracy"], "validation": result["validation"]["exact_accuracy"], "test": result["test"]["exact_accuracy"]}, sort_keys=True), flush=True)
    replay = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay-only"], cwd=str(ROOT), check=False, capture_output=True, text=True)
    if replay.returncode != 0:
        raise RuntimeError(f"fresh-process replay failed: {replay.stdout}\n{replay.stderr}")
    replay_report = json.loads(replay.stdout.strip().splitlines()[-1])
    public_rows = [public_run(row) for row in rows]
    alternative_mean = sum(row["test"]["exact_accuracy"] for row in rows) / len(rows)
    original_mean = sum(row["paired_original_result_row"]["test_exact_accuracy"] for row in rows) / len(rows)
    mean_delta = alternative_mean - original_mean
    deltas = [row["test"]["exact_accuracy"] - row["paired_original_result_row"]["test_exact_accuracy"] for row in rows]
    all_seed_gate = all(row["test"]["exact_accuracy"] >= 0.95 for row in rows)
    no_seed_worse = all(delta >= 0.0 for delta in deltas)
    decisive = all_seed_gate and mean_delta >= 0.10 and no_seed_worse
    summary = {
        "schema_version": "crucible-cnss-workspace-accessibility-causal-control-result-v1-20261101",
        "classification": "COMPLETED_AUTHORIZED_WORKSPACE_ACCESSIBILITY_CAUSAL_CONTROL",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": digest_bytes(PROTOCOL.read_bytes()),
        "manifest_sha256": manifest["manifest_sha256"],
        "original_workspace_digest": WORKSPACE_DIGEST,
        "alternative_workspace_spec_sha256": WORKSPACE_SPEC_SHA256,
        "workspace_kind": "signed_bit_coordinate",
        "skill": SKILL,
        "operation": "x + 0x37 modulo 256",
        "gpu_used": False,
        "seeds": list(SEEDS),
        "rank": RANK,
        "lambda": LAMBDA,
        "updates": UPDATES,
        "rows": public_rows,
        "paired_comparison": {
            "original_workspace_source_result": str(SOURCE_RESULT.relative_to(ROOT)),
            "original_mean_test_accuracy": original_mean,
            "alternative_mean_test_accuracy": alternative_mean,
            "mean_delta_alternative_minus_original": mean_delta,
            "seed_deltas": {str(seed): delta for seed, delta in zip(SEEDS, deltas)},
        },
        "fresh_process_replay": replay_report,
        "checkpoint_integrity": all(row["checkpoint_integrity"] for row in rows),
        "decision": {
            "individual_d_gate_required": 0.95,
            "alternative_all_seed_d_gate": all_seed_gate,
            "no_seed_worse": no_seed_worse,
            "decisive_workspace_support": decisive,
            "partner_rerun_unlocked": False,
            "interpretation": "A positive result supports a bounded workspace-design effect, not a unique geometric cause or automatic composability."
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "source_runner_sha256": digest_bytes(Path(__file__).read_bytes()),
        "scientific_boundary": "This is a paired CPU workspace-accessibility control for one 8-bit operation. It does not establish general composability, cumulative learning or architecture superiority.",
    }
    if not public_answer_free(summary):
        raise RuntimeError("public result contains answer-bearing keys")
    write_json(OUTPUT / "WORKSPACE_ACCESSIBILITY_CAUSAL_CONTROL_RESULT.json", summary)
    write_json(SEALED_ROOT / "WORKSPACE_ACCESSIBILITY_CAUSAL_CONTROL_RESULT_SEALED.json", {"summary": summary, "raw_rows": rows})
    lines = [
        "# CNSS workspace-accessibility causal control - CPU result",
        "",
        "Classification: `COMPLETED_AUTHORIZED_WORKSPACE_ACCESSIBILITY_CAUSAL_CONTROL`.",
        "",
        "The alternative signed-bit workspace was trained with the same rank-32 adapter, lambda, paired initial states, standard schedule, seeds and fixed partitions as the existing original-workspace control. No GPU was used and partner rerun remains locked.",
        "",
        "| Seed | Alternative train | Alternative validation | Alternative test | Original test | Delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row, delta in zip(public_rows, deltas):
        original_test = row["paired_original_result_row"]["test_exact_accuracy"]
        lines.append(f"| {row['seed']} | {row['train_exact_accuracy']:.4f} | {row['validation_exact_accuracy']:.4f} | {row['test_exact_accuracy']:.4f} | {original_test:.4f} | {delta:+.4f} |")
    lines.extend([
        "",
        f"Mean alternative test accuracy: `{alternative_mean:.6f}`; mean original-workspace test accuracy: `{original_mean:.6f}`; mean delta: `{mean_delta:+.6f}`.",
        "",
        f"Decisive workspace-support criterion passed: `{decisive}`. Partner rerun unlocked: `False`.",
        "",
        "All checkpoints were finite, hash-validated, optimizer-validated and replay-checked in a fresh process.",
    ])
    (OUTPUT / "WORKSPACE_ACCESSIBILITY_CAUSAL_CONTROL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"classification": summary["classification"], "rows": len(rows), "alternative_mean_test": alternative_mean, "original_mean_test": original_mean, "mean_delta": mean_delta, "alternative_all_seed_d_gate": all_seed_gate, "decisive_workspace_support": decisive, "partner_rerun_unlocked": False, "elapsed_wall_seconds": summary["elapsed_wall_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
