#!/usr/bin/env python3
"""Run the bounded CNSS shared-contract lambda sensitivity sweep.

This is a separate CPU-only experiment.  It reuses the verified frozen
workspace artifact and exact registered manifest, then trains the shared
contract adapter pair for lambda=0.1..1.0 at the registered three seeds.
The original CNSS artifacts are never overwritten.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import run_crucible_track_a_cnss_semantic_interface_compatibility_v1_20261101 as runner
from verify_crucible_cnss_workspace_artifact_20261101 import load_bundle


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

SWEEP_ROOT = ROOT / "experiments" / "crucible_track_a_cnss_lambda_sweep_v1_20261101"
SEALED_ROOT = Path("/tmp/crucible_cnss_lambda_sweep_sealed_20261101")
ORIGINAL_ROOT = ROOT / "experiments" / "crucible_track_a_cnss_semantic_interface_compatibility_v1_rerun_contractfix_20261101"
WORKSPACE_ARTIFACT = ORIGINAL_ROOT / "workspace_artifacts" / "a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c"
EXPECTED_WORKSPACE_DIGEST = "a1ec947075a8a100afee37a3595d606b09b6f8befdb94a73605b15f016bdfe4c"
LAMBDA_VALUES = tuple(round(index / 10.0, 1) for index in range(1, 11))
SEEDS = tuple(runner.SKILL_SEEDS)


def finite_nested(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
    if isinstance(value, dict):
        return all(finite_nested(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_nested(item) for item in value)
    return True


def verify_saved_checkpoints(result: dict[str, Any]) -> None:
    for checkpoint in result["checkpoints"]:
        path = ROOT / checkpoint["path"]
        if not path.is_file():
            raise RuntimeError(f"missing checkpoint: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("workspace_digest") != EXPECTED_WORKSPACE_DIGEST:
            raise RuntimeError(f"workspace dependency mismatch: {path}")
        if not finite_nested(payload):
            raise RuntimeError(f"non-finite checkpoint payload: {path}")
        if runner.digest_bytes(path.read_bytes()) != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")


def load_frozen_workspace() -> runner.Workspace:
    state, architecture, hashes = load_bundle(WORKSPACE_ARTIFACT, EXPECTED_WORKSPACE_DIGEST)
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


def verify_manifest() -> dict[str, Any]:
    manifest = runner.make_manifest()
    existing_path = ORIGINAL_ROOT / "MANIFEST_SEALED.json"
    if not existing_path.is_file():
        raise RuntimeError(f"missing registered manifest: {existing_path}")
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    if manifest["manifest_sha256"] != existing["manifest_sha256"]:
        raise RuntimeError("sweep manifest differs from the registered manifest")
    if not all(manifest["checks"].values()):
        raise RuntimeError(f"manifest checks failed: {manifest['checks']}")
    return manifest


def latent_geometry(workspace: runner.Workspace, adapter: runner.SkillAdapter, values: list[int]) -> dict[str, float]:
    tensor_values = torch.tensor(values, dtype=torch.int64)
    with torch.no_grad():
        base = workspace.encode(runner.bits_from_int(tensor_values))
        target_values = runner.skill_target("A", tensor_values)
        canonical = workspace.encode(runner.bits_from_int(target_values))
        actual = adapter(base, runner.operand_bits("A", len(values)))
        error = actual - canonical
        displacement = torch.linalg.vector_norm(error, dim=1)
        actual_norm = torch.linalg.vector_norm(actual, dim=1).clamp_min(1e-12)
        canonical_norm = torch.linalg.vector_norm(canonical, dim=1).clamp_min(1e-12)
        cosine = (actual * canonical).sum(dim=1) / (actual_norm * canonical_norm)
    if not torch.isfinite(displacement).all() or not torch.isfinite(cosine).all():
        raise RuntimeError("non-finite latent geometry")
    return {
        "state_count": len(values),
        "mean_displacement_l2": float(displacement.mean().item()),
        "max_displacement_l2": float(displacement.max().item()),
        "mean_canonical_cosine": float(cosine.mean().item()),
        "min_canonical_cosine": float(cosine.min().item()),
    }


def make_public(value: Any) -> Any:
    return runner.redact_answers(value)


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(f"{run['lambda']:.1f}", []).append(run)
    summary: dict[str, Any] = {}
    for lambda_label, items in grouped.items():
        def values(path: tuple[str, ...]) -> list[float]:
            output = []
            for item in items:
                current: Any = item
                for key in path:
                    current = current[key]
                output.append(float(current))
            return output

        def mean(path: tuple[str, ...]) -> float:
            selected = values(path)
            return sum(selected) / len(selected)

        def minimum(path: tuple[str, ...]) -> float:
            return min(values(path))

        summary[lambda_label] = {
            "seed_count": len(items),
            "seeds": [item["seed"] for item in items],
            "a_test_accuracy_mean": mean(("skill_a", "test", "exact_accuracy")),
            "a_test_accuracy_min_seed": minimum(("skill_a", "test", "exact_accuracy")),
            "b_test_accuracy_mean": mean(("skill_b", "test", "exact_accuracy")),
            "b_test_accuracy_min_seed": minimum(("skill_b", "test", "exact_accuracy")),
            "a_to_b_composition_accuracy_mean": mean(("composition_a_to_b", "exact_accuracy")),
            "a_to_b_composition_accuracy_min_seed": minimum(("composition_a_to_b", "exact_accuracy")),
            "b_to_a_composition_accuracy_mean": mean(("composition_b_to_a", "exact_accuracy")),
            "b_to_a_composition_accuracy_min_seed": minimum(("composition_b_to_a", "exact_accuracy")),
            "retention_drop_mean": mean(("retention_drop",)),
            "latent_displacement_l2_mean": mean(("latent_geometry", "mean_displacement_l2")),
            "latent_canonical_cosine_mean": mean(("latent_geometry", "mean_canonical_cosine")),
            "training_wall_seconds_mean": mean(("training_wall_seconds",)),
            "training_wall_seconds_total": sum(values(("training_wall_seconds",))),
            "a_to_b_all_seeds_at_least_90_percent": minimum(("composition_a_to_b", "exact_accuracy")) >= 0.90,
            "individual_all_seeds_at_least_95_percent": min(
                minimum(("skill_a", "test", "exact_accuracy")),
                minimum(("skill_b", "test", "exact_accuracy")),
            ) >= 0.95,
        }
    return summary


def main() -> int:
    started = time.perf_counter()
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    SEALED_ROOT.mkdir(parents=True, exist_ok=True)
    (SEALED_ROOT / "runs").mkdir(parents=True, exist_ok=True)

    manifest = verify_manifest()
    workspace = load_frozen_workspace()
    runner.OUTPUT = SWEEP_ROOT
    authorization = {
        "schema_version": "crucible-cnss-lambda-sweep-authorization-v1-20261101",
        "authorization_source": "explicit user instruction: test lambda 0.1 through 1.0 in 0.1 increments",
        "scientific_training": True,
        "gpu": False,
        "lambda_values": list(LAMBDA_VALUES),
        "seeds": list(SEEDS),
        "updates_per_skill": runner.SKILL_STEPS,
        "manifest_sha256": manifest["manifest_sha256"],
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "original_artifacts_preserved": True,
        "new_seed_validation": False,
        "larger_domain_validation": False,
    }
    (SWEEP_ROOT / "AUTHORIZATION.json").write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (SWEEP_ROOT / "MANIFEST_PUBLIC.json").write_text(
        json.dumps({key: value for key, value in manifest.items() if key not in {"a_test_outputs"}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preflight = {
        "schema_version": "crucible-cnss-lambda-sweep-preflight-v1-20261101",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_checks": manifest["checks"],
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "workspace_artifact": str(WORKSPACE_ARTIFACT.relative_to(ROOT)),
        "workspace_artifact_hashes": json.loads((WORKSPACE_ARTIFACT / "HASHES.json").read_text(encoding="utf-8")),
        "gpu_used": False,
        "lambda_values": list(LAMBDA_VALUES),
        "seeds": list(SEEDS),
        "updates_per_skill": runner.SKILL_STEPS,
        "checkpoints": list(runner.CHECKPOINT_STEPS),
        "same_manifest_as_original": True,
        "composition_rows_in_training_or_validation": False,
    }
    (SWEEP_ROOT / "PREFLIGHT_REPORT.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runs: list[dict[str, Any]] = []
    for lambda_value in LAMBDA_VALUES:
        config = f"lambda_{lambda_value:.1f}"
        for seed in SEEDS:
            skill_a = runner.train_adapter(workspace, manifest, seed, "A", "B_shared_contract", config, lambda_value, 0.0)
            verify_saved_checkpoints(skill_a)
            skill_b = runner.train_adapter(workspace, manifest, seed, "B", "B_shared_contract", config, lambda_value, 0.0)
            verify_saved_checkpoints(skill_b)
            adapter_a = runner.SkillAdapter().to(runner.DEVICE)
            adapter_b = runner.SkillAdapter().to(runner.DEVICE)
            adapter_a.load_state_dict(skill_a["adapter_state"])
            adapter_b.load_state_dict(skill_b["adapter_state"])
            adapter_a.eval()
            adapter_b.eval()
            composition_ab = runner.evaluate_composition(workspace, adapter_a, adapter_b, manifest["primary_composition_inputs"], "A_to_B")
            composition_ba = runner.evaluate_composition(workspace, adapter_a, adapter_b, manifest["primary_composition_inputs"], "B_to_A")
            retention = runner.evaluate_single(workspace, adapter_a, manifest["partitions"]["A"]["test"], "A")
            run = {
                "arm": "B_shared_contract",
                "config": config,
                "lambda": lambda_value,
                "gamma": 0.0,
                "seed": seed,
                "skill_a": runner.strip_state(skill_a),
                "skill_b": runner.strip_state(skill_b),
                "composition_a_to_b": composition_ab,
                "composition_b_to_a": composition_ba,
                "retention_a_after_b": retention,
                "retention_drop": skill_a["test"]["exact_accuracy"] - retention["exact_accuracy"],
                "adapter_a_digest": skill_a["adapter_digest"],
                "adapter_b_digest": skill_b["adapter_digest"],
                "latent_geometry": latent_geometry(workspace, adapter_a, manifest["primary_composition_inputs"]),
                "training_wall_seconds": skill_a["wall_seconds"] + skill_b["wall_seconds"],
            }
            runs.append(run)
            sealed_path = SEALED_ROOT / "runs" / f"{config}_seed_{seed}.json"
            sealed_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            public_path = SWEEP_ROOT / "runs" / f"{config}_seed_{seed}.json"
            public_path.parent.mkdir(parents=True, exist_ok=True)
            public_path.write_text(json.dumps(make_public(run), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "schema_version": "crucible-cnss-lambda-sweep-result-v1-20261101",
        "classification": "COMPLETED_EXPLORATORY_CPU_LAMBDA_SWEEP",
        "manifest_sha256": manifest["manifest_sha256"],
        "workspace_digest": EXPECTED_WORKSPACE_DIGEST,
        "gpu_used": False,
        "lambda_values": list(LAMBDA_VALUES),
        "seeds": list(SEEDS),
        "updates_per_skill": runner.SKILL_STEPS,
        "checkpoints": list(runner.CHECKPOINT_STEPS),
        "runs_count": len(runs),
        "adapter_pairs_count": len(runs),
        "summary_by_lambda": summarize_runs(runs),
        "runs": make_public(runs),
        "scientific_boundary": "This is a bounded 8-bit sensitivity sweep. It does not establish broad generalization, forward transfer, lifelong learning, or superiority over other architectures. New seeds, larger state domains, and longer chains remain untested.",
        "elapsed_wall_seconds": time.perf_counter() - started,
        "source_runner_sha256": runner.digest_bytes((TOOLS / "run_crucible_track_a_cnss_semantic_interface_compatibility_v1_20261101.py").read_bytes()),
        "sweep_runner_sha256": runner.digest_bytes(Path(__file__).read_bytes()),
    }
    (SWEEP_ROOT / "CNSS_LAMBDA_SWEEP_RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (SEALED_ROOT / "CNSS_LAMBDA_SWEEP_RESULT_SEALED.json").write_text(
        json.dumps(summary | {"runs": runs}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# CNSS λ Sweep v1 — CPU result",
        "",
        "Classification: **COMPLETED_EXPLORATORY_CPU_LAMBDA_SWEEP**",
        "",
        "The verified frozen workspace and the original registered manifest were reused.",
        "The original CNSS artifacts were not overwritten.",
        "",
        "| λ | A test mean | B test mean | A→B mean | A→B minimum seed | B→A mean | displacement L2 | cosine | wall seconds |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lambda_label, values in summary["summary_by_lambda"].items():
        lines.append(
            f"| {lambda_label} | {values['a_test_accuracy_mean']:.4f} | {values['b_test_accuracy_mean']:.4f} | "
            f"{values['a_to_b_composition_accuracy_mean']:.4f} | {values['a_to_b_composition_accuracy_min_seed']:.4f} | "
            f"{values['b_to_a_composition_accuracy_mean']:.4f} | {values['latent_displacement_l2_mean']:.4f} | "
            f"{values['latent_canonical_cosine_mean']:.4f} | {values['training_wall_seconds_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Adapter-pair runs: `{len(runs)}`; seeds per λ: `{len(SEEDS)}`; updates per skill: `{runner.SKILL_STEPS}`.",
            "",
            "This sweep is exploratory and remains limited to the registered 8-bit mechanism probe.",
            "Promising λ values require independent seeds, a larger held-out domain and longer-chain validation before selection.",
            "",
        ]
    )
    (SWEEP_ROOT / "CNSS_LAMBDA_SWEEP_RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"classification": summary["classification"], "runs": len(runs), "output": str((SWEEP_ROOT / "CNSS_LAMBDA_SWEEP_RESULT.json").relative_to(ROOT)), "summary_by_lambda": summary["summary_by_lambda"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
