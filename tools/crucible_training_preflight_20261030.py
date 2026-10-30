#!/usr/bin/env python3
"""Fail-closed source and checkpoint audit for future Crucible training runs.

This is intentionally small and dependency-light. It does not authorize a
scientific experiment; it checks that the runner and starting checkpoint carry
the safeguards required by the persistent bug ledger.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def check(name: str, passed: bool, detail: str, checks: List[Dict[str, Any]]) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def finite_state(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item()) if value.is_floating_point() or value.is_complex() else True
    if isinstance(value, dict):
        return all(finite_state(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(item) for item in value)
    return True


def audit_ledger(path: Path, checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    incidents = payload.get("incidents", [])
    ids = [item.get("id") for item in incidents]
    unique = len(ids) == len(set(ids)) and all(re.fullmatch(r"CT-\d{3}", str(item)) for item in ids)
    check("ledger_schema", bool(payload.get("schema_version")) and unique, f"{len(incidents)} incidents; unique CT-IDs={unique}", checks)
    missing_tests = [item.get("id") for item in incidents if not item.get("regression_test")]
    check("ledger_regression_tests", not missing_tests, f"missing regression tests: {missing_tests}", checks)
    return payload


def audit_runner(path: Path, checks: List[Dict[str, Any]]) -> None:
    text = path.read_text(encoding="utf-8")
    required_markers = {
        "finite_state_validation": "finite_state_issue",
        "nonfinite_abort": "NONFINITE_ABORT",
        "resource_abort": "RESOURCE_ABORT",
        "atomic_checkpoint_replace": "os.replace",
        "per_run_token_accounting": "training_tokens_this_run",
        "cpu_rng_restore": "torch_rng_state",
        "cuda_rng_restore": "cuda_rng_state_all",
        "sealed_test_not_loaded": "test.jsonl",
    }
    dependency_text = ""
    if "from run_crucible_foundation_training_20261029 import" in text:
        dependency = path.parent / "run_crucible_foundation_training_20261029.py"
        if dependency.exists():
            dependency_text = dependency.read_text(encoding="utf-8")
    # A training runner should not open the sealed test manifest. The marker is
    # expected to be absent, unlike the safety markers above.
    sealed_access = "test.jsonl" in text
    missing_safety = [
        name
        for name, marker in required_markers.items()
        if name != "sealed_test_not_loaded" and marker not in text and marker not in dependency_text
    ]
    passed = not missing_safety and not sealed_access
    detail = f"missing safety markers={missing_safety}; sealed test access={sealed_access}"
    check(f"runner_safety_markers:{path.name}", passed, detail, checks)


def audit_checkpoint(path: Path, checks: List[Dict[str, Any]]) -> None:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        required = ["model", "optimizer", "step", "training_tokens", "torch_rng_state", "cuda_rng_state_all"]
        missing = [key for key in required if key not in checkpoint]
        finite = finite_state(checkpoint.get("model")) and finite_state(checkpoint.get("optimizer"))
        progress = isinstance(checkpoint.get("step"), int) and int(checkpoint["step"]) >= 0 and int(checkpoint.get("training_tokens", -1)) >= 0
        rng = checkpoint.get("torch_rng_state") is not None
        check("checkpoint_schema", not missing, f"missing fields={missing}", checks)
        check("checkpoint_finite", finite, "model and optimizer floating tensors finite", checks)
        check("checkpoint_progress", progress, f"step={checkpoint.get('step')} training_tokens={checkpoint.get('training_tokens')}", checks)
        check("checkpoint_rng", rng, f"cpu RNG present; CUDA RNG field present={checkpoint.get('cuda_rng_state_all') is not None}", checks)
    except Exception as exc:  # noqa: BLE001 - preflight must report a structured failure
        check("checkpoint_load", False, f"{type(exc).__name__}: {exc}", checks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=ROOT / "docs" / "CRUCIBLE_TRAINING_BUG_LEDGER.json")
    parser.add_argument("--runner", type=Path, action="append")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checks: List[Dict[str, Any]] = []
    audit_ledger(args.ledger.resolve(), checks)
    if args.runner:
        for runner in args.runner:
            audit_runner(runner.resolve(), checks)
    else:
        check("runner_supplied", False, "no runner supplied; source audit not performed", checks)
    if args.checkpoint:
        audit_checkpoint(args.checkpoint.resolve(), checks)
    else:
        check("checkpoint_supplied", False, "no checkpoint supplied; checkpoint audit not performed", checks)

    failures = [item for item in checks if item["status"] == "FAIL"]
    result = {
        "schema_version": "crucible-training-preflight-v1-20261030",
        "status": "FAIL" if failures else "PASS",
        "scientific_authorization": False,
        "checks": checks,
        "failures": failures,
        "policy": "A PASS means implementation safeguards were found; it never authorizes target evaluation or a long training run.",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

