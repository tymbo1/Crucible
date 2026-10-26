#!/usr/bin/env python3
"""Run the paired role-swapped unequal control without changing the prior run."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "run_crucible_digitwise_diagonal_ablation_20251025.py"
OUTPUT = ROOT / "experiments" / "crucible_digitwise_role_swapped_control_20251025"


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


role = import_module("role_swapped_diagonal_ablation_runner", SOURCE)
role.OUTPUT = OUTPUT
role.ANCHOR_DIR = OUTPUT / "generated_reference_e4" / "crucible_manifest"

diagonal = {(2, 2), (3, 3)}
role_swapped = {(4, 2), (1, 3)}
role.COMMON_BACKGROUND = tuple(role.COMMON_BACKGROUND)
role.VALIDATION_PAIRS = {
    "diagonal": tuple(sorted(set(role.COMMON_BACKGROUND) | diagonal)),
    "unequal_role_swapped": tuple(sorted(set(role.COMMON_BACKGROUND) | role_swapped)),
}
role.SPECIAL = {
    "diagonal": diagonal,
    "unequal_role_swapped": role_swapped,
}


def custom_preflight(records: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]]) -> Dict[str, Any]:
    audits = {condition: role.audit_condition(condition, records[condition]["canonical"]) for condition in role.VALIDATION_PAIRS}
    interface_errors: Dict[str, list[str]] = {}
    for condition in role.VALIDATION_PAIRS:
        for interface in role.INTERFACES:
            errors: list[str] = []
            for split in ("train", "validation"):
                for record in records[condition][interface][split]:
                    if record.get("interface") != interface or record.get("condition") != condition:
                        errors.append(f"metadata:{condition}:{interface}:{split}:{record['record_id']}")
            interface_errors[f"{condition}:{interface}"] = errors

    validation_sets = {condition: set(values) for condition, values in role.VALIDATION_PAIRS.items()}
    errors = [f"{condition}:{error}" for condition, audit in audits.items() for error in audit["errors"]]
    errors.extend(f"{key}:{error}" for key, values in interface_errors.items() for error in values)
    if any(set(role.COMMON_BACKGROUND) - validation_sets[condition] for condition in validation_sets):
        errors.append("common_background")
    if sorted(a + b for a, b in diagonal) != sorted(a + b for a, b in role_swapped):
        errors.append("special_sum_multiset")
    if Counter(b for _, b in diagonal) != Counter(b for _, b in role_swapped):
        errors.append("second_operand_match")
    if {condition: sum(a == b for a, b in values) for condition, values in validation_sets.items()} != {"diagonal": 2, "unequal_role_swapped": 0}:
        errors.append("diagonal_counts")
    return {
        "experiment": "crucible_digitwise_role_swapped_control_20251025",
        "status": "frozen_before_training",
        "conditions": {condition: [list(pair) for pair in values] for condition, values in role.VALIDATION_PAIRS.items()},
        "common_background": [list(pair) for pair in role.COMMON_BACKGROUND],
        "special_pairs": {condition: [list(pair) for pair in values] for condition, values in role.SPECIAL.items()},
        "condition_audits": audits,
        "interface_errors": interface_errors,
        "balance_diagnostics": {
            "special_sum_values": {condition: sorted(a + b for a, b in values) for condition, values in role.SPECIAL.items()},
            "special_second_operand_counts": {condition: dict(sorted(Counter(b for _, b in values).items())) for condition, values in role.SPECIAL.items()},
            "special_first_operand_counts": {condition: dict(sorted(Counter(a for a, _ in values).items())) for condition, values in role.SPECIAL.items()},
            "unavoidable_first_operand_difference": Counter(a for a, _ in diagonal) != Counter(a for a, _ in role_swapped),
        },
        "interfaces": list(role.INTERFACES),
        "seeds": list(role.SEEDS),
        "steps": role.STEPS,
        "architecture_changed": False,
        "sealed_test_accessed": False,
        "errors": errors,
        "pass": not errors,
    }


role.preflight = custom_preflight


if __name__ == "__main__":
    raise SystemExit(role.main())
