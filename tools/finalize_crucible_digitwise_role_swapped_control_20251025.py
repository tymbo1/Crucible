#!/usr/bin/env python3
"""Finalize the role-swapped control using the shared read-only diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "finalize_crucible_digitwise_diagonal_ablation_20251025.py"
OUTPUT = ROOT / "experiments" / "crucible_digitwise_role_swapped_control_20251025"

spec = importlib.util.spec_from_file_location("role_swapped_finalizer", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUTPUT = OUTPUT

if __name__ == "__main__":
    raise SystemExit(module.main())
