#!/usr/bin/env python3
"""Evaluate completed S4 D2 readout arms on the unseen-layout control."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_crucible_d2_readout_competence_ladder_20261026.py"


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


runner = import_module("d2_readout_ladder_runner_for_unseen_eval", RUNNER)
torch = runner.torch


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    output = runner.OUTPUT
    for interface in runner.INTERFACES:
        records = load_jsonl(output / "data" / interface / "S3_UNSEEN_LAYOUT" / "validation.jsonl")
        for seed in runner.SEEDS:
            arm = output / "trained_arms" / "S4_END_TO_END_AUTHORITY" / interface / f"seed{seed}"
            checkpoint = arm / "checkpoint.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = runner.DecisionPointerParser(seed)
            model.load_state_dict(state["model"], strict=True)
            model.eval()
            prepared = runner.prepare_records(records)
            predictions: List[Dict[str, Any]] = []
            for start in range(0, len(prepared), runner.READOUT_BATCH):
                indices = list(range(start, min(start + runner.READOUT_BATCH, len(prepared))))
                batch = runner.batchify(prepared, indices)
                with torch.inference_mode():
                    logits = model.decision_logits(batch)
                predictions.extend(runner.pointer_rows([records[index] for index in indices], logits, "S4_END_TO_END_AUTHORITY", seed, interface, "end_to_end_unseen_layout"))
            result = runner.load_json(arm / "RESULTS.json")
            result["unseen_layout_metrics"] = runner.row_metrics(predictions, "validation_unseen_layout")
            runner.write_json(arm / "RESULTS.json", result)
            existing = [row for row in load_csv(arm / "PREDICTIONS.csv") if row.get("source") != "end_to_end_unseen_layout"]
            runner.write_csv(arm / "PREDICTIONS.csv", existing + predictions)
            print(json.dumps({"interface": interface, "seed": seed, "metrics": result["unseen_layout_metrics"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
