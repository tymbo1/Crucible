#!/usr/bin/env python3
"""CPU-only audit and tiny-set overfit control for composition learnability.

This diagnostic independently checks the generated composition oracle and byte
serialization, then trains a fresh dense control on eight fixed composition
examples.  It does not evaluate the held-out test split and does not compare
against E4.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
LADDER_PATH = ROOT / "tools" / "run_crucible_cognitive_learnability_ladder_20261022.py"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_cognitive_composition_diagnostic_20261022_seed42"
RAW_BYTE_START = 3
EOS_ID = 2
STEPS = 500
TINY_COUNT = 8
PROMPT_RE = re.compile(
    r"Task: arithmetic is modulo ten\. Apply both operations left-to-right\. "
    r"start=(?P<start>[0-9]); step1=(?P<op1>add|sub|mul)\((?P<arg1>[1-4])\); "
    r"step2=(?P<op2>add|sub|mul)\((?P<arg2>[1-4])\)\. Return one digit only\."
)

spec = importlib.util.spec_from_file_location("cognitive_ladder", LADDER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import ladder runner: {LADDER_PATH}")
ladder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ladder
spec.loader.exec_module(ladder)
stage_a = ladder.stage_a


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_mod(value: int, operation: str, operand: int) -> int:
    if operation == "add":
        return (value + operand) % 10
    if operation == "sub":
        return (value - operand) % 10
    if operation == "mul":
        return (value * operand) % 10
    raise ValueError(operation)


def independent_oracle(record: Mapping[str, Any]) -> Dict[str, Any]:
    match = PROMPT_RE.fullmatch(str(record["input"]))
    if match is None:
        raise ValueError(f"prompt does not match independent grammar: {record['record_id']}")
    start = int(match.group("start"))
    op1, arg1 = match.group("op1"), int(match.group("arg1"))
    op2, arg2 = match.group("op2"), int(match.group("arg2"))
    intermediate = apply_mod(start, op1, arg1)
    expected = apply_mod(intermediate, op2, arg2)
    return {
        "start": start,
        "operations": [[op1, arg1], [op2, arg2]],
        "intermediate": intermediate,
        "expected": expected,
    }


def audit_records(records_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    errors: List[str] = []
    split_summary: Dict[str, Any] = {}
    seen_ids = set()
    for split, records in records_by_split.items():
        targets = []
        serialized_target_tokens = []
        for record in records:
            record_id = str(record["record_id"])
            if record_id in seen_ids:
                errors.append(f"duplicate record id: {record_id}")
            seen_ids.add(record_id)
            try:
                oracle = independent_oracle(record)
            except Exception as exc:  # pragma: no cover - diagnostic report path
                errors.append(f"{record_id}: {exc}")
                continue
            expected = str(oracle["expected"])
            if str(record["target"]) != expected:
                errors.append(f"{record_id}: target {record['target']!r} != independent {expected!r}")
            metadata_ops = list(record.get("metadata", {}).get("composition", []))
            if metadata_ops != [pair[0] for pair in oracle["operations"]]:
                errors.append(f"{record_id}: metadata operation order mismatch")
            prepared = ladder.prepare_records([record])[0]
            full = prepared["_full"]
            prefix_len = int(prepared["_prefix_len"])
            expected_target_token = RAW_BYTE_START + ord(expected)
            if len(full) != prefix_len + 2:
                errors.append(f"{record_id}: expected one target byte plus EOS")
            elif full[prefix_len] != expected_target_token or full[prefix_len + 1] != EOS_ID:
                errors.append(f"{record_id}: serialized target alignment mismatch")
            targets.append(int(expected))
            serialized_target_tokens.append(full[prefix_len] if len(full) > prefix_len else None)
        split_summary[split] = {
            "records": len(records),
            "unique_record_ids": len({str(record["record_id"]) for record in records}),
            "unique_inputs": len({str(record["input"]) for record in records}),
            "target_values": sorted(set(targets)),
            "serialized_target_tokens": sorted(set(token for token in serialized_target_tokens if token is not None)),
        }
    return {"errors": errors, "error_count": len(errors), "splits": split_summary}


@stage_a.torch.no_grad()
def prediction_diagnostics(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    torch = stage_a.torch
    model.eval()
    batch = stage_a.batchify(records, list(range(len(records))))
    logits = stage_a.forward_model(model, batch)
    mask = batch["target_mask"] & (batch["y"] != stage_a.PAD_ID)
    # The shared training mask includes the answer byte and EOS.  For this
    # diagnostic, report answer-byte confidence separately from the framing
    # token so the output distribution is interpretable.
    answer_mask = mask & (batch["y"] != EOS_ID)
    target_logits = logits[answer_mask, batch["y"][answer_mask]]
    probabilities = torch.softmax(logits[answer_mask], dim=-1)
    target_probabilities = probabilities.gather(1, batch["y"][answer_mask].unsqueeze(1)).squeeze(1)
    ranks = 1 + (logits[answer_mask] > target_logits.unsqueeze(1)).sum(dim=1)
    predicted = logits[answer_mask].argmax(dim=-1).tolist()
    digit_start = RAW_BYTE_START + ord("0")
    digit_end = RAW_BYTE_START + ord("9")
    predicted_digits = [token - digit_start for token in predicted if digit_start <= token <= digit_end]
    return {
        "supervised_positions_including_eos": int(mask.sum().item()),
        "answer_positions": int(answer_mask.sum().item()),
        "mean_target_logit": float(target_logits.mean().item()),
        "mean_target_probability": float(target_probabilities.mean().item()),
        "mean_target_rank": float(ranks.float().mean().item()),
        "predicted_digit_histogram": {str(digit): predicted_digits.count(digit) for digit in range(10)},
    }


def run_tiny_overfit(seed: int, records: Sequence[Mapping[str, Any]], output: Path) -> Dict[str, Any]:
    torch = stage_a.torch
    tiny = [dict(record) for record in records[:TINY_COUNT]]
    prepared = ladder.prepare_records(tiny)
    order = list(range(len(prepared)))
    random.Random(seed + 61_022).shuffle(order)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(checkpoint_dir / "step_0000.pt", model, optimizer, 0, order)
    evaluation_steps = {0, 1, 10, 25, 50, 100, 200, 300, STEPS}
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(STEPS + 1):
        train_info = None
        if step > 0:
            indices = stage_a.batch_indices(order, step)
            train_info = stage_a.train_step(model, optimizer, prepared, indices)
        if step in evaluation_steps:
            evaluation = stage_a.evaluate(model, prepared)
            row = {
                "step": step,
                "train_step": train_info,
                "evaluation": evaluation,
                "prediction_diagnostics": prediction_diagnostics(model, prepared),
            }
            rows.append(row)
            print(json.dumps({"seed": seed, "step": step, "exact": evaluation["teacher_forced_exact_accuracy"], "loss": evaluation["mean_target_loss"]}, sort_keys=True), flush=True)
    stage_a.save_checkpoint(checkpoint_dir / f"step_{STEPS:04d}.pt", model, optimizer, STEPS, order)
    final = rows[-1]["evaluation"]
    return {
        "seed": seed,
        "steps": STEPS,
        "examples": len(prepared),
        "elapsed_seconds": time.perf_counter() - started,
        "final": final,
        "final_prediction_diagnostics": rows[-1]["prediction_diagnostics"],
        "records": rows,
        "checkpoint_paths": ["checkpoints/step_0000.pt", f"checkpoints/step_{STEPS:04d}.pt"],
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    all_records = ladder.generate_ladder(args.seed)["composition"]
    audit = audit_records(all_records)
    write_json(output / "AUDIT.json", audit)
    tiny = [dict(record) for record in all_records["train"][:TINY_COUNT]]
    (output / "tiny_train.jsonl").write_text("\n".join(json.dumps(record, sort_keys=True) for record in tiny) + "\n", encoding="utf-8")
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_cognitive_composition_diagnostic_20261022",
        "status": "cpu_only_dense_control",
        "seed": args.seed,
        "steps": STEPS,
        "tiny_examples": TINY_COUNT,
        "test_evaluated": False,
        "test_model_evaluation": "sealed; generator audit only",
        "acceptance": "final tiny-set teacher-forced exact accuracy >= 0.99",
    })
    if audit["error_count"]:
        write_json(output / "RESULTS.json", {
            "status": "blocked_by_audit",
            "seed": args.seed,
            "test_evaluated": False,
            "audit": audit,
        })
        return 2

    result = run_tiny_overfit(args.seed, tiny, output)
    result["overfit_pass"] = result["final"]["teacher_forced_exact_accuracy"] >= 0.99
    final = {
        "experiment": "crucible_cognitive_composition_diagnostic_20261022",
        "status": "completed",
        "seed": args.seed,
        "test_evaluated": False,
        "audit_pass": audit["error_count"] == 0,
        "overfit_pass": result["overfit_pass"],
        "audit": audit,
        "tiny_overfit": result,
        "interpretation": "composition interface and positive-control diagnostic only; no E4 comparison or cognitive-capability claim",
    }
    write_json(output / "RESULTS.json", final)
    (output / "RUNS.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in result["records"]) + "\n", encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"output": str(output), "audit_pass": final["audit_pass"], "overfit_pass": final["overfit_pass"], "status": "completed"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
