#!/usr/bin/env python3
"""Run frozen-inference source competence diagnostics on the repaired curriculum."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
REPAIR_ROOT = ROOT / "experiments" / "crucible_source_curriculum_repair_20261031"
ORACLE_ROOT = Path("/home/timbushnell/crucible_source_curriculum_repair_oracles_20261031")
CHECKPOINT = Path("/home/timbushnell/crucible_120m_checkpoints_20261029/crucible_120m_foundation_step_015000.pt")
OUTPUT = ROOT / "experiments" / "crucible_source_competence_frozen_diagnostic_20261031"
THROUGHPUT = ROOT / "experiments" / "crucible_heterogeneous_training_throughput_20261029"
sys.path.insert(0, str(THROUGHPUT))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark import make_model  # noqa: E402
from run_crucible_foundation_training_20261029 import foundation_model_config  # noqa: E402
from run_crucible_source_curriculum_repair_20261031 import (  # noqa: E402
    ONE_TOKEN,
    ZERO_TOKEN,
    decode_one_step,
    read_jsonl,
    sha256_file,
)


SCHEMA = "crucible-source-competence-frozen-diagnostic-v1-20261031"
SEEDS = (42, 1337)
SPLITS = (
    ("source_validation", "query_identifiable_induction"),
    ("source_function_validation", "function_identifiable_induction"),
    ("execution_validation", "explicit_execution"),
    ("source_test", "query_identifiable_induction"),
    ("execution_test", "explicit_execution"),
)
PAD_ID = 0
BOS_ID = 1
RAW_BYTE_START = 3
CONTEXT = 512


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def format_induction_prompt(row: Mapping[str, Any]) -> str:
    lines = ["BOOLEAN_FUNCTION\n", "DEMONSTRATIONS:\n"]
    for example in row["demonstrations"]:
        values = ",".join(str(int(value)) for value in example["input"])
        lines.append(f"INPUT={values};OUTPUT={int(example['output'])}\n")
    query = ",".join(str(int(value)) for value in row["query"])
    lines.append(f"QUERY={query};ANSWER=")
    return "".join(lines)


def format_execution_prompt(row: Mapping[str, Any]) -> str:
    query = ",".join(str(int(value)) for value in row["query"])
    procedure = canonical(row["procedure"])
    return f"BOOLEAN_PROCEDURE\nEXPRESSION={procedure}\nQUERY={query};ANSWER="


def prompt_tokens(row: Mapping[str, Any]) -> List[int]:
    prompt = format_induction_prompt(row) if row["family"] == "source_induction" else format_execution_prompt(row)
    encoded = prompt.encode("utf-8")
    if len(encoded) + 1 > CONTEXT:
        raise RuntimeError(f"prompt exceeds context: {row['task_id']}")
    return [BOS_ID] + [RAW_BYTE_START + value for value in encoded]


def token_repr(token: int) -> str:
    if token == 0:
        return "PAD"
    if token == BOS_ID:
        return "BOS"
    if token == 2:
        return "EOS"
    if RAW_BYTE_START <= token < RAW_BYTE_START + 256:
        value = token - RAW_BYTE_START
        return repr(chr(value))
    return f"token:{token}"


def load_rows(seed: int, split: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    public = read_jsonl(REPAIR_ROOT / f"seed_{seed}" / f"{split}.jsonl")
    oracle = read_jsonl(ORACLE_ROOT / f"seed_{seed}" / f"{split}.jsonl")
    if len(public) != len(oracle):
        raise RuntimeError(f"public/oracle count mismatch: seed={seed} split={split}")
    return public, oracle


def evaluate_rows(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    oracles: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        batch_oracles = oracles[start : start + batch_size]
        encoded = [prompt_tokens(row) for row in batch]
        max_length = max(len(item) for item in encoded)
        if max_length > CONTEXT:
            raise RuntimeError("batch contains a prompt beyond context")
        inputs = torch.full((len(encoded), max_length), PAD_ID, dtype=torch.long, device=device)
        for index, tokens in enumerate(encoded):
            inputs[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(inputs)
            if not bool(torch.isfinite(logits).all().item()):
                raise RuntimeError("non-finite frozen-inference logits")
        for index, (row, oracle, tokens) in enumerate(zip(batch, batch_oracles, encoded)):
            answer_position = len(tokens) - 1
            vector = logits[index, answer_position].float().cpu().tolist()
            prediction = decode_one_step(vector, ZERO_TOKEN, ONE_TOKEN)
            expected = str(int(oracle["query_output"]))
            zero_logit = float(vector[ZERO_TOKEN])
            one_logit = float(vector[ONE_TOKEN])
            top_token = max(range(len(vector)), key=lambda token: (float(vector[token]), -token))
            ranked = sorted(range(len(vector)), key=lambda token: float(vector[token]), reverse=True)
            forced_prediction = "" if zero_logit == one_logit else ("0" if zero_logit > one_logit else "1")
            answered = prediction in {"0", "1"}
            correct = answered and prediction == expected
            results.append(
                {
                    "task_id": row["task_id"],
                    "prediction": prediction,
                    "expected": expected,
                    "answered": answered,
                    "correct": correct,
                    "classification": "correct" if correct else ("abstain" if not answered else "wrong"),
                    "zero_logit": zero_logit,
                    "one_logit": one_logit,
                    "margin_zero_minus_one": zero_logit - one_logit,
                    "forced_binary_prediction": forced_prediction,
                    "forced_binary_correct": forced_prediction == expected,
                    "top_token": top_token,
                    "top_token_repr": token_repr(top_token),
                    "zero_rank": ranked.index(ZERO_TOKEN) + 1,
                    "one_rank": ranked.index(ONE_TOKEN) + 1,
                    "answer_position": answer_position,
                    "prompt_tokens": len(tokens),
                }
            )
    eligible = len(results)
    answered = sum(int(item["answered"]) for item in results)
    correct = sum(int(item["correct"]) for item in results)
    forced_correct = sum(int(item["forced_binary_correct"]) for item in results)
    forced_answered = sum(int(item["forced_binary_prediction"] in {"0", "1"}) for item in results)
    top_tokens: Dict[str, int] = {}
    for item in results:
        key = f"{item['top_token']}:{item['top_token_repr']}"
        top_tokens[key] = top_tokens.get(key, 0) + 1
    return {
        "eligible": eligible,
        "answered": answered,
        "correct": correct,
        "wrong": answered - correct,
        "abstained": eligible - answered,
        "exact_accuracy": correct / max(1, eligible),
        "selective_accuracy": correct / max(1, answered),
        "coverage": answered / max(1, eligible),
        "abstention_rate": (eligible - answered) / max(1, eligible),
        "auxiliary_forced_binary_readout": {
            "purpose": "diagnose answer-format abstention; not the registered primary gate",
            "correct": forced_correct,
            "eligible": eligible,
            "answered": forced_answered,
            "exact_accuracy": forced_correct / max(1, eligible),
            "coverage": forced_answered / max(1, eligible),
        },
        "top_token_histogram": dict(sorted(top_tokens.items(), key=lambda item: (-item[1], item[0]))),
        "rows": results,
    }


def positive_decoder_control() -> Dict[str, Any]:
    vocab = max(ONE_TOKEN + 1, 300)
    rows = []
    for expected in ("0", "1") * 8:
        vector = [0.0] * vocab
        vector[ZERO_TOKEN if expected == "0" else ONE_TOKEN] = 4.0
        prediction = decode_one_step(vector, ZERO_TOKEN, ONE_TOKEN)
        rows.append({"expected": expected, "prediction": prediction, "correct": prediction == expected})
    return {"rows": rows, "correct": sum(int(item["correct"]) for item in rows), "total": len(rows), "passed": all(item["correct"] for item in rows)}


def run(checkpoint: Path, device: torch.device, batch_size: int) -> Dict[str, Any]:
    output = OUTPUT.resolve()
    if output.exists() and any(path.name != "DIAGNOSTIC_PROTOCOL.json" for path in output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(REPAIR_ROOT / "MANIFEST.json")
    hashes = read_json(REPAIR_ROOT / "HASHES.json")
    if sha256_file(REPAIR_ROOT / "MANIFEST.json") != hashes["manifest_sha256"]:
        raise RuntimeError("curriculum manifest hash mismatch")
    if not checkpoint.exists():
        raise RuntimeError(f"missing frozen checkpoint: {checkpoint}")

    started = time.perf_counter()
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_seed = int(checkpoint_payload.get("seed", 42))
    checkpoint_step = int(checkpoint_payload.get("step", 15000))
    cfg = foundation_model_config("120m", False)
    model = make_model(cfg, device, checkpoint_seed)
    model.load_state_dict(checkpoint_payload["model"], strict=True)
    del checkpoint_payload
    gc.collect()
    model.eval()

    reports: Dict[str, Any] = {}
    protected_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for split, category in SPLITS:
            public, oracle = load_rows(seed, split)
            result = evaluate_rows(model, public, oracle, device, batch_size)
            key = f"seed_{seed}/{split}"
            reports[key] = {"seed": seed, "split": split, "category": category, **result}
            if split == "source_test":
                protected_rows.extend(
                    [
                        {"seed": seed, "task_id": item["task_id"], "initial_correct": bool(item["correct"])}
                        for item in result["rows"]
                    ]
                )

    report = {
        "schema_version": SCHEMA,
        "classification": "frozen_inference_diagnostic_only",
        "status": "completed",
        "scientific_authorization": True,
        "training_started": False,
        "source_training_updates": 0,
        "target_data_loaded": False,
        "sealed_target_answers_loaded": False,
        "checkpoint": {
            "path": "<local-path-redacted>",
            "sha256": sha256_file(checkpoint),
            "step": checkpoint_step,
            "checkpoint_seed": checkpoint_seed,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "curriculum_manifest_sha256": sha256_file(REPAIR_ROOT / "MANIFEST.json"),
        "device": str(device),
        "precision": "bf16_autocast" if device.type == "cuda" else "fp32",
        "batch_size": batch_size,
        "elapsed_seconds": time.perf_counter() - started,
        "positive_decoder_control": positive_decoder_control(),
        "reports": reports,
        "protected_initial_results": {
            "source": "source_test",
            "rows": protected_rows,
            "correct_instances": sum(int(item["initial_correct"]) for item in protected_rows),
            "transition_counts": "not_applicable_frozen_inference_only",
            "promotion_eligibility": "not_applicable",
        },
        "interpretation_boundary": [
            "This run measures a frozen checkpoint only; it does not establish learning.",
            "No source-task updates were performed, so retention transitions are not applicable.",
            "The original two-arm dense-baseline comparison remains blocked.",
            "C0-C3 productive-retention training remains blocked.",
        ],
    }
    (output / "FROZEN_DIAGNOSTIC_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Crucible frozen source-competence diagnostic",
        "",
        "Status: **completed — frozen inference only**",
        "",
        f"Checkpoint step: {checkpoint_step}",
        f"Device: {device}",
        f"Precision: {report['precision']}",
        f"Elapsed seconds: {report['elapsed_seconds']:.2f}",
        "",
        "| Seed/split | Category | Primary exact | Primary coverage | Forced-binary exact |",
        "|---|---|---:|---:|---:|",
    ]
    for key, result in sorted(reports.items()):
        forced = result["auxiliary_forced_binary_readout"]
        lines.append(
            f"| {key} | {result['category']} | {result['correct']}/{result['eligible']} | "
            f"{result['coverage']:.3f} | {forced['correct']}/{forced['eligible']} |"
        )
    lines.extend([
        "",
        "The full per-task results are in `FROZEN_DIAGNOSTIC_REPORT.json`.",
        "No source training, target evaluation, productive-retention claim or architectural comparison was performed.",
        "",
        "## Positive decoder control",
        "",
        json.dumps(report["positive_decoder_control"], indent=2, sort_keys=True),
    ])
    (output / "FROZEN_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    report = run(args.checkpoint.resolve(), device, args.batch_size)
    print(json.dumps({"status": report["status"], "classification": report["classification"], "device": report["device"], "training_started": report["training_started"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
