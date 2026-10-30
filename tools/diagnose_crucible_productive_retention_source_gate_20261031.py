#!/usr/bin/env python3
"""CPU-only diagnostic of source-task identifiability and source-gate fit."""

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
THROUGHPUT = ROOT / "experiments" / "crucible_heterogeneous_training_throughput_20261029"
sys.path.insert(0, str(THROUGHPUT))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark import make_model  # noqa: E402
from run_crucible_foundation_training_20261029 import foundation_model_config  # noqa: E402
from run_crucible_productive_retention_source_gate_20261031 import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_MANIFEST,
    CONTEXT,
    BOS_ID,
    RAW_BYTE_START,
    format_prompt,
)


SCHEMA = "crucible-productive-retention-source-gate-diagnostic-v1-20261031"
OPS = ("and", "or", "xor", "not")
DEFAULT_OUTPUT = ROOT / "experiments" / "crucible_productive_retention_transfer_20261031" / "source_gate_diagnostic"
SOURCE_GATE_REPORT_ROOT = ROOT / "experiments" / "crucible_productive_retention_transfer_20261031" / "source_gate"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def combine(a: Tuple[int, ...], b: Tuple[int, ...], op: str) -> Tuple[int, ...]:
    if op == "and":
        return tuple(x & y for x, y in zip(a, b))
    if op == "or":
        return tuple(x | y for x, y in zip(a, b))
    if op == "xor":
        return tuple(x ^ y for x, y in zip(a, b))
    raise ValueError(op)


def enumerate_source_functions() -> List[Tuple[int, ...]]:
    vectors = [[(mask >> bit) & 1 for bit in range(4)] for mask in range(16)]
    by_depth: List[set[Tuple[int, ...]]] = [{tuple(vector[index] for vector in vectors) for index in range(4)}]
    for _depth in range(1, 3):
        previous = by_depth[-1]
        current = {tuple(1 - value for value in table) for table in previous}
        for left in previous:
            for right in previous:
                for op in OPS:
                    if op != "not":
                        current.add(combine(left, right, op))
        by_depth.append(current)
    return sorted(set().union(*by_depth))


def row_index(values: Sequence[int]) -> int:
    return sum(int(value) << index for index, value in enumerate(values))


def identifiability(row: Mapping[str, Any], functions: Sequence[Tuple[int, ...]]) -> Dict[str, Any]:
    compatible = [
        table
        for table in functions
        if all(table[row_index(example["input"])] == int(example["output"]) for example in row["demonstrations"])
    ]
    query_index = row_index(row["query"])
    query_values = sorted({table[query_index] for table in compatible})
    expected = int(row["query_output"])
    return {
        "task_id": row["task_id"],
        "compatible_function_count": len(compatible),
        "compatible_query_values": query_values,
        "query_identifiable": len(query_values) == 1,
        "expected_query_output": expected,
        "expected_is_compatible": expected in query_values,
        "demonstration_count": len(row["demonstrations"]),
        "demonstration_input_count": len({tuple(example["input"]) for example in row["demonstrations"]}),
        "query_is_new_input": tuple(row["query"]) not in {tuple(example["input"]) for example in row["demonstrations"]},
    }


def task_statistics(rows: Sequence[Mapping[str, Any]], functions: Sequence[Tuple[int, ...]]) -> Dict[str, Any]:
    details = [identifiability(row, functions) for row in rows]
    labels = [int(row["query_output"]) for row in rows]
    counts = {str(value): labels.count(value) for value in (0, 1)}
    majority = max(counts.values()) if counts else 0
    return {
        "rows": len(rows),
        "query_label_counts": counts,
        "majority_constant_accuracy": majority / max(1, len(rows)),
        "all_zero_accuracy": counts.get("0", 0) / max(1, len(rows)),
        "all_one_accuracy": counts.get("1", 0) / max(1, len(rows)),
        "identifiable_rows": sum(int(item["query_identifiable"]) for item in details),
        "ambiguous_rows": sum(int(not item["query_identifiable"]) for item in details),
        "no_compatible_function_rows": sum(int(item["compatible_function_count"] == 0) for item in details),
        "expected_compatible_rows": sum(int(item["expected_is_compatible"]) for item in details),
        "all_queries_new_inputs": all(bool(item["query_is_new_input"]) for item in details),
        "details": details,
    }


def prompt_batch(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    prompts = []
    for row in rows:
        tokens = [BOS_ID] + [RAW_BYTE_START + value for value in format_prompt(row).encode("utf-8")]
        if len(tokens) >= CONTEXT:
            raise RuntimeError(f"prompt exceeds context: {row['task_id']}")
        prompts.append(tokens)
    lengths = {len(tokens) for tokens in prompts}
    if len(lengths) != 1:
        raise RuntimeError("source prompts unexpectedly have different lengths")
    return torch.tensor(prompts, dtype=torch.long)


@torch.no_grad()
def model_accuracy(model: torch.nn.Module, rows: Sequence[Mapping[str, Any]], batch_size: int) -> Dict[str, Any]:
    model.eval()
    details: List[Dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        x = prompt_batch(batch)
        logits = model(x)
        predictions = torch.argmax(logits[:, -1, :], dim=-1).tolist()
        for row, token in zip(batch, predictions):
            prediction = chr(token - RAW_BYTE_START) if token in {RAW_BYTE_START + ord("0"), RAW_BYTE_START + ord("1")} else ""
            expected = str(int(row["query_output"]))
            details.append({"task_id": row["task_id"], "prediction": prediction, "expected": expected, "correct": prediction == expected})
    correct = sum(int(item["correct"]) for item in details)
    return {"correct": correct, "rows": len(details), "accuracy": correct / max(1, len(details)), "details": details}


def checkpoint_accuracy(checkpoint_path: Path, train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_model(foundation_model_config("120m", False), torch.device("cpu"), int(checkpoint["seed"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    result = {
        "checkpoint": f"<source-gate-seed{checkpoint['seed']}-update{checkpoint['update']}>",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "seed": checkpoint["seed"],
        "update": checkpoint["update"],
        "execution": "cpu_fp32_replay",
        "training": model_accuracy(model, train_rows, 8),
        "validation": model_accuracy(model, validation_rows, 8),
    }
    del model, checkpoint
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest_dir = args.manifest_dir.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    output_dir = args.output_dir.resolve()
    functions = enumerate_source_functions()
    if len(functions) != 429:
        raise RuntimeError(f"source grammar enumeration expected 429 functions, got {len(functions)}")

    by_seed: Dict[str, Any] = {}
    started = time.perf_counter()
    for seed in (42, 1337):
        train_rows = read_jsonl(manifest_dir / f"seed_{seed}" / "source_train.jsonl")
        validation_rows = read_jsonl(manifest_dir / f"seed_{seed}" / "source_validation.jsonl")
        train_stats = task_statistics(train_rows, functions)
        validation_stats = task_statistics(validation_rows, functions)
        final_checkpoint = checkpoint_root / f"crucible_120m_source_gate_seed{seed}_update_001000.pt"
        recorded_report = read_json(SOURCE_GATE_REPORT_ROOT / f"seed_{seed}" / "SOURCE_GATE_REPORT.json")
        by_seed[str(seed)] = {
            "train": train_stats,
            "validation": validation_stats,
            "recorded_gpu_source_gate": recorded_report["source_gate"]["final"],
            "model_fit": checkpoint_accuracy(final_checkpoint, train_rows, validation_rows),
        }

    report = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "classification": "cpu_read_only_source_identifiability_and_fit_diagnostic",
        "scientific_authorization": False,
        "training_started": False,
        "target_data_loaded": False,
        "sealed_answers_loaded": False,
        "grammar": {"operators": list(OPS), "max_depth": 2, "distinct_truth_tables": len(functions)},
        "manifest_sha256": sha256_file(manifest_dir / "MANIFEST.json"),
        "seeds": by_seed,
        "summary": {
            "validation_rows": 32,
            "validation_ambiguous_rows": sum(by_seed[str(seed)]["validation"]["ambiguous_rows"] for seed in (42, 1337)),
            "validation_identifiable_rows": sum(by_seed[str(seed)]["validation"]["identifiable_rows"] for seed in (42, 1337)),
            "validation_majority_baseline_accuracy_mean": sum(by_seed[str(seed)]["validation"]["majority_constant_accuracy"] for seed in (42, 1337)) / 2,
            "training_validation_accuracy": {
                str(seed): {
                    "train": by_seed[str(seed)]["model_fit"]["training"]["accuracy"],
                    "validation": by_seed[str(seed)]["model_fit"]["validation"]["accuracy"],
                }
                for seed in (42, 1337)
            },
            "recorded_gpu_source_gate_accuracy": {
                str(seed): by_seed[str(seed)]["recorded_gpu_source_gate"]["accuracy"]
                for seed in (42, 1337)
            },
        },
        "interpretation": [
            "This diagnostic does not retrain the model and does not open target answers.",
            "Ambiguous episodes are not logically identifiable from the supplied demonstrations alone; probabilistic generalization may still be possible, but exact competence claims must report this limitation.",
            "The saved source-gate checkpoints show fit and validation behavior under the tested prompt/optimization configuration, not a causal diagnosis of failure.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "SOURCE_GATE_DIAGNOSTIC.json", report)
    (output_dir / "SOURCE_GATE_DIAGNOSTIC.md").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "grammar_functions": len(functions), "validation_ambiguous_rows": report["summary"]["validation_ambiguous_rows"], "training_started": False, "sealed_answers_loaded": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
