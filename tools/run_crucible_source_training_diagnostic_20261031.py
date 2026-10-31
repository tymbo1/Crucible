#!/usr/bin/env python3
"""Preflight and, only when explicitly authorized, run the bounded source diagnostic.

This runner is intentionally separate from the frozen-inference diagnostic and
the old productive-retention source gate.  It trains only on the repaired
public source splits, uses the fixed answer-position decoder for both loss and
evaluation, and never opens target data.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
THROUGHPUT = ROOT / "experiments" / "crucible_heterogeneous_training_throughput_20261029"
import sys

sys.path.insert(0, str(THROUGHPUT))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark import make_model, parameter_groups, state_digest  # noqa: E402
from run_crucible_foundation_training_20261029 import foundation_model_config  # noqa: E402


SCHEMA = "crucible-source-training-diagnostic-v1-20261031"
PROTOCOL = ROOT / "experiments" / "crucible_source_training_diagnostic_20261031" / "DIAGNOSTIC_PROTOCOL.json"
AUTHORIZATION = ROOT / "experiments" / "crucible_source_training_diagnostic_20261031" / "AUTHORIZATION.json"
REPAIR_ROOT = ROOT / "experiments" / "crucible_source_curriculum_repair_20261031"
ORACLE_ROOT = Path("/home/timbushnell/crucible_source_curriculum_repair_oracles_20261031")
CHECKPOINT = Path("/home/timbushnell/crucible_120m_checkpoints_20261029/crucible_120m_foundation_step_015000.pt")
OUTPUT = ROOT / "experiments" / "crucible_source_training_diagnostic_20261031"
CHECKPOINT_ROOT = Path("/home/timbushnell/crucible_source_training_diagnostic_checkpoints_20261031")

SEEDS = (42, 1337)
CONTEXT = 512
PAD_ID = 0
BOS_ID = 1
RAW_BYTE_START = 3
ZERO_TOKEN = RAW_BYTE_START + ord("0")
ONE_TOKEN = RAW_BYTE_START + ord("1")
MICROBATCH_SIZE = 4
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.95)
EPS = 1e-8
GRAD_CLIP = 1.0

PHASES = {
    "execution": {
        "training_split": "execution_train",
        "updates": 500,
        "checkpoints": (0, 50, 100, 250, 500),
        "seed_offset": 1701,
    },
    "induction": {
        "training_split": "source_train",
        "updates": 1000,
        "checkpoints": (0, 100, 250, 500, 750, 1000),
        "seed_offset": 2701,
    },
}

EVAL_SPLITS = (
    "execution_train",
    "execution_validation",
    "execution_test",
    "source_train",
    "source_validation",
    "source_function_validation",
    "source_test",
)

CATEGORIES = {
    "execution": "explicit_procedure_execution",
    "source": "query_identifiable_induction",
    "source_function": "function_identifiable_induction",
}

THRESHOLDS = {
    "explicit_procedure_execution": {"min_correct": 14, "min_coverage": 15},
    "query_identifiable_induction": {"min_correct": 26, "min_coverage": 29},
    "function_identifiable_induction": {"min_correct": 13, "min_coverage": 14},
}


class DiagnosticError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return not (torch.is_floating_point(value) or torch.is_complex(value)) or bool(torch.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def finite_model(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter.detach()).all().item()) for parameter in model.parameters())


def finite_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    return finite_tree(optimizer.state_dict())


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
    return f"BOOLEAN_PROCEDURE\nEXPRESSION={canonical(row['procedure'])}\nQUERY={query};ANSWER="


def prompt_tokens(row: Mapping[str, Any]) -> List[int]:
    prompt = format_induction_prompt(row) if row["family"] == "source_induction" else format_execution_prompt(row)
    encoded = prompt.encode("utf-8")
    if len(encoded) + 1 > CONTEXT:
        raise DiagnosticError(f"prompt exceeds context: {row['task_id']}")
    return [BOS_ID] + [RAW_BYTE_START + value for value in encoded]


def answer_token(value: Any) -> int:
    text = str(int(value))
    if text not in {"0", "1"}:
        raise DiagnosticError(f"invalid binary answer: {value}")
    return ZERO_TOKEN if text == "0" else ONE_TOKEN


def build_batch(rows: Sequence[Mapping[str, Any]], values: Sequence[Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [prompt_tokens(row) for row in rows]
    width = max(len(item) for item in encoded)
    inputs = torch.full((len(encoded), width), PAD_ID, dtype=torch.long, device=device)
    positions = []
    for index, tokens in enumerate(encoded):
        inputs[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        positions.append(len(tokens) - 1)
    targets = torch.tensor([answer_token(value) for value in values], dtype=torch.long, device=device)
    return inputs, torch.tensor(positions, dtype=torch.long, device=device), targets


def decode_one_step(vector: Sequence[float]) -> Tuple[str, Dict[str, Any]]:
    if not vector or not all(math.isfinite(float(value)) for value in vector):
        raise DiagnosticError("non-finite or empty answer logits")
    ranked = sorted(range(len(vector)), key=lambda index: (float(vector[index]), -index), reverse=True)
    top = ranked[0]
    zero = float(vector[ZERO_TOKEN])
    one = float(vector[ONE_TOKEN])
    if zero == one:
        prediction = ""
    elif top == ZERO_TOKEN:
        prediction = "0"
    elif top == ONE_TOKEN:
        prediction = "1"
    else:
        prediction = ""
    forced = "" if zero == one else ("0" if zero > one else "1")
    return prediction, {
        "top_token": int(top),
        "zero_rank": ranked.index(ZERO_TOKEN) + 1,
        "one_rank": ranked.index(ONE_TOKEN) + 1,
        "zero_logit": zero,
        "one_logit": one,
        "margin_zero_minus_one": zero - one,
        "forced_binary_prediction": forced,
    }


def decoder_fixture() -> Dict[str, Any]:
    vector = [0.0] * 300
    vector[ZERO_TOKEN] = 3.0
    zero, _ = decode_one_step(vector)
    vector[ZERO_TOKEN] = 0.0
    vector[ONE_TOKEN] = 3.0
    one, _ = decode_one_step(vector)
    vector[ONE_TOKEN] = 0.0
    vector[RAW_BYTE_START + ord("x")] = 3.0
    other, _ = decode_one_step(vector)
    vector[RAW_BYTE_START + ord("x")] = 0.0
    vector[ZERO_TOKEN] = 3.0
    vector[ONE_TOKEN] = 3.0
    tie, _ = decode_one_step(vector)
    return {
        "zero": zero == "0",
        "one": one == "1",
        "non_answer_abstains": other == "",
        "ties_abstain": tie == "",
        "passed": zero == "0" and one == "1" and other == "" and tie == "",
    }


def load_bundle() -> Tuple[Dict[str, Any], Dict[Tuple[int, str], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]], Dict[str, Any]]:
    manifest_path = REPAIR_ROOT / "MANIFEST.json"
    hashes_path = REPAIR_ROOT / "HASHES.json"
    if not manifest_path.is_file() or not hashes_path.is_file():
        raise DiagnosticError("repaired curriculum manifest or hashes are missing")
    manifest = read_json(manifest_path)
    hashes = read_json(hashes_path)
    if sha256_file(manifest_path) != hashes.get("manifest_sha256"):
        raise DiagnosticError("curriculum manifest hash mismatch")
    if manifest.get("training_started") or manifest.get("target_data_loaded") or manifest.get("sealed_target_answers_loaded"):
        raise DiagnosticError("curriculum manifest has an unsafe state flag")
    bundle: Dict[Tuple[int, str], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    seen_truth: set[str] = set()
    seen_structure: set[str] = set()
    expected_counts = {
        "source_train": 32,
        "source_validation": 32,
        "source_function_validation": 16,
        "source_test": 32,
        "execution_train": 16,
        "execution_validation": 16,
        "execution_test": 16,
    }
    for seed in SEEDS:
        for split, expected_count in expected_counts.items():
            public_path = REPAIR_ROOT / f"seed_{seed}" / f"{split}.jsonl"
            oracle_path = ORACLE_ROOT / f"seed_{seed}" / f"{split}.jsonl"
            if not public_path.is_file() or not oracle_path.is_file():
                raise DiagnosticError(f"missing source split or private oracle: seed={seed} split={split}")
            if hashes.get("public_files", {}).get(f"seed_{seed}/{split}.jsonl") != sha256_file(public_path):
                raise DiagnosticError(f"public split hash mismatch: seed={seed} split={split}")
            if manifest.get("source_oracle_files_outside_git", {}).get(f"seed_{seed}/{split}.jsonl") != sha256_file(oracle_path):
                raise DiagnosticError(f"private source oracle hash mismatch: seed={seed} split={split}")
            public = read_jsonl(public_path)
            oracle = read_jsonl(oracle_path)
            if len(public) != expected_count or len(oracle) != expected_count or len(public) != len(oracle):
                raise DiagnosticError(f"unexpected row count: seed={seed} split={split}")
            for row, truth in zip(public, oracle):
                if row["task_id"] != truth["task_id"]:
                    raise DiagnosticError(f"public/private task mismatch: {row['task_id']}")
                training = split.endswith("_train")
                if training != ("query_output" in row):
                    raise DiagnosticError(f"answer-label boundary violation: {row['task_id']}")
                if row.get("canonical_ast_sha256") != truth.get("canonical_ast_sha256") or row.get("truth_table_sha256") != truth.get("truth_table_sha256"):
                    raise DiagnosticError(f"oracle hash metadata mismatch: {row['task_id']}")
                if truth["truth_table_sha256"] in seen_truth or truth["canonical_ast_sha256"] in seen_structure:
                    raise DiagnosticError(f"cross-split function or structure overlap: {row['task_id']}")
                seen_truth.add(truth["truth_table_sha256"])
                seen_structure.add(truth["canonical_ast_sha256"])
                prompt_tokens(row)
                answer_token(truth["query_output"])
            bundle[(seed, split)] = (public, oracle)
    return manifest, bundle, {
        "manifest_sha256": sha256_file(manifest_path),
        "public_hashes_verified": True,
        "private_oracle_hashes_verified": True,
        "rows_checked": sum(len(value[0]) for value in bundle.values()),
        "function_hashes_unique": len(seen_truth) == sum(len(value[0]) for value in bundle.values()),
        "structure_hashes_unique": len(seen_structure) == sum(len(value[0]) for value in bundle.values()),
    }


def checkpoint_payload() -> Dict[str, Any]:
    if not CHECKPOINT.is_file():
        raise DiagnosticError(f"base checkpoint missing: {CHECKPOINT}")
    if sha256_file(CHECKPOINT) != "e49faa226aaa96bc26f4abbed700ecf313ad59cbf68869cfc0327c61f5902d5a":
        raise DiagnosticError("base checkpoint hash differs from the registered checkpoint")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != 15000:
        raise DiagnosticError("base checkpoint is not step 15,000")
    if not finite_tree(payload.get("model")) or not finite_tree(payload.get("optimizer")):
        raise DiagnosticError("base checkpoint contains non-finite model or optimizer tensors")
    required = {"model", "optimizer", "step", "torch_rng_state", "cuda_rng_state_all"}
    missing = sorted(required - set(payload))
    if missing:
        raise DiagnosticError(f"base checkpoint missing fields: {missing}")
    return payload


def verify_authorization(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise DiagnosticError(f"missing authorization artifact: {path}")
    authorization = read_json(path)
    if authorization.get("training_authorized") is not True:
        raise DiagnosticError("authorization artifact does not authorize training")
    if authorization.get("registration_commit") != "d6d983d":
        raise DiagnosticError("authorization artifact does not identify the registered protocol commit")
    if authorization.get("protocol_sha256") != sha256_file(PROTOCOL):
        raise DiagnosticError("authorization artifact protocol hash mismatch")
    if authorization.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise DiagnosticError("authorization artifact runner hash mismatch")
    for relative, expected in authorization.get("dependency_hashes", {}).items():
        path_to_check = ROOT / relative
        if not path_to_check.is_file() or sha256_file(path_to_check) != expected:
            raise DiagnosticError(f"diagnostic dependency changed: {relative}")
    for relative, expected in authorization.get("dirty_worktree_hashes", {}).items():
        path_to_check = ROOT / relative
        if not path_to_check.is_file() or sha256_file(path_to_check) != expected:
            raise DiagnosticError(f"pre-existing dirty file changed after audit: {relative}")
    if authorization.get("target_data_loaded") is not False or authorization.get("sealed_target_answers_loaded") is not False:
        raise DiagnosticError("authorization artifact has unsafe target-access flags")
    return authorization


def load_model(payload: Mapping[str, Any], seed: int, device: torch.device) -> torch.nn.Module:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = make_model(foundation_model_config("120m", False), device, seed)
    model.load_state_dict(payload["model"], strict=True)
    if not finite_model(model):
        raise DiagnosticError("loaded model is non-finite")
    return model


def evaluate_split(
    model: torch.nn.Module,
    public: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(public), batch_size):
        public_batch = public[start : start + batch_size]
        oracle_batch = oracle[start : start + batch_size]
        inputs, positions, _ = build_batch(public_batch, [item["query_output"] for item in oracle_batch], device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(inputs)
            if not bool(torch.isfinite(logits).all().item()):
                raise DiagnosticError("non-finite evaluation logits")
            answer_logits = logits[torch.arange(len(public_batch), device=device), positions].float().cpu()
        for row, truth, vector in zip(public_batch, oracle_batch, answer_logits):
            prediction, detail = decode_one_step(vector.tolist())
            expected = str(int(truth["query_output"]))
            rows.append({
                "task_id": row["task_id"],
                "prediction": prediction,
                "expected": expected,
                "answered": prediction in {"0", "1"},
                "correct": prediction == expected,
                "classification": "correct" if prediction == expected else ("abstain" if prediction == "" else "wrong"),
                **detail,
            })
    labels = {"0": sum(int(str(int(item["query_output"]))) == 0 for item in oracle), "1": sum(int(str(int(item["query_output"]))) == 1 for item in oracle)}
    eligible = len(rows)
    answered = sum(int(row["answered"]) for row in rows)
    correct = sum(int(row["correct"]) for row in rows)
    forced_correct = sum(int(row["forced_binary_prediction"] == row["expected"]) for row in rows)
    majority = max(labels.values(), default=0) / max(1, eligible)
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
        "forced_binary_correct": forced_correct,
        "forced_binary_accuracy": forced_correct / max(1, eligible),
        "label_counts": labels,
        "constant_zero_accuracy": labels["0"] / max(1, eligible),
        "constant_one_accuracy": labels["1"] / max(1, eligible),
        "majority_baseline_accuracy": majority,
        "balanced_random_baseline_accuracy": 0.5,
        "rows": rows,
    }


def evaluate_all(model: torch.nn.Module, seed: int, bundle: Mapping[Tuple[int, str], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]], device: torch.device) -> Dict[str, Any]:
    return {
        split: evaluate_split(model, *bundle[(seed, split)], device, MICROBATCH_SIZE)
        for split in EVAL_SPLITS
    }


def state_class(row: Mapping[str, Any]) -> str:
    return "correct" if row["correct"] else ("abstain" if not row["answered"] else "wrong")


def transition(previous: Mapping[str, Any], current: Mapping[str, Any]) -> Dict[str, Any]:
    previous_rows = {row["task_id"]: row for row in previous["rows"]}
    current_rows = {row["task_id"]: row for row in current["rows"]}
    matrix: Dict[str, int] = {}
    changed: List[str] = []
    for task_id in sorted(previous_rows):
        before = state_class(previous_rows[task_id])
        after = state_class(current_rows[task_id])
        key = f"{before}->{after}"
        matrix[key] = matrix.get(key, 0) + 1
        if before != after:
            changed.append(task_id)
    return {
        "matrix": matrix,
        "changed": len(changed),
        "changed_task_ids": changed,
        "regressions": matrix.get("correct->wrong", 0) + matrix.get("correct->abstain", 0),
        "improvements": matrix.get("wrong->correct", 0) + matrix.get("abstain->correct", 0),
    }


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, seed: int, phase: str, update: int, manifest_sha256: str) -> str:
    if not finite_model(model) or not finite_optimizer(optimizer):
        raise DiagnosticError(f"refusing to save non-finite state at {phase}/{seed}/{update}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA,
        "phase": phase,
        "seed": seed,
        "update": update,
        "base_checkpoint_sha256": sha256_file(CHECKPOINT),
        "curriculum_manifest_sha256": manifest_sha256,
        "model": cpu_tree(model.state_dict()),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": cpu_tree(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,
    }
    if not finite_tree(payload):
        raise DiagnosticError(f"checkpoint payload is non-finite at {phase}/{seed}/{update}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
    if not finite_tree(reloaded) or set(reloaded["model"]) != set(payload["model"]):
        temporary.unlink(missing_ok=True)
        raise DiagnosticError(f"checkpoint round-trip failed at {phase}/{seed}/{update}")
    os.replace(temporary, path)
    return sha256_file(path)


def run_update(model: torch.nn.Module, optimizer: torch.optim.Optimizer, rows: Sequence[Mapping[str, Any]], schedule: Sequence[Sequence[int]], schedule_position: int, device: torch.device) -> Tuple[float, float, int]:
    optimizer.zero_grad(set_to_none=True)
    losses: List[float] = []
    for micro in range(GRADIENT_ACCUMULATION):
        indices = schedule[schedule_position + micro]
        batch_rows = [rows[index] for index in indices]
        values = [row["query_output"] for row in batch_rows]
        inputs, positions, targets = build_batch(batch_rows, values, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(inputs)
            if not bool(torch.isfinite(logits).all().item()):
                raise DiagnosticError("non-finite training logits")
            answer_logits = logits[torch.arange(len(batch_rows), device=device), positions].float()
            loss = F.cross_entropy(answer_logits, targets) / GRADIENT_ACCUMULATION
        if not bool(torch.isfinite(loss).item()):
            raise DiagnosticError("non-finite training loss")
        loss.backward()
        losses.append(float(loss.detach().cpu().item() * GRADIENT_ACCUMULATION))
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP, error_if_nonfinite=True)
    if not bool(torch.isfinite(gradient_norm).item()):
        raise DiagnosticError("non-finite gradient norm")
    optimizer.step()
    if not finite_model(model) or not finite_optimizer(optimizer):
        raise DiagnosticError("non-finite committed model or optimizer state")
    return sum(losses) / max(1, len(losses)), float(gradient_norm.detach().cpu().item()), schedule_position + GRADIENT_ACCUMULATION


def build_schedule(seed: int, phase: str, row_count: int) -> Tuple[List[List[int]], str]:
    rng = random.Random(seed * 1000003 + PHASES[phase]["seed_offset"])
    schedule: List[List[int]] = []
    for _ in range(PHASES[phase]["updates"]):
        for _ in range(GRADIENT_ACCUMULATION):
            order = list(range(row_count))
            rng.shuffle(order)
            schedule.append(order[:MICROBATCH_SIZE])
    digest = hashlib.sha256(canonical(schedule).encode("utf-8")).hexdigest()
    return schedule, digest


def run_phase(seed: int, phase: str, bundle: Mapping[Tuple[int, str], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]], device: torch.device, manifest_sha256: str) -> Dict[str, Any]:
    payload = checkpoint_payload()
    model = load_model(payload, seed, device)
    optimizer = torch.optim.AdamW(parameter_groups(model, WEIGHT_DECAY), lr=LEARNING_RATE, betas=BETAS, eps=EPS)
    train_rows = bundle[(seed, PHASES[phase]["training_split"])][0]
    schedule, schedule_digest = build_schedule(seed, phase, len(train_rows))
    checkpoint_updates = set(PHASES[phase]["checkpoints"])
    previous: Dict[str, Any] | None = None
    initial: Dict[str, Any] | None = None
    records: List[Dict[str, Any]] = []
    schedule_position = 0
    started = time.perf_counter()
    for update in range(PHASES[phase]["updates"] + 1):
        if update in checkpoint_updates:
            evaluations = evaluate_all(model, seed, bundle, device)
            transitions_previous = None if previous is None else {split: transition(previous[split], evaluations[split]) for split in EVAL_SPLITS}
            transitions_initial = None if initial is None else {split: transition(initial[split], evaluations[split]) for split in EVAL_SPLITS}
            record = {
                "update": update,
                "elapsed_seconds": time.perf_counter() - started,
                "evaluations": evaluations,
                "transitions_from_previous": transitions_previous,
                "transitions_from_initial": transitions_initial,
                "protected_regressions": sum(item["regressions"] for item in (transitions_initial or {}).values()),
                "checkpoint_promotable": transitions_initial is None or all(item["regressions"] == 0 for item in transitions_initial.values()),
                "validation_qualification": {
                    "explicit_procedure_execution": qualification_from_evaluations(evaluations, "execution_validation"),
                    "query_identifiable_induction": qualification_from_evaluations(evaluations, "source_validation"),
                    "function_identifiable_induction": qualification_from_evaluations(evaluations, "source_function_validation"),
                },
            }
            if update > 0:
                checkpoint_path = CHECKPOINT_ROOT / phase / f"seed_{seed}" / f"update_{update:04d}.pt"
                record["checkpoint"] = {
                    "path": "<local-checkpoint-path-redacted>",
                    "sha256": save_checkpoint(checkpoint_path, model, optimizer, seed, phase, update, manifest_sha256),
                }
            records.append(record)
            if initial is None:
                initial = evaluations
            previous = evaluations
        if update == PHASES[phase]["updates"]:
            break
        model.train()
        _, _, schedule_position = run_update(model, optimizer, train_rows, schedule, schedule_position, device)
    report = {
        "schema_version": SCHEMA,
        "status": "completed",
        "phase": phase,
        "seed": seed,
        "training_started": True,
        "target_data_loaded": False,
        "sealed_target_answers_loaded": False,
        "training": {
            "updates": PHASES[phase]["updates"],
            "training_split": PHASES[phase]["training_split"],
            "rows": len(train_rows),
            "microbatch_size": MICROBATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "effective_examples_per_update": MICROBATCH_SIZE * GRADIENT_ACCUMULATION,
            "schedule_sha256": schedule_digest,
            "learning_rate": LEARNING_RATE,
            "precision": "bf16_autocast" if device.type == "cuda" else "fp32",
            "device": str(device),
        },
        "checkpoints": records,
        "final_validation_qualification": records[-1]["validation_qualification"],
        "final_protected_regressions": records[-1]["protected_regressions"],
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_interpretation": "source-training diagnostic only; no productive-retention or architectural-superiority claim",
    }
    write_json(OUTPUT / "TRAINING_RESULTS" / phase / f"seed_{seed}" / "REPORT.json", report)
    del model, optimizer, payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def qualification_from_evaluations(evaluations: Mapping[str, Any], split: str) -> bool:
    category = CATEGORIES["execution" if split.startswith("execution") else ("source_function" if split == "source_function_validation" else "source")]
    threshold = THRESHOLDS[category]
    evaluation = evaluations[split]
    return evaluation["correct"] >= threshold["min_correct"] and evaluation["answered"] >= threshold["min_coverage"]


def qualification(report: Mapping[str, Any], split: str) -> bool:
    return qualification_from_evaluations(report["evaluations"], split)


def preflight() -> Dict[str, Any]:
    manifest, bundle, data_checks = load_bundle()
    payload = checkpoint_payload()
    decoder = decoder_fixture()
    if not decoder["passed"]:
        raise DiagnosticError("decoder fixture failed")
    model = load_model(payload, 42, torch.device("cpu"))
    smoke_rows = [bundle[(42, "source_train")][0][0], bundle[(42, "execution_train")][0][0]]
    inputs, positions, _ = build_batch(smoke_rows, [0, 1], torch.device("cpu"))
    with torch.no_grad():
        logits = model(inputs)
        if not bool(torch.isfinite(logits).all().item()):
            raise DiagnosticError("CPU model smoke produced non-finite logits")
        answer_logits = logits[torch.arange(2), positions]
    roundtrip = False
    with tempfile.TemporaryDirectory(prefix="crucible-source-diagnostic-") as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save(cpu_tree(payload), path)
        restored = torch.load(path, map_location="cpu", weights_only=False)
        roundtrip = finite_tree(restored) and set(restored["model"]) == set(payload["model"])
    if not roundtrip:
        raise DiagnosticError("base checkpoint round-trip failed")
    report = {
        "schema_version": SCHEMA,
        "status": "preflight_passed",
        "scientific_authorization": False,
        "training_authorized": False,
        "training_started": False,
        "target_data_loaded": False,
        "sealed_target_answers_loaded": False,
        "cuda_training": "not run by preflight",
        "base_checkpoint": {
            "sha256": sha256_file(CHECKPOINT),
            "step": int(payload["step"]),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "finite_model": finite_tree(payload["model"]),
            "finite_optimizer": finite_tree(payload["optimizer"]),
        },
        "curriculum": data_checks,
        "decoder_fixture": decoder,
        "answer_position_smoke": {
            "rows": len(smoke_rows),
            "logits_shape": list(answer_logits.shape),
            "finite": bool(torch.isfinite(answer_logits).all().item()),
        },
        "checkpoint_roundtrip": roundtrip,
        "training_configuration": {
            "phases": PHASES,
            "microbatch_size": MICROBATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": BETAS,
            "eps": EPS,
            "gradient_clip_norm": GRAD_CLIP,
            "answer_loss": "fixed_answer_position_only",
        },
        "interpretation_boundary": [
            "Preflight validates implementation and data boundaries only.",
            "No source-task updates or held-out scientific training occurred.",
            "The original frozen diagnostic and old source-gate run remain unchanged.",
            "C0-C3 and the original two-arm comparison remain blocked.",
        ],
    }
    write_json(OUTPUT / "PREFLIGHT_REPORT.json", report)
    lines = [
        "# Crucible source-training diagnostic preflight",
        "",
        "Status: **PASS — implementation/data preflight only**",
        "",
        f"Base checkpoint SHA-256: `{report['base_checkpoint']['sha256']}`",
        f"Rows checked: {data_checks['rows_checked']}",
        f"Decoder fixture: `{decoder['passed']}`",
        f"Checkpoint round-trip: `{roundtrip}`",
        "",
        "No source training, target loading or sealed-answer evaluation was performed.",
    ]
    (OUTPUT / "PREFLIGHT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "train"), default="preflight")
    parser.add_argument("--phase", choices=("execution", "induction", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--authorize-source-training", action="store_true")
    parser.add_argument("--authorization-artifact", type=Path, default=AUTHORIZATION)
    args = parser.parse_args()
    try:
        if args.mode == "preflight":
            report = preflight()
            print(json.dumps({"status": report["status"], "training_started": False, "target_data_loaded": False}, sort_keys=True))
            return 0
        if not args.authorize_source_training:
            raise DiagnosticError("training requires explicit --authorize-source-training")
        authorization = verify_authorization(args.authorization_artifact.resolve())
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise DiagnosticError("CUDA requested but unavailable")
            if not torch.cuda.is_bf16_supported():
                raise DiagnosticError("CUDA device does not support registered BF16 mode")
        device = torch.device(args.device)
        manifest, bundle, _ = load_bundle()
        execution_reports = [run_phase(seed, "execution", bundle, device, sha256_file(REPAIR_ROOT / "MANIFEST.json")) for seed in SEEDS]
        execution_pass = all(qualification(report["checkpoints"][-1], "execution_validation") for report in execution_reports)
        reports: Dict[str, Any] = {"execution": execution_reports, "induction": []}
        if args.phase in {"induction", "all"}:
            if not execution_pass:
                raise DiagnosticError("execution positive-control gate failed; induction phase was not started")
            reports["induction"] = [run_phase(seed, "induction", bundle, device, sha256_file(REPAIR_ROOT / "MANIFEST.json")) for seed in SEEDS]
        summary = {
            "schema_version": SCHEMA,
            "status": "completed",
            "training_started": True,
            "target_data_loaded": False,
            "sealed_target_answers_loaded": False,
            "authorization_artifact": str(args.authorization_artifact.resolve()),
            "authorization_id": authorization.get("authorization_id"),
            "execution_positive_control_pass_both_seeds": execution_pass,
            "reports": reports,
        }
        write_json(OUTPUT / "TRAINING_SUMMARY.json", summary)
        print(json.dumps({"status": summary["status"], "execution_positive_control_pass_both_seeds": execution_pass, "induction_started": bool(reports["induction"])}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema_version": SCHEMA, "status": "halted", "error_type": type(exc).__name__, "error": str(exc), "training_started": args.mode == "train", "target_data_loaded": False, "sealed_target_answers_loaded": False}
        write_json(OUTPUT / "LAST_FAILURE.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
