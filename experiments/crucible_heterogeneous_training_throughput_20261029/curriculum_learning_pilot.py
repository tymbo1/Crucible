#!/usr/bin/env python3
"""Bounded GPU/CPU pilot for verified-procedure curriculum handoff.

This is an exploratory systems-and-learning pilot, not a registered Track B
experiment.  CPU workers use Crucible's current grammar candidate generator
and independent verifier.  Accepted procedures are serialized into later
GPU language-model batches in the curriculum arm.  The controls are:

* ``gpu_no_curriculum``: fixed Crucible byte-LM batches only;
* ``cpu_work_no_curriculum``: the same verified CPU work runs and is discarded;
* ``verified_curriculum``: verified procedures enter later GPU batches.

The worker receives demonstrations and independent verification examples, but
never receives a target AST.  The exploratory evaluation set is sampled from
the frozen representative grammar pool after excluding source/validation
functional hashes; it is not the sealed scientific target set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import (  # noqa: E402
    LOSS_SCALE,
    assemble_current,
    build_batches,
    loss_sum,
    make_model,
    model_config,
    parameter_groups,
    state_digest,
    unscale_grads,
    write_json,
)
from tools import crucible_cumulative_adaptation_scientific_runner_20261028 as actual_runner  # noqa: E402


SCHEMA_VERSION = "crucible-verified-curriculum-learning-pilot-v1-20261029"
RAW_BYTE_START = 3
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
DEFAULT_PAIRS = ((0, 0), (1, 2), (4, 3), (9, 8))
VERIFICATION_PAIRS = ((9, 9), (8, 7), (6, 5), (4, 2))
MODEL_CONTEXT = 128
WORKER_ARM = "A3_VERIFIED_ADAPTIVE_LIBRARY"

_ACTUAL_WORKER_READY = False


def now_ns() -> int:
    return time.perf_counter_ns()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_digest(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def worker_init(
    pool: Sequence[Mapping[str, Any]],
    truth_tables: Any,
    promoted_records: Sequence[Any],
) -> None:
    global _ACTUAL_WORKER_READY
    actual_runner._init_crucible_cpu_worker(
        pool,
        truth_tables,
        tuple(promoted_records),
        WORKER_ARM,
    )
    _ACTUAL_WORKER_READY = True


def worker_generate_and_verify(payload: Tuple[str, Sequence[Mapping[str, int]], Sequence[Mapping[str, int]]]) -> Dict[str, Any]:
    """Generate and independently verify one procedure without target labels."""
    if not _ACTUAL_WORKER_READY:
        raise RuntimeError("actual Crucible worker was not initialized")
    task_id, demonstrations, verification_examples = payload
    started = now_ns()
    _, candidates = actual_runner._crucible_cpu_candidate_job((str(task_id), tuple(demonstrations)))
    verifier = actual_runner.IndependentVerifier()
    traces: List[Dict[str, Any]] = []
    accepted: Optional[Dict[str, Any]] = None
    for slot, candidate in enumerate(candidates[:4]):
        demo_ok, demo_checks = verifier.verify(candidate["ast"], demonstrations)
        independent_ok = False
        independent_checks = 0
        if demo_ok:
            independent_ok, independent_checks = verifier.verify(candidate["ast"], verification_examples)
        traces.append({
            "slot": int(slot),
            "canonical_ast_hash": str(candidate["canonical_ast_hash"]),
            "demonstration_ok": bool(demo_ok),
            "independent_ok": bool(independent_ok),
            "demo_checks": int(demo_checks),
            "independent_checks": int(independent_checks),
            "source": candidate.get("source"),
        })
        if demo_ok and independent_ok and accepted is None:
            accepted = dict(candidate)
    stable = {
        "task_id": str(task_id),
        "candidate_hashes": [str(item["canonical_ast_hash"]) for item in candidates[:4]],
        "accepted_hash": None if accepted is None else str(accepted["canonical_ast_hash"]),
        "traces": traces,
    }
    return {
        "task_id": str(task_id),
        "accepted": accepted is not None,
        "accepted_ast": None if accepted is None else actual_runner.canonical_ast(accepted["ast"]),
        "accepted_hash": None if accepted is None else str(accepted["canonical_ast_hash"]),
        "candidate_count": len(candidates[:4]),
        "traces": traces,
        "result_digest": json_digest(stable),
        "worker_pid": os.getpid(),
        "task_wall_ns": now_ns() - started,
    }


class OrderedVerifiedProducer:
    """Bounded ordered CPU reservoir for verified procedure records."""

    def __init__(
        self,
        payloads: Sequence[Tuple[str, Sequence[Mapping[str, int]], Sequence[Mapping[str, int]]]],
        pool: Sequence[Mapping[str, Any]],
        truth_tables: Any,
        promoted_records: Sequence[Any],
        workers: int,
        window: int,
    ) -> None:
        self.payloads = list(payloads)
        if not self.payloads:
            raise ValueError("producer payloads cannot be empty")
        self.workers = max(1, int(workers))
        self.window = max(1, int(window))
        context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        kwargs: Dict[str, Any] = {
            "max_workers": self.workers,
            "initializer": worker_init,
            "initargs": (list(pool), truth_tables, tuple(promoted_records)),
        }
        if context is not None:
            kwargs["mp_context"] = context
        self.pool = ProcessPoolExecutor(**kwargs)
        self.pending: Dict[int, Tuple[Future[Dict[str, Any]], int]] = {}
        self.next_submit = 0
        self.next_commit = 0
        self.result_digests: List[str] = []
        self.results: List[Dict[str, Any]] = []
        self.max_occupancy = 0

    def _fill(self) -> None:
        while self.next_submit < len(self.payloads) and len(self.pending) < self.window:
            index = self.next_submit
            self.pending[index] = (self.pool.submit(worker_generate_and_verify, self.payloads[index]), now_ns())
            self.next_submit += 1
            self.max_occupancy = max(self.max_occupancy, len(self.pending))

    def prime(self) -> None:
        self._fill()

    def get(self) -> Dict[str, Any]:
        if self.next_commit >= len(self.payloads):
            raise RuntimeError("verified producer exhausted")
        self._fill()
        future, submitted = self.pending.pop(self.next_commit)
        result = future.result()
        result["queue_age_ns"] = now_ns() - submitted
        result["ordered_index"] = self.next_commit
        self.next_commit += 1
        self.results.append(result)
        self.result_digests.append(str(result["result_digest"]))
        self._fill()
        return result

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=False)

    def summary(self) -> Dict[str, Any]:
        return {
            "declared_workers": self.workers,
            "window": self.window,
            "max_queue_occupancy": self.max_occupancy,
            "submitted": self.next_submit,
            "committed": self.next_commit,
            "accepted": sum(int(item["accepted"]) for item in self.results),
            "worker_pids": sorted({int(item["worker_pid"]) for item in self.results}),
            "result_digest": sha256_bytes("|".join(self.result_digests).encode("ascii")),
            "task_wall_ns": sum(int(item["task_wall_ns"]) for item in self.results),
        }


def make_prompt(demonstrations: Sequence[Mapping[str, int]]) -> str:
    rows = ";".join(
        f"{int(row['a'])},{int(row['b'])},{int(row['output'])}"
        for row in list(demonstrations)[:4]
    )
    return f"D:{rows}|A:"


def make_lm_sample(
    prompt: str,
    answer_ast: Mapping[str, Any],
    context: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    prompt_bytes = prompt.encode("utf-8")
    answer_bytes = (canonical_json(answer_ast) + "\n").encode("utf-8")
    text_bytes = prompt_bytes + answer_bytes
    if len(text_bytes) + 1 > context:
        return None
    x = torch.zeros(context, dtype=torch.long)
    y = torch.zeros(context, dtype=torch.long)
    text_tokens = torch.tensor([RAW_BYTE_START + value for value in text_bytes], dtype=torch.long)
    x[0] = BOS_ID
    x[1 : 1 + len(text_tokens)] = text_tokens
    y[: len(text_tokens)] = text_tokens
    y[len(text_tokens)] = EOS_ID
    # Only the answer and EOS are supervised.  The prompt is conditioning data.
    y[: len(prompt_bytes)] = PAD_ID
    return x, y


def make_base_batch(raw: torch.Tensor, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y, _, _ = assemble_current(raw, batch)
    return x.clone(), y.clone()


def build_source_work(
    source_tasks: Sequence[Mapping[str, Any]],
) -> List[Tuple[str, Sequence[Mapping[str, int]], Sequence[Mapping[str, int]]]]:
    payloads = []
    for task in source_tasks:
        payloads.append((
            str(task["task_id"]),
            tuple(dict(row) for row in task.get("demonstrations", [])),
            tuple(actual_runner.examples_for_ast(task["target_ast"], VERIFICATION_PAIRS)),
        ))
    return payloads


def load_curriculum_inputs() -> Dict[str, Any]:
    root = ROOT / "experiments" / "crucible_cumulative_adaptation_preflight_20261028"
    frozen_pool = actual_runner.load_frozen_pool()
    source_rows = actual_runner.load_jsonl(root / "SOURCE_MANIFEST.jsonl")
    validation_rows = actual_runner.load_jsonl(root / "OPERATOR_VALIDATION_MANIFEST.jsonl")
    source_tasks = actual_runner.make_source_tasks(source_rows)
    validation_tasks = actual_runner.make_validation_tasks(validation_rows)
    runner = actual_runner.FourArmRunner(frozen_pool, root, scientific=False)
    state = runner._prepare_arm(WORKER_ARM, source_tasks, validation_tasks)
    source_truth = {str(task["truth_table_hash"]) for task in source_tasks}
    validation_truth = {str(task["truth_table_hash"]) for task in validation_tasks}
    targets: List[Dict[str, Any]] = []
    seen_truth: set[str] = set()
    for row in sorted(frozen_pool, key=lambda item: str(item["canonical_ast_hash"])):
        truth = str(row["truth_table_hash"])
        if truth in source_truth or truth in validation_truth or truth in seen_truth:
            continue
        ast = actual_runner.canonical_ast(row["canonical_ast"])
        prompt = make_prompt(actual_runner.examples_for_ast(ast, DEFAULT_PAIRS))
        answer = canonical_json(ast) + "\n"
        # Freeze only tasks whose complete prompt and gold answer fit the
        # model context.  This is a pre-run feasibility rule, not an accuracy
        # filter applied after seeing outcomes.
        if len(prompt.encode("utf-8")) + len(answer.encode("utf-8")) + 1 > MODEL_CONTEXT:
            continue
        targets.append({
            "task_id": f"exploratory_target_{len(targets):03d}",
            "canonical_ast": ast,
            "canonical_ast_hash": str(row["canonical_ast_hash"]),
            "truth_table_hash": truth,
            "demonstrations": actual_runner.examples_for_ast(ast, DEFAULT_PAIRS),
        })
        seen_truth.add(truth)
        if len(targets) == 16:
            break
    if len(targets) < 16:
        raise RuntimeError("could not construct 16 functionally disjoint exploratory targets")
    return {
        "root": root,
        "pool": frozen_pool,
        "source_tasks": source_tasks,
        "validation_tasks": validation_tasks,
        "promoted_records": tuple(state["library"].promoted),
        "activation_status": state["library"].activation_status,
        "pool_truth_tables": runner.pool_truth_tables,
        "targets": targets,
    }


def make_curriculum_records(
    producer_results: Sequence[Mapping[str, Any]],
    source_task_by_id: Mapping[str, Mapping[str, Any]],
    context: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for result in producer_results:
        if not result.get("accepted") or result.get("accepted_ast") is None:
            continue
        task = source_task_by_id[str(result["task_id"])]
        sample = make_lm_sample(make_prompt(task["demonstrations"]), result["accepted_ast"], context)
        if sample is None:
            continue
        records.append({
            "task_id": str(result["task_id"]),
            "accepted_hash": str(result["accepted_hash"]),
            "ast": result["accepted_ast"],
            "sample": sample,
        })
    return records


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    started = now_ns()
    optimizer.zero_grad(set_to_none=True)
    x = x_cpu.to(device, non_blocking=False)
    y = y_cpu.to(device, non_blocking=False)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(x)
        ce_sum, valid = loss_sum(logits, y)
        loss = ce_sum / float(max(1, valid))
    (loss * LOSS_SCALE).backward()
    finite = unscale_grads(model, LOSS_SCALE)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu().item()),
        "valid_tokens": int(valid),
        "finite_gradients": bool(finite),
        "wall_ns": now_ns() - started,
    }


@torch.no_grad()
def generate_answer(model: torch.nn.Module, prompt: str, device: torch.device, context: int, max_tokens: int = 100) -> Tuple[str, str]:
    model.eval()
    prefix = [BOS_ID] + [RAW_BYTE_START + value for value in prompt.encode("utf-8")]
    generated: List[int] = []
    for _ in range(max_tokens):
        if len(prefix) >= context:
            break
        input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        logits = model(input_ids)
        next_id = int(torch.argmax(logits[0, -1]).item())
        if next_id == EOS_ID:
            break
        if next_id < RAW_BYTE_START or next_id >= 256 + RAW_BYTE_START:
            break
        generated.append(next_id - RAW_BYTE_START)
        prefix.append(next_id)
    try:
        decoded = bytes(generated).decode("utf-8").strip()
    except UnicodeDecodeError:
        decoded = ""
    return decoded, sha256_bytes(bytes(generated))


@torch.no_grad()
def evaluate_targets(
    model: torch.nn.Module,
    targets: Sequence[Mapping[str, Any]],
    device: torch.device,
    context: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for target in targets:
        prompt = make_prompt(target["demonstrations"])
        decoded, output_hash = generate_answer(model, prompt, device, context)
        teacher_forced_valid = 0
        teacher_forced_correct = 0
        teacher_sample = make_lm_sample(prompt, target["canonical_ast"], context)
        if teacher_sample is not None:
            teacher_x, teacher_y = teacher_sample
            logits = model(teacher_x.unsqueeze(0).to(device))
            valid = teacher_y != PAD_ID
            predictions = torch.argmax(logits[0], dim=-1).detach().cpu()
            teacher_forced_valid = int(valid.sum().item())
            teacher_forced_correct = int((predictions[valid] == teacher_y[valid]).sum().item())
        parsed = False
        functional = False
        structural = False
        parse_error: Optional[str] = None
        try:
            candidate = actual_runner.GRAMMAR.validate(json.loads(decoded))
            parsed = True
            structural = actual_runner.ast_hash(candidate) == str(target["canonical_ast_hash"])
            functional = actual_runner.truth_table_hash(candidate) == str(target["truth_table_hash"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            parse_error = type(exc).__name__
        rows.append({
            "task_id": str(target["task_id"]),
            "decoded_sha256": output_hash,
            "decoded_length": len(decoded),
            "parsed": parsed,
            "structurally_correct": structural,
            "functionally_correct": functional,
            "parse_error": parse_error,
            "teacher_forced_valid_tokens": teacher_forced_valid,
            "teacher_forced_correct_tokens": teacher_forced_correct,
        })
    total_teacher_tokens = sum(int(row["teacher_forced_valid_tokens"]) for row in rows)
    total_teacher_correct = sum(int(row["teacher_forced_correct_tokens"]) for row in rows)
    return {
        "task_count": len(rows),
        "parsed": sum(int(row["parsed"]) for row in rows),
        "structurally_correct": sum(int(row["structurally_correct"]) for row in rows),
        "functionally_correct": sum(int(row["functionally_correct"]) for row in rows),
        "teacher_forced_valid_tokens": total_teacher_tokens,
        "teacher_forced_correct_tokens": total_teacher_correct,
        "teacher_forced_token_accuracy": total_teacher_correct / max(1, total_teacher_tokens),
        "rows": rows,
    }


def run_arm_fixed(
    arm: str,
    base_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    payloads: Sequence[Tuple[str, Sequence[Mapping[str, int]], Sequence[Mapping[str, int]]]],
    inputs: Mapping[str, Any],
    cfg: Mapping[str, int],
    device: torch.device,
    steps: int,
    workers: int,
    window: int,
    context: int,
    seed: int,
) -> Dict[str, Any]:
    """Run an arm with explicit curriculum queue bookkeeping."""
    model = make_model(cfg, device, seed)
    initial = state_digest(model)
    optimizer = torch.optim.AdamW(parameter_groups(model, 0.01), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    producer: Optional[OrderedVerifiedProducer] = None
    if arm in {"cpu_work_no_curriculum", "verified_curriculum"}:
        producer = OrderedVerifiedProducer(
            payloads,
            inputs["pool"],
            inputs["pool_truth_tables"],
            inputs["promoted_records"],
            workers,
            window,
        )
        producer.prime()
    accepted_buffer: List[Dict[str, Any]] = []
    producer_results: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    started = now_ns()
    try:
        for step in range(steps):
            if arm == "verified_curriculum":
                while len(accepted_buffer) < 4:
                    result = producer.get() if producer is not None else None
                    if result is None:
                        raise RuntimeError("curriculum producer unavailable")
                    producer_results.append(result)
                    if not result.get("accepted") or result.get("accepted_ast") is None:
                        continue
                    task = next(item for item in inputs["source_tasks"] if str(item["task_id"]) == str(result["task_id"]))
                    sample = make_lm_sample(make_prompt(task["demonstrations"]), result["accepted_ast"], context)
                    if sample is not None:
                        accepted_buffer.append({"x": sample[0], "y": sample[1]})
                batch_records = [accepted_buffer.pop(0) for _ in range(4)]
                x_cpu = torch.stack([record["x"] for record in batch_records])
                y_cpu = torch.stack([record["y"] for record in batch_records])
            else:
                x_cpu, y_cpu = base_batches[step % len(base_batches)]
            train = train_step(model, optimizer, x_cpu, y_cpu, device)
            train["step"] = step
            step_rows.append(train)
            if producer is not None and arm == "cpu_work_no_curriculum":
                target_commits = ((step + 1) * len(payloads)) // steps
                commits_this_step = target_commits - len(producer_results)
                for _ in range(max(0, commits_this_step)):
                    producer_results.append(producer.get())
        # Both CPU arms consume the identical ordered task prefix.  The
        # curriculum arm may need fewer results to fill a batch because some
        # proposals fail independent verification; drain the remainder after
        # training so the CPU-work control has an exactly matched workload.
        if producer is not None:
            while producer.next_commit < len(payloads):
                producer.get()
    finally:
        if producer is not None:
            producer.close()
    elapsed = now_ns() - started
    evaluation = evaluate_targets(model, inputs["targets"], device, context)
    return {
        "arm": arm,
        "initial_state_sha256": initial,
        "final_state_sha256": state_digest(model),
        "steps": steps,
        "elapsed_ns": elapsed,
        "mean_loss": sum(float(row["loss"]) for row in step_rows) / max(1, len(step_rows)),
        "step_rows": step_rows,
        "evaluation": evaluation,
        "producer": None if producer is None else producer.summary(),
        "producer_results": producer_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exploratory verified-procedure curriculum pilot")
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--target-count", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use host GPU access or --device cpu")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.synchronize(device)
    torch.set_num_threads(1)
    context = MODEL_CONTEXT
    inputs = load_curriculum_inputs()
    if len(inputs["targets"]) != args.target_count:
        inputs["targets"] = inputs["targets"][: args.target_count]
    _, data_sha, _, raw, batches, data_report = build_batches(args.context_root.resolve(), args.steps)
    by_step: Dict[int, List[Any]] = {}
    for batch in batches:
        by_step.setdefault(int(batch.step), []).append(batch)
    base_batches = [make_base_batch(raw, by_step[step][0]) for step in sorted(by_step) if step <= args.steps]
    if len(base_batches) < args.steps:
        raise RuntimeError(f"context schedule provides {len(base_batches)} steps, need {args.steps}")
    base_batches = base_batches[: args.steps]
    source_payloads = build_source_work(inputs["source_tasks"])
    payloads = [source_payloads[index % len(source_payloads)] for index in range(args.steps * 12)]
    cfg = model_config(args.model_size)
    arms = []
    for arm in ("gpu_no_curriculum", "cpu_work_no_curriculum", "verified_curriculum"):
        result = run_arm_fixed(
            arm,
            base_batches,
            payloads,
            inputs,
            cfg,
            device,
            args.steps,
            args.workers,
            args.window,
            context,
            42,
        )
        arms.append(result)
    curriculum_result = next(item for item in arms if item["arm"] == "verified_curriculum")
    cpu_result = next(item for item in arms if item["arm"] == "cpu_work_no_curriculum")
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "classification": "exploratory_curriculum_learning_pilot",
        "scientific_status": "not_registered_not_track_b",
        "device": str(device),
        "model": {"family": "Crucible latent-attention + SwiGLU dense", "size": args.model_size, "config": cfg, "parameter_count": sum(parameter.numel() for parameter in make_model(cfg, device, 42).parameters())},
        "training": {"steps": args.steps, "batch_size": 4, "context": context, "seed": 42, "workers": args.workers, "window": args.window},
        "data_manifest_sha256": data_sha,
        "data_report": {key: {k: v for k, v in value.items() if k != "path"} for key, value in data_report.items()},
        "source_task_count": len(inputs["source_tasks"]),
        "validation_task_count": len(inputs["validation_tasks"]),
        "operator_activation_status": inputs["activation_status"],
        "target_set": {
            "count": len(inputs["targets"]),
            "functional_disjoint_from_source_and_validation": True,
            "sealed_target_answers_accessed_by_workers": False,
            "evaluation_used_exploratory_pool_answers_after_training": True,
            "truth_hashes": [str(item["truth_table_hash"]) for item in inputs["targets"]],
        },
        "producer_task_manifest_sha256": json_digest(payloads),
        "arms": arms,
        "paired_comparison": {
            "verified_curriculum_minus_gpu_no_curriculum_functional": curriculum_result["evaluation"]["functionally_correct"] - next(item for item in arms if item["arm"] == "gpu_no_curriculum")["evaluation"]["functionally_correct"],
            "verified_curriculum_minus_cpu_work_no_curriculum_functional": curriculum_result["evaluation"]["functionally_correct"] - cpu_result["evaluation"]["functionally_correct"],
            "cpu_work_digest_matches_curriculum": cpu_result["producer"]["result_digest"] == curriculum_result["producer"]["result_digest"],
        },
        "limitations": [
            "This is a bounded exploratory curriculum handoff, not a registered scientific learning result.",
            "The small model is a Crucible-family proxy, not the planned 500M model.",
            "The curriculum serializes verified ASTs into a byte-LM stream; it does not establish autonomous procedure induction.",
            "The exploratory held-out answers are used only after training and are not the sealed Track B target set.",
            "No persistence/reset or contradiction-detection claim is tested here.",
        ],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "CURRICULUM_PILOT.json", report)
    lines = [
        "# Verified-procedure curriculum pilot",
        "",
        f"Classification: `{report['classification']}`",
        "",
        "This exploratory pilot measures whether verified procedure records can enter later GPU batches. It is not Track B and does not reopen the registered benchmark.",
        "",
        "| Arm | Functional target success | Parsed | Teacher-forced token accuracy | Mean loss | Verified CPU procedures | Wall (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in arms:
        producer = item.get("producer") or {}
        lines.append(
            f"| {item['arm']} | {item['evaluation']['functionally_correct']}/{item['evaluation']['task_count']} | {item['evaluation']['parsed']}/{item['evaluation']['task_count']} | {item['evaluation']['teacher_forced_token_accuracy']:.3f} | {item['mean_loss']:.4f} | {producer.get('accepted', 0)} | {item['elapsed_ns'] / 1e9:.3f} |"
        )
    lines.extend([
        "",
        f"Producer result digest matched between CPU-work control and curriculum arm: `{report['paired_comparison']['cpu_work_digest_matches_curriculum']}`.",
        "",
        "Interpretation: any curriculum-arm improvement is exploratory evidence that verified records affected this small model's training stream. It is not evidence of cumulative learning or general neural procedure induction.",
        "",
        "## Boundary",
        "",
        "CPU workers received demonstrations and independent verification examples only. The exploratory target ASTs were not sent to workers and were evaluated only after training.",
        "",
    ])
    (output / "CURRICULUM_PILOT.md").write_text("\n".join(lines), encoding="utf-8")
    hashes = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "HASHES.json":
            hashes[path.name] = sha256_bytes(path.read_bytes())
    write_json(output / "HASHES.json", {"schema_version": SCHEMA_VERSION, "files": hashes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
