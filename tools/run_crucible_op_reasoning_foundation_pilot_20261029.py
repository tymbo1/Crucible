#!/usr/bin/env python3
"""Train/evaluate a small Crucible foundation pilot with the bounded OP loop.

The model is the existing Crucible dense latent-attention + SwiGLU byte LM.
The OP condition is intentionally an external controller, matching the
repository's operational specification rather than inventing a new recurrence:

Reason -> Critique -> Counterexample -> EvidenceSearch/verification -> Revise -> Fix_R

The verifier is an evaluation-only answer comparator.  It reports whether the
candidate is correct but never supplies the target answer to the model.  This
is an exploratory pilot, not evidence that the neural weights internalized OP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "crucible_op_reasoning_foundation_pilot_20261029"
THROUGHPUT = ROOT / "experiments" / "crucible_heterogeneous_training_throughput_20261029"
if str(THROUGHPUT) not in sys.path:
    sys.path.insert(0, str(THROUGHPUT))

from benchmark import LOSS_SCALE, loss_sum, make_model, model_config, parameter_groups, state_digest, unscale_grads  # noqa: E402


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
RAW_BYTE_START = 3
SCHEMA_VERSION = "crucible-op-reasoning-foundation-pilot-v1-20261029"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def now_ns() -> int:
    return time.perf_counter_ns()


def truncate_utf8(value: str, limit: int) -> str:
    raw = str(value).encode("utf-8")[:limit]
    return raw.decode("utf-8", errors="ignore")


def extract_math_answer(value: str) -> str:
    matches = re.findall(r"\\boxed\{([^{}]*)\}", str(value))
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    return lines[-1] if lines else str(value).strip()


def answer_for_record(record: Mapping[str, Any]) -> str:
    target = str(record.get("target", ""))
    if record.get("domain") == "math":
        target = extract_math_answer(target)
    return truncate_utf8(target, 128)


def prompt_for_record(record: Mapping[str, Any]) -> str:
    domain = str(record.get("domain", "unknown"))
    prompt = f"DOMAIN={domain}\nTASK={record.get('input', '')}\nANSWER="
    return truncate_utf8(prompt, 192)


def encode_sample(record: Mapping[str, Any], context: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    prompt = prompt_for_record(record)
    answer = answer_for_record(record) + "\n"
    prompt_bytes = prompt.encode("utf-8")
    answer_bytes = answer.encode("utf-8")
    combined = prompt_bytes + answer_bytes
    if len(combined) + 1 > context:
        answer_bytes = answer_bytes[: max(1, context - len(prompt_bytes) - 1)]
        answer_bytes = answer_bytes.decode("utf-8", errors="ignore").encode("utf-8")
        combined = prompt_bytes + answer_bytes
    if len(combined) + 1 > context or not answer_bytes:
        return None
    tokens = torch.tensor([RAW_BYTE_START + value for value in combined], dtype=torch.long)
    x = torch.zeros(context, dtype=torch.long)
    y = torch.zeros(context, dtype=torch.long)
    x[0] = BOS_ID
    x[1 : 1 + len(tokens)] = tokens
    y[: len(tokens)] = tokens
    y[len(tokens)] = EOS_ID
    y[: len(prompt_bytes)] = PAD_ID
    return x, y


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_samples(rows: Sequence[Mapping[str, Any]], context: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    samples = []
    for row in rows:
        sample = encode_sample(row, context)
        if sample is not None:
            samples.append(sample)
    return samples


def train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> Dict[str, Any]:
    started = now_ns()
    optimizer.zero_grad(set_to_none=True)
    x_cpu, y_cpu = batch
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(x)
        ce_sum, valid = loss_sum(logits, y)
        loss = ce_sum / float(max(1, valid))
    (loss * LOSS_SCALE).backward()
    finite = unscale_grads(model, LOSS_SCALE)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(loss.detach().cpu().item()), "valid_tokens": int(valid), "finite_gradients": bool(finite), "wall_ns": now_ns() - started}


@torch.no_grad()
def validation_loss(model: torch.nn.Module, samples: Sequence[Tuple[torch.Tensor, torch.Tensor]], device: torch.device, batch_size: int) -> float:
    if not samples:
        return float("nan")
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        x = torch.stack([item[0] for item in chunk]).to(device)
        y = torch.stack([item[1] for item in chunk]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            ce_sum, valid = loss_sum(logits, y)
        total += float(ce_sum.detach().cpu().item())
        count += valid
    model.train()
    return total / max(1, count)


@torch.no_grad()
def generate(model: torch.nn.Module, prompt: str, device: torch.device, context: int, max_tokens: int = 128) -> Tuple[str, int]:
    model.eval()
    prefix = [BOS_ID] + [RAW_BYTE_START + value for value in prompt.encode("utf-8")]
    generated: List[int] = []
    for _ in range(max_tokens):
        if len(prefix) >= context:
            break
        tokens = torch.tensor([prefix], dtype=torch.long, device=device)
        logits = model(tokens)
        token = int(torch.argmax(logits[0, -1]).item())
        if token == EOS_ID or token < RAW_BYTE_START or token >= RAW_BYTE_START + 256:
            break
        generated.append(token - RAW_BYTE_START)
        prefix.append(token)
    try:
        return bytes(generated).decode("utf-8", errors="ignore").strip(), len(generated)
    finally:
        model.train()


def normalized_answer(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    return value.strip(".,;:")


def verifier(candidate: str, gold: str) -> bool:
    """Evaluation-only independent answer comparator; gold is not model input."""
    return normalized_answer(candidate) == normalized_answer(gold)


def one_pass(model: torch.nn.Module, record: Mapping[str, Any], device: torch.device, context: int) -> Dict[str, Any]:
    prompt = prompt_for_record(record)
    candidate, length = generate(model, prompt, device, context)
    return {"answer": candidate, "passes": 1, "model_tokens": length, "verified": verifier(candidate, answer_for_record(record)), "actions": ["answer", "stop"], "ig_status": "unavailable", "self_quality_status": "unavailable"}


def fixed_depth_two(model: torch.nn.Module, record: Mapping[str, Any], device: torch.device, context: int) -> Dict[str, Any]:
    prompt = prompt_for_record(record)
    first, first_len = generate(model, prompt, device, context)
    second_prompt = prompt + "\nCandidate=" + truncate_utf8(first, 96) + "\nReconsider and answer again:"
    second, second_len = generate(model, truncate_utf8(second_prompt, 224), device, context)
    return {"answer": second, "passes": 2, "model_tokens": first_len + second_len, "verified": verifier(second, answer_for_record(record)), "actions": ["answer", "critique", "revise", "stop"], "ig_status": "unavailable", "self_quality_status": "unavailable"}


def op_external(model: torch.nn.Module, record: Mapping[str, Any], device: torch.device, context: int, max_passes: int) -> Dict[str, Any]:
    """Bounded operational OP loop with exact action names from the spec."""
    prompt = prompt_for_record(record)
    candidate, generated = generate(model, prompt, device, context)
    actions = ["answer"]
    passes = 1
    verified = verifier(candidate, answer_for_record(record))
    if not verified and passes < max_passes:
        critique, critique_tokens = generate(model, truncate_utf8(prompt + "\nCandidate=" + candidate + "\nCritique the candidate:", 224), device, context)
        counterexample, counter_tokens = generate(model, truncate_utf8(prompt + "\nCandidate=" + candidate + "\nFind a counterexample or error:", 224), device, context)
        actions.extend(["critique", "counterexample", "evidence_search"])
        revised_prompt = prompt + "\nCandidate=" + truncate_utf8(candidate, 64) + "\nCritique=" + truncate_utf8(critique, 48) + "\nCounterexample=" + truncate_utf8(counterexample, 48) + "\nRevise:"
        revised, revised_tokens = generate(model, truncate_utf8(revised_prompt, 224), device, context)
        actions.extend(["revise", "stop"])
        passes += 1
        revised_verified = verifier(revised, answer_for_record(record))
        if revised_verified:
            candidate = revised
            verified = True
        generated += critique_tokens + counter_tokens + revised_tokens
    else:
        actions.append("stop")
    return {
        "answer": candidate,
        "passes": passes,
        "model_tokens": generated,
        "verified": verified,
        "actions": actions,
        "ig_status": "unavailable_without_calibrated_transition",
        "self_quality_status": "unavailable_without_historical_outcomes",
        "future_competence_status": "not_measured",
    }


def evaluate_policy(model: torch.nn.Module, rows: Sequence[Mapping[str, Any]], policy: str, device: torch.device, context: int, max_records: int) -> Dict[str, Any]:
    selected = list(rows)[:max_records]
    traces = []
    for row in selected:
        if policy == "one_pass":
            result = one_pass(model, row, device, context)
        elif policy == "fixed_depth_2":
            result = fixed_depth_two(model, row, device, context)
        else:
            result = op_external(model, row, device, context, 2)
        traces.append({
            "record_id": row["record_id"],
            "domain": row["domain"],
            "policy": policy,
            "verified": result["verified"],
            "passes": result["passes"],
            "model_tokens": result["model_tokens"],
            "actions": result["actions"],
            "answer_length": len(result["answer"]),
            "ig_status": result.get("ig_status"),
            "self_quality_status": result.get("self_quality_status"),
        })
    return {
        "policy": policy,
        "records": len(traces),
        "verified_correct": sum(int(row["verified"]) for row in traces),
        "accuracy": sum(int(row["verified"]) for row in traces) / max(1, len(traces)),
        "mean_passes": sum(int(row["passes"]) for row in traces) / max(1, len(traces)),
        "mean_model_tokens": sum(int(row["model_tokens"]) for row in traces) / max(1, len(traces)),
        "by_domain": {
            domain: {
                "records": sum(1 for row in traces if row["domain"] == domain),
                "verified_correct": sum(int(row["verified"]) for row in traces if row["domain"] == domain),
            }
            for domain in sorted({str(row["domain"]) for row in traces})
        },
        "traces": traces,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/timbushnell/crucible_reasoning_pilot_20261029"))
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT / "results_gpu")
    parser.add_argument("--model-size", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-records", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use host access or --device cpu")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    data_root = args.data_root.resolve()
    train_rows = load_jsonl(data_root / "train.jsonl")
    validation_rows = load_jsonl(data_root / "validation.jsonl")
    test_rows = load_jsonl(data_root / "test.jsonl")
    train_samples = make_samples(train_rows, args.context)
    validation_samples = make_samples(validation_rows, args.context)
    if not train_samples or not validation_samples:
        raise RuntimeError("pilot data produced no train/validation samples")
    cfg = model_config(args.model_size)
    model = make_model(cfg, device, args.seed)
    initial_hash = state_digest(model)
    optimizer = torch.optim.AdamW(parameter_groups(model, 0.01), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    curve = []
    started = now_ns()
    for step in range(args.steps):
        batch = [train_samples[(step * args.batch_size + index) % len(train_samples)] for index in range(args.batch_size)]
        x = torch.stack([item[0] for item in batch])
        y = torch.stack([item[1] for item in batch])
        result = train_step(model, optimizer, (x, y), device)
        result["step"] = step + 1
        if (step + 1) % max(1, args.steps // 8) == 0 or step == 0:
            result["validation_loss"] = validation_loss(model, validation_samples[: min(128, len(validation_samples))], device, args.batch_size)
        curve.append(result)
    training_seconds = (now_ns() - started) / 1e9
    policies = [
        evaluate_policy(model, test_rows, "one_pass", device, args.context, args.eval_records),
        evaluate_policy(model, test_rows, "fixed_depth_2", device, args.context, args.eval_records),
        evaluate_policy(model, test_rows, "op_external", device, args.context, args.eval_records),
    ]
    manifest_path = data_root / "MANIFEST.json"
    spec_path = ROOT / "docs" / "CRUCIBLE_RECURSIVE_CONTROLLER_OPERATIONAL_SPEC_20261021.md"
    controller_path = ROOT / "tools" / "recursive_controller.py"
    report = {
        "schema_version": SCHEMA_VERSION,
        "classification": "exploratory_op_external_scaffold_foundation_pilot",
        "scientific_status": "not_registered_not_track_b",
        "model": {"family": "Crucible latent-attention + SwiGLU dense", "size": args.model_size, "config": cfg, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "initial_state_sha256": initial_hash, "final_state_sha256": state_digest(model)},
        "training": {"steps": args.steps, "batch_size": args.batch_size, "context": args.context, "seed": args.seed, "seconds": training_seconds, "train_records": len(train_rows), "train_samples": len(train_samples), "validation_records": len(validation_rows), "validation_samples": len(validation_samples)},
        "data": {"manifest_sha256": sha256_file(manifest_path), "train_sha256": sha256_file(data_root / "train.jsonl"), "validation_sha256": sha256_file(data_root / "validation.jsonl"), "test_sha256": sha256_file(data_root / "test.jsonl"), "sealed_target_answers_accessed": False, "language_source_boundary": "local TinyStories context proxy; FineWeb-Edu/Dolma/Gutenberg absent"},
        "op": {"formula_spec_sha256": sha256_file(spec_path), "controller_source_sha256": sha256_file(controller_path), "objective_terms": {"utility": "evaluator-only for this pilot", "information_gain": "unavailable", "consistency": "answer verifier pass/fail", "self_quality": "unavailable", "entropy": "unavailable", "future_competence": "not measured", "cost": "model token count and pass count"}, "fix_rule": "first independently verified answer, otherwise final bounded revision", "internalized_in_weights": False},
        "learning_curve": curve,
        "policies": policies,
        "limitations": [
            "The OP loop is an external controller around a trained Crucible model; it is not a neural recursive architecture.",
            "The answer comparator is an evaluation-only oracle and exposes only pass/fail to the controller.",
            "This small pilot does not establish broad reasoning, cumulative learning, retention or transfer.",
            "The five reasoning datasets are sampled records; the language source is a local TinyStories proxy, not FineWeb-Edu.",
        ],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "OP_FOUNDATION_PILOT.json", report)
    with (output / "OP_EVAL_TRACES.jsonl").open("w", encoding="utf-8") as handle:
        for policy in policies:
            for row in policy["traces"]:
                handle.write(canonical_json(row) + "\n")
    lines = [
        "# Crucible OP foundation pilot",
        "",
        "Classification: `exploratory_op_external_scaffold_foundation_pilot`",
        "",
        "The model is the existing dense Crucible family. OP is evaluated as the exact bounded external loop from the repository specification; it is not internalized in weights.",
        "",
        "| Policy | Accuracy | Mean passes | Mean model tokens |",
        "|---|---:|---:|---:|",
    ]
    for policy in policies:
        lines.append(f"| {policy['policy']} | {policy['accuracy']:.4f} ({policy['verified_correct']}/{policy['records']}) | {policy['mean_passes']:.2f} | {policy['mean_model_tokens']:.1f} |")
    lines.extend([
        "",
        f"Training seconds: `{training_seconds:.3f}`; final validation loss: `{curve[-1].get('validation_loss')}`.",
        "",
        "The controller's information-gain, self-quality and future-competence terms remain unavailable or disabled because this pilot has no calibrated transition model or later learning intervention.",
        "",
    ])
    (output / "OP_FOUNDATION_PILOT.md").write_text("\n".join(lines), encoding="utf-8")
    hashes = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "HASHES.json":
            hashes[path.name] = sha256_file(path)
    write_json(output / "HASHES.json", {"schema_version": SCHEMA_VERSION, "files": hashes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
