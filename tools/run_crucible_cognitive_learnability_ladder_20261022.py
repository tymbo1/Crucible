#!/usr/bin/env python3
"""CPU-only cognitive learnability ladder for Crucible.

This study uses the validated Stage A dense control only.  It deliberately
separates in-distribution learnability from held-out generalization before any
E4-versus-dense comparison is attempted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
STAGE_A_PATH = ROOT / "tools" / "run_crucible_stage_a_instrumentation_20261022.py"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_cognitive_learnability_ladder_20261022_seed42"
STAGES = ("atomic", "composition", "relational", "logic")
SPLITS = ("train", "validation", "heldout", "test")
RECORDS_PER_STAGE = {"train": 128, "validation": 64, "heldout": 64, "test": 64}
ANSWER_MARKER = b"\nAnswer:"

spec = importlib.util.spec_from_file_location("stage_a_runner", STAGE_A_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import Stage A runner: {STAGE_A_PATH}")
stage_a = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stage_a
spec.loader.exec_module(stage_a)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def split_rng(seed: int, stage: str, split: str, index: int) -> random.Random:
    material = f"crucible-learnability-ladder-v1|{seed}|{stage}|{split}|{index}"
    return random.Random(int.from_bytes(hashlib.sha256(material.encode()).digest()[:16], "big"))


def record_id(seed: int, stage: str, split: str, index: int) -> str:
    return sha256_bytes(f"ladder-v1|{seed}|{stage}|{split}|{index}".encode())[:20]


def make_record(
    seed: int,
    stage: str,
    split: str,
    index: int,
    prompt: str,
    target: str,
    oracle: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": "crucible_cognitive_learnability_ladder_v1",
        "record_id": record_id(seed, stage, split, index),
        "stage": stage,
        "split": split,
        "input": prompt,
        "target": target,
        "oracle": dict(oracle),
        "metadata": dict(metadata),
    }


def make_atomic(seed: int, split: str, index: int) -> Dict[str, Any]:
    rng = split_rng(seed, "atomic", split, index)
    digit = rng.randrange(10)
    template = index % 3
    if template == 0:
        prompt = f"Task: successor modulo ten. Input digit: {digit}. Return one digit only."
    elif template == 1:
        prompt = f"Rule successor_mod_10. Given digit={digit}, output the next digit modulo 10."
    else:
        prompt = f"Apply successor_mod_10 to x={digit}; answer with the resulting digit."
    target = str((digit + 1) % 10)
    return make_record(seed, "atomic", split, index, prompt, target, {"kind": "successor_mod_10", "value": int(target)}, {"difficulty": 0})


COMPOSITIONS = {
    "train": (("add", "add"), ("add", "mul")),
    "validation": (("add", "add"), ("add", "mul")),
    "heldout": (("mul", "add"), ("sub", "add")),
    "test": (("mul", "mul"), ("sub", "mul")),
}


def apply_mod(value: int, operation: str, operand: int) -> int:
    if operation == "add":
        return (value + operand) % 10
    if operation == "sub":
        return (value - operand) % 10
    if operation == "mul":
        return (value * operand) % 10
    raise ValueError(operation)


def make_composition(seed: int, split: str, index: int) -> Dict[str, Any]:
    rng = split_rng(seed, "composition", split, index)
    operations = COMPOSITIONS[split][index % len(COMPOSITIONS[split])]
    value = rng.randrange(10)
    start = value
    steps = []
    for operation in operations:
        operand = rng.randrange(1, 5)
        value = apply_mod(value, operation, operand)
        steps.append(f"{operation}({operand})")
    prompt = (
        "Task: arithmetic is modulo ten. Apply both operations left-to-right. "
        f"start={start}; step1={steps[0]}; step2={steps[1]}. Return one digit only."
    )
    return make_record(
        seed,
        "composition",
        split,
        index,
        prompt,
        str(value),
        {"kind": "mod10_composition", "value": value, "operations": operations},
        {"difficulty": 1, "composition": operations, "heldout_composition": split in {"heldout", "test"}},
    )


def make_chain(seed: int, split: str, index: int) -> Dict[str, Any]:
    rng = split_rng(seed, "relational", split, index)
    path_lengths = {"train": 2, "validation": 2, "heldout": 3, "test": 4}
    distractors = {"train": 1, "validation": 1, "heldout": 2, "test": 4}
    length = path_lengths[split]
    entities = [f"N{index}_{position}" for position in range(length + 1)]
    facts = [f"{entities[i]} -> {entities[i + 1]}" for i in range(length)]
    for number in range(distractors[split]):
        facts.append(f"D{index}_{number}_L -> D{index}_{number}_R")
    rng.shuffle(facts)
    positive = index % 2 == 0
    left, right = (entities[0], entities[-1]) if positive else (entities[-1], entities[0])
    answer = "yes" if positive else "no"
    prompt = "Directed facts: " + " | ".join(facts) + f". Query: {left} -> {right}? Return yes or no only."
    return make_record(
        seed,
        "relational",
        split,
        index,
        prompt,
        answer,
        {"kind": "directed_chain", "path_length": length, "answer": answer},
        {"difficulty": 2, "path_length": length, "distractors": distractors[split], "heldout_depth": split in {"heldout", "test"}},
    )


def closure(facts: Iterable[str], rules: Iterable[Tuple[str, str]]) -> set[str]:
    result = set(facts)
    changed = True
    while changed:
        changed = False
        for premise, conclusion in rules:
            if premise in result and conclusion not in result:
                result.add(conclusion)
                changed = True
    return result


def make_logic(seed: int, split: str, index: int) -> Dict[str, Any]:
    rng = split_rng(seed, "logic", split, index)
    depths = {"train": 2, "validation": 2, "heldout": 3, "test": 4}
    distractors = {"train": 1, "validation": 2, "heldout": 3, "test": 5}
    depth = depths[split]
    atoms = [f"P{index}_{position}" for position in range(depth + 1)]
    facts = [atoms[0]]
    rules = [(atoms[position], atoms[position + 1]) for position in range(depth)]
    for number in range(distractors[split]):
        facts.append(f"D{index}_{number}")
        rules.append((f"D{index}_{number}", f"E{index}_{number}"))
    rng.shuffle(facts)
    rng.shuffle(rules)
    case = index % 3
    if case == 0:
        query, answer = atoms[-1], "entailed"
    elif case == 1:
        query, answer = f"not {atoms[-1]}", "contradicted"
    else:
        query, answer = f"Q{index}", "unknown"
    prompt = (
        "Facts: " + ", ".join(facts) + ". Rules: " +
        ", ".join(f"if {left} then {right}" for left, right in rules) +
        f". Query: {query}. Return entailed, contradicted or unknown only."
    )
    return make_record(
        seed,
        "logic",
        split,
        index,
        prompt,
        answer,
        {"kind": "forward_chain", "depth": depth, "answer": answer, "closure": sorted(closure(facts, rules))},
        {"difficulty": 3, "rule_depth": depth, "distractors": distractors[split], "heldout_depth": split in {"heldout", "test"}},
    )


GENERATORS = {"atomic": make_atomic, "composition": make_composition, "relational": make_chain, "logic": make_logic}


def generate_ladder(seed: int) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for stage in STAGES:
        result[stage] = {}
        for split in SPLITS:
            result[stage][split] = [
                GENERATORS[stage](seed, split, index)
                for index in range(RECORDS_PER_STAGE[split])
            ]
    return result


def prepare_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared = []
    for original in records:
        record = dict(original)
        prompt = record["input"].encode("utf-8") + ANSWER_MARKER
        target = record["target"].encode("utf-8")
        full = [stage_a.BOS_ID] + stage_a.serialize_bytes(prompt) + stage_a.serialize_bytes(target) + [stage_a.EOS_ID]
        if len(full) > stage_a.CONTEXT_LENGTH:
            raise RuntimeError(f"task exceeds context: {record['record_id']} ({len(full)})")
        record["_full"] = full
        record["_prefix_len"] = 1 + len(stage_a.serialize_bytes(prompt))
        prepared.append(record)
    return prepared


def write_ladder_manifest(output: Path, seed: int, records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "schema_version": "crucible_cognitive_learnability_ladder_v1",
        "seed": seed,
        "stages": list(STAGES),
        "splits": list(SPLITS),
        "records_per_stage": RECORDS_PER_STAGE,
        "test_evaluated": False,
        "task_roles": {
            "train": "in-distribution learning",
            "validation": "in-distribution checkpoint gate",
            "heldout": "depth/composition/distractor generalization diagnostic",
            "test": "untouched and not evaluated in this run",
        },
        "files": {},
    }
    for stage in STAGES:
        for split in SPLITS:
            path = output / "manifests" / stage / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [canonical_json(record) for record in records[stage][split]]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            manifest["files"][f"{stage}/{split}"] = {"path": str(path.relative_to(output)), "records": len(lines), "sha256": sha256_file(path)}
    write_json(output / "LADDER_MANIFEST.json", manifest)
    return manifest


def run_stage(seed: int, stage: str, stage_index: int, steps: int, output: Path, records: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    torch = stage_a.torch
    torch.manual_seed(seed)
    random.seed(seed)
    factory = lambda: stage_a.make_dense(seed)
    model = factory()
    optimizer = stage_a.optimizer_for(model)
    prepared = {split: prepare_records(values) for split, values in records.items()}
    order = list(range(len(prepared["train"])))
    random.Random(seed + 10_000 + stage_index).shuffle(order)
    stage_dir = output / "checkpoints" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_a.save_checkpoint(stage_dir / "step_0000.pt", model, optimizer, 0, order)
    evaluation_steps = {0, 100, 200, steps}
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(0, steps + 1):
        if step > 0:
            indices = stage_a.batch_indices(order, step)
            train_info = stage_a.train_step(model, optimizer, prepared["train"], indices)
        else:
            train_info = None
        if step in evaluation_steps:
            evaluations = {
                split: stage_a.evaluate(model, prepared[split])
                for split in ("train", "validation", "heldout")
            }
            row = {"stage": stage, "seed": seed, "step": step, "train_step": train_info, "evaluations": evaluations}
            if step == steps:
                checkpoint = stage_dir / f"step_{step:04d}.pt"
                stage_a.save_checkpoint(checkpoint, model, optimizer, step, order)
                row["checkpoint"] = str(checkpoint.relative_to(output))
            rows.append(row)
        if step > 0:
            print(json.dumps({"stage": stage, "seed": seed, "step": step, "loss": train_info["loss"]}, sort_keys=True), flush=True)
    final = rows[-1]["evaluations"]
    train_exact = final["train"]["teacher_forced_exact_accuracy"]
    validation_exact = final["validation"]["teacher_forced_exact_accuracy"]
    return {
        "stage": stage,
        "seed": seed,
        "steps": steps,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "final": final,
        "heldout_generalization_exact": final["heldout"]["teacher_forced_exact_accuracy"],
        "learnability_pass": train_exact >= 0.90 and validation_exact >= 0.70,
        "records": rows,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, choices=(200, 300), default=300)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = generate_ladder(args.seed)
    manifest = write_ladder_manifest(output, args.seed, records)
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_cognitive_learnability_ladder_20261022",
        "status": "cpu_only_dense_control",
        "seed": args.seed,
        "steps_per_stage": args.steps,
        "model_arm": "dense_active_compute_only",
        "parameter_count": 21968640,
        "test_evaluated": False,
        "selection_rule": "in_distribution validation only; heldout is diagnostic",
        "manifest_sha256": sha256_file(output / "LADDER_MANIFEST.json"),
    })
    results = []
    for stage_index, stage in enumerate(STAGES):
        result = run_stage(args.seed, stage, stage_index, args.steps, output, records[stage])
        results.append(result)
        if not result["learnability_pass"]:
            print(json.dumps({"stage": stage, "status": "ladder_stop", "reason": "learnability_gate_failed"}, sort_keys=True), flush=True)
            break
    result = {
        "experiment": "crucible_cognitive_learnability_ladder_20261022",
        "status": "completed",
        "seed": args.seed,
        "test_evaluated": False,
        "stopped_after": results[-1]["stage"] if results else None,
        "ladder_completed": len(results) == len(STAGES) and all(item["learnability_pass"] for item in results),
        "results": results,
        "interpretation": "dense-control learnability diagnostic only; no E4 comparison or cognitive-capability claim",
    }
    write_json(output / "RESULTS.json", result)
    (output / "RUNS.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for item in results for row in item["records"]) + "\n",
        encoding="utf-8",
    )
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"output": str(output), "status": result["status"], "ladder_completed": result["ladder_completed"], "stopped_after": result["stopped_after"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
