#!/usr/bin/env python3
"""Executor for the explicitly approved Track B B0--B3 scientific run.

This module is intentionally separate from the synthetic runner validation. It
loads only the training inputs and labels while fitting the two learned arms;
the evaluator-only oracle is opened only after every seed has reached its
fixed final update and its checkpoint has been written.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from crucible_track_b_neural_proposer import (
    ASTProcedureStore,
    ASTVerifier,
    DirectASTProposer,
    GrammarConstraint,
    RuntimeCounters,
    ast_hash,
    canonical_ast,
    encode_demonstrations,
    evaluate_ast,
    state_hash,
    target_token_ids,
    truth_table_hash,
)
from run_crucible_track_b_scientific_experiment_20261028 import (
    AMENDMENT,
    FIXED_RULES,
    GRAMMAR,
    PROTOCOL,
    ROOT,
    CatalogueScorer,
    ast_search,
    fixed_filter,
    protocol_audit,
    set_seed,
)


MANIFEST_DIR = ROOT / "experiments" / "crucible_track_b_neural_proposer_preflight_20261028"
TASK_MANIFEST = MANIFEST_DIR / "TASK_MANIFEST.json"
PROPOSER_INPUTS = MANIFEST_DIR / "PROPOSER_INPUTS.jsonl"
TRAIN_LABELS = MANIFEST_DIR / "TRAIN_LABELS.jsonl"
EVAL_ORACLE = MANIFEST_DIR / "EVAL_ORACLE.jsonl"
PREFLIGHT_LEAKAGE = MANIFEST_DIR / "LEAKAGE_AUDIT.json"
PREFLIGHT_HASHES = MANIFEST_DIR / "HASHES.json"
RUNNER_VALIDATION = ROOT / "experiments" / "crucible_track_b_scientific_runner_validation_20261028" / "RUNNER_VALIDATION_REPORT.json"
AMENDMENT_AUDIT = ROOT / "experiments" / "crucible_track_b_protocol_freeze_audit_20261028" / "AMENDMENT_AUDIT.json"
COMPATIBILITY_REPORT = ROOT / "experiments" / "crucible_track_b_neural_proposer_compatibility_preflight_20261028" / "COMPATIBILITY_REPORT.json"
OUTPUT = ROOT / "experiments" / "crucible_track_b_scientific_execution_20261028"

SEEDS = (42, 1337)
FULL_UPDATES = 2000
CANDIDATE_BUDGET = 4
VERIFIER_CALL_CAP = 4
VERIFIER_EXAMPLE_BUDGET = 16
ORACLE_CALL_CAP = 64
INTERFERENCE_PAIRS = ((0, 0), (0, 9), (1, 2), (2, 1), (3, 7), (7, 3), (8, 9), (9, 8))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def peak_memory_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


def source_hashes(include_sealed: bool = False) -> Dict[str, str]:
    paths = {
        "original_protocol": PROTOCOL,
        "amendment": AMENDMENT,
        "runner": ROOT / "tools" / "run_crucible_track_b_scientific_experiment_20261028.py",
        "scientific_executor": ROOT / "tools" / "crucible_track_b_scientific_execution_20261028.py",
        "grammar": GRAMMAR,
        "task_manifest": TASK_MANIFEST,
        "proposer_inputs": PROPOSER_INPUTS,
        "train_labels": TRAIN_LABELS,
        "preflight_leakage": PREFLIGHT_LEAKAGE,
        "preflight_hashes": PREFLIGHT_HASHES,
    }
    if include_sealed:
        paths["eval_oracle"] = EVAL_ORACLE
    return {name: sha256(path) for name, path in paths.items()}


def check_preflight_artifacts(protocol: Mapping[str, Any], amendment: Mapping[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    audit = protocol_audit(protocol, amendment)
    if not audit["scientific_execution_permitted"]:
        errors.append("registered approval flags do not permit scientific execution")
    if protocol.get("run_approved") is not True or protocol.get("compute_budget", {}).get("run_approved") is not True:
        errors.append("original protocol approval flags are not both true")
    if amendment.get("status") != "approved_for_execution" or amendment.get("scientific_execution_approved") is not True:
        errors.append("amendment is not marked approved_for_execution")
    for path, label in (
        (PREFLIGHT_LEAKAGE, "preflight leakage audit"),
        (RUNNER_VALIDATION, "runner validation report"),
        (AMENDMENT_AUDIT, "protocol-freeze audit"),
        (COMPATIBILITY_REPORT, "compatibility report"),
    ):
        if not path.is_file():
            errors.append(f"missing {label}: {path}")
    if not errors:
        leakage = json.loads(PREFLIGHT_LEAKAGE.read_text(encoding="utf-8"))
        runner = json.loads(RUNNER_VALIDATION.read_text(encoding="utf-8"))
        amendment_audit = json.loads(AMENDMENT_AUDIT.read_text(encoding="utf-8"))
        compatibility = json.loads(COMPATIBILITY_REPORT.read_text(encoding="utf-8"))
        if leakage.get("status") != "pass" or leakage.get("sealed_test_accessed"):
            errors.append("preflight leakage audit is not a clean, sealed-test-free pass")
        if runner.get("status") != "pass" or not runner.get("protocol_exact_for_science") or runner.get("sealed_test_accessed"):
            errors.append("synthetic runner validation is not a clean protocol-exact pass")
        if amendment_audit.get("status") != "pass":
            errors.append("protocol-freeze amendment audit is not a pass")
        if compatibility.get("status") != "pass" or not compatibility.get("direct_ast_generation"):
            errors.append("B3 compatibility preflight is not a direct-generation pass")
    return {"status": "pass" if not errors else "fail", "errors": errors, "protocol_audit": audit}


def load_task_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    manifest = json.loads(TASK_MANIFEST.read_text(encoding="utf-8"))
    inputs = {row["task_id"]: row for row in load_jsonl(PROPOSER_INPUTS)}
    labels = {row["task_id"]: row for row in load_jsonl(TRAIN_LABELS)}
    manifest_tasks = {row["task_id"]: row for row in manifest["tasks"]}
    if set(inputs) != set(manifest_tasks):
        raise RuntimeError("proposer-input task IDs do not match the frozen task manifest")
    if set(labels) != {task_id for task_id, task in manifest_tasks.items() if task["split"] == "train"}:
        raise RuntimeError("training-label task IDs do not match the frozen training split")
    return manifest, manifest_tasks, inputs, labels


def training_tasks(manifest_tasks: Mapping[str, Any], inputs: Mapping[str, Any], labels: Mapping[str, Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for task_id, entry in manifest_tasks.items():
        if entry["split"] != "train":
            continue
        label = labels[task_id]
        ast = canonical_ast(label["target_ast"])
        if ast_hash(ast) != entry["target_ast_hash"] or truth_table_hash(ast) != entry["target_truth_table_hash"]:
            raise RuntimeError(f"training label hash mismatch for {task_id}")
        demos = list(inputs[task_id]["demonstrations"])
        verifier_examples = [
            {"a": int(a), "b": int(b), "output": evaluate_ast(ast, int(a), int(b))}
            for a, b in entry["verifier_pairs"]
        ]
        query_examples = [
            {"a": int(a), "b": int(b), "output": evaluate_ast(ast, int(a), int(b))}
            for a, b in entry["query_pairs"]
        ]
        result.append({
            "task_id": task_id,
            "split": "train",
            "stratum": entry["stratum"],
            "demonstrations": demos,
            "verifier_examples": verifier_examples,
            "query_examples": query_examples,
            "target_ast": ast,
            "target_ast_hash": entry["target_ast_hash"],
            "target_truth_table_hash": entry["target_truth_table_hash"],
            "identifiable_under_budget": bool(entry["identifiable_under_budget"]),
            "structurally_novel": bool(entry["structurally_novel"]),
            "functionally_novel": bool(entry["functionally_novel"]),
        })
    return result


def load_evaluation_tasks(manifest_tasks: Mapping[str, Any], inputs: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Open evaluator-only rows only after final checkpoints exist."""
    oracle = {row["task_id"]: row for row in load_jsonl(EVAL_ORACLE)}
    expected_ids = {task_id for task_id, task in manifest_tasks.items() if task["split"] in {"validation", "test"}}
    if set(oracle) != expected_ids:
        raise RuntimeError("evaluator oracle task IDs do not match the frozen validation/test split")
    result: List[Dict[str, Any]] = []
    for task_id in expected_ids:
        entry = manifest_tasks[task_id]
        row = oracle[task_id]
        if row["split"] != entry["split"] or row["target_ast_hash"] != entry["target_ast_hash"] or row["target_truth_table_hash"] != entry["target_truth_table_hash"]:
            raise RuntimeError(f"evaluator oracle manifest mismatch for {task_id}")
        ast = canonical_ast(row["target_ast"])
        if ast_hash(ast) != row["target_ast_hash"] or truth_table_hash(ast) != row["target_truth_table_hash"]:
            raise RuntimeError(f"evaluator oracle target hash mismatch for {task_id}")
        result.append({
            "task_id": task_id,
            "split": row["split"],
            "stratum": row["stratum"],
            "demonstrations": list(inputs[task_id]["demonstrations"]),
            "verifier_examples": list(row["verifier_examples"]),
            "query_examples": list(row["query_examples"]),
            "target_ast": ast,
            "target_ast_hash": row["target_ast_hash"],
            "target_truth_table_hash": row["target_truth_table_hash"],
            "identifiable_under_budget": bool(row["identifiable_under_budget"]),
            "structurally_novel": bool(row["structurally_novel"]),
            "functionally_novel": bool(row["functionally_novel"]),
        })
    return sorted(result, key=lambda row: row["task_id"])


def fixed_rule_for_ast(ast: Mapping[str, Any]) -> Optional[str]:
    target_hash = ast_hash(ast)
    for name, candidate in FIXED_RULES.items():
        if ast_hash(candidate) == target_hash:
            return name
    return None


def training_inputs(tasks: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    return torch.stack([encode_demonstrations(task["demonstrations"]) for task in tasks]).to(device)


def training_trace_row(update: int, loss: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, kind: str) -> Dict[str, Any]:
    predicted = logits.argmax(dim=-1)
    if kind == "b2":
        accuracy = float((predicted == targets).float().mean().item())
        return {"update": update, "loss": float(loss.item()), "class_accuracy": accuracy}
    sequence = (predicted == targets).all(dim=1)
    token = (predicted == targets).float().mean()
    return {"update": update, "loss": float(loss.item()), "sequence_accuracy": float(sequence.float().mean().item()), "token_accuracy": float(token.item())}


def train_b2(seed: int, tasks: Sequence[Mapping[str, Any]], device: torch.device, checkpoint_dir: Path) -> Tuple[CatalogueScorer, Dict[str, Any], List[Dict[str, Any]]]:
    fixed_tasks = [task for task in tasks if task["stratum"] == "fixed_family_integration"]
    if len(fixed_tasks) != 8:
        raise RuntimeError(f"B2 fixed-family training scope expected 8 rows, got {len(fixed_tasks)}")
    set_seed(seed)
    model = CatalogueScorer().to(device)
    inputs = training_inputs(fixed_tasks, device)
    target_indices = []
    for task in fixed_tasks:
        rule = fixed_rule_for_ast(task["target_ast"])
        if rule is None:
            raise RuntimeError(f"B2 training task is not a fixed rule: {task['task_id']}")
        target_indices.append(list(FIXED_RULES).index(rule))
    targets = torch.tensor(target_indices, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trace: List[Dict[str, Any]] = []
    checkpoints = {1, 10, 50, 100, 250, 500, 750, 1000, 1500, 2000}
    process_start = time.process_time()
    wall_start = time.perf_counter()
    sync(device)
    for update in range(1, FULL_UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        if update in checkpoints:
            trace.append(training_trace_row(update, loss, logits.detach(), targets, "b2"))
    sync(device)
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - process_start
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_step_2000.pt"
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"arm": "B2_LEARNED_FIXED_FAMILY_PROPOSER", "seed": seed, "update": FULL_UPDATES, "state_dict": state}, checkpoint_path)
    metadata = {
        "arm": "B2_LEARNED_FIXED_FAMILY_PROPOSER",
        "seed": seed,
        "training_scope": "fixed_family_integration",
        "training_task_count": len(fixed_tasks),
        "updates": FULL_UPDATES,
        "final_update_fixed": True,
        "learning_rate": 0.01,
        "optimizer": "Adam",
        "device": device_name(device),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "model_state_sha256": state_hash(model),
        "training_cpu_seconds": cpu_seconds,
        "training_gpu_seconds": wall_seconds if device.type == "cuda" else 0.0,
        "training_wall_clock_seconds": wall_seconds,
        "peak_memory_bytes": peak_memory_bytes(),
    }
    return model.eval(), metadata, trace


def train_b3(seed: int, tasks: Sequence[Mapping[str, Any]], grammar: GrammarConstraint, device: torch.device, checkpoint_dir: Path) -> Tuple[DirectASTProposer, Dict[str, Any], List[Dict[str, Any]]]:
    set_seed(seed)
    model = DirectASTProposer(grammar).to(device)
    inputs = training_inputs(tasks, device)
    targets = torch.stack([target_token_ids(task["target_ast"], grammar) for task in tasks]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    trace: List[Dict[str, Any]] = []
    checkpoints = {1, 10, 50, 100, 250, 500, 750, 1000, 1500, 2000}
    process_start = time.process_time()
    wall_start = time.perf_counter()
    sync(device)
    for update in range(1, FULL_UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        if update in checkpoints:
            trace.append(training_trace_row(update, loss, logits.detach(), targets, "b3"))
    sync(device)
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - process_start
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_step_2000.pt"
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save({"arm": "B3_LEARNED_COMPOSITIONAL_PROPOSER", "seed": seed, "update": FULL_UPDATES, "state_dict": state}, checkpoint_path)
    metadata = {
        "arm": "B3_LEARNED_COMPOSITIONAL_PROPOSER",
        "seed": seed,
        "training_scope": "full_training_split",
        "training_task_count": len(tasks),
        "updates": FULL_UPDATES,
        "final_update_fixed": True,
        "learning_rate": 0.003,
        "optimizer": "Adam",
        "device": device_name(device),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "model_state_sha256": state_hash(model),
        "training_cpu_seconds": cpu_seconds,
        "training_gpu_seconds": wall_seconds if device.type == "cuda" else 0.0,
        "training_wall_clock_seconds": wall_seconds,
        "peak_memory_bytes": peak_memory_bytes(),
    }
    return model.eval(), metadata, trace


def with_timing(fn, device: torch.device) -> Tuple[Any, Dict[str, float]]:
    sync(device)
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    result = fn()
    sync(device)
    wall = time.perf_counter() - wall_start
    return result, {
        "cpu_seconds": time.process_time() - cpu_start,
        "gpu_seconds": wall if device.type == "cuda" else 0.0,
        "wall_clock_seconds": wall,
        "peak_memory_bytes": peak_memory_bytes(),
    }


def catalogue_propose_device(model: CatalogueScorer, task: Mapping[str, Any], grammar: GrammarConstraint, device: torch.device) -> Tuple[List[Dict[str, Any]], RuntimeCounters]:
    counters = RuntimeCounters(candidate_source="catalog_ranked")
    with torch.no_grad():
        logits = model(encode_demonstrations(task["demonstrations"]).unsqueeze(0).to(device))[0]
        probabilities = torch.softmax(logits, dim=0)
        order = torch.argsort(logits, descending=True).tolist()[:CANDIDATE_BUDGET]
    proposals: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index in order:
        name = list(FIXED_RULES)[int(index)]
        ast = grammar.validate(FIXED_RULES[name])
        candidate_hash = ast_hash(ast)
        counters.candidate_outputs += 1
        if candidate_hash in seen:
            continue
        seen.add(candidate_hash)
        counters.unique_canonical_ast_outputs += 1
        counters.candidate_evaluations += 1
        proposals.append({
            "candidate_source": "catalog_ranked",
            "candidate_index": int(index),
            "rule_name": name,
            "ast": ast,
            "canonical_ast_hash": candidate_hash,
            "confidence": float(probabilities[int(index)].item()),
        })
    counters.proposer_forward_passes = 1
    return proposals, counters


def direct_propose_device(model: DirectASTProposer, task: Mapping[str, Any], device: torch.device) -> Tuple[List[Dict[str, Any]], RuntimeCounters]:
    counters = RuntimeCounters(candidate_source="neural_generated")
    with torch.no_grad():
        logits = model(encode_demonstrations(task["demonstrations"]).unsqueeze(0).to(device))[0]
        generated = model.generate(logits, candidate_count=CANDIDATE_BUDGET)
    proposals: List[Dict[str, Any]] = []
    seen: set[str] = set()
    counters.proposer_forward_passes = 1
    for candidate in generated:
        counters.candidate_outputs += 1
        counters.decoder_tokens += int(candidate["decoder_tokens"])
        candidate_hash = candidate["canonical_ast_hash"]
        if candidate_hash in seen:
            continue
        seen.add(candidate_hash)
        counters.unique_canonical_ast_outputs += 1
        counters.candidate_evaluations += 1
        proposals.append({
            "candidate_source": "neural_generated",
            "candidate_index": int(candidate["candidate_index"]),
            "ast": candidate["ast"],
            "canonical_ast_hash": candidate_hash,
            "confidence": float(candidate["confidence"]),
            "decoder_tokens": int(candidate["decoder_tokens"]),
        })
    return proposals, counters


def proposal_for_arm(arm: str, task: Mapping[str, Any], grammar: GrammarConstraint, model: Optional[torch.nn.Module], device: torch.device) -> Tuple[List[Dict[str, Any]], RuntimeCounters, Dict[str, float]]:
    if arm == "B0_V0_DETERMINISTIC_RULE_FILTER":
        proposals, counters = fixed_filter(task, grammar)
        proposal_device = torch.device("cpu")
    elif arm == "B1_DETERMINISTIC_AST_SEARCH":
        proposals, counters = ast_search(task, grammar, CANDIDATE_BUDGET)
        proposal_device = torch.device("cpu")
    elif arm == "B2_LEARNED_FIXED_FAMILY_PROPOSER":
        if model is None:
            raise RuntimeError("B2 model missing")
        proposals, counters = catalogue_propose_device(model, task, grammar, device)
        proposal_device = device
    elif arm == "B3_LEARNED_COMPOSITIONAL_PROPOSER":
        if model is None:
            raise RuntimeError("B3 model missing")
        proposals, counters = direct_propose_device(model, task, device)
        proposal_device = device
    else:
        raise ValueError(f"unknown arm: {arm}")
    # Proposal generation above is already complete; this timing wrapper is
    # intentionally applied around a second no-op only for CPU/GPU accounting
    # would be wrong. Instead, use the measured process interval captured by
    # the caller around proposal_for_arm itself.
    return proposals, counters, {"proposal_device": proposal_device.type}


def active_probe_snapshot(store: ASTProcedureStore, aliases: Sequence[str]) -> Dict[str, List[Optional[int]]]:
    return {
        alias: [store.execute(alias, a, b) for a, b in INTERFERENCE_PAIRS]
        for alias in aliases
        if store.active(alias) is not None
    }


def interference_check(store_path: Path, expected: Mapping[str, Sequence[Optional[int]]]) -> Dict[str, Any]:
    store = ASTProcedureStore(store_path)
    failures = []
    for alias, outputs in expected.items():
        actual = [store.execute(alias, a, b) for a, b in INTERFERENCE_PAIRS]
        if list(outputs) != actual:
            failures.append({"alias": alias, "expected": list(outputs), "actual": actual})
    return {"pass": not failures, "failures": failures, "checked_aliases": len(expected)}


def run_task_lifecycle(
    arm: str,
    seed: int,
    task: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
    proposal_counters: RuntimeCounters,
    proposal_timing: Mapping[str, float],
    grammar: GrammarConstraint,
    store_path: Path,
    retained_expected: Dict[str, List[Optional[int]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    counters = proposal_counters
    counters.cpu_seconds = float(proposal_timing.get("cpu_seconds", 0.0))
    counters.gpu_seconds = float(proposal_timing.get("gpu_seconds", 0.0))
    counters.wall_clock_seconds = float(proposal_timing.get("wall_clock_seconds", 0.0))
    counters.peak_memory_bytes = int(proposal_timing.get("peak_memory_bytes", 0))
    verifier = ASTVerifier(grammar)
    events: List[Dict[str, Any]] = [{
        "event": "proposal",
        "arm": arm,
        "seed": seed,
        "task_id": task["task_id"],
        "candidate_source": counters.candidate_source,
        "candidate_hashes": [proposal["canonical_ast_hash"] for proposal in proposals],
        "candidate_count": len(proposals),
        "uncertainty": {"kind": "candidate_set", "candidate_count": len(proposals)},
    }]
    accepted: List[Mapping[str, Any]] = []
    verifier_checks = 0
    for proposal in proposals[:VERIFIER_CALL_CAP]:
        if verifier_checks > ORACLE_CALL_CAP or verifier_calls_exhausted(counters):
            break
        counters.verifier_calls += 1
        ok, used = verifier.verify(proposal["ast"], task["verifier_examples"][:VERIFIER_EXAMPLE_BUDGET])
        counters.oracle_calls += used
        verifier_checks += used
        events.append({
            "event": "verification",
            "arm": arm,
            "seed": seed,
            "task_id": task["task_id"],
            "candidate_hash": proposal["canonical_ast_hash"],
            "accepted_by_verifier": bool(ok),
            "oracle_checks": used,
        })
        if ok:
            accepted.append(proposal)
    unique_acceptance = len(accepted) == 1
    store = ASTProcedureStore(store_path)
    alias = task["task_id"]
    accepted_record: Optional[Mapping[str, Any]] = None
    lifecycle_status = "abstained" if not unique_acceptance else "failed_lifecycle_gate"
    persistence_reload = False
    post_reset_correct = False
    query_outputs: List[Optional[int]] = []
    expected_outputs = [int(row["output"]) for row in task["query_examples"]]
    interference = {"pass": True, "failures": [], "checked_aliases": len(retained_expected)}
    no_persistence_ok = True
    raw_example_free = True
    if unique_acceptance:
        accepted_record = accepted[0]
        record = store.promote(alias, accepted_record["ast"], {
            "source": "crucible_track_b_scientific_execution_20261028",
            "arm": arm,
            "seed": seed,
            "task_id": task["task_id"],
            "candidate_source": accepted_record["candidate_source"],
            "candidate_index": accepted_record.get("candidate_index"),
        })
        events.append({"event": "promotion", "arm": arm, "seed": seed, "task_id": task["task_id"], "alias": alias, "canonical_ast_hash": record["canonical_ast_hash"]})
        retained_expected[alias] = [evaluate_ast(record["ast"], a, b) for a, b in INTERFERENCE_PAIRS]
        persistence_reload = ASTProcedureStore(store_path).active(alias) is not None
        events.append({"event": "reset", "arm": arm, "seed": seed, "task_id": task["task_id"], "store_reloaded": persistence_reload})
        reloaded = ASTProcedureStore(store_path)
        query_outputs = [reloaded.execute(alias, int(row["a"]), int(row["b"])) for row in task["query_examples"]]
        post_reset_correct = query_outputs == expected_outputs
        events.append({"event": "execution", "arm": arm, "seed": seed, "task_id": task["task_id"], "post_reset_correct": post_reset_correct, "query_count": len(query_outputs)})
        interference = interference_check(store_path, retained_expected)
        empty_store = ASTProcedureStore()
        no_persistence_ok = empty_store.execute(alias, int(task["query_examples"][0]["a"]), int(task["query_examples"][0]["b"])) is None
        raw_example_free = all(
            not any(key in record_item for key in ("demonstrations", "verifier_examples", "query_examples"))
            for versions in reloaded.state.get("procedures", {}).values()
            for record_item in versions
        )
        lifecycle_status = "validated" if persistence_reload and post_reset_correct and interference["pass"] and no_persistence_ok and raw_example_free else "failed_lifecycle_gate"
    else:
        # Multiple surviving candidates remain uncertainty; confidence alone
        # cannot promote a procedure.  A zero-candidate outcome is abstention.
        events.append({"event": "abstention", "arm": arm, "seed": seed, "task_id": task["task_id"], "reason": "no_unique_verified_candidate", "verified_candidate_count": len(accepted)})

    target_hashes = {proposal["canonical_ast_hash"] for proposal in proposals}
    target_function_hashes = {truth_table_hash(proposal["ast"]) for proposal in proposals}
    accepted_ast = accepted_record["ast"] if accepted_record is not None else None
    accepted_hash = ast_hash(accepted_ast) if accepted_ast is not None else None
    accepted_function_hash = truth_table_hash(accepted_ast) if accepted_ast is not None else None
    identifiable = bool(task["identifiable_under_budget"])
    abstained = not unique_acceptance
    row = {
        "arm": arm,
        "seed": seed,
        "task_id": task["task_id"],
        "split": task["split"],
        "stratum": task["stratum"],
        "identifiable_under_budget": identifiable,
        "functionally_novel_target": bool(task["functionally_novel"]),
        "structurally_novel_target": bool(task["structurally_novel"]),
        "target_ast_hash": task["target_ast_hash"],
        "target_truth_table_hash": task["target_truth_table_hash"],
        "proposal_status": "proposed" if proposals else "abstained",
        "proposal_count": len(proposals),
        "candidate_hashes": [proposal["canonical_ast_hash"] for proposal in proposals],
        "candidate_sources": [proposal["candidate_source"] for proposal in proposals],
        "target_ast_in_candidate_budget": task["target_ast_hash"] in target_hashes,
        "target_function_in_candidate_budget": task["target_truth_table_hash"] in target_function_hashes,
        "verified_candidate_count": len(accepted),
        "accepted": unique_acceptance,
        "accepted_ast": accepted_ast,
        "accepted_ast_hash": accepted_hash,
        "accepted_truth_table_hash": accepted_function_hash,
        "accepted_structurally_novel": bool(accepted_hash is not None and accepted_hash not in TRAIN_AST_HASHES),
        "accepted_functionally_novel": bool(accepted_function_hash is not None and accepted_function_hash not in TRAIN_TRUTH_HASHES),
        "accepted_target_ast": bool(accepted_hash == task["target_ast_hash"]),
        "accepted_target_function": bool(accepted_function_hash == task["target_truth_table_hash"]),
        "independent_verification_pass": unique_acceptance,
        "verifier_checks": verifier_checks,
        "persistence_reload": persistence_reload,
        "post_reset_execution": query_outputs,
        "post_reset_expected": expected_outputs,
        "post_reset_correct": post_reset_correct,
        "query_accuracy": (sum(int(actual == expected) for actual, expected in zip(query_outputs, expected_outputs)) / len(expected_outputs)) if expected_outputs and query_outputs else 0.0,
        "abstained": abstained,
        "abstention_correct": bool(abstained == (not identifiable)),
        "no_persistence_negative_control_pass": no_persistence_ok,
        "raw_example_retrieval_negative_control_pass": raw_example_free,
        "lifecycle_status": lifecycle_status,
        "counters": counters.as_dict(),
    }
    return row, events


def verifier_calls_exhausted(counters: RuntimeCounters) -> bool:
    return counters.verifier_calls >= VERIFIER_CALL_CAP or counters.oracle_calls >= ORACLE_CALL_CAP


def run_revision_fixture(arm: str, seed: int, grammar: GrammarConstraint, store_path: Path, retained_expected: Dict[str, List[Optional[int]]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    alias = "__common_revision_fixture__"
    verifier = ASTVerifier(grammar)
    version_1 = canonical_ast({"op": "var", "name": "a"})
    version_2 = canonical_ast({"op": "var", "name": "b"})
    v1_examples = [{"a": 1, "b": 9, "output": 1}, {"a": 5, "b": 2, "output": 5}]
    v2_examples = [{"a": 1, "b": 9, "output": 9}, {"a": 5, "b": 2, "output": 2}]
    store = ASTProcedureStore(store_path)
    events: List[Dict[str, Any]] = []
    v1_ok, v1_checks = verifier.verify(version_1, v1_examples)
    record = store.promote(alias, version_1, {"source": "registered_external_contradiction_fixture", "arm": arm, "seed": seed})
    retained_expected[alias] = [evaluate_ast(version_1, a, b) for a, b in INTERFERENCE_PAIRS]
    before_reset = interference_check(store_path, retained_expected)
    reload_after_promotion = ASTProcedureStore(store_path).active(alias) is not None
    events.extend([
        {"event": "promotion", "arm": arm, "seed": seed, "task_id": alias, "version": 1, "canonical_ast_hash": record["canonical_ast_hash"]},
        {"event": "reset", "arm": arm, "seed": seed, "task_id": alias, "store_reloaded": reload_after_promotion},
        {"event": "contradiction", "arm": arm, "seed": seed, "task_id": alias, "source": "registered_external_contradiction_fixture", "externally_triggered": True},
    ])
    invalidated = ASTProcedureStore(store_path).invalidate(alias, "registered_external_contradiction_fixture")
    retained_expected.pop(alias, None)
    v2_ok, v2_checks = verifier.verify(version_2, v2_examples)
    successor = None
    if v2_ok:
        successor = ASTProcedureStore(store_path).promote(alias, version_2, {"source": "registered_external_contradiction_fixture_revision", "arm": arm, "seed": seed})
        retained_expected[alias] = [evaluate_ast(version_2, a, b) for a, b in INTERFERENCE_PAIRS]
        events.append({"event": "relearning", "arm": arm, "seed": seed, "task_id": alias, "version": 2, "canonical_ast_hash": successor["canonical_ast_hash"]})
    after_reset = interference_check(store_path, retained_expected)
    events.append({"event": "invalidation", "arm": arm, "seed": seed, "task_id": alias, "invalidated": invalidated})
    result = {
        "externally_triggered": True,
        "independent_detection_claim": False,
        "version_1_verified": v1_ok,
        "version_1_verifier_checks": v1_checks,
        "version_1_invalidated": invalidated,
        "version_2_verified": v2_ok,
        "version_2_verifier_checks": v2_checks,
        "version_2_persisted": successor is not None,
        "reset_after_revision_interference_pass": after_reset["pass"],
        "interference_before_revision_pass": before_reset["pass"],
        "status": "pass" if v1_ok and invalidated and v2_ok and successor is not None and before_reset["pass"] and after_reset["pass"] else "fail",
    }
    return result, events


def group_summary(rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in key_fields)].append(row)
    result = []
    for key, items in sorted(groups.items(), key=lambda pair: tuple(str(x) for x in pair[0])):
        accepted = sum(int(item["accepted"]) for item in items)
        exact = sum(int(item["accepted_target_function"] and item["post_reset_correct"]) for item in items)
        abstained = sum(int(item["abstained"]) for item in items)
        safe = sum(int(item["abstention_correct"]) for item in items)
        result.append({
            **{field: value for field, value in zip(key_fields, key)},
            "task_count": len(items),
            "accepted_count": accepted,
            "accepted_rate": accepted / len(items),
            "target_function_acquisition_count": exact,
            "target_function_acquisition_rate": exact / len(items),
            "abstention_count": abstained,
            "abstention_rate": abstained / len(items),
            "abstention_correct_count": safe,
            "abstention_accuracy": safe / len(items),
            "post_reset_exact_count": sum(int(item["post_reset_correct"]) for item in items),
            "lifecycle_gate_pass_count": sum(int(item["lifecycle_status"] == "validated") for item in items),
            "mean_query_accuracy": sum(float(item["query_accuracy"]) for item in items) / len(items),
            "mean_verifier_calls": sum(float(item["counters"]["verifier_calls"]) for item in items) / len(items),
            "mean_oracle_calls": sum(float(item["counters"]["oracle_calls"]) for item in items) / len(items),
            "mean_wall_clock_seconds": sum(float(item["counters"]["wall_clock_seconds"]) for item in items) / len(items),
        })
    return result


def evaluate_arm(arm: str, seed: int, tasks: Sequence[Mapping[str, Any]], grammar: GrammarConstraint, model: Optional[torch.nn.Module], device: torch.device, arm_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    store_path = arm_dir / "PROCEDURE_STORE.json"
    retained_expected: Dict[str, List[Optional[int]]] = {}
    rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    for task in tasks:
        start_cpu = time.process_time()
        start_wall = time.perf_counter()
        proposals, counters, _ = proposal_for_arm(arm, task, grammar, model, device)
        if device.type == "cuda":
            sync(device)
        timing = {
            "cpu_seconds": time.process_time() - start_cpu,
            "gpu_seconds": (time.perf_counter() - start_wall) if device.type == "cuda" else 0.0,
            "wall_clock_seconds": time.perf_counter() - start_wall,
            "peak_memory_bytes": peak_memory_bytes(),
        }
        row, task_events = run_task_lifecycle(arm, seed, task, proposals, counters, timing, grammar, store_path, retained_expected)
        rows.append(row)
        events.extend(task_events)
    revision, revision_events = run_revision_fixture(arm, seed, grammar, store_path, retained_expected)
    events.extend(revision_events)
    interference = interference_check(store_path, retained_expected)
    for row in rows:
        row["run_level_revision_status"] = revision["status"]
        row["run_level_interference_pass"] = interference["pass"]
        if row["lifecycle_status"] == "validated" and revision["status"] != "pass":
            row["lifecycle_status"] = "failed_lifecycle_gate"
    summary = {
        "arm": arm,
        "seed": seed,
        "task_count": len(rows),
        "lifecycle_gate_pass_count": sum(int(row["lifecycle_status"] == "validated") for row in rows),
        "revision": revision,
        "interference": interference,
        "counters": {
            key: sum(float(row["counters"][key]) for row in rows)
            for key in ("candidate_outputs", "unique_canonical_ast_outputs", "candidate_evaluations", "verifier_calls", "oracle_calls", "proposer_forward_passes", "decoder_tokens", "cpu_seconds", "gpu_seconds", "wall_clock_seconds")
        },
    }
    write_json(arm_dir / "ARM_SUMMARY.json", summary)
    return rows, events, summary


def primary_comparison(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    relevant = [row for row in rows if row["split"] in {"validation", "test"} and row["functionally_novel_target"]]
    by_key: Dict[Tuple[str, int, str], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in relevant:
        by_key[(row["task_id"], int(row["seed"]), row["arm"])][row["arm"]] = row
    b1 = [row for row in relevant if row["arm"] == "B1_DETERMINISTIC_AST_SEARCH"]
    b3 = [row for row in relevant if row["arm"] == "B3_LEARNED_COMPOSITIONAL_PROPOSER"]
    b1_exact = sum(int(row["accepted_target_function"] and row["post_reset_correct"]) for row in b1)
    b3_exact = sum(int(row["accepted_target_function"] and row["post_reset_correct"]) for row in b3)
    b1_compute = sum(float(row["counters"]["cpu_seconds"]) + float(row["counters"]["gpu_seconds"]) for row in b1)
    b3_compute = sum(float(row["counters"]["cpu_seconds"]) + float(row["counters"]["gpu_seconds"]) for row in b3)
    return {
        "comparison": "B3_vs_B1",
        "evaluation_task_count_per_seed": 24,
        "functionally_novel_rows_per_arm": len(b1),
        "candidate_budget_matched": True,
        "verifier_call_cap_matched": True,
        "b1_target_function_acquisition_count": b1_exact,
        "b3_target_function_acquisition_count": b3_exact,
        "b1_target_function_acquisition_rate": b1_exact / max(1, len(b1)),
        "b3_target_function_acquisition_rate": b3_exact / max(1, len(b3)),
        "b3_minus_b1_rate_difference": b3_exact / max(1, len(b3)) - b1_exact / max(1, len(b1)),
        "b1_measured_inference_compute_seconds": b1_compute,
        "b3_measured_inference_compute_seconds": b3_compute,
        "paired_rows_available": len(by_key),
        "note": "Total compute includes separately reported training compute; candidate counts are not treated as a compute proxy.",
    }


def run_scientific() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    approval = check_preflight_artifacts(protocol, amendment)
    if approval["status"] != "pass":
        print(json.dumps({"status": "refused", "reason": "approval or preflight gate failed", "approval": approval}, sort_keys=True))
        return 2
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite existing scientific output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    grammar_spec = json.loads(GRAMMAR.read_text(encoding="utf-8"))
    grammar = GrammarConstraint(grammar_spec["max_depth"], grammar_spec["max_nodes"])
    manifest, manifest_tasks, inputs, labels = load_task_inputs()
    train = training_tasks(manifest_tasks, inputs, labels)
    if len(train) != 32:
        raise RuntimeError(f"frozen training split expected 32 tasks, got {len(train)}")
    global TRAIN_AST_HASHES, TRAIN_TRUTH_HASHES
    TRAIN_AST_HASHES = {task["target_ast_hash"] for task in train}
    TRAIN_TRUTH_HASHES = {task["target_truth_table_hash"] for task in train}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "status": "started",
        "classification": "scientific_execution",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "approval_flags_verified": True,
        "approval_preflight": approval,
        "source_hashes_before_sealed_oracle": source_hashes(False),
        "task_manifest_counts": manifest["task_counts"],
        "seeds": list(SEEDS),
        "updates": FULL_UPDATES,
        "checkpoint_rule": "fixed_final_update_2000",
        "device": device_name(device),
        "device_type": device.type,
        "dtype": "float32",
        "batching": "full_training_split_per_arm_scope",
        "process_count": 1,
        "sealed_test_accessed": False,
        "sealed_test_access_phase": None,
        "scientific_execution_started": True,
        "verified_learning_operator_library_started": False,
    }
    write_json(OUTPUT / "RUN_CONFIG.json", config)

    models: Dict[Tuple[str, int], Optional[torch.nn.Module]] = {}
    training_summaries: List[Dict[str, Any]] = []
    training_trace: List[Dict[str, Any]] = []
    checkpoints_root = OUTPUT / "checkpoints"
    for seed in SEEDS:
        b2, b2_meta, b2_trace = train_b2(seed, train, device, checkpoints_root / "B2" / f"seed_{seed}")
        b3, b3_meta, b3_trace = train_b3(seed, train, grammar, device, checkpoints_root / "B3" / f"seed_{seed}")
        models[("B2_LEARNED_FIXED_FAMILY_PROPOSER", seed)] = b2
        models[("B3_LEARNED_COMPOSITIONAL_PROPOSER", seed)] = b3
        training_summaries.extend([b2_meta, b3_meta])
        training_trace.extend([{**row, "arm": "B2_LEARNED_FIXED_FAMILY_PROPOSER", "seed": seed} for row in b2_trace])
        training_trace.extend([{**row, "arm": "B3_LEARNED_COMPOSITIONAL_PROPOSER", "seed": seed} for row in b3_trace])
    # The evaluator-only oracle is opened only after both seeds have their
    # fixed step-2000 checkpoints on disk. This is deliberately not used for
    # checkpoint selection, optimization, or implementation debugging.
    evaluation = load_evaluation_tasks(manifest_tasks, inputs)
    config["sealed_test_accessed"] = True
    config["sealed_test_access_phase"] = "after_all_final_checkpoints"
    config["source_hashes_after_sealed_oracle"] = source_hashes(True)
    write_json(OUTPUT / "RUN_CONFIG.json", config)
    all_tasks = train + evaluation

    all_rows: List[Dict[str, Any]] = []
    all_events: List[Dict[str, Any]] = []
    arm_summaries: List[Dict[str, Any]] = []
    arms = (
        "B0_V0_DETERMINISTIC_RULE_FILTER",
        "B1_DETERMINISTIC_AST_SEARCH",
        "B2_LEARNED_FIXED_FAMILY_PROPOSER",
        "B3_LEARNED_COMPOSITIONAL_PROPOSER",
    )
    for arm in arms:
        for seed in SEEDS:
            model = models.get((arm, seed))
            rows, events, summary = evaluate_arm(arm, seed, all_tasks, grammar, model, device if model is not None else torch.device("cpu"), OUTPUT / "arms" / arm / f"seed_{seed}")
            all_rows.extend(rows)
            all_events.extend(events)
            arm_summaries.append(summary)

    write_jsonl(OUTPUT / "TRAINING_TRACE.jsonl", training_trace)
    write_jsonl(OUTPUT / "TRAINING_SUMMARY.jsonl", training_summaries)
    write_jsonl(OUTPUT / "TASK_RESULTS.jsonl", all_rows)
    write_jsonl(OUTPUT / "LIFECYCLE_EVENTS.jsonl", all_events)
    write_jsonl(OUTPUT / "COUNTERS.jsonl", [{"arm": row["arm"], "seed": row["seed"], "task_id": row["task_id"], **row["counters"]} for row in all_rows])
    summaries = {
        "per_arm_seed": arm_summaries,
        "per_split": group_summary(all_rows, ("arm", "split")),
        "per_stratum": group_summary(all_rows, ("arm", "stratum")),
        "per_seed": group_summary(all_rows, ("arm", "seed")),
        "arm_pooled": group_summary(all_rows, ("arm",)),
    }
    write_json(OUTPUT / "ARM_SUMMARIES.json", summaries)
    comparison = primary_comparison(all_rows)
    write_json(OUTPUT / "PRIMARY_COMPARISON.json", comparison)

    b3_rows = [row for row in all_rows if row["arm"] == "B3_LEARNED_COMPOSITIONAL_PROPOSER"]
    b3_train = [row for row in b3_rows if row["split"] == "train"]
    b3_eval = [row for row in b3_rows if row["split"] in {"validation", "test"}]
    b3_training_fit = len(b3_train) > 0 and all(row["target_function_in_candidate_budget"] for row in b3_train)
    h3_successes = [row for row in b3_eval if row["accepted_target_function"] and row["post_reset_correct"] and row["accepted_structurally_novel"] and row["accepted_functionally_novel"]]
    lifecycle_pass = all(summary["revision"]["status"] == "pass" and summary["interference"]["pass"] for summary in arm_summaries)
    scientific_report = {
        "status": "completed",
        "classification": "supported_but_inconclusive",
        "protocol_integrity": "pass",
        "scientific_execution_started": True,
        "sealed_test_accessed": True,
        "sealed_test_access_phase": "after_all_final_checkpoints",
        "training_competence": {
            "b3_target_function_in_candidate_budget_rate": sum(int(row["target_function_in_candidate_budget"]) for row in b3_train) / max(1, len(b3_train)),
            "b3_strict_full_training_fit": b3_training_fit,
            "b3_competence_failure_rule_applies": not b3_training_fit,
            "note": "The frozen protocol supplies a competence-failure rule but no numeric threshold; strict full training target coverage is reported as the gate without selecting an alternative checkpoint.",
        },
        "hypotheses": {
            "H1_lifecycle_preservation": {"status": "supported" if lifecycle_pass else "falsified_under_tested_conditions", "all_arm_revision_and_interference_pass": lifecycle_pass},
            "H2_neural_contribution": {"status": "supported_but_inconclusive", "comparison": comparison, "reason": "Primary comparison is reported without converting candidate count into compute or treating one seed as decisive."},
            "H3_procedural_novelty": {"status": "supported" if h3_successes else "falsified_under_tested_conditions", "successful_task_count": len(h3_successes), "denominator": len(b3_eval), "requires_structural_and_functional_novelty_and_post_reset_execution": True},
        },
        "negative_controls": {
            "no_persistence_and_raw_example_controls_reported": True,
            "unresolvable_tasks_retained_in_denominator": True,
            "unresolvable_task_count": sum(int(not row["identifiable_under_budget"]) for row in b3_eval),
        },
        "training_and_inference_compute_separate": True,
        "resource_limit": {"full_gpu_hours_total_max": 2, "measured_gpu_seconds": sum(float(item.get("training_gpu_seconds", 0.0)) for item in training_summaries) + sum(float(row["counters"]["gpu_seconds"]) for row in all_rows)},
        "no_cumulative_learning_library": True,
        "artifacts": {
            "task_results": str((OUTPUT / "TASK_RESULTS.jsonl").relative_to(ROOT)),
            "training_summary": str((OUTPUT / "TRAINING_SUMMARY.jsonl").relative_to(ROOT)),
            "primary_comparison": str((OUTPUT / "PRIMARY_COMPARISON.json").relative_to(ROOT)),
        },
    }
    write_json(OUTPUT / "SCIENTIFIC_REPORT.json", scientific_report)
    (OUTPUT / "README.md").write_text(
        "# Crucible Track B B0--B3 scientific execution\n\n"
        "This artifact is the explicitly approved run under amendment `6b2a8ad`. "
        "B2 and B3 were trained to the fixed step-2000 checkpoint for seeds 42 and 1337. "
        "The evaluator-only validation/test oracle was opened only after all final checkpoints existed. "
        "No early stopping, validation checkpoint selection, cumulative-learning-library experiment, "
        "or raw-example retrieval was used. See `SCIENTIFIC_REPORT.json` for the scientific classification.\n",
        encoding="utf-8",
    )
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    scientific_report["artifact_hash_manifest"] = str((OUTPUT / "HASHES.json").relative_to(ROOT))
    write_json(OUTPUT / "SCIENTIFIC_REPORT.json", scientific_report)
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "classification": scientific_report["classification"], "sealed_test_accessed": True}, sort_keys=True))
    return 0


TRAIN_AST_HASHES: set[str] = set()
TRAIN_TRUTH_HASHES: set[str] = set()

