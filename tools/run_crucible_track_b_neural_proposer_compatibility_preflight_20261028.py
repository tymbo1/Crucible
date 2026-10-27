#!/usr/bin/env python3
"""Run the training-only 50-update Track B B3 compatibility preflight.

This is an implementation/compatibility check, not a scientific run. It uses
only the registered training rows and labels, runs on CPU, performs no
validation/test scoring, and never reads EVAL_ORACLE.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import random
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from crucible_track_b_neural_proposer import (
    ASTProcedureStore,
    ASTVerifier,
    DirectASTProposer,
    GrammarConstraint,
    RuntimeCounters,
    ast_hash,
    encode_demonstrations,
    evaluate_ast,
    state_hash,
    target_token_ids,
    truth_table_hash,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments" / "crucible_track_b_neural_proposer_preflight_20261028"
OUTPUT = ROOT / "experiments" / "crucible_track_b_neural_proposer_compatibility_preflight_20261028"
UPDATES = 50
SEED = 42
CANDIDATE_COUNT = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(name: str) -> Any:
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def load_jsonl(name: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in (PACKAGE / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def current_peak_memory_bytes() -> int:
    # Linux reports ru_maxrss in KiB; this runner is intended for the local
    # Linux research environment. The fallback keeps the counter explicit.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def load_training_records() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    inputs = load_jsonl("PROPOSER_INPUTS.jsonl")
    labels = {row["task_id"]: row for row in load_jsonl("TRAIN_LABELS.jsonl")}
    manifest = load_json("TASK_MANIFEST.json")
    # Keep the compatibility runner's working manifest explicitly training
    # only. Held-out task metadata is not needed for this gate and must not
    # influence implementation changes.
    manifest_by_id = {row["task_id"]: row for row in manifest["tasks"] if row.get("split") == "train"}
    train_inputs = [row for row in inputs if row.get("split") == "train"]
    if not train_inputs:
        raise RuntimeError("no training-only proposer inputs found")
    if any(row.get("split") != "train" for row in train_inputs):
        raise RuntimeError("non-training proposer input entered the compatibility set")
    missing_labels = sorted({row["task_id"] for row in train_inputs} - set(labels))
    if missing_labels:
        raise RuntimeError(f"missing training labels: {missing_labels}")
    missing_manifest = sorted({row["task_id"] for row in train_inputs} - set(manifest_by_id))
    if missing_manifest:
        raise RuntimeError(f"missing training manifest rows: {missing_manifest}")
    return train_inputs, labels, manifest_by_id


def make_batch(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    grammar: GrammarConstraint,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = torch.stack([encode_demonstrations(row["demonstrations"]) for row in rows])
    targets = torch.stack([target_token_ids(labels[row["task_id"]]["target_ast"], grammar) for row in rows])
    return encoded, targets


def training_trace_row(update: int, loss: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
    predictions = logits.argmax(dim=-1)
    return {
        "update": update,
        "loss": float(loss.detach().cpu()),
        "token_accuracy": float((predictions == targets).float().mean().detach().cpu()),
        "sequence_accuracy": float((predictions == targets).all(dim=1).float().mean().detach().cpu()),
    }


def run_once(
    seed: int,
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    manifest_by_id: Mapping[str, Mapping[str, Any]],
    grammar: GrammarConstraint,
) -> Dict[str, Any]:
    set_seed(seed)
    model = DirectASTProposer(grammar)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    encoded, targets = make_batch(rows, labels, grammar)
    trace: List[Dict[str, Any]] = []
    training_forward_passes = 0
    training_start = time.perf_counter()
    for update in range(1, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(encoded)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        training_forward_passes += 1
        trace.append(training_trace_row(update, loss, logits, targets))
    training_wall = time.perf_counter() - training_start

    verifier = ASTVerifier(grammar)
    proposals: List[Dict[str, Any]] = []
    task_counters: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            task_id = str(row["task_id"])
            task_start_wall = time.perf_counter()
            task_start_cpu = time.process_time()
            encoded_task = encode_demonstrations(row["demonstrations"]).unsqueeze(0)
            logits = model(encoded_task)[0]
            candidates = model.generate(logits, candidate_count=CANDIDATE_COUNT)
            task = manifest_by_id[task_id]
            target_ast = labels[task_id]["target_ast"]
            verifier_examples = [
                {
                    "a": int(pair[0]),
                    "b": int(pair[1]),
                    "output": evaluate_ast(target_ast, int(pair[0]), int(pair[1])),
                }
                for pair in task["verifier_pairs"]
            ]
            counters = RuntimeCounters()
            counters.proposer_forward_passes = 1
            counters.candidate_outputs = len(candidates)
            counters.unique_canonical_ast_outputs = len({candidate["canonical_ast_hash"] for candidate in candidates})
            counters.decoder_tokens = sum(candidate["decoder_tokens"] for candidate in candidates)
            for candidate in candidates:
                counters.candidate_evaluations += 1
                counters.verifier_calls += 1
                accepted, checks = verifier.verify(candidate["ast"], verifier_examples)
                counters.oracle_calls += checks
                proposal = {
                    "task_id": task_id,
                    "candidate_index": candidate["candidate_index"],
                    "candidate_source": "neural_generated",
                    "ast": candidate["ast"],
                    "canonical_ast_hash": candidate["canonical_ast_hash"],
                    "confidence": candidate["confidence"],
                    "uncertainty": {"kind": "candidate_set", "candidate_count": len(candidates)},
                    "grammar_valid": True,
                    "training_only_verification_pass": accepted,
                    "training_only_verifier_checks": checks,
                    "provenance": {"seed": seed, "updates": UPDATES, "source": "compatibility_preflight"},
                }
                proposals.append(proposal)
            counters.cpu_seconds = time.process_time() - task_start_cpu
            counters.wall_clock_seconds = time.perf_counter() - task_start_wall
            counters.peak_memory_bytes = current_peak_memory_bytes()
            task_counters.append({"task_id": task_id, **counters.as_dict()})

    aggregate = RuntimeCounters()
    aggregate.proposer_forward_passes = training_forward_passes + sum(row["proposer_forward_passes"] for row in task_counters)
    for row in task_counters:
        for field in (
            "candidate_outputs", "unique_canonical_ast_outputs", "candidate_evaluations", "verifier_calls",
            "oracle_calls", "decoder_tokens", "cpu_seconds", "wall_clock_seconds",
        ):
            setattr(aggregate, field, getattr(aggregate, field) + row[field])
        aggregate.peak_memory_bytes = max(aggregate.peak_memory_bytes, row["peak_memory_bytes"])
    aggregate.cpu_seconds += training_wall
    aggregate.wall_clock_seconds += training_wall
    return {
        "seed": seed,
        "updates": UPDATES,
        "device": "cpu",
        "training_only": True,
        "training_forward_passes": training_forward_passes,
        "training_wall_clock_seconds": training_wall,
        "trace": trace,
        "state_hash": state_hash(model),
        "proposals": proposals,
        "task_counters": task_counters,
        "aggregate_counters": aggregate.as_dict(),
    }


def lifecycle_smoke(grammar: GrammarConstraint) -> Dict[str, Any]:
    fixture_ast = {"op": "add_mod10", "args": [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]}
    fixture_examples = [
        {"a": 2, "b": 3, "output": 5},
        {"a": 7, "b": 8, "output": 5},
        {"a": 9, "b": 6, "output": 5},
    ]
    verifier = ASTVerifier(grammar)
    accepted, checks = verifier.verify(fixture_ast, fixture_examples)
    store_path = OUTPUT / "LIFECYCLE_STORE.json"
    store = ASTProcedureStore(store_path)
    first = store.promote("compatibility_fixture", fixture_ast, {"source": "synthetic_fixture", "verified_checks": checks})
    reloaded = ASTProcedureStore(store_path)
    reset_execution = [reloaded.execute("compatibility_fixture", row["a"], row["b"]) for row in fixture_examples]
    reset_correct = reset_execution == [row["output"] for row in fixture_examples]
    invalidated = reloaded.invalidate("compatibility_fixture", "synthetic_contradiction_event")
    successor_ast = {"op": "sub_mod10", "args": [{"op": "var", "name": "a"}, {"op": "var", "name": "b"}]}
    successor_examples = [{"a": 9, "b": 6, "output": 3}, {"a": 7, "b": 2, "output": 5}]
    successor_ok, successor_checks = verifier.verify(successor_ast, successor_examples)
    second = reloaded.promote("compatibility_fixture", successor_ast, {"source": "synthetic_revision", "verified_checks": successor_checks})
    return {
        "fixture_only": True,
        "proposal_not_claimed": True,
        "schema_validation": accepted,
        "initial_verifier_checks": checks,
        "persistence_reload": reloaded.active("compatibility_fixture") is not None,
        "post_reset_execution_correct": reset_correct,
        "external_contradiction_invalidation": invalidated,
        "successor_verification": successor_ok,
        "successor_active": reloaded.active("compatibility_fixture") == second,
        "versions": [first["version"], second["version"]],
        "store_path": str(store_path.relative_to(ROOT)),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    grammar_spec = load_json("GRAMMAR_SPEC.json")
    grammar = GrammarConstraint(grammar_spec["max_depth"], grammar_spec["max_nodes"])
    rows, labels, manifest_by_id = load_training_records()
    run_one = run_once(SEED, rows, labels, manifest_by_id, grammar)
    run_two = run_once(SEED, rows, labels, manifest_by_id, grammar)
    replay_equal = (
        run_one["state_hash"] == run_two["state_hash"]
        and run_one["trace"] == run_two["trace"]
        and run_one["proposals"] == run_two["proposals"]
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generated_names = {
        "RUN_CONFIG.json", "TRAINING_TRACE.jsonl", "PROPOSALS.jsonl", "COUNTERS.jsonl",
        "REPLAY_COMPARISON.json", "LIFECYCLE_SMOKE.json", "LIFECYCLE_STORE.json",
        "COMPATIBILITY_REPORT.json", "HASHES.json", "README.md",
    }
    existing = list(OUTPUT.iterdir())
    unexpected = [path.name for path in existing if path.name not in generated_names or not path.is_file()]
    if unexpected:
        raise SystemExit(f"refusing to overwrite unexpected compatibility paths: {unexpected}")
    for path in existing:
        path.unlink()
    smoke = lifecycle_smoke(grammar)

    run_config = {
        "schema_version": "crucible_track_b_b3_compatibility_preflight_v1",
        "status": "compatibility_only",
        "seed": SEED,
        "updates": UPDATES,
        "device": "cpu",
        "candidate_count": CANDIDATE_COUNT,
        "input_files": ["PROPOSER_INPUTS.jsonl training rows only", "TRAIN_LABELS.jsonl", "TASK_MANIFEST.json training rows only"],
        "eval_oracle_loaded": False,
        "validation_or_test_scored": False,
        "sealed_test_accessed": False,
        "direct_candidate_source": "neural_generated",
        "grammar": grammar_spec["schema_version"],
        "source_hashes": {
            "proposer_module": sha256(ROOT / "tools" / "crucible_track_b_neural_proposer.py"),
            "runner": sha256(ROOT / "tools" / "run_crucible_track_b_neural_proposer_compatibility_preflight_20261028.py"),
        },
    }
    write_json(OUTPUT / "RUN_CONFIG.json", run_config)
    write_jsonl(OUTPUT / "TRAINING_TRACE.jsonl", run_one["trace"])
    write_jsonl(OUTPUT / "PROPOSALS.jsonl", run_one["proposals"])
    write_jsonl(OUTPUT / "COUNTERS.jsonl", run_one["task_counters"])
    write_json(OUTPUT / "REPLAY_COMPARISON.json", {
        "same_seed": SEED,
        "state_hash_equal": run_one["state_hash"] == run_two["state_hash"],
        "training_trace_equal": run_one["trace"] == run_two["trace"],
        "proposal_outputs_equal": run_one["proposals"] == run_two["proposals"],
        "deterministic_replay": replay_equal,
        "run_one_state_hash": run_one["state_hash"],
        "run_two_state_hash": run_two["state_hash"],
    })
    write_json(OUTPUT / "LIFECYCLE_SMOKE.json", smoke)
    report = {
        "status": "pass" if replay_equal and smoke["post_reset_execution_correct"] and smoke["successor_verification"] else "fail",
        "classification": "compatibility_only_not_scientific_evidence",
        "training_only": True,
        "updates": UPDATES,
        "training_task_count": len(rows),
        "direct_ast_generation": True,
        "generated_candidate_source": "neural_generated",
        "grammar_valid_proposal_count": sum(1 for proposal in run_one["proposals"] if proposal["grammar_valid"]),
        "proposal_count": len(run_one["proposals"]),
        "deterministic_replay": replay_equal,
        "lifecycle_interface_smoke": smoke,
        "aggregate_counters": run_one["aggregate_counters"],
        "validation_or_test_scored": False,
        "sealed_test_accessed": False,
        "scientific_checkpoint_selected": False,
        "scientific_execution_approved": False,
        "note": "Passing this preflight does not establish acquisition, generalisation, lifecycle success for B3, or a neural advantage over B1.",
    }
    write_json(OUTPUT / "COMPATIBILITY_REPORT.json", report)
    (OUTPUT / "README.md").write_text(
        "# Track B B3 compatibility preflight\n\n"
        "Training-only, CPU-only 50-update implementation check. B3 emits grammar-constrained prefix ASTs directly; it does not rerank an enumerated catalogue. Validation/test records and EVAL_ORACLE.jsonl were not loaded or scored. A passing report is not scientific evidence of procedural acquisition or generalisation.\n",
        encoding="utf-8",
    )
    hashes = {}
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name not in {"HASHES.json", "COMPATIBILITY_REPORT.json"}:
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    print(json.dumps({"status": report["status"], "output": str(OUTPUT), "deterministic_replay": replay_equal, "training_only": True, "updates": UPDATES}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
