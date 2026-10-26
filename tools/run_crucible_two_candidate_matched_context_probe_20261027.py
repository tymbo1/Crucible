#!/usr/bin/env python3
"""Run the checkpoint-only matched-context two-candidate probe.

The probe evaluates familiar and recombined candidate sets on the same union
of source operand contexts. It performs no training and never touches sealed
test data.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUNNER = ROOT / "tools" / "run_crucible_shared_candidate_scoring_cardinality_20261026.py"
TRAIN_SOURCE = ROOT / "experiments" / "crucible_shared_candidate_scoring_cardinality_20261026_v1" / "data" / "canonical" / "train.jsonl"
VALIDATION_SOURCE = ROOT / "experiments" / "crucible_shared_candidate_scoring_cardinality_20261026_v1" / "data" / "canonical" / "validation.jsonl"
GATE_ROOT = ROOT / "experiments" / "crucible_two_candidate_competence_gate_20261027_v1"
GATE_SUMMARY = GATE_ROOT / "SUMMARY.json"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_MATCHED_CONTEXT_PROBE_PROTOCOL_20261027.md"
NOVELTY = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_MATCHED_CONTEXT_PROBE_NOVELTY_AUDIT_20261027.json"
OUTPUT = ROOT / "experiments" / "crucible_two_candidate_matched_context_probe_20261027_v1"

SEEDS = (42, 1337)
PASSING_CELLS = {
    42: "DENSE_CYCLIC_SEED42",
    1337: "DENSE_CYCLIC_SEED1337",
}
REGIMES = {
    "FAMILIAR": ((0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)),
    "RECOMBINED": ((0, 3), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2), (3, 6), (6, 3)),
}
REGIME_ORDER = ("FAMILIAR", "RECOMBINED")
CONTEXT_SOURCE_ORDER = ("TRAIN_SOURCE", "VALIDATION_SOURCE")


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


source = import_module("shared_candidate_scoring_source_for_matched_probe", SOURCE_RUNNER)
torch = source.torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def oracle_target(row: Mapping[str, Any]) -> int:
    candidates = [int(value) for value in row["metadata"]["candidates"]]
    authority = int(row["metadata"]["authority_id"])
    return candidates.index(authority) if authority in candidates else 8


def context_inventory(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, int], int]:
    contexts = Counter((int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in rows if len(row["metadata"]["candidates"]) == 2)
    if any(count != 24 for count in contexts.values()):
        raise RuntimeError(f"unexpected source context counts: {contexts}")
    return dict(contexts)


def make_probe_rows(contexts: Sequence[Tuple[int, int]], regime: str, candidate_sets: Sequence[Sequence[int]], context_source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ordinal = 0
    for context_index, (a, b) in enumerate(contexts):
        for candidate_index, candidate_tuple in enumerate(candidate_sets):
            candidates = tuple(int(value) for value in candidate_tuple)
            pair_key = f"matched_context|{context_source}|{regime}|{a}|{b}|{candidates}"
            pair_id = hashlib.sha256(pair_key.encode("utf-8")).hexdigest()[:20]
            for authority in (candidates[0], candidates[1], 9):
                target = candidates.index(authority) if authority in candidates else 8
                record_key = f"matched_context|{context_source}|{regime}|{a}|{b}|{candidates}|{authority}|{ordinal}"
                record_id = hashlib.sha256(record_key.encode("utf-8")).hexdigest()[:20]
                rows.append({
                    "schema_version": "crucible_two_candidate_matched_context_probe_v1",
                    "record_id": record_id,
                    "interface": "canonical",
                    "split": "matched_context_probe",
                    "layout": "TRAIN_COMPACT",
                    "input": source.input_text("canonical", "TRAIN_COMPACT", a, b, candidates, authority, ordinal),
                    "target": str(target),
                    "oracle": {"kind": "identifier_match", "target_slot": target, "authority_id": authority},
                    "metadata": {
                        "a": a,
                        "b": b,
                        "candidates": list(candidates),
                        "authority_id": authority,
                        "target_slot": target,
                        "counterfactual_pair_id": pair_id if authority != 9 else None,
                        "candidate_regime": regime,
                        "context_source": context_source,
                        "context_index": context_index,
                        "candidate_index": candidate_index,
                    },
                })
                ordinal += 1
    return rows


def audit_rows(name: str, rows: Sequence[Mapping[str, Any]], expected_context_source: str, expected_regime: str, expected_context_count: int) -> Dict[str, Any]:
    expected_rows = expected_context_count * 8 * 3
    if len(rows) != expected_rows:
        raise RuntimeError(f"{name}: expected {expected_rows} rows, got {len(rows)}")
    ids = [str(row["record_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{name}: duplicate record IDs")
    for row in rows:
        if row["metadata"]["context_source"] != expected_context_source:
            raise RuntimeError(f"{name}: context source mismatch")
        if row["metadata"]["candidate_regime"] != expected_regime:
            raise RuntimeError(f"{name}: regime mismatch")
        if int(row["metadata"]["target_slot"]) != oracle_target(row):
            raise RuntimeError(f"{name}: oracle mismatch for {row['record_id']}")
        if not str(row["input"]):
            raise RuntimeError(f"{name}: empty input")
    return {
        "rows": len(rows),
        "unique_record_ids": len(set(ids)),
        "context_source": expected_context_source,
        "candidate_regime": expected_regime,
        "contexts": sorted({f"{row['metadata']['a']},{row['metadata']['b']}" for row in rows}),
        "candidate_sets": sorted({",".join(str(value) for value in row["metadata"]["candidates"]) for row in rows}),
        "target_slot_counts": {str(key): value for key, value in sorted(Counter(int(row["metadata"]["target_slot"]) for row in rows).items())},
        "oracle_rows_correct": sum(int(int(row["metadata"]["target_slot"]) == oracle_target(row)) for row in rows),
        "layouts": dict(Counter(str(row["layout"]) for row in rows)),
    }


def load_model(checkpoint: Path, seed: int) -> Any:
    model = source.CandidateReadout(seed, "SHARED_SCORER")
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def evaluate(model: Any, records: Sequence[Mapping[str, Any]], seed: int) -> List[Dict[str, Any]]:
    prepared = source.prepare_records(records)
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(prepared), int(source.EFFECTIVE_BATCH)):
        indices = list(range(start, min(start + int(source.EFFECTIVE_BATCH), len(prepared))))
        batch = source.batchify(prepared, indices)
        with torch.inference_mode():
            predictions = model.decision_logits(batch).argmax(dim=-1).tolist()
        for local, index in enumerate(indices):
            item = prepared[index]
            metadata = item["metadata"]
            target = int(metadata["target_slot"])
            rows.append({
                "seed": seed,
                "candidate_regime": str(metadata["candidate_regime"]),
                "context_source": str(metadata["context_source"]),
                "context_a": int(metadata["a"]),
                "context_b": int(metadata["b"]),
                "split": str(item["split"]),
                "layout": str(item["layout"]),
                "record_id": str(item["record_id"]),
                "counterfactual_pair_id": metadata["counterfactual_pair_id"],
                "candidates": ",".join(str(value) for value in metadata["candidates"]),
                "authority_id": int(metadata["authority_id"]),
                "target_slot": target,
                "predicted_slot": int(predictions[local]),
                "pointer_correct": int(predictions[local]) == target,
            })
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "seed", "candidate_regime", "context_source", "context_a", "context_b",
        "split", "layout", "record_id", "counterfactual_pair_id", "candidates",
        "authority_id", "target_slot", "predicted_slot", "pointer_correct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["counterfactual_pair_id"]:
            groups[str(row["counterfactual_pair_id"])].append(row)
    complete = [values for values in groups.values() if len(values) == 2]
    return {
        "groups": len(complete),
        "both_correct": sum(int(all(bool(row["pointer_correct"]) for row in values)) for values in complete),
        "predicted_flips": sum(int(values[0]["predicted_slot"] != values[1]["predicted_slot"]) for values in complete),
        "target_flips": sum(int(values[0]["target_slot"] != values[1]["target_slot"]) for values in complete),
    }


def metric_block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    correct = sum(int(bool(row["pointer_correct"])) for row in rows)
    by_candidate_set: Dict[str, Dict[str, Any]] = {}
    for candidate_set in sorted({str(row["candidates"]) for row in rows}):
        subset = [row for row in rows if str(row["candidates"]) == candidate_set]
        set_correct = sum(int(bool(row["pointer_correct"])) for row in subset)
        by_candidate_set[candidate_set] = {"rows": len(subset), "correct": set_correct, "accuracy": set_correct / len(subset) if subset else None}
    by_context: Dict[str, Dict[str, Any]] = {}
    for context in sorted({(int(row["context_a"]), int(row["context_b"])) for row in rows}):
        subset = [row for row in rows if (int(row["context_a"]), int(row["context_b"])) == context]
        context_correct = sum(int(bool(row["pointer_correct"])) for row in subset)
        key = f"{context[0]},{context[1]}"
        by_context[key] = {"rows": len(subset), "correct": context_correct, "accuracy": context_correct / len(subset) if subset else None}
    confusion = Counter((int(row["target_slot"]), int(row["predicted_slot"])) for row in rows)
    abstention = [row for row in rows if int(row["target_slot"]) == 8]
    abstention_correct = sum(int(bool(row["pointer_correct"])) for row in abstention)
    return {
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "abstention": {"rows": len(abstention), "correct": abstention_correct, "accuracy": abstention_correct / len(abstention) if abstention else None},
        "by_candidate_set": by_candidate_set,
        "by_context": by_context,
        "target_prediction_confusion": {f"target_{target}_pred_{predicted}": count for (target, predicted), count in sorted(confusion.items())},
        "paired": paired_metrics(rows),
        "errors_by_candidate_set": dict(Counter(str(row["candidates"]) for row in rows if not bool(row["pointer_correct"]))),
        "errors_by_context": dict(Counter(f"{row['context_a']},{row['context_b']}" for row in rows if not bool(row["pointer_correct"]))),
    }


def regime_delta(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    familiar = metrics["FAMILIAR"]
    recombined = metrics["RECOMBINED"]
    return {
        "row_accuracy_recombined_minus_familiar": float(recombined["accuracy"] - familiar["accuracy"]),
        "paired_both_correct_recombined_minus_familiar": int(recombined["paired"]["both_correct"] - familiar["paired"]["both_correct"]),
        "predicted_flips_recombined_minus_familiar": int(recombined["paired"]["predicted_flips"] - familiar["paired"]["predicted_flips"]),
    }


def main() -> int:
    if OUTPUT.exists() and (OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"refusing to overwrite completed output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    train_rows = [row for row in load_jsonl(TRAIN_SOURCE) if len(row["metadata"]["candidates"]) == 2]
    validation_rows = [row for row in load_jsonl(VALIDATION_SOURCE) if len(row["metadata"]["candidates"]) == 2]
    train_context_counts = context_inventory(train_rows)
    validation_context_counts = context_inventory(validation_rows)
    train_contexts = sorted(train_context_counts)
    validation_contexts = sorted(validation_context_counts)
    if set(train_contexts) & set(validation_contexts):
        raise RuntimeError("source context overlap")
    context_sets = {"TRAIN_SOURCE": train_contexts, "VALIDATION_SOURCE": validation_contexts}

    conditions: Dict[str, List[Dict[str, Any]]] = {}
    audits: Dict[str, Any] = {}
    for context_source in CONTEXT_SOURCE_ORDER:
        for regime in REGIME_ORDER:
            name = f"{context_source}_{regime}"
            conditions[name] = make_probe_rows(context_sets[context_source], regime, REGIMES[regime], context_source)
            audits[name] = audit_rows(name, conditions[name], context_source, regime, len(context_sets[context_source]))

    data_audit = {
        "experiment": "crucible_two_candidate_matched_context_probe_20261027_v1",
        "train_source": rel(TRAIN_SOURCE),
        "validation_source": rel(VALIDATION_SOURCE),
        "train_source_sha256": sha256(TRAIN_SOURCE),
        "validation_source_sha256": sha256(VALIDATION_SOURCE),
        "source_train_two_candidate_context_counts": {f"{a},{b}": count for (a, b), count in sorted(train_context_counts.items())},
        "source_validation_two_candidate_context_counts": {f"{a},{b}": count for (a, b), count in sorted(validation_context_counts.items())},
        "source_train_contexts": [f"{a},{b}" for a, b in train_contexts],
        "source_validation_contexts": [f"{a},{b}" for a, b in validation_contexts],
        "conditions": audits,
        "record_id_overlap_between_source_splits": 0,
        "training_performed": False,
        "sealed_test_accessed": False,
        "checkpoints_modified": False,
    }
    write_json(OUTPUT / "DATA_AUDIT.json", data_audit)
    write_json(OUTPUT / "NOVELTY_AUDIT.json", json.loads(NOVELTY.read_text(encoding="utf-8")))

    checkpoint_hashes: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}
    rows_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for seed in SEEDS:
        cell = PASSING_CELLS[seed]
        checkpoint = GATE_ROOT / cell / "checkpoint.pt"
        checkpoint_hashes[cell] = sha256(checkpoint)
        model = load_model(checkpoint, seed)
        seed_rows: List[Dict[str, Any]] = []
        for condition_name in ("TRAIN_SOURCE_FAMILIAR", "TRAIN_SOURCE_RECOMBINED", "VALIDATION_SOURCE_FAMILIAR", "VALIDATION_SOURCE_RECOMBINED"):
            seed_rows.extend(evaluate(model, conditions[condition_name], seed))
        del model
        rows_by_seed[seed] = seed_rows
        write_csv(OUTPUT / f"PREDICTIONS_SEED{seed}.csv", seed_rows)
        metrics[str(seed)] = {"overall": {}, "by_context_source": {}}
        for regime in REGIME_ORDER:
            regime_rows = [row for row in seed_rows if row["candidate_regime"] == regime]
            metrics[str(seed)]["overall"][regime] = metric_block(regime_rows)
        for context_source in CONTEXT_SOURCE_ORDER:
            metrics[str(seed)]["by_context_source"][context_source] = {}
            for regime in REGIME_ORDER:
                subset = [row for row in seed_rows if row["candidate_regime"] == regime and row["context_source"] == context_source]
                metrics[str(seed)]["by_context_source"][context_source][regime] = metric_block(subset)

    deltas = {
        str(seed): {
            "overall": regime_delta(metrics[str(seed)]["overall"]),
            "by_context_source": {
                context_source: regime_delta(metrics[str(seed)]["by_context_source"][context_source])
                for context_source in CONTEXT_SOURCE_ORDER
            },
        }
        for seed in SEEDS
    }
    summary = {
        "experiment": "crucible_two_candidate_matched_context_probe_20261027_v1",
        "status": "completed",
        "protocol": rel(PROTOCOL),
        "novelty_audit": rel(NOVELTY),
        "data_audit": rel(OUTPUT / "DATA_AUDIT.json"),
        "source_train": rel(TRAIN_SOURCE),
        "source_validation": rel(VALIDATION_SOURCE),
        "gate_summary": rel(GATE_SUMMARY),
        "checkpoint_hashes": checkpoint_hashes,
        "seeds": list(SEEDS),
        "regimes": list(REGIME_ORDER),
        "context_sources": list(CONTEXT_SOURCE_ORDER),
        "training_performed": False,
        "sealed_test_accessed": False,
        "checkpoints_modified": False,
        "metrics": metrics,
        "regime_deltas": deltas,
    }
    write_json(OUTPUT / "SUMMARY.json", summary)

    report_lines = [
        "# Crucible Two-Candidate Matched Context Probe v1",
        "",
        "Checkpoint-only exploratory probe. Familiar and recombined candidate sets were evaluated on the same source-context strata in fixed `TRAIN_COMPACT` serialization. No training was performed.",
        "",
        "## Results by source-context stratum",
        "",
        "| Seed | Context source | Familiar row accuracy | Recombined row accuracy | Familiar paired | Recombined paired | Familiar flips | Recombined flips |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for context_source in CONTEXT_SOURCE_ORDER:
            block = metrics[str(seed)]["by_context_source"][context_source]
            fam = block["FAMILIAR"]
            rec = block["RECOMBINED"]
            report_lines.append(
                f"| {seed} | {context_source} | {fam['correct']}/{fam['rows']} ({fam['accuracy']:.3f}) | {rec['correct']}/{rec['rows']} ({rec['accuracy']:.3f}) | {fam['paired']['both_correct']}/{fam['paired']['groups']} | {rec['paired']['both_correct']}/{rec['paired']['groups']} | {fam['paired']['predicted_flips']}/{fam['paired']['target_flips']} | {rec['paired']['predicted_flips']}/{rec['paired']['target_flips']} |"
            )
    report_lines += [
        "",
        "## Overall matched-regime results",
        "",
        "| Seed | Familiar | Recombined | Recombined minus familiar |",
        "|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        overall = metrics[str(seed)]["overall"]
        fam = overall["FAMILIAR"]
        rec = overall["RECOMBINED"]
        report_lines.append(f"| {seed} | {fam['correct']}/{fam['rows']} ({fam['accuracy']:.3f}) | {rec['correct']}/{rec['rows']} ({rec['accuracy']:.3f}) | {deltas[str(seed)]['overall']['row_accuracy_recombined_minus_familiar']:+.3f} |")
    report_lines += [
        "",
        "## Interpretation boundary",
        "",
        "The same source-context values are used in both candidate regimes within each stratum. This removes the original context-regime confound descriptively, but the probe is generated from source distributions and is not an independent held-out benchmark. It does not establish a general identifier-comparison algorithm or an architectural mechanism.",
        "",
        "## Artifacts",
        "",
        f"- [Protocol](../../{rel(PROTOCOL)})",
        f"- [Novelty audit](../../{rel(NOVELTY)})",
        "- [Summary](SUMMARY.json)",
        "- [Data audit](DATA_AUDIT.json)",
        "- Per-seed prediction matrices in `PREDICTIONS_SEED42.csv` and `PREDICTIONS_SEED1337.csv`",
    ]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    interpretation = f"""# Scientific Interpretation

This checkpoint-only probe evaluates the same source operand contexts with
both the familiar and recombined candidate-set regimes. It therefore tests
whether the candidate-regime effect survives when the original `(a,b)` context
shift is removed within each comparison.

The probe did not train, modify checkpoints or access sealed test data. It is
exploratory and generated from the canonical source context populations, not a
new independent validation benchmark.

## Result summary

{chr(10).join(
    f"- Seed {seed}, {context_source}: familiar {metrics[str(seed)]['by_context_source'][context_source]['FAMILIAR']['correct']}/{metrics[str(seed)]['by_context_source'][context_source]['FAMILIAR']['rows']} versus recombined {metrics[str(seed)]['by_context_source'][context_source]['RECOMBINED']['correct']}/{metrics[str(seed)]['by_context_source'][context_source]['RECOMBINED']['rows']} (row-accuracy delta {deltas[str(seed)]['by_context_source'][context_source]['row_accuracy_recombined_minus_familiar']:+.3f})."
    for seed in SEEDS for context_source in CONTEXT_SOURCE_ORDER
)}

## Interpretation rule

If recombined performance is lower within both context strata, the result is
consistent with a candidate-pair-regime effect independent of the original
source-context shift. If both regimes degrade mainly on validation-source
contexts, context shift is implicated. If the pattern differs by stratum, the
correct description is an interaction. These are behavioral descriptions, not
mechanistic or architectural conclusions.

See `SUMMARY.json` for denominators, paired correctness and counterfactual
flip counts; see `DATA_AUDIT.json` for the source-context construction.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")

    hash_paths = [
        SOURCE_RUNNER,
        PROTOCOL,
        NOVELTY,
        TRAIN_SOURCE,
        VALIDATION_SOURCE,
        GATE_SUMMARY,
        OUTPUT / "DATA_AUDIT.json",
        OUTPUT / "SUMMARY.json",
        OUTPUT / "REPORT_FINAL.md",
        OUTPUT / "SCIENTIFIC_INTERPRETATION.md",
    ]
    for seed in SEEDS:
        hash_paths.append(GATE_ROOT / PASSING_CELLS[seed] / "checkpoint.pt")
        hash_paths.append(OUTPUT / f"PREDICTIONS_SEED{seed}.csv")
    hash_paths.append(Path(__file__).resolve())
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths})
    write_json(OUTPUT / "FINALIZATION.json", {
        "experiment": summary["experiment"],
        "status": "completed",
        "verification_pass": True,
        "training_performed": False,
        "sealed_test_accessed": False,
        "checkpoints_modified": False,
        "seed_count": len(SEEDS),
        "regime_count": len(REGIME_ORDER),
        "context_source_count": len(CONTEXT_SOURCE_ORDER),
        "hashes": rel(OUTPUT / "HASHES.json"),
    })
    print(json.dumps({"experiment": summary["experiment"], "status": "completed", "training_performed": False, "sealed_test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
