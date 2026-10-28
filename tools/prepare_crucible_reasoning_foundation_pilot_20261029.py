#!/usr/bin/env python3
"""Prepare a small, provenance-preserving Crucible reasoning pilot.

Raw datasets remain outside Git.  This adapter emits JSONL records with a
common ``input``/``target`` contract plus a manifest of source revisions and
record hashes.  It uses the five prepared reasoning sources and the existing
local TinyStories context corpus as a language-foundation proxy.  FineWeb-Edu,
Dolma and Gutenberg are deliberately not silently substituted: they are
recorded as not present in the local preparation manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "crucible-reasoning-foundation-pilot-v1-20261029"


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


def record_id(domain: str, source_key: str, index: int) -> str:
    return f"{domain}:{source_key}:{index:08d}"


def make_record(
    *,
    domain: str,
    source_key: str,
    index: int,
    split: str,
    prompt: str,
    target: str,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    rid = record_id(domain, source_key, index)
    core = {
        "record_id": rid,
        "domain": domain,
        "source_key": source_key,
        "split": split,
        "input": str(prompt),
        "target": str(target),
        "metadata": dict(metadata),
    }
    core["record_sha256"] = sha256_bytes(canonical_json(core).encode("utf-8"))
    return core


def split_hash(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(str(record["record_id"]).encode("utf-8")).hexdigest()


def assign_split(record: Mapping[str, Any]) -> str:
    value = int(split_hash(record)[:8], 16) % 10
    return "train" if value < 8 else "validation" if value == 8 else "test"


def load_math(root: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    path = root / "math_data" / "data" / "train-00000-of-00001-7320a6f3aba8ebd2.parquet"
    rows = pq.read_table(path).to_pylist()
    result = []
    for index, row in enumerate(rows):
        result.append(make_record(
            domain="math",
            source_key="competition_math",
            index=index,
            split="pending",
            prompt="Solve the following mathematics problem. Show the reasoning and final answer.\n\n" + row["problem"],
            target=row["solution"],
            metadata={"level": row.get("level"), "type": row.get("type"), "source_revision": "e839825f9ec5c6cfa585c654a59610969ec13993"},
        ))
    return result


def load_bbh(root: Path) -> List[Dict[str, Any]]:
    result = []
    index = 0
    for path in sorted((root / "bbh" / "bbh").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for example in payload.get("examples", []):
            result.append(make_record(
                domain="bbh",
                source_key=path.stem,
                index=index,
                split="pending",
                prompt="Answer the reasoning task.\n\n" + str(example["input"]),
                target=str(example["target"]),
                metadata={"task": path.stem, "source_revision": "9ee07bd481feebf959a6b59d61ea57bdcf30964d"},
            ))
            index += 1
    return result


def load_scibench(root: Path) -> List[Dict[str, Any]]:
    result = []
    index = 0
    base = root / "scibench" / "dataset" / "original"
    for path in sorted(base.glob("*.json")):
        if path.stem.endswith("_sol"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not row.get("problem_text") or row.get("answer_number") is None:
                continue
            result.append(make_record(
                domain="scibench",
                source_key=path.stem,
                index=index,
                split="pending",
                prompt="Solve the following scientific problem. Give the final numerical answer and explain the reasoning.\n\n" + str(row["problem_text"]),
                target=str(row["answer_number"]),
                metadata={"unit": row.get("unit", ""), "source": row.get("source", path.stem), "problemid": row.get("problemid", ""), "source_revision": "e14e0ca3d3db0f02928a18d435b9c3d15656a7c5"},
            ))
            index += 1
    return result


def load_ethics(root: Path) -> List[Dict[str, Any]]:
    result = []
    index = 0
    for path in sorted((root / "ethics").glob("*/*.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("input") or row.get("label") is None:
                    continue
                filename_split = "test" if "_test" in path.stem or "ambig" in path.stem else "train"
                result.append(make_record(
                    domain="ethics",
                    source_key=f"{path.parent.name}/{path.stem}",
                    index=index,
                    split=filename_split,
                    prompt="Classify the following normative or commonsense statement. Explain the judgment.\n\n" + row["input"],
                    target=str(row["label"]),
                    metadata={"subdomain": path.parent.name, "source_revision": "fa022dbc62dc6736c2f3780ced6412cae97efb90"},
                ))
                index += 1
    return result


def load_bigbench(root: Path) -> List[Dict[str, Any]]:
    result = []
    index = 0
    base = root / "big_bench" / "bigbench" / "benchmark_tasks"
    for path in sorted(base.glob("*/task.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for example in payload.get("examples", []):
            scores = example.get("target_scores", {})
            if not scores:
                continue
            target = next((str(choice) for choice, score in scores.items() if score == 1), None)
            if target is None:
                continue
            choices = ", ".join(str(choice) for choice in scores)
            prompt = f"Choose the correct answer from: {choices}.\n\n{example['input']}"
            result.append(make_record(
                domain="bigbench_lite",
                source_key=path.parent.name,
                index=index,
                split="pending",
                prompt=prompt,
                target=target,
                metadata={"task": path.parent.name, "source_revision": "092b196c1f8f14a54bbc62f24759d43bde46dd3b"},
            ))
            index += 1
    return result


def load_language(root: Path, limit: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source = root / "context_128" / "source" / "packed_train.jsonl"
    validation = root / "context_128" / "source" / "packed_validation.jsonl"
    if not source.is_file() or not validation.is_file():
        return [], {"status": "not_present", "role": "language_foundation_proxy"}
    result = []
    for path, split, local_limit in ((source, "train", limit), (validation, "validation", max(1, limit // 8))):
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= local_limit:
                    break
                row = json.loads(line)
                result.append(make_record(
                    domain="language_foundation_proxy",
                    source_key="local_tinystories_context",
                    index=(0 if split == "train" else limit) + index,
                    split=split,
                    prompt="Continue the following text:\n\n" + str(row["text"]),
                    target=str(row["text"]),
                    metadata={"source_revision": "local_context_manifest", "source_example_index": row.get("example_index")},
                ))
    return result, {"status": "present", "role": "language_foundation_proxy", "source_sha256": sha256_file(source), "validation_sha256": sha256_file(validation)}


def bound_records(records: Sequence[Mapping[str, Any]], per_domain: int) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        item = dict(record)
        if item["split"] == "pending":
            item["split"] = assign_split(item)
        grouped.setdefault(str(item["domain"]), []).append(item)
    output = {"train": [], "validation": [], "test": []}
    for domain in sorted(grouped):
        rows = grouped[domain]
        for split in output:
            candidates = [row for row in rows if row["split"] == split]
            output[split].extend(candidates[:per_domain])
    for split in output:
        output[split].sort(key=lambda row: str(row["record_id"]))
    return output


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/timbushnell/crucible_datasets_20261029"))
    parser.add_argument("--language-root", type=Path, default=Path("/home/timbushnell/crucible_v2_production_telemetry_v1_20261019"))
    parser.add_argument("--output-root", type=Path, default=Path("/home/timbushnell/crucible_reasoning_pilot_20261029"))
    parser.add_argument("--per-domain", type=int, default=128)
    parser.add_argument("--language-count", type=int, default=512)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    rows: List[Dict[str, Any]] = []
    source_status: Dict[str, Any] = {}
    loaders = (("math", load_math), ("bbh", load_bbh), ("scibench", load_scibench), ("ethics", load_ethics), ("bigbench_lite", load_bigbench))
    for name, loader in loaders:
        loaded = loader(root)
        rows.extend(loaded)
        source_status[name] = {"records_seen": len(loaded)}
    language_rows, language_status = load_language(args.language_root.resolve(), args.language_count)
    rows.extend(language_rows)
    source_status["language_foundation_proxy"] = language_status
    splits = bound_records(rows, args.per_domain)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for split, split_rows in splits.items():
        file_hashes[f"{split}.jsonl"] = write_jsonl(output / f"{split}.jsonl", split_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "raw_data_committed_to_git": False,
        "source_status": source_status,
        "limits": {"per_domain_per_split": args.per_domain, "language_train_limit": args.language_count},
        "counts": {split: len(rows_for_split) for split, rows_for_split in splits.items()},
        "counts_by_domain": {
            split: {
                domain: sum(1 for row in rows_for_split if row["domain"] == domain)
                for domain in sorted({row["domain"] for row in rows_for_split})
            }
            for split, rows_for_split in splits.items()
        },
        "files": file_hashes,
        "missing_recommended_language_sources": ["FineWeb-Edu", "Dolma", "Project Gutenberg Australia"],
        "interpretation_boundary": "This is a data/pretraining pilot manifest; benchmark performance does not establish cumulative learning.",
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output), "counts": manifest["counts"], "source_status": source_status}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
