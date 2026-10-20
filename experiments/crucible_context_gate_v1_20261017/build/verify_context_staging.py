#!/usr/bin/env python3
"""Independent verifier for a context-length staging corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List


END_OF_TEXT = b"<|endoftext|>"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_story(raw_story: bytes) -> str:
    text = unicodedata.normalize("NFC", raw_story.decode("utf-8", errors="strict"))
    text = text.replace("\ufeff", " ")
    cleaned: List[str] = []
    for character in text:
        category = unicodedata.category(character)
        if category.startswith("C"):
            if character in "\r\n\t":
                cleaned.append(" ")
            continue
        cleaned.append(character)
    return re.sub(r"\s+", " ", "".join(cleaned)).strip()


def verify_source_range(
    source_path: Path,
    record: Dict[str, object],
    provenance: Dict[str, object],
) -> Dict[str, object]:
    start = int(provenance["source_start_byte"])
    end = int(provenance["source_end_byte"])
    if not (0 <= start < end <= source_path.stat().st_size):
        raise AssertionError("invalid source byte range")
    with source_path.open("rb") as handle:
        handle.seek(start)
        raw_story = handle.read(end - start)
        delimiter = handle.read(len(END_OF_TEXT))
    if delimiter != END_OF_TEXT:
        raise AssertionError("source range does not terminate at story delimiter")
    if sha256_bytes(raw_story) != provenance["source_story_sha256"]:
        raise AssertionError("source story hash mismatch")
    clean = clean_story(raw_story)
    for component in provenance["components"]:
        start_char = int(component["clean_story_char_start"])
        end_char = int(component["clean_story_char_end"])
        component_text = clean[start_char:end_char]
        if len(component_text.encode("utf-8")) != int(component["byte_length"]):
            raise AssertionError("component byte length mismatch")
        if sha256_bytes(component_text.encode("utf-8")) != component["text_utf8_sha256"]:
            raise AssertionError("component text hash mismatch")
    if int(provenance["component_count"]) != len(provenance["components"]):
        raise AssertionError("component count mismatch")
    return {"source_story_hash": provenance["source_story_sha256"]}


def verify_split(
    root: Path,
    split_name: str,
    max_bytes: int,
) -> Dict[str, object]:
    build = root / "build"
    records = read_jsonl(build / f"packed_{split_name}.jsonl")
    provenance = read_jsonl(build / f"provenance_{split_name}.jsonl")
    if len(records) != len(provenance):
        raise AssertionError(f"{split_name} record/provenance count mismatch")
    source_paths = {
        "source/TinyStoriesV2-GPT4-train.txt": root_source / "TinyStoriesV2-GPT4-train.txt",
        "source/TinyStoriesV2-GPT4-valid.txt": root_source / "TinyStoriesV2-GPT4-valid.txt",
    }
    seen = set()
    total_bytes = 0
    for local_id, (record, sidecar) in enumerate(zip(records, provenance)):
        if record.get("split") != split_name:
            raise AssertionError("split field mismatch")
        if int(record["example_index"]) != local_id:
            raise AssertionError("non-deterministic example index")
        if int(sidecar["local_split_id"]) != local_id:
            raise AssertionError("non-deterministic provenance index")
        text = str(record["text"])
        encoded = text.encode("utf-8")
        if not encoded or len(encoded) > max_bytes:
            raise AssertionError("invalid source text byte length")
        if "\n" in text or "\r" in text or END_OF_TEXT.decode() in text:
            raise AssertionError("delimiter or line break leaked into text")
        if encoded in seen:
            raise AssertionError("duplicate text in split")
        seen.add(encoded)
        if int(record["byte_length"]) != len(encoded):
            raise AssertionError("record byte length mismatch")
        text_hash = sha256_bytes(encoded)
        if record["text_utf8_sha256"] != text_hash or sidecar["text_utf8_sha256"] != text_hash:
            raise AssertionError("text hash mismatch")
        if int(sidecar["source_record_index"]) < 0:
            raise AssertionError("negative source story index")
        source_path = source_paths[str(sidecar["source_file"])]
        verify_source_range(source_path, record, sidecar)
        total_bytes += len(encoded)

    text_path = build / f"packed_{split_name}.txt"
    lines = text_path.read_bytes().splitlines()
    if lines != [str(record["text"]).encode("utf-8") for record in records]:
        raise AssertionError(f"text corpus mismatch for {split_name}")
    return {
        "count": len(records),
        "total_valid_utf8_bytes": total_bytes,
        "total_nonpad_target_tokens": total_bytes + len(records),
        "jsonl_sha256": sha256_file(build / f"packed_{split_name}.jsonl"),
        "text_sha256": sha256_file(text_path),
        "provenance_sha256": sha256_file(build / f"provenance_{split_name}.jsonl"),
        "source_offset_records_verified": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    global root_source
    root_source = args.source_root.resolve()
    root = args.dataset_root.resolve()
    stats = json.loads((root / "context_dataset_statistics.json").read_text(encoding="utf-8"))
    max_bytes = int(stats["max_source_utf8_bytes"])
    result = {
        "status": "verified",
        "context_length": int(stats["context_length"]),
        "max_source_utf8_bytes": max_bytes,
        "train": verify_split(root, "train", max_bytes),
        "validation": verify_split(root, "validation", max_bytes),
    }
    train_texts = {
        record["text"]
        for record in read_jsonl(root / "build" / "packed_train.jsonl")
    }
    validation_texts = {
        record["text"]
        for record in read_jsonl(root / "build" / "packed_validation.jsonl")
    }
    cross_split = train_texts & validation_texts
    if cross_split:
        raise AssertionError("cross-split exact duplicate found")
    result["cross_split_exact_duplicates"] = 0
    output = root / "context_staging_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
