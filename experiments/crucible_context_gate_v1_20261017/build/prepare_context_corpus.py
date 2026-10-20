#!/usr/bin/env python3
"""Build deterministic, story-preserving corpora for context-length comparisons.

This is a staging builder.  It does not create model checkpoints, launch a
runner, or modify any existing Crucible artifact.  It keeps the official
TinyStories train/validation files separate, packs consecutive sentence/phrase
units within each story, and selects a deterministic prefix whose raw UTF-8
source-byte budget is as close as possible to the frozen E16 reference.  The
resulting non-PAD target-token budget is reported separately because each
example contributes one EOS target.

The source files are deliberately referenced by logical paths in the records;
their absolute location and hashes are recorded in source_metadata.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


END_OF_TEXT = b"<|endoftext|>"
SENTENCE_BOUNDARY = re.compile(r'([.!?]+["”’»)\]]*)\s+')


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def iter_source_stories(path: Path) -> Iterator[Tuple[int, int, int, bytes]]:
    record_index = 0
    story_start: int | None = None
    story_lines: List[bytes] = []
    with path.open("rb") as handle:
        while True:
            line_offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_body = line.rstrip(b"\r\n")
            if line_body.strip() == END_OF_TEXT:
                if story_lines and story_start is not None:
                    yield record_index, story_start, line_offset, b"".join(story_lines)
                    record_index += 1
                story_start = None
                story_lines = []
                continue
            if story_start is None and not line_body.strip():
                continue
            if story_start is None:
                story_start = line_offset
            story_lines.append(line)
    if story_lines and story_start is not None:
        yield record_index, story_start, path.stat().st_size, b"".join(story_lines)


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


def utf8_length(text: str) -> int:
    return len(text.encode("utf-8"))


def sentence_spans(text: str) -> Iterator[Tuple[str, int, int, int]]:
    cursor = 0
    sentence_index = 0
    for boundary in SENTENCE_BOUNDARY.finditer(text):
        raw_part = text[cursor : boundary.start()] + boundary.group(1)
        part = raw_part.strip()
        if part:
            leading = len(raw_part) - len(raw_part.lstrip())
            start = cursor + leading
            yield part, start, start + len(part), sentence_index
            sentence_index += 1
        cursor = boundary.end()
    raw_part = text[cursor:]
    part = raw_part.strip()
    if part:
        leading = len(raw_part) - len(raw_part.lstrip())
        start = cursor + leading
        yield part, start, start + len(part), sentence_index


def split_phrase_units(
    phrase: str,
    phrase_start: int,
    sentence_index: int,
    next_unit_id: int,
    max_bytes: int,
) -> Tuple[List[Dict[str, object]], int]:
    words = list(re.finditer(r"\S+", phrase))
    units: List[Dict[str, object]] = []
    unit_id = next_unit_id
    current_start: int | None = None
    current_end: int | None = None

    def emit(start: int, end: int, wrapped: bool, long_word: bool) -> None:
        nonlocal unit_id
        text = phrase[start:end]
        if not text or utf8_length(text) > max_bytes:
            raise AssertionError("invalid phrase unit")
        units.append(
            {
                "unit_id": unit_id,
                "unit_label": f"sentence_{sentence_index}_unit_{unit_id}",
                "sentence_index": sentence_index,
                "text": text,
                "clean_char_start": phrase_start + start,
                "clean_char_end": phrase_start + end,
                "wrapped_from_oversized_sentence": wrapped,
                "split_long_word_by_codepoint": long_word,
            }
        )
        unit_id += 1

    for word_match in words:
        word_start, word_end = word_match.span()
        candidate_start = word_start if current_start is None else current_start
        candidate_end = word_end
        candidate = phrase[candidate_start:candidate_end]
        if utf8_length(candidate) <= max_bytes:
            current_start = candidate_start
            current_end = candidate_end
            continue

        if current_start is not None and current_end is not None:
            emit(current_start, current_end, True, False)
            current_start = None
            current_end = None

        word = phrase[word_start:word_end]
        if utf8_length(word) <= max_bytes:
            current_start = word_start
            current_end = word_end
            continue

        piece_start = word_start
        piece_end = word_start
        for character_offset in range(word_start, word_end):
            candidate_piece = phrase[piece_start : character_offset + 1]
            if utf8_length(candidate_piece) <= max_bytes:
                piece_end = character_offset + 1
            else:
                if piece_end <= piece_start:
                    raise AssertionError("failed to split long Unicode word")
                emit(piece_start, piece_end, True, True)
                piece_start = character_offset
                piece_end = character_offset + 1
        current_start = piece_start
        current_end = piece_end

    if current_start is not None and current_end is not None:
        emit(current_start, current_end, utf8_length(phrase) > max_bytes, False)
    return units, unit_id


def make_units(clean_text: str, max_bytes: int) -> List[Dict[str, object]]:
    units: List[Dict[str, object]] = []
    next_unit_id = 0
    for phrase, start, _end, sentence_index in sentence_spans(clean_text):
        phrase_units, next_unit_id = split_phrase_units(
            phrase, start, sentence_index, next_unit_id, max_bytes
        )
        units.extend(phrase_units)
    return units


def unit_byte_offsets(clean_text: str, unit: Dict[str, object]) -> Tuple[int, int]:
    start = int(unit["clean_char_start"])
    end = int(unit["clean_char_end"])
    return len(clean_text[:start].encode("utf-8")), len(clean_text[:end].encode("utf-8"))


def build_chunk(
    units: Sequence[Dict[str, object]],
    clean_text: str,
    story_index: int,
    source_start: int,
    source_end: int,
    story_sha: str,
    max_bytes: int,
    is_final: bool,
    tail_merge_applied: bool,
) -> Dict[str, object]:
    packed_text = " ".join(str(unit["text"]) for unit in units)
    byte_length = utf8_length(packed_text)
    if not packed_text or byte_length > max_bytes:
        raise AssertionError("packed chunk exceeds context byte limit")
    components = []
    for unit in units:
        component_start, component_end = unit_byte_offsets(clean_text, unit)
        component_text = str(unit["text"])
        components.append(
            {
                "component_unit_id": unit["unit_id"],
                "component_unit_label": unit["unit_label"],
                "sentence_index": unit["sentence_index"],
                "text_utf8_sha256": sha256_bytes(component_text.encode("utf-8")),
                "byte_length": utf8_length(component_text),
                "clean_story_char_start": unit["clean_char_start"],
                "clean_story_char_end": unit["clean_char_end"],
                "clean_story_utf8_byte_start": component_start,
                "clean_story_utf8_byte_end": component_end,
                "wrapped_from_oversized_sentence": unit[
                    "wrapped_from_oversized_sentence"
                ],
                "split_long_word_by_codepoint": unit[
                    "split_long_word_by_codepoint"
                ],
            }
        )
    return {
        "text": packed_text,
        "byte_length": byte_length,
        "source_record_index": story_index,
        "source_start_byte": source_start,
        "source_end_byte": source_end,
        "source_story_sha256": story_sha,
        "component_unit_ids": [unit["unit_id"] for unit in units],
        "component_count": len(units),
        "components": components,
        "is_story_final_chunk": is_final,
        "tail_merge_applied": tail_merge_applied,
        "unavoidable_final_tail": False,
        "packing_strategy": "greedy_consecutive_sentence_units_context_v1",
    }


def pack_story(
    clean_text: str,
    story_index: int,
    source_start: int,
    source_end: int,
    story_sha: str,
    max_bytes: int,
    preferred_floor: int,
) -> List[Dict[str, object]]:
    units = make_units(clean_text, max_bytes)
    if not units:
        return []
    chunks: List[Dict[str, object]] = []
    chunk_units: List[List[Dict[str, object]]] = []
    current: List[Dict[str, object]] = []
    for unit in units:
        candidate = current + [unit]
        if current and utf8_length(" ".join(str(item["text"]) for item in candidate)) > max_bytes:
            chunks.append(
                build_chunk(
                    current,
                    clean_text,
                    story_index,
                    source_start,
                    source_end,
                    story_sha,
                    max_bytes,
                    False,
                    False,
                )
            )
            chunk_units.append(list(current))
            current = [unit]
        else:
            current = candidate
    if current:
        chunks.append(
            build_chunk(
                current,
                clean_text,
                story_index,
                source_start,
                source_end,
                story_sha,
                max_bytes,
                True,
                False,
            )
        )
        chunk_units.append(list(current))

    if len(chunks) >= 2 and int(chunks[-1]["byte_length"]) < preferred_floor:
        previous = chunks[-2]
        tail = chunks[-1]
        merged_text = f"{previous['text']} {tail['text']}"
        if utf8_length(merged_text) <= max_bytes:
            merged_units = chunk_units[-2] + chunk_units[-1]
            chunks[-2] = build_chunk(
                merged_units,
                clean_text,
                story_index,
                source_start,
                source_end,
                story_sha,
                max_bytes,
                True,
                True,
            )
            chunks.pop()
            chunk_units[-2] = merged_units
            chunk_units.pop()

    for index, chunk in enumerate(chunks):
        chunk["is_story_final_chunk"] = index == len(chunks) - 1
        chunk["unavoidable_final_tail"] = (
            bool(chunk["is_story_final_chunk"])
            and int(chunk["byte_length"]) < preferred_floor
        )
    return chunks


def percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    lengths = [int(record["byte_length"]) for record in records]
    return {
        "count": len(lengths),
        "total_valid_utf8_bytes": sum(lengths),
        "total_nonpad_target_tokens": sum(lengths) + len(lengths),
        "min_bytes": min(lengths) if lengths else 0,
        "p05_bytes": percentile(lengths, 0.05),
        "p25_bytes": percentile(lengths, 0.25),
        "median_bytes": percentile(lengths, 0.50),
        "mean_bytes": statistics.fmean(lengths) if lengths else 0.0,
        "p75_bytes": percentile(lengths, 0.75),
        "p95_bytes": percentile(lengths, 0.95),
        "max_bytes": max(lengths) if lengths else 0,
        "below_thresholds": {
            str(threshold): {
                "count": sum(length < threshold for length in lengths),
                "percentage": (100.0 * sum(length < threshold for length in lengths) / len(lengths))
                if lengths
                else 0.0,
            }
            for threshold in (32, 64, 96, 112)
        },
        "exactly_max_bytes": sum(length == max(lengths) for length in lengths)
        if lengths
        else 0,
        "unavoidable_final_tails": sum(
            bool(record["unavoidable_final_tail"]) for record in records
        ),
        "tiny_standalone_chunks_below_16": sum(
            int(record["byte_length"]) < 16 and int(record["component_count"]) == 1
            for record in records
        ),
    }


def provenance_record(local_id: int, record: Dict[str, object]) -> Dict[str, object]:
    return {
        "schema_version": "crucible_context_provenance_v1",
        "local_split_id": local_id,
        "text_utf8_sha256": record["text_utf8_sha256"],
        "source_file": record["source_file"],
        "source_record_index": record["source_record_index"],
        "source_start_byte": record["source_start_byte"],
        "source_end_byte": record["source_end_byte"],
        "source_story_sha256": record["source_story_sha256"],
        "component_unit_ids": record["component_unit_ids"],
        "component_count": record["component_count"],
        "components": record["components"],
        "is_story_final_chunk": record["is_story_final_chunk"],
        "tail_merge_applied": record["tail_merge_applied"],
        "unavoidable_final_tail": record["unavoidable_final_tail"],
        "packing_strategy": record["packing_strategy"],
    }


def candidate_record(
    split_name: str,
    logical_source_file: str,
    local_id: int,
    chunk: Dict[str, object],
) -> Dict[str, object]:
    encoded = str(chunk["text"]).encode("utf-8")
    return {
        "example_index": local_id,
        "split": split_name,
        "text": chunk["text"],
        "text_utf8_sha256": sha256_bytes(encoded),
        "byte_length": chunk["byte_length"],
        "source_file": logical_source_file,
        "source_record_index": chunk["source_record_index"],
        "source_start_byte": chunk["source_start_byte"],
        "source_end_byte": chunk["source_end_byte"],
        "source_story_sha256": chunk["source_story_sha256"],
        "component_unit_ids": chunk["component_unit_ids"],
        "component_count": chunk["component_count"],
        "is_story_final_chunk": chunk["is_story_final_chunk"],
        "tail_merge_applied": chunk["tail_merge_applied"],
        "unavoidable_final_tail": chunk["unavoidable_final_tail"],
        "packing_strategy": chunk["packing_strategy"],
    }


def select_budgeted_split(
    source_path: Path,
    split_name: str,
    logical_source_file: str,
    max_bytes: int,
    preferred_floor: int,
    target_source_bytes: int,
    count_multiple: int,
    already_selected: set[bytes],
) -> Tuple[List[Dict[str, object]], Dict[str, object], set[bytes]]:
    records: List[Dict[str, object]] = []
    selected_texts: set[bytes] = set()
    scanned_stories = 0
    scanned_candidates = 0
    duplicates_within = 0
    duplicates_cross = 0
    source_story_bytes_scanned = 0
    long_word_splits = 0
    multiple_components = 0
    tail_merges = 0
    # The margin is large enough to find a nearby train prefix whose count is
    # divisible by 32, while still keeping the selection a deterministic prefix.
    search_margin = max(4 * max_bytes, count_multiple * max_bytes * 2)
    best_count = 0
    best_score: Tuple[int, int, int] | None = None
    running_source_bytes = 0

    def consider() -> None:
        nonlocal best_count, best_score
        if not records or (count_multiple > 1 and len(records) % count_multiple):
            return
        source_bytes = running_source_bytes
        score = (
            abs(source_bytes - target_source_bytes),
            0 if source_bytes <= target_source_bytes else 1,
            len(records),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_count = len(records)

    for story_index, source_start, source_end, raw_story in iter_source_stories(source_path):
        scanned_stories += 1
        source_story_bytes_scanned += len(raw_story)
        story_sha = sha256_bytes(raw_story)
        clean_text = clean_story(raw_story)
        chunks = pack_story(
            clean_text,
            story_index,
            source_start,
            source_end,
            story_sha,
            max_bytes,
            preferred_floor,
        )
        for chunk in chunks:
            scanned_candidates += 1
            text_bytes = str(chunk["text"]).encode("utf-8")
            long_word_splits += sum(
                bool(component["split_long_word_by_codepoint"])
                for component in chunk["components"]
            )
            if text_bytes in selected_texts:
                duplicates_within += 1
                continue
            if text_bytes in already_selected:
                duplicates_cross += 1
                continue
            record = candidate_record(
                split_name,
                logical_source_file,
                len(records),
                chunk,
            )
            # Components are deliberately kept only in the provenance sidecar;
            # the compact selected JSONL remains usable without duplicating them.
            record["_provenance"] = provenance_record(len(records), record | {"components": chunk["components"]})
            records.append(record)
            selected_texts.add(text_bytes)
            running_source_bytes += int(record["byte_length"])
            if int(record["component_count"]) > 1:
                multiple_components += 1
            if bool(record["tail_merge_applied"]):
                tail_merges += 1
            consider()
            if running_source_bytes >= target_source_bytes + search_margin:
                selected = records[:best_count]
                for local_id, item in enumerate(selected):
                    item["example_index"] = local_id
                    item["_provenance"]["local_split_id"] = local_id
                stats = {
                    "stories_scanned": scanned_stories,
                    "source_story_bytes_scanned": source_story_bytes_scanned,
                    "packed_candidates_scanned": scanned_candidates,
                    "duplicate_within_split_removed": duplicates_within,
                    "duplicate_against_previous_split_removed": duplicates_cross,
                    "packed_chunks_with_multiple_components": multiple_components,
                    "packed_chunks_with_tail_merge": tail_merges,
                    "long_word_split_components": long_word_splits,
                    "selection_search_margin_source_bytes": search_margin,
                    "target_source_bytes": target_source_bytes,
                    "selected_source_bytes": sum(int(item["byte_length"]) for item in selected),
                    "selected_nonpad_tokens": sum(int(item["byte_length"]) for item in selected) + len(selected),
                    "selected_prefix_candidate_count": best_count,
                }
                return selected, stats, {
                    str(item["text"]).encode("utf-8") for item in selected
                }

    if best_count == 0:
        raise RuntimeError(f"{split_name} source exhausted before target budget")
    selected = records[:best_count]
    for local_id, item in enumerate(selected):
        item["example_index"] = local_id
        item["_provenance"]["local_split_id"] = local_id
    stats = {
        "stories_scanned": scanned_stories,
        "source_story_bytes_scanned": source_story_bytes_scanned,
        "packed_candidates_scanned": scanned_candidates,
        "duplicate_within_split_removed": duplicates_within,
        "duplicate_against_previous_split_removed": duplicates_cross,
        "packed_chunks_with_multiple_components": multiple_components,
        "packed_chunks_with_tail_merge": tail_merges,
        "long_word_split_components": long_word_splits,
        "selection_search_margin_source_bytes": search_margin,
        "target_source_bytes": target_source_bytes,
        "selected_source_bytes": sum(int(item["byte_length"]) for item in selected),
        "selected_nonpad_tokens": sum(int(item["byte_length"]) for item in selected) + len(selected),
        "selected_prefix_candidate_count": best_count,
    }
    return selected, stats, {
        str(item["text"]).encode("utf-8") for item in selected
    }


def strip_internal(record: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> str:
    data = (
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in records
        )
    ).encode("utf-8")
    atomic_write(path, data)
    return sha256_bytes(data)


def write_text(path: Path, records: Sequence[Dict[str, object]]) -> str:
    data = "".join(str(record["text"]) + "\n" for record in records).encode("utf-8")
    atomic_write(path, data)
    return sha256_bytes(data)


def write_split(
    output_dir: Path,
    split_name: str,
    records: Sequence[Dict[str, object]],
) -> Dict[str, str]:
    build_dir = output_dir / "build"
    compact = [strip_internal(record) for record in records]
    provenance = [record["_provenance"] for record in records]
    return {
        "jsonl_sha256": write_jsonl(build_dir / f"packed_{split_name}.jsonl", compact),
        "text_sha256": write_text(build_dir / f"packed_{split_name}.txt", compact),
        "provenance_sha256": write_jsonl(
            build_dir / f"provenance_{split_name}.jsonl", provenance
        ),
    }


def split_report(
    records: Sequence[Dict[str, object]],
    stats: Dict[str, object],
    hashes: Dict[str, str],
) -> Dict[str, object]:
    return {
        **stats,
        "selected_unique_examples": len(records),
        "distribution": distribution(records),
        **hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--train-target-source-bytes", type=int, required=True)
    parser.add_argument("--validation-target-source-bytes", type=int, required=True)
    parser.add_argument("--train-count-multiple", type=int, default=32)
    args = parser.parse_args()

    if args.context_length < 4:
        raise ValueError("context length is too small")
    output_root = args.output_root.resolve()
    source_root = args.source_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty staging directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    train_source = source_root / "TinyStoriesV2-GPT4-train.txt"
    validation_source = source_root / "TinyStoriesV2-GPT4-valid.txt"
    if not train_source.is_file() or not validation_source.is_file():
        raise FileNotFoundError("official TinyStoriesV2-GPT4 source files are required")

    max_bytes = args.context_length - 1
    preferred_floor = (3 * args.context_length) // 4
    train, train_stats, train_texts = select_budgeted_split(
        train_source,
        "train",
        "source/TinyStoriesV2-GPT4-train.txt",
        max_bytes,
        preferred_floor,
        args.train_target_source_bytes,
        args.train_count_multiple,
        set(),
    )
    validation, validation_stats, validation_texts = select_budgeted_split(
        validation_source,
        "validation",
        "source/TinyStoriesV2-GPT4-valid.txt",
        max_bytes,
        preferred_floor,
        args.validation_target_source_bytes,
        1,
        train_texts,
    )
    cross_split = train_texts & validation_texts
    if cross_split:
        raise AssertionError("cross-split exact duplicates remain")

    source_metadata = {
        "dataset_name": "TinyStoriesV2-GPT4 context-length staging corpus",
        "language": "English",
        "license": "cdla-sharing-1.0",
        "upstream_repository": "roneneldan/TinyStories",
        "upstream_revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "source_root": str(source_root),
        "source_files": [
            {
                "split": "train",
                "local_path": str(train_source),
                "bytes": train_source.stat().st_size,
                "sha256": sha256_file(train_source),
            },
            {
                "split": "validation",
                "local_path": str(validation_source),
                "bytes": validation_source.stat().st_size,
                "sha256": sha256_file(validation_source),
            },
        ],
        "packing": {
            "context_length": args.context_length,
            "max_source_utf8_bytes": max_bytes,
            "preferred_floor_bytes": preferred_floor,
            "strategy": "greedy consecutive sentence/phrase units within story",
            "final_tail_merge": "merge backward when result remains within context byte limit",
            "official_split_preserved": True,
            "exact_deduplication": "after packing, independently within split",
            "cross_split_exact_duplicates": len(cross_split),
        },
    }
    source_bytes = json.dumps(source_metadata, indent=2, sort_keys=True).encode("utf-8")
    atomic_write(output_root / "source_metadata.json", source_bytes)

    hashes_train = write_split(output_root, "train", train)
    hashes_validation = write_split(output_root, "validation", validation)
    report = {
        "status": "staged",
        "context_length": args.context_length,
        "max_source_utf8_bytes": max_bytes,
        "preferred_floor_bytes": preferred_floor,
        "budget_basis": "raw_utf8_source_bytes",
        "preprocessor": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "truncation_count": 0,
        },
        "source": source_metadata,
        "splits": {
            "train": split_report(train, train_stats, hashes_train),
            "validation": split_report(validation, validation_stats, hashes_validation),
        },
        "deduplication": {
            "train_exact_duplicates_removed": train_stats["duplicate_within_split_removed"],
            "validation_exact_duplicates_removed": validation_stats["duplicate_within_split_removed"],
            "validation_cross_split_duplicates_removed": validation_stats[
                "duplicate_against_previous_split_removed"
            ],
            "final_cross_split_exact_duplicates": len(cross_split),
        },
        "integrity": {
            "utf8_decode_failures": 0,
            "oversized_chunks": 0,
            "truncation_count": 0,
            "provenance_records": len(train) + len(validation),
        },
    }
    report_bytes = json.dumps(report, indent=2, sort_keys=True).encode("utf-8")
    atomic_write(output_root / "context_dataset_statistics.json", report_bytes)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
