#!/usr/bin/env python3
"""Pack a bounded FineWeb-Edu shard for the Crucible byte-level LM.

Raw Parquet remains outside Git.  The packer records the source hash, filter,
split rule and packed-array hashes so a local foundation run is replayable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pyarrow.parquet as pq


RAW_BYTE_START = 3
EOS_ID = 2
SCHEMA = "crucible-fineweb-foundation-pack-v1-20261029"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_split(identifier: str, validation_denominator: int) -> str:
    value = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16)
    return "validation" if value % validation_denominator == 0 else "train"


def encode_document(text: str) -> np.ndarray:
    payload = bytes([RAW_BYTE_START + value for value in text.encode("utf-8")])
    return np.frombuffer(payload + bytes([EOS_ID]), dtype=np.uint8).astype(np.uint16)


def pack_stream(tokens: bytearray, context: int, path: Path) -> Dict[str, Any]:
    width = context + 1
    usable = (len(tokens) // width) * width
    # The in-memory stream uses one byte per token ID; materialize explicitly
    # because the model vocabulary includes IDs above the uint8 range.
    array = np.fromiter(tokens[:usable], dtype=np.uint16, count=usable).reshape(-1, width)
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    return {
        "path": path.name,
        "dtype": "uint16",
        "shape": [int(array.shape[0]), int(array.shape[1])],
        "tokens": int(array.size),
        "discarded_tokens": int(len(tokens) - usable),
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--train-tokens", type=int, default=20_000_000)
    parser.add_argument("--validation-tokens", type=int, default=2_000_000)
    parser.add_argument("--min-score", type=float, default=3.0)
    parser.add_argument("--min-chars", type=int, default=256)
    parser.add_argument("--max-chars", type=int, default=40_000)
    parser.add_argument("--validation-denominator", type=int, default=20)
    args = parser.parse_args()

    source = args.parquet.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train = bytearray()
    validation = bytearray()
    seen = accepted = 0
    accepted_by_split = {"train": 0, "validation": 0}
    bytes_by_split = {"train": 0, "validation": 0}

    parquet = pq.ParquetFile(source)
    columns = ["id", "text", "language", "score"]
    for batch in parquet.iter_batches(columns=columns, batch_size=4096):
        rows = batch.to_pylist()
        for row in rows:
            seen += 1
            text = str(row.get("text") or "").strip()
            if row.get("language") != "en":
                continue
            if float(row.get("score") or 0.0) < args.min_score:
                continue
            if not (args.min_chars <= len(text) <= args.max_chars):
                continue
            split = row_split(str(row.get("id") or seen), args.validation_denominator)
            encoded = encode_document(text)
            destination = train if split == "train" else validation
            limit = args.train_tokens if split == "train" else args.validation_tokens
            if len(destination) >= limit:
                continue
            destination.extend(encoded[: max(0, limit - len(destination))])
            accepted += 1
            accepted_by_split[split] += 1
            bytes_by_split[split] = len(destination)
        if len(train) >= args.train_tokens and len(validation) >= args.validation_tokens:
            break

    train_info = pack_stream(train, args.context, output / "fineweb_train.bin")
    validation_info = pack_stream(validation, args.context, output / "fineweb_validation.bin")
    manifest = {
        "schema_version": SCHEMA,
        "source": {
            "path": "<external-fineweb-shard>",
            "sha256": sha256_file(source),
            "rows_in_source": int(parquet.metadata.num_rows),
            "license_record": "ODC-By as published by HuggingFaceFW/fineweb-edu",
        },
        "filter": {
            "language": "en",
            "min_score": args.min_score,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
        },
        "split": {
            "rule": "sha256(id) modulo validation_denominator",
            "validation_denominator": args.validation_denominator,
        },
        "packing": {
            "context": args.context,
            "token_ids": "raw UTF-8 bytes + 3; EOS=2; PAD=0/BOS=1 reserved",
            "target_train_tokens": args.train_tokens,
            "target_validation_tokens": args.validation_tokens,
        },
        "counts": {
            "rows_seen": seen,
            "rows_accepted": accepted,
            "rows_accepted_by_split": accepted_by_split,
            "bytes_by_split": bytes_by_split,
        },
        "packed_files": {
            "train": train_info,
            "validation": validation_info,
        },
        "raw_data_committed_to_git": False,
        "interpretation_boundary": "This is a bounded language-foundation source pack, not a completed pretraining corpus or capability result.",
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
