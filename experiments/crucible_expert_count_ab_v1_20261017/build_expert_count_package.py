#!/usr/bin/env python3
"""Build the isolated Crucible E=1/2/4 expert-count package.

This builder consumes the frozen 100K packed dataset package and creates new
variant manifests and architecture-specific initial anchors.  It never writes
to the source package.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path(
    "/home/timbushnell/crucible_large_tinystories_v2_packed_v1_20261015"
)
SOURCE_MANIFEST_DIR = SOURCE_ROOT / "crucible_manifest"
SOURCE_CONFIG = SOURCE_MANIFEST_DIR / "experiment_config.json"
MODEL_SOURCE = Path(
    "/home/timbushnell/crucible_topk_preflight_v1_20261017"
    "/model_core_generalized.py"
)

sys.path.insert(0, str(MODEL_SOURCE.parent))
from model_core_generalized import BraidPrototypeLM  # noqa: E402


VARIANTS = (
    {
        "id": "fixed_total_e1",
        "regime": "fixed_total_parameters",
        "num_experts": 1,
        "top_k": 1,
        "d_ff": 13788,
    },
    {
        "id": "fixed_total_e2",
        "regime": "fixed_total_parameters",
        "num_experts": 2,
        "top_k": 2,
        "d_ff": 6894,
    },
    {
        "id": "reference_e4",
        "regime": "shared_reference",
        "num_experts": 4,
        "top_k": 2,
        "d_ff": 3447,
    },
    {
        "id": "fixed_active_e1",
        "regime": "fixed_active_compute",
        "num_experts": 1,
        "top_k": 1,
        "d_ff": 6894,
    },
    {
        "id": "fixed_active_e2",
        "regime": "fixed_active_compute",
        "num_experts": 2,
        "top_k": 2,
        "d_ff": 3447,
    },
)

COPY_FILES = (
    "stories_train.bin",
    "stories_validation.bin",
    "schedule_seed_42.bin",
    "schedule_seed_1337.bin",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, payload)


def anchor_bytes(seed: int, variant: dict[str, Any], source_hp: dict[str, Any]) -> bytes:
    torch.manual_seed(seed)
    model = BraidPrototypeLM(
        source_hp["vocab_size"],
        source_hp["n_layers"],
        source_hp["d_model"],
        source_hp["n_heads"],
        source_hp["latent_dim_kv"],
        source_hp["latent_dim_q"],
        source_hp["pe_dim"],
        variant["num_experts"],
        variant["d_ff"],
        top_k=variant["top_k"],
    )
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def parameter_count(variant: dict[str, Any], source_hp: dict[str, Any]) -> int:
    torch.manual_seed(0)
    model = BraidPrototypeLM(
        source_hp["vocab_size"],
        source_hp["n_layers"],
        source_hp["d_model"],
        source_hp["n_heads"],
        source_hp["latent_dim_kv"],
        source_hp["latent_dim_q"],
        source_hp["pe_dim"],
        variant["num_experts"],
        variant["d_ff"],
        top_k=variant["top_k"],
    )
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> int:
    if not SOURCE_CONFIG.is_file():
        raise FileNotFoundError(SOURCE_CONFIG)
    if not MODEL_SOURCE.is_file():
        raise FileNotFoundError(MODEL_SOURCE)
    for required in (ROOT / "model_core.py", ROOT / "crucible_runner.py"):
        if not required.is_file():
            raise FileNotFoundError(required)

    source_manifest = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    source_manifest_sha = sha256_file(SOURCE_CONFIG)
    source_hp = source_manifest["hyperparameters"]
    model_sha = sha256_file(ROOT / "model_core.py")
    runner_sha = sha256_file(ROOT / "crucible_runner.py")
    builder_sha = sha256_file(Path(__file__).resolve())

    common = {
        "source_manifest_sha256": source_manifest_sha,
        "source_dataset_root": str(SOURCE_ROOT),
        "model_core_sha256": model_sha,
        "runner_sha256": runner_sha,
        "builder_sha256": builder_sha,
        "variants": [],
    }

    for variant in VARIANTS:
        variant_dir = ROOT / variant["id"]
        manifest_dir = variant_dir / "crucible_manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        for filename in COPY_FILES:
            shutil.copy2(SOURCE_MANIFEST_DIR / filename, manifest_dir / filename)

        hp = dict(source_hp)
        hp.update(
            {
                "num_experts": variant["num_experts"],
                "top_k": variant["top_k"],
                "d_ff": variant["d_ff"],
                "expected_parameter_geometry": parameter_count(variant, source_hp),
                "expert_count_variant": variant["id"],
            }
        )

        config = json.loads(json.dumps(source_manifest))
        config["meta"].update(
            {
                "dataset_name": (
                    "Crucible TinyStoriesV2-GPT4 expert-count "
                    f"{variant['id']}"
                ),
                "experiment_family": "crucible_expert_count_v1",
                "variant_id": variant["id"],
                "source_manifest_sha256": source_manifest_sha,
                "training_started": False,
            }
        )
        config["contract"] = {
            "architectural_branches": ["B"],
            "learning_rate_candidates": [0.0003],
            "seeds": [42, 1337],
            "selection_note": (
                "Branch B held fixed while comparing expert-count regimes; "
                "one packed 100K corpus pass"
            ),
        }
        config["hyperparameters"] = hp
        config["code_provenance"].update(
            {
                "model_core_sha256": model_sha,
                "runner_sha256": runner_sha,
                "package_builder_sha256": builder_sha,
                "source_manifest_compiler_sha256": config["code_provenance"].get(
                    "manifest_compiler_sha256"
                ),
            }
        )
        config["experiment_variant"] = {
            "id": variant["id"],
            "regime": variant["regime"],
            "expert_count": variant["num_experts"],
            "top_k": variant["top_k"],
            "expert_hidden_width": variant["d_ff"],
            "reference_shared_with": (
                ["fixed_total_parameters", "fixed_active_compute"]
                if variant["id"] == "reference_e4"
                else []
            ),
        }

        anchor_hashes = {}
        for seed in config["contract"]["seeds"]:
            payload = anchor_bytes(seed, variant, source_hp)
            name = f"initial_seed_anchor_{seed}.pt"
            atomic_write_bytes(manifest_dir / name, payload)
            anchor_hash = hashlib.sha256(payload).hexdigest()
            anchor_hashes[str(seed)] = anchor_hash
            config["anchors"][str(seed)] = {
                "relative_path": name,
                "binary_payload_sha256": anchor_hash,
            }

        config_path = manifest_dir / "experiment_config.json"
        atomic_write_json(config_path, config)
        manifest_sha = sha256_file(config_path)
        (variant_dir / "manifest_sha256.txt").write_text(
            manifest_sha + "\n", encoding="ascii"
        )

        files = {}
        for path in sorted(variant_dir.rglob("*")):
            if path.is_file() and path.name != "manifest_sha256.txt":
                files[str(path.relative_to(variant_dir))] = sha256_file(path)
        atomic_write_json(variant_dir / "variant_hashes.json", files)

        common["variants"].append(
            {
                "id": variant["id"],
                "regime": variant["regime"],
                "num_experts": variant["num_experts"],
                "top_k": variant["top_k"],
                "d_ff": variant["d_ff"],
                "parameter_count": hp["expected_parameter_geometry"],
                "manifest_sha256": manifest_sha,
                "anchor_sha256": anchor_hashes,
            }
        )

    atomic_write_json(ROOT / "package_plan.json", common)
    print(json.dumps(common, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
