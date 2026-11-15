"""Run the isolated, frozen AFS-D embedding-to-depth control."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import platform
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bsg.config import load_config
from bsg.data.c7 import build_dataset as build_c7, transform_state as c7_transform
from bsg.data.d4 import build_dataset as build_d4, transform_state as d4_transform
from bsg.data.splits import split_templates_three_way
from bsg.experiments.afs_d_outputs import (
    bootstrap_depth_contrasts,
    classify_outcomes,
    pair_depth_contrasts,
    prepare_location_metrics,
    write_stable_outputs,
)
from bsg.experiments.algebra_falsification_units import (
    generator_composition_rows,
    stable_cell_seed,
    verify_embedding_architecture,
)
from bsg.experiments.canonical_units import evaluate_operators, fit_base_artifact
from bsg.experiments.fitting import GroupSpec
from bsg.experiments.main_config import CANONICAL_DEPTH_LAYERS, CANONICAL_REVISIONS
from bsg.experiments.main_outputs import artifact_checksums
from bsg.experiments.manifest import git_state, package_versions
from bsg.experiments.runtime_state import (
    RunLedger,
    RuntimeTelemetry,
    atomic_write_frame,
    atomic_write_json,
    atomic_write_text,
    experiment_key,
    sha256_file,
)
from bsg.groups import c7, d4
from bsg.metrics.relations import c7_relations, d4_relations
from bsg.representations.cache import activation_cache_key
from bsg.representations.extract import extract_embedding_representations, load_tokenizer_and_model


EXPERIMENT_NAME = "beyond_shared_geometry_afs_d_rank_aware"
EXPECTED_MODELS = {
    "llama_1b": "meta-llama/Llama-3.2-1B",
    "qwen_1_5b": "Qwen/Qwen2.5-1.5B",
}
EXPECTED_GROUPS = ["C7", "D4"]
EXPECTED_DEPTHS = [0.2, 0.4, 0.6, 0.8]
EXPECTED_SPLITS = [101, 303, 909, 2701, 4243]
EXPECTED_TEMPLATE_SPLIT = {"train_templates": 28, "validation_templates": 6, "test_templates": 6}
EXPECTED_GROUP_DIMS = {"C7": 6, "D4": 3}
EXPECTED_EMBEDDING_RANKS = {"C7": 6, "D4": 3}


def group_specs() -> dict[str, GroupSpec]:
    return {
        "C7": GroupSpec("C7", build_c7(), c7.ELEMENTS, c7.IDENTITY, c7.compose, c7.inverse, c7_transform, c7_relations()),
        "D4": GroupSpec("D4", build_d4(), d4.ELEMENTS, d4.IDENTITY, d4.compose, d4.inverse, d4_transform, d4_relations()),
    }


def template_split(spec: GroupSpec, config: dict, seed: int):
    counts = config["template_split"]
    return split_templates_three_way(
        spec.frame,
        seed=int(seed),
        train_templates=int(counts["train_templates"]),
        validation_templates=int(counts["validation_templates"]),
        test_templates=int(counts["test_templates"]),
    )


def resolve_config(path) -> dict:
    config = copy.deepcopy(load_config(path))
    selected = []
    for key, model in config.get("models", {}).items():
        entry = copy.deepcopy(model)
        entry["key"] = key
        entry["depth_layers"] = {round(float(depth), 2): int(layer) for depth, layer in entry["depth_layers"].items()}
        entry["depth_slots"] = list(config["depth_slots"])
        entry["primary_layers"] = [entry["depth_layers"][round(float(depth), 2)] for depth in config["depth_slots"]]
        entry["depth_slot_by_layer"] = {str(layer): depth for depth, layer in entry["depth_layers"].items()}
        selected.append(entry)
    config["selected_models"] = selected
    config["depth_slots"] = [round(float(depth), 2) for depth in config["depth_slots"]]
    config["split_seeds"] = list(map(int, config["split_seeds"]))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("AFS-D experiment namespace changed")
    if config.get("groups") != EXPECTED_GROUPS:
        raise ValueError("AFS-D groups must be C7 and D4")
    if config.get("template_split") != EXPECTED_TEMPLATE_SPLIT:
        raise ValueError("AFS-D template split must remain 28/6/6")
    if list(map(int, config.get("split_seeds", []))) != EXPECTED_SPLITS:
        raise ValueError("AFS-D split seeds changed")
    if [round(float(value), 2) for value in config.get("depth_slots", [])] != EXPECTED_DEPTHS:
        raise ValueError("AFS-D frozen depth slots changed")
    dims = {str(key): int(value) for key, value in config.get("group_latent_dimensions", {}).items()}
    if dims != EXPECTED_GROUP_DIMS:
        raise ValueError("AFS-D repaired group latent dimensions must be C7=6 and D4=3")
    ranks = {str(key): int(value) for key, value in config.get("expected_embedding_centered_ranks", {}).items()}
    if ranks != EXPECTED_EMBEDDING_RANKS:
        raise ValueError("AFS-D repaired expected embedding ranks must be C7=6 and D4=3")
    if float(config.get("ridge_alpha", -1)) != 0.01:
        raise ValueError("AFS-D ridge coefficient changed")
    if int(config.get("bootstrap_samples", -1)) != 5000 or int(config.get("bootstrap_seed", -1)) != 314159:
        raise ValueError("AFS-D frozen bootstrap changed")
    if float(config.get("confidence_level", -1)) != 0.95:
        raise ValueError("AFS-D confidence level changed")
    if config.get("embedding_convention") != "Hugging Face hidden_states[0], before transformer block 1; mean over the unique target-token span":
        raise ValueError("AFS-D embedding convention changed")
    internal = config.get("rank_matched_internal_inputs", {})
    if internal.get("source") != "main_full_003_activation_cache":
        raise ValueError("AFS-D repaired internal source must be the immutable FULL activation cache")
    if internal.get("activation_extraction_convention") != "mean over the unique target-token span at transformer block output":
        raise ValueError("AFS-D internal activation convention changed")
    digest = str(internal.get("archive_sha256", ""))
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("AFS-D internal activation archive sha256 must be frozen")
    observed_models = {model["key"]: model["id"] for model in config.get("selected_models", [])}
    if observed_models != EXPECTED_MODELS:
        raise ValueError("AFS-D must contain exactly the frozen two-model panel")
    for model in config["selected_models"]:
        if model["revision"] != CANONICAL_REVISIONS[model["id"]]:
            raise ValueError(f"AFS-D revision changed for {model['id']}")
        if model["depth_layers"] != CANONICAL_DEPTH_LAYERS[model["id"]]:
            raise ValueError(f"AFS-D depth mapping changed for {model['id']}")
        if int(model["blocks"]) not in {16, 28}:
            raise ValueError("AFS-D block count changed")
        expected_architecture = "llama" if model["id"].startswith("meta-llama/") else "qwen2"
        if model.get("architecture") != expected_architecture:
            raise ValueError("AFS-D model architecture changed")
    metrics = config.get("metrics", {})
    if metrics.get("co_primary") != ["E_gen_eq", "E_rel_data"]:
        raise ValueError("AFS-D co-primary metrics changed")
    if metrics.get("supporting") != ["E_hom_data", "E_comp_data"]:
        raise ValueError("AFS-D supporting metrics changed")
    if metrics.get("secondary") != ["E_eq", "E_hom_op"]:
        raise ValueError("AFS-D secondary metrics changed")


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def matrix(config: dict) -> dict:
    models = len(config["selected_models"])
    groups = len(config["groups"])
    splits = len(config["split_seeds"])
    depths = len(config["depth_slots"])
    return {
        "models": models,
        "groups": groups,
        "split_seeds": splits,
        "representation_locations": 1 + depths,
        "embedding_extraction_sets": models * groups,
        "embedding_rank_diagnostic_cells": models * groups * splits,
        "embedding_analysis_cells": models * groups * splits,
        "rank_matched_internal_cells": models * groups * depths * splits,
        "paired_cells": models * groups * (1 + depths) * splits,
    }


def group_latent_dim(config: dict, group: str) -> int:
    return int(config["group_latent_dimensions"][str(group)])


class FrozenMetricSources:
    """Read-only, checksum-verified views of FULL and AFS-A/B/C metrics."""

    def __init__(self, full_root: Path, afs_root: Path, config: dict):
        self.full_root = Path(full_root).resolve()
        self.afs_root = Path(afs_root).resolve()
        self.config = config
        if self.full_root == self.afs_root:
            raise ValueError("FULL and AFS-A/B/C roots must be distinct")
        self._verify_source("full", self.full_root)
        self._verify_source("afs_abc", self.afs_root)
        self._validate_manifests()
        self._snapshots = {
            "full": self._snapshot(self.full_root),
            "afs_abc": self._snapshot(self.afs_root),
        }

    @staticmethod
    def _snapshot(root: Path) -> dict:
        return {
            str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()
        }

    def _verify_source(self, name: str, root: Path) -> None:
        if not root.is_dir():
            raise FileNotFoundError(f"AFS-D immutable {name} root not found: {root}")
        expected = self.config["immutable_sources"][name]
        for relative, digest in expected["files"].items():
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"AFS-D immutable source file missing: {path}")
            if sha256_file(path) != digest:
                raise ValueError(f"AFS-D immutable source checksum mismatch: {path}")

    def _validate_manifests(self) -> None:
        for name, root in (("full", self.full_root), ("afs_abc", self.afs_root)):
            expected = self.config["immutable_sources"][name]
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            for field in ("run_id", "experiment", "tier", "status", "git_commit", "config_hash"):
                if manifest.get(field) != expected.get(field):
                    raise ValueError(f"AFS-D immutable {name} {field} mismatch")
            if manifest.get("git_dirty") is not False:
                raise ValueError(f"AFS-D immutable {name} source was dirty")
            if name == "afs_abc" and manifest.get("execution_modules") != expected["execution_modules"]:
                raise ValueError("AFS-D AFS-A/B/C module provenance mismatch")

    def assert_unchanged(self) -> None:
        if self._snapshot(self.full_root) != self._snapshots["full"]:
            raise RuntimeError("immutable main_full_003 changed during AFS-D")
        if self._snapshot(self.afs_root) != self._snapshots["afs_abc"]:
            raise RuntimeError("immutable afs_full_abc_001 changed during AFS-D")

    def source_provenance(self) -> dict:
        result = {}
        for name, root in (("full", self.full_root), ("afs_abc", self.afs_root)):
            expected = self.config["immutable_sources"][name]
            result[name] = {
                "root": str(root),
                "run_id": expected["run_id"],
                "git_commit": expected["git_commit"],
                "config_hash": expected["config_hash"],
                "verified_files": dict(expected["files"]),
            }
        return result

    def load_internal_metrics(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        operator = pd.read_csv(self.full_root / "structural" / "template_metrics.csv")
        generator = pd.read_csv(self.afs_root / "afs" / "generator_composition_metrics.csv")
        operator = self._select_operator(operator)
        generator = self._select_generator(generator)
        self.assert_unchanged()
        return operator, generator

    def _select_operator(self, frame: pd.DataFrame) -> pd.DataFrame:
        selected_models = {model["id"]: model for model in self.config["selected_models"]}
        depth = pd.to_numeric(frame["depth_slot"], errors="coerce")
        selected = frame[
            frame["model"].isin(selected_models)
            & frame["group"].isin(self.config["groups"])
            & depth.round(2).isin(self.config["depth_slots"])
            & frame["split_seed"].isin(self.config["split_seeds"])
            & (frame["latent_dim"] == 16)
            & (frame["method"] == "independent_ridge")
            & frame["data_split"].isin(["validation", "test"])
        ].copy()
        selected["depth_slot"] = pd.to_numeric(selected["depth_slot"]).round(2)
        selected["model_family"] = selected["model"].map(
            lambda value: "Llama" if str(value).startswith("meta-llama/") else "Qwen"
        )
        selected["representation_location"] = "internal"
        selected["source_run"] = "main_full_003"
        self._validate_internal_cells(selected, selected_models, rows_per_cell=12)
        return selected.reset_index(drop=True)

    def _select_generator(self, frame: pd.DataFrame) -> pd.DataFrame:
        selected_models = {model["id"]: model for model in self.config["selected_models"]}
        depth = pd.to_numeric(frame["depth_slot"], errors="coerce")
        selected = frame[
            frame["model"].isin(selected_models)
            & frame["group"].isin(self.config["groups"])
            & depth.round(2).isin(self.config["depth_slots"])
            & frame["split_seed"].isin(self.config["split_seeds"])
            & (frame["latent_dim"] == 16)
        ].copy()
        selected["depth_slot"] = pd.to_numeric(selected["depth_slot"]).round(2)
        selected["representation_location"] = "internal"
        selected["source_run"] = "afs_full_abc_001"
        self._validate_internal_cells(selected, selected_models, rows_per_cell=None)
        if set(selected[selected["evaluation_role"] == "unseen_composite"]["method"].unique()) != {
            "direct_ridge", "generator_composed", "identity", "matched_random_generator", "matched_wrong_word"
        }:
            methods_by_group = selected[selected["evaluation_role"] == "unseen_composite"].groupby("group")["method"].apply(set)
            if methods_by_group.get("C7") != {"direct_ridge", "generator_composed", "identity", "matched_random_generator"}:
                raise ValueError("AFS-D imported C7 generator controls changed")
            if methods_by_group.get("D4") != {"direct_ridge", "generator_composed", "identity", "matched_random_generator", "matched_wrong_word"}:
                raise ValueError("AFS-D imported D4 generator controls changed")
        return selected.reset_index(drop=True)

    def _validate_internal_cells(self, frame: pd.DataFrame, selected_models: dict, rows_per_cell: int | None) -> None:
        keys = ["model", "group", "depth_slot", "split_seed"]
        observed = set(frame[keys].drop_duplicates().itertuples(index=False, name=None))
        expected = {
            (model, group, depth, seed)
            for model in selected_models for group in self.config["groups"]
            for depth in self.config["depth_slots"] for seed in self.config["split_seeds"]
        }
        if observed != expected:
            raise ValueError(f"AFS-D internal source coverage mismatch: missing={len(expected-observed)} extra={len(observed-expected)}")
        if rows_per_cell is not None and not (frame.groupby(keys).size() == rows_per_cell).all():
            raise ValueError("AFS-D internal operator row count changed")
        for model, metadata in selected_models.items():
            rows = frame[frame["model"] == model]
            for depth, layer in metadata["depth_layers"].items():
                if set(rows[rows["depth_slot"] == depth]["layer"].astype(int).unique()) != {int(layer)}:
                    raise ValueError(f"AFS-D internal layer mapping mismatch for {model} depth {depth}")


class InternalActivationSource:
    """Read-only FULL activation-cache source for AFS-D rank-matched internal fits."""

    def __init__(self, source_path: Path, config: dict):
        self.source_path = Path(source_path)
        self.config = config
        self.archive = zipfile.ZipFile(self.source_path) if self.source_path.is_file() else None
        if self.archive is None and not self.source_path.is_dir():
            raise FileNotFoundError(f"AFS-D internal activation source not found: {self.source_path}")
        if self.archive is not None:
            expected = config["rank_matched_internal_inputs"]["archive_sha256"]
            observed = sha256_file(self.source_path)
            if observed != expected:
                raise ValueError(f"AFS-D internal activation archive checksum mismatch: {self.source_path}")
            self.cache_subdir = str(config["rank_matched_internal_inputs"].get("cache_subdir", "")).strip("/")
        else:
            self.cache_subdir = ""

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()

    def _member_name(self, key: str) -> str:
        name = f"{key}.npz"
        return f"{self.cache_subdir}/{name}" if self.cache_subdir else name

    def has(self, key: str) -> bool:
        if self.archive is not None:
            return self._member_name(key) in set(self.archive.namelist())
        return (self.source_path / f"{key}.npz").is_file()

    def load(self, key: str, expected_metadata: dict) -> np.ndarray:
        if self.archive is not None:
            member = self._member_name(key)
            try:
                payload = self.archive.read(member)
            except KeyError as error:
                raise FileNotFoundError(f"AFS-D internal activation cache missing: {member}") from error
            path_label = f"{self.source_path}!{member}"
            handle = io.BytesIO(payload)
        else:
            path = self.source_path / f"{key}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"AFS-D internal activation cache missing: {path}")
            path_label = str(path)
            handle = path
        with np.load(handle, allow_pickle=False) as data:
            if "activations" not in data or "metadata" not in data:
                raise ValueError(f"AFS-D internal activation cache schema mismatch: {path_label}")
            metadata = json.loads(str(data["metadata"].item()))
            if metadata != expected_metadata:
                raise ValueError(f"AFS-D internal activation metadata mismatch: {path_label}")
            return np.asarray(data["activations"], dtype=np.float32)

    def member_hash(self, key: str) -> str | None:
        if self.archive is not None:
            member = self._member_name(key)
            try:
                payload = self.archive.read(member)
            except KeyError:
                return None
            return hashlib.sha256(payload).hexdigest()
        path = self.source_path / f"{key}.npz"
        return sha256_file(path) if path.is_file() else None

    def provenance(self) -> dict:
        data = {
            "source_path": str(self.source_path),
            "source_type": "zip" if self.archive is not None else "directory",
            "source": self.config["rank_matched_internal_inputs"]["source"],
            "activation_extraction_convention": self.config["rank_matched_internal_inputs"]["activation_extraction_convention"],
        }
        if self.archive is not None:
            data["archive_sha256"] = sha256_file(self.source_path)
            data["cache_subdir"] = self.cache_subdir
        return data


def _internal_cache_metadata(config: dict, spec: GroupSpec, model_meta: dict, group: str, layer: int) -> dict:
    split_identity = {
        "kind": "template_pool",
        "template_ids": sorted(map(int, spec.frame.template_id.unique())),
        "split_seeds": list(map(int, config["split_seeds"])),
        "split_counts": config["template_split"],
    }
    return {
        "model_id": model_meta["id"],
        "model_revision": model_meta["revision"],
        "tokenizer_revision": model_meta["revision"],
        "task": f"structural_{group}",
        "group_task": f"structural_{group}",
        "layer": int(layer),
        "prompt_template_split": split_identity,
        "template_ids": sorted(map(int, spec.frame.template_id.unique())),
        "tokenizer_settings": config["tokenizer"],
        "prompt_construction_version": config["prompt_construction_version"],
        "activation_extraction_convention": config["rank_matched_internal_inputs"]["activation_extraction_convention"],
    }


def _internal_activation_requirements(config: dict, specs: dict[str, GroupSpec]) -> list[dict]:
    rows = []
    for model_meta in config["selected_models"]:
        for group in config["groups"]:
            spec = specs[group]
            for depth, layer in model_meta["depth_layers"].items():
                metadata = _internal_cache_metadata(config, spec, model_meta, group, int(layer))
                rows.append({
                    "model_key": model_meta["key"],
                    "model": model_meta["id"],
                    "group": group,
                    "depth_slot": round(float(depth), 2),
                    "layer": int(layer),
                    "cache_key": activation_cache_key(**metadata),
                    "metadata": metadata,
                })
    return rows


def verify_internal_activation_source(config: dict, specs: dict[str, GroupSpec], source: InternalActivationSource) -> pd.DataFrame:
    rows = []
    missing = []
    for requirement in _internal_activation_requirements(config, specs):
        key = requirement["cache_key"]
        digest = source.member_hash(key)
        available = digest is not None
        rows.append({
            key_name: value for key_name, value in requirement.items()
            if key_name != "metadata"
        } | {
            "available": bool(available),
            "sha256": digest,
        })
        if not available:
            missing.append((requirement["model_key"], requirement["group"], requirement["layer"], key))
    frame = pd.DataFrame(rows)
    if missing:
        raise FileNotFoundError(f"AFS-D missing {len(missing)} required internal activation caches: {missing[:3]}")
    return frame


def ensure_isolated_output(output_dir: Path, sources: FrozenMetricSources) -> None:
    output = Path(output_dir).resolve()
    for source in (sources.full_root, sources.afs_root):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("AFS-D output directory overlaps an immutable experiment root")


def create_output_tree(root, run_id: str) -> Path:
    output = Path(root) / run_id
    if output.exists() and not (output / "run_ledger.json").is_file():
        raise FileExistsError(f"existing AFS-D directory is not resumable: {output}")
    for relative in (
        "afs_d", "checkpoints/embedding_extraction", "checkpoints/embedding_analysis",
        "checkpoints/source_import", "logs", "provenance", "results/tables", "results/figures",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)
    return output


def _checkpoint(output_dir: Path, module: str, key: str, suffix: str) -> Path:
    digest = key.split(":", 1)[1]
    path = output_dir / "checkpoints" / module / f"{digest}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _embedding_metadata(model: dict, group: str, split_seed: int | None = None, latent_dim: int | None = None) -> dict:
    dim = int(latent_dim if latent_dim is not None else EXPECTED_GROUP_DIMS[str(group)])
    fields = {
        "model_key": model["key"],
        "model": model["id"],
        "model_family": "Llama" if model["id"].startswith("meta-llama/") else "Qwen",
        "group": group,
        "representation_location": "embedding",
        "depth_slot": 0.0,
        "layer": 0,
        "realized_depth": 0.0,
        "latent_dim": dim,
    }
    if split_seed is not None:
        fields["split_seed"] = int(split_seed)
    return fields


def _save_embedding(path: Path, hidden, metadata: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(
        tmp,
        hidden=np.asarray(hidden, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    tmp.replace(path)
    return path


def _load_embedding(path: Path, expected: dict) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "hidden" not in data or "metadata" not in data:
            raise ValueError("AFS-D embedding cache schema mismatch")
        metadata = json.loads(str(data["metadata"].item()))
        if metadata != expected:
            raise ValueError("AFS-D embedding cache metadata mismatch")
        hidden = np.asarray(data["hidden"], dtype=np.float32)
    return hidden


def _attach(rows, metadata: dict, source_run: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for key, value in metadata.items():
        frame[key] = value
    frame["source_run"] = source_run
    return frame


def _internal_metadata(model: dict, group: str, depth: float, layer: int, split_seed: int, latent_dim: int) -> dict:
    return {
        "model_key": model["key"],
        "model": model["id"],
        "model_family": "Llama" if model["id"].startswith("meta-llama/") else "Qwen",
        "group": group,
        "representation_location": "internal",
        "depth_slot": round(float(depth), 2),
        "layer": int(layer),
        "realized_depth": int(layer) / int(model["blocks"]),
        "latent_dim": int(latent_dim),
        "split_seed": int(split_seed),
    }


def _raw_matrix_rank(matrix: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=float)))


def _unique_row_count(matrix: np.ndarray) -> int:
    arr = np.ascontiguousarray(np.asarray(matrix))
    if arr.ndim != 2:
        raise ValueError("rank diagnostics require a 2D matrix")
    return int(np.unique(arr, axis=0).shape[0])


def embedding_rank_diagnostic_rows(spec: GroupSpec, hidden: np.ndarray, split, model: dict, config: dict) -> list[dict]:
    train_hidden = np.asarray(hidden[split.train_mask], dtype=np.float64)
    centered = train_hidden - train_hidden.mean(axis=0, keepdims=True)
    observed_rank = _raw_matrix_rank(centered)
    raw_rank = _raw_matrix_rank(train_hidden)
    unique_vectors = _unique_row_count(train_hidden)
    latent_dim = group_latent_dim(config, spec.name)
    expected_rank = int(config["expected_embedding_centered_ranks"][spec.name])
    row = {
        **_embedding_metadata(model, spec.name, split.seed, latent_dim),
        "rows": int(hidden.shape[0]),
        "train_rows": int(train_hidden.shape[0]),
        "unique_embedding_vectors": unique_vectors,
        "raw_matrix_rank": raw_rank,
        "centered_training_rank": observed_rank,
        "max_admissible_pca_dim": observed_rank,
        "frozen_group_latent_dim": int(latent_dim),
        "expected_centered_rank": expected_rank,
        "rank_status": "PASS" if observed_rank == expected_rank and observed_rank >= latent_dim else "FAIL",
        "duplicate_accounting": "exact_unique_rows",
    }
    return [row]


def assert_rank_diagnostics_pass(rows: list[dict]) -> None:
    bad = [
        row for row in rows
        if row["rank_status"] != "PASS"
        or int(row["centered_training_rank"]) < int(row["frozen_group_latent_dim"])
    ]
    if bad:
        first = bad[0]
        raise ValueError(
            "AFS-D embedding rank diagnostic failed before metric interpretation: "
            f"{first['model_key']} {first['group']} split={first['split_seed']} "
            f"rank={first['centered_training_rank']} required={first['frozen_group_latent_dim']} "
            f"expected={first['expected_centered_rank']}"
        )


def _run_dataframe_analysis(
    spec: GroupSpec,
    hidden: np.ndarray,
    split,
    model: dict,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latent_dim = group_latent_dim(config, spec.name)
    artifact = fit_base_artifact(spec, hidden, split, latent_dim, config)
    direct = evaluate_operators(
        spec,
        artifact,
        split,
        "independent_ridge",
        optimization_seed=None,
        random_baseline_seed=None,
    )
    metadata = _embedding_metadata(model, spec.name, split.seed, latent_dim)
    generator = generator_composition_rows(
        spec,
        artifact,
        split,
        alpha=float(config["ridge_alpha"]),
        random_seed=stable_cell_seed(config["random_generator_seed"], model["key"], spec.name, split.seed, "embedding"),
    )
    return _attach(direct, metadata, "afs_d_embedding"), _attach(generator, metadata, "afs_d_embedding")


def _run_internal_dataframe_analysis(
    spec: GroupSpec,
    hidden: np.ndarray,
    split,
    model: dict,
    depth: float,
    layer: int,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latent_dim = group_latent_dim(config, spec.name)
    artifact = fit_base_artifact(spec, hidden, split, latent_dim, config)
    direct = evaluate_operators(
        spec,
        artifact,
        split,
        "independent_ridge",
        optimization_seed=None,
        random_baseline_seed=None,
    )
    metadata = _internal_metadata(model, spec.name, depth, layer, split.seed, latent_dim)
    generator = generator_composition_rows(
        spec,
        artifact,
        split,
        alpha=float(config["ridge_alpha"]),
        random_seed=stable_cell_seed(config["random_generator_seed"], model["key"], spec.name, split.seed, f"internal_{depth}"),
    )
    return _attach(direct, metadata, "afs_d_internal_rank_matched"), _attach(generator, metadata, "afs_d_internal_rank_matched")


def _analysis_paths(output_dir: Path, fields: dict) -> tuple[Path, Path, str]:
    key = experiment_key("embedding_analysis", fields)
    return (
        _checkpoint(output_dir, "embedding_analysis", key, "operator.csv"),
        _checkpoint(output_dir, "embedding_analysis", key, "generator.csv"),
        key,
    )


def _rank_diagnostic_path(output_dir: Path, fields: dict) -> tuple[Path, str]:
    key = experiment_key("embedding_rank_diagnostic", fields)
    return _checkpoint(output_dir, "embedding_rank_diagnostic", key, "csv"), key


def _models_requiring_extraction(config: dict, output_dir: Path, ledger: RunLedger) -> set[str]:
    needed = set()
    for model in config["selected_models"]:
        for group in config["groups"]:
            fields = {**_embedding_metadata(model, group), "revision": model["revision"]}
            if not ledger.should_skip(experiment_key("embedding_extraction", fields)):
                needed.add(model["key"])
    return needed


def _run_embedding_cells(config, specs, output_dir, ledger, telemetry, manifest) -> set[str]:
    required = set()
    needs_model = _models_requiring_extraction(config, output_dir, ledger)
    for index, model_meta in enumerate(config["selected_models"]):
        tokenizer = model = None
        if model_meta["key"] in needs_model:
            fields = {"model": model_meta["id"], "revision": model_meta["revision"]}
            load_key = experiment_key("embedding_model_load", fields)
            marker = _checkpoint(output_dir, "embedding_model_load", load_key, "json")
            started = time.perf_counter()
            RuntimeTelemetry.reset_gpu_peak()
            tokenizer, model = load_tokenizer_and_model(
                model_meta["id"],
                dtype=config["dtype"],
                device_map=config["device_map"],
                revision=model_meta["revision"],
                tokenizer_settings=config["tokenizer"],
            )
            verify_embedding_architecture(model, model_meta["architecture"], model_meta["blocks"])
            resolved = str(getattr(model.config, "_commit_hash", None) or model_meta["revision"])
            if resolved != model_meta["revision"]:
                raise ValueError("resolved AFS-D model revision mismatch")
            if not ledger.should_skip(load_key):
                ledger.begin(load_key, fields)
                atomic_write_json(marker, {**fields, "resolved_revision": resolved})
                ledger.complete(load_key, fields, [marker])
            telemetry.record(load_key, "embedding_model_load", fields, started)
            required.add(load_key)
            manifest["models"][index]["resolved_revision"] = resolved
        for group in config["groups"]:
            spec = specs[group]
            extraction_fields = {**_embedding_metadata(model_meta, group), "revision": model_meta["revision"]}
            extraction_key = experiment_key("embedding_extraction", extraction_fields)
            cache_path = _checkpoint(output_dir, "embedding_extraction", extraction_key, "npz")
            if ledger.should_skip(extraction_key):
                hidden = _load_embedding(cache_path, extraction_fields)
            else:
                if tokenizer is None or model is None:
                    raise RuntimeError("AFS-D missing embedding model for incomplete extraction")
                started = time.perf_counter()
                RuntimeTelemetry.reset_gpu_peak()
                ledger.begin(extraction_key, extraction_fields)
                try:
                    hidden = extract_embedding_representations(
                        spec.frame, tokenizer, model, batch_size=int(config["batch_size"])
                    )
                    hidden = hidden.numpy() if hasattr(hidden, "numpy") else np.asarray(hidden)
                    _save_embedding(cache_path, hidden, extraction_fields)
                    ledger.complete(
                        extraction_key,
                        extraction_fields,
                        [cache_path],
                        {"rows": int(hidden.shape[0]), "hidden_dim": int(hidden.shape[1])},
                    )
                except Exception as error:
                    ledger.fail(extraction_key, extraction_fields, error)
                    raise
                finally:
                    telemetry.record(extraction_key, "embedding_extraction", extraction_fields, started)
            if hidden.shape[0] != len(spec.frame):
                raise ValueError("AFS-D embedding row contract mismatch")
            required.add(extraction_key)
            for seed in config["split_seeds"]:
                split = template_split(spec, config, seed)
                rank_fields = _embedding_metadata(model_meta, group, seed, group_latent_dim(config, group))
                rank_path, rank_key = _rank_diagnostic_path(output_dir, rank_fields)
                if not ledger.should_skip(rank_key):
                    started = time.perf_counter()
                    ledger.begin(rank_key, rank_fields)
                    try:
                        rank_rows = embedding_rank_diagnostic_rows(spec, hidden, split, model_meta, config)
                        assert_rank_diagnostics_pass(rank_rows)
                        atomic_write_frame(pd.DataFrame(rank_rows), rank_path)
                        ledger.complete(rank_key, rank_fields, [rank_path], {
                            "centered_training_rank": int(rank_rows[0]["centered_training_rank"]),
                            "frozen_group_latent_dim": int(rank_rows[0]["frozen_group_latent_dim"]),
                        })
                    except Exception as error:
                        ledger.fail(rank_key, rank_fields, error)
                        raise
                    finally:
                        telemetry.record(rank_key, "embedding_rank_diagnostic", rank_fields, started)
                required.add(rank_key)
                fields = _embedding_metadata(model_meta, group, seed)
                operator_path, generator_path, analysis_key = _analysis_paths(output_dir, fields)
                if not ledger.should_skip(analysis_key):
                    started = time.perf_counter()
                    ledger.begin(analysis_key, fields)
                    try:
                        operator, generator = _run_dataframe_analysis(spec, hidden, split, model_meta, config)
                        atomic_write_frame(operator, operator_path)
                        atomic_write_frame(generator, generator_path)
                        ledger.complete(
                            analysis_key,
                            fields,
                            [operator_path, generator_path],
                            {"operator_rows": len(operator), "generator_rows": len(generator)},
                        )
                    except Exception as error:
                        ledger.fail(analysis_key, fields, error)
                        raise
                    finally:
                        telemetry.record(analysis_key, "embedding_analysis", fields, started)
                required.add(analysis_key)
        del model, tokenizer
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass
    ledger.assert_complete(required)
    return required


def _collect_embedding(ledger: RunLedger) -> tuple[pd.DataFrame, pd.DataFrame]:
    operators, generators = [], []
    for key, unit in ledger.data["units"].items():
        if not key.startswith("embedding_analysis:") or unit.get("status") != "COMPLETE":
            continue
        for output in unit.get("outputs", []):
            path = Path(output["path"])
            if path.name.endswith(".operator.csv"):
                operators.append(pd.read_csv(path))
            elif path.name.endswith(".generator.csv"):
                generators.append(pd.read_csv(path))
    if len(operators) != 20 or len(generators) != 20:
        raise ValueError("AFS-D expected exactly 20 completed embedding analysis cells")
    return pd.concat(operators, ignore_index=True), pd.concat(generators, ignore_index=True)


def _collect_embedding_rank_diagnostics(ledger: RunLedger) -> pd.DataFrame:
    frames = []
    for key, unit in ledger.data["units"].items():
        if not key.startswith("embedding_rank_diagnostic:") or unit.get("status") != "COMPLETE":
            continue
        for output in unit.get("outputs", []):
            path = Path(output["path"])
            if path.name.endswith(".csv"):
                frames.append(pd.read_csv(path))
    if len(frames) != 20:
        raise ValueError("AFS-D expected exactly 20 completed embedding rank diagnostic cells")
    return pd.concat(frames, ignore_index=True)


def _internal_analysis_paths(output_dir: Path, fields: dict) -> tuple[Path, Path, str]:
    key = experiment_key("internal_rank_matched_analysis", fields)
    return (
        _checkpoint(output_dir, "internal_rank_matched_analysis", key, "operator.csv"),
        _checkpoint(output_dir, "internal_rank_matched_analysis", key, "generator.csv"),
        key,
    )


def _run_internal_cells(config, specs, source: InternalActivationSource, output_dir, ledger, telemetry) -> set[str]:
    required = set()
    availability_path = output_dir / "provenance" / "internal_activation_availability.csv"
    availability_key = experiment_key("internal_activation_source", {"source": source.provenance()})
    if not ledger.should_skip(availability_key):
        started = time.perf_counter()
        ledger.begin(availability_key, source.provenance())
        try:
            availability = verify_internal_activation_source(config, specs, source)
            atomic_write_frame(availability, availability_path)
            ledger.complete(availability_key, source.provenance(), [availability_path], {"required_caches": len(availability)})
        except Exception as error:
            ledger.fail(availability_key, source.provenance(), error)
            raise
        finally:
            telemetry.record(availability_key, "internal_activation_source", {}, started)
    required.add(availability_key)
    hidden_cache = {}
    for model_meta in config["selected_models"]:
        for group in config["groups"]:
            spec = specs[group]
            for depth, layer in model_meta["depth_layers"].items():
                metadata = _internal_cache_metadata(config, spec, model_meta, group, int(layer))
                cache_key = activation_cache_key(**metadata)
                hidden_cache[(model_meta["key"], group, int(layer))] = source.load(cache_key, metadata)
            for seed in config["split_seeds"]:
                split = template_split(spec, config, seed)
                for depth, layer in model_meta["depth_layers"].items():
                    fields = _internal_metadata(model_meta, group, depth, int(layer), seed, group_latent_dim(config, group))
                    operator_path, generator_path, analysis_key = _internal_analysis_paths(output_dir, fields)
                    if not ledger.should_skip(analysis_key):
                        started = time.perf_counter()
                        ledger.begin(analysis_key, fields)
                        try:
                            hidden = hidden_cache[(model_meta["key"], group, int(layer))]
                            operator, generator = _run_internal_dataframe_analysis(
                                spec, hidden, split, model_meta, depth, int(layer), config
                            )
                            atomic_write_frame(operator, operator_path)
                            atomic_write_frame(generator, generator_path)
                            ledger.complete(
                                analysis_key,
                                fields,
                                [operator_path, generator_path],
                                {"operator_rows": len(operator), "generator_rows": len(generator)},
                            )
                        except Exception as error:
                            ledger.fail(analysis_key, fields, error)
                            raise
                        finally:
                            telemetry.record(analysis_key, "internal_rank_matched_analysis", fields, started)
                    required.add(analysis_key)
    ledger.assert_complete(required)
    return required


def _collect_internal_rank_matched(ledger: RunLedger) -> tuple[pd.DataFrame, pd.DataFrame]:
    operators, generators = [], []
    for key, unit in ledger.data["units"].items():
        if not key.startswith("internal_rank_matched_analysis:") or unit.get("status") != "COMPLETE":
            continue
        for output in unit.get("outputs", []):
            path = Path(output["path"])
            if path.name.endswith(".operator.csv"):
                operators.append(pd.read_csv(path))
            elif path.name.endswith(".generator.csv"):
                generators.append(pd.read_csv(path))
    expected = 2 * 2 * 4 * 5
    if len(operators) != expected or len(generators) != expected:
        raise ValueError(f"AFS-D expected exactly {expected} completed internal rank-matched cells")
    return pd.concat(operators, ignore_index=True), pd.concat(generators, ignore_index=True)


def _build_manifest(config: dict, run_id: str, repo_root: Path, sources: FrozenMetricSources) -> dict:
    commit, dirty = git_state(repo_root)
    return {
        "experiment": config["experiment"],
        "run_id": run_id,
        "tier": "frozen",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "config_hash": config_hash(config),
        "scientific_design_changed": True,
        "design_repair": "rank_aware_repair_preregistered_in_docs/AFS_D_RANK_AWARE_REDESIGN.md",
        "final_scientific_run": True,
        "models": [
            {
                "key": model["key"], "id": model["id"], "requested_revision": model["revision"],
                "resolved_revision": None, "blocks": model["blocks"], "depth_layers": model["depth_layers"],
            }
            for model in config["selected_models"]
        ],
        "groups": config["groups"],
        "depth_slots": config["depth_slots"],
        "split_seeds": config["split_seeds"],
        "group_latent_dimensions": config["group_latent_dimensions"],
        "expected_embedding_centered_ranks": config["expected_embedding_centered_ranks"],
        "embedding_convention": config["embedding_convention"],
        "rank_matched_internal_inputs": config["rank_matched_internal_inputs"],
        "matrix": matrix(config),
        "immutable_sources": sources.source_provenance(),
        "seed_namespaces": {"template_split": config["split_seeds"], "bootstrap": config["bootstrap_seed"], "random_generator": config["random_generator_seed"]},
        "package_versions": package_versions(),
        "hardware": {"platform": platform.platform()},
        "run_ledger": "run_ledger.json",
        "status": "RUNNING",
        "artifacts": {},
    }


def _postprocess(config, output_dir, ledger, telemetry, sources, manifest) -> str:
    fields = {"bootstrap_samples": config["bootstrap_samples"], "bootstrap_seed": config["bootstrap_seed"]}
    key = experiment_key("afs_d_postprocessing", fields)
    report = output_dir / "AFS_D_SCIENTIFIC_REPORT.md"
    if ledger.should_skip(key):
        return key
    started = time.perf_counter()
    ledger.begin(key, fields)
    try:
        rank_diagnostics = _collect_embedding_rank_diagnostics(ledger)
        if (rank_diagnostics["rank_status"] != "PASS").any():
            raise ValueError("AFS-D rank diagnostics did not all pass")
        embedding_operator, embedding_generator = _collect_embedding(ledger)
        internal_operator, internal_generator = _collect_internal_rank_matched(ledger)
        operator = pd.concat([embedding_operator, internal_operator], ignore_index=True)
        generator = pd.concat([embedding_generator, internal_generator], ignore_index=True)
        location_metrics = prepare_location_metrics(operator, generator)
        contrasts = pair_depth_contrasts(location_metrics, config["depth_slots"])

        bootstrap_started = time.perf_counter()
        def progress(label, done, total):
            progress_fields = {"model": label[0], "group": label[1]}
            telemetry.record("afs_d_bootstrap_progress", "bootstrap_statistics", progress_fields, bootstrap_started, event="PROGRESS", progress_completed=done, progress_total=total)
        bootstrap = bootstrap_depth_contrasts(
            contrasts,
            n_boot=int(config["bootstrap_samples"]),
            seed=int(config["bootstrap_seed"]),
            ci=float(config["confidence_level"]),
            progress_callback=progress,
        )
        telemetry.record("afs_d_bootstrap", "bootstrap_statistics", {}, bootstrap_started)
        interpretation = classify_outcomes(bootstrap)
        manifest_for_report = {**manifest, "status": "SUCCESS"}
        outputs = write_stable_outputs(
            output_dir,
            embedding_operator=embedding_operator,
            embedding_generator=embedding_generator,
            embedding_rank_diagnostics=rank_diagnostics,
            generator_calibration=generator,
            location_metrics=location_metrics,
            contrasts=contrasts,
            bootstrap=bootstrap,
            interpretation=interpretation,
            manifest=manifest_for_report,
        )
        ledger.complete(key, fields, outputs, {
            "rank_diagnostic_rows": len(rank_diagnostics),
            "location_rows": len(location_metrics),
            "contrast_rows": len(contrasts),
            "bootstrap_rows": len(bootstrap),
        })
    except Exception as error:
        ledger.fail(key, fields, error)
        raise
    finally:
        telemetry.record(key, "afs_d_postprocessing", fields, started)
    sources.assert_unchanged()
    return key


def run(config, sources, internal_source, output_dir, ledger, telemetry, manifest) -> None:
    ensure_isolated_output(output_dir, sources)
    specs = group_specs()
    sources.assert_unchanged()
    required = _run_embedding_cells(config, specs, output_dir, ledger, telemetry, manifest)
    required |= _run_internal_cells(config, specs, internal_source, output_dir, ledger, telemetry)
    post_key = _postprocess(
        config, output_dir, ledger, telemetry, sources, manifest
    )
    required.add(post_key)
    ledger.assert_complete(required)
    sources.assert_unchanged()


def preflight(config: dict, full_root: Path, afs_root: Path, internal_activation_source: Path) -> dict:
    sources = FrozenMetricSources(full_root, afs_root, config)
    specs = group_specs()
    source = InternalActivationSource(internal_activation_source, config)
    availability = verify_internal_activation_source(config, specs, source)
    source.close()
    sources.assert_unchanged()
    return {
        "status": "VALID",
        "matrix": matrix(config),
        "group_latent_dimensions": dict(config["group_latent_dimensions"]),
        "expected_embedding_centered_ranks": dict(config["expected_embedding_centered_ranks"]),
        "required_internal_activation_caches": len(availability),
        "internal_activation_source": source.provenance(),
        "immutable_sources": sources.source_provenance(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Frozen AFS-D embedding-to-depth control")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", default="configs/afs_d.yaml")
    parser.add_argument("--full-results-root")
    parser.add_argument("--afs-abc-root")
    parser.add_argument("--internal-activation-source")
    parser.add_argument("--output-root", default="run_outputs/algebra_falsification")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    config = resolve_config(args.config)
    if args.validate_only:
        print(json.dumps({"status": "VALID", "matrix": matrix(config)}, sort_keys=True))
        return 0
    if not args.full_results_root or not args.afs_abc_root:
        raise RuntimeError("AFS-D requires --full-results-root and --afs-abc-root")
    if not args.internal_activation_source:
        default_source = config["rank_matched_internal_inputs"].get("archive")
        if default_source and Path(default_source).exists():
            args.internal_activation_source = default_source
        else:
            raise RuntimeError("rank-aware AFS-D requires --internal-activation-source")
    if args.preflight:
        print(json.dumps(
            preflight(config, Path(args.full_results_root), Path(args.afs_abc_root), Path(args.internal_activation_source)),
            indent=2,
            sort_keys=True,
        ))
        return 0

    repo_root = Path(__file__).resolve().parents[3]
    run_id = args.run_id or datetime.now(timezone.utc).strftime("afs_d_rank_aware_%Y%m%dT%H%M%SZ")
    prospective_output = Path(args.output_root) / run_id
    full_root = Path(args.full_results_root).resolve()
    afs_root = Path(args.afs_abc_root).resolve()
    for source_root in (full_root, afs_root):
        resolved_output = prospective_output.resolve()
        if resolved_output == source_root or resolved_output.is_relative_to(source_root) or source_root.is_relative_to(resolved_output):
            raise ValueError("AFS-D output directory overlaps an immutable experiment root")
    sources = FrozenMetricSources(full_root, afs_root, config)
    internal_source = InternalActivationSource(Path(args.internal_activation_source), config)
    output_dir = create_output_tree(args.output_root, run_id)
    ensure_isolated_output(output_dir, sources)
    ledger = RunLedger(output_dir / "run_ledger.json", run_id, config_hash(config))
    telemetry = RuntimeTelemetry(output_dir / "runtime_profile.csv")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else _build_manifest(config, run_id, repo_root, sources)
    if manifest.get("config_hash") != config_hash(config) or manifest.get("experiment") != config["experiment"]:
        raise ValueError("AFS-D resume manifest does not match frozen config")
    atomic_write_json(manifest_path, manifest)
    try:
        import yaml
        config_text = yaml.safe_dump(config, sort_keys=True)
    except ModuleNotFoundError:
        config_text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    atomic_write_text(output_dir / "config_snapshot.yaml", config_text)
    atomic_write_json(output_dir / "provenance" / "full_source.json", sources.source_provenance()["full"])
    atomic_write_json(output_dir / "provenance" / "afs_abc_source.json", sources.source_provenance()["afs_abc"])
    atomic_write_json(output_dir / "provenance" / "internal_activation_source.json", internal_source.provenance())
    try:
        run(config, sources, internal_source, output_dir, ledger, telemetry, manifest)
        manifest["status"] = "SUCCESS"
        checksums = artifact_checksums(output_dir)
        atomic_write_json(output_dir / "artifact_checksums.json", checksums)
        manifest["artifacts"] = artifact_checksums(output_dir)
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        internal_source.close()
        atomic_write_json(manifest_path, manifest)
    print(f"AFS-D results: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
