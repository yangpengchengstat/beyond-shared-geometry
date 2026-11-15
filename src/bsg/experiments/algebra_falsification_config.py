"""Frozen configuration resolver for the Algebra Falsification Suite."""

from __future__ import annotations

import copy
import hashlib
import json

from bsg.config import load_config
from bsg.experiments.main_config import CANONICAL_DEPTH_LAYERS, CANONICAL_MODELS, CANONICAL_REVISIONS


TIERS = ("smoke", "pre_full", "full")
EXPECTED_SPLITS = {"train_templates": 28, "validation_templates": 6, "test_templates": 6}
EXPECTED_SCRAMBLE_SEEDS = [70001, 70003, 70009, 70019, 70039]


def resolve_afs_config(path, tier: str) -> dict:
    if tier not in TIERS:
        raise ValueError(f"unknown AFS tier: {tier}")
    config = copy.deepcopy(load_config(path))
    tier_config = copy.deepcopy(config["tiers"][tier])
    config["tier"] = tier
    for key, value in tier_config.items():
        if key != "model_keys":
            config[key] = value
    selected = []
    for key in tier_config["model_keys"]:
        model = copy.deepcopy(config["models"][key])
        model["key"] = key
        model["depth_layers"] = {round(float(k), 2): int(v) for k, v in model["depth_layers"].items()}
        model["selected_layers"] = [model["depth_layers"][round(float(d), 2)] for d in config["depth_slots"]]
        selected.append(model)
    config["selected_models"] = selected
    validate_afs_config(config)
    return config


def validate_afs_config(config: dict) -> None:
    if config.get("experiment") != "beyond_shared_geometry_algebra_falsification":
        raise ValueError("AFS experiment namespace changed")
    if config.get("groups") != ["C7", "D4"]:
        raise ValueError("AFS groups must be C7 and D4")
    if config.get("template_split") != EXPECTED_SPLITS:
        raise ValueError("AFS template split must remain 28/6/6")
    if int(config.get("primary_latent_dimension", -1)) != 16:
        raise ValueError("AFS primary latent dimension must remain 16")
    configured = {m["id"]: int(m["blocks"]) for m in config["models"].values()}
    if configured != CANONICAL_MODELS:
        raise ValueError("AFS canonical model panel changed")
    for model in config["models"].values():
        model_id = model["id"]
        if model["revision"] != CANONICAL_REVISIONS[model_id]:
            raise ValueError(f"AFS immutable revision changed for {model_id}")
        depths = {round(float(k), 2): int(v) for k, v in model["depth_layers"].items()}
        if depths != CANONICAL_DEPTH_LAYERS[model_id]:
            raise ValueError(f"AFS depth map changed for {model_id}")
        expected_arch = "llama" if model_id.startswith("meta-llama/") else "qwen2"
        if model.get("architecture") != expected_arch:
            raise ValueError(f"AFS embedding architecture contract changed for {model_id}")
    tier = config["tier"]
    expected = {
        "smoke": (1, 1, 1, 100, 1, 200),
        "pre_full": (2, 2, 2, 1000, 5, 2000),
        "full": (6, 4, 5, 5000, 5, 5000),
    }[tier]
    observed = (
        len(config["selected_models"]), len(config["depth_slots"]), len(config["split_seeds"]),
        int(config["permutation_samples"]), len(config["scramble_seeds"]), int(config["bootstrap_samples"]),
    )
    if observed != expected:
        raise ValueError(f"AFS {tier} matrix changed: {observed} != {expected}")
    if tier in {"pre_full", "full"} and list(map(int, config["scramble_seeds"])) != EXPECTED_SCRAMBLE_SEEDS:
        raise ValueError("AFS scramble seed namespace changed")
    namespaces = [
        set(map(int, config["split_seeds"])), set(map(int, config["scramble_seeds"])),
        {int(config["algebra_permutation_seed"])}, {int(config["bootstrap_seed"])},
        {int(config["random_generator_seed"])},
    ]
    if any(namespaces[i] & namespaces[j] for i in range(len(namespaces)) for j in range(i + 1, len(namespaces))):
        raise ValueError("AFS RNG namespaces overlap")
    source = config.get("historical_source", {})
    required_source = {
        "run_id": "main_full_003", "experiment": "beyond_shared_geometry_main",
        "tier": "full", "status": "SUCCESS",
    }
    if any(source.get(k) != v for k, v in required_source.items()):
        raise ValueError("AFS historical source identity changed")


def afs_config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def afs_matrix(config: dict) -> dict:
    model_count = len(config["selected_models"])
    group_count = len(config["groups"])
    depth_count = len(config["depth_slots"])
    split_count = len(config["split_seeds"])
    primary_cells = model_count * group_count * depth_count * split_count
    embedding_cells = model_count * group_count * split_count
    return {
        "primary_cells": primary_cells,
        "embedding_cells": embedding_cells,
        "scientific_units": 1 + 3 * primary_cells + model_count + model_count * group_count + embedding_cells,
        "permutation_draws": primary_cells * int(config["permutation_samples"]),
        "scramble_fits": primary_cells * len(config["scramble_seeds"]) * 2,
        "bootstrap_replicates_per_contrast": int(config["bootstrap_samples"]),
    }


def validate_primary_coverage(frame, config: dict, *, embedding: bool = False) -> None:
    if embedding:
        expected = {
            (m["id"], group, int(seed))
            for m in config["selected_models"] for group in config["groups"] for seed in config["split_seeds"]
        }
        columns = ("model", "group", "split_seed")
    else:
        expected = {
            (m["id"], group, round(float(depth), 2), int(seed))
            for m in config["selected_models"] for group in config["groups"]
            for depth in config["depth_slots"] for seed in config["split_seeds"]
        }
        columns = ("model", "group", "depth_slot", "split_seed")
    observed = set()
    for values in frame.loc[:, columns].drop_duplicates().itertuples(index=False, name=None):
        normalized = tuple(round(float(value), 2) if column == "depth_slot" else int(value) if column == "split_seed" else value for column, value in zip(columns, values))
        observed.add(normalized)
    if observed != expected:
        raise ValueError(f"AFS primary coverage mismatch: missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
