"""Canonical final-experiment configuration resolution and validation."""

from __future__ import annotations

import copy
import hashlib
import json

from bsg.config import load_config
from bsg.experiments.fitting import centered_training_rank


CANONICAL_MODELS = {
    "meta-llama/Llama-3.2-1B": 16,
    "meta-llama/Llama-3.2-3B": 28,
    "meta-llama/Llama-3.1-8B": 32,
    "Qwen/Qwen2.5-1.5B": 28,
    "Qwen/Qwen2.5-3B": 36,
    "Qwen/Qwen2.5-7B": 28,
}

CANONICAL_REVISIONS = {
    "meta-llama/Llama-3.2-1B": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    "meta-llama/Llama-3.2-3B": "13afe5124825b4f3751f836b40dafda64c1ed062",
    "meta-llama/Llama-3.1-8B": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "Qwen/Qwen2.5-1.5B": "8faed761d45a263340a0528343f099c05c9a4323",
    "Qwen/Qwen2.5-3B": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
    "Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796",
}

CANONICAL_DEPTH_LAYERS = {
    "meta-llama/Llama-3.2-1B": {0.20: 3, 0.40: 6, 0.60: 10, 0.80: 13},
    "meta-llama/Llama-3.2-3B": {0.20: 6, 0.40: 11, 0.60: 17, 0.80: 22},
    "meta-llama/Llama-3.1-8B": {0.20: 6, 0.40: 13, 0.60: 19, 0.80: 26},
    "Qwen/Qwen2.5-1.5B": {0.20: 6, 0.40: 11, 0.60: 17, 0.80: 22},
    "Qwen/Qwen2.5-3B": {0.20: 7, 0.40: 14, 0.60: 22, 0.80: 29},
    "Qwen/Qwen2.5-7B": {0.20: 6, 0.40: 11, 0.60: 17, 0.80: 22},
}

CANONICAL_PAIR_KEYS = (
    ("llama_1b", "llama_3b"),
    ("llama_3b", "llama_8b"),
    ("llama_1b", "llama_8b"),
    ("qwen_1_5b", "qwen_3b"),
    ("qwen_3b", "qwen_7b"),
    ("qwen_1_5b", "qwen_7b"),
    ("llama_1b", "qwen_1_5b"),
    ("llama_3b", "qwen_3b"),
    ("llama_8b", "qwen_7b"),
)


def _normalize_depth_map(values: dict) -> dict[float, int]:
    return {round(float(slot), 2): int(layer) for slot, layer in values.items()}


def resolve_main_config(path, tier: str) -> dict:
    raw = load_config(path)
    if tier not in {"smoke", "core", "pre_full", "full"}:
        raise ValueError(f"unknown tier: {tier}")
    config = copy.deepcopy(raw)
    config["tier"] = tier
    tier_cfg = copy.deepcopy(config["tiers"][tier])
    if "template_split" in tier_cfg:
        config["template_split"] = tier_cfg["template_split"]
    for key, value in tier_cfg.items():
        if key not in {"model_keys", "template_split", "causal_baselines", "causal_d4_wrong_order_controls"}:
            config[key] = value
    if "causal_baselines" in tier_cfg:
        config["causal"]["baselines"] = list(tier_cfg["causal_baselines"])
    if "causal_d4_wrong_order_controls" in tier_cfg:
        config["causal"]["d4_wrong_order_controls"] = list(tier_cfg["causal_d4_wrong_order_controls"])

    slots = [round(float(slot), 2) for slot in config["depth_slots"]]
    selected = []
    for key in tier_cfg["model_keys"]:
        model = copy.deepcopy(config["models"][key])
        model["key"] = key
        depth_map = _normalize_depth_map(model["depth_layers"])
        model["depth_layers"] = depth_map
        model["depth_slots"] = slots
        model["primary_layers"] = [depth_map[slot] for slot in slots]
        model["causal_layers"] = list(model["primary_layers"])
        model["layers"] = list(model["primary_layers"])
        if config.get("include_terminal_structural"):
            model["layers"].append(int(model["terminal_structural_layer"]))
        model["depth_slot_by_layer"] = {str(depth_map[slot]): slot for slot in slots}
        if config.get("include_terminal_structural"):
            model["depth_slot_by_layer"][str(model["terminal_structural_layer"])] = "terminal"
        selected.append(model)
    config["selected_models"] = selected
    config["selected_cross_model_pairs"] = [
        list(pair) for pair in config["cross_model_pairs"]
        if pair[0] in tier_cfg["model_keys"] and pair[1] in tier_cfg["model_keys"]
    ]
    config["random_seed_by_optimization_seed"] = {
        str(opt): int(random_seed)
        for opt, random_seed in zip(config["optimization_seeds"], config["random_baseline_seeds"])
    }
    semantics = config.get("learned_fit_semantics", "historical_seed_replicated")
    config["learned_fit_seeds"] = (
        list(map(int, config["optimization_seeds"]))
        if semantics == "historical_seed_replicated"
        else [int(config["deterministic_fit_seed"])]
    )
    validate_main_config(config)
    return config


def _validate_rank_policy(config: dict) -> None:
    train_templates = int(config["template_split"]["train_templates"])
    max_dim = max(map(int, config["structural_latent_dimensions"]))
    for group, order in (("C7", 7), ("D4", 4)):
        rank = centered_training_rank(train_templates * order, 10**9)
        if max_dim > rank:
            raise ValueError(
                f"{config['tier']} {group} latent_dim={max_dim} exceeds centered training rank {rank}"
            )


def validate_main_config(config: dict) -> None:
    configured = {entry["id"]: int(entry["blocks"]) for entry in config["models"].values()}
    if configured != CANONICAL_MODELS:
        raise ValueError("canonical model matrix or block counts changed")
    if config.get("groups") != ["C7", "D4"]:
        raise ValueError("canonical groups must be [C7, D4]")
    if config.get("estimators") != ["independent_ridge", "group_consistent_joint"]:
        raise ValueError("canonical estimators changed")
    if tuple(tuple(pair) for pair in config["cross_model_pairs"]) != CANONICAL_PAIR_KEYS:
        raise ValueError("canonical cross-model pair matrix changed")
    semantics = config.get("learned_fit_semantics")
    if semantics not in {"historical_seed_replicated", "deterministic_single"}:
        raise ValueError("unknown learned-fit semantics")
    if semantics == "historical_seed_replicated" and len(config["optimization_seeds"]) != len(config["random_baseline_seeds"]):
        raise ValueError("historical optimization and paired random seed counts differ")
    if semantics == "deterministic_single" and len(config["learned_fit_seeds"]) != 1:
        raise ValueError("deterministic learned fitting requires exactly one technical fit seed")
    seed_sets = [
        set(map(int, config["split_seeds"])),
        set(map(int, config["optimization_seeds"])),
        set(map(int, config["random_baseline_seeds"])),
        {int(config["bootstrap_seed"])},
    ]
    if any(seed_sets[i] & seed_sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("scientific seed namespaces must be disjoint")
    counts = config["template_split"]
    expected_templates = int(config.get("template_limit") or 40)
    if sum(int(counts[name]) for name in ("train_templates", "validation_templates", "test_templates")) != expected_templates:
        raise ValueError("template split counts do not match selected template count")
    if int(config["primary_latent_dimension"]) != 16:
        raise ValueError("primary latent dimension must remain 16")
    if list(map(int, config["causal_latent_dimensions"])) != [16]:
        raise ValueError("causal analysis is frozen to latent dimension 16")
    if config.get("tier") in {"pre_full", "full"}:
        if list(map(int, config.get("primary_structural_latent_dimensions", []))) != [16]:
            raise ValueError("revised primary structural dimension must be [16]")
        sensitivity = set(map(int, config.get("sensitivity_latent_dimensions", [])))
        if config["tier"] == "full" and sensitivity != {8, 32, 64}:
            raise ValueError("revised FULL sensitivity dimensions must be 8, 32, and 64")
        if list(map(int, config.get("cross_model_latent_dimensions", []))) != [16]:
            raise ValueError("revised cross-model analysis is primary dimension 16 only")
        if config["causal"]["baselines"] != ["identity", "random_orthogonal", "source_conditioned_additive"]:
            raise ValueError("revised causal baselines changed")
    _validate_rank_policy(config)
    for key, raw_model in config["models"].items():
        model_id = raw_model["id"]
        if raw_model["revision"] != CANONICAL_REVISIONS[model_id]:
            raise ValueError(f"mutable or unexpected model revision for {model_id}")
        if _normalize_depth_map(raw_model["depth_layers"]) != CANONICAL_DEPTH_LAYERS[model_id]:
            raise ValueError(f"canonical depth mapping changed for {model_id}")
        if int(raw_model["terminal_structural_layer"]) != int(raw_model["blocks"]):
            raise ValueError(f"terminal diagnostic must equal block count for {model_id}")
    for model in config["selected_models"]:
        blocks = int(model["blocks"])
        if not all(1 <= int(layer) <= blocks for layer in model["layers"]):
            raise ValueError(f"invalid structural layer for {model['id']}")
        if not all(1 <= int(layer) < blocks for layer in model["causal_layers"]):
            raise ValueError(f"causal layers must exclude terminal block for {model['id']}")


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
