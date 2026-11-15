"""Validated prospective configurations; never resolve moving model revisions silently."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

from .io import identity


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be an object")
    value = deepcopy(value)
    for override in overrides or []:
        key, separator, raw = override.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError("overrides use dotted.key=JSON_value")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        target = value
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError("override crosses a non-object configuration value")
        target[parts[-1]] = parsed
    validate_config(value)
    return value


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported prospective configuration schema")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", config.get("study_id", "")):
        raise ValueError("study_id must be a portable identifier")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    models = config.get("models", [])
    if not models:
        raise ValueError("at least one pinned model specification is required")
    names = []
    for model in models:
        name = model.get("id", model.get("model_id"))
        if not isinstance(name, str) or not name:
            raise ValueError("model specification lacks id")
        names.append(name)
        revision = model.get("revision", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"model {name} requires a resolved 40-character commit revision")
        if model.get("tokenizer_revision", revision) != revision:
            raise ValueError("a different tokenizer revision requires a separately registered protocol")
        layers = model.get("layers", model.get("hidden_state_indices", []))
        if not layers or any(not isinstance(i, int) or i < 0 for i in layers):
            raise ValueError(f"model {name} has invalid hidden-state indices")
    if len(names) != len(set(names)):
        raise ValueError("duplicate model identifiers")
    technical = bool(config.get("technical_test", False))
    data = config.get("data", {})
    test = data.get("test_families", 32)
    if not technical and test not in (32, 48, 64):
        raise ValueError("production test-family size must be power-selected from 32/48/64")
    if data.get("max_test_families", 64) < test:
        raise ValueError("maximum sealed test pool is smaller than the selected test")
    calibrated_families=int(config.get("synthetic",{}).get("test_families",32))
    sizes=config.get("power",{}).get("test_families",[32,48,64])
    if not technical and any(int(n)<calibrated_families for n in sizes):
        raise ValueError("Structural power candidates cannot be smaller than the calibrated test-family count")
    if config.get("runtime", {}).get("trust_remote_code", False):
        raise ValueError("remote model code is not part of this pinned runtime")
    for name in ("main_alpha", "s3_alpha"):
        alpha = config.get("statistics", {}).get(name, .04 if name == "main_alpha" else .01)
        if not 0 < alpha < 1:
            raise ValueError("invalid multiplicity allocation")
    stats = config.get("statistics", {})
    if stats.get("main_alpha", .04) + stats.get("s3_alpha", .01) > .05 + 1e-12:
        raise ValueError("structural familywise allocations exceed .05")
    for model in config.get("native", {}).get("models", []):
        if model not in names:
            raise ValueError("native model must be a pinned study model")


def model_key(model: dict | str) -> str:
    name = model if isinstance(model, str) else model.get("id", model.get("model_id"))
    return re.sub(r"[^A-Za-z0-9_.-]", "__", name)


def config_identity(config: dict) -> str:
    return identity(config)


def smoke_config(config: dict) -> dict:
    """Technical smoke limits never qualify outputs for manuscript promotion."""
    result = deepcopy(config)
    result["technical_test"] = True
    result["study_id"] += "-technical"
    result.setdefault("synthetic", {}).update({"replicates": 2, "replicates_per_cell": 2,
        "negative_replicates": 2, "crossed_replicates": 2,"timing_replicates":1,
        "bootstrap_datasets_per_cell":1,"crossed_bootstrap_datasets":1,"bootstrap_replicates":99,
        "limit": 4})
    result.setdefault("power", {})["replicates"] = 50
    native=result.setdefault("native", {})
    native["technical_run"]=True
    native.setdefault("power",{}).update({"replicates":50,"require_measured":False})
    result.setdefault("peers",{})["technical_run"]=True
    result.setdefault("crossmodel",{})["technical_run"]=True
    result.setdefault("execution",{})["require_historical_archive"]=False
    result.setdefault("structural", {}).update({"secondary": ["affine", "procrustes", "generator"],
                                                 "bootstrap_sensitivity": False})
    return result
