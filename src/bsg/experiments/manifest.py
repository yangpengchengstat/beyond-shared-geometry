"""Run provenance for the canonical experiment."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from bsg.experiments.main_config import config_hash


REQUIRED_FIELDS = {
    "experiment", "run_id", "tier", "timestamp_utc", "git_commit", "git_dirty",
    "config_hash", "models", "layer_convention", "activation_extraction_convention",
    "tokenizer", "primary_latent_dimension", "structural_latent_dimensions",
    "causal_latent_dimensions", "seed_namespaces", "package_versions", "hardware",
    "run_ledger", "status", "artifacts",
}


def _git_output(root: Path, args: list[str], default: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def git_state(root: Path) -> tuple[str, bool]:
    commit = _git_output(root, ["rev-parse", "HEAD"], "UNKNOWN")
    status = _git_output(root, ["status", "--porcelain"], "UNKNOWN")
    if commit == "UNKNOWN":
        provenance_path = root / "DEPLOYMENT_PROVENANCE.json"
        if provenance_path.is_file():
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                return str(provenance.get("source_git_commit", "UNKNOWN")), bool(provenance.get("source_git_dirty", True))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    return commit, bool(status)


def validate_git_policy(tier: str, dirty: bool, allow_dirty_full: bool = False) -> None:
    if tier == "full" and dirty and not allow_dirty_full:
        raise RuntimeError("FULL requires a clean Git working tree; pass --allow-dirty-full only with explicit authorization")


def package_versions() -> dict:
    names = ["numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers", "accelerate", "pyyaml"]
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def hardware_environment() -> dict:
    result = {"platform": platform.platform(), "cuda_version": None, "gpu_model": None}
    try:
        import torch
        result["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            result["gpu_model"] = torch.cuda.get_device_name(0)
            result["gpu_count"] = torch.cuda.device_count()
            result["gpu_total_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    except ModuleNotFoundError:
        pass
    return result


def build_manifest(config: dict, run_id: str, repo_root: Path, *, ledger_path: str = "run_ledger.json") -> dict:
    commit, dirty = git_state(repo_root)
    return {
        "experiment": "beyond_shared_geometry_main",
        "run_id": run_id,
        "tier": config["tier"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "config_hash": config_hash(config),
        "models": [
            {
                "key": model["key"],
                "id": model["id"],
                "requested_revision": model["revision"],
                "resolved_revision": None,
                "blocks": model["blocks"],
                "layers": model["layers"],
                "causal_layers": model["causal_layers"],
                "depth_slots": model["depth_slots"],
                "tokenizer_identity": None,
            }
            for model in config["selected_models"]
        ],
        "layer_convention": config["layer_convention"],
        "activation_extraction_convention": config["activation_extraction_convention"],
        "tokenizer": config["tokenizer"],
        "primary_latent_dimension": config["primary_latent_dimension"],
        "structural_latent_dimensions": config["structural_latent_dimensions"],
        "causal_latent_dimensions": config["causal_latent_dimensions"],
        "seed_namespaces": {
            "template_split": config["split_seeds"],
            "operator_optimization": config["optimization_seeds"],
            "learned_fit": config["learned_fit_seeds"],
            "random_baseline": config["random_baseline_seeds"],
            "bootstrap": config["bootstrap_seed"],
        },
        "learned_fit_semantics": config["learned_fit_semantics"],
        "template_split": config["template_split"],
        "prompt_construction_version": config["prompt_construction_version"],
        "package_versions": package_versions(),
        "hardware": hardware_environment(),
        "run_ledger": ledger_path,
        "status": "RUNNING",
        "artifacts": {},
    }


def validate_manifest(manifest: dict) -> None:
    missing = REQUIRED_FIELDS - set(manifest)
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    if not manifest["models"]:
        raise ValueError("manifest requires at least one model")
    if any(not model.get("requested_revision") for model in manifest["models"]):
        raise ValueError("manifest model revisions must be immutable and nonempty")


def write_manifest(path, manifest: dict) -> None:
    validate_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
