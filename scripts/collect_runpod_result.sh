#!/usr/bin/env bash
set -euo pipefail

stage="${1:-}"
source_path="${2:-}"
requested_run_id="${3:-}"
max_mb="${RUNPOD_RESULT_MAX_MB:-50}"

if [[ -z "${stage}" || -z "${source_path}" ]]; then
  echo "usage: bash scripts/collect_runpod_result.sh <stage> <source_dir_or_zip> [run_id]" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
if [[ -n "${RUNPOD_RESULT_PROJECT_ROOT_OVERRIDE:-}" ]]; then
  project_root="${RUNPOD_RESULT_PROJECT_ROOT_OVERRIDE}"
fi

export RUNPOD_RESULT_STAGE="${stage}"
export RUNPOD_RESULT_SOURCE="${source_path}"
export RUNPOD_RESULT_RUN_ID="${requested_run_id}"
export RUNPOD_RESULT_MAX_MB="${max_mb}"
export PROJECT_ROOT="${project_root}"

python3 - <<'PY'
from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
stage = os.environ["RUNPOD_RESULT_STAGE"]
source = Path(os.environ["RUNPOD_RESULT_SOURCE"]).expanduser()
requested_run_id = os.environ.get("RUNPOD_RESULT_RUN_ID", "")
max_bytes = int(float(os.environ.get("RUNPOD_RESULT_MAX_MB", "50")) * 1024 * 1024)
result_root = project_root / "run_outputs" / "imported"

stages = {"main_experiment"}
approved_names = {
    "RUN_METADATA.json",
    "console.log",
    "config_snapshot.yaml",
    "DEPLOYMENT_PROVENANCE.json",
    "summary.json",
    "summary.md",
    "raw_metrics.csv",
    "results.csv",
    "paired_effects.csv",
    "seed_summary.csv",
    "layer_summary.csv",
    "latent_dim_summary.csv",
    "group_word_results.csv",
    "control_results.csv",
    "behavior_summary.csv",
    "intertwiner_summary.csv",
    "raw_intertwiner_metrics.csv",
    "depth_summary.csv",
    "dimension_summary.csv",
    "control_summary.csv",
    "automorphism_summary.csv",
    "heldout_power_summary.csv",
    "floor_diagnostic_summary.csv",
}
approved_suffixes = {".json", ".yaml", ".yml", ".md", ".csv", ".txt", ".log", ".tex", ".svg"}
dangerous_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".gguf", ".npy", ".npz", ".parquet", ".feather", ".pkl", ".pickle"}
dangerous_names = {".env"}
dangerous_dirs = {".git", "__pycache__", ".pytest_cache", ".cache", "hf_cache", "huggingface", "transformers", "models", "checkpoints", "activation_cache"}
credential_patterns = [
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"huggingface_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY"),
]


def fail(message: str) -> None:
    raise SystemExit(f"RUNPOD RESULT IMPORT FAILED: {message}")


def validate_stage(value: str) -> None:
    if value not in stages:
        fail(f"unsupported stage {value!r}; expected one of {sorted(stages)}")


def allocate_run_id(stage_dir: Path, value: str) -> str:
    if value:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            fail("run_id contains unsafe characters")
        if (stage_dir / value).exists():
            fail(f"run_id already exists: {value}")
        return value
    max_seen = 0
    for child in stage_dir.iterdir() if stage_dir.exists() else []:
        match = re.fullmatch(rf"{re.escape(stage)}_([0-9]{{3}})", child.name)
        if match:
            max_seen = max(max_seen, int(match.group(1)))
    return f"{stage}_{max_seen + 1:03d}"


def is_dangerous(rel: Path) -> str | None:
    parts = rel.parts
    if any(part in dangerous_dirs for part in parts):
        return "dangerous directory"
    if rel.name in dangerous_names or rel.name.startswith(".env."):
        return "environment file"
    if rel.suffix.lower() in dangerous_suffixes and not (stage == "intertwiner" and rel.name.startswith("intertwiner_artifact_") and rel.suffix.lower() == ".npz"):
        return "dangerous binary/tensor suffix"
    return None


def is_approved(rel: Path) -> bool:
    if rel.name in approved_names:
        return True
    if stage == "intertwiner" and rel.name.startswith("intertwiner_artifact_") and rel.suffix.lower() == ".npz":
        return True
    if rel.suffix.lower() in approved_suffixes:
        return True
    return False


def is_textish(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    if not data:
        return True
    sample = data[:4096]
    return sum(byte < 9 or 13 < byte < 32 for byte in sample) / len(sample) < 0.05


def scan_secret(data: bytes) -> bool:
    return any(pattern.search(data) for pattern in credential_patterns)


def prepare_source(path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None, str]:
    if not path.exists():
        fail(f"source does not exist: {path}")
    if path.is_dir():
        return path.resolve(), None, str(path)
    if path.is_file() and path.suffix == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="runpod_result_import_")
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                parts = Path(name).parts
                if any(part == ".." for part in parts) or Path(name).is_absolute():
                    fail("zip contains unsafe path")
            zf.extractall(tmp.name)
        return Path(tmp.name), tmp, str(path)
    fail("source must be a directory or .zip file")


validate_stage(stage)
stage_dir = result_root / stage
stage_dir.mkdir(parents=True, exist_ok=True)
run_id = allocate_run_id(stage_dir, requested_run_id)
dest = stage_dir / run_id

src_root, tmp_obj, source_label = prepare_source(source)
try:
    candidates = sorted(path for path in src_root.rglob("*") if path.is_file())
    if not candidates:
        fail("source contains no files")
    approved: list[tuple[Path, Path]] = []
    for path in candidates:
        rel = path.relative_to(src_root)
        if any(part.startswith("__MACOSX") for part in rel.parts):
            continue
        reason = is_dangerous(rel)
        if reason:
            fail(f"{reason}: {rel}")
        size = path.stat().st_size
        if size > max_bytes:
            fail(f"file exceeds {max_bytes} bytes: {rel}")
        data = path.read_bytes()
        if is_textish(data) and scan_secret(data):
            fail(f"credential-like literal detected in {rel}")
        if is_approved(rel):
            parts = rel.parts
            if parts and parts[0] == "Beyond_Shared_Geometry_main_experiment_results":
                rel = Path(*parts[1:])
            approved.append((path, rel))
    if not approved:
        fail("no approved lightweight result files found")
    if dest.exists():
        fail(f"destination already exists: {dest}")
    tmp_dest = dest.with_name(f".{run_id}.importing")
    if tmp_dest.exists():
        shutil.rmtree(tmp_dest)
    tmp_dest.mkdir(parents=True)
    imported = []
    for src, rel in approved:
        target = tmp_dest / rel
        if target.exists():
            fail(f"duplicate imported filename: {rel}")
        shutil.copy2(src, target)
        data = target.read_bytes()
        imported.append((rel.as_posix(), target.stat().st_size, hashlib.sha256(data).hexdigest()))
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metadata_commit = "unknown"
    metadata_path = tmp_dest / "RUN_METADATA.json"
    if metadata_path.exists():
        text = metadata_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r'"git_commit"\s*:\s*"([^"]+)"', text)
        if match:
            metadata_commit = match.group(1)
    manifest_lines = [
        f"RUN ID: {run_id}",
        f"STAGE: {stage}",
        f"IMPORT TIMESTAMP UTC: {timestamp}",
        f"SOURCE: {source_label}",
        f"GIT COMMIT IF KNOWN: {metadata_commit}",
        f"VALIDATION STATUS: PASS",
        f"MAX FILE SIZE BYTES: {max_bytes}",
        "FILES:",
    ]
    for name, size, digest in imported:
        manifest_lines.append(f"- {name}\t{size}\t{digest}")
    manifest_lines.append("")
    (tmp_dest / "IMPORT_MANIFEST.txt").write_text("\n".join(manifest_lines), encoding="utf-8")
    tmp_dest.replace(dest)
finally:
    if tmp_obj is not None:
        tmp_obj.cleanup()

print(f"IMPORTED: {dest}")
print(f"FILES: {len(approved)}")
PY
