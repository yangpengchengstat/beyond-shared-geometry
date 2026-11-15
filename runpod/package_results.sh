#!/usr/bin/env bash
set -euo pipefail

source_dir="${1:-}"
if [[ -z "${source_dir}" || ! -d "${source_dir}" ]]; then
  echo "usage: bash runpod/package_results.sh <main_experiment_run_directory>" >&2
  exit 2
fi

bundle="${RUNPOD_RESULT_BUNDLE_PATH:-/workspace/Beyond_Shared_Geometry_main_experiment_results.zip}"
export BSG_RESULT_SOURCE="${source_dir}"
export BSG_RESULT_BUNDLE="${bundle}"
python3 - <<'PY'
from __future__ import annotations
import csv, hashlib, json, os, re, time, zipfile
from datetime import datetime, timezone
from pathlib import Path

source = Path(os.environ["BSG_RESULT_SOURCE"]).resolve()
bundle = Path(os.environ["BSG_RESULT_BUNDLE"]).resolve()
sha = bundle.with_name(bundle.name + ".sha256")
packaging_started = time.perf_counter()
allowed = {".json", ".yaml", ".yml", ".csv", ".tex", ".svg", ".pdf", ".log", ".txt", ".md"}
forbidden_parts = {"activation_cache", ".git", "__pycache__", "checkpoints", "models"}
secret = re.compile(rb"(?:hf_|sk-|ghp_|github_pat_)[A-Za-z0-9_\-]{20,}")


def append_packaging_telemetry(event: str) -> bytes:
    path = source / "runtime_profile.csv"
    if not path.is_file():
        raise SystemExit("RESULT PACKAGE VALIDATION FAILED: runtime_profile.csv is missing")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    required = [
        "experiment_key", "experiment_module", "wall_clock_seconds", "event",
        "progress_completed", "progress_total", "timestamp_utc",
    ]
    fieldnames.extend(name for name in required if name not in fieldnames)
    elapsed = max(0.0, time.perf_counter() - packaging_started)
    row = {name: "" for name in fieldnames}
    row.update({
        "experiment_key": "result_packaging:main_experiment",
        "experiment_module": "result_packaging",
        "wall_clock_seconds": f"{elapsed:.9f}",
        "event": event,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    rows.append(row)
    tmp_profile = path.with_name(f".{path.name}.packaging.tmp")
    with tmp_profile.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_profile.replace(path)
    return path.read_bytes()


def refresh_runtime_profile_manifest(runtime_data: bytes) -> bytes:
    path = source / "manifest.json"
    if not path.is_file():
        raise SystemExit("RESULT PACKAGE VALIDATION FAILED: manifest.json is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["runtime_profile.csv"] = hashlib.sha256(runtime_data).hexdigest()
    tmp_manifest = path.with_name(f".{path.name}.packaging.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_manifest.replace(path)
    return path.read_bytes()


append_packaging_telemetry("STARTED")
files = []
for path in sorted(source.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(source)
    if any(part in forbidden_parts for part in rel.parts):
        continue
    if path.suffix.lower() not in allowed:
        raise SystemExit(f"RESULT PACKAGE VALIDATION FAILED: forbidden file {rel}")
    data = path.read_bytes()
    if secret.search(data):
        raise SystemExit(f"RESULT PACKAGE VALIDATION FAILED: credential-like text in {rel}")
    files.append((rel, data))
if not files:
    raise SystemExit("RESULT PACKAGE VALIDATION FAILED: no files")
bundle.parent.mkdir(parents=True, exist_ok=True)
tmp = bundle.with_suffix(".tmp.zip")
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel, data in files:
        if rel in {Path("runtime_profile.csv"), Path("manifest.json")}:
            continue
        zf.writestr(str(Path("Beyond_Shared_Geometry_main_experiment_results") / rel), data)
    runtime_data = append_packaging_telemetry("COMPLETE")
    manifest_data = refresh_runtime_profile_manifest(runtime_data)
    if secret.search(runtime_data):
        raise SystemExit("RESULT PACKAGE VALIDATION FAILED: credential-like text in runtime_profile.csv")
    if secret.search(manifest_data):
        raise SystemExit("RESULT PACKAGE VALIDATION FAILED: credential-like text in manifest.json")
    zf.writestr(
        str(Path("Beyond_Shared_Geometry_main_experiment_results") / "runtime_profile.csv"),
        runtime_data,
    )
    zf.writestr(
        str(Path("Beyond_Shared_Geometry_main_experiment_results") / "manifest.json"),
        manifest_data,
    )
with zipfile.ZipFile(tmp) as zf:
    names = zf.namelist()
    required = {
        "manifest.json", "environment.json", "config_snapshot.yaml", "run_ledger.json",
        "runtime_profile.csv", "FULL_COST_PROJECTION.md", "go_no_go.json",
        "structural/template_metrics.csv", "irrep/representation_diagnostics.csv",
        "intertwiner/cross_model_metrics.csv", "causal/intervention_metrics.csv",
        "bootstrap/confidence_intervals.csv",
        "results/tables/structural_main.csv", "results/tables/c7_causal_main.csv",
        "results/tables/d4_causal_main.csv", "results/tables/cross_model_main.csv",
        "results/tables/latent_dim_sensitivity.csv", "results/tables/model_scale_summary.csv",
        "results/tables/joint_vs_ridge_inference.csv",
        "results/tables/causal_additive_baseline_summary.csv",
        "results/tables/d4_order_control_summary.csv",
    }
    inner = {str(Path(*Path(name).parts[1:])) for name in names}
    missing = required - inner
    if missing:
        raise SystemExit(f"RESULT PACKAGE VALIDATION FAILED: missing {sorted(missing)}")
tmp.replace(bundle)
digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
sha.write_text(f"{digest}  {bundle.name}\n", encoding="utf-8")
print(f"RESULT ZIP: {bundle}")
print(f"SHA256: {digest}")
print(f"FILES: {len(files)}")
PY
