#!/usr/bin/env bash
set -euo pipefail

run_dir="${1:-}"
if [[ -z "${run_dir}" || ! -f "${run_dir}/manifest.json" ]]; then
  echo "usage: bash runpod/package_afs_d_results.sh <successful_afs_d_run_directory>" >&2
  exit 2
fi

bundle="${AFS_D_RESULT_BUNDLE_PATH:-/workspace/Beyond_Shared_Geometry_afs_d_results.zip}"
RUN_DIR="${run_dir}" BUNDLE="${bundle}" PYTHONPATH=src python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

from bsg.experiments.main_outputs import artifact_checksums
from bsg.experiments.runtime_state import RuntimeTelemetry, atomic_write_json

root = Path(os.environ["RUN_DIR"]).resolve()
bundle = Path(os.environ["BUNDLE"]).resolve()
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("experiment") != "beyond_shared_geometry_afs_d_rank_aware" or manifest.get("status") != "SUCCESS":
    raise SystemExit("refusing to package an incomplete or non-AFS-D run")
required = [
    "interpretation.json", "AFS_D_SCIENTIFIC_REPORT.md",
    "afs_d/embedding_rank_diagnostics.csv",
    "afs_d/bootstrap_depth_contrasts.csv", "results/tables/afs_d_depth_contrasts.csv",
]
missing = [relative for relative in required if not (root / relative).is_file()]
if missing:
    raise SystemExit(f"AFS-D result package missing required outputs: {missing}")

started = time.perf_counter()
RuntimeTelemetry(root / "runtime_profile.csv").record(
    "afs_d_result_packaging", "result_packaging", {}, started
)
manifest["artifacts"] = artifact_checksums(root)
atomic_write_json(root / "artifact_checksums.json", manifest["artifacts"])
manifest["artifacts"] = artifact_checksums(root)
atomic_write_json(manifest_path, manifest)

bundle.parent.mkdir(parents=True, exist_ok=True)
tmp = bundle.with_name("." + bundle.name + ".tmp")
with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "checkpoints" in path.parts:
            continue
        archive.write(path, Path("Beyond_Shared_Geometry_afs_d_results") / path.relative_to(root))
tmp.replace(bundle)
digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
bundle.with_suffix(bundle.suffix + ".sha256").write_text(f"{digest}  {bundle.name}\n", encoding="utf-8")
print(bundle)
print(digest)
PY
