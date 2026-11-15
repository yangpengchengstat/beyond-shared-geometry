#!/usr/bin/env bash
set -euo pipefail

run_dir="${1:-}"
if [[ -z "${run_dir}" || ! -f "${run_dir}/manifest.json" ]]; then
  echo "usage: bash runpod/package_afs_results.sh <successful_afs_run_directory>" >&2
  exit 2
fi

bundle="${AFS_RESULT_BUNDLE_PATH:-/workspace/Beyond_Shared_Geometry_algebra_falsification_results.zip}"
RUN_DIR="${run_dir}" BUNDLE="${bundle}" python3 - <<'PY'
import hashlib, json, os, zipfile
from pathlib import Path

root = Path(os.environ["RUN_DIR"]).resolve()
bundle = Path(os.environ["BUNDLE"]).resolve()
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("status") != "SUCCESS":
    raise SystemExit("refusing to package an incomplete AFS run")
tmp = bundle.with_name("." + bundle.name + ".tmp")
with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "checkpoints" in path.parts:
            continue
        archive.write(path, Path("Beyond_Shared_Geometry_algebra_falsification_results") / path.relative_to(root))
tmp.replace(bundle)
digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
bundle.with_suffix(bundle.suffix + ".sha256").write_text(f"{digest}  {bundle.name}\n", encoding="utf-8")
print(bundle)
print(digest)
PY
