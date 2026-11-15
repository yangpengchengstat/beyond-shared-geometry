#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
package_name="Beyond_Shared_Geometry_algebra_falsification_runpod.zip"
package_dir="${project_root}/runpod_packages"
zip_path="${package_dir}/${package_name}"
mkdir -p "${package_dir}"

cd "${project_root}"
test_python="${BSG_TEST_PYTHON:-python3}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bsg_pycache PYTHONPATH=src "${test_python}" tests/run_cpu_checks.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bsg_pycache PYTHONPATH=src "${test_python}" tests/run_runtime_checks.py

PROJECT_ROOT="${project_root}" ZIP_PATH="${zip_path}" PACKAGE_NAME="${package_name}" "${test_python}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["PROJECT_ROOT"]).resolve()
zip_path = Path(os.environ["ZIP_PATH"]).resolve()
package_name = os.environ["PACKAGE_NAME"]
status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
if status:
    raise SystemExit("AFS PACKAGE FAILED: Git tree must be clean")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
build_info = {
    "source_commit": commit,
    "git_dirty": False,
    "build_timestamp_utc": timestamp,
    "package_name": package_name,
}
include = ["src", "configs", "runpod", "tests", "scripts", "docs", "pyproject.toml", "requirements.txt", "requirements-gpu.txt", "README.md", "LICENSE"]
forbidden_parts = {".git", "__pycache__", ".pytest_cache", "runpod_packages", "run_outputs", "paper", "archive"}
forbidden_suffixes = {".zip", ".pt", ".pth", ".safetensors", ".bin", ".pdf"}
files = []
for relative in include:
    path = root / relative
    if not path.exists():
        raise SystemExit(f"AFS PACKAGE FAILED: missing {relative}")
    candidates = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in candidates:
        rel = item.relative_to(root)
        if any(part in forbidden_parts for part in rel.parts) or rel.suffix.lower() in forbidden_suffixes or rel.name == "test_release_contract.py":
            continue
        files.append(item)
tmp = zip_path.with_name("." + zip_path.name + ".tmp")
with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        rel = Path("beyond-shared-geometry") / path.relative_to(root)
        archive.write(path, rel)
    archive.writestr("BUILD_INFO.json", json.dumps(build_info, indent=2, sort_keys=True) + "\n")
    archive.writestr("beyond-shared-geometry/DEPLOYMENT_PROVENANCE.json", json.dumps({
        "source_git_commit": commit, "source_git_dirty": False,
        "package_filename": package_name, "package_stage": "ALGEBRA FALSIFICATION SUITE",
        "build_timestamp_utc": timestamp,
    }, indent=2, sort_keys=True) + "\n")
required = {
    "BUILD_INFO.json",
    "beyond-shared-geometry/configs/algebra_falsification.yaml",
    "beyond-shared-geometry/src/bsg/experiments/algebra_falsification.py",
    "beyond-shared-geometry/src/bsg/experiments/algebra_falsification_units.py",
    "beyond-shared-geometry/tests/test_algebra_falsification.py",
    "beyond-shared-geometry/runpod/run_algebra_falsification.sh",
    "beyond-shared-geometry/runpod/preflight_algebra_falsification.sh",
    "beyond-shared-geometry/docs/ALGEBRA_FALSIFICATION_PROTOCOL.md",
}
with zipfile.ZipFile(tmp) as archive:
    names = set(archive.namelist())
    missing = required - names
    if missing:
        raise SystemExit(f"AFS PACKAGE FAILED: missing archive members {sorted(missing)}")
    if any("/.git/" in name or name.endswith(".zip") for name in names):
        raise SystemExit("AFS PACKAGE FAILED: forbidden archive content")
    if json.loads(archive.read("BUILD_INFO.json")) != build_info:
        raise SystemExit("AFS PACKAGE FAILED: BUILD_INFO mismatch")
with tempfile.TemporaryDirectory(prefix="afs_package_verify_") as temporary:
    extract = Path(temporary)
    with zipfile.ZipFile(tmp) as archive:
        archive.extractall(extract)
    project = extract / "beyond-shared-geometry"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = "/tmp/bsg_pycache"
    env["PATH"] = str(Path(os.sys.executable).parent) + os.pathsep + env.get("PATH", "")
    checks = [
        [os.sys.executable, "-c", "import bsg; from bsg.experiments.algebra_falsification import main; print('AFS IMPORT: PASS')"],
        ["bash", "runpod/preflight_algebra_falsification.sh", "--cpu-contract"],
    ]
    for command in checks:
        result = subprocess.run(command, cwd=project, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, end="")
        if result.returncode:
            raise SystemExit(f"AFS PACKAGE FAILED: extracted check failed: {command}")
tmp.replace(zip_path)
digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
(zip_path.parent / (zip_path.name + ".sha256")).write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
manifest = zip_path.parent / "Beyond_Shared_Geometry_algebra_falsification_runpod.manifest.txt"
manifest.write_text(
    f"PACKAGE: {zip_path}\nBYTES: {zip_path.stat().st_size}\nSHA256: {digest}\nSOURCE COMMIT: {commit}\nBUILD UTC: {timestamp}\nGIT DIRTY: false\n",
    encoding="utf-8",
)
print(f"AFS PACKAGE: {zip_path}")
print(f"AFS PACKAGE BYTES: {zip_path.stat().st_size}")
print(f"AFS PACKAGE SHA256: {digest}")
PY
