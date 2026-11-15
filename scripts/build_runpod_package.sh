#!/usr/bin/env bash
set -euo pipefail

stage="${1:-main_experiment}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
package_dir="${project_root}/runpod_packages"
project_name="beyond-shared-geometry"

if [[ "${stage}" != "main_experiment" ]]; then
  echo "Only the canonical main_experiment package is supported" >&2
  exit 2
fi
package_base="Beyond_Shared_Geometry_main_experiment_runpod"
intended_stage="CANONICAL MAIN PAPER EXPERIMENT"

zip_path="${package_dir}/${package_base}.zip"
sha_path="${zip_path}.sha256"
manifest_path="${package_dir}/${package_base}.manifest.txt"

mkdir -p "${package_dir}"

export PROJECT_ROOT="${project_root}"
export PACKAGE_DIR="${package_dir}"
export PROJECT_NAME="${project_name}"
export ZIP_PATH="${zip_path}"
export SHA_PATH="${sha_path}"
export MANIFEST_PATH="${manifest_path}"
export PACKAGE_FILENAME="${package_base}.zip"
export INTENDED_STAGE="${intended_stage}"

python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
package_dir = Path(os.environ["PACKAGE_DIR"]).resolve()
project_name = os.environ["PROJECT_NAME"]
zip_path = Path(os.environ["ZIP_PATH"]).resolve()
sha_path = Path(os.environ["SHA_PATH"]).resolve()
manifest_path = Path(os.environ["MANIFEST_PATH"]).resolve()
package_filename = os.environ["PACKAGE_FILENAME"]
intended_stage = os.environ["INTENDED_STAGE"]

include_paths = [
    "src",
    "configs",
    "runpod",
    "tests",
    "scripts",
    "pyproject.toml",
    "requirements.txt",
    "requirements-gpu.txt",
    "README.md",
    "LICENSE",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/MATHEMATICAL_DEFINITIONS.md",
    "docs/MODEL_CONVENTIONS.md",
    "docs/PAPER_EXPERIMENT_MAPPING.md",
    "docs/PRE_FULL_ADDITIVE_AUDIT.md",
    "docs/PRE_FULL_PROTOCOL.md",
    "docs/PROVENANCE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/SCIENTIFIC_SCOPE.md",
    "docs/ARTIFACTS.md",
    "docs/FINAL_EXPERIMENT_DESIGN.md",
    "docs/FULL_FREEZE_RECORD.md",
    "docs/FULL_CHECKPOINT_HYDRATION_REPAIR.md",
]

root_forbidden_dir_names = {
    "paper",
    "notebooks",
    "archive",
    "runpod_packages",
}
forbidden_dir_names = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".cache",
    "hf_cache",
    "huggingface",
}
# Publication checks require paper/evidence files intentionally omitted here.
# Keep the inherited numerical CPU/runtime checks in the slim GPU deployment.
forbidden_file_names = {".env", "test_release_contract.py"}
forbidden_suffixes = {
    ".pdf",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".gguf",
    ".zip",
    ".tar",
    ".gz",
}
credential_patterns = [
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"huggingface_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY"),
]


def fail(message: str) -> None:
    for artifact in (zip_path, sha_path, manifest_path):
        try:
            artifact.unlink()
        except FileNotFoundError:
            pass
    raise SystemExit(f"PACKAGE VALIDATION FAILED: {message}")


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True).strip()
    except Exception:
        return "unavailable"


def run_full_cpu_tests() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/bsg_pycache")
    test_python = os.environ.get("BSG_TEST_PYTHON", sys.executable)
    result = subprocess.run(
        [test_python, str(project_root / "tests" / "run_cpu_checks.py")],
        cwd=project_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        fail("full local CPU checks failed; refusing to build deployment package")
    match = re.search(r"(\d+) CPU checks passed", result.stdout)
    if not match:
        fail("full local CPU check count was not reported")
    return {
        "full_cpu_tests_status": "PASS",
        "full_cpu_test_count": int(match.group(1)),
        "full_cpu_test_timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def is_forbidden_path(rel: Path) -> str | None:
    parts = rel.parts
    if parts and parts[0] in root_forbidden_dir_names:
        return "forbidden root directory"
    if any(part in forbidden_dir_names for part in parts[:-1]):
        return "forbidden directory"
    if rel.name in forbidden_file_names or rel.name.startswith(".env."):
        return "forbidden environment file"
    if rel.suffix.lower() in forbidden_suffixes:
        return "forbidden generated/binary suffix"
    return None


def should_scan_as_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    sample = data[:4096]
    return bool(sample) and sum(byte < 9 or 13 < byte < 32 for byte in sample) / len(sample) < 0.05


def scan_credentials(data: bytes) -> bool:
    return any(pattern.search(data) for pattern in credential_patterns)


def validate_extracted_package(zip_candidate: Path, expected_build_info: dict) -> None:
    test_python = os.environ.get("BSG_TEST_PYTHON", sys.executable)
    with tempfile.TemporaryDirectory(prefix="runpod_package_install_") as tmp_name:
        extract_root = Path(tmp_name)
        with zipfile.ZipFile(zip_candidate) as zf:
            zf.extractall(extract_root)
        extracted_project = extract_root / project_name
        if not extracted_project.exists():
            fail("extracted project directory is missing")
        build_info_path = extract_root / "BUILD_INFO.json"
        if not build_info_path.is_file():
            fail("root BUILD_INFO.json is missing")
        if json.loads(build_info_path.read_text(encoding="utf-8")) != expected_build_info:
            fail("root BUILD_INFO.json does not match the source build identity")
        performance_test = extracted_project / "tests" / "test_bootstrap_performance.py"
        if not performance_test.is_file():
            fail("bootstrap performance regression test is missing")
        hydration_test = extracted_project / "tests" / "test_checkpoint_hydration.py"
        if not hydration_test.is_file():
            fail("checkpoint hydration regression test is missing")
        hierarchical_source = (
            extracted_project / "src" / "bsg" / "statistics" / "hierarchical.py"
        ).read_text(encoding="utf-8")
        mean_bootstrap = hierarchical_source.split("def hierarchical_mean_ci", 1)[1].split(
            "def _spearman", 1
        )[0]
        if "_prepare_clustered_index" not in mean_bootstrap or "_clustered_draw_indexed" not in mean_bootstrap:
            fail("optimized indexed hierarchical mean bootstrap is missing")
        if "data[data.split_seed" in mean_bootstrap or "opt_frame[opt_frame.template_id" in mean_bootstrap:
            fail("hierarchical mean bootstrap still filters pandas frames inside replicate loops")
        experiment_source = (
            extracted_project / "src" / "bsg" / "experiments" / "main_experiment.py"
        ).read_text(encoding="utf-8")
        if "_can_resume_postprocessing" not in experiment_source or "post-processing-only mode" not in experiment_source:
            fail("postprocessing-only resume implementation is missing")
        if "_hydrate_structural_checkpoints" not in experiment_source or "_can_resume_cross_model" not in experiment_source:
            fail("checkpoint-only cross-model recovery implementation is missing")
        install = subprocess.run(
            [test_python, "-m", "pip", "install", "--no-deps", "-e", str(extracted_project)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if install.returncode != 0:
            print(install.stdout, end="")
            fail("editable install from extracted package failed")
        imports = subprocess.run(
            [
                test_python,
                "-c",
                (
                    "import bsg; "
                    "print(bsg.__version__ if hasattr(bsg, '__version__') else 'bsg import PASS'); "
                    "from bsg.paper.labels import paper_metric_label; "
                    "print('PASS', paper_metric_label('E_hom_data'))"
                ),
            ],
            cwd=extracted_project,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(imports.stdout, end="")
        if imports.returncode != 0:
            fail("runtime package import check from extracted package failed")
        preflight_env = os.environ.copy()
        preflight_env["PATH"] = f"{Path(test_python).parent}{os.pathsep}{preflight_env.get('PATH', '')}"
        preflight = subprocess.run(
            ["bash", "runpod/preflight.sh", "--cpu-contract"],
            cwd=extracted_project,
            env=preflight_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(preflight.stdout, end="")
        if preflight.returncode != 0 or "RUNPOD CPU CONTRACT: PASS" not in preflight.stdout:
            fail("RunPod CPU preflight contract failed from extracted package")


files: list[Path] = []
for include in include_paths:
    root = project_root / include
    if not root.exists():
        fail(f"required include path is missing: {include}")
    if root.is_file():
        candidates = [root]
    else:
        candidates = sorted(path for path in root.rglob("*") if path.is_file())
    for path in candidates:
        rel = path.relative_to(project_root)
        reason = is_forbidden_path(rel)
        if reason:
            continue
        files.append(path)

if not files:
    fail("no files selected")

with tempfile.NamedTemporaryFile(prefix="runpod_package_", suffix=".zip", dir=package_dir, delete=False) as tmp:
    tmp_zip_path = Path(tmp.name)

try:
    full_cpu_status = run_full_cpu_tests()
    source_status = git_output(["status", "--short"])
    source_commit = git_output(["rev-parse", "HEAD"])
    if source_status:
        fail("Git working tree is dirty; refusing to build canonical deployment package")
    if source_commit == "unavailable":
        fail("source Git commit is unavailable")
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    build_info = {
        "source_commit": source_commit,
        "git_dirty": False,
        "build_timestamp_utc": timestamp,
        "package_name": package_filename,
    }
    provenance = {
        "source_git_commit": source_commit,
        "source_git_dirty": bool(source_status),
        "package_filename": package_filename,
        "package_sha256": None,
        "package_stage": intended_stage,
        "build_timestamp_utc": timestamp,
        **full_cpu_status,
    }
    with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            rel = path.relative_to(project_root)
            data = path.read_bytes()
            if should_scan_as_text(data) and scan_credentials(data):
                fail(f"credential-like literal detected in {rel}")
            arc = Path(project_name) / rel
            info = zipfile.ZipInfo(str(arc))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (path.stat().st_mode & 0o777) << 16
            zf.writestr(info, data)
        zf.writestr(
            f"{project_name}/DEPLOYMENT_PROVENANCE.json",
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        )
        zf.writestr("BUILD_INFO.json", json.dumps(build_info, indent=2, sort_keys=True) + "\n")

    with zipfile.ZipFile(tmp_zip_path) as zf:
        names = zf.namelist()
        if not names:
            fail("zip is empty")
        for name in names:
            rel = Path(name)
            if name == "BUILD_INFO.json":
                continue
            if rel.parts[0] != project_name:
                fail(f"unexpected top-level path: {name}")
            inner = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path("")
            reason = is_forbidden_path(inner)
            if reason:
                fail(f"{reason}: {name}")
            data = zf.read(name)
            if should_scan_as_text(data) and scan_credentials(data):
                fail(f"credential-like literal detected in packaged file {name}")

        required = [
            "src",
            "configs",
            "runpod",
            "tests",
            "scripts",
            "pyproject.toml",
            "requirements.txt",
            "LICENSE",
            "runpod/bootstrap.sh",
            "runpod/preflight.sh",
            "runpod/run_main_experiment.sh",
            "configs/main_experiment.yaml",
            "src/bsg/experiments/main_experiment.py",
            "docs/EXPERIMENT_PROTOCOL.md",
            "docs/PRE_FULL_PROTOCOL.md",
            "docs/PRE_FULL_ADDITIVE_AUDIT.md",
            "docs/FULL_FREEZE_RECORD.md",
            "docs/FULL_CHECKPOINT_HYDRATION_REPAIR.md",
            "runpod/full_authorization/manifest.json",
            "runpod/full_authorization/go_no_go.json",
            "runpod/full_authorization/FULL_COST_PROJECTION.md",
            "tests/run_runtime_checks.py",
            "tests/test_bootstrap_performance.py",
            "src/bsg/paper/__init__.py",
            "src/bsg/paper/labels.py",
            "DEPLOYMENT_PROVENANCE.json",
        ]
        name_set = set(names)
        for required_path in required:
            prefix = f"{project_name}/{required_path}"
            if required_path in {"src", "configs", "runpod", "tests", "scripts"}:
                if not any(name.startswith(prefix + "/") for name in names):
                    fail(f"missing required directory after extraction: /workspace/{prefix}")
            elif prefix not in name_set:
                fail(f"missing required file after extraction: /workspace/{prefix}")
        if "BUILD_INFO.json" not in name_set:
            fail("missing required file after extraction: /workspace/BUILD_INFO.json")

    validate_extracted_package(tmp_zip_path, build_info)

    zip_path.unlink(missing_ok=True)
    tmp_zip_path.replace(zip_path)
    zip_path.chmod(0o644)

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    sha_path.write_text(f"{digest}  {package_filename}\n", encoding="utf-8")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    top_level = sorted({Path(name).parts[1] for name in names if len(Path(name).parts) > 1})
    status = source_status
    if not status:
        status = "clean"
    commit = source_commit
    excluded = [
        ".git/",
        "paper/",
        "run_outputs/ and activation caches",
        "archive/",
        "runpod_packages/",
        ".env and .env.*",
        "HF tokens and likely literal credentials",
        "model weights and Hugging Face caches",
        "__pycache__/ and test/tool caches",
        "PDFs and previous deployment archives",
    ]
    manifest = [
        f"PACKAGE FILENAME: {package_filename}",
        f"CREATION TIMESTAMP UTC: {timestamp}",
        f"GIT COMMIT HASH: {commit}",
        "GIT WORKING-TREE STATUS:",
        status,
        f"PACKAGE SHA-256: {digest}",
        f"PACKAGE SIZE BYTES: {zip_path.stat().st_size}",
        f"NUMBER OF FILES: {len(names)}",
        f"FULL CPU TESTS: {full_cpu_status['full_cpu_tests_status']}",
        f"FULL CPU TEST COUNT: {full_cpu_status['full_cpu_test_count']}",
        f"FULL CPU TEST TIMESTAMP UTC: {full_cpu_status['full_cpu_test_timestamp_utc']}",
        "INCLUDED TOP-LEVEL PATHS:",
        *[f"- {item}" for item in top_level],
        "EXCLUDED CATEGORIES:",
        *[f"- {item}" for item in excluded],
        "INTENDED STAGE:",
        intended_stage,
        "",
    ]
    manifest_path.write_text("\n".join(manifest), encoding="utf-8")
except BaseException:
    try:
        tmp_zip_path.unlink()
    except FileNotFoundError:
        pass
    raise

print(f"PACKAGE: {zip_path}")
print(f"SHA256: {digest}")
print(f"MANIFEST: {manifest_path}")
print(f"FILES: {len(names)}")
PY
