"""Verify the opt-in spectral source capsule without loading coordinates or models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> dict:
    study = root / "studies/spectral_finite_bank_v1"
    manifest_path = study / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("missing or invalid spectral source manifest")
    actual = {}
    for path in study.rglob("*"):
        rel = path.relative_to(study)
        if any(part in {"__pycache__", ".pytest_cache"} for part in rel.parts):
            continue
        if path.suffix in {".pyc", ".pyo"} or rel.as_posix() == "source_manifest.json":
            continue
        if path.is_symlink():
            raise ValueError("spectral source contains a symlink")
        if path.is_file():
            actual[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != manifest["files"]:
        raise ValueError("spectral source capsule changed or contains unrecorded files")
    original = actual.get("reference/gamma_r_original.py")
    if original != "12fff1cd64e801c5261a791c1700baf485512a6cbeac5704a845d636604057cb":
        raise ValueError("supplied gamma prototype was not preserved byte-for-byte")
    return {"source_files": len(actual), "original_prototype_preserved": True,
            "real_coordinate_analysis_executed_by_verifier": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root), sort_keys=True))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"spectral source verification failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
