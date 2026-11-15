#!/usr/bin/env python3
"""Verify frozen execution sources and explicitly bound publication wrappers.

No source is executed. The original prospective manifest and strict verifier stay
unchanged. Only four named engineering files may bind preserved historical bytes;
scientific code/configuration/test sources must still match at their active paths.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ENGINEERING_PATHS = frozenset({
    ".github/workflows/ci.yml", "pyproject.toml",
    "scripts/check_release.py", "scripts/package_release.py",
})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(root, value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid source path: " + repr(value))
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Nonrelative source path: " + value)
    target = root / path
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source path escapes repository: " + value)
    if any(parent.is_symlink() for parent in target.parents if parent != root and root in parent.parents):
        raise ValueError("Source path traverses symlink: " + value)
    return target


def prospective_paths(root):
    # Preserve the selection contract in the unchanged verify_prospective.py.
    selected = []
    for directory in ("src/bsg_next", "configs/prospective", "tests/prospective"):
        selected.extend(p for p in (root / directory).rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts
                        and p.suffix in (".py", ".json", ".yaml", ".yml"))
    selected.extend(p for p in (root / "runpod").glob("next_*") if p.is_file())
    selected.extend(p for p in (root / "studies/maintenance_v2").rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts
                    and p.suffix in (".py", ".md", ".txt", ".json", ".sh"))
    selected.extend(root / name for name in (
        "scripts/verify_prospective.py", "scripts/package_runpod_next.py",
        "scripts/check_release.py", "scripts/package_release.py",
        "tests/run_prospective_checks.py", "pyproject.toml", ".github/workflows/ci.yml",
    ) if (root / name).is_file())
    return {str(p.relative_to(root)) for p in selected}


def verify(root):
    root = Path(root).resolve()
    manifest_path = safe(root, "release/prospective_source_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["files"]
    binding = json.loads(safe(root, "release/engineering_source_bindings.json").read_text())
    if binding.get("schema_version") != 1 or binding.get("frozen_manifest") != "release/prospective_source_manifest.json":
        raise ValueError("Invalid engineering source binding schema/manifest")
    if binding.get("frozen_manifest_sha256") != digest(manifest_path):
        raise ValueError("Engineering binding refers to a different frozen manifest")
    entries = binding.get("engineering_bindings", {})
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Engineering bindings must be explicit and nonempty")
    if set(entries) - ENGINEERING_PATHS:
        raise ValueError("Historical fallback is forbidden for scientific/non-engineering sources: " + ", ".join(sorted(set(entries) - ENGINEERING_PATHS)))
    if set(entries) - set(expected):
        raise ValueError("Engineering binding is not a frozen source member")
    current_paths = prospective_paths(root)
    if current_paths != set(expected):
        raise ValueError("Prospective inventory differs: " + ", ".join(sorted(current_paths ^ set(expected))))
    resolutions = []
    active_matches = 0
    for name, expected_hash in sorted(expected.items()):
        current = safe(root, name)
        if not current.is_file():
            raise ValueError("Missing active source: " + name)
        current_hash = digest(current)
        entry = entries.get(name)
        if entry is not None:
            archived_name = entry.get("execution_source", "")
            archived = safe(root, archived_name)
            if Path(archived_name).parts[:2] != ("release", "history"):
                raise ValueError("Execution source must be preserved under release/history: " + name)
            if entry.get("execution_sha256") != expected_hash:
                raise ValueError("Execution binding differs from frozen manifest: " + name)
            if not archived.is_file() or digest(archived) != expected_hash:
                raise ValueError("Missing or changed preserved execution source: " + name)
            if current_hash != entry.get("publication_sha256"):
                raise ValueError("Unbound publication wrapper change: " + name)
            if not entry.get("reason"):
                raise ValueError("Engineering derivative lacks reason: " + name)
            resolutions.append({"path": name, "execution_source": archived_name,
                                "execution_sha256": expected_hash,
                                "publication_sha256": current_hash})
        elif current_hash != expected_hash:
            raise ValueError("Frozen scientific/unbound source changed: " + name)
        else:
            active_matches += 1
    return {
        "status": "PASS", "prospective_files": len(expected),
        "active_frozen_matches": active_matches,
        "engineering_derivatives": resolutions,
        "version": manifest.get("version"),
        "scope": "Frozen execution-source identity plus explicitly bound publication derivatives; no source execution",
        "frozen_manifest_changed": False,
        "model_execution_during_verification": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root), indent=2))
    except (ValueError, KeyError, OSError) as error:
        sys.exit(str(error))
