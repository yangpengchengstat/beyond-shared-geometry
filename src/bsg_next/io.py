"""Content identities and atomic, resumable experiment records."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        return jsonable(value.item())
    if isinstance(value, float) and not __import__("math").isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def identity(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(jsonable(value), handle, indent=2, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_run(output: str | Path, config: dict, stage: str,
                *, provenance: dict | None = None) -> dict:
    """Reject identity changes and nonempty foreign outputs; allow exact resume."""
    output = Path(output)
    if output.is_symlink():
        raise ValueError("a run output cannot be a symbolic link")
    descriptor = {"schema_version": 1, "stage": stage, "config": config,
                  "provenance": provenance or {}}
    digest = identity(descriptor)
    marker = output / "run_identity.json"
    if marker.exists():
        existing = read_json(marker)
        if existing.get("identity") != digest:
            raise ValueError("output belongs to a different configuration/input identity; choose a new run directory")
        return existing
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing a nonempty output directory without matching run identity")
    output.mkdir(parents=True, exist_ok=True)
    record = {**descriptor, "identity": digest}
    atomic_json(marker, record)
    return record


def source_identity(root: str | Path) -> str:
    root = Path(root)
    entries = {str(p.relative_to(root)): file_sha256(p)
               for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts}
    return identity(entries)
