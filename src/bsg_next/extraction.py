"""Bounded float32 activation caches with durable resume and strict completion.

No model is loaded on a verified cache hit. Partial files are never returned by
read_cache. Every batch is flushed before a checksummed progress record commits.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from contextlib import contextmanager

import numpy as np

from .data import dataset_identity, validate_records
from .io import atomic_json, file_sha256, identity, read_json

CACHE_SCHEMA_VERSION = 1


@contextmanager
def _writer_lock(directory):
    import fcntl
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Cache directory cannot be a symbolic link")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is writing this cache") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _code_identity():
    directory = Path(__file__).parent
    return identity({name: file_sha256(directory / name) for name in ("extraction.py", "hf.py", "data.py", "groups.py")})


def _root(path):
    path = Path(path)
    return path.parent if path.name == "manifest.json" else path


def _safe(root, name):
    value = Path(name)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Cache file references must be relative and remain in their cache")
    result = root / value
    if result.is_symlink() or root.resolve() not in result.resolve().parents:
        raise ValueError("Unsafe cache file reference")
    return result


def _rows_bytes(records):
    return ("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records)).encode()


def _chunk_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def load_cache(path, verify_hashes=True):
    directory = _root(path)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Activation cache is incomplete: no completion manifest")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ValueError("Activation cache is not a complete supported cache")
    if "descriptor" in manifest:
        descriptor = manifest["descriptor"]
        if identity(descriptor) != manifest.get("cache_key") or any(manifest.get(k) != descriptor.get(k) for k in ("dataset_sha256", "model", "layers", "readout", "extraction_code_sha256")):
            raise ValueError("Cache completion descriptor identity mismatch")
    rows_path = _safe(directory, manifest["rows_path"])
    if verify_hashes and file_sha256(rows_path) != manifest["rows_sha256"]:
        raise ValueError("Cached row manifest checksum mismatch")
    records = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    if dataset_identity(records) != manifest["dataset_sha256"]:
        raise ValueError("Cached dataset identity mismatch")
    if len(records) != manifest["n_rows"] or len({r["row_id"] for r in records}) != len(records):
        raise ValueError("Cache row cardinality mismatch")
    entries = list(manifest["arrays"].values())
    if manifest.get("potential_features"):
        entries.append(manifest["potential_features"])
    for entry in entries:
        file = _safe(directory, entry["path"])
        if verify_hashes and file_sha256(file) != entry["sha256"]:
            raise ValueError(f"Activation array checksum mismatch: {entry['path']}")
        array = np.load(file, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"] or array.shape[0] != len(records):
            raise ValueError("Activation cache array header mismatch")
    return manifest


def load_cached_rows(path):
    manifest = load_cache(path)
    return [json.loads(line) for line in (_root(path) / manifest["rows_path"]).read_text().splitlines() if line]


def load_cached_array(path, layer, mmap_mode="r"):
    manifest = load_cache(path)
    key = str(int(layer))
    if key not in manifest["arrays"]:
        raise ValueError(f"Cache has no hidden-state index {layer}")
    return np.load(_root(path) / manifest["arrays"][key]["path"], mmap_mode=mmap_mode, allow_pickle=False)


def read_cache(path):
    manifest = load_cache(path)
    directory = _root(path)
    rows = [json.loads(line) for line in (directory / manifest["rows_path"]).read_text().splitlines() if line]
    arrays = {int(layer): np.load(directory / entry["path"], mmap_mode="r", allow_pickle=False)
              for layer, entry in manifest["arrays"].items()}
    return rows, arrays, manifest


def extract_features(runtime, records, layers, readout, cache_dir, resume=True, potential_features=False):
    with _writer_lock(cache_dir):
        return _extract_features_locked(runtime, records, layers, readout, cache_dir, resume, potential_features)


def _extract_features_locked(runtime, records, layers, readout, cache_dir, resume=True, potential_features=False):
    validate_records(records)
    readout = "span_mean" if readout == "source_span" else readout
    layers = sorted(set(map(int, layers)))
    if not layers:
        raise ValueError("At least one layer required")
    directory = Path(cache_dir)
    if directory.is_symlink():
        raise ValueError("Cache output cannot be a symlink")
    descriptor = {"schema_version": CACHE_SCHEMA_VERSION, "dataset_sha256": dataset_identity(records),
                  "model": runtime.identity(), "layers": layers, "readout": readout,
                  "potential_features": bool(potential_features), "extraction_code_sha256": _code_identity()}
    cache_key = identity(descriptor)
    if (directory / "manifest.json").exists():
        complete = load_cache(directory)
        if complete["cache_key"] != cache_key:
            raise ValueError("Cache identity differs; choose a fresh output namespace")
        return complete
    directory.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "progress.json"
    if progress_path.exists():
        if not resume:
            raise ValueError("Partial cache exists and resume=False")
        progress = read_json(progress_path)
        if progress.get("cache_key") != cache_key:
            raise ValueError("Partial cache belongs to a different dataset/model/readout/code identity")
    else:
        if any(p.name != ".writer.lock" for p in directory.iterdir()):
            raise ValueError("Refusing nonempty cache without a valid progress identity")
        rows_path = directory / "rows.jsonl"
        rows_path.write_bytes(_rows_bytes(records))
        progress = {"cache_key": cache_key, "descriptor": descriptor, "completed_rows": 0,
                    "arrays": {}, "chunks": [], "rows_sha256": file_sha256(rows_path)}
        atomic_json(progress_path, progress)
    if file_sha256(directory / "rows.jsonl") != progress["rows_sha256"]:
        raise ValueError("Partial-cache row manifest was modified")
    arrays = {key: np.load(_safe(directory, info["path"]), mmap_mode="r+", allow_pickle=False)
              for key, info in progress["arrays"].items()}
    for key, array in arrays.items():
        info = progress["arrays"][key]
        if list(array.shape) != info["shape"] or str(array.dtype) != "float32":
            raise ValueError("Partial array header mismatch")
    cursor = 0
    for chunk in progress["chunks"]:
        if chunk["start"] != cursor or chunk["end"] <= cursor:
            raise ValueError("Noncontiguous cache progress")
        for key, digest in chunk["hashes"].items():
            if key not in arrays or _chunk_hash(arrays[key][chunk["start"]:chunk["end"]]) != digest:
                raise ValueError("Committed partial-cache chunk failed verification")
        cursor = chunk["end"]
    if cursor != progress["completed_rows"] or cursor > len(records):
        raise ValueError("Cache progress row count mismatch")
    initial_cursor = cursor
    remaining = runtime.features(records[initial_cursor:], layers=layers, readout=readout) if initial_cursor < len(records) else []
    for relative_indices, outputs in remaining:
        indices = [initial_cursor + int(i) for i in relative_indices]
        if indices != list(range(cursor, cursor + len(indices))) or not indices or indices[-1] >= len(records):
            raise ValueError("Runtime must emit every row once in contiguous source order")
        batches = {str(layer): np.asarray(outputs[layer], dtype=np.float32) for layer in layers}
        if potential_features:
            batches["potential_features"] = np.asarray(runtime.input_entity_features([records[i] for i in indices]), dtype=np.float32)
        for key, value in batches.items():
            if value.ndim != 2 or value.shape[0] != len(indices) or not np.isfinite(value).all():
                raise ValueError("Runtime emitted malformed/nonfinite pooled vectors")
            if key not in arrays:
                if cursor != 0:
                    raise ValueError("Partial cache lacks a requested array")
                file_name = "potential_features.npy" if key == "potential_features" else f"layer_{key}.npy"
                arrays[key] = np.lib.format.open_memmap(directory / file_name, mode="w+", dtype=np.float32,
                                                       shape=(len(records), value.shape[1]))
                progress["arrays"][key] = {"path": file_name, "shape": [len(records), value.shape[1]], "dtype": "float32"}
            if arrays[key].shape[1] != value.shape[1]:
                raise ValueError("Runtime feature dimension changed across batches")
            arrays[key][indices] = value
            arrays[key].flush()
            with (directory / progress["arrays"][key]["path"]).open("rb") as handle:
                os.fsync(handle.fileno())
        end = cursor + len(indices)
        progress["chunks"].append({"start": cursor, "end": end,
                                   "hashes": {key: _chunk_hash(value) for key, value in batches.items()}})
        progress["completed_rows"] = end
        atomic_json(progress_path, progress)
        cursor = end
    if cursor != len(records):
        raise ValueError("Runtime stopped before all rows; cache remains incomplete")
    entries = {key: {**info, "sha256": file_sha256(directory / info["path"])} for key, info in progress["arrays"].items()}
    manifest = {**descriptor, "descriptor": descriptor, "cache_key": cache_key, "status": "complete", "n_rows": len(records),
                "rows_path": "rows.jsonl", "rows_sha256": progress["rows_sha256"],
                "arrays": {str(layer): entries[str(layer)] for layer in layers},
                "potential_features": entries.get("potential_features"),
                "telemetry": list(getattr(runtime, "telemetry", []))}
    atomic_json(directory / "manifest.json", manifest)
    load_cache(directory)
    return manifest


def merge_feature_caches(manifests, outdir, ordered_records=None):
    """Merge verified disjoint phases with no inference or duplicate observations."""
    if not manifests:
        raise ValueError("No cache phases supplied")
    sources = [(Path(path), *read_cache(path)) for path in manifests]
    first = sources[0][3]
    compatibility = lambda m: {k: m[k] for k in ("model", "layers", "readout", "extraction_code_sha256")}
    lookup, merged_rows = {}, []
    for si, (_path, rows, _arrays, manifest) in enumerate(sources):
        if compatibility(manifest) != compatibility(first):
            raise ValueError("Cannot merge caches with different model/layer/readout/code identities")
        if bool(manifest.get("potential_features")) != bool(first.get("potential_features")):
            raise ValueError("Potential features must exist in every merged phase or none")
        for ri, row in enumerate(rows):
            if row["row_id"] in lookup:
                raise ValueError(f"Duplicate phase row {row['row_id']}")
            lookup[row["row_id"]] = (si, ri, row)
            merged_rows.append(row)
    if ordered_records is not None:
        merged_rows = list(ordered_records)
        for row in merged_rows:
            if row["row_id"] not in lookup or identity(row) != identity(lookup[row["row_id"]][2]):
                raise ValueError("Requested merge row missing or differs from sealed phase data")
    validate_records(merged_rows)
    directory = Path(outdir)
    merge_descriptor = {"dataset_sha256": dataset_identity(merged_rows),
                        "source_cache_keys": [s[3]["cache_key"] for s in sources], **compatibility(first)}
    key = identity(merge_descriptor)
    if (directory / "manifest.json").exists():
        result = load_cache(directory)
        if result["cache_key"] != key:
            raise ValueError("Merge output belongs to a different phase selection")
        return result
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Partial merge output exists; choose a fresh merge output")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "rows.jsonl").write_bytes(_rows_bytes(merged_rows))
    entries = {}
    keys = [str(layer) for layer in first["layers"]] + (["potential_features"] if first.get("potential_features") else [])
    for name in keys:
        info = first["potential_features"] if name == "potential_features" else first["arrays"][name]
        path = f"{name}.npy" if name == "potential_features" else f"layer_{name}.npy"
        target = np.lib.format.open_memmap(directory / path, mode="w+", dtype=np.float32, shape=(len(merged_rows), info["shape"][1]))
        source_arrays = [np.load(_root(source[0]) / (source[3]["potential_features"] if name == "potential_features" else source[3]["arrays"][name])["path"], mmap_mode="r", allow_pickle=False) for source in sources]
        for i, row in enumerate(merged_rows):
            si, ri, _ = lookup[row["row_id"]]
            target[i] = source_arrays[si][ri]
        target.flush()
        entries[name] = {"path": path, "shape": list(target.shape), "dtype": "float32", "sha256": file_sha256(directory / path)}
        del target
    manifest = {"schema_version": CACHE_SCHEMA_VERSION, **merge_descriptor, "descriptor": merge_descriptor, "cache_key": key, "status": "complete",
                "n_rows": len(merged_rows), "rows_path": "rows.jsonl", "rows_sha256": file_sha256(directory / "rows.jsonl"),
                "arrays": {str(layer): entries[str(layer)] for layer in first["layers"]},
                "potential_features": entries.get("potential_features"),
                "excluded_source_rows": len(lookup) - len(merged_rows)}
    atomic_json(directory / "manifest.json", manifest)
    load_cache(directory)
    return manifest
