"""Revision-aware activation cache with embedded provenance validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class CacheMetadataMismatch(RuntimeError):
    pass


def activation_cache_key(**metadata) -> str:
    required = {"model_id", "model_revision", "task", "layer", "prompt_template_split", "tokenizer_settings"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"activation cache metadata missing fields: {sorted(missing)}")
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ActivationCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.last_event = {"status": "none", "bytes_read": 0, "bytes_written": 0}

    def path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def load(self, key: str, expected_metadata: dict | None = None):
        path = self.path(key)
        if not path.exists():
            self.last_event = {"status": "miss", "bytes_read": 0, "bytes_written": 0}
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if "activations" not in data or "metadata" not in data:
                    raise CacheMetadataMismatch(f"cache file lacks required arrays: {path}")
                stored_metadata = json.loads(str(data["metadata"].item()))
                if expected_metadata is not None and stored_metadata != expected_metadata:
                    raise CacheMetadataMismatch(f"activation cache metadata mismatch: {path}")
                activations = np.asarray(data["activations"])
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise CacheMetadataMismatch(f"invalid activation cache file {path}: {error}") from error
        self.last_event = {"status": "hit", "bytes_read": path.stat().st_size, "bytes_written": 0}
        return activations

    def store(self, key: str, activations, metadata: dict) -> Path:
        expected_key = activation_cache_key(**metadata)
        if expected_key != key:
            raise ValueError("activation cache key does not match metadata")
        path = self.path(key)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            activations=np.asarray(activations, dtype=np.float32),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        tmp.replace(path)
        self.last_event = {"status": "write", "bytes_read": 0, "bytes_written": path.stat().st_size}
        return path

    def discard(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)
