"""Atomic run ledger, checkpoint verification, and runtime telemetry."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_key(module: str, fields: dict) -> str:
    payload = {"experiment_module": str(module), **fields}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"{module}:{hashlib.sha256(raw).hexdigest()}"


def atomic_write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def atomic_write_json(path: Path, payload) -> Path:
    return atomic_write_text(Path(path), json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


class RunLedger:
    schema_version = 1

    def __init__(self, path: Path, run_id: str, config_hash: str):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("run_id") != run_id or self.data.get("config_hash") != config_hash:
                raise ValueError("run ledger does not match run/config identity")
        else:
            self.data = {
                "schema_version": self.schema_version,
                "run_id": run_id,
                "config_hash": config_hash,
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "units": {},
            }
            self._save()

    def _save(self) -> None:
        self.data["updated_at_utc"] = utc_now()
        atomic_write_json(self.path, self.data)

    def begin(self, key: str, fields: dict) -> None:
        attempts = int(self.data["units"].get(key, {}).get("attempts", 0)) + 1
        self.data["units"][key] = {
            "status": "RUNNING",
            "fields": fields,
            "attempts": attempts,
            "started_at_utc": utc_now(),
        }
        self._save()

    def complete(self, key: str, fields: dict, output_paths: list[Path], metrics: dict | None = None) -> None:
        outputs = []
        for path in output_paths:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"cannot complete ledger unit without output: {path}")
            outputs.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        previous = self.data["units"].get(key, {})
        self.data["units"][key] = {
            "status": "COMPLETE",
            "fields": fields,
            "attempts": int(previous.get("attempts", 1)),
            "started_at_utc": previous.get("started_at_utc"),
            "completed_at_utc": utc_now(),
            "outputs": outputs,
            "metrics": metrics or {},
        }
        self._save()

    def fail(self, key: str, fields: dict, error: BaseException) -> None:
        previous = self.data["units"].get(key, {})
        self.data["units"][key] = {
            "status": "FAILED",
            "fields": fields,
            "attempts": int(previous.get("attempts", 1)),
            "started_at_utc": previous.get("started_at_utc"),
            "failed_at_utc": utc_now(),
            "exception_type": type(error).__name__,
            "message": str(error),
        }
        self._save()

    def should_skip(self, key: str) -> bool:
        unit = self.data["units"].get(key)
        if not unit or unit.get("status") != "COMPLETE":
            return False
        for output in unit.get("outputs", []):
            path = Path(output["path"])
            if not path.is_file() or path.stat().st_size != int(output["bytes"]) or sha256_file(path) != output["sha256"]:
                unit["status"] = "STALE"
                unit["stale_at_utc"] = utc_now()
                unit["stale_reason"] = f"missing or corrupt checkpoint: {path}"
                self._save()
                return False
        return True

    def failed_units(self) -> list[dict]:
        return [unit for unit in self.data["units"].values() if unit.get("status") == "FAILED"]

    def assert_complete(self, required_keys: set[str]) -> None:
        missing = sorted(key for key in required_keys if not self.should_skip(key))
        if missing:
            raise RuntimeError(f"run finalization blocked by {len(missing)} incomplete required units")


class RuntimeTelemetry:
    COLUMNS = (
        "experiment_key", "experiment_module", "model", "group", "layer", "latent_dim",
        "split_seed", "optimization_seed", "wall_clock_seconds", "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes", "cache_bytes_read", "cache_bytes_written", "event",
        "progress_completed", "progress_total", "timestamp_utc",
    )

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            self.frame = pd.read_csv(self.path)
        else:
            self.frame = pd.DataFrame(columns=self.COLUMNS)

    @staticmethod
    def reset_gpu_peak() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ModuleNotFoundError:
            pass

    @staticmethod
    def gpu_peaks() -> tuple[int, int]:
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
        except ModuleNotFoundError:
            pass
        return 0, 0

    def record(
        self,
        key: str,
        module: str,
        fields: dict,
        started: float,
        *,
        cache_read: int = 0,
        cache_written: int = 0,
        event: str = "COMPLETE",
        progress_completed: int | None = None,
        progress_total: int | None = None,
    ) -> None:
        allocated, reserved = self.gpu_peaks()
        row = {
            "experiment_key": key,
            "experiment_module": module,
            "model": fields.get("model"),
            "group": fields.get("group"),
            "layer": fields.get("layer"),
            "latent_dim": fields.get("latent_dim"),
            "split_seed": fields.get("split_seed"),
            "optimization_seed": fields.get("optimization_seed"),
            "wall_clock_seconds": max(0.0, time.perf_counter() - started),
            "gpu_peak_allocated_bytes": allocated,
            "gpu_peak_reserved_bytes": reserved,
            "cache_bytes_read": int(cache_read),
            "cache_bytes_written": int(cache_written),
            "event": event,
            "progress_completed": progress_completed,
            "progress_total": progress_total,
            "timestamp_utc": utc_now(),
        }
        self.frame = pd.concat([self.frame, pd.DataFrame([row])], ignore_index=True)
        atomic_write_frame(self.frame, self.path)
