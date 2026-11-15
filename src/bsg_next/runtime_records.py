"""Durable maintenance-v2 execution records; no scientific decisions live here."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import os
import tempfile
import time
import uuid

from .io import atomic_json, identity, read_json

CHECKPOINT_SCHEMA = 2


def save_record(path, payload):
    """Commit a completed payload; an interrupted temporary file is never valid."""
    if Path(path).is_symlink():
        raise ValueError("Maintenance checkpoint cannot be a symbolic link")
    atomic_json(path, {"schema_version": CHECKPOINT_SCHEMA, "payload": payload,
                       "payload_sha256": identity(payload)})


def load_record(path):
    if Path(path).is_symlink():
        raise ValueError("Maintenance checkpoint cannot be a symbolic link")
    record = read_json(path)
    if (record.get("schema_version") != CHECKPOINT_SCHEMA or
            identity(record.get("payload")) != record.get("payload_sha256")):
        raise ValueError(f"Maintenance checkpoint checksum/schema mismatch: {path}")
    return record["payload"]


def atomic_npz(path, parameters):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **parameters)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Session:
    def __init__(self, root, stage, total):
        self.root, self.stage, self.total = Path(root), stage, int(total)
        self.started = time.perf_counter()
        self.completed = self.resumed = self.new = 0
        self.record = {"session_id": uuid.uuid4().hex, "stage": stage,
                       "started_at_utc": datetime.now(timezone.utc).isoformat(),
                       "status": "running", "total": self.total}
        self.path = self.root / "sessions" / (self.record["session_id"] + ".json")
        self.update()

    def update(self, *, status="running", detail=None):
        elapsed = time.perf_counter() - self.started
        self.record.update(status=status, completed=self.completed, resumed=self.resumed,
                           new=self.new, elapsed_this_session_seconds=elapsed, detail=detail)
        atomic_json(self.path, self.record)
        atomic_json(self.root / "progress.json", self.record)

    def advance(self, resumed=False, detail=None):
        self.completed += 1
        self.resumed += int(resumed)
        self.new += int(not resumed)
        self.update(detail=detail)
        print(f"PROGRESS stage={self.stage} completed={self.completed}/{self.total} "
              f"resumed={self.resumed} new={self.new} "
              f"elapsed_session={self.record['elapsed_this_session_seconds']:.1f}s "
              f"detail={detail}", flush=True)


@contextmanager
def recorded_session(output, stage, total):
    """Single writer, with a separate durable record for every invocation.

    Call only after prepare_run: old source/config/output identities are rejected
    before this helper creates operational bookkeeping. A killed process leaves
    its session marked running, never falsely complete; later attempts are new.
    """
    import fcntl
    output = Path(output)
    with (output / ".maintenance-v2.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is writing this maintenance run") from exc
        session = Session(output, stage, total)
        try:
            yield session
            if session.completed != session.total:
                raise ValueError("Maintenance task returned before all planned units completed")
        except BaseException as exc:
            session.update(status="interrupted_or_failed", detail=f"{type(exc).__name__}: {exc}")
            raise
        else:
            session.update(status="complete")
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
