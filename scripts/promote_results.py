"""Promote reviewed, lightweight run artifacts into publication directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ALLOWED = {".csv", ".tex", ".svg", ".png", ".pdf"}
MAX_BYTES = 10 * 1024 * 1024


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def promote(run_dir: Path, results_dir: Path, replace: bool = False) -> list[dict]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SUCCESS":
        raise ValueError("only successful runs may promote artifacts")
    records = []
    artifact_root = run_dir / "results" if (run_dir / "results").is_dir() else run_dir
    for category in ("tables", "figures"):
        source_dir = artifact_root / category
        target_dir = results_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(source_dir.iterdir() if source_dir.is_dir() else []):
            if not source.is_file() or source.suffix.lower() not in ALLOWED:
                continue
            if source.stat().st_size > MAX_BYTES:
                raise ValueError(f"artifact exceeds {MAX_BYTES} bytes: {source}")
            target = target_dir / source.name
            if target.exists() and not replace:
                raise FileExistsError(f"refusing to overwrite {target}; pass --replace after review")
            shutil.copy2(source, target)
            records.append({"source": str(source), "destination": str(target), "bytes": target.stat().st_size, "sha256": sha256(target)})
    if not records:
        raise ValueError("run contains no promotable table or figure artifacts")
    promotion = {
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": manifest.get("run_id"),
        "source_git_commit": manifest.get("git_commit"),
        "source_config_hash": manifest.get("config_hash"),
        "artifacts": records,
    }
    (results_dir / "PROMOTION_MANIFEST.json").write_text(json.dumps(promotion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    records = promote(args.run_dir.resolve(), args.results_dir.resolve(), args.replace)
    print(f"Promoted {len(records)} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
