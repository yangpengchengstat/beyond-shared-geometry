"""Read-only integrity check for copied evidence; no scientific recomputation."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path


PRIVATE_PATH_RE = re.compile(rb"(?:/(?:Users|home)/[A-Za-z0-9_.-]+/|/root/|/private/(?:tmp|var)/|[A-Za-z]:\\Users\\)")


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "source_manifest.json").read_text())
    counts = {
        "dimension_rank/dimension_rank_audit.csv": 13784,
        "dimension_rank/dimension_sensitivity_existing.csv": 6144,
        "dimension_rank/dimension_sensitivity_summary.csv": 128,
        "dimension_rank/dimension_sensitivity_coverage.csv": 8,
    }
    checked = []
    for item in manifest["sources"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest entry escapes the evidence directory")
        data = path.read_bytes()
        if len(data) != item["packaged_bytes"]:
            raise ValueError("Byte-count mismatch: " + item["path"])
        if hashlib.sha256(data).hexdigest() != item["packaged_sha256"]:
            raise ValueError("SHA256 mismatch: " + item["path"])
        if PRIVATE_PATH_RE.search(data):
            raise ValueError("Personal absolute path in scientific copy: " + item["path"])
        if item["path"] in counts:
            rows = list(csv.reader(io.StringIO(data.decode(), newline="")))
            if len(rows) - 1 != counts[item["path"]]:
                raise ValueError("Row-count mismatch: " + item["path"])
        else:
            json.loads(data)
        checked.append(item["path"])
    if len(checked) != 9 or len(set(checked)) != 9:
        raise ValueError("Expected nine distinct scientific evidence files")
    records_root = root / "dimension_rank/audit_records"
    historical = json.loads((records_root / "source_manifest.json").read_text())
    historical_checked = []
    for item in historical["sources"]:
        path = (records_root / item["path"]).resolve()
        if not path.is_relative_to(records_root):
            raise ValueError("Historical record escapes its evidence directory")
        data = path.read_bytes()
        if len(data) != item["packaged_bytes"] or hashlib.sha256(data).hexdigest() != item["packaged_sha256"]:
            raise ValueError("Historical record identity mismatch: " + item["path"])
        if PRIVATE_PATH_RE.search(data):
            raise ValueError("Local absolute path in historical evidence: " + item["path"])
        historical_checked.append(item["path"])
    if len(historical_checked) != 59 or len(set(historical_checked)) != 59:
        raise ValueError("Expected 59 distinct historical records")
    print(json.dumps({"status": "PASS", "scientific_files_checked": len(checked), "historical_records_checked": len(historical_checked), "new_scientific_analysis": False}, indent=2))


if __name__ == "__main__":
    main()
