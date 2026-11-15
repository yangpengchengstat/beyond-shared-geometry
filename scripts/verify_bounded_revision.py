"""Check bounded-analysis provenance without executing any study or resampling."""
from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    manifest = json.loads((root / "release/bounded_revision_manifest.json").read_text())
    for group in ("files", "unchanged_inputs"):
        for name, expected in manifest[group].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Nonrelative manifest entry")
            full = root / path
            if full.is_symlink() or hashlib.sha256(full.read_bytes()).hexdigest() != expected:
                raise ValueError("Bounded revision hash mismatch: " + name)
    report = json.loads((root / "results/main30_bounded_v1/validation_report.json").read_text())
    assert report["status"] == "DONE" and report["source_files_unchanged"]
    assert report["counts"] == {"spectral_cells": 48, "image_conditions": 2952,
                                "type_rows": 216, "identity_direct_checks": 48}
    assert not report["family_sensitivity_run"] and not report["new_model_execution"]
    config = json.loads((root / "analysis/rank_sensitivity_v1/config.json").read_text())
    availability = json.loads((root / "analysis/rank_sensitivity_v1/preparation_audit/availability.json").read_text())
    assert config["full_run"] is False
    assert availability["status"] == "PREPARED_NOT_RUN"
    assert availability["real_family_deletions_executed"] == 0
    assert not availability["coordinate_values_loaded"]
    return {"status": "verified", "files": len(manifest["files"]),
            "unchanged_inputs": len(manifest["unchanged_inputs"]),
            "new_scientific_execution_during_check": False,
            "family_sensitivity": "PREPARED_NOT_RUN"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
