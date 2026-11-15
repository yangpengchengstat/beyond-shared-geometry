from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "promote_results.py"
    spec = importlib.util.spec_from_file_location("promote_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_successful_small_artifacts_can_be_promoted_with_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = root / "run"
        (run / "tables").mkdir(parents=True)
        (run / "figures").mkdir()
        (run / "manifest.json").write_text(json.dumps({"status": "SUCCESS", "run_id": "smoke_1", "git_commit": "abc", "config_hash": "def"}), encoding="utf-8")
        (run / "tables" / "summary.csv").write_text("metric,value\nE_eq,0.1\n", encoding="utf-8")
        (run / "figures" / "summary.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
        records = _module().promote(run, root / "results")
        assert len(records) == 2
        promotion = json.loads((root / "results" / "PROMOTION_MANIFEST.json").read_text(encoding="utf-8"))
        assert promotion["source_run_id"] == "smoke_1"
