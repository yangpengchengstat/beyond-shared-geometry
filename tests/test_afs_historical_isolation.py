from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from bsg.data.c7 import build_dataset, transform_state
from bsg.experiments.algebra_falsification_units import HistoricalCheckpointSource
from bsg.experiments.fitting import GroupSpec
from bsg.groups import c7
from bsg.metrics.relations import c7_relations
from bsg.experiments.main_config import CANONICAL_REVISIONS


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_historical_import_validates_hash_and_cannot_write_source():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint = root / "checkpoints" / "structural_base" / "cell.npz"
        checkpoint.parent.mkdir(parents=True)
        z = np.zeros((280, 16))
        np.savez_compressed(
            checkpoint, pca_mean=np.zeros(20), pca_components=np.zeros((16, 20)),
            pca_explained=np.zeros(16), pca_fit_rows=np.asarray(196), z=z,
            operators=np.stack([np.eye(16) for _ in c7.ELEMENTS]),
        )
        expected = {"run_id": "main_full_003", "experiment": "beyond_shared_geometry_main", "tier": "full", "status": "SUCCESS", "git_commit": "abc", "config_hash": "def"}
        models = [{"id": model, "requested_revision": revision} for model, revision in CANONICAL_REVISIONS.items()]
        (root / "manifest.json").write_text(json.dumps({**expected, "git_dirty": False, "models": models, "primary_latent_dimension": 16, "template_split": {"train_templates": 28, "validation_templates": 6, "test_templates": 6}}), encoding="utf-8")
        fields = {"model_key": "m", "model": "model", "group": "C7", "depth_slot": 0.4, "layer": 6, "split_seed": 101, "latent_dim": 16}
        terminal_fields = {**fields, "depth_slot": "terminal", "layer": 16}
        ledger = {"units": {
            "structural_base:terminal": {"status": "COMPLETE", "fields": terminal_fields, "outputs": []},
            "structural_base:key": {"status": "COMPLETE", "fields": fields, "outputs": [{"path": "/old/cell.npz", "bytes": checkpoint.stat().st_size, "sha256": _digest(checkpoint)}]},
        }}
        (root / "run_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
        source = HistoricalCheckpointSource(root, expected)
        spec = GroupSpec("C7", build_dataset(), c7.ELEMENTS, c7.IDENTITY, c7.compose, c7.inverse, transform_state, c7_relations())
        artifact = source.load_cell(spec, model_key="m", model="model", depth_slot=0.4, layer=6, split_seed=101, latent_dim=16)
        assert artifact["z"].shape == (280, 16)
        source.assert_unchanged()
        checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
        try:
            source.assert_unchanged()
        except RuntimeError:
            pass
        else:
            raise AssertionError("historical mutation was not detected")


def test_historical_import_rejects_bad_checksum_before_loading():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); (root / "checkpoints" / "structural_base").mkdir(parents=True)
        checkpoint = root / "checkpoints" / "structural_base" / "cell.npz"
        checkpoint.write_bytes(b"not-an-npz")
        expected = {"run_id": "main_full_003", "experiment": "beyond_shared_geometry_main", "tier": "full", "status": "SUCCESS", "git_commit": "abc", "config_hash": "def"}
        models = [{"id": model, "requested_revision": revision} for model, revision in CANONICAL_REVISIONS.items()]
        (root / "manifest.json").write_text(json.dumps({**expected, "git_dirty": False, "models": models, "primary_latent_dimension": 16, "template_split": {"train_templates": 28, "validation_templates": 6, "test_templates": 6}}), encoding="utf-8")
        fields = {"model_key": "m", "model": "model", "group": "C7", "depth_slot": 0.4, "layer": 6, "split_seed": 101, "latent_dim": 16}
        ledger = {"units": {"structural_base:key": {"status": "COMPLETE", "fields": fields, "outputs": [{"path": "/old/cell.npz", "bytes": checkpoint.stat().st_size, "sha256": "0" * 64}]}}}
        (root / "run_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
        source = HistoricalCheckpointSource(root, expected)
        spec = GroupSpec("C7", build_dataset(), c7.ELEMENTS, c7.IDENTITY, c7.compose, c7.inverse, transform_state, c7_relations())
        try:
            source.load_cell(spec, model_key="m", model="model", depth_slot=0.4, layer=6, split_seed=101, latent_dim=16)
        except ValueError as error:
            assert "checksum" in str(error)
        else:
            raise AssertionError("bad historical checksum was accepted")
