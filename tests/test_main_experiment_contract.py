from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from bsg.analysis.representation_theory import c7_diagnostics, d4_diagnostics
from bsg.experiments.main_config import CANONICAL_MODELS, config_hash, resolve_main_config
from bsg.experiments.main_experiment import _group_specs, _limit_templates, _mean_latent_delta, _split
from bsg.experiments.main_outputs import SUBDIRS, create_output_tree
from bsg.experiments.fitting import GroupSpec
from bsg.experiments.manifest import REQUIRED_FIELDS, build_manifest, validate_manifest
from bsg.groups import c7, d4
from bsg.interventions.causal import residual_preserving_intervention
from bsg.metrics.cross_model import linear_cka, orthogonal_procrustes
from bsg.representations.cache import ActivationCache, activation_cache_key
from bsg.representations.pca import PCASubspace
from bsg.statistics.hierarchical import hierarchical_mean_ci, hierarchical_spearman_ci, paired_baseline_ci


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "main_experiment.yaml"


def _permutation_ops(group):
    elements, states, apply = (c7.ELEMENTS, range(7), c7.apply) if group == "C7" else (d4.ELEMENTS, range(4), d4.apply)
    ops = {}
    for g in elements:
        matrix = np.zeros((len(states), len(states)))
        for state in states:
            matrix[state, apply(g, state)] = 1.0
        ops[g] = matrix
    return ops


def test_main_config_is_reproducible_and_exact():
    first = resolve_main_config(CONFIG, "full")
    second = resolve_main_config(CONFIG, "full")
    assert config_hash(first) == config_hash(second)
    assert {model["id"]: model["blocks"] for model in first["selected_models"]} == CANONICAL_MODELS
    assert all(layer < model["blocks"] for model in first["selected_models"] for layer in model["causal_layers"])


def test_smoke_tier_exercises_every_stage_contract():
    config = resolve_main_config(CONFIG, "smoke")
    assert len(config["selected_models"]) == 1
    assert config["groups"] == ["C7", "D4"]
    assert config["estimators"] == ["independent_ridge", "group_consistent_joint"]
    assert config["causal"]["c7_steps"] == list(range(1, 8))
    assert set(config["causal"]["d4_relations"]) == {"r4_identity", "s2_identity", "srs_r_inverse"}


def test_manifest_complete_and_output_tree_canonical():
    config = resolve_main_config(CONFIG, "smoke")
    manifest = build_manifest(config, "smoke_test", ROOT)
    validate_manifest(manifest)
    assert REQUIRED_FIELDS <= set(manifest)
    with tempfile.TemporaryDirectory() as tmp_path:
        output = create_output_tree(tmp_path, "smoke_test")
        assert all((output / name).is_dir() for name in SUBDIRS)


def test_activation_cache_key_and_round_trip():
    kwargs = dict(model_id="m", model_revision="abc", task="C7", layer=4, prompt_template_split={"seed": 42}, tokenizer_settings={"use_fast": True})
    assert activation_cache_key(**kwargs) == activation_cache_key(**kwargs)
    assert activation_cache_key(**kwargs) != activation_cache_key(**{**kwargs, "layer": 5})
    with tempfile.TemporaryDirectory() as tmp_path:
        cache = ActivationCache(tmp_path)
        values = np.arange(12).reshape(3, 4)
        cache.store(activation_cache_key(**kwargs), values, kwargs)
        np.testing.assert_array_equal(cache.load(activation_cache_key(**kwargs)), values)


def test_intervention_preserves_outside_subspace_residual():
    rng = np.random.default_rng(2)
    pca = PCASubspace(3).fit(rng.normal(size=(20, 7)))
    hidden = rng.normal(size=7)
    operator, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    intervened = residual_preserving_intervention(hidden, pca, operator)
    projector = pca.components_.T @ pca.components_
    np.testing.assert_allclose((intervened - hidden) @ (np.eye(7) - projector), 0.0, atol=1e-10)


def test_mean_difference_baseline_uses_training_rows_only():
    config = resolve_main_config(CONFIG, "smoke")
    base = _group_specs()["D4"]
    frame = _limit_templates(base.frame, config["template_limit"])
    spec = GroupSpec(
        base.name, frame, base.elements, base.identity, base.compose,
        base.inverse, base.transform_state, base.relations,
    )
    split = _split(spec, config, config["split_seeds"][0])
    rng = np.random.default_rng(91)
    z = rng.normal(size=(len(frame), 16))
    expected = _mean_latent_delta(spec, z, split.train_mask, d4.R)
    altered = z.copy()
    altered[~split.train_mask] += 1_000_000
    np.testing.assert_allclose(_mean_latent_delta(spec, altered, split.train_mask, d4.R), expected)


def test_hierarchical_bootstrap_and_pairing_reproducible():
    rows = []
    for split in (1, 2):
        for opt in (10, 20):
            for template in (3, 4, 5):
                for method, shift in (("learned", -0.5), ("baseline", 0.0)):
                    rows.append({"split_seed": split, "optimization_seed": opt, "template_id": template, "method": method, "value": split + opt / 10 + template / 100 + shift})
    frame = pd.DataFrame(rows)
    learned = frame[frame.method == "learned"]
    a = hierarchical_mean_ci(learned, "value", n_boot=100, seed=7)
    b = hierarchical_mean_ci(learned, "value", n_boot=100, seed=7)
    assert a == b
    paired = paired_baseline_ci(frame, "value", "learned", "baseline", n_boot=100, seed=7)
    assert abs(paired["mean"] + 0.5) < 1e-12
    association_frame = learned.assign(x=np.arange(len(learned)), y=np.arange(len(learned)) ** 2)
    c = hierarchical_spearman_ci(association_frame, "x", "y", n_boot=100, seed=7)
    d = hierarchical_spearman_ci(association_frame, "x", "y", n_boot=100, seed=7)
    assert c == d and c["mean"] > 0.99


def test_cross_model_and_irrep_diagnostics_exact_cases():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 5))
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    y = x @ q
    assert linear_cka(x, y) > 0.999999
    _, error = orthogonal_procrustes(x, y)
    assert error < 1e-20
    assert c7_diagnostics(_permutation_ops("C7"))["cycle_operator_error"] < 1e-12
    assert d4_diagnostics(_permutation_ops("D4"))["r_fourth_power_error"] < 1e-12


def test_archive_is_never_imported_by_active_runtime():
    active_roots = [ROOT / "src", ROOT / "runpod", ROOT / "scripts"]
    offenders = []
    for active_root in active_roots:
        for path in active_root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh"}:
                text = path.read_text(encoding="utf-8")
                if "import archive" in text or "from archive" in text or "archive/historical_experiments" in text:
                    offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_protocol_documents_cycle_and_terminal_layer_rules():
    protocol = (ROOT / "docs" / "EXPERIMENT_PROTOCOL.md").read_text(encoding="utf-8")
    assert "k=7" in protocol and "cycle recovery" in protocol
    assert "Terminal blocks are diagnostic structural endpoints and are never causal" in protocol
    assert "h_intervened = h - z C + (z R^k) C" in protocol
