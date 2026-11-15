from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from bsg.experiments.main_config import CANONICAL_PAIR_KEYS, resolve_main_config
from bsg.experiments.main_outputs import STABLE_FIGURES, STABLE_TABLES, validate_stable_artifacts
from bsg.experiments.main_experiment import (
    _bootstrap_rows,
    _cross_work_pairs,
    _estimate_full_cost,
    _full_calibration_multipliers,
    _full_workload_unit_targets,
    _generate_stable_artifacts,
    _render_cost_projection,
    _scientific_unit_targets,
    _can_resume_postprocessing,
)
from bsg.experiments.manifest import validate_git_policy
from bsg.experiments.runtime_state import RunLedger, experiment_key
from bsg.groups import d4
from bsg.metrics.relations import d4_relation_components_data, d4_relation_components_op
from bsg.representations.cache import ActivationCache, CacheMetadataMismatch, activation_cache_key


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "main_experiment.yaml"


def test_seed_namespaces_are_separate_and_deterministic_full():
    config = resolve_main_config(CONFIG, "full")
    namespaces = [
        set(config["split_seeds"]), set(config["optimization_seeds"]),
        set(config["random_baseline_seeds"]), {config["bootstrap_seed"]},
    ]
    assert all(not (namespaces[i] & namespaces[j]) for i in range(4) for j in range(i + 1, 4))
    assert config["learned_fit_semantics"] == "deterministic_single"
    assert config["learned_fit_seeds"] == [42]
    assert len(config["random_baseline_seeds"]) == 5


def test_smoke_uses_approved_rank_feasible_primary_dimension():
    config = resolve_main_config(CONFIG, "smoke")
    assert config["structural_latent_dimensions"] == [16]
    assert config["causal_latent_dimensions"] == [16]
    assert config["template_limit"] == 7
    assert config["template_split"] == {"train_templates": 5, "validation_templates": 1, "test_templates": 1}
    assert config["selected_models"][0]["primary_layers"] == [6]
    assert _cross_work_pairs(config) == [("llama_1b", "llama_1b", "self_smoke_diagnostic")]


def test_terminal_layers_are_never_causal_and_pair_matrix_is_exact():
    config = resolve_main_config(CONFIG, "full")
    assert tuple(tuple(pair) for pair in config["cross_model_pairs"]) == CANONICAL_PAIR_KEYS
    for model in config["selected_models"]:
        assert model["terminal_structural_layer"] in model["layers"]
        assert model["terminal_structural_layer"] not in model["causal_layers"]


def test_d4_srs_exact_toy_case_and_reversed_order_failure():
    operators = {}
    for element in d4.ELEMENTS:
        matrix = np.zeros((4, 4))
        for state in range(4):
            matrix[state, d4.apply(element, state)] = 1.0
        operators[element] = matrix
    r, s = operators[d4.R], operators[d4.S]
    np.testing.assert_allclose(s @ r @ s, operators[d4.inverse(d4.R)])
    assert not np.allclose(r @ s @ s, operators[d4.inverse(d4.R)])
    op_components = d4_relation_components_op(operators)
    assert set(op_components) == {"r4", "s2", "srs_r_inverse"}
    assert max(op_components.values()) < 1e-12
    z = np.eye(4)
    data_components = d4_relation_components_data(z, np.ones(4, dtype=bool), operators)
    assert max(data_components.values()) < 1e-12


def test_stale_cache_metadata_is_rejected():
    metadata = {
        "model_id": "m", "model_revision": "abc", "task": "C7", "layer": 4,
        "prompt_template_split": {"test_ids": [1]}, "tokenizer_settings": {"use_fast": True},
        "prompt_construction_version": "v1", "activation_extraction_convention": "target mean",
    }
    with tempfile.TemporaryDirectory() as tmp:
        cache = ActivationCache(tmp)
        key = activation_cache_key(**metadata)
        cache.store(key, np.ones((2, 3)), metadata)
        stale = {**metadata, "prompt_construction_version": "v2"}
        try:
            cache.load(key, expected_metadata=stale)
        except CacheMetadataMismatch:
            pass
        else:
            raise AssertionError("stale cache metadata was accepted")


def test_checkpoint_resume_skips_verified_and_recomputes_corrupt_output():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ledger = RunLedger(root / "run_ledger.json", "run", "cfg")
        fields = {"model": "m", "group": "C7", "layer": 6}
        key = experiment_key("structural_base", fields)
        output = root / "unit.csv"
        output.write_text("value\n1\n", encoding="utf-8")
        ledger.begin(key, fields)
        ledger.complete(key, fields, [output])
        assert ledger.should_skip(key)
        output.write_text("corrupt\n", encoding="utf-8")
        assert not ledger.should_skip(key)
        state = json.loads((root / "run_ledger.json").read_text(encoding="utf-8"))
        assert state["units"][key]["status"] == "STALE"


def test_clean_tree_full_guard_and_stable_artifact_names():
    validate_git_policy("full", dirty=False)
    try:
        validate_git_policy("full", dirty=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("dirty FULL tree was not rejected")
    assert STABLE_TABLES == (
        "structural_main.csv", "c7_causal_main.csv", "d4_causal_main.csv",
        "cross_model_main.csv", "latent_dim_sensitivity.csv", "model_scale_summary.csv",
        "joint_vs_ridge_inference.csv", "causal_additive_baseline_summary.csv",
        "d4_order_control_summary.csv",
    )
    assert len(STABLE_FIGURES) == 9


def test_final_artifact_schemas_are_generated_from_saved_tables():
    structural = pd.DataFrame([{
        "data_split": "test", "latent_dim": 16, "depth_slot": 0.4, "model": "m",
        "group": "C7", "method": "group_consistent_joint", "layer": 6,
        "E_eq": 0.1, "E_rel_data": 0.2,
    }])
    causal = pd.DataFrame([
        {"group": "C7", "model": "m", "method": "learned_operator", "evaluation_role": "transformation", "intervention": "R^1", "depth_slot": 0.4, "layer": 6, "target_logit_change": 0.2, "relation_final_logit_distance": np.nan},
        {"group": "D4", "model": "m", "method": "learned_operator", "evaluation_role": "relation_consistency", "intervention": "srs_r_inverse", "depth_slot": 0.4, "layer": 6, "target_logit_change": 0.1, "relation_final_logit_distance": 0.05},
    ])
    cross = pd.DataFrame([
        {"model_A": "a", "model_B": "b", "group": "C7", "depth_slot": 0.4, "layer_A": 6, "layer_B": 11, "split_seed": 101, "optimization_seed": 42, "latent_dim": 16, "metric": metric, "value": value}
        for metric, value in (
            ("linear_cka", 0.8), ("fitted_intertwiner_commutation_error", 0.1),
            ("procrustes_map_commutation_error", 0.2),
        )
    ])
    irrep = pd.DataFrame([{"generator_root_angle_rmse": 0.1}])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "results" / "tables").mkdir(parents=True)
        (root / "results" / "figures").mkdir()
        _generate_stable_artifacts(root, structural, irrep, cross, causal, pd.DataFrame())
        validate_stable_artifacts(root / "results")


def test_primary_bootstrap_excludes_terminal_layers_and_sensitivity_dimensions():
    rows = []
    for depth_slot, latent_dim, learned, random in (
        (0.2, 16, 1.0, 2.0),
        ("terminal", 16, 1000.0, 0.0),
        (0.2, 64, 1000.0, 0.0),
    ):
        for method, value in (("group_consistent_joint", learned), ("random_orthogonal", random)):
            rows.append({
                "data_split": "test", "model": "m", "group": "C7", "method": method,
                "depth_slot": depth_slot, "layer": 3, "latent_dim": latent_dim,
                "split_seed": 101, "optimization_seed": 42, "random_baseline_seed": 104729,
                "template_id": 1, "E_eq": value, "E_rel_data": value,
                "E_hom_data": value, "E_comp_data": value, "E_inverse_data": value,
                "E_word": value,
            })
    config = {"primary_latent_dimension": 16, "bootstrap_samples": 20, "bootstrap_seed": 7}
    result = _bootstrap_rows(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame(), config)
    contrast = result[(result.analysis == "learned_minus_random_orthogonal") & (result.metric == "E_eq")]
    assert set(contrast["mean"]) == {-1.0}


def test_finite_smoke_telemetry_produces_finite_preliminary_full_projection():
    config = resolve_main_config(CONFIG, "smoke")
    targets = _full_workload_unit_targets(config)
    assert targets == {
        "model_load": 6,
        "activation_extraction": 72,
        "structural_base": 444,
        "structural_joint": 444,
        "causal": 240,
        "cross_model": 360,
    }
    profile = pd.DataFrame({
        "experiment_module": [
            "structural_joint", "structural_joint", "causal", "causal", "cross_model", "cross_model",
        ],
        "wall_clock_seconds": [
            1.320338879013434, 1.3178639730904251,
            3.6072879799176008, 1.967289285035804,
            1.3861898710019886, 1.1872702590189874,
        ],
    })
    projection = _estimate_full_cost(profile, targets)
    assert np.isfinite(projection["projected_seconds"])
    assert abs(projection["projected_seconds"] - 1717.853128365241) < 1e-9
    assert projection["coverage"] == "PARTIAL"
    rendered = _render_cost_projection(projection, "smoke")
    assert "1717.85 seconds (0.48 hours)" in rendered
    assert "SMOKE-derived FULL estimates are preliminary" in rendered
    assert "nan" not in rendered.lower()
    adjusted = _estimate_full_cost(profile, targets, _full_calibration_multipliers(config))
    assert abs(adjusted["projected_seconds"] - 3304.6197258634493) < 1e-9
    assert all(np.isfinite(row["calibration_multiplier"]) for row in adjusted["modules"] if row["status"] == "MEASURED")


def test_nonfinite_runtime_timing_is_rejected_before_projection():
    profile = pd.DataFrame({
        "experiment_module": ["structural_joint"],
        "wall_clock_seconds": [float("inf")],
    })
    try:
        _estimate_full_cost(profile, {"structural_joint": 6000})
    except ValueError as error:
        assert "invalid timing" in str(error)
    else:
        raise AssertionError("non-finite runtime timing was accepted")


def test_completed_smoke_scientific_ledger_enters_postprocessing_only_resume():
    core = resolve_main_config(CONFIG, "core")
    assert _scientific_unit_targets(core) == {
        "model_load": 4,
        "activation_extraction": 8,
        "causal_activation_extraction": 24,
        "structural_base": 48,
        "structural_joint": 144,
        "causal": 144,
        "cross_model": 144,
    }
    config = resolve_main_config(CONFIG, "smoke")
    expected = _scientific_unit_targets(config)
    assert expected == {
        "model_load": 1,
        "activation_extraction": 2,
        "causal_activation_extraction": 2,
        "structural_base": 2,
        "structural_joint": 2,
        "causal": 2,
        "cross_model": 2,
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ledger = RunLedger(root / "run_ledger.json", "smoke", "config")
        markers = []
        for module, count in expected.items():
            for index in range(count):
                fields = {"index": index}
                key = f"{module}:{index}"
                marker = root / "checkpoints" / module / f"{index}.txt"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("complete\n", encoding="utf-8")
                ledger.begin(key, fields)
                ledger.complete(key, fields, [marker])
                markers.append(marker)
        aggregate_fixtures = {
            root / "structural" / "template_metrics.csv": "layer,E_eq\n1,0.1\n",
            root / "irrep" / "representation_diagnostics.csv": "layer,method\n1,joint\n",
            root / "intertwiner" / "cross_model_metrics.csv": "metric,value\ncka,1.0\n",
            root / "causal" / "intervention_metrics.csv": "layer,method\n1,learned\n",
        }
        for path, contents in aggregate_fixtures.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        assert _can_resume_postprocessing(root, ledger, config)
        markers[-1].write_text("corrupt\n", encoding="utf-8")
        assert not _can_resume_postprocessing(root, ledger, config)


def test_result_packaging_telemetry_preserves_manifest_checksum_integrity():
    required = (
        "environment.json", "config_snapshot.yaml", "run_ledger.json",
        "FULL_COST_PROJECTION.md", "go_no_go.json",
        "structural/template_metrics.csv", "irrep/representation_diagnostics.csv",
        "intertwiner/cross_model_metrics.csv", "causal/intervention_metrics.csv",
        "bootstrap/confidence_intervals.csv", "results/tables/structural_main.csv",
        "results/tables/c7_causal_main.csv", "results/tables/d4_causal_main.csv",
        "results/tables/cross_model_main.csv", "results/tables/latent_dim_sensitivity.csv",
        "results/tables/model_scale_summary.csv",
        "results/tables/joint_vs_ridge_inference.csv",
        "results/tables/causal_additive_baseline_summary.csv",
        "results/tables/d4_order_control_summary.csv",
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = root / "run"
        run.mkdir()
        (run / "manifest.json").write_text(
            json.dumps({"status": "SUCCESS", "artifacts": {"runtime_profile.csv": "stale"}}),
            encoding="utf-8",
        )
        (run / "runtime_profile.csv").write_text(
            "experiment_key,experiment_module,wall_clock_seconds,timestamp_utc\n"
            "cross:1,cross_model,1.25,2026-01-01T00:00:00+00:00\n",
            encoding="utf-8",
        )
        for relative in required:
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder\n", encoding="utf-8")
        bundle = root / "results.zip"
        environment = {**os.environ, "RUNPOD_RESULT_BUNDLE_PATH": str(bundle)}
        subprocess.run(
            ["bash", str(ROOT / "runpod" / "package_results.sh"), str(run)],
            check=True,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        prefix = "Beyond_Shared_Geometry_main_experiment_results/"
        with zipfile.ZipFile(bundle) as archive:
            runtime_data = archive.read(prefix + "runtime_profile.csv")
            manifest = json.loads(archive.read(prefix + "manifest.json"))
        rows = list(csv.DictReader(runtime_data.decode("utf-8").splitlines()))
        packaging_events = [
            row["event"] for row in rows if row["experiment_module"] == "result_packaging"
        ]
        assert packaging_events == ["STARTED", "COMPLETE"]
        assert manifest["artifacts"]["runtime_profile.csv"] == hashlib.sha256(runtime_data).hexdigest()
