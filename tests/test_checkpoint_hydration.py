from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

import bsg.experiments.main_experiment as experiment
from bsg.experiments.canonical_units import load_artifact, save_artifact
from bsg.experiments.runtime_state import (
    RunLedger,
    RuntimeTelemetry,
    atomic_write_frame,
    atomic_write_json,
    experiment_key,
)
from bsg.representations.cache import ActivationCache, activation_cache_key
from bsg.representations.pca import PCASubspace


ROOT = Path(__file__).resolve().parents[1]


def _model(key: str, model_id: str) -> dict:
    return {
        "key": key,
        "id": model_id,
        "revision": f"{key}-immutable-revision",
        "blocks": 2,
        "layers": [1],
        "primary_layers": [1],
        "causal_layers": [1],
        "depth_slots": [0.5],
        "depth_layers": {0.5: 1},
        "depth_slot_by_layer": {"1": 0.5},
    }


def _config() -> dict:
    return {
        "tier": "full",
        "selected_models": [_model("model_a", "fixture/a"), _model("model_b", "fixture/b")],
        "selected_cross_model_pairs": [["model_a", "model_b"]],
        "groups": ["C7"],
        "depth_slots": [0.5],
        "split_seeds": [101],
        "template_limit": 7,
        "template_split": {"train_templates": 5, "validation_templates": 1, "test_templates": 1},
        "structural_latent_dimensions": [2],
        "primary_structural_latent_dimensions": [2],
        "sensitivity_latent_dimensions": [],
        "sensitivity_depth_slots": [],
        "sensitivity_split_seeds": [],
        "cross_model_latent_dimensions": [2],
        "primary_latent_dimension": 2,
        "learned_fit_semantics": "deterministic_single",
        "learned_fit_seeds": [42],
        "random_baseline_seeds": [104729],
        "optimization_seeds": [42],
        "random_seed_by_optimization_seed": {},
        "ridge_alpha": 0.01,
        "tokenizer": {"use_fast": True, "padding_side": "right"},
        "prompt_construction_version": "fixture-v1",
        "activation_extraction_convention": "fixture",
    }


def _artifact(spec, model_offset: float) -> dict:
    row_count = len(spec.frame)
    z = np.column_stack((np.arange(row_count), np.arange(row_count) % 7)).astype(float)
    z += model_offset
    pca = PCASubspace(2, seed=101)
    pca.mean_ = np.array([model_offset, model_offset + 1.0, model_offset + 2.0])
    pca.components_ = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pca.explained_variance_ratio_ = np.array([0.6, 0.3])
    pca._fit_rows = 35
    operators = {element: np.eye(2) for element in spec.elements}
    return {"pca": pca, "z": z, "operators": operators}


def _complete(ledger, key, fields, paths):
    ledger.begin(key, fields)
    ledger.complete(key, fields, paths)


def _write_structural_unit(root, ledger, config, spec, model_meta, artifact, value):
    split = experiment._split(spec, config, 101)
    base_fields = experiment._common_fields(model_meta, "C7", 1, 2, split.seed)
    base_key = experiment_key("structural_base", base_fields)
    base_paths = experiment._checkpoint_paths(root, "structural_base", base_key, ("csv", "npz", "irrep.csv"))
    atomic_write_frame(pd.DataFrame([{"layer": 1, "E_eq": value, "method": "independent_ridge"}]), base_paths[0])
    save_artifact(base_paths[1], artifact, spec.elements)
    atomic_write_frame(pd.DataFrame([{"layer": 1, "method": "independent_ridge"}]), base_paths[2])
    _complete(ledger, base_key, base_fields, base_paths)

    fields = experiment._common_fields(model_meta, "C7", 1, 2, split.seed, 42, None)
    checkpoint_fields = experiment._full_checkpoint_fields(fields, config, config["random_baseline_seeds"])
    joint_key = experiment_key("structural_joint", checkpoint_fields)
    joint_paths = experiment._checkpoint_paths(root, "structural_joint", joint_key, ("csv", "npz", "irrep.csv"))
    atomic_write_frame(pd.DataFrame([{"layer": 1, "E_eq": value + 0.01, "method": "group_consistent_joint"}]), joint_paths[0])
    save_artifact(joint_paths[1], artifact, spec.elements)
    atomic_write_frame(pd.DataFrame([{"layer": 1, "method": "group_consistent_joint"}]), joint_paths[2])
    _complete(ledger, joint_key, checkpoint_fields, joint_paths)
    return joint_paths


def _write_nonstructural_prerequisites(root, ledger, config):
    for model_meta in config["selected_models"]:
        load_fields = {"model": model_meta["id"], "model_revision": model_meta["revision"]}
        load_key = experiment_key("model_load", load_fields)
        [load_marker] = experiment._checkpoint_paths(root, "model_load", load_key, ("json",))
        payload = {**load_fields, "resolved_revision": model_meta["revision"]}
        if model_meta["key"] == "model_a":
            payload["tokenizer_identity"] = {
                "class": "FixtureTokenizer", "name_or_path": model_meta["id"],
                "revision": model_meta["revision"], "padding_side": "right",
                "pad_token_id": 0, "eos_token_id": 1,
            }
        atomic_write_json(load_marker, payload)
        _complete(ledger, load_key, load_fields, [load_marker])

        extraction_fields = {"model": model_meta["id"], "group": "C7"}
        extraction_key = experiment_key("activation_extraction", extraction_fields)
        [marker] = experiment._checkpoint_paths(root, "activation_extraction", extraction_key, ("json",))
        atomic_write_json(marker, extraction_fields)
        _complete(ledger, extraction_key, extraction_fields, [marker])

        causal_extract_fields = {"model": model_meta["id"], "group": "C7", "split_seed": 101}
        causal_extract_key = experiment_key("causal_activation_extraction", causal_extract_fields)
        [marker] = experiment._checkpoint_paths(root, "causal_activation_extraction", causal_extract_key, ("json",))
        atomic_write_json(marker, causal_extract_fields)
        _complete(ledger, causal_extract_key, causal_extract_fields, [marker])

        causal_fields = experiment._common_fields(model_meta, "C7", 1, 2, 101, 42, None)
        causal_checkpoint_fields = experiment._full_causal_checkpoint_fields(
            causal_fields,
            experiment._configured_specs(config)["C7"],
            {**config, "causal": {"c7_steps": list(range(1, 8)), "baselines": [], "d4_wrong_order_controls": []}},
            config["random_baseline_seeds"],
        )
        causal_key = experiment_key("causal", causal_checkpoint_fields)
        [causal_path] = experiment._checkpoint_paths(root, "causal", causal_key)
        atomic_write_frame(pd.DataFrame([{"layer": 1, "method": "learned_operator"}]), causal_path)
        _complete(ledger, causal_key, causal_checkpoint_fields, [causal_path])


def _write_activation_caches(cache, config, specs):
    for model_index, model_meta in enumerate(config["selected_models"]):
        frame = specs["C7"].frame
        identity = {
            "kind": "template_pool",
            "template_ids": sorted(map(int, frame.template_id.unique())),
            "split_seeds": config["split_seeds"],
            "split_counts": config["template_split"],
        }
        metadata = experiment._cache_metadata(
            config, model_meta, model_meta["revision"], "structural_C7", 1, frame, identity,
        )
        key = activation_cache_key(**metadata)
        values = np.arange(len(frame) * 4, dtype=np.float32).reshape(len(frame), 4)
        cache.store(key, values + model_index, metadata)


def _prepare_fixture(root: Path):
    config = _config()
    specs = experiment._configured_specs(config)
    ledger = RunLedger(root / "run_ledger.json", "fixture_full", "fixture_config")
    originals = {}
    for index, model_meta in enumerate(config["selected_models"]):
        artifact = _artifact(specs["C7"], float(index))
        paths = _write_structural_unit(root, ledger, config, specs["C7"], model_meta, artifact, 0.1 + index)
        originals[model_meta["key"]] = (artifact, paths)
    _write_nonstructural_prerequisites(root, ledger, config)
    cache = ActivationCache(root / "cache")
    _write_activation_caches(cache, config, specs)
    return config, specs, ledger, originals


def exercise_restored_cross(root: Path) -> None:
    config = _config()
    specs = experiment._configured_specs(config)
    ledger = RunLedger(root / "run_ledger.json", "fixture_full", "fixture_config")
    cache = ActivationCache(root / "cache")
    assert experiment._can_resume_cross_model(ledger, config)
    artifact_index, structural_paths, irrep_paths, required = experiment._hydrate_structural_checkpoints(
        config, specs, root, ledger,
    )
    expected_key = ("model_a", "C7", 0.5, 101, 2, 42)
    assert expected_key in artifact_index
    restored = load_artifact(artifact_index[expected_key], specs["C7"].elements, seed=101)
    assert restored["pca"].dim == 2
    hidden_index = experiment._hydrate_hidden_index(config, specs, cache)

    fields = {
        "model_A": "fixture/a", "model_B": "fixture/b",
        "model_key_A": "model_a", "model_key_B": "model_b",
        "comparison_type": "fixed_pair", "group": "C7", "depth_slot": 0.5,
        "layer_A": 1, "layer_B": 1, "split_seed": 101,
        "optimization_seed": 42, "random_baseline_seed": None, "latent_dim": 2,
    }
    checkpoint_fields = experiment._full_checkpoint_fields(
        fields, config, config["random_baseline_seeds"],
    )
    cross_key = experiment_key("cross_model", checkpoint_fields)
    ledger.begin(cross_key, checkpoint_fields)
    ledger.fail(cross_key, checkpoint_fields, RuntimeError("simulated interruption"))

    def forbidden(*args, **kwargs):
        raise AssertionError("structural scientific fitting was invoked during hydration")

    experiment.fit_base_artifact = forbidden
    experiment.fit_joint_artifact = forbidden
    experiment.load_tokenizer_and_model = forbidden
    experiment._cached_layers = forbidden
    experiment._run_causal = forbidden
    telemetry = RuntimeTelemetry(root / "runtime_profile.csv")
    cross_paths = experiment._run_cross_units(
        config, specs, artifact_index, hidden_index, cache, root, ledger, telemetry, required,
    )
    assert len(cross_paths) == 1
    cross = pd.read_csv(cross_paths[0])
    assert {"metric", "value"} <= set(cross.columns) and not cross.empty
    assert ledger.data["units"][cross_key]["status"] == "COMPLETE"
    assert ledger.data["units"][cross_key]["attempts"] == 2
    completed_bytes = cross_paths[0].read_bytes()
    reused = experiment._run_cross_units(
        config, specs, artifact_index, hidden_index, cache, root, ledger, telemetry, required,
    )
    assert reused == cross_paths
    assert cross_paths[0].read_bytes() == completed_bytes
    assert ledger.data["units"][cross_key]["attempts"] == 2

    ledger.begin(cross_key, checkpoint_fields)
    assert ledger.data["units"][cross_key]["status"] == "RUNNING"
    retried = experiment._run_cross_units(
        config, specs, artifact_index, hidden_index, cache, root, ledger, telemetry, required,
    )
    assert retried == cross_paths
    assert ledger.data["units"][cross_key]["status"] == "COMPLETE"
    assert ledger.data["units"][cross_key]["attempts"] == 4
    frames = experiment._aggregate_checkpoint_frames(
        root, ledger, structural_paths, irrep_paths, cross_paths,
    )
    assert not frames["structural"].empty
    assert {"layer", "E_eq"} <= set(frames["structural"].columns)
    np.testing.assert_array_equal(
        np.sort(frames["structural"]["E_eq"].to_numpy()),
        np.array([0.1, 0.11, 1.1, 1.11]),
    )
    atomic_write_json(root / "subprocess_result.json", {
        "cross_key": cross_key,
        "lookup_key": list(expected_key),
        "cross_rows": len(cross),
        "structural_rows": len(frames["structural"]),
    })


def test_fresh_and_restored_checkpoint_state_are_exact_and_aggregate_schemas_survive():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config, specs, ledger, originals = _prepare_fixture(root)
        artifact_index, structural_paths, irrep_paths, _ = experiment._hydrate_structural_checkpoints(
            config, specs, root, ledger,
        )
        for model_meta in config["selected_models"]:
            original, paths = originals[model_meta["key"]]
            key = (model_meta["key"], "C7", 0.5, 101, 2, 42)
            restored = load_artifact(artifact_index[key], specs["C7"].elements, seed=101)
            np.testing.assert_array_equal(restored["pca"].mean_, original["pca"].mean_)
            np.testing.assert_array_equal(restored["pca"].components_, original["pca"].components_)
            np.testing.assert_array_equal(restored["z"], original["z"])
            for element in specs["C7"].elements:
                np.testing.assert_array_equal(restored["operators"][element], original["operators"][element])
            serialized_metrics = pd.read_csv(paths[0])
            assert serialized_metrics.iloc[0]["method"] == "group_consistent_joint"
        assert len(structural_paths) == 4
        assert len(irrep_paths) == 4


def test_fresh_structural_computation_populates_the_same_cross_model_lookup():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = _config()
        specs = experiment._configured_specs(config)
        spec = specs["C7"]
        split = experiment._split(spec, config, 101)
        ledger = RunLedger(root / "run_ledger.json", "fresh_fixture", "fixture_config")
        telemetry = RuntimeTelemetry(root / "runtime_profile.csv")
        artifact_index = {}
        cache = ActivationCache(root / "cache")
        _write_activation_caches(cache, config, specs)
        hidden_index = experiment._hydrate_hidden_index(config, specs, cache)
        original_joint = experiment.fit_joint_artifact

        def deterministic_joint(_spec, base, _split, _seed, _config):
            return {
                "pca": base["pca"], "z": base["z"],
                "operators": {element: matrix.copy() for element, matrix in base["operators"].items()},
            }, {name: 0.0 for name in ("L_data", "L_hom", "L_rel", "L_identity", "L_orth", "L_total")}

        experiment.fit_joint_artifact = deterministic_joint
        try:
            for model_index, model_meta in enumerate(config["selected_models"]):
                x = np.arange(len(spec.frame), dtype=float)
                hidden = np.column_stack((x, x ** 2, np.sin(x), np.cos(x)))
                hidden += model_index
                base, _, _, _ = experiment._load_or_run_base(
                    spec, hidden, split, model_meta, 1, 2, config,
                    root, ledger, telemetry, set(),
                )
                learned, _, _, artifact_path = experiment._load_or_run_joint(
                    spec, base, split, model_meta, 1, 2, 42,
                    config["random_baseline_seeds"], config,
                    root, ledger, telemetry, set(),
                )
                assert learned is not None, ledger.failed_units()
                artifact_index[(model_meta["key"], "C7", 0.5, 101, 2, 42)] = artifact_path
        finally:
            experiment.fit_joint_artifact = original_joint

        paths = experiment._run_cross_units(
            config, specs, artifact_index, hidden_index, cache,
            root, ledger, telemetry, set(),
        )
        assert len(paths) == 1
        frame = pd.read_csv(paths[0])
        assert not frame.empty and {"metric", "value"} <= set(frame.columns)


def test_new_process_hydrates_complete_structural_state_and_retries_cross_model():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _prepare_fixture(root)
        code = (
            "import importlib.util, pathlib, sys; "
            "p=pathlib.Path(sys.argv[1]); "
            f"s=importlib.util.spec_from_file_location('hydration_fixture', {str(Path(__file__))!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "m.exercise_restored_cross(p)"
        )
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        subprocess.run([sys.executable, "-c", code, str(root)], check=True, cwd=ROOT, env=env)
        result = json.loads((root / "subprocess_result.json").read_text(encoding="utf-8"))
        assert result["lookup_key"] == ["model_a", "C7", 0.5, 101, 2, 42]
        assert result["cross_rows"] > 0
        assert result["structural_rows"] == 4


def test_manifest_provenance_hydrates_exact_and_legacy_model_load_markers():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config, _, ledger, _ = _prepare_fixture(root)
        manifest = {"models": [{"resolved_revision": None, "tokenizer_identity": None} for _ in range(2)]}
        experiment._hydrate_manifest_provenance(config, root, ledger, manifest)
        assert manifest["models"][0]["tokenizer_identity"]["class"] == "FixtureTokenizer"
        assert manifest["models"][0]["resolved_revision"] == config["selected_models"][0]["revision"]
        assert manifest["models"][1]["tokenizer_identity"]["recording_status"] == "PARTIAL_LEGACY_CHECKPOINT"
        assert manifest["models"][1]["tokenizer_identity"]["revision"] == config["selected_models"][1]["revision"]


def test_empty_promoted_structural_csv_never_enters_postprocessing_only_resume():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config, _, ledger, _ = _prepare_fixture(root)
        for path in experiment._aggregate_output_paths(root).values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n", encoding="utf-8")
        assert not experiment._can_resume_postprocessing(root, ledger, config)
