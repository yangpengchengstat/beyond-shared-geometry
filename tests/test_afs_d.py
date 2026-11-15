from __future__ import annotations

import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bsg.data.c7 import build_dataset as build_c7, transform_state as c7_transform
from bsg.data.d4 import build_dataset as build_d4, transform_state as d4_transform
from bsg.data.splits import split_templates_three_way
from bsg.experiments.afs_d import (
    _embedding_metadata,
    _internal_metadata,
    _models_requiring_extraction,
    assert_rank_diagnostics_pass,
    embedding_rank_diagnostic_rows,
    ensure_isolated_output,
    group_latent_dim,
    matrix,
    resolve_config,
)
from bsg.experiments.afs_d_outputs import (
    bootstrap_depth_contrasts,
    classify_outcomes,
    pair_depth_contrasts,
    prepare_location_metrics,
)
from bsg.experiments.algebra_falsification_units import d4_canonical_words, fit_generators
from bsg.experiments.canonical_units import fit_base_artifact
from bsg.experiments.fitting import GroupSpec
from bsg.experiments.runtime_state import RunLedger, experiment_key
from bsg.groups import c7, d4
from bsg.metrics.relations import c7_relations, d4_relations


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "afs_d.yaml"
MODELS = [
    ("llama_1b", "meta-llama/Llama-3.2-1B", "Llama"),
    ("qwen_1_5b", "Qwen/Qwen2.5-1.5B", "Qwen"),
]
DEPTHS = [0.2, 0.4, 0.6, 0.8]


def _location_fields(model_key, model, family, group, location, depth, split):
    return {
        "model_key": model_key,
        "model": model,
        "model_family": family,
        "group": group,
        "representation_location": location,
        "depth_slot": depth,
        "layer": 0 if location == "embedding" else int(round(depth * 10)),
        "realized_depth": depth,
        "latent_dim": 6 if group == "C7" else 3,
        "split_seed": split,
        "source_run": "synthetic",
    }


def _raw_metric_fixture():
    operator_rows, generator_rows = [], []
    for model_key, model, family in MODELS:
        for group in ("C7", "D4"):
            for location, depth in [("embedding", 0.0), *[("internal", value) for value in DEPTHS]]:
                for split in (101, 303):
                    fields = _location_fields(model_key, model, family, group, location, depth, split)
                    for template_id in (1, 2):
                        base = depth + template_id / 100 + split / 100000
                        operator_rows.append({
                            **fields,
                            "data_split": "test",
                            "method": "independent_ridge",
                            "template_id": template_id,
                            "E_eq": base + 0.01,
                            "E_hom_data": base + 0.02,
                            "E_hom_op": depth + split / 100000,
                            "E_comp_data": base + 0.03,
                            "E_rel_data": base + 0.04,
                            "E_rel_r4_data": base + 0.05 if group == "D4" else np.nan,
                            "E_rel_s2_data": base + 0.06 if group == "D4" else np.nan,
                            "E_rel_srs_r_inverse_data": base + 0.07 if group == "D4" else np.nan,
                        })
                        for target in range(5):
                            generator_rows.append({
                                **fields,
                                "data_split": "test",
                                "template_id": template_id,
                                "target_element": str(target),
                                "evaluation_role": "unseen_composite",
                                "method": "generator_composed",
                                "E_gen_eq": base + target / 1000,
                            })
    return pd.DataFrame(operator_rows), pd.DataFrame(generator_rows)


def test_afs_d_config_and_matrix_are_frozen():
    config = resolve_config(CONFIG)
    assert [model["id"] for model in config["selected_models"]] == [
        "meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-1.5B",
    ]
    assert config["groups"] == ["C7", "D4"]
    assert config["split_seeds"] == [101, 303, 909, 2701, 4243]
    assert config["experiment"] == "beyond_shared_geometry_afs_d_rank_aware"
    assert config["group_latent_dimensions"] == {"C7": 6, "D4": 3}
    assert config["expected_embedding_centered_ranks"] == {"C7": 6, "D4": 3}
    assert group_latent_dim(config, "C7") == 6
    assert group_latent_dim(config, "D4") == 3
    assert config["embedding_convention"].startswith("Hugging Face hidden_states[0]")
    assert matrix(config) == {
        "models": 2,
        "groups": 2,
        "split_seeds": 5,
        "representation_locations": 5,
        "embedding_extraction_sets": 4,
        "embedding_rank_diagnostic_cells": 20,
        "embedding_analysis_cells": 20,
        "rank_matched_internal_cells": 80,
        "paired_cells": 100,
    }


def test_afs_d_generator_and_operator_units_are_aggregated_without_pseudoreplication():
    operator, generator = _raw_metric_fixture()
    metrics = prepare_location_metrics(operator, generator)
    cells = 2 * 2 * 5 * 2
    assert len(metrics[metrics.metric == "E_hom_op"]) == cells
    assert len(metrics[metrics.metric == "E_gen_eq"]) == cells * 2
    assert set(metrics[metrics.metric == "E_gen_eq"].unit_type) == {"template"}
    assert set(metrics[metrics.metric == "E_hom_op"].unit_type) == {"split"}
    one = metrics[
        (metrics.model_key == "llama_1b") & (metrics.group == "C7")
        & (metrics.representation_location == "embedding") & (metrics.split_seed == 101)
        & (metrics.template_id == 1) & (metrics.metric == "E_gen_eq")
    ]
    expected = np.mean([0.0 + 0.01 + 101 / 100000 + target / 1000 for target in range(5)])
    assert len(one) == 1 and np.isclose(one.iloc[0].value, expected)


def test_afs_d_pairing_keeps_depth_fixed_and_bootstrap_is_deterministic():
    operator, generator = _raw_metric_fixture()
    metrics = prepare_location_metrics(operator, generator)
    contrasts = pair_depth_contrasts(metrics, DEPTHS)
    per_unit = contrasts.groupby(
        ["model_key", "group", "metric", "split_seed", "template_id"], dropna=False
    ).depth_slot.nunique()
    assert (per_unit == 4).all()
    first = bootstrap_depth_contrasts(contrasts, n_boot=80, seed=314159)
    second = bootstrap_depth_contrasts(contrasts, n_boot=80, seed=314159)
    pd.testing.assert_frame_equal(first, second)
    fixed = first[first.scope == "fixed_depth_average"]
    assert set(fixed.cluster_columns) == {"split_seed,template_id", "split_seed"}
    assert set(fixed.n_depths) == {4}
    assert all("depth_slot" not in value for value in fixed.cluster_columns)


def test_afs_d_dimension_mismatched_pairing_is_rejected():
    operator, generator = _raw_metric_fixture()
    operator.loc[
        (operator["representation_location"] == "internal") & (operator["group"] == "C7"),
        "latent_dim",
    ] = 16
    try:
        pair_depth_contrasts(prepare_location_metrics(operator, generator), DEPTHS)
    except ValueError as error:
        assert "dimension-mismatched" in str(error)
    else:
        raise AssertionError("AFS-D accepted dimension-mismatched embedding/internal pairing")


def _interpretation_summary(group_patterns):
    rows = []
    metrics = ("E_gen_eq", "E_rel_data", "E_hom_data", "E_comp_data")
    for group, outcome in group_patterns.items():
        for _, model, _ in MODELS:
            for metric in metrics:
                if outcome == "A":
                    mean, low, high = -1.0, -1.2, -0.8
                elif outcome == "C":
                    mean, low, high = 1.0, 0.8, 1.2
                elif outcome == "B":
                    mean, low, high = 0.0, -0.2, 0.2
                else:
                    mean, low, high = (-1.0, -1.2, -0.8) if model.startswith("meta-") else (1.0, 0.8, 1.2)
                rows.append({
                    "model": model,
                    "group": group,
                    "metric": metric,
                    "scope": "fixed_depth_average",
                    "mean_difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                })
    return pd.DataFrame(rows)


def test_afs_d_outcome_rules_cover_a_b_c_d_and_mixed():
    assert classify_outcomes(_interpretation_summary({"C7": "A", "D4": "A"}))["group_outcomes"] == {"C7": "A", "D4": "A"}
    assert classify_outcomes(_interpretation_summary({"C7": "B", "D4": "B"}))["group_outcomes"] == {"C7": "B", "D4": "B"}
    assert classify_outcomes(_interpretation_summary({"C7": "C", "D4": "C"}))["group_outcomes"] == {"C7": "C", "D4": "C"}
    d = classify_outcomes(_interpretation_summary({"C7": "B", "D4": "A"}))
    assert d["overall_interpretation"] == "D"
    mixed = classify_outcomes(_interpretation_summary({"C7": "MIXED", "D4": "A"}))
    assert mixed["group_outcomes"]["C7"] == "MIXED"
    assert mixed["gate"] == "NONE_DIAGNOSTIC_CONTROL"


def test_afs_d_output_guard_and_resume_require_no_model_when_extractions_are_complete():
    config = resolve_config(CONFIG)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        full = root / "full"; afs = root / "afs"; output = root / "outputs" / "run"
        full.mkdir(); afs.mkdir(); output.mkdir(parents=True)
        sources = SimpleNamespace(full_root=full.resolve(), afs_root=afs.resolve())
        ensure_isolated_output(output, sources)
        try:
            ensure_isolated_output(full / "new_run", sources)
        except ValueError:
            pass
        else:
            raise AssertionError("AFS-D accepted an output path inside immutable FULL")

        ledger = RunLedger(output / "run_ledger.json", "test", "hash")
        for model in config["selected_models"]:
            for group in config["groups"]:
                fields = {**_embedding_metadata(model, group), "revision": model["revision"]}
                key = experiment_key("embedding_extraction", fields)
                checkpoint = output / f"{model['key']}_{group}.npz"
                checkpoint.write_bytes(b"verified")
                ledger.begin(key, fields)
                ledger.complete(key, fields, [checkpoint])
        assert _models_requiring_extraction(config, output, ledger) == set()


def test_afs_d_rank_diagnostics_select_frozen_dimensions_without_metrics():
    config = resolve_config(CONFIG)
    c7_spec = GroupSpec("C7", build_c7(), c7.ELEMENTS, c7.IDENTITY, c7.compose, c7.inverse, c7_transform, c7_relations())
    d4_spec = GroupSpec("D4", build_d4(), d4.ELEMENTS, d4.IDENTITY, d4.compose, d4.inverse, d4_transform, d4_relations())
    model = config["selected_models"][0]
    for spec, dim, states in ((c7_spec, 6, 7), (d4_spec, 3, 4)):
        split = split_templates_three_way(spec.frame, seed=101, train_templates=28, validation_templates=6, test_templates=6)
        basis = np.eye(states, dtype=np.float32)
        hidden = basis[spec.frame.state_id.to_numpy()]
        rows = embedding_rank_diagnostic_rows(spec, hidden, split, model, config)
        assert rows[0]["frozen_group_latent_dim"] == dim
        assert rows[0]["centered_training_rank"] == dim
        assert rows[0]["rank_status"] == "PASS"
        assert_rank_diagnostics_pass(rows)


def test_afs_d_insufficient_rank_aborts_before_analysis():
    config = resolve_config(CONFIG)
    spec = GroupSpec("C7", build_c7(), c7.ELEMENTS, c7.IDENTITY, c7.compose, c7.inverse, c7_transform, c7_relations())
    split = split_templates_three_way(spec.frame, seed=101, train_templates=28, validation_templates=6, test_templates=6)
    hidden = np.ones((len(spec.frame), 4), dtype=np.float32)
    rows = embedding_rank_diagnostic_rows(spec, hidden, split, config["selected_models"][0], config)
    try:
        assert_rank_diagnostics_pass(rows)
    except ValueError as error:
        assert "before metric interpretation" in str(error)
    else:
        raise AssertionError("AFS-D accepted insufficient embedding rank")


def test_afs_d_observed_pca_rank_guard_remains_active():
    config = resolve_config(CONFIG)
    spec = GroupSpec("C7", build_c7(), c7.ELEMENTS, c7.IDENTITY, c7.compose, c7.inverse, c7_transform, c7_relations())
    split = split_templates_three_way(spec.frame, seed=101, train_templates=28, validation_templates=6, test_templates=6)
    hidden = np.zeros((len(spec.frame), 20), dtype=np.float32)
    hidden[:, :7] = np.eye(7, dtype=np.float32)[spec.frame.state_id.to_numpy()]
    try:
        fit_base_artifact(spec, hidden, split, 16, config)
    except ValueError as error:
        assert "observed centered training rank" in str(error)
    else:
        raise AssertionError("ensure_observed_pca_rank did not reject infeasible latent_dim=16")


def test_afs_d_internal_metadata_uses_rank_matched_dimensions():
    config = resolve_config(CONFIG)
    model = config["selected_models"][0]
    assert _embedding_metadata(model, "C7")["latent_dim"] == 6
    assert _embedding_metadata(model, "D4")["latent_dim"] == 3
    assert _internal_metadata(model, "C7", 0.2, 3, 101, group_latent_dim(config, "C7"))["latent_dim"] == 6
    assert _internal_metadata(model, "D4", 0.2, 3, 101, group_latent_dim(config, "D4"))["latent_dim"] == 3


def test_afs_d_embedding_pca_and_operators_are_training_only():
    config = resolve_config(CONFIG)
    spec = GroupSpec("C7", build_c7(), c7.ELEMENTS, c7.IDENTITY, c7.compose, c7.inverse, c7_transform, c7_relations())
    split = split_templates_three_way(spec.frame, seed=101, train_templates=28, validation_templates=6, test_templates=6)
    rng = np.random.default_rng(7)
    hidden = rng.normal(size=(len(spec.frame), 24))
    first = fit_base_artifact(spec, hidden, split, group_latent_dim(config, "C7"), config)
    changed = hidden.copy(); changed[~split.train_mask] += 1000.0
    second = fit_base_artifact(spec, changed, split, group_latent_dim(config, "C7"), config)
    np.testing.assert_array_equal(first["pca"].mean_, second["pca"].mean_)
    np.testing.assert_array_equal(first["pca"].components_, second["pca"].components_)
    for element in spec.elements:
        np.testing.assert_array_equal(first["operators"][element], second["operators"][element])


def test_afs_d_reuses_exact_c7_and_d4_generator_composition():
    c7_spec = GroupSpec("C7", build_c7(), c7.ELEMENTS, c7.IDENTITY, c7.compose, c7.inverse, c7_transform, c7_relations())
    d4_spec = GroupSpec("D4", build_d4(), d4.ELEMENTS, d4.IDENTITY, d4.compose, d4.inverse, d4_transform, d4_relations())
    assert c7.power(c7.GENERATOR, 7) == c7.IDENTITY
    words = d4_canonical_words()
    assert all(d4.row_word_product(word) == element for element, word in words.items())
    assert d4.row_word_product((d4.S, d4.R, d4.S)) == d4.inverse(d4.R)
    assert d4.compose(d4.R, d4.S) != d4.compose(d4.S, d4.R)
    for spec in (c7_spec, d4_spec):
        split = split_templates_three_way(spec.frame, seed=101, train_templates=28, validation_templates=6, test_templates=6)
        z = np.random.default_rng(9).normal(size=(len(spec.frame), 16))
        changed = z.copy(); changed[~split.train_mask] += 999.0
        first = fit_generators(spec, z, split.train_mask, 0.01)
        second = fit_generators(spec, changed, split.train_mask, 0.01)
        for generator in first:
            np.testing.assert_array_equal(first[generator], second[generator])


def test_afs_d_result_packaging_is_namespaced_and_excludes_checkpoints():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = root / "afs_d_synthetic"
        for relative in ("afs_d", "results/tables", "checkpoints/embedding_extraction"):
            (run / relative).mkdir(parents=True, exist_ok=True)
        (run / "manifest.json").write_text(json.dumps({
            "experiment": "beyond_shared_geometry_afs_d_rank_aware",
            "status": "SUCCESS",
            "artifacts": {},
        }), encoding="utf-8")
        (run / "interpretation.json").write_text("{}\n", encoding="utf-8")
        (run / "AFS_D_SCIENTIFIC_REPORT.md").write_text("synthetic\n", encoding="utf-8")
        (run / "afs_d" / "bootstrap_depth_contrasts.csv").write_text("metric,mean_difference\nE_eq,0\n", encoding="utf-8")
        (run / "afs_d" / "embedding_rank_diagnostics.csv").write_text("group,centered_training_rank\nC7,6\n", encoding="utf-8")
        (run / "results" / "tables" / "afs_d_depth_contrasts.csv").write_text("metric,mean_difference\nE_eq,0\n", encoding="utf-8")
        (run / "checkpoints" / "embedding_extraction" / "secret.npz").write_bytes(b"checkpoint")
        bundle = root / "result.zip"
        env = {
            **os.environ,
            "AFS_D_RESULT_BUNDLE_PATH": str(bundle),
            "PYTHONPATH": str(ROOT / "src"),
            "PATH": str(Path(os.sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        }
        subprocess.run(
            ["bash", str(ROOT / "runpod" / "package_afs_d_results.sh"), str(run)],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            assert all(name.startswith("Beyond_Shared_Geometry_afs_d_results/") for name in names)
            assert not any("checkpoints" in name for name in names)
            manifest = json.loads(archive.read("Beyond_Shared_Geometry_afs_d_results/manifest.json"))
            assert manifest["status"] == "SUCCESS"
            assert "runtime_profile.csv" in manifest["artifacts"]
