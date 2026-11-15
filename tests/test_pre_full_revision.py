from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from bsg.experiments.main_config import resolve_main_config
from bsg.experiments.main_experiment import (
    _bootstrap_rows,
    _bootstrap_analysis_targets,
    _combine_gate_statuses,
    _depth_point_rule,
    _full_workload_unit_targets,
    _full_checkpoint_fields,
    _full_causal_checkpoint_fields,
    _generate_stable_artifacts,
    _group_specs,
    _cka_association_summary,
    _scientific_unit_targets,
    _validate_calibration,
)
from bsg.experiments.runtime_state import experiment_key
from bsg.groups import d4
from bsg.interventions.causal import (
    fit_source_conditioned_additive,
    source_conditioned_additive_intervention,
)
from bsg.statistics.hierarchical import hierarchical_mean_ci, paired_baseline_ci


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "main_experiment.yaml"


def _balanced_d4_fixture():
    frame = pd.DataFrame([
        {"template_id": template, "state_id": state}
        for template in range(3)
        for state in range(4)
    ])
    hidden = np.asarray([
        np.eye(4)[row.state_id] + np.asarray([row.template_id * 0.1] * 4)
        for row in frame.itertuples()
    ])
    train_mask = frame.template_id.to_numpy() < 2
    return frame, hidden, train_mask


def test_balanced_orbit_global_element_average_cancels_but_source_conditioned_does_not():
    frame, hidden, train_mask = _balanced_d4_fixture()
    lookup = {
        (row.template_id, row.state_id): index
        for index, row in enumerate(frame.itertuples())
    }
    global_deltas = []
    for row in frame[train_mask].itertuples():
        target = d4.apply(d4.R, row.state_id)
        global_deltas.append(hidden[lookup[(row.template_id, target)]] - hidden[lookup[(row.template_id, row.state_id)]])
    np.testing.assert_allclose(np.mean(global_deltas, axis=0), 0.0, atol=1e-12)

    vectors = fit_source_conditioned_additive(
        frame, hidden, train_mask, [d4.R], lambda state, element: d4.apply(element, state)
    )
    assert all(np.linalg.norm(vectors[(d4.R, state)]) > 0 for state in range(4))
    assert not np.allclose(vectors[(d4.R, 0)], vectors[(d4.R, 1)])


def test_source_conditioned_additive_is_train_only_and_uses_exact_d4_action():
    frame, hidden, train_mask = _balanced_d4_fixture()
    elements = (d4.R, d4.S, d4.power(d4.R, 2), d4.power(d4.R, 3), d4.inverse(d4.R))
    transform = lambda state, element: d4.apply(element, state)
    vectors = fit_source_conditioned_additive(frame, hidden, train_mask, elements, transform)
    altered = hidden.copy()
    for index in np.flatnonzero(~train_mask):
        altered[index] += (int(frame.iloc[index].state_id) + 1) * 1_000_000
    altered_vectors = fit_source_conditioned_additive(frame, altered, train_mask, elements, transform)
    for element in elements:
        for source_state in range(4):
            np.testing.assert_allclose(vectors[(element, source_state)], altered_vectors[(element, source_state)])
            expected = np.eye(4)[d4.apply(element, source_state)] - np.eye(4)[source_state]
            np.testing.assert_allclose(vectors[(element, source_state)], expected)
            held_out_index = int(frame.index[(frame.template_id == 2) & (frame.state_id == source_state)][0])
            steered = source_conditioned_additive_intervention(
                hidden[held_out_index], vectors, element, source_state,
            )
            np.testing.assert_allclose(steered, hidden[held_out_index] + expected)


def test_deterministic_duplicates_do_not_change_weight_but_random_seeds_remain():
    learned_rows = []
    for split_seed in (101, 303):
        for template_id in (1, 2):
            value = split_seed / 1000 + template_id
            for optimization_seed in (42, 202, 777):
                learned_rows.append({
                    "model": "m", "depth_slot": 0.4, "split_seed": split_seed,
                    "template_id": template_id, "optimization_seed": optimization_seed,
                    "value": value,
                })
    duplicated = pd.DataFrame(learned_rows)
    single = duplicated[duplicated.optimization_seed == 42]
    duplicate_summary = hierarchical_mean_ci(duplicated, "value", n_boot=100, seed=17)
    single_summary = hierarchical_mean_ci(single, "value", n_boot=100, seed=17)
    assert duplicate_summary == single_summary
    assert duplicate_summary["cluster_columns"] == "model,depth_slot,split_seed,template_id"

    paired_rows = [{
        "model": "m", "depth_slot": 0.4, "split_seed": 101, "template_id": 1,
        "method": "learned_operator", "value": 1.0,
    }]
    paired_rows.extend({
        "model": "m", "depth_slot": 0.4, "split_seed": 101, "template_id": 1,
        "method": "random_orthogonal", "random_baseline_seed": seed, "value": value,
    } for seed, value in zip((11, 13, 17, 19, 23), (2.0, 3.0, 4.0, 5.0, 6.0)))
    paired = paired_baseline_ci(pd.DataFrame(paired_rows), "value", "learned_operator", "random_orthogonal", n_boot=20, seed=9)
    assert paired["n_pairs"] == 5
    assert paired["n_random_seeds"] == 5
    assert paired["mean"] == -3.0
    assert paired["cluster_columns"] == "model,depth_slot,split_seed,random_baseline_seed,template_id"


def test_joint_vs_independent_ridge_paired_inference_is_emitted():
    rows = []
    for split_seed in (101, 303):
        for template_id in (1, 2):
            for method, value in (("group_consistent_joint", 0.12), ("independent_ridge", 0.10)):
                rows.append({
                    "data_split": "test", "model": "m", "group": "C7", "method": method,
                    "depth_slot": 0.4, "layer": 6, "latent_dim": 16,
                    "split_seed": split_seed, "optimization_seed": 42,
                    "template_id": template_id, "E_eq": value, "E_rel_data": value,
                    "E_hom_data": value, "E_comp_data": value,
                    "E_inverse_data": value, "E_word": value,
                })
    config = {"primary_latent_dimension": 16, "bootstrap_samples": 20, "bootstrap_seed": 5}
    result = _bootstrap_rows(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame(), config)
    joint = result[(result.analysis == "joint_minus_independent_ridge") & (result.metric == "E_eq")]
    assert set(joint.model) == {"m", "ALL"}
    np.testing.assert_allclose(joint["mean"], 0.02, atol=1e-12)
    np.testing.assert_allclose(joint["learned_estimate"], 0.12, atol=1e-12)
    np.testing.assert_allclose(joint["baseline_estimate"], 0.10, atol=1e-12)


def test_depth_gate_pairs_deterministic_learned_rows_with_random_replications():
    rows = []
    for depth_slot in (0.2, 0.4, 0.6, 0.8):
        rows.append({
            "model": "m", "group": "C7", "depth_slot": depth_slot,
            "split_seed": 101, "template_id": 1,
            "method": "group_consistent_joint", "E_eq": 0.1,
        })
        for seed in (11, 13, 17, 19, 23):
            rows.append({
                "model": "m", "group": "C7", "depth_slot": depth_slot,
                "split_seed": 101, "template_id": 1,
                "method": "random_orthogonal", "random_baseline_seed": seed,
                "E_eq": 0.5,
            })
    result = _depth_point_rule(
        pd.DataFrame(rows), "C7", "E_eq",
        "group_consistent_joint", "random_orthogonal", direction="negative",
    )
    assert result["status"] == "PASS"
    assert result["successful_depths"] == 4


def test_gate_wording_distinguishes_not_evaluable_and_nonsignificant_association():
    assert _combine_gate_statuses(["PASS", "NOT_EVALUABLE"]) == "NOT_EVALUABLE"
    assert _combine_gate_statuses(["PASS", "FAIL", "NOT_EVALUABLE"]) == "FAIL"
    association = _cka_association_summary(pd.DataFrame([{
        "mean": -0.14, "ci_low": -0.38, "ci_high": 0.09,
    }]))
    assert association["status"] == "NO_RELIABLE_ASSOCIATION"
    assert "independ" not in association["interpretation"].lower()


def test_d4_word_products_distinguish_true_relation_and_wrong_order():
    assert d4.row_word_product((d4.S, d4.R, d4.S)) == d4.inverse(d4.R)
    assert d4.row_word_product((d4.R, d4.S, d4.R)) == d4.S
    assert d4.S != d4.inverse(d4.R)


def test_d4_true_relation_is_paired_against_wrong_order_distance():
    rows = []
    for split_seed in (101, 303):
        for template_id in (1, 2):
            for intervention, role, distance in (
                ("srs_r_inverse", "relation_consistency", 0.1),
                ("rsr_wrong_order", "order_control", 0.5),
            ):
                rows.append({
                    "model": "m", "group": "D4", "method": "learned_operator",
                    "depth_slot": 0.4, "layer": 6, "latent_dim": 16,
                    "split_seed": split_seed, "template_id": template_id,
                    "source_state": "NW", "target_state": "SW",
                    "intervention": intervention, "evaluation_role": role,
                    "relation_final_logit_distance": distance,
                    "relation_downstream_hidden_distance": distance,
                    "relation_target_probability_difference": distance,
                    "relation_logit_margin_difference": distance,
                    "relation_sequence_log_probability_difference": distance,
                    "target_logit_change": 0.0, "target_probability_change": 0.0,
                    "top1_success": 0, "logit_margin_vs_original": 0.0,
                })
    config = {"primary_latent_dimension": 16, "bootstrap_samples": 20, "bootstrap_seed": 5}
    result = _bootstrap_rows(pd.DataFrame(), pd.DataFrame(rows), pd.DataFrame(), config)
    contrast = result[
        (result.analysis == "d4_true_minus_wrong_order")
        & (result.metric == "relation_final_logit_distance")
    ]
    assert set(contrast.model) == {"m", "ALL"}
    np.testing.assert_allclose(contrast["mean"], -0.4, atol=1e-12)


def test_pre_full_and_revised_full_workload_matrices_are_exact():
    historical_core = resolve_main_config(CONFIG, "core")
    assert historical_core["learned_fit_semantics"] == "historical_seed_replicated"
    assert historical_core["causal"]["baselines"] == ["identity", "random_orthogonal", "mean_difference"]
    assert historical_core["causal"]["d4_wrong_order_controls"] == []
    pre_full = resolve_main_config(CONFIG, "pre_full")
    assert pre_full["learned_fit_semantics"] == "deterministic_single"
    assert pre_full["bootstrap_samples"] == 2000
    assert _scientific_unit_targets(pre_full) == {
        "model_load": 2,
        "activation_extraction": 4,
        "causal_activation_extraction": 8,
        "structural_base": 16,
        "structural_joint": 16,
        "causal": 16,
        "cross_model": 8,
    }
    assert _bootstrap_analysis_targets(pre_full) == 398
    full = resolve_main_config(CONFIG, "full")
    assert _scientific_unit_targets(full) == {
        "model_load": 6,
        "activation_extraction": 12,
        "causal_activation_extraction": 60,
        "structural_base": 444,
        "structural_joint": 444,
        "causal": 240,
        "cross_model": 360,
    }
    assert _bootstrap_analysis_targets(full) == 926
    assert _full_workload_unit_targets(full) == {
        "model_load": 6,
        "activation_extraction": 72,
        "structural_base": 444,
        "structural_joint": 444,
        "causal": 240,
        "cross_model": 360,
    }


def test_revised_full_gate_accepts_pre_full_and_rejects_historical_core():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "go_no_go.json").write_text('{"overall":"GO"}\n', encoding="utf-8")
        (root / "FULL_COST_PROJECTION.md").write_text("finite\n", encoding="utf-8")
        pre_full = root / "pre_full.json"
        pre_full.write_text('{"tier":"pre_full","status":"SUCCESS"}\n', encoding="utf-8")
        _validate_calibration(pre_full, "pre_full")
        core = root / "core.json"
        core.write_text('{"tier":"core","status":"SUCCESS"}\n', encoding="utf-8")
        try:
            _validate_calibration(core, "pre_full")
        except RuntimeError as error:
            assert "PRE_FULL" in str(error)
        else:
            raise AssertionError("historical CORE incorrectly authorized revised FULL")


def test_packaged_pre_full_authorization_is_explicit_and_self_contained():
    manifest = ROOT / "runpod" / "full_authorization" / "manifest.json"
    _validate_calibration(manifest, "pre_full")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["run_id"] == "pre_full_001"
    assert payload["additive_audit_verdict"] == "PASS WITH REPORTING LIMITATION"
    assert payload["full_decision"] == "AUTHORIZE FULL"
    assert not str(manifest).startswith(str(ROOT / "runpod_output_by_date"))


def test_full_compound_checkpoint_keys_bind_controls_words_and_roles_only_prospectively():
    full = resolve_main_config(CONFIG, "full")
    pre_full = resolve_main_config(CONFIG, "pre_full")
    fields = {
        "model": "m", "group": "D4", "layer": 6, "depth_slot": 0.4,
        "latent_dim": 16, "split_seed": 101, "optimization_seed": 42,
        "random_baseline_seed": None,
    }
    seeds = full["random_baseline_seeds"]
    bound = _full_checkpoint_fields(fields, full, seeds)
    assert bound["random_baseline_seeds"] == seeds
    assert _full_checkpoint_fields(fields, pre_full, pre_full["random_baseline_seeds"]) == fields
    changed = _full_checkpoint_fields(fields, full, seeds[:-1])
    assert experiment_key("structural_joint", bound) != experiment_key("structural_joint", changed)

    specs = _group_specs()
    d4_fields = _full_causal_checkpoint_fields(fields, specs["D4"], full, seeds)
    d4_contract = {
        item["intervention"]: item["role"] for item in d4_fields["interventions"]
    }
    assert d4_contract["srs_r_inverse"] == "relation_consistency"
    assert d4_contract["rsr_wrong_order"] == "order_control"
    assert d4_contract["r_inverse_direct"] == "relation_reference"
    assert "source_conditioned_additive" in d4_fields["methods"]
    assert d4_fields["random_baseline_seeds"] == seeds

    c7_fields = _full_causal_checkpoint_fields(fields, specs["C7"], full, seeds)
    c7_contract = {
        item["intervention"]: item["role"] for item in c7_fields["interventions"]
    }
    assert c7_contract["R^7"] == "cycle_recovery"
    assert all(c7_contract[f"R^{step}"] == "transformation" for step in range(1, 7))


def test_c7_transformation_and_cycle_have_separate_figure_sources():
    structural = pd.DataFrame([{
        "data_split": "test", "latent_dim": 16, "depth_slot": 0.4,
        "model": "m", "group": "C7", "method": "group_consistent_joint",
        "layer": 6, "E_eq": 0.1, "E_rel_data": 0.2,
    }])
    causal = pd.DataFrame([
        {"group": "C7", "model": "m", "method": "learned_operator", "evaluation_role": "transformation", "intervention": "R^1", "depth_slot": 0.4, "layer": 6, "target_logit_change": 0.2, "cycle_hidden_relative_error": np.nan},
        {"group": "C7", "model": "m", "method": "learned_operator", "evaluation_role": "cycle_recovery", "intervention": "R^7", "depth_slot": 0.4, "layer": 6, "target_logit_change": 99.0, "cycle_hidden_relative_error": 0.03},
    ])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "results" / "tables").mkdir(parents=True)
        (root / "results" / "figures").mkdir()
        _generate_stable_artifacts(root, structural, pd.DataFrame(), pd.DataFrame(), causal, pd.DataFrame())
        steps = pd.read_csv(root / "results" / "figures" / "c7_causal_by_step.source.csv")
        cycle = pd.read_csv(root / "results" / "figures" / "c7_cycle_recovery.source.csv")
        assert steps.step.tolist() == [1.0]
        assert steps.target_logit_change.tolist() == [0.2]
        assert cycle.cycle_hidden_relative_error.tolist() == [0.03]
