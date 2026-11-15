"""Scientific-contract tests: no model inference, no prior calibration rerun."""
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact import fit_models, compose, evaluate_compact, Predictor
from common import make_cell


def bank():
    rng = np.random.default_rng(24)
    x = rng.normal(size=(80, 4))
    xv = rng.normal(size=(20, 4))
    transform = lambda q, a: q + (a + 1) * np.sin(q * .7)
    targets = {str(a): transform(x, a) for a in range(2)}
    validation = {str(a): (xv, transform(xv, a)) for a in range(2)}
    return x, targets, validation


def small_config(methods=None):
    return {"methods": methods or ["ridge", "affine", "limited_support", "rff", "nystrom"],
            "budget_multipliers": [.5, 1, 2], "lambda_grid": [0, .001, .1],
            "bandwidth_multipliers": [.5, 1], "seed": 5}


def test_storage_counts_actual_retained_numeric_payload_and_infeasible_budgets():
    x, targets, validation = bank()
    models = fit_models(x, targets, validation, small_config())
    assert any(m.predict is None for m in models)
    for model in models:
        if model.predict is None:
            assert model.metadata["status"] == "infeasible"
            continue
        actual = sum(np.asarray(v).nbytes for v in model.predict.params.values())
        reported = model.metadata["storage"]
        assert actual == reported["parameter_bytes"] == reported["materialized_numeric_parameter_bytes"]
        budget = model.metadata["budget_float64_equivalent_coordinates"]
        if budget is not None:
            assert actual <= 8 * budget
        assert all(np.asarray(v).shape != x.shape for v in model.predict.params.values())
        json.dumps(model.metadata, allow_nan=False)


def test_all_pair_regression_responds_to_nonlandmark_target_but_field_does_not():
    x, targets, validation = bank()
    config = small_config(["limited_support", "rff", "nystrom"])
    config["budget_multipliers"] = [2]
    before = fit_models(x, targets, validation, config)
    field = next(m for m in before if m.metadata["family"] == "limited_support")
    excluded = next(i for i in range(len(x)) if i not in field.metadata["support_source_indices"])
    changed = {a: y.copy() for a, y in targets.items()}
    changed["0"][excluded] += np.asarray([20, -10, 30, 40])
    after = fit_models(x, changed, validation, config)
    probe = x[:15]
    for first, second in zip(before, after):
        difference = np.max(np.abs(first.predict(probe, "0") - second.predict(probe, "0")))
        if first.metadata["family"] == "limited_support":
            assert difference == 0
            assert first.metadata["all_training_pairs_exposed"] is False
        else:
            assert difference > 1e-6
            assert first.metadata["all_training_pairs_exposed"] is True
            assert first.metadata["training_target_pairs_used"] == 160


def test_centroid_redecodes_each_predicted_intermediate():
    x = np.asarray([[0.], [1.], [3.]])
    target = x[[1, 2, 0]]
    model = fit_models(x, {"r": target}, {"r": (x, target)},
                       {"methods": ["centroid"]}, states=np.arange(3), tables={"r": (1, 2, 0)})[0]
    assert model.metadata["intermediate_state_redecoded"]
    np.testing.assert_allclose(compose(model.predict, np.asarray([[.8]]), ("r", "r")), [[-.2]], atol=1e-12)
    assert model.metadata["storage"]["integer64_coordinates"] == 3


def test_fit_api_cannot_receive_heldout_targets_and_rejects_misalignment():
    signature = inspect.signature(fit_models)
    assert "test" not in signature.parameters and "evaluation" not in signature.parameters
    x, targets, validation = bank()
    with pytest.raises(ValueError, match="identical string"):
        fit_models(x, targets, {"0": validation["0"]}, small_config())
    with pytest.raises(ValueError, match="align"):
        fit_models(x, {"0": targets["0"][:-1], "1": targets["1"]}, validation, small_config())


def circular_cell():
    rows, z = [], []
    for family, split in enumerate(("train", "validation", "pilot", "test")):
        for state in range(7):
            for paraphrase in range(2):
                angle = 2 * np.pi * state / 7
                z.append([np.cos(angle), np.sin(angle)])
                rows.append({"row_id": f"{family}-{state}-{paraphrase}", "family_id": str(family),
                             "orbit_id": "0", "split": split, "state_id": state, "paraphrase_id": paraphrase})
    return make_cell(np.asarray(z), rows, "C7")


def test_generator_compositions_use_predictions_and_training_energy():
    cell = circular_cell()
    result = evaluate_compact(cell, {"methods": ["ridge"], "lambda_grid": [0], "technical_test": True}, ["test"])
    assert result["technical_test"] is True
    np.testing.assert_allclose(result["training_energy"], 1., atol=1e-12)
    assert "unseen_composite_equal_action_mean_nmse" not in result["tracks"]["all_element"]["models"][0]["evaluation"]["test"]
    generators = result["tracks"]["generator_only"]
    assert generators["fitting_actions"] == ["1"]
    outcome = generators["models"][0]["evaluation"]["test"]
    assert outcome["unseen_composite_equal_action_mean_nmse"] < 1e-25
    assert outcome["composition_diagnostics"]["cycles"]["r7"]["mean_nmse"] < 1e-25
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="Evaluation partitions"):
        evaluate_compact(cell, {"methods": ["ridge"]}, ["train"])
    with pytest.raises(ValueError, match="absent"):
        evaluate_compact(cell, {"methods": ["ridge"]}, ["test_joint"])


def test_method_filter_and_determinism():
    x, targets, validation = bank()
    cfg = small_config(["rff"])
    first, second = fit_models(x, targets, validation, cfg), fit_models(x, targets, validation, cfg)
    assert all(m.name.startswith("rff") for m in first)
    for left, right in zip(first, second):
        assert left.metadata == right.metadata
        if left.predict is not None:
            np.testing.assert_array_equal(left.predict(x[:8], "0"), right.predict(x[:8], "0"))


def test_zero_penalty_matches_audited_singular_cutoff():
    from compact import _ridge_solutions
    x = np.diag([1., 1e-7])
    target = np.eye(2)
    penalty, coefficient = next(_ridge_solutions(x, target, [0]))
    np.testing.assert_allclose(x @ coefficient, target, atol=1e-12)
    assert coefficient[1, 1] > 1e6


def test_d4_order_control_excludes_equal_target_states():
    rows, z = [], []
    for family, split in enumerate(("train", "validation", "test")):
        for state in range(4):
            for paraphrase in range(2):
                z.append(np.eye(4)[state])
                rows.append({"row_id": f"{family}-{state}-{paraphrase}", "family_id": str(family),
                             "orbit_id": "0", "split": split, "state_id": state, "paraphrase_id": paraphrase})
    cell = make_cell(np.asarray(z), rows, "D4_corner")
    result = evaluate_compact(cell, {"methods": ["ridge"], "lambda_grid": [0]}, ["test"])
    diagnostic = result["tracks"]["generator_only"]["models"][0]["evaluation"]["test"]["composition_diagnostics"]
    order = diagnostic["d4_discriminating_order"]
    assert order["correct_word_to_designated_target"]["n"] == 4
    assert order["correct_word_to_designated_target"]["mean_nmse"] < 1e-25
    assert order["wrong_word_to_designated_target"]["mean_nmse"] > 1
    commutator = diagnostic["d4_discriminating_noncommutativity"]
    assert commutator["correct_word_to_designated_target"]["n"] == 8


def test_nystrom_collapsed_coefficients_preserve_whitened_predictions():
    from compact import _collapse_nystrom_coefficients, _distances
    rng = np.random.default_rng(901)
    d, m, actions = 4, 6, 2
    landmarks = rng.normal(size=(m, d))
    matrix = rng.normal(size=(m, m))
    whitener = matrix @ matrix.T + np.eye(m)
    beta = rng.normal(size=(m + 1, actions * d))
    collapsed = _collapse_nystrom_coefficients(whitener, beta)
    params = {"mean": rng.normal(size=d), "scale": 1.7, "bandwidth": .9,
              "landmarks": landmarks,
              "coef": collapsed.reshape(m + 1, actions, d).transpose(1, 0, 2).copy()}
    predictor = Predictor("nystrom", ("0", "1"), params)
    query = rng.normal(size=(13, d))
    normalized = (query - params["mean"]) / params["scale"]
    kernel = np.exp(-_distances(normalized, landmarks) / (2 * params["bandwidth"] ** 2))
    original_residual = np.column_stack((kernel @ whitener, np.ones(len(query)))) @ beta
    for action in range(actions):
        expected = query + params["scale"] * original_residual[:, action * d:(action + 1) * d]
        np.testing.assert_allclose(predictor(query, str(action)), expected, rtol=1e-12, atol=1e-12)
    assert "whitener" not in predictor.params
    x, targets, validation = bank()
    fitted = fit_models(x, targets, validation, {"methods": ["nystrom"], "budget_multipliers": [2],
                       "lambda_grid": [.001], "bandwidth_multipliers": [1]})[0]
    assert "whitener" not in fitted.predict.params
    expected_features = (64 - (4 + 2 + 2 * 4)) // (4 + 2 * 4)
    assert fitted.metadata["feature_count"] == expected_features
    assert fitted.metadata["storage"]["parameter_bytes"] <= 64 * 8
