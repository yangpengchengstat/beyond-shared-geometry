"""Meaningful pairing, input-information, split, and normalization regressions."""
from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import action_catalog, make_cell
from context import evaluate_context, pair_context, predict_category, _score


def fixture_cell(domain='C7', registration='semantic'):
    elements, identity, _, tables, _ = action_catalog(domain)
    nstates = len(tables[identity])
    if domain == 'S3':
        partitions = ['train', 'train', 'validation', 'pilot', 'test_joint',
                      'test_context', 'test_entity', 'support', 'reserved']
    else:
        partitions = ['train', 'train', 'validation', 'pilot', 'test', 'support', 'reserved']
    rows = []
    coordinates = []
    for block, split in enumerate(partitions):
        for state in range(nstates):
            center = np.asarray([state * 5., state ** 2., block * .1])
            difference = np.asarray([2. + state / 10., 1. + block / 20., .5 + state / 20.])
            for paraphrase in range(2):
                row = {'row_id': f'{block}-{state}-{paraphrase}',
                       'family_id': f'family-{block}', 'orbit_id': f'orbit-{block}',
                       'state_id': state, 'paraphrase_id': paraphrase, 'split': split}
                rows.append(row)
                coordinates.append(center + (1 if paraphrase == 0 else -1) * difference / 2)
    return make_cell(np.asarray(coordinates), rows, domain, registration)


def config():
    return {'context_nonlinear': False, 'lambda_grid': [0., .001, .1],
            'scalar_grid': [0., .5, 1., 1.25]}


def test_difference_energy_is_not_recentered():
    cell = fixture_cell()
    paired = pair_context(cell)
    expected = np.mean(np.sum(paired.u[paired.train] ** 2, axis=1))
    centered = np.mean(np.sum((paired.u[paired.train] - paired.u[paired.train].mean(0)) ** 2, axis=1))
    result = evaluate_context(cell, config(), ['test'])
    assert result['train_energy'] == pytest.approx(expected)
    assert expected > centered * 20
    identity = next(m for m in result['methods'] if m['name'] == 'identity_context')
    action = cell.generators[0]
    selected = np.asarray([r['split'] == 'test' for r in paired.rows])
    target = paired.u[paired.targets[action][selected]]
    manual = np.mean(np.sum((paired.u[selected] - target) ** 2, axis=1)) / expected
    assert identity['splits']['test']['per_action'][action]['mean_normalized_squared_error'] == pytest.approx(manual)


def test_missing_duplicate_or_inconsistent_paraphrases_rejected():
    cell = fixture_cell()
    cell.rows = cell.rows[:-1]
    cell.z = cell.z[:-1]
    with pytest.raises(ValueError, match='exactly two'):
        pair_context(cell)
    cell = fixture_cell()
    cell.rows[1]['paraphrase_id'] = cell.rows[0]['paraphrase_id']
    with pytest.raises(ValueError, match='Duplicate paraphrase'):
        pair_context(cell)
    cell = fixture_cell()
    cell.rows[0]['paraphrase_id'] = 8
    with pytest.raises(ValueError, match='ordering'):
        pair_context(cell)


def test_cross_partition_pair_rejected():
    cell = fixture_cell()
    cell.rows[1]['split'] = 'test'
    with pytest.raises(ValueError, match='crosses partitions'):
        pair_context(cell)


def test_parent_registration_is_preserved_and_checked():
    semantic = pair_context(fixture_cell())
    alternate_cell = fixture_cell(registration='consistent_swap01')
    alternate = pair_context(alternate_cell)
    action = alternate_cell.generators[0]
    assert not np.array_equal(semantic.targets[action], alternate.targets[action])
    alternate_cell.targets[action] = fixture_cell().targets[action]
    with pytest.raises(ValueError, match='parent registration'):
        pair_context(alternate_cell)


def test_test_and_support_coordinates_cannot_affect_fits_or_selection():
    original = fixture_cell()
    mutated = fixture_cell()
    protected = np.asarray([r['split'] not in ('train', 'validation') for r in mutated.rows])
    random = np.random.default_rng(2026)
    mutated.z[protected] = random.normal(size=mutated.z[protected].shape) * 1000
    before = evaluate_context(original, config(), ['test'])
    after = evaluate_context(mutated, config(), ['test'])
    assert before['train_energy'] == after['train_energy']
    assert [m['metadata'] for m in before['methods']] == [m['metadata'] for m in after['methods']]
    assert before['methods'][0]['splits'] != after['methods'][0]['splits']


def test_decoded_predictor_has_no_label_or_target_argument():
    assert list(inspect.signature(predict_category).parameters) == ['u', 'centers', 'alphas', 'centroids']
    u = np.asarray([[1., 2.], [3., 4.]])
    centers = np.asarray([[9., 9.], [0., 0.]])
    centroids = np.asarray([[0., 0.], [10., 10.]])
    result = predict_category(u, centers, np.asarray([2., 3.]), centroids)
    np.testing.assert_array_equal(result, u * np.asarray([[3.], [2.]]))
    records = evaluate_context(fixture_cell(), config(), ['test'])['methods']
    decoded = next(r for r in records if r['name'] == 'decoded_category_retention_context')
    oracle = next(r for r in records if r['name'] == 'oracle_category_retention_context')
    assert not decoded['metadata']['oracle_state_labels_at_evaluation']
    assert oracle['metadata']['oracle_state_labels_at_evaluation']
    assert decoded['metadata']['storage']['float64_coordinates'] - oracle['metadata']['storage']['float64_coordinates'] == 7 * 3


def test_s3_strata_are_separate_and_extra_rows_never_fit():
    cell = fixture_cell('S3')
    result = evaluate_context(cell, config(), ['test_joint', 'test_context', 'test_entity'])
    assert result['training']['pairs'] == 12
    assert result['training']['validation_pairs'] == 6
    assert result['training']['all_split_pair_counts']['support'] == 6
    assert result['training']['all_split_pair_counts']['reserved'] == 6
    assert result['training']['support_and_reserved_used'] is False
    for method in result['methods']:
        assert set(method['splits']) == {'test_joint', 'test_context', 'test_entity'}
        for split in method['splits'].values():
            assert split['pooled']['n'] == 30  # five nonidentity permutations, six states
    for forbidden in ('support', 'reserved', 'train'):
        with pytest.raises(ValueError, match='evaluation must'):
            evaluate_context(cell, config(), [forbidden])


def test_d4_evaluates_only_changed_states_but_fits_all_pairs():
    cell = fixture_cell('D4')
    paired = pair_context(cell)
    result = evaluate_context(cell, config(), ['pilot'])
    assert result['training']['pairs'] == 8
    expected = sum(sum(i != j for i, j in enumerate(cell.tables[a]))
                   for a in cell.elements if a != cell.identity)
    assert expected < 4 * 7
    assert result['methods'][0]['splits']['pilot']['pooled']['n'] == expected
    assert result['training']['training_includes_fixed_state_pairs'] is True


def test_nonfinite_predictions_are_counted_not_dropped():
    paired = pair_context(fixture_cell())
    energy = np.mean(np.sum(paired.u[paired.train] ** 2, axis=1))
    def invalid(indices, action):
        out = np.copy(paired.u[indices])
        out[0] = np.nan
        return out
    result = _score(paired, ['1'], energy, invalid, ['test'])['test']
    assert result['pooled']['n'] == 7
    assert result['pooled']['invalid_predictions'] == 1
    assert result['pooled']['mean_normalized_squared_error'] is None
    assert result['pooled']['equal_family_mean'] is None


def test_zero_energy_is_reported_undefined_instead_of_invented_denominator():
    cell = fixture_cell()
    for i in range(0, len(cell.z), 2):
        cell.z[i + 1] = cell.z[i]
    with pytest.raises(ValueError, match='normalization is undefined'):
        evaluate_context(cell, config(), ['test'])


def test_integrated_nonlinear_controls_expose_all_train_pairs_and_honor_budgets():
    cfg = {'lambda_grid': [0., .01], 'bandwidth_multipliers': [.5, 1.],
           'budget_multipliers': [1., 4.], 'seed': 99}
    result = evaluate_context(fixture_cell(), cfg, ['pilot'])
    nonlinear = [m for m in result['methods'] if m['name'].startswith(('rff_', 'nystrom_'))]
    assert len(nonlinear) == 4
    for model in nonlinear:
        meta = model['metadata']
        assert meta['all_training_pairs_exposed']
        assert meta['training_target_pairs_available'] == 14 * 6
        assert meta['storage']['parameter_bytes'] <= meta['budget_float64_equivalent_coordinates'] * 8
        assert model['validation_selected_predictor']['pooled']['n'] == 7 * 6
        assert model['splits']['pilot']['pooled']['invalid_predictions'] == 0


def test_requested_absent_evaluation_partition_fails_loudly():
    with pytest.raises(ValueError, match='partition is absent'):
        evaluate_context(fixture_cell(), config(), ['test_joint'])


def test_fractional_state_metadata_rejected_before_pairing():
    cell = fixture_cell()
    cell.rows[0]['state_id'] = .1
    with pytest.raises(ValueError, match='state identity must be an integer'):
        pair_context(cell)


def test_coordinate_fitted_reference_receives_z_only_and_is_applied_to_u(monkeypatch):
    import compact
    cell = fixture_cell()
    paired = pair_context(cell)
    seen = {}
    def spy(x, targets, validation, configuration, **kwargs):
        assert configuration['methods'] == ['ridge']
        seen['x'] = np.copy(x)
        seen['targets'] = copy.deepcopy(targets)
        seen['validation'] = copy.deepcopy(validation)
        actions = list(targets)
        d = x.shape[1]
        def predict(query, action):
            return np.asarray(query) * 2.
        return [compact.Fitted('ridge', predict, {
            'storage': {'float64_coordinates': len(actions) * d * d,
                        'parameter_bytes': 8 * len(actions) * d * d},
            'selection_ledger': [{'lambda': .01, 'validation_mse': .123}],
            'all_training_pairs_exposed': True,
        })]
    monkeypatch.setattr(compact, 'fit_models', spy)
    result = evaluate_context(cell, config(), ['test'])
    np.testing.assert_array_equal(seen['x'], cell.z[cell.train])
    assert len(seen['x']) == 2 * paired.train.sum()
    for action, target in seen['targets'].items():
        np.testing.assert_array_equal(target, cell.z[cell.targets[action][cell.train]])
        mask = cell.validation & (cell.states != cell.states[cell.targets[action]])
        np.testing.assert_array_equal(seen['validation'][action][0], cell.z[mask])
        np.testing.assert_array_equal(seen['validation'][action][1], cell.z[cell.targets[action][mask]])
    reference = next(m for m in result['methods'] if m['name'] == 'coordinate_fitted_operator_context')
    assert reference['metadata']['training_objective'] == 'state_coordinates'
    assert reference['metadata']['context_targets_used_for_fitting_or_selection'] is False
    assert reference['metadata']['selection_ledger'] == [{'lambda': .01, 'validation_mse': .123}]
    action = cell.generators[0]
    test = np.asarray([r['split'] == 'test' for r in paired.rows])
    expected = np.mean(np.sum((2 * paired.u[test] - paired.u[paired.targets[action][test]]) ** 2, axis=1))
    expected /= result['train_energy']
    actual = reference['splits']['test']['per_action'][action]['mean_normalized_squared_error']
    assert actual == pytest.approx(expected)


def test_context_linear_coefficient_norms_match_fitted_operators():
    cell = fixture_cell()
    paired = pair_context(cell)
    cfg = {**config(), 'lambda_grid': [0.]}
    result = evaluate_context(cell, cfg, ['pilot'])
    linear = next(m for m in result['methods'] if m['name'] == 'linear_transport_context')
    expected_squared = 0.
    for action in cell.elements:
        if action == cell.identity:
            continue
        matrix = np.linalg.lstsq(paired.u[paired.train],
                                paired.u[paired.targets[action][paired.train]], rcond=1e-12)[0]
        expected = np.linalg.norm(matrix, ord='fro')
        assert linear['metadata']['per_action_coefficient_frobenius_norm'][action] == pytest.approx(expected)
        expected_squared += expected ** 2
    assert linear['metadata']['coefficient_frobenius_norm'] == pytest.approx(np.sqrt(expected_squared))
