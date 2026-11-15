"""Context-difference transport controls using exported coordinates only.

This module fits on the declared train split and selects hyperparameters on
changed-state validation pairs. No model forward, old calibration, or hypothesis
promotion is performed. The returned statistics are descriptive.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class ContextPairs:
    u: np.ndarray
    centers: np.ndarray
    rows: list[dict[str, Any]]
    states: np.ndarray
    targets: dict[str, np.ndarray]
    train: np.ndarray
    validation: np.ndarray
    source_indices: np.ndarray


def pair_context(cell: Any) -> ContextPairs:
    """Pair exactly two ordered paraphrases; reject incomplete/ambiguous banks."""
    z = np.asarray(cell.z, dtype=np.float64)
    if z.ndim != 2 or len(z) != len(cell.rows) or not np.isfinite(z).all():
        raise ValueError('Context source coordinates must be a finite row matrix')
    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    ids: set[str] = set()
    for i, row in enumerate(cell.rows):
        required = ('family_id', 'orbit_id', 'state_id', 'paraphrase_id', 'split', 'row_id')
        if any(key not in row for key in required):
            raise ValueError('Missing context row metadata')
        if isinstance(row['state_id'], bool) or not isinstance(row['state_id'], (int, np.integer)):
            raise ValueError('Context state identity must be an integer')
        rid = str(row['row_id'])
        if rid in ids:
            raise ValueError('Duplicate source row identity')
        ids.add(rid)
        groups[(str(row['family_id']), str(row['orbit_id']), int(row['state_id']))].append(i)
    pairs = []
    pair_rows = []
    common_paraphrases = None
    lookup = {}
    partition_lookup = {}
    for key, indices in groups.items():
        if len(indices) != 2:
            raise ValueError('Each context/state needs exactly two paraphrases')
        # Paraphrase identifiers in the frozen export are integers. Lexical
        # fallback supports portable string identifiers with an explicit order.
        def order(i):
            value = cell.rows[i]['paraphrase_id']
            if isinstance(value, bool) or not isinstance(value, (str, int, np.integer)):
                raise ValueError('Paraphrase identifiers must be integers or strings')
            if isinstance(value, (int, np.integer)):
                return (0, int(value))
            try:
                return (0, int(value))
            except ValueError:
                return (1, str(value))
        indices = sorted(indices, key=order)
        paraphrases = tuple(order(i) for i in indices)
        if paraphrases[0] == paraphrases[1]:
            raise ValueError('Duplicate paraphrase identifier')
        if common_paraphrases is None:
            common_paraphrases = paraphrases
        elif common_paraphrases != paraphrases:
            raise ValueError('Paraphrase ordering must be consistent across states')
        left, right = (cell.rows[i] for i in indices)
        if left['split'] != right['split']:
            raise ValueError('A context difference crosses partitions')
        block = key[:2]
        if block in partition_lookup and partition_lookup[block] != left['split']:
            raise ValueError('Family/orbit block crosses partitions')
        partition_lookup[block] = left['split']
        record = dict(left)
        record.update(row_id=f"context:{left['row_id']}:{right['row_id']}",
                      source_row_ids=[left['row_id'], right['row_id']],
                      paraphrase_pair=[left['paraphrase_id'], right['paraphrase_id']])
        lookup[key] = len(pairs)
        pairs.append(indices)
        pair_rows.append(record)
    source = np.asarray(pairs, dtype=np.int64)
    if not len(source):
        raise ValueError('Empty context bank')
    targets = {}
    for action, table in cell.tables.items():
        selected = []
        for row, indices in zip(pair_rows, pairs):
            state = int(row['state_id'])
            if not 0 <= state < len(table):
                raise ValueError('Invalid context state')
            key = (str(row['family_id']), str(row['orbit_id']), int(table[state]))
            if key not in lookup:
                raise ValueError('Incomplete context state orbit')
            j = lookup[key]
            if pair_rows[j]['split'] != row['split']:
                raise ValueError('Context action crosses partitions')
            # The parent registration is authoritative: check both paraphrases
            # against it rather than quietly introducing a second action table.
            actual = tuple(int(cell.targets[action][i]) for i in indices)
            if actual != tuple(pairs[j]):
                raise ValueError('Context pairing disagrees with parent registration')
            selected.append(j)
        targets[action] = np.asarray(selected, dtype=np.int64)
    return ContextPairs(
        u=z[source[:, 0]] - z[source[:, 1]],
        centers=(z[source[:, 0]] + z[source[:, 1]]) * .5,
        rows=pair_rows,
        states=np.asarray([int(r['state_id']) for r in pair_rows], dtype=np.int64),
        targets=targets,
        train=np.asarray([r['split'] == 'train' for r in pair_rows]),
        validation=np.asarray([r['split'] == 'validation' for r in pair_rows]),
        source_indices=source,
    )


def _ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    if penalty < 0 or not np.isfinite(penalty):
        raise ValueError('Ridge penalty must be finite and nonnegative')
    if penalty == 0:
        return np.linalg.lstsq(x, y, rcond=1e-12)[0]
    scale = float(np.sum(x * x) / x.shape[1])
    return np.linalg.solve(x.T @ x + penalty * scale * np.eye(x.shape[1]), x.T @ y)


def _select(candidates: list[tuple[float, float, Any]]) -> tuple[float, float, Any]:
    if not candidates or any(not np.isfinite(c[0]) for c in candidates):
        raise ValueError('Hyperparameter selection contains invalid validation losses')
    # The objective is unrounded validation MSE; exact ties retain grid order.
    return min(candidates, key=lambda item: item[0])


def _training_centroids(data: ContextPairs, state_count: int) -> np.ndarray:
    result = []
    for state in range(state_count):
        selected = data.train & (data.states == state)
        if not selected.any():
            raise ValueError('Training category decoder is missing a state')
        result.append(data.centers[selected].mean(axis=0))
    return np.asarray(result)


def predict_category(u: np.ndarray, centers: np.ndarray, alphas: np.ndarray,
                     centroids: np.ndarray) -> np.ndarray:
    """Target-blind decoded arm: only source u and paired source center enter."""
    labels = np.argmin(np.sum((centers[:, None] - centroids[None]) ** 2, axis=2), axis=1)
    return np.asarray(u) * alphas[labels, None]


def _category_alphas(data: ContextPairs, action: str, penalty: float,
                     state_count: int, energy: float) -> np.ndarray:
    values = []
    for state in range(state_count):
        selected = data.train & (data.states == state)
        x = data.u[selected]
        y = data.u[data.targets[action][selected]]
        # Ridge toward the identity retention factor 1. The regularizer uses
        # train-only energy and cell count, and has no target/test dependence.
        regularizer = penalty * energy * len(x)
        denominator = float(np.sum(x * x)) + regularizer
        alpha = ((float(np.sum(x * y)) + regularizer) / denominator
                 if denominator > 0 else 1.)
        values.append(alpha)
    return np.asarray(values)


def _summary(losses: list[float], norm_ratios: list[float]) -> dict[str, Any]:
    loss = np.asarray(losses, dtype=float)
    norms = np.asarray(norm_ratios, dtype=float)
    invalid = int(np.count_nonzero(~np.isfinite(loss)))
    n = len(loss)
    return {
        'n': n,
        'invalid_predictions': invalid,
        # Invalid predictions make the primary mean unavailable; no successful
        # subset is silently reported as the full method's score.
        'mean_normalized_squared_error': (float(np.mean(loss)) if n and not invalid else None),
        'mean_output_squared_norm_over_train_energy': (
            float(np.mean(norms)) if n and np.isfinite(norms).all() else None),
    }


def _score(data: ContextPairs, actions: list[str], energy: float,
           predictor: Callable[[np.ndarray, str], np.ndarray],
           eval_splits: list[str]) -> dict[str, Any]:
    result = {}
    for split in eval_splits:
        action_records = {}
        families: dict[str, tuple[list[float], list[float]]] = {}
        orbits: dict[str, tuple[list[float], list[float]]] = {}
        all_losses = []
        all_norms = []
        errors = []
        for action in actions:
            selected = np.asarray([
                r['split'] == split for r in data.rows
            ]) & (data.states != data.states[data.targets[action]])
            indices = np.flatnonzero(selected)
            if not len(indices):
                action_records[action] = _summary([], [])
                continue
            try:
                pred = np.asarray(predictor(indices, action), dtype=np.float64)
                if pred.shape != (len(indices), data.u.shape[1]):
                    raise ValueError('Prediction shape mismatch')
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                pred = np.full((len(indices), data.u.shape[1]), np.nan)
                errors.append({'action': action, 'exception': type(exc).__name__, 'detail': str(exc)})
            with np.errstate(over='ignore', invalid='ignore'):
                loss = np.sum((pred - data.u[data.targets[action][indices]]) ** 2, axis=1) / energy
                norms = np.sum(pred * pred, axis=1) / energy
            action_records[action] = _summary(loss.tolist(), norms.tolist())
            all_losses.extend(loss.tolist())
            all_norms.extend(norms.tolist())
            for index, value, norm in zip(indices, loss, norms):
                row = data.rows[index]
                family = str(row['family_id'])
                orbit = str(row['orbit_id'])
                for groups, key in ((families, family), (orbits, orbit)):
                    entry = groups.setdefault(key, ([], []))
                    entry[0].append(float(value))
                    entry[1].append(float(norm))
        family_summaries = {key: _summary(*values) for key, values in families.items()}
        orbit_summaries = {key: _summary(*values) for key, values in orbits.items()}
        summary = _summary(all_losses, all_norms)
        family_means = [row['mean_normalized_squared_error'] for row in family_summaries.values()]
        orbit_means = [row['mean_normalized_squared_error'] for row in orbit_summaries.values()]
        summary['equal_family_mean'] = (float(np.mean(family_means))
                                        if family_means and None not in family_means else None)
        summary['equal_orbit_mean'] = (float(np.mean(orbit_means))
                                       if orbit_means and None not in orbit_means else None)
        result[split] = {'pooled': summary, 'per_action': action_records,
                         'per_family': family_summaries, 'per_orbit': orbit_summaries,
                         'prediction_errors': errors}
    return result


def evaluate_context(cell: Any, config: dict[str, Any], eval_splits: list[str]) -> dict[str, Any]:
    """Fit all controls and report descriptive, split-preserving transport loss."""
    allowed = {'pilot', 'validation', 'test', 'test_joint', 'test_context', 'test_entity'}
    if len(set(eval_splits)) != len(eval_splits) or not set(eval_splits) <= allowed:
        raise ValueError('Context evaluation must explicitly use pilot, validation, or test strata')
    data = pair_context(cell)
    present_splits = {r['split'] for r in data.rows}
    if not set(eval_splits) <= present_splits:
        raise ValueError('Requested context evaluation partition is absent')
    if not data.train.any() or not data.validation.any():
        raise ValueError('Context controls require nonempty train and validation splits')
    energy = float(np.mean(np.sum(data.u[data.train] ** 2, axis=1)))
    if not np.isfinite(energy) or energy <= np.finfo(float).tiny:
        raise ValueError('Zero/nonfinite training context energy; normalization is undefined')
    d = data.u.shape[1]
    actions = [a for a in cell.elements if a != cell.identity]
    state_count = len(next(iter(cell.tables.values())))
    train_x = data.u[data.train]
    targets = {a: data.u[data.targets[a][data.train]] for a in actions}
    validation = {}
    validation_indices = {}
    for action in actions:
        selected = data.validation & (data.states != data.states[data.targets[action]])
        if not selected.any():
            raise ValueError('Every fitted nonidentity action requires changed-state validation')
        validation_indices[action] = np.flatnonzero(selected)
        validation[action] = (data.u[selected], data.u[data.targets[action][selected]])
    reference = len(actions) * d * d
    scalar_grid = list(config.get('scalar_grid', [0., .25, .5, .75, 1., 1.25]))
    penalty_grid = list(config.get('lambda_grid', [0., 1e-5, .001, .1, 1.]))
    if (not scalar_grid or not penalty_grid or
            not np.isfinite(scalar_grid).all() or
            not np.isfinite(penalty_grid).all() or min(penalty_grid) < 0):
        raise ValueError('Invalid context hyperparameter grid')
    records = []

    def add(name, predictor, coordinates, metadata):
        stored = {'float64_coordinates': int(coordinates),
                  'parameter_bytes': int(coordinates) * 8,
                  'logical_serialized_parameter_bytes': int(coordinates) * 8,
                  'materialized_numeric_parameter_bytes': int(coordinates) * 8,
                  'identity_pruned_linear_reference_coordinates': reference,
                  'ratio_to_linear_reference': float(coordinates / reference),
                  'integer_state_or_action_table_entries': 0}
        combined = {'storage': stored, **metadata}
        records.append({'name': name, 'metadata': combined,
                        'validation_selected_predictor': _score(data, actions, energy, predictor, ['validation'])['validation'],
                        'splits': _score(data, actions, energy, predictor, eval_splits)})

    add('identity_context', lambda i, a: data.u[i], 0,
        {'input_information': 'source_context_difference_only', 'fit': 'none'})
    scalar = {}
    scalar_ledger = {}
    for action, (xv, yv) in validation.items():
        candidates = [(float(np.mean(np.sum((xv * alpha - yv) ** 2, axis=1))),
                       float(alpha), None) for alpha in scalar_grid]
        loss, alpha, _ = _select(candidates)
        scalar[action] = alpha
        scalar_ledger[action] = {'selected_alpha': alpha, 'validation_mse': loss,
                                'candidates': [{'alpha': c[1], 'validation_mse': c[0]}
                                               for c in candidates]}
    add('scalar_retention_context', lambda i, a: data.u[i] * scalar[a], len(actions),
        {'input_information': 'source_context_difference_only',
         'selection': 'separate scalar per action; changed-state validation; grid-order exact ties',
         'selection_ledger': scalar_ledger})
    matrices = {}
    linear_ledger = {}
    for action, (xv, yv) in validation.items():
        candidates = []
        for penalty in penalty_grid:
            matrix = _ridge(train_x, targets[action], float(penalty))
            loss = float(np.mean(np.sum((xv @ matrix - yv) ** 2, axis=1)))
            candidates.append((loss, float(penalty), matrix))
        loss, penalty, matrix = _select(candidates)
        matrices[action] = matrix
        linear_ledger[action] = {'selected_lambda': penalty, 'validation_mse': loss,
                                'candidates': [{'lambda': c[1], 'validation_mse': c[0]}
                                               for c in candidates]}
    add('linear_transport_context', lambda i, a: data.u[i] @ matrices[a], reference,
        {'input_information': 'source_context_difference_only',
         'all_training_pairs_exposed': True, 'selection_ledger': linear_ledger,
         'training_objective': 'context_differences',
         'selection_objective': 'context_difference_validation_mse',
         'coefficient_frobenius_norm': float(np.sqrt(sum(np.sum(matrix * matrix) for matrix in matrices.values()))),
         'per_action_coefficient_frobenius_norm': {a: float(np.linalg.norm(matrix, ord='fro')) for a, matrix in matrices.items()},
         'regularization': 'lambda * train_squared_frobenius / dimension; no bias'})
    # Original operator reference: fit and select on state coordinates z, then
    # apply that same zero-bias linear operator to u. This is deliberately not
    # the stronger transport competitor trained directly on context targets.
    try:
        from .compact import fit_models
    except ImportError:
        from compact import fit_models
    coordinate_train = np.asarray([r['split'] == 'train' for r in cell.rows])
    coordinate_validation = np.asarray([r['split'] == 'validation' for r in cell.rows])
    coordinate_targets = {a: cell.z[cell.targets[a][coordinate_train]] for a in actions}
    coordinate_valid_pairs = {}
    for action in actions:
        changed = coordinate_validation & (cell.states != cell.states[cell.targets[action]])
        coordinate_valid_pairs[action] = (cell.z[changed], cell.z[cell.targets[action][changed]])
    coordinate_models = fit_models(cell.z[coordinate_train], coordinate_targets,
                                  coordinate_valid_pairs, {**config, 'methods': ['ridge']})
    if len(coordinate_models) != 1 or coordinate_models[0].predict is None:
        raise ValueError('State-coordinate ridge reference was not fitted')
    coordinate_model = coordinate_models[0]
    if coordinate_model.metadata['storage']['float64_coordinates'] != reference:
        raise ValueError('State-coordinate ridge reference has unexpected storage')
    add('coordinate_fitted_operator_context',
        lambda i, a: coordinate_model.predict(data.u[i], a), reference,
        {**coordinate_model.metadata,
         'input_information': 'source_context_difference_only',
         'training_objective': 'state_coordinates',
         'selection_objective': 'state_coordinate_validation_mse',
         'context_targets_used_for_fitting_or_selection': False,
         'application': 'same zero-bias coordinate-trained operator applied to context difference',
         'reference_identity_pruned_ridge_float64_coordinates': reference})
    centroids = _training_centroids(data, state_count)
    for oracle in (False, True):
        selected_alphas = {}
        ledger = {}
        for action in actions:
            indices = validation_indices[action]
            yv = data.u[data.targets[action][indices]]
            candidates = []
            for penalty in penalty_grid:
                alphas = _category_alphas(data, action, float(penalty), state_count, energy)
                predicted = (data.u[indices] * alphas[data.states[indices], None] if oracle
                             else predict_category(data.u[indices], data.centers[indices], alphas, centroids))
                loss = float(np.mean(np.sum((predicted - yv) ** 2, axis=1)))
                candidates.append((loss, float(penalty), alphas))
            loss, penalty, alphas = _select(candidates)
            selected_alphas[action] = alphas
            ledger[action] = {'selected_lambda': penalty, 'validation_mse': loss,
                              'training_category_alphas': alphas.tolist(),
                              'candidates': [{'lambda': c[1], 'validation_mse': c[0]}
                                             for c in candidates]}
        if oracle:
            predictor = lambda i, a, weights=selected_alphas: data.u[i] * weights[a][data.states[i], None]
        else:
            predictor = lambda i, a, weights=selected_alphas: predict_category(
                data.u[i], data.centers[i], weights[a], centroids)
        add('oracle_category_retention_context' if oracle else 'decoded_category_retention_context',
            predictor, state_count * len(actions) + (0 if oracle else state_count * d),
            {'input_information': ('source_difference_plus_oracle_source_state_label' if oracle
                                   else 'source_difference_plus_paired_source_center'),
             'additional_information_arm': True,
             'oracle_state_labels_at_evaluation': oracle,
             'decoder': None if oracle else 'nearest training-only paired-source centroid; fixed state-index ties',
             'all_training_pairs_exposed': True,
             'selection_ledger': ledger,
             'scalar_fit': 'per-state training fit shrunk toward identity, penalty selected on validation'})
    if config.get('context_nonlinear', True):
        nonlinear_config = {**config, 'methods': ['rff', 'nystrom']}
        for fitted in fit_models(train_x, targets, validation, nonlinear_config):
            if fitted.predict is None:
                records.append({'name': f'{fitted.name}_context', 'metadata': fitted.metadata,
                                'splits': {}, 'status': 'infeasible',
                                'validation_selected_predictor': None})
                continue
            records.append({
                'name': f'{fitted.name}_context',
                'metadata': {**fitted.metadata, 'input_information': 'source_context_difference_only'},
                'validation_selected_predictor': _score(data, actions, energy,
                                 lambda i, a, model=fitted: model.predict(data.u[i], a), ['validation'])['validation'],
                'splits': _score(data, actions, energy,
                                 lambda i, a, model=fitted: model.predict(data.u[i], a), eval_splits),
            })
    split_counts = dict(Counter(str(r['split']) for r in data.rows))
    return {
        'definition': 'u = z(lower paraphrase ID) - z(higher paraphrase ID)',
        'normalization': 'mean training squared norm of u, without recentering',
        'train_energy': energy,
        'training': {'pairs': int(data.train.sum()), 'validation_pairs': int(data.validation.sum()),
                     'all_split_pair_counts': split_counts,
                     'nonidentity_actions': actions,
                     'identity_pruned_linear_reference_coordinates': reference,
                     'training_includes_fixed_state_pairs': True,
                     'selection_and_evaluation_exclude_fixed_state_pairs': True,
                     'support_and_reserved_used': False},
        'evaluation_splits': list(eval_splits),
        'validation_selected_predictor_role': 'post-selection descriptive validation diagnostics; not an unbiased heldout score',
        'aggregation': 'changed-state row/action pairs; pooled and equal family/orbit summaries; no significance tests',
        'methods': records,
        'technical_test': bool(config.get('technical_test', False)),
        'paper_eligible': False,
        'confirmation_authorized': False,
        'interpretation': 'Exploratory cached-coordinate context controls after prior test inspection. No model intervention or confirmatory promotion.',
    }
