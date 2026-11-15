#!/usr/bin/env python3
"""Independently refit exported coordinates; NumPy only, never worker execution.

Input must first pass import_results.py. This additional audit verifies selected
estimators conditional on coordinates and row metadata, not model forwards/PCA,
Torch joint/potential fits, every legacy secondary estimator, or paper eligibility.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import hashlib
from itertools import permutations
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

_LOCATION = Path(__file__).resolve()
_STUDY_ROOT = _LOCATION.parents[1]
# Check only ancestors inside this tool's own release layout, never search a
# personal directory. Both nested installed and standalone public releases have
# release/release_manifest.json at the same local publication root.
_RELEASE_CANDIDATES = tuple(_LOCATION.parents)[:4]
PROJECT_ROOT = next((p for p in _RELEASE_CANDIDATES
                     if (p/'release/release_manifest.json').is_file()
                     and (p/'paper/main.tex').is_file()), _STUDY_ROOT)
BANDWIDTHS = (.125, .25, .5, 1., 2., 4., 8., math.inf)
DEFAULT_GRID = (0., 1e-6, 1e-5, 1e-4, .001, .01, .1, 1.)
METHODS = ('generator_ridge', 'generator_input_only_field')


class AuditError(ValueError):
    pass


def bounded_path(path, project_root=PROJECT_ROOT):
    """Reject path escapes and symlink traversal before reading/writing data."""
    path = Path(os.path.abspath(path))
    root = Path(project_root).resolve()
    if not path.is_relative_to(root):
        raise AuditError('Path must be inside the authorized project')
    cursor = root
    for component in path.relative_to(root).parts:
        cursor = cursor/component
        # lstat-like symlink check happens before resolving any child target.
        if cursor.is_symlink():
            raise AuditError('Symlinks are forbidden for audit input/output paths')
    if not path.resolve().is_relative_to(root):
        raise AuditError('Resolved path escaped the authorized project')
    return path


def read_json(path):
    return json.loads(bounded_path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with bounded_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Checks:
    def __init__(self):
        self.count = 0
        self.max_absolute_error = 0.

    def equal(self, actual, expected, label):
        self.count += 1
        if isinstance(actual, dict):
            if not isinstance(expected, dict):
                raise AuditError(label + ': expected object')
            for key, value in actual.items():
                if key not in expected:
                    raise AuditError(label + ': missing ' + str(key))
                self.equal(value, expected[key], label + '/' + str(key))
        elif isinstance(actual, (list, tuple)):
            if not isinstance(expected, (list, tuple)) or len(actual) != len(expected):
                raise AuditError(label + ': sequence mismatch')
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.equal(a, b, label + '/' + str(i))
        elif isinstance(actual, (float, int, np.number)) and not isinstance(actual, bool):
            if not isinstance(expected, (float, int)) or isinstance(expected, bool):
                raise AuditError(label + ': nonnumeric value')
            if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
                raise AuditError(label + ': nonfinite result')
            delta = abs(float(actual) - float(expected))
            self.max_absolute_error = max(self.max_absolute_error, delta)
            if not math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=1e-10):
                raise AuditError(label + ': numerical mismatch')
        elif actual != expected:
            raise AuditError(label + ': value mismatch')


def action_catalog(domain):
    """Literal finite actions, independent of worker group code."""
    if domain == 'C7':
        elements = tuple(range(7)); identity = 0; generators = (1,)
        apply = lambda g, i: (i + g) % 7
    elif domain in ('D4', 'D4_corner'):
        elements = tuple((k, b) for k in range(4) for b in (0, 1))
        identity = (0, 0); generators = ((1, 0), (0, 1))
        apply = lambda g, i: ((1 - i if g[1] else i) + g[0]) % 4
    elif domain in ('S3', 'S3_role'):
        elements = tuple(permutations(range(3))); identity = (0, 1, 2)
        generators = ((1, 0, 2), (0, 2, 1))
        def apply(g, i):
            assignment = elements[i]
            transformed = tuple(assignment[g.index(role)] for role in range(3))
            return elements.index(transformed)
    else:
        raise AuditError('Unsupported action domain')
    states = 7 if domain == 'C7' else 4 if domain in ('D4', 'D4_corner') else 6
    tables = {g: tuple(apply(g, i) for i in range(states)) for g in elements}
    words = {identity: ()}; queue = deque([identity])
    while queue:
        current = queue.popleft()
        for generator in generators:
            table = tuple(tables[generator][tables[current][i]] for i in range(states))
            following = next(g for g in elements if tables[g] == table)
            if following not in words:
                words[following] = words[current] + (generator,); queue.append(following)
    return elements, identity, generators, tables, words


def paired_targets(rows, tables):
    lookup = {}; ids = set(); partition_keys = {}
    for i, row in enumerate(rows):
        required = ('row_id', 'family_id', 'split', 'state_id', 'paraphrase_id', 'orbit_id')
        if any(key not in row for key in required):
            raise AuditError('Incomplete exported row metadata')
        if row['row_id'] in ids:
            raise AuditError('Duplicate row identity')
        ids.add(row['row_id'])
        # S3 cross blocks intentionally reuse context families across entity
        # strata; a family/orbit pair must still stay in a single partition.
        family_orbit = (str(row['family_id']), str(row['orbit_id']))
        if family_orbit in partition_keys and partition_keys[family_orbit] != row['split']:
            raise AuditError('Family/orbit pair crosses partitions')
        partition_keys[family_orbit] = row['split']
        key = (*family_orbit, str(row['paraphrase_id']), int(row['state_id']))
        if key in lookup:
            raise AuditError('Duplicate context/state row')
        lookup[key] = i
    result = {}
    for g, table in tables.items():
        destination = []
        for row in rows:
            state = int(row['state_id'])
            if not 0 <= state < len(table):
                raise AuditError('Invalid source state')
            key = (str(row['family_id']), str(row['orbit_id']), str(row['paraphrase_id']), table[state])
            if key not in lookup:
                raise AuditError('Incomplete state orbit')
            target = lookup[key]
            if rows[target]['split'] != row['split']:
                raise AuditError('Action pair crosses partitions')
            destination.append(target)
        result[g] = np.asarray(destination, dtype=int)
    return result


def ridge(x, y, penalty):
    """Normal equations for positive ridge; independent least squares at zero."""
    if penalty == 0:
        return np.linalg.lstsq(x, y, rcond=1e-12)[0]
    regularizer = penalty * float(np.sum(x * x) / x.shape[1])
    return np.linalg.solve(x.T @ x + regularizer * np.eye(x.shape[1]), x.T @ y)


def select_ridge(z, rows, targets, actions, grid):
    states = np.asarray([int(r['state_id']) for r in rows])
    train = np.asarray([r['split'] == 'train' for r in rows])
    val = np.asarray([r['split'] == 'validation' for r in rows])
    valid_actions = [g for g in actions if np.any(val & (states != states[targets[g]]))]
    if not train.any() or not valid_actions:
        raise AuditError('Missing training or changed validation data')
    candidates = []; scores = []; norms = []
    for penalty in grid:
        fitted = {g: ridge(z[train], z[targets[g][train]], float(penalty)) for g in valid_actions}
        losses = []
        for g in valid_actions:
            mask = val & (states != states[targets[g]])
            losses.append(float(np.mean(np.sum((z[mask] @ fitted[g] - z[targets[g][mask]])**2, axis=1))))
        candidates.append(fitted); scores.append(float(np.mean(losses)))
        norms.append(sum(np.linalg.norm(c) for c in fitted.values()))
    eligible = [i for i, score in enumerate(scores) if score <= min(scores) + 1e-8]
    chosen = min(eligible, key=lambda i: norms[i])
    result = {g: ridge(z[train], z[targets[g][train]], float(grid[chosen])) for g in actions}
    return result, {'normalized_lambda': float(grid[chosen]),
                    'validation_scores': dict(zip(map(str, grid), scores)),
                    'action_count': len(valid_actions), 'affine': False}


class Field:
    def __init__(self, x, targets):
        self.x = np.asarray(x, dtype=float)
        self.displacements = {g: np.asarray(y, dtype=float) - self.x for g, y in targets.items()}
        # Explicit lexicographic upper triangle, independent of worker rank inversion.
        left, right = np.triu_indices(len(x), k=1)
        positions = np.unique(np.linspace(0, len(left)-1, min(4096, len(left)), dtype=np.int64))
        distances = np.linalg.norm(self.x[left[positions]] - self.x[right[positions]], axis=1)
        positive = distances[distances > 1e-12]
        self.scale = float(np.median(positive)) if len(positive) else 0.
        self.bandwidth = math.inf

    def predict(self, query, action, bandwidth=None):
        bandwidth = self.bandwidth if bandwidth is None else bandwidth
        result = np.asarray(query, dtype=float).copy()
        for start in range(0, len(result), 64):
            part = query[start:start+64]
            if math.isinf(bandwidth) or self.scale <= 0:
                weights = np.full((len(part), len(self.x)), 1/len(self.x))
            else:
                distances = np.sum((part[:, None, :] - self.x[None, :, :])**2, axis=2)
                logits = -distances / (2*(self.scale*bandwidth)**2)
                weights = np.exp(logits - logits.max(axis=1, keepdims=True))
                weights /= weights.sum(axis=1, keepdims=True)
            result[start:start+len(part)] += weights @ self.displacements[action]
        return result

    def select(self, validation):
        scores = [float(np.mean([np.mean((self.predict(x, g, bandwidth)-y)**2)
                  for g, (x, y) in validation.items()])) for bandwidth in BANDWIDTHS]
        selected = max(i for i, value in enumerate(scores) if value <= min(scores)+1e-8)
        self.bandwidth = BANDWIDTHS[selected]
        return {'bandwidth_grid': ['inf' if math.isinf(b) else b for b in BANDWIDTHS],
                'validation_losses': scores, 'selected': 'inf' if math.isinf(self.bandwidth) else self.bandwidth,
                'scale': self.scale, 'fit_scope': 'paired_training_arrays_only',
                'source_state_labels_in_fit_or_prediction': False,
                'selection_scope': 'changed_validation_actions_only',
                'retained_displacement_coordinates': sum(v.size for v in self.displacements.values())}


def farthest_first(x, count):
    count = min(int(count), len(x))
    if count < 1:
        raise AuditError('Support size must be positive')
    if count == len(x):
        return np.arange(count)
    selected = [0]
    while len(selected) < count:
        minimum = np.min(np.sum((x[:, None, :] - x[selected][None, :, :])**2, axis=2), axis=1)
        minimum[selected] = -math.inf
        selected.append(int(np.argmax(minimum)))
    return np.asarray(selected, dtype=int)


def compose(x, word, predictor):
    value = x.copy()
    for generator in word:
        value = predictor(value, generator)
    return value


def pack_family(values):
    return {'mean_loss': float(np.mean([v for items in values.values() for v in items])),
            'family_losses': {family: float(np.mean(items)) for family, items in values.items()}}


def generator_refit(z, rows, domain, grid):
    elements, identity, generators, tables, words = action_catalog(domain)
    targets = paired_targets(rows, tables)
    train = np.asarray([r['split'] == 'train' for r in rows]); val = np.asarray([r['split'] == 'validation' for r in rows])
    test = np.asarray([r['split'] == 'test' for r in rows]); states = np.asarray([int(r['state_id']) for r in rows])
    fitted, ridge_selection = select_ridge(z, rows, targets, generators, grid)
    validation = {g: (z[val & (states != states[targets[g]])], z[targets[g][val & (states != states[targets[g]])]]) for g in generators}
    field = Field(z[train], {g: z[targets[g][train]] for g in generators}); kernel_selection = field.select(validation)
    predictors = {'generator_ridge': lambda x, g: x @ fitted[g], 'generator_input_only_field': field.predict}
    energy = float(np.mean(np.sum((z[train]-z[train].mean(0))**2, axis=1)))
    if energy <= 1e-12:
        raise AuditError('Insufficient training energy')
    pooled = {kind: {m: defaultdict(list) for m in METHODS} for kind in ('primitive', 'unseen_composite')}
    per_action = []
    for g in elements:
        if g == identity:
            continue
        a = np.flatnonzero(test & (states != states[targets[g]])); b = targets[g][a]
        if not len(a):
            continue
        kind = 'primitive' if g in generators else 'unseen_composite'; losses = {}
        for method, predictor in predictors.items():
            values = np.sum((compose(z[a], words[g], predictor)-z[b])**2, axis=1)/energy
            losses[method] = float(np.mean(values))
            for i, value in zip(a, values):
                pooled[kind][method][rows[i]['family_id']].append(float(value))
        per_action.append({'element': list(g) if isinstance(g, tuple) else g,
                           'word': list(words[g]), 'kind': kind, 'n_pairs': len(a), 'losses': losses})
    diagnostic_words = {'r7': (1,)*7} if domain == 'C7' else {
        'r4': (generators[0],)*4, 's2': (generators[1],)*2,
        'srs': (generators[1], generators[0], generators[1]),
        'rsr': (generators[0], generators[1], generators[0]), 'rs': generators, 'sr': generators[::-1]}
    def word_element(word):
        table = tuple(range(len(tables[identity])))
        for generator in word:
            table = tuple(tables[generator][i] for i in table)
        return next(g for g in elements if tables[g] == table)
    diagnostics = {}; orders = {}
    for name, word in diagnostic_words.items():
        g = word_element(word); a = np.flatnonzero(test); b = targets[g][a]
        diagnostics[name] = {'word': list(word), 'element': list(g) if isinstance(g, tuple) else g,
                             'n_pairs': len(a), 'mean_losses': {m: float(np.mean(np.sum((compose(z[a], word, p)-z[b])**2, axis=1)/energy)) for m, p in predictors.items()}}
    if domain != 'C7':
        for name, right, wrong in [('relation_srs_vs_rsr', 'srs', 'rsr'), ('order_rs_vs_sr', 'rs', 'sr')]:
            rg = word_element(diagnostic_words[right]); wg = word_element(diagnostic_words[wrong])
            a = np.flatnonzero(test & (targets[rg] != targets[wg])); b = targets[rg][a]
            entry = {'n_pairs': len(a), 'coincident_endpoints_excluded': True, 'methods': {}}
            for method, predictor in predictors.items():
                values = {label: float(np.mean(np.sum((compose(z[a], diagnostic_words[word], predictor)-z[b])**2, axis=1)/energy)) for label, word in [('correct', right), ('wrong', wrong)]}
                entry['methods'][method] = {**values, 'correct_minus_wrong': values['correct']-values['wrong']}
            orders[name] = entry
    return {'mean_losses': {kind: {m: pack_family(values)['mean_loss'] for m, values in methods.items()} for kind, methods in pooled.items()},
            'family_losses': {kind: {m: pack_family(values)['family_losses'] for m, values in methods.items()} for kind, methods in pooled.items()},
            'per_action': per_action, 'diagnostics': diagnostics, 'order_tests': orders,
            'ridge_selection': ridge_selection, 'kernel_selection': kernel_selection,
            'storage': {'ridge_matrix_coordinates': sum(v.size for v in fitted.values()),
                        'kernel_source_coordinates': field.x.size,
                        'kernel_displacement_coordinates': sum(v.size for v in field.displacements.values()),
                        'storage_not_independent_parameter_count': True},
            'fit_action_count': len(generators), 'training_scale': energy,
            'intermediate_true_state_used': False, 'intermediate_projection_or_snapping': False}


def storage_refit(z, rows, domain, grid):
    elements, identity, generators, tables, words = action_catalog(domain)
    targets = paired_targets(rows, tables); train = np.asarray([r['split']=='train' for r in rows])
    val = np.asarray([r['split']=='validation' for r in rows]); test = np.asarray([r['split']=='test' for r in rows])
    states = np.asarray([int(r['state_id']) for r in rows]); x = z[train]
    energy = float(np.mean(np.sum((x-x.mean(0))**2, axis=1)))
    validation = {}
    for g in elements:
        mask = val & (states != states[targets[g]])
        if mask.any(): validation[g] = (z[mask], z[targets[g][mask]])
    curves = []
    for track in ('all_element', 'generator_only'):
        actions = elements if track == 'all_element' else generators
        matched = int(len(actions)*z.shape[1]/(len(actions)+1))
        budgets = sorted({min(len(x), k) for k in (8,16,32,64,128,256,len(x),matched)})
        for k in budgets:
            support = farthest_first(x, k)
            field = Field(x[support], {g: z[targets[g][train]][support] for g in actions})
            selection = field.select({g: pair for g, pair in validation.items() if g in actions})
            values = {kind: defaultdict(list) for kind in (('all_changed',) if track=='all_element' else ('primitive','unseen_composite'))}
            for g in elements:
                a = np.flatnonzero(test & (states != states[targets[g]])); b = targets[g][a]
                if not len(a): continue
                if track == 'all_element':
                    prediction = field.predict(z[a], g); kind = 'all_changed'
                else:
                    prediction = compose(z[a], words[g], field.predict)
                    kind = 'primitive' if g in generators else 'unseen_composite'
                losses = np.sum((prediction-z[b])**2, axis=1)/energy
                for i, loss in zip(a, losses): values[kind][rows[i]['family_id']].append(float(loss))
            coordinates = field.x.size + sum(d.size for d in field.displacements.values())
            curves.append({'track': track, 'K': k, 'full_training_support': k==len(x),
                           'coordinate_storage_matched_budget_K': matched,
                           'selected_training_indices': support.tolist(), 'selection': selection,
                           'counted_float64_coordinates': coordinates, 'counted_predictor_array_bytes': coordinates*8,
                           'linear_map_array_bytes': len(actions)*z.shape[1]**2*8,
                           'capacity_or_total_RAM_matched': False,
                           'results': {kind: pack_family(items) for kind, items in values.items()}})
    return curves


def direct_refit(z, rows, domain, grid, split):
    elements, _, _, tables, _ = action_catalog(domain)
    targets = paired_targets(rows, tables)
    fitted, selection = select_ridge(z, rows, targets, elements, grid)
    train = np.asarray([r['split']=='train' for r in rows]); test = np.asarray([r['split']==split for r in rows])
    states = np.asarray([int(r['state_id']) for r in rows]); energy = float(np.mean(np.sum((z[train]-z[train].mean(0))**2, axis=1)))
    values = []; families = defaultdict(list)
    for g in elements:
        a = np.flatnonzero(test & (states != states[targets[g]])); b = targets[g][a]
        if not len(a): continue
        loss = np.sum((z[a]@fitted[g]-z[b])**2, axis=1)/energy
        values.extend(loss.tolist())
        for i, value in zip(a, loss): families[str(rows[i]['family_id'])+'|'+str(rows[i]['orbit_id'])].append(float(value))
    if not values: raise AuditError('No changed evaluation pairs')
    return {'selection': selection, 'training_scale': energy, 'coordinate_dimension': z.shape[1],
            'mean_losses': {'primary_operator': float(np.mean(values))}}, {f: float(np.mean(v)) for f, v in families.items()}


def audit_raw(raw, config=None):
    raw = bounded_path(raw)
    config = config or read_json(raw/'provenance/study_config.json')
    if config.get('study_id') != 'icml2027-characterization-v1':
        raise AuditError('Wrong study identity')
    checks = Checks(); records = []
    analysis = bounded_path(raw/'analysis')
    # Bound each child before is_dir/is_file; glob must not traverse symlinked
    # analysis/cell directories even while discovering input filenames.
    folders = []
    for entry in sorted(analysis.iterdir()):
        entry = bounded_path(entry)
        if entry.is_dir():
            cell = bounded_path(entry/'legacy.json')
            if cell.is_file(): folders.append(entry)
    for folder in folders:
        cell = bounded_path(folder/'legacy.json')
        # Shared S3 metadata references must remain inside the same raw archive.
        if bounded_path(folder/'rows.json').is_file():
            row_path = folder/'rows.json'
        else:
            export = read_json(folder/'coordinate_export.json')
            relative = Path(export['rows_and_potential_archive_relative_directory'])
            if relative.is_absolute() or '..' in relative.parts:
                raise AuditError('Unsafe S3 metadata reference')
            row_path = raw/relative/'rows.json'
        row_path = bounded_path(row_path)
        rows = read_json(row_path)['rows']
        with np.load(bounded_path(folder/'coordinates.npz'), allow_pickle=False) as array:
            z = array['z']
        if z.shape != (len(rows), 16) or z.dtype != np.float64 or not np.isfinite(z).all():
            raise AuditError('Invalid exported float64 q16 coordinates')
        legacy = read_json(cell)
        if not isinstance(legacy, list) or not legacy:
            raise AuditError('Missing legacy split records')
        domain = legacy[0]['group']; grid = config['structural']['lambda_grid']
        for expected in legacy:
            if expected['group'] != domain: raise AuditError('Mixed groups within cell')
            direct, family = direct_refit(z, rows, domain, grid, expected['test_split'])
            checks.equal(direct, expected, folder.name+'/'+expected['test_split']+'/direct')
        coverage = ['direct_ridge_fit_selection_and_mean_all_exported_strata']
        if domain in ('C7','D4_corner'):
            fields = read_json(folder/'fields.json')
            gen = generator_refit(z, rows, domain, grid)
            checks.equal(gen, fields['generator'], folder.name+'/generator')
            curves = storage_refit(z, rows, domain, grid)
            checks.equal(curves, fields['input_comparisons']['storage_curves'], folder.name+'/storage')
            checks.equal(direct['mean_losses']['primary_operator'], fields['input_comparisons']['controls']['primary_operator']['mean_loss'], folder.name+'/primary_replay')
            coverage += ['primitive_only_ridge_and_kernel_selection', 'own_intermediate_composite_predictions',
                         'generator_per_action_pooled_family_losses', 'cycles_and_endpoint_discriminating_orders',
                         'all_element_and_generator_kernel_storage_curves_support_indices_and_bytes']
        records.append({'cell': folder.name, 'group': domain, 'strata': [x['test_split'] for x in legacy],
                        'coordinate_sha256': sha(folder/'coordinates.npz'), 'rows_sha256': sha(row_path),
                        'legacy_sha256': sha(cell),
                        'fields_sha256': sha(folder/'fields.json') if domain in ('C7','D4_corner') else None,
                        'coverage': coverage})
        print('REFIT_CELL '+folder.name, flush=True)
    if not records:
        raise AuditError('No cells to refit')
    return {'schema_version': 1, 'study_id': config['study_id'], 'status': 'verified_selected_coordinate_refits',
            'paper_eligible': False, 'formal_inference': False, 'cells_audited': len(records),
            'numeric_checks': checks.count, 'maximum_absolute_checked_difference': checks.max_absolute_error,
            'numpy_version': np.__version__, 'auditor_source_sha256': sha(Path(__file__)),
            'study_configuration_sha256': sha(raw/'provenance/study_config.json'), 'cells': records,
            'limitations': ['The Python API requires prior archive/config/completeness validation; the CLI additionally rechecks the complete import receipt and all raw-member hashes before and after refits.',
                           'Conditional on exported q16 coordinates and row metadata; raw model forwards and hidden-space PCA are not verified.',
                           'Torch joint and S3 potential fits are not independently reconstructed.',
                           'Oracle/decoded local fields, affine/Procrustes/residual controls, context-difference endpoints, bootstrap intervals and coherent-action references are not independently refitted here.',
                           'Coordinate storage is not statistical capacity, total RAM or training exposure.',
                           'No independent-test archive was available when this auditor was prepared; only model-free synthetic fixtures were executed.']}


def verify_receipt(raw):
    """Revalidate materialized bytes against the original complete import receipt."""
    raw = bounded_path(raw)
    receipt_path = bounded_path(raw.parent/'receipt.json')
    receipt = read_json(receipt_path)
    requirements = {'study_id': 'icml2027-characterization-v1',
                    'archive_integrity': 'passed', 'source_contract': 'matches_frozen_v1',
                    'panel_validation': 'complete', 'imported': True,
                    'import_status': 'imported_complete_pending_scientific_audit',
                    'paper_eligible': False}
    for key, expected in requirements.items():
        if receipt.get(key) != expected:
            raise AuditError('Complete validated import receipt required: '+key)
    expected_members = receipt.get('member_sha256')
    if not isinstance(expected_members, dict) or not expected_members:
        raise AuditError('Receipt lacks member fingerprints')
    actual = set(); pending = [raw]
    while pending:
        directory = bounded_path(pending.pop())
        for child in directory.iterdir():
            child = bounded_path(child)
            if child.is_dir(): pending.append(child)
            elif child.is_file(): actual.add(child.relative_to(raw).as_posix())
            else: raise AuditError('Unexpected raw input file type')
    if actual != set(expected_members):
        raise AuditError('Materialized input inventory differs from import receipt')
    for name, digest in expected_members.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise AuditError('Unsafe import receipt member path')
        if sha(raw/relative) != digest:
            raise AuditError('Imported member changed since validation: '+name)
    return {'receipt_sha256': sha(receipt_path),
            'original_archive_sha256': receipt.get('input_sha256'),
            'member_count': len(actual)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    raw = bounded_path(args.raw); output = bounded_path(args.output)
    if output.exists() or output.is_relative_to(raw):
        raise AuditError('Report output must be fresh and outside imported raw inputs')
    receipt_before = verify_receipt(raw)
    report = audit_raw(raw)
    if verify_receipt(raw) != receipt_before:
        raise AuditError('Import receipt changed while refitting')
    report['import_receipt_reverified_before_and_after'] = receipt_before
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print('REFIT_REPORT '+str(output), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (AuditError, KeyError, OSError, ValueError) as exc:
        print('coordinate-refit audit failed: '+str(exc), file=sys.stderr)
        raise SystemExit(2)
