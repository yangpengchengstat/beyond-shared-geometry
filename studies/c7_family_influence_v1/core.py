"""C7 family influence: compose immutable spectral and orthogonal kernels.

No fitting, centering, rank repair, registration search, or input-file writes.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

spectral = load('family_frozen_spectral', PROJECT/'studies/spectral_dimension_sensitivity_v1/core/_vendor/gamma_r_frozen.py')
orthogonal = load('family_frozen_orthogonal', PROJECT/'studies/orthogonal_comparator_v1/core.py')

def validate_contexts(Z, contexts):
    if Z.ndim != 3 or Z.shape[1] != 7 or len(contexts) != Z.shape[0]:
        raise ValueError('Complete C7 blocks and context length required')
    families, row_ids = {}, set()
    for index, context in enumerate(contexts):
        family, variant = context['family_id'], context['paraphrase_id']
        if not isinstance(family, str) or not family or type(variant) is not int or variant not in (0, 1):
            raise ValueError('Invalid family/paired variant')
        if context['split'] != 'test' or context['state_ids'] != list(range(7)):
            raise ValueError('Complete test states in canonical order required')
        rows = context['row_ids']
        if len(rows) != 7 or len(set(rows)) != 7 or row_ids.intersection(rows):
            raise ValueError('Missing or duplicate row IDs')
        row_ids.update(rows)
        members = families.setdefault(family, {})
        if variant in members:
            raise ValueError('Duplicate variant within family')
        members[variant] = index
    if len(families) < 2 or any(set(v) != {0, 1} for v in families.values()):
        raise ValueError('Every family must have both variants; at least two families')
    return families

def delete_family(Z, contexts, family):
    families = validate_contexts(Z, contexts)
    if family not in families:
        raise ValueError('Unknown family')
    removed = sorted(families[family].values())
    keep = [i for i in range(len(contexts)) if i not in removed]
    result = np.array(Z[keep], copy=True)
    return result, {'deleted_family': family, 'removed_context_indices': removed,
        'removed_contexts': [contexts[i] for i in removed],
        'retained_context_indices': keep, 'retained_families': len(families)-1,
        'retained_contexts': len(keep), 'retained_states_per_context': 7,
        'state_weight': 'uniform', 'context_weight': 'uniform', 'action_weight': 'uniform'}

def analyze_bank(Z):
    s, sa = spectral.analyze_bank(Z, 'C7')
    actions = np.asarray(spectral.group_spec('C7')['actions'], dtype=int)
    o, oa = orthogonal.analyze_bank(Z, actions, expected_action_count=7, identity_index=0)
    out = {'status': 'not_evaluable_kernel', 'spectral': s, 'orthogonal': o,
           'checks': {}, 'metrics': {}, 'diagnostics': s.get('diagnostics', {}),
           'fresh_bank_whitening': True, 'model_forward': False, 'PCA_refit': False,
           'predictor_training': False, 'statistical_certificate': False}
    if s['status'] != 'complete' or o['status'] != 'complete':
        out['reason'] = 'spectral='+s['status']+';orthogonal='+o['status']
        return out
    checks = orthogonal.Checks(out['checks'])
    try:
        checks.matrix('same_second_moment', sa['Sigma'], oa['Sigma'])
        checks.matrix('same_whitening', sa['whitening'], oa['whitening'])
        sm, om = s['metrics'], o['metrics']
        checks.scalar('same_eta', sm['eta_LS'], om['eta'])
        trivial = sum(t['total_energy'] for t in s['types'] if t['real_dimension'] == 1)
        checks.scalar('identity_from_trivial_energy', 2*(1-trivial), om['L_identity_direct'])
        eta, orth, gamma = om['eta'], om['L_orth'], sm['gamma']
        lower, upper = sm['L_lower'], sm['L_upper']
        pminus, pplus = max(0., lower-eta), upper-eta
        ellminus, ellplus = max(0., lower-orth), upper-orth
        checks.le('orth_le_upper', orth, upper)
        checks.le('joint_interval_ordered', pminus, pplus)
        checks.le('law_interval_ordered', ellminus, ellplus)
        checks.le('law_upper_nonnegative', 0., ellplus)
        tol = orthogonal.scalar_tolerance(ellminus)
        out['metrics'] = {'eta': eta, 'L_orth': orth, 'beta': om['beta'], 'gamma': gamma,
            'L_lower': lower, 'L_upper': upper, 'p_minus': pminus, 'p_plus': pplus,
            'extra_law_lower': ellminus, 'extra_law_upper': ellplus,
            'extra_law_lower_above_tolerance': bool(ellminus > tol),
            'extra_law_comparison_tolerance': tol,
            'extra_law_lower_tolerance_multiple': ellminus/tol,
            'p_minus_above_tolerance': bool(pminus > orthogonal.scalar_tolerance(pminus)),
            'L_identity': om['L_identity_direct'], 'delta_type_normalized': sm['delta_type_normalized'],
            'total_trivial_energy': trivial}
        out['status'] = 'complete'
    except orthogonal.ComparatorError as exc:
        out.update(status='not_evaluable_cross_kernel_check', reason=str(exc))
    return out
