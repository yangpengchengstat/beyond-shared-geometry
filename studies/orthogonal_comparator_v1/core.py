"""Independent O(r) comparison on a complete uniform bank; PREPARED, NOT RUN.

Calling analyze_bank performs numerical research computation. Importing has no
IO or analysis side effects. The caller binds real banks to archived identities.
No group-law constraint is imposed on the separately chosen Q_g matrices.
"""
from __future__ import annotations
import math
import numpy as np

MATRIX_TOL = 1e-8
SCALAR_ATOL = 1e-8
SCALAR_RTOL = 1e-7
RANK_RATIO = 1e-10


class ComparatorError(ValueError):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status, self.reason = status, reason


def scalar_tolerance(a, b=0.0):
    return SCALAR_ATOL + SCALAR_RTOL * max(abs(float(a)), abs(float(b)))


def finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


class Checks:
    def __init__(self, records):
        self.records = records

    def condition(self, name, passed, **details):
        self.records[name] = {'passed': bool(passed), **details}
        if not passed:
            raise ComparatorError('not_evaluable_check', name)

    def scalar(self, name, a, b):
        delta, tol = abs(float(a) - float(b)), scalar_tolerance(a, b)
        self.condition(name, math.isfinite(delta) and delta <= tol,
                       observed=finite_or_none(a), reference=finite_or_none(b),
                       absolute_difference=finite_or_none(delta), tolerance=finite_or_none(tol))

    def matrix(self, name, a, b):
        residual = float(np.linalg.norm(a-b, 'fro') /
                         max(1.0, np.linalg.norm(a, 'fro'), np.linalg.norm(b, 'fro')))
        self.condition(name, math.isfinite(residual) and residual <= MATRIX_TOL,
                       residual=finite_or_none(residual), tolerance=MATRIX_TOL)

    def le(self, name, a, b):
        tol = scalar_tolerance(a, b)
        self.condition(name, math.isfinite(float(a)) and math.isfinite(float(b)) and a <= b + tol,
                       left=finite_or_none(a), right=finite_or_none(b), tolerance=finite_or_none(tol))


def require_uniform(weights, size, name):
    if weights is None:
        return
    raw = np.asarray(weights)
    if raw.shape != (size,) or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise ComparatorError('not_evaluable_weight_contract', name + ' weights invalid')
    # Exact declared uniform weights; never renormalize supplied weights or masks.
    if not np.array_equal(raw, np.full(size, 1.0 / size)):
        raise ComparatorError('not_evaluable_weight_contract', name + ' weights are not the declared uniform weights')


def analyze_bank(Z, actions, *, expected_action_count, identity_index,
                 context_weights=None, state_weights=None, action_weights=None,
                 include_identity=True, include_fixed_points=True):
    """Compute independent orthogonal and OLS finite-bank risks.

    Z has shape (contexts, states, retained coordinates). No additional centering,
    jitter, clipping, regularization, rank reduction or pseudoinverse is allowed.
    The second-moment construction/whitening and tolerances match the retained
    spectral kernel. The statistical object is the current empirical bank, not
    learned prediction on another bank. Q_g belongs to O(r), not SO(r).
    """
    summary = {'status': 'not_evaluable_input', 'checks': {}, 'diagnostics': {},
               'per_action': [], 'extra_centering': False, 'jitter': False,
               'eigenvalue_clipping': False, 'singular_value_clipping': False,
               'eta_clipping': False, 'pseudoinverse': False, 'regularization': False,
               'automatic_rank_reduction': False, 'PCA_refit': False,
               'orthogonal_class': 'O(r), determinant unconstrained',
               'group_laws_imposed': False, 'witness_trained': False,
               'native_causal_inference': False, 'population_certificate': False,
               'numerical_intervals_are_rigorous': False}
    arrays = {}
    checks = Checks(summary['checks'])
    try:
        raw = np.asarray(Z)
        if raw.ndim != 3 or min(raw.shape, default=0) < 1 or raw.dtype.kind not in 'fiu' or np.iscomplexobj(raw) or not np.isfinite(raw).all():
            raise ComparatorError('not_evaluable_input', 'Z must be a finite real complete (context,state,coordinate) array')
        Z = np.array(raw, dtype=np.float64, order='C', copy=True)
        n, p, r = Z.shape
        if not np.isfinite(Z).all():
            raise ComparatorError('not_evaluable_input', 'nonfinite float64 conversion')
        actions = np.asarray(actions)
        if actions.dtype.kind not in 'iu' or actions.shape != (expected_action_count, p):
            raise ComparatorError('not_evaluable_action_contract', 'action count/state shape/integer type mismatch')
        if not isinstance(identity_index, int) or isinstance(identity_index, bool) or not 0 <= identity_index < expected_action_count:
            raise ComparatorError('not_evaluable_action_contract', 'invalid identity index')
        if include_identity is not True or include_fixed_points is not True:
            raise ComparatorError('not_evaluable_weight_contract', 'identity and all fixed points are mandatory')
        if any(not np.array_equal(np.sort(a), np.arange(p)) for a in actions):
            raise ComparatorError('not_evaluable_action_contract', 'every action must permute all states')
        if not np.array_equal(actions[identity_index], np.arange(p)):
            raise ComparatorError('not_evaluable_action_contract', 'declared identity missing')
        keys = {tuple(a) for a in actions.tolist()}
        if len(keys) != expected_action_count:
            raise ComparatorError('not_evaluable_action_contract', 'duplicate action permutations')
        if any(tuple(a[b]) not in keys for a in actions for b in actions):
            raise ComparatorError('not_evaluable_action_contract', 'declared complete action image is not closed')
        require_uniform(context_weights, n, 'context')
        require_uniform(state_weights, p, 'state')
        require_uniform(action_weights, expected_action_count, 'action')
        summary.update(n_contexts=n, n_states=p, r=r, group_order=expected_action_count,
                       state_weight='uniform', context_weight='uniform', action_weight='uniform',
                       identity_index=identity_index, include_identity=True, include_fixed_points=True)
        summary['checks']['complete_uniform_action_contract'] = {'passed': True}
        flat = Z.reshape(-1, r)
        with np.errstate(over='ignore', invalid='ignore'):
            Sigma_raw = flat.T @ flat / len(flat)
        checks.condition('finite_second_moment', np.isfinite(Sigma_raw).all())
        checks.matrix('second_moment_symmetry', Sigma_raw, Sigma_raw.T)
        Sigma = (Sigma_raw + Sigma_raw.T) / 2
        w, U = np.linalg.eigh(Sigma)
        checks.condition('finite_second_moment_eigensystem', np.isfinite(w).all() and np.isfinite(U).all())
        smallest, largest = float(w[0]), float(w[-1])
        summary['diagnostics'].update(sigma_eigenvalues=w.tolist(), sigma_min=smallest,
                                      sigma_max=largest,
                                      sigma_min_max_ratio=smallest/largest if largest > 0 else None)
        arrays.update(Sigma_raw=Sigma_raw, Sigma=Sigma, sigma_eigenvalues=w)
        full_rank = largest > 0 and smallest > RANK_RATIO * largest
        summary['checks']['second_moment_rank'] = {'passed': bool(full_rank), 'required_min_max_ratio_strictly_above': RANK_RATIO}
        if not full_rank:
            raise ComparatorError('not_evaluable_rank', 'empirical second moment fails original fixed-rank rule')
        whitening = (U * (w ** -0.5)) @ U.T
        X = Z @ whitening
        Xf = X.reshape(-1, r)
        checks.matrix('whitening_identity', Xf.T @ Xf / len(Xf), np.eye(r))
        arrays['whitening'] = whitening
        sigmas, matrices, orthogonal_maps = [], [], []
        eta_direct, eta_svd, orth_direct, orth_svd, beta_svd = [], [], [], [], []
        identity_losses = []
        for g, action in enumerate(actions):
            Xg = X[:, action, :].reshape(-1, r)
            checks.matrix(f'target_second_moment_{g}', Xg.T @ Xg / len(Xg), np.eye(r))
            M = Xf.T @ Xg / len(Xf)
            checks.condition(f'finite_cross_moment_{g}', np.isfinite(M).all())
            left, singular, right_t = np.linalg.svd(M, full_matrices=False)
            checks.condition(f'finite_cross_svd_{g}', np.isfinite(singular).all())
            checks.le(f'cross_singular_max_{g}', float(singular[0]), 1.0)
            checks.le(f'cross_singular_min_{g}', 0.0, float(singular[-1]))
            Q = left @ right_t  # O(r), without any determinant correction.
            checks.matrix(f'orthogonality_{g}', Q.T @ Q, np.eye(r))
            checks.matrix(f'svd_reconstruction_{g}', (left * singular) @ right_t, M)
            if g == identity_index:
                checks.matrix('identity_cross_moment', M, np.eye(r))
                checks.matrix('identity_orthogonal_map', Q, np.eye(r))
            e = float(np.mean(np.sum((Xf @ M-Xg)**2, axis=1)) / r)
            o = float(np.mean(np.sum((Xf @ Q-Xg)**2, axis=1)) / r)
            # (1-s)(1+s) avoids subtracting two close squared values. Raw singular
            # values are retained; neither the singular spectrum nor eta is clipped.
            es = float(np.sum((1.0-singular)*(1.0+singular)) / r)
            os = float(2.0*np.sum(1.0-singular) / r)
            bs = float(np.sum((1.0-singular)**2) / r)
            checks.scalar(f'eta_direct_svd_{g}', e, es)
            checks.scalar(f'orth_direct_svd_{g}', o, os)
            checks.scalar(f'orth_minus_eta_svd_{g}', o-e, bs)
            checks.le(f'ols_le_orth_{g}', e, o)
            checks.le(f'orth_le_twice_ols_{g}', o, 2.0*e)
            identity_losses.append(float(np.mean(np.sum((Xf-Xg)**2, axis=1)) / r))
            eta_direct.append(e); eta_svd.append(es)
            orth_direct.append(o); orth_svd.append(os); beta_svd.append(bs)
            sigmas.append(singular); matrices.append(M); orthogonal_maps.append(Q)
            summary['per_action'].append({'action_index': g, 'state_permutation': action.tolist(),
                'is_identity': g == identity_index, 'fixed_point_count': int(np.sum(action == np.arange(p))),
                'singular_values': singular.tolist(), 'eta_direct': e, 'eta_svd': es,
                'L_orth_direct': o, 'L_orth_svd': os, 'beta_direct': o-e,
                'beta_svd': bs, 'Q_determinant': float(np.linalg.det(Q)),
                'Q_completion_unique_under_rank_rule': bool(singular[-1] > RANK_RATIO*max(1.0,float(singular[0])))})
        eta = float(np.mean(eta_direct)); orth = float(np.mean(orth_direct))
        beta = orth-eta; beta_stable = float(np.mean(beta_svd))
        es = float(np.mean(eta_svd)); os = float(np.mean(orth_svd))
        identity = float(np.mean(identity_losses))
        checks.scalar('eta_direct_svd', eta, es)
        checks.scalar('orth_direct_svd', orth, os)
        checks.scalar('beta_direct_svd', beta, beta_stable)
        checks.le('eta_le_L_orth', eta, orth)
        checks.le('L_orth_le_2_eta', orth, 2.0*eta)
        checks.le('L_orth_le_identity_action', orth, identity)
        checks.le('L_orth_le_2', orth, 2.0)
        checks.le('eta_nonnegative', 0.0, eta)
        checks.le('eta_at_most_1', eta, 1.0)
        # A tolerance-passing eta outside [0,1] is NOT clipped to define sqrt.
        beta_floor = None
        beta_floor_status = 'unavailable_raw_eta_outside_exact_unit_interval'
        if 0.0 <= eta <= 1.0:
            beta_floor = eta*eta / (1.0+math.sqrt(1.0-eta))**2
            beta_floor_status = 'computed_stable_eta_formula'
            checks.le('eta_forced_orthogonality_floor', beta_floor, beta)
        summary['metrics'] = {'eta': eta, 'eta_definition': 'direct_independent_OLS_residual',
            'eta_svd': es, 'L_orth': orth, 'L_orth_svd': os, 'beta': beta,
            'beta_svd': beta_stable, 'beta_eta_floor': beta_floor,
            'beta_eta_floor_status': beta_floor_status,
            'eta_above_tolerance': bool(eta > scalar_tolerance(eta)),
            'beta_eta_floor_above_tolerance': None if beta_floor is None else bool(beta_floor > scalar_tolerance(beta_floor)),
            'L_identity_direct': identity,
            'beta_above_tolerance': bool(beta > scalar_tolerance(beta)),
            'beta_comparison_tolerance': scalar_tolerance(beta)}
        arrays.update(M=np.asarray(matrices), Q=np.asarray(orthogonal_maps),
                      cross_singular_values=np.asarray(sigmas), actions=actions)
        summary.update(status='complete', interpretation='Same-bank independent O(r) optimum; matrices are not asserted to satisfy group laws or predict new contexts.')
    except ComparatorError as exc:
        summary.update(status=exc.status, reason=exc.reason)
    except (ValueError, TypeError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as exc:
        summary.update(status='not_evaluable_numerics', reason=f'{type(exc).__name__}: {exc}')
    return summary, arrays
