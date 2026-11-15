"""CPU-only numerical extensions; no import-time experiments or model execution.

Witnesses use one train-bank empirical metric. Old-test prediction is evaluated
with that frozen metric and is not certified by the train-bank risk bounds.
Only C7 and the four-corner D4 action are supported.
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path
import sys
import numpy as np

CORE = Path(__file__).resolve().parents[1] / "spectral_finite_bank_v1"
sys.path.insert(0, str(CORE))
import gamma_r as spectral

RIDGE_ALPHA = 1e-3
SUPPORT_RELATIVE_THRESHOLD = 1e-10
SUPPORT_ABSOLUTE_THRESHOLD = 1e-12


class WitnessFailure(ValueError):
    pass


def _tol(a=0.0, b=0.0):
    return spectral.SCALAR_ATOL + spectral.SCALAR_RTOL * max(abs(a), abs(b))


def _matrix(name, A, B, checks):
    residual = spectral._matrix_residual(A, B)
    checks[name] = {"residual": residual if np.isfinite(residual) else None,
                    "tolerance": spectral.MATRIX_TOL,
                    "passed": bool(np.isfinite(residual) and residual <= spectral.MATRIX_TOL)}
    if not checks[name]["passed"]:
        raise WitnessFailure(name)


def _scalar(name, a, b, checks):
    residual = abs(float(a) - float(b)); tolerance = _tol(a, b)
    checks[name] = {"residual": residual, "tolerance": tolerance, "passed": bool(residual <= tolerance)}
    if not checks[name]["passed"]:
        raise WitnessFailure(name)


def _canonical_vector(projector):
    # The maximal diagonal provides a stable deterministic coordinate pivot.
    norms = np.sum(projector * projector, axis=0)
    index = int(np.argmax(norms)); norm = float(np.sqrt(norms[index]))
    if not np.isfinite(norm) or norm <= SUPPORT_ABSOLUTE_THRESHOLD:
        raise WitnessFailure("cannot_extract_declared_invariant_copy")
    vector = projector[:, index] / norm
    if vector[index] < 0:
        vector = -vector
    return vector


def _invariant_columns(K, spec, core_summary, arrays, checks):
    """Select whole irreps, including when a spectral boundary is degenerate.

    In C7 complex-type isotypic spaces, a vector and its rotation span a real
    two-dimensional copy. In D4 E, first choose a +1 reflection vector, whose
    quarter-turn spans the other direction. Picking arbitrary eigenvector pairs
    in a degenerate eigenspace would not have this guarantee.
    """
    r = core_summary["r"]; D = K.shape[0]
    actions = arrays["block_actions"]
    R = actions[1]
    S = actions[4] if spec["domain"] == "D4" else None
    columns = []
    for j, (kind, info) in enumerate(zip(spec["types"], core_summary["types"])):
        wanted = info["selected_dimension"]
        if wanted == 0:
            continue
        P = arrays[f"type_{j}_projector"]
        wp, Up = np.linalg.eigh(P)
        B = Up[:, wp > 0.5]
        raw = B.T @ K @ B
        _matrix(f"type_{j}_restricted_symmetry", raw, raw.T, checks)
        values, vectors = np.linalg.eigh((raw + raw.T) / 2)
        values = values[::-1]; vectors = vectors[:, ::-1]
        cursor, remaining = 0, wanted
        while remaining:
            if cursor >= len(values):
                raise WitnessFailure("insufficient_type_dimension")
            end = cursor + 1
            while end < len(values) and abs(values[end] - values[cursor]) <= _tol(values[end], values[cursor]):
                end += 1
            eigenspace = B @ vectors[:, cursor:end]
            cluster = eigenspace @ eigenspace.T
            d = kind["d"]
            take = min(remaining, end - cursor)
            if (end - cursor) % d or take % d:
                raise WitnessFailure("eigenvalue_cluster_splits_real_irrep")
            within = []
            for _ in range(take // d):
                Pleft = cluster.copy()
                if within:
                    U = np.column_stack(within)
                    Pleft -= U @ U.T
                Pleft = (Pleft + Pleft.T) / 2
                if d == 2 and spec["domain"] == "D4":
                    seed_projector = (Pleft + Pleft @ S) / 2
                else:
                    seed_projector = Pleft
                v = _canonical_vector(seed_projector)
                within.append(v)
                if d == 2:
                    w = R @ v
                    w -= v * np.dot(v, w)
                    norm = float(np.linalg.norm(w))
                    if not np.isfinite(norm) or norm <= SUPPORT_ABSOLUTE_THRESHOLD:
                        raise WitnessFailure("degenerate_rotation_copy")
                    within.append(w / norm)
            columns.extend(within)
            remaining -= take
            cursor = end
    C = np.column_stack(columns) if columns else np.zeros((D, 0))
    Q = C @ C.T
    _matrix("selected_basis_orthogonality", C.T @ C, np.eye(C.shape[1]), checks)
    _matrix("selected_projection_idempotence", Q @ Q, Q, checks)
    for g, action in enumerate(actions):
        _matrix(f"selected_projection_invariance_{g}", Q @ action, action @ Q, checks)
    _scalar("selected_budget_value", np.trace(K @ Q), core_summary["metrics"]["C_r"], checks)
    if C.shape[1] > r or C.shape[1] != core_summary["budget"]["used_dimension"]:
        raise WitnessFailure("selected_dimension_budget")
    return C, Q


def _risk(X, operators, actions):
    r = X.shape[-1]
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.mean([np.mean(np.sum((X @ A - X[:, action, :]) ** 2, axis=-1)) / r
                               for A, action in zip(operators, actions)]))
    if not np.isfinite(value):
        raise WitnessFailure("nonfinite_prediction_risk")
    return value


def witness_comparison(Ztrain, Ztest, domain, spec=None, ridge_alpha=RIDGE_ALPHA):
    """Fit a witness and OLS/Ridge on old training data; score frozen old test.

    Returns (JSON-safe summary, arrays). A numerical/identity/rank failure is a
    recorded not-evaluable result, never repaired by clipping or changed rank.
    The full uniform source/state/action grid includes identity and fixed points.
    """
    spec = spectral.group_spec(domain) if spec is None else copy.deepcopy(spec)
    arrays, checks = {}, {}
    result = {"status": "not_evaluable_witness", "domain": domain, "checks": checks,
              "new_context_validation": False, "statistical_certificate": False,
              "test_metric": "frozen_training_uncentered_second_moment",
              "bound_scope": "training bank only", "ridge_alpha": ridge_alpha}
    try:
        if spec.get("domain") != domain or domain not in ("C7", "D4"):
            raise WitnessFailure("unsupported_or_mismatched_domain")
        if ridge_alpha != RIDGE_ALPHA:
            raise WitnessFailure("ridge_alpha_differs_from_frozen_protocol")
        train, ca = spectral._analyze_spec(Ztrain, spec)
        result["train_spectral"] = train
        if train["status"] != "complete":
            result.update(status=train["status"], reason="training_spectral_contract_failed")
            return result, {"train_" + k: v for k, v in ca.items()}
        raw_test = np.asarray(Ztest)
        if raw_test.ndim != 3 or min(raw_test.shape) < 1 or raw_test.dtype.kind not in "fiu" or not np.isfinite(raw_test).all():
            raise WitnessFailure("invalid_test_array")
        with np.errstate(over="ignore", invalid="ignore"):
            Ztrain = np.asarray(Ztrain, dtype=np.float64); Ztest = np.asarray(raw_test, dtype=np.float64)
        if not np.isfinite(Ztest).all():
            raise WitnessFailure("test_array_not_finite_in_float64")
        if Ztest.shape[1:] != Ztrain.shape[1:]:
            raise WitnessFailure("train_test_state_or_coordinate_dimensions_differ")
        r = Ztrain.shape[-1]; m = spec["m"]
        actions = np.asarray(spec["actions"], dtype=int)
        Wwhite = ca["whitening"]
        with np.errstate(over="ignore", invalid="ignore"):
            X = Ztrain @ Wwhite; Xt = Ztest @ Wwhite
        if not np.isfinite(Xt).all():
            raise WitnessFailure("test_array_not_finite_in_training_metric")
        nrows = X.shape[0] * X.shape[1]
        V = np.stack([X[:, action, :] for action in actions], axis=2).reshape(nrows, m*r) / np.sqrt(m)
        W = V / np.sqrt(nrows)
        C, Q = _invariant_columns(ca["K"], spec, train, ca, checks)
        k = C.shape[1]
        arrays.update(selected_coefficient_basis=C, selected_coefficient_projection=Q)
        if k:
            lifted = W @ C
            gram = lifted.T @ lifted
            _matrix("lifted_gram_symmetry", gram, gram.T, checks)
            eig, U = np.linalg.eigh((gram + gram.T) / 2)
            arrays["selected_support_eigenvalues"] = eig
            if (not np.isfinite(eig).all() or eig[0] <= SUPPORT_ABSOLUTE_THRESHOLD or
                    eig[0] <= SUPPORT_RELATIVE_THRESHOLD * eig[-1]):
                raise WitnessFailure("selected_invariant_support_numerically_unresolved_no_directions_dropped")
            basis = lifted @ ((U * eig ** -0.5) @ U.T)
        else:
            basis = np.zeros((nrows, 0))
        _matrix("lifted_basis_orthogonality", basis.T @ basis, np.eye(k), checks)
        # Auxiliary trivial dimensions complete the representation dimension to r.
        Vbasis = np.zeros((nrows + r-k, r))
        Vbasis[:nrows, :k] = basis
        Vbasis[nrows:, k:] = np.eye(r-k)
        J = np.zeros_like(Vbasis); J[:nrows] = X.reshape(nrows, r) / np.sqrt(nrows)
        L, singular_values, Rt = np.linalg.svd(Vbasis.T @ J, full_matrices=True)
        alignment = L @ Rt
        operators = []
        for g, action in enumerate(actions):
            Ag = np.eye(r)
            if k:
                transformed = basis.reshape(X.shape[0], X.shape[1], k)[:, action, :].reshape(nrows, k)
                Ag[:k, :k] = basis.T @ transformed
                _matrix(f"lifted_intertwining_{g}", transformed, basis @ Ag[:k, :k], checks)
            operators.append(alignment.T @ Ag @ alignment)
        rho = np.asarray(operators)
        table = np.asarray(spec["multiplication"])
        _matrix("witness_identity", rho[spec["identity"]], np.eye(r), checks)
        for g in range(m):
            _matrix(f"witness_orthogonality_{g}", rho[g].T @ rho[g], np.eye(r), checks)
            for h in range(m):
                _matrix(f"witness_group_law_{g}_{h}", rho[g] @ rho[h], rho[table[h, g]], checks)
        train_risk = _risk(X, rho, actions)
        lower, upper = train["metrics"]["L_lower"], train["metrics"]["L_upper"]
        checks["witness_training_bound"] = {"passed": bool(lower-_tol(lower,train_risk) <= train_risk <= upper+_tol(upper,train_risk)),
            "risk": train_risk, "lower": lower, "upper": upper,
            "note": "same training bank and whitening; numerical residual check, not rigorous rounding enclosure"}
        if not checks["witness_training_bound"]["passed"]:
            raise WitnessFailure("witness_training_bound")
        Xf = X.reshape(-1,r)
        Aols = np.array([Xf.T @ X[:,a,:].reshape(-1,r) / len(Xf) for a in actions])
        Aridge = np.array([np.linalg.solve(Xf.T @ Xf/len(Xf) + ridge_alpha*np.eye(r), A) for A in Aols])
        inverse_white = np.linalg.inv(Wwhite)
        raw_operators = np.array([Wwhite @ A @ inverse_white for A in rho])
        arrays.update(rho=rho, raw_coordinate_operators=raw_operators,
            train_whitening=Wwhite, training_second_moment=ca["Sigma"],
            procrustes_alignment=alignment, procrustes_singular_values=singular_values,
            ols_operators=Aols, ridge_operators=Aridge,
            training_K=ca["K"])
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise WitnessFailure("nonfinite_witness_artifact")
        result.update(status="complete", n_train=Ztrain.shape[0], n_test=Ztest.shape[0], r=r,
            metrics={"train_gamma": train["metrics"]["gamma"], "train_L_lower": lower,
                "train_L_upper": upper, "train_witness_risk": train_risk,
                "test_witness_risk": _risk(Xt,rho,actions), "train_ols_risk": _risk(X,Aols,actions),
                "test_ols_risk": _risk(Xt,Aols,actions), "train_ridge_risk": _risk(X,Aridge,actions),
                "test_ridge_risk": _risk(Xt,Aridge,actions)},
            interpretation="Witness bounds checked on training bank only. Frozen old-test prediction compares estimators; "
            "it is retrospective, is not independent-context validation, and does not estimate a population exactness cost.")
    except (WitnessFailure, ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError) as exc:
        result.update(status="not_evaluable_witness", reason=str(exc))
    return result, arrays


def subgroup_specs(domain):
    """Enumerate distinct conjugate image subgroups, not all generator names.

    Representatives are deterministic lexicographically minimal conjugated action
    lists. C7 has 120 distinct order-seven image subgroups; corner D4 has three.
    Every representative changes its targets consistently; rankings are descriptive.
    """
    base = spectral.group_spec(domain)
    p = base["p"]
    original = np.asarray(base["actions"], dtype=int)
    found = {}
    for relabel in itertools.permutations(range(p)):
        relabel = np.asarray(relabel, dtype=int); inverse = np.argsort(relabel)
        actions = relabel[original[:, inverse]]
        key = tuple(sorted(tuple(row) for row in actions.tolist()))
        representative = tuple(tuple(row) for row in actions.tolist())
        if key not in found or representative < found[key]:
            found[key] = representative
    semantic = tuple(sorted(tuple(row) for row in original.tolist()))
    result = []
    for i, key in enumerate(sorted(found)):
        spec = copy.deepcopy(base); spec["actions"] = [list(row) for row in found[key]]
        spectral.validate_group(spec)
        result.append({"id": f"{domain}_image_{i:03d}", "semantic": key == semantic,
                       "image_permutations": [list(row) for row in key], "spec": spec})
    expected = {"C7":120,"D4":3}[domain]
    if len(result) != expected or sum(x["semantic"] for x in result) != 1:
        raise ValueError("subgroup enumeration count/semantic identity failed")
    return result


def alternative_comparison(Ztrain,Ztest,domain,subgroup):
    """One independently checkpointable image subgroup; all canonical grid entries."""
    spec = subgroup["spec"]
    result, arrays = witness_comparison(Ztrain,Ztest,domain,spec=spec)
    test_spectral, test_arrays = spectral._analyze_spec(Ztest,spec)
    witness_status = result["status"]
    if witness_status == "complete" and test_spectral["status"] != "complete":
        result["status"] = "completed_with_not_evaluable"
    result.update(subgroup_id=subgroup["id"], semantic_registration=subgroup["semantic"],
                  witness_status=witness_status, test_spectral_status=test_spectral["status"],
                  test_bank_spectral=test_spectral, p_value=None,
                  registration_changes_targets=True)
    arrays.update({"test_bank_"+k:v for k,v in test_arrays.items()})
    return result, arrays


def registration_ranks(summaries):
    """Descriptive semantic rank interval, all failed subgroups explicitly counted."""
    valid=[]
    for s in summaries:
        q=s.get("test_bank_spectral",{})
        if q.get("status")=="complete":
            valid.append((s["subgroup_id"],s["semantic_registration"],q["metrics"]["gamma"]))
    semantic=next((v for _,is_semantic,v in valid if is_semantic),None)
    output={"planned":len(summaries),"spectral_evaluable":len(valid),"failed_spectral":len(summaries)-len(valid),
            "p_value":None,"rank_direction":"larger gamma first","ties_use_frozen_scalar_tolerance":True}
    if semantic is None:
        return dict(output,status="semantic_registration_not_evaluable")
    better=sum(v > semantic+_tol(v,semantic) for _,_,v in valid)
    tied=sum(abs(v-semantic)<=_tol(v,semantic) for _,_,v in valid)
    complete = len(valid) == len(summaries)
    failed = len(summaries) - len(valid)
    return dict(output,status="complete_descriptive" if complete else "incomplete_descriptive",
                semantic_gamma=semantic,
                rank_low=better+1 if complete else None,
                rank_high=better+tied if complete else None,
                rank_among_evaluable_low=better+1,
                rank_among_evaluable_high=better+tied,
                all_planned_possible_rank_low=better+1,
                all_planned_possible_rank_high=better+tied+failed,
                failed_subgroups_have_unknown_positions=bool(failed),tied=tied)


def simulate_case(domain,mechanism,n_train=32,n_test=32,noise=0.0,seed=2026102900):
    """Fixed small mechanism constructions, not calibrated null hypotheses.

    M3 uses an oblique state/context direct sum and is an exact shared-action
    construction before noise. M2 is heterogeneous conjugation; its name does
    not certify that every generated finite instance excludes all shared actions.
    """
    if domain not in ("C7","D4") or mechanism not in ("M1","M2","M3"):
        raise ValueError("unsupported declared mechanism/domain")
    if n_train < 16 or n_test < 1 or noise not in (0.0,0.05):
        raise ValueError("simulation size/noise differs from supported protocol")
    rng=np.random.default_rng(seed); n=n_train+n_test;r=16;p=7 if domain=="C7" else 4
    if mechanism in ("M1","M2"):
        if domain=="C7":
            R=np.eye(r)
            for j in range(1,4):
                a=2*np.pi*j/7
                R[2*j-2:2*j,2*j-2:2*j]=[[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]
            starts=rng.normal(size=(n,r))
            Z=np.stack([starts@np.linalg.matrix_power(R,s) for s in range(p)],axis=1)
        else:
            a=rng.normal(size=(n,4))
            Z=np.array([[np.kron(np.eye(4)[s],a[c]) for s in range(p)] for c in range(n)])
        if mechanism=="M2":
            for c in range(n):
                O,_=np.linalg.qr(rng.normal(size=(r,r)))
                Z[c]=Z[c]@O
    else:
        # Orthonormal centered state code of dimension p-1; independent context.
        basis=np.linalg.qr(np.eye(p)[:,:p-1]-np.ones((p,p-1))/p)[0]
        context=rng.normal(size=(n,r-(p-1)))
        Z=np.array([np.c_[basis,np.broadcast_to(context[c],(p,len(context[c])))] for c in range(n)])
        shear=np.eye(r);shear[p-1:,:p-1]=0.3*rng.normal(size=(r-(p-1),p-1))
        Z=Z@shear
    if noise:
        Z=Z+noise*rng.normal(size=Z.shape)
    meta={"domain":domain,"mechanism":mechanism,"n_train":n_train,"n_test":n_test,"rank":r,
          "noise":noise,"seed":seed,"shared_action_before_noise":True if mechanism in ("M1","M3") else None,
          "context_specific_conjugations":mechanism == "M2",
          "mechanism_is_valid_negative_null":False,"M3_is_negative":False,
          "interpretation":"Construction-specific sensitivity demonstration; no false-positive/power calibration."}
    return Z[:n_train],Z[n_train:],meta
