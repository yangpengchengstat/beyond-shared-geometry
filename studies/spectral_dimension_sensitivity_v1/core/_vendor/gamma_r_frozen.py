"""Model-free finite-bank spectral diagnostic for the fixed C7/D4 protocol.

``analyze_bank(Z, domain)`` returns a JSON-safe summary and auditable NumPy
arrays. It does not read files, write results, download models, or fit a neural
network. The caller must separately verify row/state identities and provenance;
algebraic residuals cannot detect absent observations replaced by zero vectors.

All risks use the same complete, uniformly weighted empirical bank and its
uncentered second-moment metric. No confidence interval, causal certificate,
held-out prediction guarantee, or witness action is produced. The original
prototype is preserved separately; this implementation adds explicit validation.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np

PROTOCOL_ID = "spectral_finite_bank_v1_0.4.9"
MATRIX_TOL = 1e-8
SCALAR_ATOL = 1e-8
SCALAR_RTOL = 1e-7
RANK_RATIO = 1e-10
EIGEN_TOL = 1e-8


class ValidationFailure(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _scalar_tol(a: float, b: float) -> float:
    return SCALAR_ATOL + SCALAR_RTOL * max(abs(a), abs(b))


def _matrix_residual(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, "fro") /
                 max(1.0, np.linalg.norm(a, "fro"), np.linalg.norm(b, "fro")))


def _real_irreps(orders: tuple[int, ...], elements: list[tuple[int, ...]]) -> list[dict]:
    """Internal abelian construction, also used by analytic software fixtures.

    Only real and complex Schur types occur here. This is not a generic
    quaternionic character-projector implementation.
    """
    result, seen = [], set()
    for a in itertools.product(*(range(n) for n in orders)):
        if a in seen:
            continue
        conjugate = tuple((-x) % n for x, n in zip(a, orders))
        seen.update((a, conjugate))
        character = np.cos([2 * np.pi * sum(x * g / n for x, g, n in zip(a, e, orders))
                            for e in elements])
        real = a == conjugate
        result.append({"name": "trivial" if all(x == 0 for x in a) else
                       ("sign_" if real else "plane_") + "_".join(map(str, a)),
                       "d": 1 if real else 2,
                       "coefficient": 1.0 / len(elements),
                       "character": (character if real else 2 * character).tolist(),
                       "absent": False})
    return result


def _abelian_spec(orders: tuple[int, ...]) -> dict:
    elements = list(itertools.product(*(range(n) for n in orders)))
    index = {e: i for i, e in enumerate(elements)}
    table = [[index[tuple((a + b) % n for a, b, n in zip(g, h, orders))]
              for h in elements] for g in elements]
    return {"domain": "x".join(f"C{n}" for n in orders), "p": len(elements),
            "m": len(elements), "identity": 0, "elements": [list(e) for e in elements],
            "actions": table, "multiplication": table,
            "inverse": [next(h for h in range(len(elements)) if table[g][h] == 0)
                        for g in range(len(elements))],
            "types": _real_irreps(orders, elements)}


def group_spec(domain: str) -> dict:
    """Return the complete, JSON-safe canonical action contract.

    C7 uses state order 0,...,6. D4 uses state order 0,...,3 and element
    order R^k S^b (b=0,1 outside k=0,...,3), R(s)=s+1, S(s)=1-s.
    Product gh means first h then g; coefficient actions are opposite/row actions.
    """
    if domain == "C7":
        spec = _abelian_spec((7,))
    elif domain == "D4":
        elements = [(k, b) for b in (0, 1) for k in range(4)]
        index = {e: i for i, e in enumerate(elements)}
        table = [[index[((k + (-1) ** b * l) % 4, (b + a) % 2)]
                  for l, a in elements] for k, b in elements]
        types = []
        for a, b in itertools.product((1, -1), repeat=2):
            types.append({"name": f"chi_{'p' if a == 1 else 'm'}{'p' if b == 1 else 'm'}",
                          "d": 1, "coefficient": 1.0 / 8,
                          "character": [float(a ** k * b ** h) for k, h in elements],
                          "absent": a != b})
        types.append({"name": "E", "d": 2, "coefficient": 2.0 / 8,
                      "character": [float(2 * np.cos(np.pi * k / 2)) if b == 0 else 0.0
                                    for k, b in elements], "absent": False})
        spec = {"domain": domain, "p": 4, "m": 8, "identity": 0,
                "elements": [list(e) for e in elements],
                "actions": [[(k + (1 - s if b else s)) % 4 for s in range(4)]
                            for k, b in elements], "multiplication": table,
                "inverse": [next(h for h in range(8) if table[g][h] == 0) for g in range(8)],
                "types": types}
    else:
        raise ValueError("domain must be C7 or D4; no arbitrary/nonfaithful action is inferred")
    spec["group_product"] = "gh means first h, then g"
    spec["block_action"] = "(Lambda_t a)_k = a_(k t^-1); Lambda_g Lambda_h = Lambda_(h g)"
    return spec


def validate_group(spec: dict) -> None:
    """Validate explicit tables rather than infer an abstract group from its image."""
    try:
        p, m, e = spec["p"], spec["m"], spec["identity"]
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (p, m, e)):
            raise ValueError("integer group dimensions required")
        if min(p, m) <= 0 or e not in range(m):
            raise ValueError("invalid group dimensions/identity")
        actions = np.asarray(spec["actions"])
        table = np.asarray(spec["multiplication"])
        inverse = np.asarray(spec["inverse"])
        if (actions.shape != (m, p) or table.shape != (m, m) or inverse.shape != (m,) or
                actions.dtype.kind not in "iu" or table.dtype.kind not in "iu" or
                inverse.dtype.kind not in "iu"):
            raise ValueError("malformed group tables")
        if (not np.array_equal(np.sort(actions, axis=1), np.tile(np.arange(p), (m, 1))) or
                np.any(table < 0) or np.any(table >= m) or np.any(inverse < 0) or np.any(inverse >= m)):
            raise ValueError("nonbijective action or invalid multiplication indices")
        if not np.array_equal(actions[e], np.arange(p)):
            raise ValueError("identity action is not identity")
        if not np.array_equal(table[e], np.arange(m)) or not np.array_equal(table[:, e], np.arange(m)):
            raise ValueError("identity multiplication failed")
        for g in range(m):
            if table[g, inverse[g]] != e or table[inverse[g], g] != e:
                raise ValueError("inverse failed")
            for h in range(m):
                if not np.array_equal(actions[g, actions[h]], actions[table[g, h]]):
                    raise ValueError("state action does not respect multiplication")
                for k in range(m):
                    if table[table[g, h], k] != table[g, table[h, k]]:
                        raise ValueError("associativity failed")
        types = spec["types"]
        if not types or len({t["name"] for t in types}) != len(types):
            raise ValueError("empty or repeated irreducible types")
        for t in types:
            if not isinstance(t["d"], int) or isinstance(t["d"], bool) or t["d"] not in (1, 2):
                raise ValueError("only declared one/two-dimensional real types are supported")
            c = np.asarray(t["character"])
            if c.shape != (m,) or c.dtype.kind not in "fiu" or not np.isfinite(c).all():
                raise ValueError("invalid real character")
            if not np.isfinite(t["coefficient"]) or t["coefficient"] <= 0:
                raise ValueError("invalid projector coefficient")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValidationFailure("not_evaluable_input", "group_contract: " + str(error)) from error


def integer_budget(types: list[tuple[int, np.ndarray]], r: int) -> dict:
    """Exact finite enumeration DP over whole irreducible copies, never eigenvectors.

    At most two best distinct allocations per exact cost are retained to expose
    ties. Values are not rounded or clipped. Exact ties prefer less dimension,
    then the lexicographically smaller multiplicity tuple in fixed type order.
    Near ties use the declared scalar tolerance as a diagnostic only and do not
    alter which raw objective value is maximized.
    """
    if not isinstance(r, int) or isinstance(r, bool) or r < 1 or not types:
        raise ValueError("positive integer budget and nonempty types required")
    states = {0: [(0.0, ())]}
    for d, spectrum in types:
        lam = np.asarray(spectrum, dtype=np.float64)
        if (not isinstance(d, int) or isinstance(d, bool) or d < 1 or lam.ndim != 1 or
                not np.isfinite(lam).all() or np.any(lam[1:] > lam[:-1])):
            raise ValueError("invalid dimension or descending multiplicity spectrum")
        cumulative = np.r_[0.0, np.cumsum(lam)]
        candidates: dict[int, list] = {}
        for cost, solutions in states.items():
            for k in range(min(len(lam), (r - cost) // d) + 1):
                for score, allocation in solutions:
                    candidates.setdefault(cost + d * k, []).append(
                        (float(score + d * cumulative[k]), allocation + (k,)))
        states = {cost: sorted(values, key=lambda item: (-item[0], item[1]))[:2]
                  for cost, values in candidates.items()}
    alternatives = sorted([(score, cost, ks) for cost, values in states.items()
                           for score, ks in values], key=lambda item: (-item[0], item[1], item[2]))
    best, used, ks = alternatives[0]
    runner_up = alternatives[1][0] if len(alternatives) > 1 else None
    return {"value": best, "selected_copies": list(ks), "used_dimension": used,
            "unused_dimension": r - used,
            "exact_allocation_tie": runner_up is not None and runner_up == best,
            "near_allocation_tie": runner_up is not None and
                abs(best - runner_up) <= _scalar_tol(best, runner_up),
            "runner_up_value": runner_up,
            "tie_rule": "maximize raw value; exact ties: fewer dimensions, then lexicographic counts"}


class _Checks:
    def __init__(self, output: dict):
        self.output = output

    def condition(self, name: str, passed: bool, **details: Any) -> None:
        self.output[name] = {"passed": bool(passed), **details}
        if not passed:
            raise ValidationFailure("not_evaluable_numerics", name)

    def matrix(self, name: str, a: np.ndarray, b: np.ndarray) -> None:
        residual = _matrix_residual(a, b)
        self.condition(name, np.isfinite(residual) and residual <= MATRIX_TOL,
                       residual=residual if np.isfinite(residual) else None, tolerance=MATRIX_TOL)

    def scalar(self, name: str, a: float, b: float) -> None:
        a, b = float(a), float(b)
        residual, tolerance = abs(a - b), _scalar_tol(a, b)
        self.condition(name, np.isfinite(residual) and residual <= tolerance,
                       residual=residual if np.isfinite(residual) else None,
                       tolerance=tolerance if np.isfinite(tolerance) else None)

    def interval(self, name: str, value: float, low: float, high: float,
                 *, tolerance: float | None = None) -> None:
        value = float(value)
        tolerance = (_scalar_tol(value, low) if abs(value - low) < abs(value - high)
                     else _scalar_tol(value, high)) if tolerance is None else tolerance
        self.condition(name, np.isfinite(value) and low - tolerance <= value <= high + tolerance,
                       raw_value=value if np.isfinite(value) else None,
                       lower=low, upper=high, tolerance=float(tolerance))


def _analyze_spec(Z: Any, spec: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """Internal kernel; noncanonical specs are used only by software fixtures."""
    summary: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID, "status": "not_evaluable_input",
        "domain": spec.get("domain"), "checks": {}, "diagnostics": {},
        "numeric_intervals_are_rigorous": False,
        "provenance_verified_by_core": False,
        "scope": "complete uniform finite bank; empirical uncentered second-moment metric",
    }
    arrays: dict[str, np.ndarray] = {}
    checks = _Checks(summary["checks"])
    try:
        validate_group(spec)
        summary["checks"]["group_contract"] = {"passed": True}
        try:
            raw = np.asarray(Z)
        except (TypeError, ValueError) as error:
            raise ValidationFailure("not_evaluable_input", "Z is not a rectangular numeric array") from error
        if raw.ndim != 3 or min(raw.shape, default=0) < 1:
            raise ValidationFailure("not_evaluable_input", "Z must be nonempty (contexts, states, coordinates)")
        if raw.dtype.kind not in "fiu" or np.iscomplexobj(raw):
            raise ValidationFailure("not_evaluable_input", "Z must contain real numeric values; no complex/object conversion")
        if not np.isfinite(raw).all():
            raise ValidationFailure("not_evaluable_input", "Z contains nonfinite values")
        Z = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(Z).all():
            raise ValidationFailure("not_evaluable_input", "Z is not finite after float64 conversion")
        n, p, r = Z.shape
        if p != spec["p"]:
            raise ValidationFailure("not_evaluable_input", "state count does not match canonical domain")
        m, D = spec["m"], spec["m"] * r
        summary.update(n_contexts=n, n_states=p, r=r, group_order=m, orbit_dimension=D)
        summary["checks"]["real_finite_complete_array_shape"] = {"passed": True,
            "note": "array shape is not observation provenance or state-ID validation"}
        flat = Z.reshape(-1, r)
        with np.errstate(over="ignore", invalid="ignore"):
            Sigma_raw = flat.T @ flat / len(flat)
        arrays["Sigma_raw"] = Sigma_raw
        checks.condition("finite_second_moment", np.isfinite(Sigma_raw).all())
        checks.matrix("second_moment_symmetry", Sigma_raw, Sigma_raw.T)
        Sigma = (Sigma_raw + Sigma_raw.T) / 2
        arrays["Sigma"] = Sigma
        w, U = np.linalg.eigh(Sigma)
        arrays["sigma_eigenvalues"] = w
        checks.condition("finite_second_moment_eigensystem", np.isfinite(w).all() and np.isfinite(U).all())
        largest, smallest = float(w[-1]), float(w[0])
        ratio = smallest / largest if largest > 0 else None
        condition = largest / smallest if smallest > 0 else None
        condition_overflow = condition is not None and not np.isfinite(condition)
        summary["diagnostics"].update(sigma_eigenvalues=w.tolist(),
            sigma_max=largest, sigma_min=smallest,
            sigma_min_max_ratio=ratio,
            sigma_condition_number=None if condition_overflow else condition,
            sigma_condition_number_overflow=bool(condition_overflow))
        full_rank = largest > 0 and smallest > RANK_RATIO * largest
        summary["checks"]["second_moment_rank"] = {"passed": bool(full_rank),
            "required_min_max_ratio_strictly_above": RANK_RATIO}
        if not full_rank:
            raise ValidationFailure("not_evaluable_rank", "empirical second moment fails fixed full-rank rule")
        whitening = (U * (w ** -0.5)) @ U.T
        arrays["whitening"] = whitening
        X = Z @ whitening
        Xf = X.reshape(-1, r)
        whitened_moment = Xf.T @ Xf / len(Xf)
        arrays["whitened_second_moment"] = whitened_moment
        checks.matrix("whitening_identity", whitened_moment, np.eye(r))
        actions = np.asarray(spec["actions"], dtype=np.int64)
        table = np.asarray(spec["multiplication"], dtype=np.int64)
        inverse = np.asarray(spec["inverse"], dtype=np.int64)
        arrays.update(actions=actions, multiplication=table, inverse=inverse)
        V = np.stack([X[:, action, :] for action in actions], axis=2).reshape(-1, D) / np.sqrt(m)
        K_raw = V.T @ V / len(V)
        arrays["K_raw"] = K_raw
        checks.condition("finite_K", np.isfinite(K_raw).all())
        checks.matrix("K_symmetry", K_raw, K_raw.T)
        K = (K_raw + K_raw.T) / 2
        arrays["K"] = K
        spectrum_K = np.linalg.eigvalsh(K)[::-1]
        arrays["K_eigenvalues"] = spectrum_K
        checks.condition("finite_K_spectrum", np.isfinite(spectrum_K).all())
        summary["diagnostics"].update(K_min_eigenvalue=float(spectrum_K[-1]),
            K_max_eigenvalue=float(spectrum_K[0]), trace_K=float(np.trace(K)))
        checks.interval("K_psd", float(spectrum_K[-1]), 0.0, 1.0, tolerance=EIGEN_TOL)
        checks.interval("K_contraction", float(spectrum_K[0]), 0.0, 1.0, tolerance=EIGEN_TOL)
        checks.scalar("K_trace", float(np.trace(K)), r)
        # Lambda_t has block (k, k t^-1)=I; v Lambda_t has block g equal v_(g t).
        block_indices = np.array([table[np.arange(m), inverse[t]] for t in range(m)])
        Lambdas = np.array([np.kron(np.eye(m)[indices], np.eye(r)) for indices in block_indices])
        arrays["block_actions"] = Lambdas
        for g in range(m):
            # Exact permutations validate the full block action without dense cubic products.
            checks.condition(f"block_orthogonal_{g}",
                np.array_equal(np.sort(block_indices[g]), np.arange(m)))
            for h in range(m):
                checks.condition(f"block_law_{g}_{h}", np.array_equal(
                    block_indices[h][block_indices[g]], block_indices[table[h, g]]))
            checks.matrix(f"K_commutation_{g}", K @ Lambdas[g], Lambdas[g] @ K)
        projectors = []
        identity = np.eye(D)
        for j, t in enumerate(spec["types"]):
            P_raw = t["coefficient"] * np.einsum("g,gij->ij", t["character"], Lambdas)
            arrays[f"type_{j}_projector_raw"] = P_raw
            checks.matrix(f"projector_{j}_symmetry", P_raw, P_raw.T)
            P = (P_raw + P_raw.T) / 2
            checks.matrix(f"projector_{j}_idempotence", P @ P, P)
            projectors.append(P)
            arrays[f"type_{j}_projector"] = P
            for g in range(m):
                checks.matrix(f"projector_{j}_commutation_{g}", P @ Lambdas[g], Lambdas[g] @ P)
        checks.matrix("projector_completeness", sum(projectors, np.zeros_like(K)), identity)
        for j, P in enumerate(projectors):
            for k in range(j):
                checks.matrix(f"projector_orthogonality_{j}_{k}", P @ projectors[k], np.zeros_like(K))
        types, infos = [], []
        for j, (t, P) in enumerate(zip(spec["types"], projectors)):
            wp, Up = np.linalg.eigh(P)
            arrays[f"type_{j}_projector_eigenvalues"] = wp
            for k, value in enumerate(wp):
                checks.scalar(f"projector_{j}_eigenvalue_{k}", float(value), float(value > 0.5))
            B = Up[:, wp > 0.5]
            dim, d = B.shape[1], t["d"]
            checks.condition(f"type_{j}_dimension", dim > 0 and dim % d == 0,
                             observed=dim, real_irrep_dimension=d)
            restricted_raw = B.T @ K @ B
            checks.matrix(f"type_{j}_K_symmetry", restricted_raw, restricted_raw.T)
            restricted = (restricted_raw + restricted_raw.T) / 2
            eigenvalues = np.linalg.eigvalsh(restricted)[::-1]
            arrays[f"type_{j}_eigenvalues"] = eigenvalues
            pairs = eigenvalues.reshape(-1, d)
            for k, values in enumerate(pairs):
                checks.scalar(f"type_{j}_pairing_{k}", float(values.max()), float(values.min()))
            # Average only after every repetition check passes. No eigenvalue clipping.
            multiplicity = pairs.mean(axis=1)
            arrays[f"type_{j}_multiplicity_eigenvalues"] = multiplicity
            energy = float(np.trace(restricted) / r)
            checks.interval(f"type_{j}_energy", energy, 0.0, 1.0)
            if t.get("absent"):
                checks.scalar(f"type_{j}_analytic_absent_energy", energy, 0.0)
            types.append((d, multiplicity))
            infos.append({"name": t["name"], "real_dimension": d, "isotypic_dimension": dim,
                "multiplicity": len(multiplicity), "total_energy": energy, "e": energy,
                "expected_absent": bool(t.get("absent")),
                "eigenvalue_pairing_max_spread": float(np.max(pairs.max(axis=1) - pairs.min(axis=1))),
                "multiplicity_eigenvalues": multiplicity.tolist()})
        total_energy = sum(t["total_energy"] for t in infos)
        checks.scalar("type_energy_sum", total_energy, 1.0)
        budget = integer_budget(types, r)
        best, ks = budget["value"], budget["selected_copies"]
        used = sum(d * k for (d, _), k in zip(types, ks))
        checks.condition("integer_budget_feasible", used <= r and used == budget["used_dimension"] and
                         all(0 <= k <= len(lam) for (_, lam), k in zip(types, ks)))
        reconstructed = sum(float(d * np.sum(lam[:k])) for (d, lam), k in zip(types, ks))
        checks.scalar("integer_budget_value", reconstructed, best)
        top = float(np.sum(spectrum_K[:r]))
        delta = top - best
        gamma = best / r
        checks.interval("gamma_range", gamma, 0.0, 1.0)
        checks.interval("typed_budget_vs_untyped", delta, 0.0, float(r))
        if spec["domain"] in ("C7", "D4"):
            checks.interval("analytic_type_correction_bound", delta, 0.0, 1.0)
        for j, (t, (d, lam), k) in enumerate(zip(infos, types, ks)):
            selected = float(d * np.sum(lam[:k]) / r)
            t.update(selected_copies=k, selected_dimension=d * k,
                     selected_energy=selected, b=selected)
            checks.interval(f"type_{j}_selected_energy", selected, 0.0, t["total_energy"])
        checks.scalar("selected_energy_sum", sum(t["b"] for t in infos), gamma)
        eta_spec = 1.0 - float(np.trace(K @ K)) / r
        errors, operators = [], []
        for action in actions:
            Xg = X[:, action, :].reshape(-1, r)
            A = Xf.T @ Xg / len(Xf)
            operators.append(A)
            errors.append(float(np.mean(np.sum((Xf @ A - Xg) ** 2, axis=1)) / r))
        arrays["unconstrained_LS_operators"] = np.asarray(operators)
        arrays["unconstrained_LS_risks"] = np.asarray(errors)
        eta_LS = float(np.mean(errors))
        checks.scalar("eta_LS_spectral_identity", eta_spec, eta_LS)
        checks.interval("eta_range", eta_spec, 0.0, 1.0)
        lower, upper = 2 * (1 - gamma), 2 * (1 - gamma * gamma)
        metrics = {"gamma": gamma, "C_r": best, "C_untyped": top,
                   "delta_type": delta, "delta_type_normalized": delta / r,
                   "L_lower": lower, "L_upper": upper, "eta_spec": eta_spec, "eta_LS": eta_LS}
        summary.update(status="complete", metrics=metrics, types=infos, budget=budget,
            near_zero_numerically_unresolved=[key for key, value in metrics.items()
                if abs(value) <= _scalar_tol(value, 0.0)],
            interpretation="Finite-bank class-optimal risk bounds in the declared empirical metric; "
                "raw floating-point approximations, not rigorous error enclosures or semantic/causal identification.",
            witness_action_constructed=False, statistical_certificate=False)
    except ValidationFailure as error:
        summary.update(status=error.status, reason=error.reason)
    except (ValueError, TypeError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as error:
        summary.update(status="not_evaluable_numerics", reason=f"numeric_operation: {type(error).__name__}: {error}")
    return summary, arrays


def analyze_bank(Z: Any, domain: str) -> tuple[dict, dict[str, np.ndarray]]:
    """Analyze an already authenticated complete array; no external side effects.

    The production adapter separately enforces r=16 and the fixed 128 blocks.
    This numerical kernel accepts any positive rank for analytic unit fixtures.
    No result field named ``metrics`` is returned on any validation failure.
    """
    try:
        spec = group_spec(domain)
    except (TypeError, ValueError) as error:
        return ({"protocol_id": PROTOCOL_ID, "status": "not_evaluable_input",
                 "reason": str(error), "checks": {}, "diagnostics": {},
                 "provenance_verified_by_core": False}, {})
    return _analyze_spec(Z, spec)
