"""Compact, validation-selected coordinate predictors. No model inference or calibration.

Only construction pairs enter fitting. Validation targets select hyperparameters.
Every prediction object retains only its declared numeric payload. Gaussian-field
landmarks retain their own target displacements; RFF and Nyström regression expose
all construction target pairs at every budget. These are different information
conditions and must remain separately labeled in result tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import numpy as np


@dataclass
class Fitted:
    name: str
    predict: Callable | None
    metadata: dict


class Predictor:
    """Inference payload; deliberately contains no training or validation bank."""
    def __init__(self, kind, actions, params):
        self.kind = kind
        self.actions = tuple(actions)
        self.params = params

    def __call__(self, query, action):
        q = np.asarray(query, dtype=np.float64)
        p = self.params
        a = self.actions.index(str(action))
        if self.kind == "ridge":
            return q @ p["coef"][a]
        if self.kind == "affine":
            return q @ p["coef"][a, :-1] + p["coef"][a, -1]
        if self.kind == "centroid":
            c = p["centroids"]
            labels = _distances(q, c).argmin(axis=1)
            return q + c[p["action_table"][a, labels]] - c[labels]
        if self.kind == "limited_support":
            out = np.empty_like(q)
            for start in range(0, len(q), 512):
                sl = slice(start, start + 512)
                logits = -_distances(q[sl], p["source"]) / (2 * p["bandwidth"] ** 2)
                logits -= logits.max(axis=1, keepdims=True)
                weights = np.exp(logits)
                weights /= weights.sum(axis=1, keepdims=True)
                out[sl] = q[sl] + weights @ p["displacements"][a]
            return out
        if self.kind in ("rff", "nystrom"):
            xn = (q - p["mean"]) / p["scale"]
            if self.kind == "rff":
                phi = np.cos(xn @ p["frequencies"] + p["phases"])
                phi *= math.sqrt(2.0 / len(p["phases"]))
            else:
                kernel = np.exp(-_distances(xn, p["landmarks"]) / (2 * p["bandwidth"] ** 2))
                # Whitening is absorbed into fitted coefficients once; no m²
                # transform is retained in the compact inference payload.
                phi = kernel
            return q + p["scale"] * (phi @ p["coef"][a, :-1] + p["coef"][a, -1])
        raise ValueError(f"Unknown predictor kind: {self.kind}")


def _distances(x, y):
    # Multiplication avoids an n x m x d allocation.
    return np.maximum(np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2 * x @ y.T, 0.0)


def _payload(params):
    floats = integers = 0
    shapes = {}
    for key, value in params.items():
        arr = np.asarray(value)
        shapes[key] = list(arr.shape)
        if arr.dtype.kind == "f":
            if arr.dtype != np.float64:
                raise ValueError("Inference floating payload must be float64")
            floats += arr.size
        elif arr.dtype.kind in "iu":
            if arr.dtype != np.int64:
                raise ValueError("Inference integer payload must be int64")
            integers += arr.size
        else:
            raise ValueError(f"Unaccounted inference payload: {key}")
    return {
        "float64_coordinates": int(floats),
        "integer64_coordinates": int(integers),
        "float64_bytes": int(8 * floats),
        "integer64_bytes": int(8 * integers),
        "parameter_bytes": int(8 * (floats + integers)),
        "logical_serialized_parameter_bytes": int(8 * (floats + integers)),
        "materialized_numeric_parameter_bytes": int(8 * (floats + integers)),
        "shapes": shapes,
        "excluded_common_inputs": "PCA coordinates and common action identifiers; not Python-object, JSON, ZIP or filesystem overhead",
        "accounting": "All retained inference numeric arrays/scalars, including decoder tables; no seed-regeneration discount",
    }


def _as_matrix(value, name, d=None):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or not len(arr) or (d is not None and arr.shape[1] != d) or not np.isfinite(arr).all():
        raise ValueError(f"Invalid finite nonempty matrix: {name}")
    return arr


def _farthest(x, count):
    """Deterministic source-only farthest-first; first point farthest from mean."""
    count = min(int(count), len(x))
    selected = []
    distance = np.full(len(x), np.inf)
    available = np.ones(len(x), dtype=bool)
    initial = np.sum((x - x.mean(axis=0)) ** 2, axis=1)
    index = int(initial.argmax())
    for _ in range(count):
        selected.append(index)
        available[index] = False
        distance = np.minimum(distance, np.sum((x - x[index]) ** 2, axis=1))
        distance[~available] = -np.inf
        if len(selected) < count:
            index = int(distance.argmax())
    return np.asarray(selected, dtype=np.int64)


def _median_distance(x):
    # Fixed deterministic pairs, independent of targets and evaluation rows.
    n = len(x)
    if n < 2:
        return 1.0
    total = n * (n - 1) // 2
    ranks = np.linspace(0, total - 1, min(total, 4096), dtype=np.int64)
    offsets = np.arange(n, dtype=np.int64)
    offsets = offsets * (2 * n - offsets - 1) // 2
    first = np.searchsorted(offsets, ranks, side="right") - 1
    second = first + 1 + ranks - offsets[first]
    distances = np.sqrt(np.sum((x[first] - x[second]) ** 2, axis=1))
    positive = distances[distances > 1e-12]
    return float(np.median(positive)) if len(positive) else 1.0


def _collapse_nystrom_coefficients(whitener, beta):
    """Preserve the whitened-feature fit exactly without retaining its basis."""
    return np.vstack((whitener @ beta[:-1], beta[-1:]))


def _ridge_solutions(features, target, penalties):
    gram = features.T @ features
    eigenvalues, vectors = np.linalg.eigh((gram + gram.T) * 0.5)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    rhs = vectors.T @ (features.T @ target)
    scale = float(np.trace(gram)) / gram.shape[0]
    for penalty in penalties:
        if penalty == 0:
            # Match the audited baseline's singular-value cutoff, not a cutoff
            # on X'X eigenvalues (which would square the condition number).
            yield float(penalty), np.linalg.lstsq(features, target, rcond=1e-12)[0]
        else:
            reciprocal = 1.0 / (eigenvalues + penalty * max(scale, 1e-15))
            yield float(penalty), vectors @ (reciprocal[:, None] * rhs)


def _validation_loss(predictor, validation):
    losses = []
    for action, (x, y) in validation.items():
        pred = predictor(x, action)
        if pred.shape != y.shape or not np.isfinite(pred).all():
            return math.inf
        losses.append(float(np.mean((pred - y) ** 2)))
    return float(np.mean(losses))


def _finish(name, predictor, ledger, selected, n, actions, reference, *, budget=None, all_pairs=True, extra=None):
    selection_finite = np.isfinite(selected.get("validation_mse", 0.0))
    ledger = [dict(candidate, validation_mse=(float(candidate["validation_mse"])
                if np.isfinite(candidate["validation_mse"]) else None),
                validation_status=("finite" if np.isfinite(candidate["validation_mse"]) else "nonfinite"))
              for candidate in ledger]
    if not selection_finite:
        return Fitted(name, None, {"status": "numerical_failure", "reason": "Every validation candidate produced nonfinite error",
            "selection_ledger": ledger, "training_rows": int(n), "training_actions": list(actions),
            "reference_identity_pruned_ridge_float64_coordinates": int(reference),
            "budget_float64_equivalent_coordinates": int(budget) if budget is not None else None})
    storage = _payload(predictor.params)
    if budget is not None and storage["parameter_bytes"] > int(budget) * 8:
        raise AssertionError(f"Storage accounting exceeded budget for {name}")
    metadata = {
        "status": "fit", "family": predictor.kind,
        "training_rows": int(n), "training_actions": list(actions),
        "training_target_pairs_available": int(n * len(actions)),
        "all_training_pairs_exposed": bool(all_pairs),
        "selection_ledger": ledger, "selected": selected,
        "selection_rule": "Lowest equal-action validation MSE; deterministic candidate order resolves exact ties",
        "selection_inputs": "construction source/target pairs and changed-state validation pairs only",
        "reference_identity_pruned_ridge_float64_coordinates": int(reference),
        "budget_float64_equivalent_coordinates": int(budget) if budget is not None else None,
        "storage": storage,
        "scientific_status": "exploratory_after_previous_holdout_access",
    }
    if "coef" in predictor.params:
        coefficients = predictor.params["coef"]
        norm = float(np.linalg.norm(coefficients))
        metadata["coefficient_frobenius_norm"] = norm if np.isfinite(norm) else None
        metadata["per_action_coefficient_frobenius_norm"] = {
            action: float(np.linalg.norm(coefficients[i])) if np.isfinite(np.linalg.norm(coefficients[i])) else None
            for i, action in enumerate(actions)}
    if extra:
        metadata.update(extra)
    return Fitted(name, predictor, metadata)


def _unavailable(name, reason, reference, budget=None):
    return Fitted(name, None, {"status": "infeasible", "reason": reason,
        "reference_identity_pruned_ridge_float64_coordinates": int(reference),
        "budget_float64_equivalent_coordinates": int(budget) if budget is not None else None})


def fit_models(x, train_targets, validation, config, *, states=None, tables=None):
    """Fit independent-action predictors with no evaluation-bank argument.

    x is construction source rows. Targets and validation have identical string
    action keys; validation[action] is a (source, target) pair of changed rows.
    The caller removes identity actions and applies the intended primitive-only
    or all-element information condition before invoking this function.
    """
    x = _as_matrix(x, "construction source")
    n, d = x.shape
    actions = tuple(str(a) for a in train_targets)
    if not actions or len(actions) != len(set(actions)):
        raise ValueError("Nonempty unique string action identifiers required")
    if any(not isinstance(a, str) for a in train_targets) or set(validation) != set(actions):
        raise ValueError("Targets and validation must have identical string action keys")
    targets = {a: _as_matrix(train_targets[a], f"target {a}", d) for a in actions}
    if any(y.shape != x.shape for y in targets.values()):
        raise ValueError("Every construction target must align with every source")
    valid = {a: (_as_matrix(validation[a][0], f"validation source {a}", d),
                 _as_matrix(validation[a][1], f"validation target {a}", d)) for a in actions}
    if any(xv.shape != yv.shape for xv, yv in valid.values()):
        raise ValueError("Validation source/target shape mismatch")
    penalties = tuple(float(v) for v in config.get("lambda_grid", [0, 1e-5, .001, .1, 1]))
    widths = tuple(float(v) for v in config.get("bandwidth_multipliers", config.get("bandwidth_grid", [.5, 1, 2])))
    multipliers = tuple(float(v) for v in config.get("budget_multipliers", [.5, 1, 2, 4, 8]))
    if not penalties or min(penalties) < 0 or not widths or min(widths) <= 0 or not multipliers or min(multipliers) <= 0:
        raise ValueError("Invalid nonempty finite positive bandwidth/budget or nonnegative regularization grids")
    if not all(np.isfinite(v) for v in penalties + widths + multipliers):
        raise ValueError("Hyperparameter grids must be finite")
    methods = set(config.get("methods", ["ridge", "affine", "centroid", "limited_support", "rff", "nystrom"]))
    unknown = methods - {"ridge", "affine", "centroid", "limited_support", "rff", "nystrom"}
    if unknown:
        raise ValueError(f"Unknown compact methods: {sorted(unknown)}")
    count = len(actions)
    reference = count * d * d
    results = []
    ys = np.concatenate([targets[a] for a in actions], axis=1)
    for method in ("ridge", "affine"):
        if method not in methods:
            continue
        features = x if method == "ridge" else np.column_stack((x, np.ones(n)))
        best, ledger = None, []
        for penalty, beta in _ridge_solutions(features, ys, penalties):
            params = {"coef": beta.reshape(features.shape[1], count, d).transpose(1, 0, 2).copy()}
            predictor = Predictor(method, actions, params)
            loss = _validation_loss(predictor, valid)
            candidate = {"lambda": penalty, "validation_mse": loss}
            ledger.append(candidate)
            if best is None or loss < best[0]:
                best = (loss, predictor, candidate)
        results.append(_finish(method, best[1], ledger, best[2], n, actions, reference))

    if "centroid" in methods:
        if states is None or tables is None:
            results.append(_unavailable("decoded_centroid", "Requires declared construction state labels and action table", reference))
        else:
            labels = np.asarray(states, dtype=np.int64)
            unique = np.unique(labels)
            if labels.shape != (n,) or not np.array_equal(unique, np.arange(len(unique))):
                raise ValueError("Centroid construction labels must cover consecutive states 0..S-1")
            centroid = np.stack([x[labels == state].mean(axis=0) for state in unique])
            action_table = np.asarray([tables[a] for a in actions], dtype=np.int64)
            if action_table.shape != (count, len(unique)) or action_table.min() < 0 or action_table.max() >= len(unique):
                raise ValueError("Invalid centroid action table")
            predictor = Predictor("centroid", actions, {"centroids": centroid, "action_table": action_table})
            loss = _validation_loss(predictor, valid)
            results.append(_finish("decoded_centroid", predictor, [{"validation_mse": loss}], {}, n, actions, reference,
                extra={"information_condition": "Construction state labels and supplied action table; query label decoded from source coordinates at every call",
                       "intermediate_state_redecoded": True, "training_target_pairs_used": 0,
                       "all_training_pairs_exposed": False}))

    if not methods.intersection({"limited_support", "rff", "nystrom"}):
        return results
    mean = np.asarray(x.mean(axis=0), dtype=np.float64)
    scale = float(np.sqrt(np.mean((x - mean) ** 2)))
    if scale <= 1e-15:
        scale = 1.0
    xn = (x - mean) / scale
    source_width = _median_distance(x)
    normalized_width = source_width / scale
    budgets = [(m, int(math.floor(m * reference))) for m in multipliers]
    # Every feature count is the largest one satisfying the declared parameter budget.
    # No hidden fixed feature cap; infeasible entries are retained explicitly.
    support_counts = {b: min(n, max(0, (b - 1) // (d * (count + 1)))) for _, b in budgets}
    nystrom_counts = {}
    for _, b in budgets:
        # mean(d), scale(1), width(1), landmarks(md), collapsed coefficients((m+1)Ad).
        # The train-only whitening basis is multiplied into the coefficients
        # before retention, preserving predictions without an m² inference cost.
        constant = d + 2 + count * d
        linear = d + count * d
        nystrom_counts[b] = min(n, max(0, (b - constant) // linear))
    farthest_count = max([support_counts[b] if "limited_support" in methods else 0 for _, b in budgets] +
                         [nystrom_counts[b] if "nystrom" in methods else 0 for _, b in budgets])
    order = _farthest(xn, farthest_count) if farthest_count else np.empty(0, dtype=np.int64)
    residual = np.concatenate([(targets[a] - x) / scale for a in actions], axis=1)

    for multiplier, budget in budgets:
        suffix = f"budget_{multiplier:g}".replace(".", "p")
        if "limited_support" in methods:
            m = support_counts[budget]
            name = f"limited_support_{suffix}"
            if m < 1:
                results.append(_unavailable(name, "Budget cannot retain one support source and all action displacements plus bandwidth", reference, budget))
            else:
                indices = order[:m]
                sx = x[indices].copy()
                delta = np.stack([targets[a][indices] - sx for a in actions])
                best, ledger = None, []
                for width in widths:
                    params = {"source": sx, "displacements": delta, "bandwidth": float(source_width * width)}
                    predictor = Predictor("limited_support", actions, params)
                    loss = _validation_loss(predictor, valid)
                    candidate = {"bandwidth_multiplier": width, "validation_mse": loss}
                    ledger.append(candidate)
                    if best is None or loss < best[0]:
                        best = (loss, predictor, candidate)
                results.append(_finish(name, best[1], ledger, best[2], n, actions, reference, budget=budget, all_pairs=m == n,
                    extra={"budget_multiplier": multiplier, "support_count": m, "support_source_indices": indices.tolist(),
                           "training_target_pairs_used": m * count, "source_selection": "All construction source coordinates, deterministic farthest-first",
                           "information_condition": "Support targets only; all source coordinates available to source selection and scale"}))

        for method in ("rff", "nystrom"):
            if method not in methods:
                continue
            name = f"{method}_{suffix}"
            if method == "rff":
                # normalizer d+1; bandwidth 1; frequencies dm, phases m, bias+coefficients (m+1)Ad.
                m = max(0, (budget - (d + 2 + count * d)) // (d + 1 + count * d))
            else:
                m = nystrom_counts[budget]
            if m < 1:
                results.append(_unavailable(name, "Budget cannot retain normalizer, one nonlinear feature and every action coefficient/bias", reference, budget))
                continue
            seed = int(config.get("seed", 20261021))
            rng = np.random.default_rng(seed)
            frequencies = rng.normal(size=(d, m)) if method == "rff" else None
            phases = rng.uniform(0, 2 * np.pi, size=m) if method == "rff" else None
            landmarks = xn[order[:m]].copy() if method == "nystrom" else None
            best, ledger = None, []
            for width in widths:
                bandwidth = float(normalized_width * width)
                base = {"mean": mean.copy(), "scale": float(scale), "bandwidth": bandwidth}
                if method == "rff":
                    base.update(frequencies=frequencies / bandwidth, phases=phases.copy())
                    phi = math.sqrt(2.0 / m) * np.cos(xn @ base["frequencies"] + phases)
                else:
                    kernel = np.exp(-_distances(landmarks, landmarks) / (2 * bandwidth ** 2))
                    eigenvalues, vectors = np.linalg.eigh((kernel + kernel.T) * .5)
                    cutoff = max(float(eigenvalues.max()), 1.0) * 1e-10
                    invroot = np.divide(1.0, np.sqrt(np.maximum(eigenvalues, 0.0)),
                                        out=np.zeros_like(eigenvalues), where=eigenvalues > cutoff)
                    whitener = (vectors * invroot) @ vectors.T
                    base.update(landmarks=landmarks.copy())
                    phi = np.exp(-_distances(xn, landmarks) / (2 * bandwidth ** 2)) @ whitener
                features = np.column_stack((phi, np.ones(n)))
                for penalty, beta in _ridge_solutions(features, residual, penalties):
                    params = dict(base)
                    inference_beta = _collapse_nystrom_coefficients(whitener, beta) if method == "nystrom" else beta
                    params["coef"] = inference_beta.reshape(m + 1, count, d).transpose(1, 0, 2).copy()
                    predictor = Predictor(method, actions, params)
                    loss = _validation_loss(predictor, valid)
                    candidate = {"bandwidth_multiplier": width, "lambda": penalty, "validation_mse": loss}
                    ledger.append(candidate)
                    if best is None or loss < best[0]:
                        best = (loss, predictor, candidate)
            results.append(_finish(name, best[1], ledger, best[2], n, actions, reference, budget=budget,
                extra={"budget_multiplier": multiplier, "feature_count": int(m), "feature_seed": seed if method == "rff" else None,
                       "landmark_source_indices": order[:m].tolist() if method == "nystrom" else None,
                       "training_target_pairs_used": n * count,
                       "information_condition": "Every construction target pair fits residual coefficients; train-only source features and validation-only hyperparameter choice",
                       "normalizer": "Construction mean and global RMS deviation; same scale for every coordinate",
                       "feature_kind": "Gaussian random Fourier cosine features" if method == "rff" else "Gaussian Nyström fit with train-only eigenvalue cutoff 1e-10; whitening absorbed exactly into retained kernel coefficients"}))
    return results


def compose(predict, source, word):
    """Autoregressive composition: intermediate predictions, never true states."""
    out = np.asarray(source, dtype=np.float64).copy()
    for action in word:
        out = predict(out, str(action))
    return out


def _row_stats(loss, indices, rows):
    finite = np.isfinite(loss)
    grouped = {}
    for local, index in enumerate(indices):
        row = rows[int(index)]
        key = (str(row["family_id"]), str(row.get("orbit_id", "0")))
        grouped.setdefault(key, []).append(float(loss[local]))
    groups = []
    for (family, orbit), values in sorted(grouped.items()):
        values = np.asarray(values)
        groups.append({"family_id": family, "orbit_id": orbit, "n": len(values),
                       "invalid": int((~np.isfinite(values)).sum()),
                       "mean_nmse": float(values.mean()) if np.isfinite(values).all() else None})
    return {"n": int(len(loss)), "invalid": int((~finite).sum()),
            "mean_nmse": float(loss.mean()) if len(loss) and finite.all() else None,
            "family_orbit": groups}


def _prediction_stats(source, target, prediction, indices, rows, energy):
    with np.errstate(over="ignore", invalid="ignore"):
        loss = np.sum((prediction - target) ** 2, axis=1) / energy
        output_norm = np.sum(prediction ** 2, axis=1) / energy
        displacement_norm = np.sum((prediction - source) ** 2, axis=1) / energy
    summary = _row_stats(loss, indices, rows)
    summary["mean_prediction_squared_norm_over_training_energy"] = float(output_norm.mean()) if len(output_norm) and np.isfinite(output_norm).all() else None
    summary["mean_displacement_squared_norm_over_training_energy"] = float(displacement_norm.mean()) if len(displacement_norm) and np.isfinite(displacement_norm).all() else None
    summary["mean_input_squared_norm_over_training_energy"] = float(np.mean(np.sum(source ** 2, axis=1)) / energy) if len(source) else None
    return summary


def _pool_action_summary(per_action, key="mean_nmse"):
    n = sum(value["n"] for value in per_action.values())
    if not n or any(value[key] is None for value in per_action.values()):
        return None
    return float(sum(value[key] * value["n"] for value in per_action.values()) / n)


def evaluate_compact(cell, config, eval_splits):
    """All-element and generator-only experiments on a verified coordinate cell."""
    z = np.asarray(cell.z, dtype=np.float64)
    train = np.asarray(cell.train)
    if train.dtype == bool:
        train = np.flatnonzero(train)
    validation = np.asarray(cell.validation)
    if validation.dtype == bool:
        validation = np.flatnonzero(validation)
    states = np.asarray(cell.states)
    energy = float(np.mean(np.sum((z[train] - z[train].mean(axis=0)) ** 2, axis=1)))
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("Nonpositive construction coordinate energy")
    all_actions = [str(a) for a in cell.elements if str(a) != str(cell.identity)]
    generator_actions = [str(a) for a in cell.generators]
    result = {"domain": cell.domain, "training_energy": energy,
              "normalization": "mean construction squared norm about construction mean, shared across actions and splits",
              "technical_test": bool(config.get("technical_test", False)), "paper_eligible": False, "confirmatory": False,
              "scientific_status": "exploratory_after_previous_holdout_access", "tracks": {}}
    eval_splits = tuple(eval_splits)
    allowed = {"pilot", "test", "test_joint", "test_context", "test_entity"}
    if not eval_splits or len(eval_splits) != len(set(eval_splits)) or not set(eval_splits).issubset(allowed):
        raise ValueError("Evaluation partitions must be unique pilot or held-out test roles")
    split_labels = np.asarray([row["split"] for row in cell.rows])
    if any(not np.any(split_labels == split) for split in eval_splits):
        raise ValueError("Requested evaluation partition is absent")
    for track, fitted_actions in (("all_element", all_actions), ("generator_only", generator_actions)):
        targets = {a: z[np.asarray(cell.targets[a])[train]] for a in fitted_actions}
        valid = {}
        for action in fitted_actions:
            paired = np.asarray(cell.targets[action])[validation]
            keep = states[validation] != states[paired]
            valid[action] = (z[validation[keep]], z[paired[keep]])
        models = fit_models(z[train], targets, valid, config, states=states[train], tables=cell.tables)
        records = []
        for fitted in models:
            entry = {"name": fitted.name, "fit": fitted.metadata, "evaluation": {}}
            if fitted.predict is None:
                records.append(entry)
                continue
            validation_summary = {}
            for action in fitted_actions:
                paired = np.asarray(cell.targets[action])[validation]
                keep = states[validation] != states[paired]
                selected = validation[keep]
                with np.errstate(over="ignore", invalid="ignore"):
                    prediction = fitted.predict(z[selected], action)
                validation_summary[action] = _prediction_stats(z[selected], z[paired[keep]], prediction,
                                                               selected, cell.rows, energy)
            entry["validation_selected_predictor"] = {
                "actions": validation_summary,
                "pooled_changed_pair_mean_nmse": _pool_action_summary(validation_summary),
                "pooled_mean_prediction_squared_norm_over_training_energy": _pool_action_summary(
                    validation_summary, "mean_prediction_squared_norm_over_training_energy"),
                "pooled_mean_displacement_squared_norm_over_training_energy": _pool_action_summary(
                    validation_summary, "mean_displacement_squared_norm_over_training_energy"),
                "purpose": "Frozen validation-only error/norm matching; no evaluation target used for matching"}
            for split in eval_splits:
                indices = np.flatnonzero(split_labels == split)
                per_action = {}
                for action in all_actions:
                    paired = np.asarray(cell.targets[action])[indices]
                    keep = states[indices] != states[paired]
                    selected = indices[keep]
                    target = z[paired[keep]]
                    with np.errstate(over="ignore", invalid="ignore"):
                        prediction = fitted.predict(z[selected], action) if track == "all_element" else compose(fitted.predict, z[selected], cell.words[action])
                    summary = _prediction_stats(z[selected], target, prediction, selected, cell.rows, energy)
                    summary["is_primitive"] = action in generator_actions
                    per_action[action] = summary
                means = [value["mean_nmse"] for value in per_action.values()]
                unseen = [value["mean_nmse"] for action, value in per_action.items() if action not in generator_actions]
                entry["evaluation"][split] = {
                    "actions": per_action,
                    "equal_action_mean_nmse": float(np.mean(means)) if means and all(v is not None for v in means) else None,
                    "pooled_changed_pair_mean_nmse": _pool_action_summary(per_action),
                    ("unseen_composite_equal_action_mean_nmse" if track == "generator_only" else "nonprimitive_direct_equal_action_mean_nmse"): float(np.mean(unseen)) if unseen and all(v is not None for v in unseen) else None,
                    "invalid_predictions": sum(value["invalid"] for value in per_action.values()),
                }
                if track == "generator_only":
                    entry["evaluation"][split]["composition_diagnostics"] = _composition_diagnostics(cell, fitted.predict, indices, energy)
            records.append(entry)
        result["tracks"][track] = {"fitting_actions": fitted_actions, "selection_actions": fitted_actions,
                                    "models": records, "identity_policy": "Identity is exact and omitted from fitted parameter payload"}
    return result


def _composition_diagnostics(cell, predict, indices, energy):
    z = np.asarray(cell.z)
    generators = [str(a) for a in cell.generators]
    domain = str(cell.domain)
    cycles = {"r7": (generators[0],) * 7} if domain == "C7" else {}
    if domain.startswith("D4"):
        cycles = {"r4": (generators[0],) * 4, "s2": (generators[1],) * 2}
    output = {"cycles": {}}
    for name, word in cycles.items():
        with np.errstate(over="ignore", invalid="ignore"):
            pred = compose(predict, z[indices], word)
            loss = np.sum((pred - z[indices]) ** 2, axis=1) / energy
        output["cycles"][name] = _row_stats(loss, indices, cell.rows)
    if domain.startswith("D4"):
        r, s = generators
        comparisons = (("d4_discriminating_order", (s, r, s), (r, s, r)),
                       ("d4_discriminating_noncommutativity", (r, s), (s, r)))
        for name, correct_word, wrong_word in comparisons:
            correct_target = indices.copy()
            wrong_target = indices.copy()
            for action in correct_word:
                correct_target = np.asarray(cell.targets[action])[correct_target]
            for action in wrong_word:
                wrong_target = np.asarray(cell.targets[action])[wrong_target]
            keep = np.asarray(cell.states)[correct_target] != np.asarray(cell.states)[wrong_target]
            selected = indices[keep]
            true = z[correct_target[keep]]
            with np.errstate(over="ignore", invalid="ignore"):
                correct = compose(predict, z[selected], correct_word)
                wrong = compose(predict, z[selected], wrong_word)
                correct_loss = np.sum((correct - true) ** 2, axis=1) / energy
                wrong_loss = np.sum((wrong - true) ** 2, axis=1) / energy
            output[name] = {
                "correct_word": list(correct_word), "comparison_word": list(wrong_word),
                "restriction": "Only source states with distinct algebraic targets under the two words",
                "correct_word_to_designated_target": _row_stats(correct_loss, selected, cell.rows),
                "wrong_word_to_designated_target": _row_stats(wrong_loss, selected, cell.rows)}
    return output
