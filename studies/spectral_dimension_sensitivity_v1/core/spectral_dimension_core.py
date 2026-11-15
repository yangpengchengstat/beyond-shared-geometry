"""Prepared, model-free dimension sensitivity on authenticated Study 2 banks.

This module has NOT been numerically executed as part of preparation. Calling
an analysis function performs a new offline analysis and requires separate
execution authorization in the adapter. Importing has no analysis side effects.

The input must be the original ordered 16-dimensional training-PCA scores, not
whitened scores. Taking the first eight columns changes the observation space;
the unmodified frozen kernel constructs its own uncentered second moment and
whitening in each new dimension. It does not reuse a sliced 16-D whitening.
No PCA is fitted, no old coordinates are changed, and no witness is trained.

The caller authenticates archive/member hashes, row/state identities, model,
readout, partition and PCA column order. Metadata is retained but is not trusted
as proof of provenance by the numerical core.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ._vendor import gamma_r_frozen as spectral
except ImportError:
    from _vendor import gamma_r_frozen as spectral


PROTOCOL_ID = "spectral_dimension_sensitivity_v1"
ALLOWED_DIMENSIONS = (8, 16)
SOURCE_DIMENSION = 16
N_CONTEXTS = 128
KERNEL_SHA256 = "e91c735890bdf888a25db07d1207d3f44cdce8ce1bf6ee7e2efcf7bf40a50f41"
IMAGE_COUNTS = {"C7": 120, "D4": 3}


def verify_kernel_source() -> dict:
    """Read source bytes only; no numeric analysis or scientific validation."""
    path = Path(spectral.__file__).resolve()
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != KERNEL_SHA256:
        raise ValueError("Frozen kernel source checksum mismatch")
    return {"source_sha256": observed, "source_unmodified": True}


def _tol(a: float, b: float = 0.0) -> float:
    return spectral.SCALAR_ATOL + spectral.SCALAR_RTOL * max(abs(a), abs(b))


def _scalar_check(a: float, b: float) -> dict:
    residual = abs(float(a) - float(b))
    tolerance = _tol(float(a), float(b))
    return {"observed": float(a), "reference": float(b),
            "absolute_difference": residual if math.isfinite(residual) else None,
            "tolerance": tolerance,
            "passed": bool(math.isfinite(residual) and residual <= tolerance)}


@lru_cache(maxsize=2)
def _enumerated_subgroups(domain: str) -> list[dict]:
    """Same deterministic image IDs as frozen extension numerics.py:254-282.

    This preserves the original semantic flag and lexicographically minimal
    conjugated representative, including full identity/fixed-point coverage.
    It does not enumerate new abstract homomorphism classes.
    """
    verify_kernel_source()
    base = spectral.group_spec(domain)
    p = base["p"]
    original = np.asarray(base["actions"], dtype=int)
    found = {}
    for relabel in itertools.permutations(range(p)):
        relabel = np.asarray(relabel, dtype=int)
        inverse = np.argsort(relabel)
        actions = relabel[original[:, inverse]]
        key = tuple(sorted(tuple(row) for row in actions.tolist()))
        representative = tuple(tuple(row) for row in actions.tolist())
        if key not in found or representative < found[key]:
            found[key] = representative
    semantic = tuple(sorted(tuple(row) for row in original.tolist()))
    result = []
    for i, key in enumerate(sorted(found)):
        spec = copy.deepcopy(base)
        spec["actions"] = [list(row) for row in found[key]]
        spectral.validate_group(spec)
        result.append({"id": f"{domain}_image_{i:03d}", "semantic": key == semantic,
                       "image_permutations": [list(row) for row in key], "spec": spec})
    if len(result) != IMAGE_COUNTS[domain] or sum(x["semantic"] for x in result) != 1:
        raise ValueError("Frozen subgroup enumeration count/semantic identity failed")
    return result


def subgroup_specs(domain: str) -> list[dict]:
    """Return an isolated copy of the frozen enumeration, cached by domain."""
    return copy.deepcopy(_enumerated_subgroups(domain))


def _source_coordinates(Z16: Any, domain: str, dimension: int) -> np.ndarray:
    if domain not in IMAGE_COUNTS:
        raise ValueError("Only canonical C7 and four-corner D4 are supported")
    if type(dimension) is not int or dimension not in ALLOWED_DIMENSIONS:
        raise ValueError("This prepared protocol permits only dimensions 8 and 16")
    raw = np.asarray(Z16)
    expected_shape = (N_CONTEXTS, 7 if domain == "C7" else 4, SOURCE_DIMENSION)
    if raw.shape != expected_shape:
        raise ValueError(f"Expected original Study 2 test bank {expected_shape}, got {raw.shape}")
    if raw.dtype.kind not in "fiu" or np.iscomplexobj(raw) or not np.isfinite(raw).all():
        raise ValueError("Original bank must contain finite real numeric PCA coordinates")
    # Make an independent array: never mutate the authenticated source object.
    return np.array(raw[..., :dimension], dtype=np.float64, order="C", copy=True)


def _derived_metrics(Z: np.ndarray, spec: dict, summary: dict, arrays: dict) -> tuple[dict, dict]:
    """Same-metric premium; eta_LS is the canonical bounded-analysis eta."""
    metrics = dict(summary["metrics"])
    r = Z.shape[-1]
    eta = float(metrics["eta_LS"])
    lower, upper = float(metrics["L_lower"]), float(metrics["L_upper"])
    trivial_name = "trivial" if spec["domain"] == "C7" else "chi_pp"
    trivial = next(t for t in summary["types"] if t["name"] == trivial_name)
    e_triv, b_triv = float(trivial["e"]), float(trivial["b"])
    identity = 2 * (1 - e_triv)
    X = Z @ arrays["whitening"]
    actions = np.asarray(spec["actions"], dtype=int)
    identity_direct = float(np.mean([
        np.mean(np.sum((X - X[:, action, :]) ** 2, axis=-1)) / r
        for action in actions
    ]))
    p_minus = max(0.0, lower - eta)  # Definition, not clipping an eigensystem.
    p_plus = upper - eta
    checks = {
        "identity_type_vs_direct": _scalar_check(identity, identity_direct),
        "eta_LS_vs_eta_spec": _scalar_check(eta, float(metrics["eta_spec"])),
        "eta_le_identity": {"passed": bool(eta <= identity + _tol(eta, identity))},
        "lower_bound_le_identity": {"passed": bool(lower <= identity + _tol(lower, identity))},
        "premium_bounds_ordered": {"passed": bool(p_minus <= p_plus + _tol(p_minus, p_plus))},
        "premium_upper_nonnegative": {"passed": bool(p_plus >= -_tol(p_plus, 0.0))},
    }
    metrics.update(eta=eta, eta_definition="eta_LS", e_triv=e_triv, b_triv=b_triv,
                   L_identity=identity, L_identity_direct=identity_direct,
                   p_minus=p_minus, p_plus=p_plus,
                   premium_lower_above_frozen_threshold=bool(p_minus > _tol(p_minus, 0.0)),
                   premium_comparison_threshold=_tol(p_minus, 0.0))
    return metrics, checks


def _analyze(Z16: Any, domain: str, dimension: int, spec: dict,
             subgroup_id: str, semantic_registration: bool,
             metadata: dict | None = None) -> tuple[dict, dict]:
    verify_kernel_source()
    base = {"protocol_id": PROTOCOL_ID, "status": "not_evaluable_input",
            "domain": domain, "dimension": dimension, "source_dimension": SOURCE_DIMENSION,
            "subgroup_id": subgroup_id, "semantic_registration": semantic_registration,
            "metadata": copy.deepcopy(metadata or {}), "checks": {},
            "source_provenance_verified_by_core": False,
            "column_selection": "first_q_columns_of_original_training_PCA_scores",
            "extra_centering": False, "jitter": False, "pseudoinverse": False,
            "PCA_refit": False, "automatic_rank_reduction": False,
            "new_model_execution": False, "witness_constructed": False,
            "p_value": None, "exact_optimal_risk_measured": False,
            "primary_16_results_replaced": False,
            "scope": "retrospective_downward_truncation_sensitivity; separate_metric_per_dimension"}
    try:
        Z = _source_coordinates(Z16, domain, dimension)
        summary, arrays = spectral._analyze_spec(Z, spec)
        base["test_bank_spectral"] = summary
        base["status"] = summary["status"]
        if summary["status"] != "complete":
            base["reason"] = summary.get("reason", "frozen_kernel_not_evaluable")
            return base, arrays
        metrics, checks = _derived_metrics(Z, spec, summary, arrays)
        base.update(metrics=metrics, checks=checks)
        if not all(item["passed"] for item in checks.values()):
            base.update(status="not_evaluable_derived_checks", reason="same_metric_derived_identity_failed")
        return base, arrays
    except (ValueError, TypeError, KeyError, StopIteration, FloatingPointError, OverflowError,
            np.linalg.LinAlgError) as exc:
        base.update(status="not_evaluable_input_or_numerics",
                    reason=f"{type(exc).__name__}: {exc}")
        return base, {}


def analyze_image(Z16: Any, domain: str, dimension: int, subgroup_id: str,
                  metadata: dict | None = None) -> tuple[dict, dict]:
    """Analyze one image in a separately checkpointable unit; does not do IO."""
    matches = [s for s in subgroup_specs(domain) if s["id"] == subgroup_id]
    if len(matches) != 1:
        raise ValueError("Unknown image ID; no omitted/replaced image is permitted")
    subgroup = matches[0]
    return _analyze(Z16, domain, dimension, subgroup["spec"], subgroup_id,
                    subgroup["semantic"], metadata)


def analyze_semantic(Z16: Any, domain: str, dimension: int,
                     metadata: dict | None = None) -> tuple[dict, dict]:
    """Canonical semantic action order for matching original spectral cells.

    An image's canonical representative can reorder abstract elements relative
    to this semantic action; its full uniform spectral risks agree but array
    coordinates need not be byte-identical. Compare scalar endpoints, not raw K.
    """
    return _analyze(Z16, domain, dimension, spectral.group_spec(domain),
                    "semantic_canonical", True, metadata)


def validate_anchor(unit: dict, saved_test_bank_spectral: dict) -> dict:
    """Check recomputed dimension 16 against the immutable original summary.

    The adapter must pair this unit to the same source cell/image and authenticate
    both summaries. Never use a rounded manuscript mean or another bank's values.
    A mismatch is a stop condition, not a reason to alter tolerances or old data.
    """
    checks = {}
    if unit.get("dimension") != 16:
        return {"status": "not_evaluable", "passed": False,
                "reason": "anchor_validation_requires_original_dimension_16", "checks": checks}
    current = unit.get("test_bank_spectral", {})
    old = saved_test_bank_spectral
    if unit.get("status") != "complete" or old.get("status") != "complete":
        return {"status": "not_evaluable", "passed": False,
                "reason": "current_or_historical_anchor_not_complete", "checks": checks}
    for field in ("domain", "n_contexts", "n_states", "r", "group_order"):
        checks[field] = {"passed": current.get(field) == old.get(field),
                         "observed": current.get(field), "reference": old.get(field)}
    keys = ("gamma", "C_r", "C_untyped", "delta_type", "delta_type_normalized",
            "L_lower", "L_upper", "eta_spec", "eta_LS")
    try:
        for key in keys:
            checks["metric_" + key] = _scalar_check(current["metrics"][key], old["metrics"][key])
        current_types = {t["name"]: t for t in current["types"]}
        old_types = {t["name"]: t for t in old["types"]}
        checks["type_names"] = {"passed": current_types.keys() == old_types.keys()}
        for name in old_types:
            for key in ("e", "b"):
                checks[f"type_{name}_{key}"] = _scalar_check(current_types[name][key], old_types[name][key])
        for key in ("sigma_min", "sigma_max", "sigma_min_max_ratio"):
            checks["diagnostic_" + key] = _scalar_check(current["diagnostics"][key], old["diagnostics"][key])
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "not_evaluable", "passed": False,
                "reason": f"malformed_anchor:{type(exc).__name__}:{exc}", "checks": checks}
    passed = all(item["passed"] for item in checks.values())
    return {"status": "matched" if passed else "mismatch", "passed": passed,
            "checks": checks, "scalar_atol": spectral.SCALAR_ATOL,
            "scalar_rtol": spectral.SCALAR_RTOL, "historical_result_replaced": False}


def registration_summary(units: list[dict], domain: str, dimension: int) -> dict:
    """Frozen tolerance ranks and explicit missing/failed images; no p-values."""
    expected = subgroup_specs(domain)
    expected_ids = {s["id"] for s in expected}
    semantic_id = next(s["id"] for s in expected if s["semantic"])
    if len({u["subgroup_id"] for u in units}) != len(units):
        raise ValueError("Duplicate image units; do not average duplicate computations")
    if any(u.get("domain") != domain or u.get("dimension") != dimension for u in units):
        raise ValueError("Mixed domain/dimension in registration summary")
    if any(u["subgroup_id"] not in expected_ids for u in units):
        raise ValueError("Unexpected image ID in registration summary")
    valid = {u["subgroup_id"]: float(u["metrics"]["gamma"]) for u in units
             if u.get("status") == "complete" and math.isfinite(float(u["metrics"]["gamma"]))}
    missing = sorted(expected_ids - {u["subgroup_id"] for u in units})
    failed = sorted(u["subgroup_id"] for u in units if u["subgroup_id"] not in valid)
    output = {"domain": domain, "dimension": dimension, "planned": len(expected_ids),
              "spectral_evaluable": len(valid), "failed_spectral": len(expected_ids) - len(valid),
              "missing_image_ids": missing, "failed_image_ids": failed,
              "semantic_image_id": semantic_id, "p_value": None,
              "rank_direction": "larger gamma first", "rank_is_Lstar_order": False,
              "ties_use_frozen_scalar_tolerance": True, "d_top": None, "d_median": None}
    if semantic_id not in valid:
        return dict(output, status="semantic_registration_not_evaluable")
    semantic = valid[semantic_id]
    better = sum(v > semantic + _tol(v, semantic) for v in valid.values())
    tied = sum(abs(v - semantic) <= _tol(v, semantic) for v in valid.values())
    all_complete = len(valid) == len(expected_ids)
    output.update(status="complete_descriptive" if all_complete else "incomplete_descriptive",
                  semantic_gamma=semantic, rank_low=better + 1 if all_complete else None,
                  rank_high=better + tied if all_complete else None,
                  rank_among_evaluable_low=better + 1, rank_among_evaluable_high=better + tied,
                  all_planned_possible_rank_low=better + 1,
                  all_planned_possible_rank_high=better + tied + len(expected_ids) - len(valid),
                  failed_subgroups_have_unknown_positions=not all_complete, tied=tied)
    if all_complete:
        alternatives = [v for image, v in valid.items() if image != semantic_id]
        output.update(d_top=semantic - max(alternatives), d_median=semantic - float(np.median(alternatives)),
                      alternative_gamma_max=max(alternatives), alternative_gamma_median=float(np.median(alternatives)),
                      full_image_gamma_spread=max(valid.values()) - min(valid.values()),
                      classification="unique_leading" if better == 0 and tied == 1 else
                                     ("tied_leading" if better == 0 else "behind"))
    return output


def analyze_cell(Z16: Any, domain: str, dimension: int, metadata: dict | None = None) -> dict:
    """Convenience full cell evaluation; production adapter should checkpoint units.

    This does not load historical anchors and MUST NOT bypass the adapter's
    dimension-16 validation gate before dimension-8 execution.
    """
    units = [analyze_image(Z16, domain, dimension, s["id"], metadata)[0]
             for s in subgroup_specs(domain)]
    return {"units": units, "registration": registration_summary(units, domain, dimension)}
