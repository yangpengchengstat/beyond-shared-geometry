from __future__ import annotations

import numpy as np

from bsg.metrics.core import EPS


def sorted_common_keys(operators_a: dict, operators_b: dict, keys=None) -> list:
    if keys is not None:
        return [key for key in keys if key in operators_a and key in operators_b]
    return sorted(set(operators_a) & set(operators_b), key=repr)


def fit_intertwiner(operators_a: dict, operators_b: dict, train_keys=None):
    """Fit P mapping A->B under row-vector convention.

Externally, rows z_A map to z_B as z_A @ P.T. The relation is
P R_A(g)^T ~= R_B(g)^T P internally, equivalent to row-vector
R_A(g) P.T ~= P.T R_B(g).
"""
    keys = sorted_common_keys(operators_a, operators_b, train_keys)
    if not keys:
        raise ValueError("fit_intertwiner requires at least one shared group element")
    da = operators_a[keys[0]].shape[0]
    db = operators_b[keys[0]].shape[0]
    blocks = []
    for g in keys:
        A = operators_a[g].T
        B = operators_b[g].T
        blocks.append(np.kron(A.T, np.eye(db)) - np.kron(np.eye(da), B))
    M = np.vstack(blocks)
    _, _, vh = np.linalg.svd(M, full_matrices=False)
    P = vh[-1].reshape(db, da, order="F")
    P /= np.linalg.norm(P, ord="fro") + EPS
    return P


def row_map(P):
    """Return the row-vector map T such that z_b ~= z_a @ T."""
    return np.asarray(P, dtype=float).T


def row_vector_residual(P, operators_a: dict, operators_b: dict, group_element):
    """Return R_A(g) T - T R_B(g), where T=P.T maps row latents A->B."""
    T = row_map(P)
    return np.asarray(operators_a[group_element], dtype=float) @ T - T @ np.asarray(operators_b[group_element], dtype=float)


def intertwining_error(P, operators_a: dict, operators_b: dict, eval_keys=None) -> float:
    keys = sorted_common_keys(operators_a, operators_b, eval_keys)
    if not keys:
        raise ValueError("intertwining_error requires at least one shared group element")
    nums, dens = [], []
    for g in keys:
        A = operators_a[g].T
        B = operators_b[g].T
        diff = P @ A - B @ P
        ref = B @ P
        nums.append(np.linalg.norm(diff, ord="fro") ** 2)
        dens.append(np.linalg.norm(ref, ord="fro") ** 2)
    return float(np.mean(nums) / (np.mean(dens) + EPS))


def row_map_commutation_error(T, operators_a: dict, operators_b: dict, eval_keys=None) -> float:
    """Score a row-vector map T using the same residual as a fitted intertwiner."""
    T = np.asarray(T, dtype=float)
    keys = sorted_common_keys(operators_a, operators_b, eval_keys)
    if not keys:
        raise ValueError("row_map_commutation_error requires shared group elements")
    nums, dens = [], []
    for g in keys:
        lhs = np.asarray(operators_a[g], dtype=float) @ T
        rhs = T @ np.asarray(operators_b[g], dtype=float)
        nums.append(np.linalg.norm(lhs - rhs, ord="fro") ** 2)
        dens.append(np.linalg.norm(rhs, ord="fro") ** 2)
    return float(np.mean(nums) / (np.mean(dens) + EPS))


def shuffled_operators(operators: dict, seed: int, identity=None) -> dict:
    """Return a deterministic nonidentity label permutation for a control."""
    rng = np.random.default_rng(int(seed))
    keys = sorted(operators, key=repr)
    movable = [key for key in keys if key != identity]
    shuffled = list(movable)
    rng.shuffle(shuffled)
    if len(shuffled) > 1 and shuffled == movable:
        shuffled = shuffled[1:] + shuffled[:1]
    result = {key: np.asarray(value, dtype=float) for key, value in operators.items()}
    result.update({key: np.asarray(operators[source], dtype=float) for key, source in zip(movable, shuffled)})
    return result


def shuffled_group_label_error(P, operators_a, operators_b, seed: int = 0, eval_keys=None):
    rng = np.random.default_rng(seed)
    keys = sorted_common_keys(operators_a, operators_b, eval_keys)
    shuffled = list(keys); rng.shuffle(shuffled)
    return intertwining_error(P, operators_a, {k: operators_b[s] for k, s in zip(keys, shuffled)}, keys)


def fit_direct_operator_alignment(operators_a: dict, operators_b: dict, train_keys=None):
    """Fit a generic direct operator alignment R_A(g) T ~= R_B(g).

This is a non-intertwining baseline: it aligns same-label operator matrices
directly and intentionally does not encode equivalence up to conjugacy.
The returned matrix uses the same P convention as fit_intertwiner.
"""
    keys = sorted_common_keys(operators_a, operators_b, train_keys)
    if not keys:
        raise ValueError("fit_direct_operator_alignment requires at least one shared group element")
    da = operators_a[keys[0]].shape[0]
    db = operators_b[keys[0]].shape[0]
    if da != db:
        raise ValueError("direct operator alignment currently requires equal latent dimensions")
    lhs = np.vstack([np.asarray(operators_a[g], dtype=float) for g in keys])
    rhs = np.vstack([np.asarray(operators_b[g], dtype=float) for g in keys])
    T, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    P = T.T
    P /= np.linalg.norm(P, ord="fro") + EPS
    return P


def random_orthogonal_intertwiner(dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(int(dim), int(dim))))
    return q.T
