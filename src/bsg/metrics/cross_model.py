"""Cross-model geometry and algebra comparison metrics."""

from __future__ import annotations

import numpy as np

from bsg.metrics.core import EPS


def linear_cka(x, y) -> float:
    x = np.asarray(x, dtype=float) - np.asarray(x, dtype=float).mean(axis=0, keepdims=True)
    y = np.asarray(y, dtype=float) - np.asarray(y, dtype=float).mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    return float(cross / (np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro") + EPS))


def orthogonal_procrustes(x, y):
    """Return the row-vector orthogonal map x @ Q ~= y and normalized error."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("orthogonal Procrustes requires equal sample and feature shapes")
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
    q = u @ vt
    error = np.linalg.norm(xc @ q - yc, ord="fro") ** 2 / (np.linalg.norm(yc, ord="fro") ** 2 + EPS)
    return q, float(error)


def geometry_algebra_gap(procrustes_error: float, intertwiner_error: float) -> float:
    """Positive values mean algebraic alignment is worse than geometric alignment."""
    return float(intertwiner_error - procrustes_error)
