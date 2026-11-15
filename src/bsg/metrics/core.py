"""Shared metric helpers under the canonical row-vector convention."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalized_state_defect(a, b, eps: float = EPS) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.sum((a - b) ** 2, axis=-1) / (np.sum(b ** 2, axis=-1) + eps)))


def relative_frobenius(a, b, eps: float = EPS) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b, ord="fro") / (np.linalg.norm(b, ord="fro") + eps))


def matrix_power(mat, n: int):
    return np.linalg.matrix_power(np.asarray(mat, dtype=float), int(n))
