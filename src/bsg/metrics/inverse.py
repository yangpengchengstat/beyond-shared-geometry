"""Inverse-consistency metrics under the repository row-vector convention."""

from __future__ import annotations

import numpy as np

from bsg.metrics.core import EPS


def inverse_error_data(latents, mask, operators, elements, inverse) -> float:
    z = np.asarray(latents, dtype=float)[np.asarray(mask, dtype=bool)]
    if len(z) == 0:
        return float("nan")
    errors = []
    for g in elements:
        recovered = z @ operators[g] @ operators[inverse(g)]
        errors.append(np.mean(np.sum((recovered - z) ** 2, axis=1)) / (np.mean(np.sum(z ** 2, axis=1)) + EPS))
    return float(np.mean(errors))


def inverse_error_op(operators, elements, inverse) -> float:
    errors = []
    for g in elements:
        dim = operators[g].shape[0]
        ident = np.eye(dim)
        errors.append(np.linalg.norm(operators[g] @ operators[inverse(g)] - ident, ord="fro") ** 2 / dim)
    return float(np.mean(errors))
