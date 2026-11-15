"""Homomorphism metrics, separated into data-conditioned and operator diagnostics."""

from __future__ import annotations

import numpy as np

from bsg.metrics.core import normalized_state_defect, relative_frobenius


def homomorphism_error_data(Z, mask, operators: dict, elements, compose) -> float:
    """Activation-conditioned E_hom_data.

For compose(g1,g2) meaning apply g2 then g1, row-vector convention tests
Z R_g2 R_g1 against Z R_compose(g1,g2).
"""
    X = np.asarray(Z)[mask]
    vals = []
    for g1 in elements:
        for g2 in elements:
            vals.append(normalized_state_defect(X @ operators[g2] @ operators[g1], X @ operators[compose(g1, g2)]))
    return float(np.mean(vals)) if vals else 0.0


def homomorphism_error_op(operators: dict, elements, compose) -> float:
    """Ambient operator-space Frobenius diagnostic E_hom_op."""
    vals = []
    for g1 in elements:
        for g2 in elements:
            vals.append(relative_frobenius(operators[g2] @ operators[g1], operators[compose(g1, g2)]))
    return float(np.mean(vals)) if vals else 0.0
