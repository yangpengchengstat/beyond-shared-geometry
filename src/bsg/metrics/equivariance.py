from __future__ import annotations

import numpy as np

from bsg.metrics.core import normalized_state_defect


def equivariance_error(frame, Z, mask, operators, elements, transform_state, build_pairs, identity=None):
    vals = []
    for g in elements:
        if identity is not None and g == identity:
            continue
        X, Y = build_pairs(frame, Z, mask, g, transform_state)
        vals.append(normalized_state_defect(X @ operators[g], Y))
    return float(np.mean(vals)) if vals else 0.0
