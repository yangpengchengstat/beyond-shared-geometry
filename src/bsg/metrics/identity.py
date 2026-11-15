from __future__ import annotations

import numpy as np

from bsg.metrics.core import normalized_state_defect


def identity_baseline_error(frame, Z, mask, elements, build_pairs, transform_state, identity):
    vals = []
    for g in elements:
        if g == identity:
            continue
        X, Y = build_pairs(frame, Z, mask, g, transform_state)
        vals.append(normalized_state_defect(X, Y))
    return float(np.mean(vals)) if vals else 0.0
