"""Independent ridge operator estimation under row-vector convention."""

from __future__ import annotations

import numpy as np


def build_pairs(frame, Z, mask, group_element, transform_state):
    lookup = {(int(row.template_id), int(row.state_id)): idx for idx, row in frame.iterrows()}
    X, Y = [], []
    for idx in np.where(mask)[0]:
        row = frame.iloc[idx]
        target_state = transform_state(int(row.state_id), group_element)
        j = lookup[(int(row.template_id), int(target_state))]
        X.append(Z[idx])
        Y.append(Z[j])
    return np.asarray(X), np.asarray(Y)


def fit_independent_ridge(frame, Z, train_mask, elements, transform_state, alpha: float = 1e-2):
    ops = {}
    for g in elements:
        Xtr, Ytr = build_pairs(frame, Z, train_mask, g, transform_state)
        gram = Xtr.T @ Xtr + float(alpha) * np.eye(Xtr.shape[1])
        ops[g] = np.linalg.solve(gram, Xtr.T @ Ytr).astype(np.float64)
    return ops
