"""Composition consistency and generator-word generalization metrics."""

from __future__ import annotations

import numpy as np

from bsg.groups import c7, d4
from bsg.metrics.core import normalized_state_defect, relative_frobenius


def composition_error_data(Z, mask, operators: dict, elements, compose, skip_identity: bool = True) -> float:
    """Conservative held-out composition consistency E_comp_data."""
    X = np.asarray(Z)[mask]
    vals = []
    for g1 in elements:
        for g2 in elements:
            if skip_identity and (g1 == elements[0] or g2 == elements[0]):
                continue
            vals.append(normalized_state_defect(X @ operators[g2] @ operators[g1], X @ operators[compose(g1, g2)]))
    return float(np.mean(vals)) if vals else 0.0


def composition_error_op(operators: dict, elements, compose, skip_identity: bool = True) -> float:
    vals = []
    for g1 in elements:
        for g2 in elements:
            if skip_identity and (g1 == elements[0] or g2 == elements[0]):
                continue
            vals.append(relative_frobenius(operators[g2] @ operators[g1], operators[compose(g1, g2)]))
    return float(np.mean(vals)) if vals else 0.0


def _lookup_rows(frame):
    return {(int(row.template_id), int(row.state_id)): i for i, row in frame.iterrows()}


def c7_generator_words(max_length: int = 6):
    return [tuple([c7.GENERATOR] * k) for k in range(1, max_length + 1)]


def d4_generator_words(max_length: int = 3):
    gens = [d4.R, d4.S]
    words = []
    def rec(prefix, depth):
        if depth > 0:
            words.append(tuple(prefix))
        if depth == max_length:
            return
        for g in gens:
            rec(prefix + [g], depth + 1)
    rec([], 0)
    return words


def word_element(word, compose, identity):
    out = identity
    for element in word:
        out = compose(element, out)
    return out


def word_operator(word, operators):
    dim = next(iter(operators.values())).shape[0]
    out = np.eye(dim)
    for element in word:
        out = out @ operators[element]
    return out


def group_word_generalization(frame, Z, mask, generator_ops: dict, words, compose, identity, transform_state) -> float:
    """Evaluate generator-only word predictions against actual transformed activations."""
    lookup = _lookup_rows(frame)
    Z = np.asarray(Z)
    vals = []
    for word in words:
        op = word_operator(word, generator_ops)
        group_element = word_element(word, compose, identity)
        X, Y = [], []
        for idx in np.where(mask)[0]:
            row = frame.iloc[idx]
            target_state = transform_state(int(row.state_id), group_element)
            j = lookup[(int(row.template_id), int(target_state))]
            X.append(Z[idx]); Y.append(Z[j])
        vals.append(normalized_state_defect(np.asarray(X) @ op, np.asarray(Y)))
    return float(np.mean(vals)) if vals else 0.0

# Backward-compatible name. Prefer composition_error_data/op.
unseen_composition_error = composition_error_op
