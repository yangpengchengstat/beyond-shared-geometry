"""Defining-relation metrics for C7 and D4."""

from __future__ import annotations

import numpy as np

from bsg.groups import d4
from bsg.metrics.core import matrix_power, normalized_state_defect, relative_frobenius


def c7_relations():
    return [((1, 1, 1, 1, 1, 1, 1), 0)]


def d4_relations():
    r = d4.R
    s = d4.S
    return [((r, r, r, r), d4.IDENTITY), ((s, s), d4.IDENTITY), ((s, r, s), d4.inverse(r))]


def relation_error_op(operators: dict, relations: list[tuple[tuple, object]]) -> float:
    dim = next(iter(operators.values())).shape[0]
    vals = []
    for word, target in relations:
        prod = np.eye(dim)
        for element in word:
            prod = prod @ operators[element]
        vals.append(relative_frobenius(prod, operators[target]))
    return float(np.mean(vals)) if vals else 0.0


def c7_relation_error_data(Z, mask, operators: dict) -> float:
    X = np.asarray(Z)[mask]
    G = operators[1]
    order = normalized_state_defect(X @ matrix_power(G, 7), X)
    powers = np.mean([normalized_state_defect(X @ matrix_power(G, k), X @ operators[k]) for k in range(7)])
    inv = np.mean([normalized_state_defect(X @ operators[k] @ operators[(-k) % 7], X) for k in range(1, 7)])
    return float(np.mean([order, powers, inv]))


def d4_relation_error_data(Z, mask, operators: dict) -> float:
    return float(np.mean(list(d4_relation_components_data(Z, mask, operators).values())))


def d4_relation_components_data(Z, mask, operators: dict) -> dict[str, float]:
    X = np.asarray(Z)[mask]
    Rr = operators[d4.R]
    Rs = operators[d4.S]
    Rinv = operators[d4.inverse(d4.R)]
    r4 = normalized_state_defect(X @ matrix_power(Rr, 4), X)
    s2 = normalized_state_defect(X @ Rs @ Rs, X)
    srs = normalized_state_defect(X @ Rs @ Rr @ Rs, X @ Rinv)
    return {"r4": r4, "s2": s2, "srs_r_inverse": srs}


def d4_relation_components_op(operators: dict) -> dict[str, float]:
    Rr = operators[d4.R]
    Rs = operators[d4.S]
    identity = operators[d4.IDENTITY]
    inverse = operators[d4.inverse(d4.R)]
    return {
        "r4": relative_frobenius(matrix_power(Rr, 4), identity),
        "s2": relative_frobenius(Rs @ Rs, identity),
        "srs_r_inverse": relative_frobenius(Rs @ Rr @ Rs, inverse),
    }


def relation_error_data(group: str, Z, mask, operators: dict) -> float:
    if group == "C7":
        return c7_relation_error_data(Z, mask, operators)
    if group == "D4":
        return d4_relation_error_data(Z, mask, operators)
    raise ValueError(f"Unsupported group {group!r}")

# Backward-compatible auxiliary name. Prefer relation_error_op or _data.
relation_error = relation_error_op
