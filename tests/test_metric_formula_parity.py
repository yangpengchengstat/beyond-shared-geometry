import numpy as np

from bsg.groups import c7, d4
from bsg.metrics.homomorphism import homomorphism_error_data
from bsg.metrics.relations import c7_relation_error_data, d4_relation_error_data


def test_c7_e_hom_data_matches_canonical_formula():
    ops = {k: np.array([[1.0 + 0.01 * k, 0.02 * k], [0.0, 1.0 - 0.01 * k]]) for k in c7.ELEMENTS}
    Z = np.array([[1.0, 2.0], [0.5, -0.3]])
    mask = np.array([True, False])
    X = Z[mask]
    vals = []
    for a in range(7):
        for b in range(7):
            A = X @ ops[a] @ ops[b]
            B = X @ ops[(a + b) % 7]
            vals.append(np.mean(np.sum((A - B) ** 2, axis=1) / (np.sum(B ** 2, axis=1) + 1e-12)))
    assert np.isclose(homomorphism_error_data(Z, mask, ops, c7.ELEMENTS, c7.compose), np.mean(vals))


def test_d4_e_hom_data_matches_canonical_order():
    rng = np.random.default_rng(0)
    ops = {g: np.eye(3) + 0.01 * rng.normal(size=(3, 3)) for g in d4.ELEMENTS}
    Z = rng.normal(size=(5, 3))
    mask = np.array([True, True, False, True, False])
    X = Z[mask]
    vals = []
    for g1 in d4.ELEMENTS:
        for g2 in d4.ELEMENTS:
            A = X @ ops[g2] @ ops[g1]
            B = X @ ops[d4.compose(g1, g2)]
            vals.append(np.mean(np.sum((A - B) ** 2, axis=1) / (np.sum(B ** 2, axis=1) + 1e-12)))
    assert np.isclose(homomorphism_error_data(Z, mask, ops, d4.ELEMENTS, d4.compose), np.mean(vals))


def test_relation_data_formulas_have_canonical_shape():
    ops_c7 = {k: np.eye(2) for k in c7.ELEMENTS}
    ops_d4 = {g: np.eye(2) for g in d4.ELEMENTS}
    Z = np.ones((3, 2))
    mask = np.array([True, True, True])
    assert c7_relation_error_data(Z, mask, ops_c7) == 0.0
    assert d4_relation_error_data(Z, mask, ops_d4) == 0.0
