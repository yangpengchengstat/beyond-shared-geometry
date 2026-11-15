import numpy as np

from bsg.groups import d4
from bsg.metrics.homomorphism import homomorphism_error_data, homomorphism_error_op
from bsg.metrics.relations import d4_relation_error_data, d4_relations, relation_error_op


def permutation_matrix(g):
    P = np.zeros((4, 4))
    for i in range(4):
        P[i, d4.apply(g, i)] = 1.0
    return P


def test_exact_d4_permutation_representation_passes_op_and_data():
    ops = {g: permutation_matrix(g) for g in d4.ELEMENTS}
    Z = np.eye(4)
    mask = np.array([True, True, True, True])
    assert homomorphism_error_op(ops, d4.ELEMENTS, d4.compose) < 1e-12
    assert homomorphism_error_data(Z, mask, ops, d4.ELEMENTS, d4.compose) < 1e-12
    assert relation_error_op(ops, d4_relations()) < 1e-12
    assert d4_relation_error_data(Z, mask, ops) < 1e-12
