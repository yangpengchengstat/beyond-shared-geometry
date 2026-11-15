import numpy as np

from bsg.groups import c7
from bsg.metrics.homomorphism import homomorphism_error_data, homomorphism_error_op
from bsg.metrics.relations import c7_relation_error_data, c7_relations, relation_error_op


def rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def test_exact_c7_representation_has_near_zero_op_and_data_errors():
    ops = {g: rotation(2 * np.pi * g / 7) for g in c7.ELEMENTS}
    Z = np.array([[1.0, 0.0], [0.2, 0.7], [-0.3, 0.4]])
    mask = np.array([True, True, True])
    assert homomorphism_error_op(ops, c7.ELEMENTS, c7.compose) < 1e-12
    assert homomorphism_error_data(Z, mask, ops, c7.ELEMENTS, c7.compose) < 1e-12
    assert relation_error_op(ops, c7_relations()) < 1e-12
    assert c7_relation_error_data(Z, mask, ops) < 1e-12
