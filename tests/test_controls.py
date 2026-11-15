from bsg.controls.random_orthogonal import random_orthogonal_operators
from bsg.groups import c7
from bsg.metrics.homomorphism import homomorphism_error_op


def test_random_orthogonal_control_does_not_trivially_pass():
    ops = random_orthogonal_operators(c7.ELEMENTS, dim=4, seed=0)
    assert homomorphism_error_op(ops, c7.ELEMENTS, c7.compose) > 0.1
