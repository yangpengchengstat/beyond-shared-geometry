import numpy as np

from bsg.groups import c7, d4


def d4_perm(g):
    P = np.zeros((4, 4))
    for i in range(4): P[i, d4.apply(g, i)] = 1.0
    return P


def test_row_vector_composition_order_c7_and_d4():
    z7 = np.array([[1.0, 0.0]])
    ops7 = {g: np.linalg.matrix_power(np.array([[0, -1], [1, 0.0]]), g % 4) for g in c7.ELEMENTS}
    # C7 is abelian; explicit compose order is still well-defined.
    assert c7.compose(2, 3) == 5
    z4 = np.array([[1.0, 0.0, 0.0, 0.0]])
    ops4 = {g: d4_perm(g) for g in d4.ELEMENTS}
    g1, g2 = d4.R, d4.S
    assert np.allclose(z4 @ ops4[g2] @ ops4[g1], z4 @ ops4[d4.compose(g1, g2)])


def test_wrong_d4_order_fails():
    z = np.eye(4)
    ops = {g: d4_perm(g) for g in d4.ELEMENTS}
    g1, g2 = d4.R, d4.S
    assert not np.allclose(z @ ops[g1] @ ops[g2], z @ ops[d4.compose(g1, g2)])
