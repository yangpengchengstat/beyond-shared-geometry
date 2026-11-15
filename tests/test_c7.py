from bsg.groups import c7


def test_c7_identity_inverse_r7_and_composition():
    for g in c7.ELEMENTS:
        assert c7.compose(c7.IDENTITY, g) == g
        assert c7.compose(g, c7.IDENTITY) == g
        assert c7.compose(g, c7.inverse(g)) == c7.IDENTITY
    assert c7.power(c7.GENERATOR, 7) == c7.IDENTITY
    assert c7.compose(5, 4) == 2
