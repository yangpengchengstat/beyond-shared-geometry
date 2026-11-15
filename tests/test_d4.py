from bsg.groups import d4


def test_d4_relations_noncommutativity_inverse_and_composition():
    r = d4.R
    s = d4.S
    assert d4.power(r, 4) == d4.IDENTITY
    assert d4.compose(s, s) == d4.IDENTITY
    assert d4.compose(d4.compose(s, r), s) == d4.inverse(r)
    assert d4.compose(r, s) != d4.compose(s, r)
    for g in d4.ELEMENTS:
        assert d4.compose(g, d4.inverse(g)) == d4.IDENTITY
        assert d4.compose(d4.IDENTITY, g) == g
        assert d4.compose(g, d4.IDENTITY) == g
