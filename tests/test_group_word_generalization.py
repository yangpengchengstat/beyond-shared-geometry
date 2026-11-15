import numpy as np

from bsg.data.c7 import build_dataset as build_c7, transform_state as c7_transform
from bsg.data.d4 import build_dataset as build_d4, transform_state as d4_transform
from bsg.groups import c7, d4
from bsg.metrics.composition import c7_generator_words, d4_generator_words, group_word_generalization


def c7_rot(k):
    theta = 2 * np.pi * k / 7
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def d4_perm(g):
    P = np.zeros((4, 4))
    for i in range(4): P[i, d4.apply(g, i)] = 1.0
    return P


def test_c7_generator_only_word_generalization_exact():
    frame = build_c7()
    Z = np.vstack([c7_rot(int(r.state_id))[0] for _, r in frame.iterrows()])
    mask = np.ones(len(frame), dtype=bool)
    ops = {c7.GENERATOR: c7_rot(1)}
    assert group_word_generalization(frame, Z, mask, ops, c7_generator_words(6), c7.compose, c7.IDENTITY, c7_transform) < 1e-12


def test_d4_generator_only_word_generalization_exact_and_noncommutative_words():
    frame = build_d4()
    Z = np.eye(4)[frame.state_id.to_numpy()]
    mask = np.ones(len(frame), dtype=bool)
    ops = {d4.R: d4_perm(d4.R), d4.S: d4_perm(d4.S)}
    assert d4.compose(d4.R, d4.S) != d4.compose(d4.S, d4.R)
    assert group_word_generalization(frame, Z, mask, ops, d4_generator_words(3), d4.compose, d4.IDENTITY, d4_transform) < 1e-12
