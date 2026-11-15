"""Canonical D4 convention defined by action lookup.

Elements are pairs (k, b), acting on square-corner state ids 0:NW, 1:NE,
2:SE, 3:SW. The action applies reflection s first when b=1, then k
clockwise quarter turns r. compose(g1, g2) means apply g2 first, then g1.
"""

from __future__ import annotations

IDENTITY = (0, 0)
R = (1, 0)
S = (0, 1)
ELEMENTS = tuple((k, b) for k in range(4) for b in (0, 1))


def r_state(i: int) -> int:
    return (int(i) + 1) % 4


def s_state(i: int) -> int:
    return {0: 1, 1: 0, 2: 3, 3: 2}[int(i)]


def apply(g: tuple[int, int], state_id: int) -> int:
    k, b = g
    x = s_state(state_id) if b else int(state_id)
    for _ in range(k % 4):
        x = r_state(x)
    return x


def compose(g1: tuple[int, int], g2: tuple[int, int]) -> tuple[int, int]:
    target = [apply(g1, apply(g2, i)) for i in range(4)]
    for h in ELEMENTS:
        if [apply(h, i) for i in range(4)] == target:
            return h
    raise RuntimeError("D4 composition lookup failed")


def inverse(g: tuple[int, int]) -> tuple[int, int]:
    for h in ELEMENTS:
        if compose(g, h) == IDENTITY and compose(h, g) == IDENTITY:
            return h
    raise RuntimeError(f"D4 inverse lookup failed for {g!r}")


def power(g: tuple[int, int], n: int) -> tuple[int, int]:
    out = IDENTITY
    for _ in range(n):
        out = compose(g, out)
    return out


def row_word_product(word) -> tuple[int, int]:
    """Return the group element represented by row-vector matrix order."""
    out = IDENTITY
    for element in word:
        out = compose(element, out)
    return out
