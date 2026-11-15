"""Canonical cyclic group C7 used for weekday transformations."""

from __future__ import annotations

ORDER = 7
IDENTITY = 0
ELEMENTS = tuple(range(ORDER))
GENERATOR = 1


def compose(g1: int, g2: int) -> int:
    """Return g1 after g2, equivalent to exponent addition modulo 7."""
    return (int(g1) + int(g2)) % ORDER


def inverse(g: int) -> int:
    return (-int(g)) % ORDER


def power(g: int, n: int) -> int:
    out = IDENTITY
    for _ in range(n):
        out = compose(g, out)
    return out


def apply(g: int, state_id: int) -> int:
    return compose(g, state_id)
