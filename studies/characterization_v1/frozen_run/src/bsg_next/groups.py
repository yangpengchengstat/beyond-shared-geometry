"""Finite groups and explicit left actions, with row-vector conventions.

compose(g, h) applies h first and g second. Thus row-vector operators obey
R[h] @ R[g] = R[compose(g,h)]. Words execute from left to right.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Iterable

Element = Any
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
CORNERS = ("NW", "NE", "SE", "SW")
PERMUTATIONS = tuple(permutations(range(3)))
D4_ELEMENTS = tuple((k, b) for k in range(4) for b in (0, 1))


def _d4_compose(g, h):
    k, b = g
    l, c = h
    return ((k + (-1 if b else 1) * l) % 4, (b + c) % 2)


def _corner(g, i):
    k, b = g
    return ((1 - int(i)) % 4 + k) % 4 if b else (int(i) + k) % 4


@dataclass(frozen=True)
class FiniteGroup:
    name: str
    elements: tuple
    identity: Element
    generators: tuple
    state_names: tuple[str, ...]

    @property
    def state_count(self):
        return len(self.state_names)

    @property
    def order(self):
        return len(self.elements)

    @property
    def size(self):
        return self.order

    def normalize(self, element):
        if isinstance(element, list):
            element = tuple(element)
        if self.name == "C7":
            aliases = {"e": 0, "identity": 0, "r": 1, "forward": 1, "backward": 6,
                       "r_inverse": 6, "r_inv": 6}
            if isinstance(element, str):
                if element in aliases:
                    element = aliases[element]
                elif element.startswith("r") and element[1:].isdigit():
                    element = int(element[1:])
                else:
                    element = int(element)
            return int(element) % 7
        if self.name.startswith("D4"):
            aliases = {"e": (0, 0), "identity": (0, 0), "r": (1, 0), "s": (0, 1),
                       "r2": (2, 0), "r3": (3, 0), "r_inverse": (3, 0), "r_inv": (3, 0)}
            if isinstance(element, str):
                if element not in aliases:
                    raise ValueError(f"Unknown D4 element {element!r}")
                element = aliases[element]
            k, b = element
            return (int(k) % 4, int(b) % 2)
        aliases = {"e": (0, 1, 2), "identity": (0, 1, 2),
                   "swap_12": (1, 0, 2), "swap12": (1, 0, 2),
                   "swap_23": (0, 2, 1), "swap23": (0, 2, 1)}
        if isinstance(element, str):
            if element not in aliases:
                raise ValueError(f"Unknown S3 element {element!r}")
            element = aliases[element]
        element = tuple(element)
        if element not in PERMUTATIONS:
            raise ValueError(f"Not an S3 permutation: {element}")
        return element

    def compose(self, g1, g2):
        g1, g2 = self.normalize(g1), self.normalize(g2)
        if self.name == "C7":
            return (g1 + g2) % 7
        if self.name.startswith("D4"):
            return _d4_compose(g1, g2)
        return tuple(g1[g2[i]] for i in range(3))

    multiply = compose

    def inverse(self, g):
        g = self.normalize(g)
        return next(h for h in self.elements if self.compose(g, h) == self.identity)

    def apply(self, g, state_id: int):
        g = self.normalize(g)
        i = int(state_id)
        if not 0 <= i < self.state_count:
            raise ValueError(f"State {i} outside {self.name}")
        if self.name == "C7":
            return (i + g) % 7
        if self.name == "D4_corner":
            return _corner(g, i)
        if self.name == "D4_free8":
            return self.elements.index(self.compose(g, self.elements[i]))
        assignment = PERMUTATIONS[i]
        moved = [None] * 3
        for old_role in range(3):
            moved[g[old_role]] = assignment[old_role]
        return PERMUTATIONS.index(tuple(moved))

    act = apply

    def apply_word(self, state_id: int, word: Iterable):
        for g in word:
            state_id = self.apply(g, state_id)
        return int(state_id)

    def evaluate_word(self, word: Iterable):
        result = self.identity
        for g in word:
            result = self.compose(g, result)
        return result

    def action_table(self):
        return [[self.apply(g, i) for i in range(self.state_count)] for g in self.elements]

    def multiplication_table(self):
        return [[self.elements.index(self.compose(g, h)) for h in self.elements] for g in self.elements]

    def as_dict(self):
        return {"name": self.name, "elements": self.elements, "identity": self.identity,
                "generators": self.generators, "state_names": self.state_names,
                "action_table": self.action_table(), "multiplication_table": self.multiplication_table(),
                "convention": "compose(g,h) applies h then g; words left to right; row R[h]@R[g]"}


def pose_tuple(state_id: int, marks=("A", "B", "C", "D")):
    """Marks at (NW,NE,SE,SW), after actively transforming the reference."""
    if len(set(marks)) != 4:
        raise ValueError("A free D4 pose requires four distinct marks")
    g = D4_ELEMENTS[int(state_id)]
    result = [None] * 4
    for i, mark in enumerate(marks):
        result[_corner(g, i)] = mark
    return tuple(result)


def get_group(name: str) -> FiniteGroup:
    key = str(name).lower().replace("-", "_")
    if key in {"c4xc4", "c4_x_c4", "toroid"}:
        from .extensions import ProductGroup
        return ProductGroup()
    if key in {"c7", "c7_native", "weekdays"}:
        return FiniteGroup("C7", tuple(range(7)), 0, (1,), DAYS)
    if key in {"d4", "d4_corner", "d4corner", "corners"}:
        return FiniteGroup("D4_corner", D4_ELEMENTS, (0, 0), ((1, 0), (0, 1)), CORNERS)
    if key in {"d4_free8", "d4_free8_native", "free8", "d4native"}:
        return FiniteGroup("D4_free8", D4_ELEMENTS, (0, 0), ((1, 0), (0, 1)),
                           tuple("(" + ", ".join(pose_tuple(i)) + ")" for i in range(8)))
    if key in {"s3", "s3_role", "s3role"}:
        return FiniteGroup("S3_role", PERMUTATIONS, (0, 1, 2), ((1, 0, 2), (0, 2, 1)),
                           tuple("".join(map(str, p)) for p in PERMUTATIONS))
    raise ValueError(f"Unknown finite-action domain {name!r}")


Group = FiniteGroup
