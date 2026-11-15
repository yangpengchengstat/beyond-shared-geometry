"""Canonical D4 square-corner dataset."""

from __future__ import annotations

import pandas as pd

from bsg.groups.d4 import apply

STATES = ["NW", "NE", "SE", "SW"]
TEMPLATES = [
    "The marked corner is {}.", "The selected corner is {}.", "The square position is {}.",
    "The reference corner is {}.", "The highlighted corner is {}.", "The indicated position is {}.",
    "The current square corner is {}.", "The diagram label is {}.", "The relevant corner is {}.",
    "The target corner is {}.", "The position under discussion is {}.", "The chosen location is {}.",
    "For this example, the corner is {}.", "The square's active corner is {}.",
    "The displayed location is {}.", "The corner shown is {}.",
    "The marked square location is {}.", "The active vertex is {}.",
    "The diagram's corner label is {}.", "The noted square position is {}.",
    "The indicated vertex is {}.", "The square's selected vertex is {}.",
    "The labeled corner reads {}.", "The active square position is {}.",
    "The diagram marks {}.", "The location label is {}.",
    "The square entry identifies {}.", "The vertex under discussion is {}.",
    "The specified corner is {}.", "The recorded square position is {}.",
    "The chosen vertex is {}.", "The diagram's active position is {}.",
    "The noted corner reads {}.", "The square label points to {}.",
    "The selected square location is {}.", "The record names the corner as {}.",
    "The highlighted square position is {}.", "The displayed vertex is {}.",
    "The corner label in the diagram is {}.", "The square's indicated location is {}.",
]


def build_dataset() -> pd.DataFrame:
    return pd.DataFrame([
        {"template_id": ti, "state_id": si, "state": state, "text": template.format(state), "target": state}
        for ti, template in enumerate(TEMPLATES)
        for si, state in enumerate(STATES)
    ])


def transform_state(state_id: int, group_element: tuple[int, int]) -> int:
    return apply(group_element, state_id)
