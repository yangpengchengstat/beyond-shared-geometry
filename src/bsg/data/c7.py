"""Canonical C7 weekday dataset."""

from __future__ import annotations

import pandas as pd

from bsg.groups.c7 import apply

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TEMPLATES = [
    "The current day is {}.", "Today is {}.", "The calendar shows {}.",
    "The selected weekday is {}.", "The starting day is {}.", "The event happens on {}.",
    "The meeting is scheduled for {}.", "The reference day is {}.", "The displayed weekday is {}.",
    "The sequence begins on {}.", "The marked weekday is {}.", "The relevant day is {}.",
    "For this example, the day is {}.", "The designated weekday is {}.", "The schedule lists {}.",
    "The record shows {}.", "The weekday under discussion is {}.", "The highlighted day is {}.",
    "The noted day of the week is {}.", "The planned weekday reads {}.",
    "The log identifies the day as {}.", "The appointment falls on {}.",
    "The timetable points to {}.", "The stated weekday is {}.", "The named day is {}.",
    "The workweek marker is {}.", "The calendar entry names {}.",
    "The day assigned to the task is {}.", "The listed day of the week is {}.",
    "The chosen weekday reads {}.", "The day's label is {}.",
    "The weekly position is labeled {}.", "The noted calendar day is {}.",
    "The schedule's weekday is {}.", "The current weekday label is {}.",
    "The recorded day of the week is {}.", "The week's selected day is {}.",
    "The day shown in the entry is {}.", "The specified day is {}.",
    "The weekday named in the note is {}.",
]


def build_dataset() -> pd.DataFrame:
    return pd.DataFrame([
        {"template_id": ti, "state_id": di, "day_id": di, "state": day, "day": day, "text": template.format(day), "target": day}
        for ti, template in enumerate(TEMPLATES)
        for di, day in enumerate(DAYS)
    ])


def transform_state(state_id: int, group_element: int) -> int:
    return apply(group_element, state_id)
