"""Deterministic template-level train/validation/test partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemplateSplit:
    train_mask: np.ndarray
    validation_mask: np.ndarray
    test_mask: np.ndarray
    train_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    seed: int

    @property
    def identity(self) -> dict:
        return {
            "seed": int(self.seed),
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "test_ids": list(self.test_ids),
        }


def split_templates_three_way(
    frame,
    *,
    seed: int,
    train_templates: int,
    validation_templates: int,
    test_templates: int,
) -> TemplateSplit:
    ids = np.array(sorted(int(value) for value in frame.template_id.unique()), dtype=int)
    counts = (int(train_templates), int(validation_templates), int(test_templates))
    if min(counts) < 1:
        raise ValueError("train, validation, and test template counts must all be positive")
    if sum(counts) != len(ids):
        raise ValueError(f"template split counts {counts} do not sum to {len(ids)} templates")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(ids)
    n_train, n_validation, _ = counts
    train_ids = tuple(sorted(ids[:n_train].tolist()))
    validation_ids = tuple(sorted(ids[n_train:n_train + n_validation].tolist()))
    test_ids = tuple(sorted(ids[n_train + n_validation:].tolist()))
    partitions = [set(train_ids), set(validation_ids), set(test_ids)]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("template leakage across train/validation/test")
    if set.union(*partitions) != set(map(int, ids)):
        raise ValueError("template partition is incomplete")
    return TemplateSplit(
        train_mask=frame.template_id.isin(train_ids).to_numpy(),
        validation_mask=frame.template_id.isin(validation_ids).to_numpy(),
        test_mask=frame.template_id.isin(test_ids).to_numpy(),
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        seed=int(seed),
    )


def split_templates(frame, seed: int, train_fraction: float = 0.70):
    """Compatibility wrapper for historical two-way CPU contracts."""
    ids = np.array(sorted(frame.template_id.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = min(max(int(round(len(ids) * train_fraction)), 1), len(ids) - 1)
    train_ids = set(ids[:n_train].tolist())
    test_ids = set(ids[n_train:].tolist())
    train_mask = frame.template_id.isin(train_ids).to_numpy()
    test_mask = frame.template_id.isin(test_ids).to_numpy()
    return train_mask, test_mask, sorted(train_ids), sorted(test_ids)
