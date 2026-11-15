"""Shared fitting contracts for the canonical experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bsg.operators.group_consistent import JointOptimizerConfig


@dataclass(frozen=True)
class GroupSpec:
    name: str
    frame: pd.DataFrame
    elements: tuple
    identity: object
    compose: object
    inverse: object
    transform_state: object
    relations: list


def objective_weights(config: dict) -> dict:
    weights = config.get("weights", {}) or {}
    return {
        "lambda_hom": float(weights.get("lambda_hom", 1.0)),
        "lambda_rel": float(weights.get("lambda_rel", 1.0)),
        "lambda_identity": float(weights.get("lambda_identity", 1.0)),
        "lambda_orth": float(weights.get("lambda_orth", 0.05)),
    }


def joint_optimizer_config(config: dict, seed: int) -> JointOptimizerConfig:
    return JointOptimizerConfig(
        learning_rate=float(config.get("joint_lr", 2e-3)),
        steps=int(config.get("joint_steps", 500)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
        seed=int(seed),
        **objective_weights(config),
    )


def ensure_pca_rank(latent_dim: int, n_train: int, n_features: int) -> None:
    max_rank = centered_training_rank(n_train, n_features)
    if int(latent_dim) > max_rank:
        raise ValueError(f"requested latent_dim={latent_dim} exceeds train PCA rank {max_rank}")


def centered_training_rank(n_train: int, n_features: int) -> int:
    return max(0, min(int(n_train) - 1, int(n_features)))


def ensure_observed_pca_rank(latent_dim: int, train_representations) -> None:
    train = np.asarray(train_representations, dtype=float)
    ensure_pca_rank(latent_dim, train.shape[0], train.shape[1])
    observed_rank = int(np.linalg.matrix_rank(train - train.mean(axis=0, keepdims=True)))
    if int(latent_dim) > observed_rank:
        raise ValueError(
            f"requested latent_dim={latent_dim} exceeds observed centered training rank {observed_rank}"
        )
