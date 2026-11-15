"""PCA helpers that enforce train-only fitting."""

from __future__ import annotations

import numpy as np


class PCASubspace:
    def __init__(self, n_components: int, seed: int = 42):
        self.n_components = int(n_components)
        self.seed = int(seed)
        self._fit_rows = None
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, train_representations):
        H = np.asarray(train_representations, dtype=float)
        r = min(self.n_components, H.shape[0] - 1, H.shape[1])
        if r < 1:
            raise ValueError("PCA requires at least two training rows and one feature")
        self.mean_ = H.mean(axis=0)
        centered = H - self.mean_
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[:r]
        denom = max(H.shape[0] - 1, 1)
        variances = (s ** 2) / denom
        total = float(variances.sum())
        self.explained_variance_ratio_ = variances[:r] / total if total > 0 else np.zeros(r)
        self._fit_rows = H.shape[0]
        return self

    def transform(self, representations):
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCASubspace must be fit on training data before transform")
        return (np.asarray(representations, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform_train_apply_all(self, train_representations, all_representations):
        self.fit(train_representations)
        return self.transform(all_representations)

    @property
    def dim(self) -> int:
        if self.components_ is None:
            raise RuntimeError("PCA is not fit")
        return int(self.components_.shape[0])

    @property
    def explained_variance(self) -> float:
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("PCA is not fit")
        return float(self.explained_variance_ratio_.sum())
