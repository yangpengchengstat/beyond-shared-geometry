from __future__ import annotations

import numpy as np


def random_orthogonal(dim: int, rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def random_orthogonal_operators(elements, dim: int, seed: int):
    rng = np.random.default_rng(seed)
    return {g: random_orthogonal(dim, rng) for g in elements}
