"""Canonical group-consistent joint operator optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointOptimizerConfig:
    learning_rate: float = 2e-3
    steps: int = 500
    lambda_hom: float = 1.0
    lambda_rel: float = 1.0
    lambda_identity: float = 1.0
    lambda_orth: float = 0.05
    weight_decay: float = 1e-5
    seed: int = 42
    device: str | None = None


def _torch_device(requested: str | None):
    import torch
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_torch_seed(seed: int):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _JointOperators:
    def __init__(self, elements, init, device):
        import torch
        self.elements = list(elements)
        self.index = {g: i for i, g in enumerate(self.elements)}
        mats = [torch.tensor(init[g], dtype=torch.float32, device=device) for g in self.elements]
        self.R = torch.nn.Parameter(torch.stack(mats))
        self.dim = int(self.R.shape[-1])

    def parameters(self):
        return [self.R]

    def mat(self, g):
        return self.R[self.index[g]]

    def as_numpy(self):
        return {g: self.mat(g).detach().cpu().numpy().astype(np.float64) for g in self.elements}


def _orth_loss(net):
    import torch
    I = torch.eye(net.dim, device=net.R.device)
    return torch.stack([((net.mat(g).T @ net.mat(g) - I) ** 2).mean() for g in net.elements]).mean()


def _identity_loss(net, identity):
    import torch
    I = torch.eye(net.dim, device=net.R.device)
    return ((net.mat(identity) - I) ** 2).mean()


def _hom_loss(net, compose):
    import torch
    return torch.stack([
        ((net.mat(g2) @ net.mat(g1) - net.mat(compose(g1, g2))) ** 2).mean()
        for g1 in net.elements for g2 in net.elements
    ]).mean()


def _c7_rel_loss(net):
    import torch
    I = torch.eye(net.dim, device=net.R.device)
    return ((torch.linalg.matrix_power(net.mat(1), 7) - I) ** 2).mean()


def _d4_rel_loss(net, inverse):
    import torch
    r = (1, 0); s = (0, 1)
    I = torch.eye(net.dim, device=net.R.device)
    return (
        ((torch.linalg.matrix_power(net.mat(r), 4) - I) ** 2).mean()
        + ((net.mat(s) @ net.mat(s) - I) ** 2).mean()
        + ((net.mat(s) @ net.mat(r) @ net.mat(s) - net.mat(inverse(r))) ** 2).mean()
    ) / 3.0


def fit_group_consistent(
    group: str,
    frame,
    Z,
    train_mask,
    init_ops: dict,
    elements,
    compose,
    inverse,
    identity,
    build_pairs,
    transform_state,
    config: JointOptimizerConfig,
):
    """Fit full-operator joint model initialized from independent ridge ops.

This uses the validated objective: data fitting over all group
transformations plus homomorphism, defining-relation, identity, and
orthogonality losses. Operators use row vectors throughout.
"""
    import torch

    if config.steps <= 0:
        raise ValueError("Joint optimizer steps must be positive")
    _set_torch_seed(config.seed)
    device = _torch_device(config.device)
    net = _JointOperators(elements, init_ops, device)

    Xs, Ys, Gs = [], [], []
    for gi, g in enumerate(elements):
        X, Y = build_pairs(frame, Z, train_mask, g, transform_state)
        Xs.append(X); Ys.append(Y); Gs.append(np.full(len(X), gi))
    X = torch.tensor(np.concatenate(Xs), dtype=torch.float32, device=device)
    Y = torch.tensor(np.concatenate(Ys), dtype=torch.float32, device=device)
    G = torch.tensor(np.concatenate(Gs), dtype=torch.long, device=device)

    opt = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    last_terms = {}
    for _ in range(config.steps):
        opt.zero_grad()
        pred = torch.empty_like(Y)
        for gi, g in enumerate(elements):
            pred[G == gi] = X[G == gi] @ net.mat(g)
        data = ((pred - Y) ** 2).mean()
        hom = _hom_loss(net, compose)
        rel = _c7_rel_loss(net) if group == "C7" else _d4_rel_loss(net, inverse)
        ident = _identity_loss(net, identity)
        orth = _orth_loss(net)
        loss = data + config.lambda_hom * hom + config.lambda_rel * rel + config.lambda_identity * ident + config.lambda_orth * orth
        loss.backward(); opt.step()
        last_terms = {
            "L_data": float(data.detach().cpu()),
            "L_hom": float(hom.detach().cpu()),
            "L_rel": float(rel.detach().cpu()),
            "L_identity": float(ident.detach().cpu()),
            "L_orth": float(orth.detach().cpu()),
            "L_total": float(loss.detach().cpu()),
        }
    ops = net.as_numpy()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ops, last_terms


def fit_group_consistent_placeholder(*args, **kwargs):
    raise RuntimeError("Placeholder removed: use fit_group_consistent with a JointOptimizerConfig.")
