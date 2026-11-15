"""Representation-theoretic diagnostics for learned C7 and D4 operators."""

from __future__ import annotations

import numpy as np

from bsg.groups import c7, d4
from bsg.metrics.core import relative_frobenius


def c7_diagnostics(operators: dict) -> dict:
    generator = np.asarray(operators[c7.GENERATOR], dtype=float)
    eigenvalues = np.linalg.eigvals(generator)
    angles = np.angle(eigenvalues)
    expected = 2 * np.pi * np.arange(c7.ORDER) / c7.ORDER
    angle_errors = [min(abs(((a - e + np.pi) % (2 * np.pi)) - np.pi) for e in expected) for a in angles]
    conjugate_pairs = sum(1 for value in eigenvalues if value.imag > 1e-8 and np.min(np.abs(eigenvalues - value.conjugate())) < 1e-6)
    return {
        "generator_spectral_radius": float(np.max(np.abs(eigenvalues))),
        "generator_unit_circle_rmse": float(np.sqrt(np.mean((np.abs(eigenvalues) - 1.0) ** 2))),
        "generator_root_angle_rmse": float(np.sqrt(np.mean(np.square(angle_errors)))),
        "real_rotation_block_count": int(conjugate_pairs),
        "candidate_real_rotation_dimension": int(2 * conjugate_pairs),
        "cycle_operator_error": float(np.linalg.norm(np.linalg.matrix_power(generator, 7) - np.eye(generator.shape[0]), ord="fro") / np.sqrt(generator.shape[0])),
        "eigenvalues": [[float(v.real), float(v.imag)] for v in eigenvalues],
    }


_D4_IRREPS = {
    "A1": {(0, 0): 1, (1, 0): 1, (2, 0): 1, (3, 0): 1, (0, 1): 1, (1, 1): 1, (2, 1): 1, (3, 1): 1},
    "A2": {(0, 0): 1, (1, 0): 1, (2, 0): 1, (3, 0): 1, (0, 1): -1, (1, 1): -1, (2, 1): -1, (3, 1): -1},
    "B1": {(k, b): ((-1) ** k) * (1 if b == 0 else 1) for k, b in d4.ELEMENTS},
    "B2": {(k, b): ((-1) ** k) * (1 if b == 0 else -1) for k, b in d4.ELEMENTS},
    "E": {(0, 0): 2, (1, 0): 0, (2, 0): -2, (3, 0): 0, (0, 1): 0, (1, 1): 0, (2, 1): 0, (3, 1): 0},
}


def _exact_permutation_character(group: str) -> dict:
    if group == "C7":
        return {g: float(sum(c7.apply(g, state) == state for state in range(7))) for g in c7.ELEMENTS}
    return {g: float(sum(d4.apply(g, state) == state for state in range(4))) for g in d4.ELEMENTS}


def learned_exact_character_rmse(group: str, operators: dict) -> float:
    exact = _exact_permutation_character(group)
    dim = next(iter(operators.values())).shape[0]
    exact_dim = 7 if group == "C7" else 4
    values = [(np.trace(operators[g]).real / dim) - (exact[g] / exact_dim) for g in operators]
    return float(np.sqrt(np.mean(np.square(values))))


def d4_diagnostics(operators: dict) -> dict:
    r = np.asarray(operators[d4.R], dtype=float)
    s = np.asarray(operators[d4.S], dtype=float)
    identity = np.asarray(operators[d4.IDENTITY], dtype=float)
    r_inverse = np.asarray(operators[d4.inverse(d4.R)], dtype=float)
    r_eigenvalues = np.linalg.eigvals(r)
    s_eigenvalues = np.linalg.eigvals(s)
    character = {g: float(np.trace(np.asarray(operators[g], dtype=float))) for g in d4.ELEMENTS}
    multiplicities = {
        name: float(sum(character[g] * chars[g] for g in d4.ELEMENTS) / 8.0)
        for name, chars in _D4_IRREPS.items()
    }
    return {
        "r_eigenvalues": [[float(v.real), float(v.imag)] for v in r_eigenvalues],
        "s_eigenvalues": [[float(v.real), float(v.imag)] for v in s_eigenvalues],
        "irrep_multiplicities": multiplicities,
        "character": {str(g): value for g, value in character.items()},
        "r_fourth_power_error": relative_frobenius(np.linalg.matrix_power(r, 4), identity),
        "s_square_error": relative_frobenius(s @ s, identity),
        "srs_inverse_error": relative_frobenius(s @ r @ s, r_inverse),
        "learned_exact_character_rmse": learned_exact_character_rmse("D4", operators),
    }
