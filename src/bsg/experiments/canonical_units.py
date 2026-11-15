"""Reusable fitting, evaluation, and checkpoint units for the final experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from bsg.analysis.representation_theory import c7_diagnostics, d4_diagnostics, learned_exact_character_rmse
from bsg.controls.random_orthogonal import random_orthogonal_operators
from bsg.experiments.fitting import ensure_observed_pca_rank, joint_optimizer_config
from bsg.groups import c7, d4
from bsg.metrics.composition import composition_error_data, composition_error_op, c7_generator_words, d4_generator_words, group_word_generalization
from bsg.metrics.core import normalized_state_defect, relative_frobenius
from bsg.metrics.equivariance import equivariance_error
from bsg.metrics.homomorphism import homomorphism_error_data, homomorphism_error_op
from bsg.metrics.identity import identity_baseline_error
from bsg.metrics.inverse import inverse_error_data, inverse_error_op
from bsg.metrics.relations import d4_relation_components_data, d4_relation_components_op, relation_error_data, relation_error_op
from bsg.operators.group_consistent import fit_group_consistent
from bsg.operators.independent import build_pairs, fit_independent_ridge
from bsg.representations.pca import PCASubspace


def fit_base_artifact(spec, hidden, split, latent_dim: int, config: dict) -> dict:
    train_hidden = hidden[split.train_mask]
    ensure_observed_pca_rank(latent_dim, train_hidden)
    pca = PCASubspace(latent_dim, split.seed).fit(train_hidden)
    z = pca.transform(hidden)
    independent = fit_independent_ridge(
        spec.frame, z, split.train_mask, spec.elements, spec.transform_state,
        alpha=float(config["ridge_alpha"]),
    )
    return {"pca": pca, "z": z, "operators": independent}


def fit_joint_artifact(spec, base: dict, split, optimization_seed: int, config: dict) -> tuple[dict, dict]:
    operators, terms = fit_group_consistent(
        spec.name,
        spec.frame,
        base["z"],
        split.train_mask,
        base["operators"],
        spec.elements,
        spec.compose,
        spec.inverse,
        spec.identity,
        build_pairs,
        spec.transform_state,
        joint_optimizer_config(config, optimization_seed),
    )
    return {"pca": base["pca"], "z": base["z"], "operators": operators}, terms


def identity_operators(spec, dim: int) -> dict:
    return {element: np.eye(int(dim), dtype=float) for element in spec.elements}


def matched_random_operators(spec, dim: int, seed: int) -> dict:
    operators = random_orthogonal_operators(spec.elements, int(dim), int(seed))
    operators[spec.identity] = np.eye(int(dim), dtype=float)
    return operators


def _word_error(spec, z, mask, operators) -> float:
    if spec.name == "C7":
        return group_word_generalization(
            spec.frame, z, mask, {c7.GENERATOR: operators[c7.GENERATOR]},
            c7_generator_words(6), spec.compose, spec.identity, spec.transform_state,
        )
    return group_word_generalization(
        spec.frame, z, mask, {d4.R: operators[d4.R], d4.S: operators[d4.S]},
        d4_generator_words(3), spec.compose, spec.identity, spec.transform_state,
    )


def evaluate_operators(
    spec,
    artifact: dict,
    split,
    method: str,
    *,
    optimization_seed: int | None,
    random_baseline_seed: int | None,
    terms: dict | None = None,
    identity_algebra_not_applicable: bool = False,
) -> list[dict]:
    pca, z, operators = artifact["pca"], artifact["z"], artifact["operators"]
    rows = []
    for split_name, split_mask, template_ids in (
        ("validation", split.validation_mask, split.validation_ids),
        ("test", split.test_mask, split.test_ids),
    ):
        for template_id in template_ids:
            mask = split_mask & (spec.frame.template_id.to_numpy() == int(template_id))
            identity_data = normalized_state_defect(z[mask] @ operators[spec.identity], z[mask])
            identity_op = relative_frobenius(operators[spec.identity], np.eye(pca.dim))
            row = {
                "group": spec.name,
                "data_split": split_name,
                "split_seed": int(split.seed),
                "optimization_seed": optimization_seed,
                "random_baseline_seed": random_baseline_seed,
                "latent_dim": int(pca.dim),
                "method": method,
                "template_id": int(template_id),
                "pca_variance": pca.explained_variance,
                "E_eq": equivariance_error(spec.frame, z, mask, operators, spec.elements, spec.transform_state, build_pairs, identity=spec.identity),
                "E_identity_baseline": identity_baseline_error(spec.frame, z, mask, spec.elements, build_pairs, spec.transform_state, spec.identity),
                "E_identity_data": identity_data,
                "E_identity_op": identity_op,
                "E_word": _word_error(spec, z, mask, operators),
            }
            if identity_algebra_not_applicable:
                row.update({name: np.nan for name in (
                    "E_hom_data", "E_hom_op", "E_rel_data", "E_rel_op",
                    "E_comp_data", "E_comp_op", "E_inverse_data", "E_inverse_op",
                )})
            else:
                row.update({
                    "E_hom_data": homomorphism_error_data(z, mask, operators, spec.elements, spec.compose),
                    "E_hom_op": homomorphism_error_op(operators, spec.elements, spec.compose),
                    "E_rel_data": relation_error_data(spec.name, z, mask, operators),
                    "E_rel_op": relation_error_op(operators, spec.relations),
                    "E_comp_data": composition_error_data(z, mask, operators, spec.elements, spec.compose),
                    "E_comp_op": composition_error_op(operators, spec.elements, spec.compose),
                    "E_inverse_data": inverse_error_data(z, mask, operators, spec.elements, spec.inverse),
                    "E_inverse_op": inverse_error_op(operators, spec.elements, spec.inverse),
                })
                if spec.name == "D4":
                    row.update({f"E_rel_{name}_data": value for name, value in d4_relation_components_data(z, mask, operators).items()})
                    row.update({f"E_rel_{name}_op": value for name, value in d4_relation_components_op(operators).items()})
            row.update(terms or {})
            rows.append(row)
    return rows


def representation_row(spec, artifact: dict) -> dict:
    if spec.name == "C7":
        diagnostics = c7_diagnostics(artifact["operators"])
        diagnostics["learned_exact_character_rmse"] = learned_exact_character_rmse("C7", artifact["operators"])
    else:
        diagnostics = d4_diagnostics(artifact["operators"])
    return {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in diagnostics.items()
    }


def save_artifact(path: Path, artifact: dict, elements) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    pca = artifact["pca"]
    operators = np.stack([np.asarray(artifact["operators"][element], dtype=np.float64) for element in elements])
    np.savez_compressed(
        tmp,
        pca_mean=np.asarray(pca.mean_, dtype=np.float64),
        pca_components=np.asarray(pca.components_, dtype=np.float64),
        pca_explained=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
        pca_fit_rows=np.asarray(int(pca._fit_rows)),
        z=np.asarray(artifact["z"], dtype=np.float64),
        operators=operators,
    )
    tmp.replace(path)
    return path


def load_artifact(path: Path, elements, *, seed: int) -> dict:
    with np.load(path, allow_pickle=False) as data:
        components = np.asarray(data["pca_components"], dtype=float)
        pca = PCASubspace(components.shape[0], seed)
        pca.mean_ = np.asarray(data["pca_mean"], dtype=float)
        pca.components_ = components
        pca.explained_variance_ratio_ = np.asarray(data["pca_explained"], dtype=float)
        pca._fit_rows = int(data["pca_fit_rows"].item())
        z = np.asarray(data["z"], dtype=float)
        matrices = np.asarray(data["operators"], dtype=float)
    if len(matrices) != len(elements):
        raise ValueError(f"operator checkpoint element count mismatch: {path}")
    return {"pca": pca, "z": z, "operators": {element: matrices[index] for index, element in enumerate(elements)}}
