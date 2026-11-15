"""Input-embedding category-code floor; never part of the primary model panel."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .data import action_target_indices, select_phase_records, load_sealed_dataset, dataset_identity
from .groups import get_group
from .io import atomic_json, prepare_run, read_json, file_sha256, identity
from .config import model_key


def _development_rows(config, cache_root=None, records_by_domain=None, dataset_root=None):
    if records_by_domain is not None:
        return {get_group(name).name: select_phase_records(list(rows), "development") for name, rows in records_by_domain.items()}
    if dataset_root is not None:
        paths = read_json(Path(dataset_root)/"datasets.json")["datasets"]
        return {get_group(name).name: select_phase_records(load_sealed_dataset(path), "development") for name, path in paths.items()}
    from .extraction import read_cache
    if cache_root is not None:
        root = Path(cache_root)
        paths = read_json(root/"caches.json")["cache_dirs"] if (root/"caches.json").exists() else [p.parent for p in root.rglob("manifest.json")]
    else:
        paths = config.get("structural", {}).get("cache_manifests", [])
    found = {}
    for path in paths:
        rows, _arrays, _manifest = read_cache(path)
        domain = get_group(rows[0]["domain"]).name
        if domain not in found:
            found[domain] = select_phase_records(rows, "development")
        elif dataset_identity(found[domain]) != dataset_identity(select_phase_records(rows, "development")):
            raise ValueError("Embedding floor source models have different development rows")
    if not found:
        raise ValueError("Embedding floor requires verified development caches, a sealed dataset_root, or explicit records_by_domain")
    return found


def _weighted_duplicate_pca(hidden, train):
    """Exact duplicated-row covariance, avoiding an unnecessarily large SVD."""
    indices = np.flatnonzero(train)
    reference = np.asarray(hidden[indices[0]])
    if all(np.all(np.asarray(hidden[part]) == reference) for part in np.array_split(indices, max(1, len(indices)//1024+1))):
        return reference.astype(np.float64), np.empty((0, hidden.shape[1])), np.asarray([0.0]), 0, 1
    values, counts = np.unique(np.asarray(hidden[train], dtype=np.float64), axis=0, return_counts=True)
    center = np.average(values, axis=0, weights=counts)
    weighted = (values-center) * np.sqrt(counts[:, None])
    _u, singular, vt = np.linalg.svd(weighted, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-12)) if len(singular) and singular[0] > 0 else 0
    return center, vt, singular, rank, len(values)


def analyze_embedding_floor(rows, hidden, domain):
    from .estimators import fit_ridge
    group = get_group(domain)
    train = np.asarray([row["split"] == "train" for row in rows])
    if not train.any():
        raise ValueError("Embedding floor needs construction rows")
    center, vt, singular, rank, unique = _weighted_duplicate_pca(hidden, train)
    result = {"group": group.name, "layer": 0, "readout": "final_token" if group.name == "S3_role" else "span_mean",
              "training_rows": int(train.sum()), "unique_training_vectors": int(unique),
              "observed_training_rank": rank, "singular_values": singular.tolist(),
              "primary_panel_eligible": False, "transformer_forwards": 0,
              "scope": "input_embedding_category_code_positive_control; not_native_group_computation"}
    if rank == 0:
        return {**result, "status": "zero_rank", "q": None, "operators_fitted": False}
    if group.name == "S3_role":
        return {**result, "status": "nonconstant_final_token_embedding", "q": None, "operators_fitted": False,
                "interpretation": "check tokenizer/readout variation; no forced q16 fit"}
    q = group.state_count-1
    if rank < q:
        return {**result, "status": "category_rank_deficient", "requested_q": q, "q": None, "operators_fitted": False}
    basis = vt[:q]
    z = np.empty((len(rows), q), dtype=np.float64)
    for start in range(0, len(rows), 4096):
        z[start:start+4096] = (np.asarray(hidden[start:start+4096], float)-center)@basis.T
    targets = action_target_indices(rows, group)
    models = {g: fit_ridge(z[train], z[np.asarray(indices)[train]], 0) for g, indices in targets.items()}
    energy = max(float(np.mean(np.sum(z[train]**2, axis=1))), 1e-12)
    scores = {}
    for split in ("validation", "pilot"):
        mask = np.asarray([row["split"] == split for row in rows])
        if not mask.any():
            continue
        losses = []
        for g, indices in targets.items():
            changed = mask & (np.arange(len(rows)) != np.asarray(indices))
            if changed.any():
                losses.extend(np.sum((models[g].predict(z[changed])-z[np.asarray(indices)[changed]])**2, axis=1).tolist())
        scores[split] = {"rows": int(mask.sum()), "mean_squared_target_error": float(np.mean(losses)),
                         "training_energy_normalized_error": float(np.mean(losses)/energy)}
    defects = [np.linalg.norm(models[h].coef@models[g].coef-models[group.compose(g,h)].coef)**2 /
               max(np.linalg.norm(models[group.compose(g,h)].coef)**2, 1e-12) for g in group.elements for h in group.elements]
    return {**result, "status": "complete", "q": q, "operators_fitted": True,
            "estimator": "independent minimum_norm_ridge_lambda0", "operator_defect_mean": float(np.mean(defects)),
            "scores": scores, "pca_mean_sha256": identity(center), "pca_basis_sha256": identity(basis)}


def run_embedding_floor(config, output, cache_root=None, *, runtime_factory=None, records_by_domain=None, dataset_root=None):
    from .hf import runtime_from_config
    from .extraction import extract_features, read_cache
    rows_by_domain = _development_rows(config, cache_root, records_by_domain, dataset_root)
    cfg = config.get("embedding", {})
    wanted = {get_group(name).name for name in cfg.get("groups", ["C7", "D4", "S3"])}
    selected = {name: rows for name, rows in rows_by_domain.items() if name in wanted}
    if not selected:
        raise ValueError("No registered development domains available for embedding floor")
    prepare_run(output, config, "input_embedding_floor", provenance={"datasets":{name:dataset_identity(rows) for name, rows in selected.items()},
                    "embedding_code_sha256":file_sha256(__file__)})
    results = []
    class EmbeddingAdapter:
        def __init__(self, runtime):
            self.runtime = runtime
        def identity(self):
            return {**self.runtime.identity(), "execution": "input_embedding_only_no_transformer"}
        @property
        def telemetry(self):
            return self.runtime.telemetry
        def features(self, records, layers, readout):
            if layers != [0]:
                raise ValueError("Embedding floor permits layer0 only")
            yield from self.runtime.embedding_only(records, readout=readout)
    for model in config["models"]:
        if not model.get("structural", True):
            continue
        runtime = runtime_factory(model, config.get("runtime", {})) if runtime_factory else runtime_from_config(model, config.get("runtime", {}))
        try:
            for domain, rows in selected.items():
                leaf = Path(output)/"caches"/model_key(model)/domain
                extract_features(EmbeddingAdapter(runtime), rows, [0], "final_token" if domain == "S3_role" else "span_mean", leaf)
                saved_rows, arrays, manifest = read_cache(leaf)
                result = {**analyze_embedding_floor(saved_rows, arrays[0], domain), "model": model["id"],
                          "cache_manifest_sha256": file_sha256(leaf/"manifest.json"), "cache_path": str(leaf.relative_to(output))}
                results.append(result)
                atomic_json(Path(output)/"cells"/(model_key(model)+"__"+domain+".json"), result)
        finally:
            if hasattr(runtime, "close"):
                runtime.close()
    report = {"study": "input_embedding_floor", "primary_panel_eligible": False, "transformer_forwards": 0,
              "datasets": {name: len(rows) for name, rows in selected.items()}, "cells": results,
              "inference_scope": "development_only; no heldout_family_or_entity_activations", "models": len({r['model'] for r in results})}
    atomic_json(Path(output)/"embedding_floor.json", report)
    return report


run_embedding = run_embedding_floor
