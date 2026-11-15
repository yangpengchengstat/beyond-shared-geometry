#!/usr/bin/env python3
"""Execute the frozen additive-field falsification protocol.

This script is intentionally isolated from FULL/AFS execution code. It reads
immutable main_full_003 structural checkpoints from the raw FULL ZIP, verifies
ledger hashes before use, and writes a new additive-specific result bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bsg.data.c7 import build_dataset as build_c7, transform_state as c7_transform
from bsg.data.d4 import build_dataset as build_d4, transform_state as d4_transform
from bsg.data.splits import split_templates_three_way
from bsg.experiments.canonical_units import load_artifact
from bsg.experiments.fitting import GroupSpec
from bsg.experiments.main_config import CANONICAL_REVISIONS
from bsg.groups import c7, d4
from bsg.metrics.core import EPS
from bsg.metrics.relations import c7_relations, d4_relations


RUN_ID = "additive_field_falsification_001"
RAW_FULL_ZIP_SHA256 = "5fe7234572120073d8dfab2e8dc0d3373a741f8d48abadc92288f98ab16721f4"
SOURCE_COMMIT = "30f3c6369a185f43e94b17ca03b0499cd5d65a78"
CONFIG_HASH = "910d430f18c5f865d88948c3284e0c9eaea531323aaf9c697ad2399cbeb372dd"
ZIP_ROOT = "run_outputs/main_experiment/main_full_003"
SPLIT_SEEDS = (101, 303, 909, 2701, 4243)
DEPTH_SLOTS = (0.20, 0.40, 0.60, 0.80)
BOOTSTRAP_SEED = 314159
BOOTSTRAP_REPS = 5000
C7_POWERS = (2, 3, 4, 5, 6)


MODELS = (
    {
        "key": "llama_1b",
        "id": "meta-llama/Llama-3.2-1B",
        "family": "Llama",
        "blocks": 16,
        "depth_layers": {0.20: 3, 0.40: 6, 0.60: 10, 0.80: 13},
    },
    {
        "key": "llama_3b",
        "id": "meta-llama/Llama-3.2-3B",
        "family": "Llama",
        "blocks": 28,
        "depth_layers": {0.20: 6, 0.40: 11, 0.60: 17, 0.80: 22},
    },
    {
        "key": "llama_8b",
        "id": "meta-llama/Llama-3.1-8B",
        "family": "Llama",
        "blocks": 32,
        "depth_layers": {0.20: 6, 0.40: 13, 0.60: 19, 0.80: 26},
    },
    {
        "key": "qwen_1_5b",
        "id": "Qwen/Qwen2.5-1.5B",
        "family": "Qwen",
        "blocks": 28,
        "depth_layers": {0.20: 6, 0.40: 11, 0.60: 17, 0.80: 22},
    },
    {
        "key": "qwen_3b",
        "id": "Qwen/Qwen2.5-3B",
        "family": "Qwen",
        "blocks": 36,
        "depth_layers": {0.20: 7, 0.40: 14, 0.60: 22, 0.80: 29},
    },
    {
        "key": "qwen_7b",
        "id": "Qwen/Qwen2.5-7B",
        "family": "Qwen",
        "blocks": 28,
        "depth_layers": {0.20: 6, 0.40: 11, 0.60: 17, 0.80: 22},
    },
)


@dataclass(frozen=True)
class ZipEntry:
    path: str
    bytes: int
    sha256: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def group_specs() -> dict[str, GroupSpec]:
    return {
        "C7": GroupSpec("C7", build_c7(), c7.ELEMENTS, c7.IDENTITY, c7.compose, c7.inverse, c7_transform, c7_relations()),
        "D4": GroupSpec("D4", build_d4(), d4.ELEMENTS, d4.IDENTITY, d4.compose, d4.inverse, d4_transform, d4_relations()),
    }


def template_split(spec: GroupSpec, seed: int):
    return split_templates_three_way(
        spec.frame,
        seed=int(seed),
        train_templates=28,
        validation_templates=6,
        test_templates=6,
    )


class FullZipSource:
    def __init__(self, zip_path: Path):
        self.zip_path = Path(zip_path).resolve()
        if sha256_path(self.zip_path) != RAW_FULL_ZIP_SHA256:
            raise ValueError("raw FULL ZIP SHA256 mismatch")
        self.zf = zipfile.ZipFile(self.zip_path)
        self.names = set(self.zf.namelist())
        self.manifest = json.loads(self.read_text(f"{ZIP_ROOT}/manifest.json"))
        self.ledger = json.loads(self.read_text(f"{ZIP_ROOT}/run_ledger.json"))
        self._validate_manifest()
        self.activation_cache_count = sum(
            1 for name in self.names
            if name.startswith("run_outputs/activation_cache/") and name.endswith(".npz")
        )
        if self.activation_cache_count != 300:
            raise ValueError(f"expected 300 activation cache NPZ files, found {self.activation_cache_count}")

    def read_text(self, name: str) -> str:
        return self.zf.read(name).decode("utf-8")

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        expected = {
            "run_id": "main_full_003",
            "experiment": "beyond_shared_geometry_main",
            "tier": "full",
            "status": "SUCCESS",
            "git_commit": SOURCE_COMMIT,
            "config_hash": CONFIG_HASH,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"FULL manifest {key} mismatch: {manifest.get(key)!r}")
        if manifest.get("git_dirty") is not False:
            raise ValueError("FULL manifest git_dirty was not false")
        revisions = {model.get("id"): model.get("requested_revision") for model in manifest.get("models", [])}
        if revisions != CANONICAL_REVISIONS:
            raise ValueError("FULL manifest model revision panel mismatch")
        if int(manifest.get("primary_latent_dimension", -1)) != 16:
            raise ValueError("FULL primary latent dimension mismatch")
        if manifest.get("template_split") != {"train_templates": 28, "validation_templates": 6, "test_templates": 6}:
            raise ValueError("FULL template split mismatch")

    def _unit_matches(self, prefix: str, spec, model, depth_slot: float, layer: int, split_seed: int) -> list[dict]:
        matches = []
        for key, unit in self.ledger.get("units", {}).items():
            fields = unit.get("fields", {})
            if not key.startswith(prefix + ":") or unit.get("status") != "COMPLETE":
                continue
            try:
                observed_depth = round(float(fields.get("depth_slot", -1)), 2)
            except (TypeError, ValueError):
                continue
            if (
                fields.get("model_key") == model["key"]
                and fields.get("model") == model["id"]
                and fields.get("group") == spec.name
                and observed_depth == round(float(depth_slot), 2)
                and int(fields.get("layer", -1)) == int(layer)
                and int(fields.get("split_seed", -1)) == int(split_seed)
                and int(fields.get("latent_dim", -1)) == 16
            ):
                matches.append(unit)
        return matches

    def load_artifact(self, prefix: str, spec, model, depth_slot: float, layer: int, split_seed: int) -> tuple[dict, ZipEntry]:
        matches = self._unit_matches(prefix, spec, model, depth_slot, layer, split_seed)
        if len(matches) != 1:
            raise ValueError(f"expected one {prefix} cell, found {len(matches)} for {model['key']} {spec.name} {depth_slot} {split_seed}")
        npz_output = next((o for o in matches[0]["outputs"] if str(o["path"]).endswith(".npz")), None)
        if npz_output is None:
            raise ValueError(f"{prefix} cell has no NPZ output")
        inner = f"{ZIP_ROOT}/checkpoints/{prefix}/{Path(npz_output['path']).name}"
        if inner not in self.names:
            raise ValueError(f"missing checkpoint in raw FULL ZIP: {inner}")
        payload = self.zf.read(inner)
        observed = ZipEntry(inner, len(payload), sha256_bytes(payload))
        if observed.bytes != int(npz_output["bytes"]):
            raise ValueError(f"{prefix} checkpoint byte mismatch: {inner}")
        if observed.sha256 != npz_output["sha256"]:
            raise ValueError(f"{prefix} checkpoint SHA256 mismatch: {inner}")

        tmp = Path("/tmp") / f"bsg_{Path(inner).name}"
        tmp.write_bytes(payload)
        try:
            artifact = load_artifact(tmp, spec.elements, seed=int(split_seed))
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        if artifact["z"].shape != (len(spec.frame), 16):
            raise ValueError("latent shape contract mismatch")
        if int(artifact["pca"]._fit_rows) != 28 * len(spec.frame.state_id.unique()):
            raise ValueError("train-only PCA fit-row contract mismatch")
        return artifact, observed


def row_error(pred, target) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    return np.sum((pred - target) ** 2, axis=1) / (np.sum(target ** 2, axis=1) + EPS)


def pair_arrays(spec, z, template_id: int, element):
    frame = spec.frame
    lookup = {(int(row.template_id), int(row.state_id)): idx for idx, row in frame.iterrows()}
    source_states = sorted(map(int, frame.state_id.unique()))
    src_idx, tgt_idx = [], []
    for state in source_states:
        src = lookup[(int(template_id), int(state))]
        target_state = spec.transform_state(int(state), element)
        tgt = lookup[(int(template_id), int(target_state))]
        src_idx.append(src)
        tgt_idx.append(tgt)
    return source_states, z[src_idx], z[tgt_idx]


def fit_additive_vectors(spec, z, train_ids):
    vectors = {}
    for element in spec.elements:
        if element == spec.identity:
            vectors[element] = np.zeros(z.shape[1], dtype=float)
            continue
        diffs = []
        for template_id in train_ids:
            _, x, y = pair_arrays(spec, z, template_id, element)
            diffs.append(y - x)
        vectors[element] = np.concatenate(diffs, axis=0).mean(axis=0)
    return vectors


def c7_power(k: int) -> int:
    return c7.power(c7.GENERATOR, int(k))


def op_power(op: np.ndarray, k: int) -> np.ndarray:
    return np.linalg.matrix_power(np.asarray(op, dtype=float), int(k))


def evaluate_cell(spec, split, metadata, base_artifact, joint_artifact) -> list[dict]:
    rows = []
    z = base_artifact["z"]
    add = fit_additive_vectors(spec, z, split.train_ids)
    operators = {
        "additive": None,
        "ridge": base_artifact["operators"],
        "joint": joint_artifact["operators"],
    }
    for template_id in split.test_ids:
        for element in spec.elements:
            if element == spec.identity:
                continue
            states, x, y = pair_arrays(spec, z, int(template_id), element)
            preds = {
                "additive": x + add[element],
                "ridge": x @ operators["ridge"][element],
                "joint": x @ operators["joint"][element],
            }
            for method, pred in preds.items():
                for source_state, value in zip(states, row_error(pred, y)):
                    rows.append({
                        **metadata,
                        "template_id": int(template_id),
                        "source_state": int(source_state),
                        "metric_family": "direct",
                        "component": str(element),
                        "method": method,
                        "value": float(value),
                        "primary": True,
                        "diagnostic": False,
                    })
        if spec.name == "C7":
            for k in C7_POWERS:
                element = c7_power(k)
                states, x, y = pair_arrays(spec, z, int(template_id), element)
                preds = {
                    "additive": x + k * add[c7.GENERATOR],
                    "ridge": x @ op_power(operators["ridge"][c7.GENERATOR], k),
                    "joint": x @ op_power(operators["joint"][c7.GENERATOR], k),
                }
                for method, pred in preds.items():
                    for source_state, value in zip(states, row_error(pred, y)):
                        rows.append({
                            **metadata,
                            "template_id": int(template_id),
                            "source_state": int(source_state),
                            "metric_family": "c7_composition",
                            "component": f"k{k}",
                            "method": method,
                            "value": float(value),
                            "primary": True,
                            "diagnostic": False,
                        })
        else:
            d4_specs = [
                ("d4_sr", d4.compose(d4.S, d4.R), (d4.R, d4.S), add[d4.R] + add[d4.S]),
                ("d4_rs", d4.compose(d4.R, d4.S), (d4.S, d4.R), add[d4.S] + add[d4.R]),
                ("d4_srs", d4.inverse(d4.R), (d4.S, d4.R, d4.S), add[d4.S] + add[d4.R] + add[d4.S]),
            ]
            for family, element, word, add_delta in d4_specs:
                states, x, y = pair_arrays(spec, z, int(template_id), element)
                preds = {"additive": x + add_delta}
                for method in ("ridge", "joint"):
                    op = np.eye(z.shape[1], dtype=float)
                    for generator in word:
                        op = op @ operators[method][generator]
                    preds[method] = x @ op
                for method, pred in preds.items():
                    for source_state, value in zip(states, row_error(pred, y)):
                        rows.append({
                            **metadata,
                            "template_id": int(template_id),
                            "source_state": int(source_state),
                            "metric_family": family,
                            "component": family.removeprefix("d4_"),
                            "method": method,
                            "value": float(value),
                            "primary": True,
                            "diagnostic": False,
                        })

            # Descriptive order diagnostics.
            states, x, y_sr = pair_arrays(spec, z, int(template_id), d4.compose(d4.S, d4.R))
            _, _, y_rs = pair_arrays(spec, z, int(template_id), d4.compose(d4.R, d4.S))
            target_sep = np.sum((y_sr - y_rs) ** 2, axis=1) / (
                0.5 * np.sum(y_sr ** 2, axis=1) + 0.5 * np.sum(y_rs ** 2, axis=1) + EPS
            )
            for source_state, value in zip(states, target_sep):
                rows.append({
                    **metadata,
                    "template_id": int(template_id),
                    "source_state": int(source_state),
                    "metric_family": "d4_target_order_separation",
                    "component": "sr_vs_rs",
                    "method": "target",
                    "value": float(value),
                    "primary": False,
                    "diagnostic": True,
                })
            for method in ("ridge", "joint"):
                op_sr = operators[method][d4.R] @ operators[method][d4.S]
                op_rs = operators[method][d4.S] @ operators[method][d4.R]
                pred_sr = x @ op_sr
                pred_rs = x @ op_rs
                sep = np.sum((pred_sr - pred_rs) ** 2, axis=1) / (
                    0.5 * np.sum(pred_sr ** 2, axis=1) + 0.5 * np.sum(pred_rs ** 2, axis=1) + EPS
                )
                for source_state, value in zip(states, sep):
                    rows.append({
                        **metadata,
                        "template_id": int(template_id),
                        "source_state": int(source_state),
                        "metric_family": "d4_operator_order_separation",
                        "component": "sr_vs_rs",
                        "method": method,
                        "value": float(value),
                        "primary": False,
                        "diagnostic": True,
                    })
                rows.append({
                    **metadata,
                    "template_id": int(template_id),
                    "source_state": -1,
                    "metric_family": "d4_additive_order_separation",
                    "component": "sr_vs_rs",
                    "method": "additive",
                    "value": 0.0,
                    "primary": False,
                    "diagnostic": True,
                })
    return rows


def assert_row_action_preflight() -> dict:
    sr = d4.compose(d4.S, d4.R)
    rs = d4.compose(d4.R, d4.S)
    srs = d4.row_word_product((d4.S, d4.R, d4.S))
    expected = d4.inverse(d4.R)
    if d4.row_word_product((d4.R, d4.S)) != sr:
        raise ValueError("row-action preflight failed for r -> s")
    if d4.row_word_product((d4.S, d4.R)) != rs:
        raise ValueError("row-action preflight failed for s -> r")
    if srs != expected:
        raise ValueError("row-action preflight failed for s -> r -> s")
    if sr == rs:
        raise ValueError("D4 order preflight failed: sr == rs")
    return {
        "r_then_s": {"group_element": str(sr), "operator_product": "R(r)R(s)", "additive": "v_r+v_s"},
        "s_then_r": {"group_element": str(rs), "operator_product": "R(s)R(r)", "additive": "v_s+v_r"},
        "s_then_r_then_s": {"group_element": str(srs), "operator_product": "R(s)R(r)R(s)", "additive": "v_s+v_r+v_s"},
        "srs_equals_r_inverse": bool(srs == expected),
    }


def aggregate_action_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model_key", "model", "model_family", "group", "depth_slot", "layer",
        "split_seed", "latent_dim", "metric_family", "component", "method",
        "primary", "diagnostic",
    ]
    return raw.groupby(keys, as_index=False, dropna=False)["value"].mean()


def primary_seed_aggregates(action: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = action[action["primary"] == True].copy()
    for method in ("additive", "ridge", "joint"):
        # Direct: equal group weight, equal model-depth weight, within-group equal action weight.
        direct = primary[(primary.method == method) & (primary.metric_family == "direct")]
        for seed, sframe in direct.groupby("split_seed"):
            group_vals = []
            for group, gframe in sframe.groupby("group"):
                md = gframe.groupby(["model_key", "depth_slot"])["value"].mean()
                group_vals.append(float(md.mean()))
            rows.append({"split_seed": int(seed), "metric": "direct", "method": method, "value": float(np.mean(group_vals))})
        c7comp = primary[(primary.method == method) & (primary.metric_family == "c7_composition")]
        for seed, sframe in c7comp.groupby("split_seed"):
            md = sframe.groupby(["model_key", "depth_slot"])["value"].mean()
            rows.append({"split_seed": int(seed), "metric": "c7_composition", "method": method, "value": float(md.mean())})
        for metric in ("d4_sr", "d4_rs", "d4_srs"):
            part = primary[(primary.method == method) & (primary.metric_family == metric)]
            for seed, sframe in part.groupby("split_seed"):
                md = sframe.groupby(["model_key", "depth_slot"])["value"].mean()
                rows.append({"split_seed": int(seed), "metric": metric, "method": method, "value": float(md.mean())})
        for seed in SPLIT_SEEDS:
            vals = {}
            for metric in ("d4_sr", "d4_rs", "d4_srs"):
                match = [r["value"] for r in rows if r["split_seed"] == seed and r["method"] == method and r["metric"] == metric]
                if len(match) == 1:
                    vals[metric] = match[0]
            if set(vals) == {"d4_sr", "d4_rs", "d4_srs"}:
                rows.append({"split_seed": int(seed), "metric": "d4_combined", "method": method, "value": float(np.mean(list(vals.values())))})
    return pd.DataFrame(rows)


def stratified_descriptives(action: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = action[action["primary"] == True].copy()
    for keys in (["metric_family", "method", "model_key"], ["metric_family", "method", "depth_slot"], ["metric_family", "method", "group"]):
        frame = primary.groupby(keys, as_index=False)["value"].mean()
        frame["stratum"] = ",".join(keys[2:])
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def bootstrap_contrasts(seed_agg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrasts = []
    for metric in ("direct", "c7_composition", "d4_sr", "d4_rs", "d4_srs", "d4_combined"):
        for comparator in ("ridge", "joint"):
            for seed in SPLIT_SEEDS:
                add = seed_agg[(seed_agg.metric == metric) & (seed_agg.method == "additive") & (seed_agg.split_seed == seed)]
                comp = seed_agg[(seed_agg.metric == metric) & (seed_agg.method == comparator) & (seed_agg.split_seed == seed)]
                if len(add) != 1 or len(comp) != 1:
                    raise ValueError(f"missing seed aggregate for {metric} {comparator} {seed}")
                contrasts.append({
                    "split_seed": int(seed),
                    "metric": metric,
                    "contrast": f"additive_minus_{comparator}",
                    "value": float(add.iloc[0].value - comp.iloc[0].value),
                })
    seed_contrasts = pd.DataFrame(contrasts)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (metric, contrast), frame in seed_contrasts.groupby(["metric", "contrast"]):
        values = np.array([frame[frame.split_seed == seed].iloc[0].value for seed in SPLIT_SEEDS], dtype=float)
        reps = np.empty(BOOTSTRAP_REPS, dtype=float)
        for i in range(BOOTSTRAP_REPS):
            indices = rng.integers(0, len(values), size=len(values))
            reps[i] = values[indices].mean()
        rows.append({
            "metric": metric,
            "contrast": contrast,
            "mean": float(values.mean()),
            "ci_low": float(np.quantile(reps, 0.025)),
            "ci_mid": float(np.quantile(reps, 0.5)),
            "ci_high": float(np.quantile(reps, 0.975)),
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        })
    return seed_contrasts, pd.DataFrame(rows)


def classify_outcome(ci: pd.DataFrame) -> tuple[str, dict]:
    def low(metric, comp):
        row = ci[(ci.metric == metric) & (ci.contrast == f"additive_minus_{comp}")].iloc[0]
        return float(row.ci_low)

    def high(metric, comp):
        row = ci[(ci.metric == metric) & (ci.contrast == f"additive_minus_{comp}")].iloc[0]
        return float(row.ci_high)

    details = {}
    for comp in ("ridge", "joint"):
        direct_degraded = low("direct", comp) > 0
        d4_components_degraded = all(low(metric, comp) > 0 for metric in ("d4_sr", "d4_rs", "d4_srs", "d4_combined"))
        d4_components_not_degraded = all(low(metric, comp) <= 0 for metric in ("d4_sr", "d4_rs", "d4_srs", "d4_combined"))
        details[comp] = {
            "direct_degraded": direct_degraded,
            "direct_not_clearly_degraded": not direct_degraded,
            "d4_all_degraded": d4_components_degraded,
            "d4_all_not_clearly_degraded": d4_components_not_degraded,
            "d4_ci_lows": {metric: low(metric, comp) for metric in ("d4_sr", "d4_rs", "d4_srs", "d4_combined")},
        }
    if any(details[comp]["direct_not_clearly_degraded"] and details[comp]["d4_all_degraded"] for comp in ("ridge", "joint")):
        return "A", details
    if all(details[comp]["direct_degraded"] for comp in ("ridge", "joint")):
        return "B", details
    if all(details[comp]["d4_all_not_clearly_degraded"] for comp in ("ridge", "joint")):
        return "C", details
    return "D", details


def write_audit(path: Path, run_dir: Path, manifest: dict, integrity: dict, ci: pd.DataFrame, seed_agg: pd.DataFrame, outcome: str) -> None:
    def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
        rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for _, row in frame.iterrows():
            values = []
            for column in columns:
                value = row[column]
                if column in {"mean", "ci_low", "ci_mid", "ci_high"}:
                    value = f"{float(value):.6f}"
                values.append(str(value))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)

    ci_table = markdown_table(
        ci,
        ["metric", "contrast", "mean", "ci_low", "ci_mid", "ci_high", "bootstrap_reps", "bootstrap_seed"],
    )
    text = f"""# Additive Field Final Scientific Audit

Audit date: {datetime.now(timezone.utc).isoformat()}
Run ID: `{RUN_ID}`
Protocol: `docs/ADDITIVE_FIELD_FALSIFICATION_PROTOCOL.md`
Execution mode: frozen one-shot additive-field falsification; no LLM forward passes

## A. Artifact Integrity

- Raw FULL ZIP: `{integrity['raw_full_zip']}`
- Raw FULL ZIP SHA256: `{integrity['raw_full_zip_sha256']}`
- Activation cache NPZ files located in raw FULL ZIP: {integrity['activation_cache_count']}
- Structural base checkpoints verified: {integrity['structural_base_verified']}
- Structural joint checkpoints verified: {integrity['structural_joint_verified']}
- Primary cells completed: {integrity['primary_cells_completed']} / {integrity['primary_cells_required']}
- Cache integrity: PASS
- Row-action preflight: PASS

## B. Protocol Compliance

The run used the frozen six-model panel, C7/D4, four primary depth slots, split
seeds 101/303/909/2701/4243, 28/6/6 template splits, train-only d=16 PCA
checkpoints, C7 powers {{2,3,4,5,6}}, D4 words `r -> s`, `s -> r`, and
`s -> r -> s`, and the paired seed-cluster bootstrap with 5,000 repetitions and
seed 314159.

No manuscript file and no frozen FULL/AFS artifact was modified.

## C. Preregistered Primary Results

Primary bootstrap contrasts are additive minus comparator. Positive intervals
mean larger additive error.

{ci_table}

## D. Descriptive Diagnostics

The D4 target-order and operator-order separation diagnostics are descriptive
only. They are included in the result bundle and are not used as additional
success criteria.

## E. Preregistered Interpretation

Preregistered outcome: **{outcome}**.

This classification uses only the Outcome A/B/C/D rules in the frozen protocol.
It does not add a new outcome category and does not reinterpret the result
beyond the protocol.

## F. Result Artifacts

- Result directory: `{run_dir}`
- Result ZIP: `{manifest['result_zip']}`
- Result ZIP SHA256: `{manifest['result_zip_sha256']}`
- SHA256 manifest: `{run_dir / 'sha256_manifest.csv'}`

## G. Integrity Issues

No artifact-integrity defect was detected during execution.
"""
    path.write_text(text, encoding="utf-8")


def file_manifest(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            rows.append({
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            })
    return pd.DataFrame(rows)


def zip_result(run_dir: Path, zip_path: Path) -> str:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(run_dir.parent))
    return sha256_path(zip_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-full-zip", type=Path, default=Path("runpod_output_by_date/result_9_6_2026/run_outputs_9_4_2026_full3.zip"))
    parser.add_argument("--output-root", type=Path, default=Path("results/additive_field_falsification"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/ADDITIVE_FIELD_FALSIFICATION_PROTOCOL.md"))
    args = parser.parse_args()

    run_dir = args.output_root / RUN_ID
    if run_dir.exists():
        raise FileExistsError(f"result directory already exists: {run_dir}")
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "tables").mkdir()
    (run_dir / "provenance").mkdir()
    (run_dir / "config").mkdir()
    (run_dir / "logs").mkdir()
    shutil.copy2(args.protocol, run_dir / "config" / "ADDITIVE_FIELD_FALSIFICATION_PROTOCOL.md")

    row_preflight = assert_row_action_preflight()
    source = FullZipSource(args.raw_full_zip)
    specs = group_specs()
    ledger_rows = []
    provenance_rows = []
    all_rows = []
    primary_required = len(MODELS) * 2 * len(DEPTH_SLOTS) * len(SPLIT_SEEDS)

    for spec in specs.values():
        if len(spec.frame.template_id.unique()) != 40:
            raise ValueError(f"{spec.name} template count mismatch")
        for seed in SPLIT_SEEDS:
            split = template_split(spec, seed)
            if (len(split.train_ids), len(split.validation_ids), len(split.test_ids)) != (28, 6, 6):
                raise ValueError(f"{spec.name} split count mismatch for seed {seed}")

    for model in MODELS:
        if CANONICAL_REVISIONS[model["id"]] != source.manifest["models"][[m["id"] for m in source.manifest["models"]].index(model["id"])]["requested_revision"]:
            raise ValueError(f"model revision mismatch: {model['id']}")
        for group_name, spec in specs.items():
            for depth_slot in DEPTH_SLOTS:
                layer = model["depth_layers"][depth_slot]
                for split_seed in SPLIT_SEEDS:
                    split = template_split(spec, split_seed)
                    meta = {
                        "run_id": RUN_ID,
                        "model_key": model["key"],
                        "model": model["id"],
                        "model_family": model["family"],
                        "group": group_name,
                        "depth_slot": float(depth_slot),
                        "layer": int(layer),
                        "realized_depth": int(layer) / int(model["blocks"]),
                        "split_seed": int(split_seed),
                        "latent_dim": 16,
                    }
                    base, base_entry = source.load_artifact("structural_base", spec, model, depth_slot, layer, split_seed)
                    joint, joint_entry = source.load_artifact("structural_joint", spec, model, depth_slot, layer, split_seed)
                    provenance_rows.append({**meta, "artifact_type": "structural_base", **base_entry.__dict__})
                    provenance_rows.append({**meta, "artifact_type": "structural_joint", **joint_entry.__dict__})
                    cell_rows = evaluate_cell(spec, split, meta, base, joint)
                    all_rows.extend(cell_rows)
                    ledger_rows.append({**meta, "status": "COMPLETE", "raw_metric_rows": len(cell_rows)})

    raw = pd.DataFrame(all_rows)
    action = aggregate_action_metrics(raw)
    seed_agg = primary_seed_aggregates(action)
    seed_contrasts, ci = bootstrap_contrasts(seed_agg)
    outcome, outcome_details = classify_outcome(ci)
    strat = stratified_descriptives(action)
    diagnostics = action[action["diagnostic"] == True].copy()

    raw.to_csv(run_dir / "raw" / "raw_additive_metrics.csv", index=False)
    action.to_csv(run_dir / "tables" / "matched_operator_comparator_metrics.csv", index=False)
    seed_agg.to_csv(run_dir / "tables" / "per_seed_aggregates.csv", index=False)
    seed_contrasts.to_csv(run_dir / "tables" / "per_seed_contrasts.csv", index=False)
    ci.to_csv(run_dir / "tables" / "bootstrap_contrasts_confidence_intervals.csv", index=False)
    strat.to_csv(run_dir / "tables" / "model_depth_descriptive_stratification.csv", index=False)
    diagnostics.to_csv(run_dir / "tables" / "d4_order_separation_diagnostics.csv", index=False)
    pd.DataFrame(ledger_rows).to_csv(run_dir / "execution_ledger.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(run_dir / "provenance" / "cache_provenance_hash_verification.csv", index=False)

    result_summary = {
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_cells_completed": len(ledger_rows),
        "primary_cells_required": primary_required,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "row_action_preflight": row_preflight,
        "preregistered_outcome": outcome,
        "outcome_details": outcome_details,
    }
    atomic_json(run_dir / "final_preregistered_outcome.json", result_summary)

    impl_config = {
        "run_id": RUN_ID,
        "models": MODELS,
        "groups": ["C7", "D4"],
        "depth_slots": DEPTH_SLOTS,
        "split_seeds": SPLIT_SEEDS,
        "template_split": {"train_templates": 28, "validation_templates": 6, "test_templates": 6},
        "latent_dim": 16,
        "c7_powers": C7_POWERS,
        "d4_words": row_preflight,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "raw_full_zip": str(args.raw_full_zip),
        "raw_full_zip_sha256": RAW_FULL_ZIP_SHA256,
    }
    atomic_json(run_dir / "config" / "implementation_config_snapshot.json", impl_config)

    manifest_df = file_manifest(run_dir)
    manifest_df.to_csv(run_dir / "sha256_manifest.csv", index=False)
    manifest_df = file_manifest(run_dir)
    manifest_df.to_csv(run_dir / "sha256_manifest.csv", index=False)

    zip_path = run_dir.parent / f"{RUN_ID}_results.zip"
    zip_sha = zip_result(run_dir, zip_path)
    final_manifest = {
        "run_id": RUN_ID,
        "status": "SUCCESS",
        "result_dir": str(run_dir.resolve()),
        "result_zip": str(zip_path.resolve()),
        "result_zip_sha256": zip_sha,
        "primary_cells_completed": len(ledger_rows),
        "primary_cells_required": primary_required,
        "preregistered_outcome": outcome,
    }
    atomic_json(run_dir / "manifest.json", final_manifest)
    # Rebuild manifest and ZIP once manifest.json exists.
    manifest_df = file_manifest(run_dir)
    manifest_df.to_csv(run_dir / "sha256_manifest.csv", index=False)
    zip_sha = zip_result(run_dir, zip_path)
    final_manifest["result_zip_sha256"] = zip_sha
    atomic_json(run_dir / "manifest.json", final_manifest)

    integrity = {
        "raw_full_zip": str(args.raw_full_zip.resolve()),
        "raw_full_zip_sha256": RAW_FULL_ZIP_SHA256,
        "activation_cache_count": source.activation_cache_count,
        "structural_base_verified": int((pd.DataFrame(provenance_rows).artifact_type == "structural_base").sum()),
        "structural_joint_verified": int((pd.DataFrame(provenance_rows).artifact_type == "structural_joint").sum()),
        "primary_cells_completed": len(ledger_rows),
        "primary_cells_required": primary_required,
    }
    atomic_json(run_dir / "provenance" / "integrity_preflight.json", integrity)
    write_audit(Path("docs/ADDITIVE_FIELD_FINAL_SCIENTIFIC_AUDIT.md"), run_dir.resolve(), final_manifest, integrity, ci, seed_agg, outcome)

    # Final rebuild after audit and integrity JSON.
    manifest_df = file_manifest(run_dir)
    manifest_df.to_csv(run_dir / "sha256_manifest.csv", index=False)
    zip_sha = zip_result(run_dir, zip_path)
    final_manifest["result_zip_sha256"] = zip_sha
    atomic_json(run_dir / "manifest.json", final_manifest)

    print(json.dumps({
        "status": "SUCCESS",
        "run_id": RUN_ID,
        "result_dir": str(run_dir.resolve()),
        "result_zip": str(zip_path.resolve()),
        "result_zip_sha256": zip_sha,
        "audit": str(Path("docs/ADDITIVE_FIELD_FINAL_SCIENTIFIC_AUDIT.md").resolve()),
        "outcome": outcome,
        "primary_cells": f"{len(ledger_rows)}/{primary_required}",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
