"""Paired inference and stable outputs for the frozen AFS-D control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bsg.experiments.main_outputs import write_figure_pair
from bsg.experiments.runtime_state import atomic_write_frame, atomic_write_json, atomic_write_text
from bsg.statistics.hierarchical import hierarchical_mean_ci


HEADLINE_METRICS = (
    "E_gen_eq", "E_rel_data", "E_hom_data", "E_comp_data", "E_eq", "E_hom_op",
)
D4_RELATION_COMPONENTS = (
    "E_rel_r4_data", "E_rel_s2_data", "E_rel_srs_r_inverse_data",
)
TEMPLATE_METRICS = set(HEADLINE_METRICS + D4_RELATION_COMPONENTS) - {"E_hom_op"}

LOCATION_COLUMNS = (
    "model_key", "model", "model_family", "group", "representation_location",
    "depth_slot", "layer", "realized_depth", "latent_dim", "split_seed",
)


def _require_unique_values(frame: pd.DataFrame, keys: list[str], value: str) -> None:
    counts = frame.groupby(keys, dropna=False)[value].nunique(dropna=False)
    if (counts != 1).any():
        raise ValueError(f"{value} is not unique at its scientific unit")


def _operator_location_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    direct = frame[(frame["method"] == "independent_ridge") & (frame["data_split"] == "test")].copy()
    if direct.empty:
        raise ValueError("AFS-D operator metrics contain no held-out independent-ridge rows")
    rows = []
    for metric in (*HEADLINE_METRICS[1:], *D4_RELATION_COMPONENTS):
        if metric not in direct:
            raise ValueError(f"AFS-D operator metrics missing {metric}")
        selected = direct.dropna(subset=[metric]).copy()
        if metric in D4_RELATION_COMPONENTS:
            selected = selected[selected["group"] == "D4"]
        if metric == "E_hom_op":
            keys = list(LOCATION_COLUMNS)
            _require_unique_values(selected, keys, metric)
            selected = selected.drop_duplicates(keys)
            selected["template_id"] = np.nan
            selected["unit_type"] = "split"
        else:
            keys = [*LOCATION_COLUMNS, "template_id"]
            _require_unique_values(selected, keys, metric)
            selected = selected.drop_duplicates(keys)
            selected["unit_type"] = "template"
        selected["metric"] = metric
        selected["value"] = selected[metric].astype(float)
        rows.append(selected[[*LOCATION_COLUMNS, "template_id", "unit_type", "metric", "value", "source_run"]])
    return pd.concat(rows, ignore_index=True)


def _generator_location_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[
        (frame["method"] == "generator_composed")
        & (frame["evaluation_role"] == "unseen_composite")
        & (frame["data_split"] == "test")
    ].copy()
    if selected.empty:
        raise ValueError("AFS-D generator metrics contain no held-out unseen-composite rows")
    keys = [*LOCATION_COLUMNS, "template_id", "source_run"]
    target_counts = selected.groupby(keys, dropna=False)["target_element"].nunique()
    expected = selected["group"].map({"C7": 5, "D4": 5})
    observed = selected.groupby(keys, dropna=False)["target_element"].transform("nunique")
    if not np.array_equal(observed.to_numpy(), expected.to_numpy()):
        raise ValueError("AFS-D E_gen_eq does not contain five preregistered unseen elements per template")
    aggregated = selected.groupby(keys, dropna=False, as_index=False)["E_gen_eq"].mean()
    if (target_counts != 5).any():
        raise ValueError("AFS-D generator target aggregation contract changed")
    aggregated["unit_type"] = "template"
    aggregated["metric"] = "E_gen_eq"
    aggregated["value"] = aggregated["E_gen_eq"].astype(float)
    return aggregated[[*LOCATION_COLUMNS, "template_id", "unit_type", "metric", "value", "source_run"]]


def prepare_location_metrics(operator_frame: pd.DataFrame, generator_frame: pd.DataFrame) -> pd.DataFrame:
    """Create one observation per scientifically valid split/template unit."""
    frame = pd.concat(
        [_operator_location_metrics(operator_frame), _generator_location_metrics(generator_frame)],
        ignore_index=True,
    )
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise ValueError("AFS-D location metrics contain NaN or Inf")
    return frame.sort_values(
        ["model_key", "group", "metric", "representation_location", "depth_slot", "split_seed", "template_id"],
        na_position="first",
    ).reset_index(drop=True)


def pair_depth_contrasts(location_metrics: pd.DataFrame, depth_slots=(0.2, 0.4, 0.6, 0.8)) -> pd.DataFrame:
    embedding = location_metrics[location_metrics["representation_location"] == "embedding"].copy()
    internal = location_metrics[location_metrics["representation_location"] == "internal"].copy()
    if embedding.empty or internal.empty:
        raise ValueError("AFS-D pairing requires embedding and internal rows")
    mismatched = []
    for group, frame in location_metrics.groupby("group"):
        dims = frame.groupby("representation_location")["latent_dim"].unique().to_dict()
        if "embedding" in dims and "internal" in dims and set(map(int, dims["embedding"])) != set(map(int, dims["internal"])):
            mismatched.append((group, sorted(map(int, dims["embedding"])), sorted(map(int, dims["internal"]))))
    if mismatched:
        raise ValueError(f"AFS-D dimension-mismatched pairing rejected: {mismatched}")
    pair_keys = ["model_key", "model", "model_family", "group", "latent_dim", "split_seed", "template_id", "unit_type", "metric"]
    embedding = embedding.rename(columns={"value": "embedding_value"})
    internal = internal.rename(columns={"value": "internal_value"})
    merged = internal.merge(
        embedding[pair_keys + ["embedding_value"]],
        on=pair_keys,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("AFS-D produced no paired embedding/internal rows")
    expected_depths = {round(float(value), 2) for value in depth_slots}
    observed_depths = {round(float(value), 2) for value in merged["depth_slot"].unique()}
    if observed_depths != expected_depths:
        raise ValueError(f"AFS-D paired depth coverage mismatch: {observed_depths}")
    base_keys = pair_keys
    counts = merged.groupby(base_keys, dropna=False)["depth_slot"].nunique()
    if not (counts == len(expected_depths)).all():
        raise ValueError("AFS-D paired units do not contain every fixed internal depth")
    merged["difference"] = merged["internal_value"] - merged["embedding_value"]
    return merged.sort_values(
        ["model_key", "group", "metric", "depth_slot", "split_seed", "template_id"],
        na_position="first",
    ).reset_index(drop=True)


def bootstrap_depth_contrasts(
    contrasts: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    ci: float = 0.95,
    progress_callback=None,
) -> pd.DataFrame:
    """Bootstrap paired differences without resampling fixed depth points."""
    rows = []
    for (model_key, model, group, metric), cell in contrasts.groupby(
        ["model_key", "model", "group", "metric"], dropna=False, sort=True
    ):
        unit_types = set(cell["unit_type"])
        if len(unit_types) != 1:
            raise ValueError("AFS-D metric mixes split- and template-level units")
        clusters = ("split_seed", "template_id") if unit_types == {"template"} else ("split_seed",)
        depth_means = cell.groupby("depth_slot", dropna=False)["difference"].mean()
        fraction = float((depth_means < 0).mean())
        selections = [("depth_specific", float(depth), cell[cell["depth_slot"] == depth]) for depth in sorted(cell["depth_slot"].unique())]
        selections.append(("fixed_depth_average", np.nan, cell))
        for scope, depth, selected in selections:
            callback = None
            if progress_callback is not None:
                callback = lambda done, total, label=(model_key, group, metric, scope, depth): progress_callback(label, done, total)
            summary = hierarchical_mean_ci(
                selected,
                "difference",
                n_boot=int(n_boot),
                seed=int(seed),
                ci=float(ci),
                cluster_columns=clusters,
                collapse_optimization_duplicates=False,
                progress_callback=callback,
                progress_every=200,
            )
            rows.append({
                "model_key": model_key,
                "model": model,
                "group": group,
                "metric": metric,
                "scope": scope,
                "depth_slot": depth,
                "mean_difference": summary.pop("mean"),
                "embedding_estimate": float(selected["embedding_value"].mean()),
                "internal_estimate": float(selected["internal_value"].mean()),
                "n_pairs": int(len(selected)),
                "fraction_depths_internal_better": fraction,
                **summary,
            })
    return pd.DataFrame(rows).sort_values(
        ["model_key", "group", "metric", "scope", "depth_slot"], na_position="last"
    ).reset_index(drop=True)


def classify_outcomes(summary: pd.DataFrame) -> dict:
    """Apply the frozen A/B/C/D interpretation without a favorable-result gate."""
    fixed = summary[summary["scope"] == "fixed_depth_average"].copy()
    models = sorted(fixed["model"].unique())
    if len(models) != 2:
        raise ValueError("AFS-D interpretation requires exactly two fixed-panel models")
    primary = ("E_gen_eq", "E_rel_data")
    supporting = ("E_hom_data", "E_comp_data")
    group_outcomes = {}
    model_patterns = {}
    for group in ("C7", "D4"):
        group_frame = fixed[fixed["group"] == group]
        patterns = {}
        for model in models:
            model_frame = group_frame[group_frame["model"] == model].set_index("metric")
            missing = set(primary + supporting) - set(model_frame.index)
            if missing:
                raise ValueError(f"AFS-D interpretation missing metrics: {sorted(missing)}")
            internal = all(float(model_frame.loc[metric, "ci_high"]) < 0 for metric in primary) and all(
                float(model_frame.loc[metric, "mean_difference"]) < 0 for metric in supporting
            )
            embedding = all(float(model_frame.loc[metric, "ci_low"]) > 0 for metric in primary) and all(
                float(model_frame.loc[metric, "mean_difference"]) > 0 for metric in supporting
            )
            primary_include_zero = all(
                float(model_frame.loc[metric, "ci_low"]) <= 0 <= float(model_frame.loc[metric, "ci_high"])
                for metric in primary
            )
            patterns[model] = {
                "internal_strengthening": bool(internal),
                "embedding_dominant": bool(embedding),
                "primary_intervals_include_zero": bool(primary_include_zero),
            }
        supporting_concordant = False
        for metric in supporting:
            rows = group_frame[group_frame["metric"] == metric]
            supporting_concordant |= bool((rows["ci_high"] < 0).all() or (rows["ci_low"] > 0).all())
        if all(pattern["internal_strengthening"] for pattern in patterns.values()):
            outcome = "A"
        elif all(pattern["embedding_dominant"] for pattern in patterns.values()):
            outcome = "C"
        elif all(pattern["primary_intervals_include_zero"] for pattern in patterns.values()) and not supporting_concordant:
            outcome = "B"
        else:
            outcome = "MIXED"
        group_outcomes[group] = outcome
        model_patterns[group] = patterns
    overall = "D" if group_outcomes["C7"] in {"B", "C"} and group_outcomes["D4"] == "A" else "GROUP_SPECIFIC_RESULTS"
    wording = {
        "A": "Internal transformer representations strengthen or refine recoverable algebra beyond the lexical baseline.",
        "B": "No systematic difference was detected; algebra may be inherited from lexical or language-statistical structure and preserved through depth. This is not formal equivalence.",
        "C": "Recoverable algebra may be strongest lexically and weakened or transformed by contextual computation.",
        "D": "C7 is embedding-competitive while D4 strengthens internally; cyclic lexical structure may explain C7 more readily than noncommutative D4 structure.",
        "MIXED": "Model, metric, or depth evidence is heterogeneous; no A-C interpretation is forced.",
        "GROUP_SPECIFIC_RESULTS": "Interpret C7 and D4 separately under their preregistered outcomes.",
    }
    return {
        "gate": "NONE_DIAGNOSTIC_CONTROL",
        "group_outcomes": group_outcomes,
        "overall_interpretation": overall,
        "interpretation_text": wording[overall],
        "group_interpretation_text": {group: wording[outcome] for group, outcome in group_outcomes.items()},
        "model_patterns": model_patterns,
    }


def location_summary(location_metrics: pd.DataFrame) -> pd.DataFrame:
    return location_metrics.groupby(
        ["model_key", "model", "model_family", "group", "representation_location", "depth_slot", "layer", "metric"],
        dropna=False,
        as_index=False,
    )["value"].mean().rename(columns={"value": "mean_value"})


def _scientific_report(manifest: dict, interpretation: dict) -> str:
    outcomes = interpretation["group_outcomes"]
    return f"""# AFS-D Scientific Report

Run ID: `{manifest['run_id']}`
Status: `{manifest['status']}`
Source commit: `{manifest['git_commit']}`

AFS-D is a rank-aware embedding-to-depth diagnostic control. It does not alter or rerun `main_full_003` or `afs_full_abc_001`, and it has no favorable-result PASS/FAIL gate.

## Frozen matrix

- Models: meta-llama/Llama-3.2-1B; Qwen/Qwen2.5-1.5B
- Groups: C7; D4
- Locations: embedding, 0.20, 0.40, 0.60, 0.80
- Split seeds: 101, 303, 909, 2701, 4243
- Rank-matched latent dimensions: C7=6, D4=3
- Intrinsic-rank diagnostics: reported before algebraic interpretation
- Bootstrap: 5,000 replicates, seed 314159, 95% percentile intervals

## Preregistered interpretation

- C7: **{outcomes['C7']}** - {interpretation['group_interpretation_text']['C7']}
- D4: **{outcomes['D4']}** - {interpretation['group_interpretation_text']['D4']}
- Overall: **{interpretation['overall_interpretation']}** - {interpretation['interpretation_text']}

The numerical evidence is in `afs_d/bootstrap_depth_contrasts.csv` and the stable tables. Outcome B is not a formal equivalence claim. No outcome proves emergence, a causal mechanism, or universality beyond this fixed panel.
"""


def write_stable_outputs(
    output_dir: Path,
    *,
    embedding_operator: pd.DataFrame,
    embedding_generator: pd.DataFrame,
    embedding_rank_diagnostics: pd.DataFrame,
    generator_calibration: pd.DataFrame,
    location_metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    bootstrap: pd.DataFrame,
    interpretation: dict,
    manifest: dict,
) -> list[Path]:
    output_dir = Path(output_dir)
    afs = output_dir / "afs_d"
    tables = output_dir / "results" / "tables"
    figures = output_dir / "results" / "figures"
    outputs = [
        atomic_write_frame(embedding_operator, afs / "embedding_operator_metrics.csv"),
        atomic_write_frame(embedding_generator, afs / "embedding_generator_metrics.csv"),
        atomic_write_frame(embedding_rank_diagnostics, afs / "embedding_rank_diagnostics.csv"),
        atomic_write_frame(generator_calibration, afs / "generator_calibration_metrics.csv"),
        atomic_write_frame(location_metrics[location_metrics["representation_location"] == "internal"], afs / "internal_metrics_import.csv"),
        atomic_write_frame(location_metrics, afs / "location_metrics.csv"),
        atomic_write_frame(contrasts, afs / "depth_contrasts.csv"),
        atomic_write_frame(bootstrap, afs / "bootstrap_depth_contrasts.csv"),
        atomic_write_json(output_dir / "interpretation.json", interpretation),
    ]
    by_location = location_summary(location_metrics)
    outputs.extend([
        atomic_write_frame(by_location, tables / "afs_d_by_model_group_location.csv"),
        atomic_write_frame(bootstrap, tables / "afs_d_depth_contrasts.csv"),
        atomic_write_frame(
            by_location[(by_location["group"] == "D4") & (by_location["metric"].isin(D4_RELATION_COMPONENTS))],
            tables / "afs_d_d4_relation_components.csv",
        ),
    ])
    figure_frame = by_location.copy()
    figure_frame["location_order"] = figure_frame["depth_slot"].astype(float)
    relation = figure_frame[figure_frame["metric"] == "E_rel_data"].groupby("location_order", as_index=False)["mean_value"].mean()
    generator = figure_frame[figure_frame["metric"] == "E_gen_eq"].groupby("location_order", as_index=False)["mean_value"].mean()
    outputs.extend(write_figure_pair(relation, "location_order", "mean_value", figures, "afs_d_operator_algebra_by_depth", "AFS-D Relation Error by Depth"))
    outputs.extend(write_figure_pair(generator, "location_order", "mean_value", figures, "afs_d_generator_transfer_by_depth", "AFS-D Generator Transfer by Depth"))
    outputs.append(atomic_write_text(output_dir / "AFS_D_SCIENTIFIC_REPORT.md", _scientific_report(manifest, interpretation)))
    return outputs
