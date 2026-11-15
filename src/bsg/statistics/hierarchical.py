"""Hierarchical inference over genuine scientific replication axes."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_CLUSTER_COLUMNS = (
    "model",
    "depth_slot",
    "split_seed",
    "random_baseline_seed",
    "template_id",
)

PAIR_CONTEXT_COLUMNS = (
    "model",
    "model_key",
    "model_A",
    "model_B",
    "group",
    "depth_slot",
    "layer",
    "layer_A",
    "layer_B",
    "latent_dim",
    "split_seed",
    "template_id",
    "data_split",
    "comparison_type",
    "step",
    "relation",
    "intervention",
    "evaluation_role",
    "source_state",
    "target_state",
)


def collapse_exact_optimization_duplicates(
    frame: pd.DataFrame,
    value_cols: tuple[str, ...] | list[str],
    *,
    preserve_random_seed: bool = False,
) -> pd.DataFrame:
    """Remove exact copies that differ only by technical optimizer seed labels."""
    values = [column for column in value_cols if column in frame.columns]
    if not values:
        raise ValueError("no requested value columns are present")
    context = [column for column in PAIR_CONTEXT_COLUMNS if column in frame.columns]
    if preserve_random_seed and "random_baseline_seed" in frame.columns:
        context.append("random_baseline_seed")
    collapsed = frame.drop_duplicates(subset=context + values, keep="first").reset_index(drop=True)
    drop_columns = ["optimization_seed"]
    if not preserve_random_seed:
        drop_columns.append("random_baseline_seed")
    return collapsed.drop(columns=[column for column in drop_columns if column in collapsed], errors="ignore")


def _resolved_cluster_columns(data: pd.DataFrame, cluster_columns=None) -> tuple[str, ...]:
    requested = DEFAULT_CLUSTER_COLUMNS if cluster_columns is None else tuple(cluster_columns)
    return tuple(
        column for column in requested
        if column in data.columns and not data[column].dropna().empty
    )


def _sorted_values(series: pd.Series) -> list:
    values = list(series.dropna().unique())
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: (type(value).__name__, str(value)))


def _clustered_draw(data: pd.DataFrame, value_col: str, rng, cluster_columns=None) -> float:
    """Slow reference draw retained for exact semantic equivalence tests."""
    columns = _resolved_cluster_columns(data, cluster_columns)

    def draw(frame: pd.DataFrame, level: int):
        if level == len(columns):
            return frame[value_col].dropna().to_numpy(dtype=float)
        column = columns[level]
        keys = _sorted_values(frame[column])
        pieces = []
        for index in rng.integers(0, len(keys), size=len(keys)):
            piece = draw(frame[frame[column] == keys[int(index)]], level + 1)
            if len(piece):
                pieces.append(piece)
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=float)

    values = draw(data, 0)
    return float(values.mean()) if len(values) else float("nan")


def _prepare_clustered_index(data: pd.DataFrame, value_cols: tuple[str, ...], cluster_columns=None) -> dict:
    columns = _resolved_cluster_columns(data, cluster_columns)

    def build(frame: pd.DataFrame, level: int):
        if level == len(columns):
            return {"values": tuple(frame[column].dropna().to_numpy(dtype=float) for column in value_cols)}
        column = columns[level]
        keys = _sorted_values(frame[column])
        return {
            "keys": keys,
            "children": {key: build(frame[frame[column] == key], level + 1) for key in keys},
        }

    return {"cluster_columns": columns, "tree": build(data, 0)}


def _draw_indexed_values(index: dict, rng) -> tuple[np.ndarray, ...]:
    def draw(node):
        if "values" in node:
            return node["values"]
        keys = node["keys"]
        pieces = [
            draw(node["children"][keys[int(key_index)]])
            for key_index in rng.integers(0, len(keys), size=len(keys))
        ]
        if not pieces:
            return (np.empty(0, dtype=float),)
        return tuple(np.concatenate([piece[i] for piece in pieces]) for i in range(len(pieces[0])))

    return draw(index["tree"])


def _clustered_draw_indexed(index: dict, rng) -> float:
    values = _draw_indexed_values(index, rng)[0]
    return float(values.mean()) if len(values) else float("nan")


def _notify_progress(callback, completed: int, total: int, every: int) -> None:
    if callback is not None and (completed == total or completed % max(1, int(every)) == 0):
        callback(completed, total)


def _cluster_counts(data: pd.DataFrame, columns: tuple[str, ...]) -> dict:
    return {
        "n_observations": int(len(data)),
        "n_models": int(data["model"].nunique()) if "model" in data else 0,
        "n_depths": int(data["depth_slot"].nunique()) if "depth_slot" in data else 0,
        "n_splits": int(data["split_seed"].nunique()) if "split_seed" in data else 0,
        "n_templates": int(data["template_id"].nunique()) if "template_id" in data else 0,
        "n_random_seeds": int(data["random_baseline_seed"].nunique()) if "random_baseline_seed" in data else 0,
        "cluster_columns": ",".join(columns),
    }


def hierarchical_mean_ci(
    frame: pd.DataFrame,
    value_col: str,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
    cluster_columns=None,
    collapse_optimization_duplicates: bool = True,
    preserve_random_seed: bool = False,
    progress_callback=None,
    progress_every: int = 200,
) -> dict:
    if value_col not in frame.columns:
        raise ValueError(f"hierarchical bootstrap missing value column: {value_col}")
    data = frame.dropna(subset=[value_col]).copy()
    if collapse_optimization_duplicates and not data.empty:
        data = collapse_exact_optimization_duplicates(
            data, (value_col,), preserve_random_seed=preserve_random_seed
        )
    if data.empty:
        return {
            "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "n_boot": int(n_boot), "n_observations": 0,
        }
    rng = np.random.default_rng(int(seed))
    columns = _resolved_cluster_columns(data, cluster_columns)
    index = _prepare_clustered_index(data, (value_col,), columns)
    draws = np.empty(int(n_boot), dtype=float)
    for replicate in range(int(n_boot)):
        draws[replicate] = _clustered_draw_indexed(index, rng)
        _notify_progress(progress_callback, replicate + 1, int(n_boot), progress_every)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        raise ValueError("clustered bootstrap produced no finite draws")
    alpha = 1.0 - float(ci)
    return {
        "mean": float(data[value_col].mean()),
        "ci_low": float(np.quantile(draws, alpha / 2)),
        "ci_high": float(np.quantile(draws, 1 - alpha / 2)),
        "n_boot": int(n_boot),
        **_cluster_counts(data, columns),
    }


def _spearman(x, y) -> float:
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    if len(x_rank) < 2 or np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def hierarchical_spearman_ci(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
    cluster_columns=None,
    progress_callback=None,
    progress_every: int = 200,
) -> dict:
    missing = {x_col, y_col} - set(frame.columns)
    if missing:
        raise ValueError(f"hierarchical Spearman bootstrap missing columns: {sorted(missing)}")
    data = frame.dropna(subset=[x_col, y_col]).copy()
    if not data.empty:
        data = collapse_exact_optimization_duplicates(data, (x_col, y_col))
    estimate = _spearman(data[x_col], data[y_col])
    if not np.isfinite(estimate):
        return {
            "mean": estimate, "ci_low": float("nan"), "ci_high": float("nan"),
            "n_boot": int(n_boot), "n_observations": int(len(data)),
        }
    rng = np.random.default_rng(int(seed))
    columns = _resolved_cluster_columns(data, cluster_columns)
    index = _prepare_clustered_index(data, (x_col, y_col), columns)
    draws = np.empty(int(n_boot), dtype=float)
    for replicate in range(int(n_boot)):
        x_values, y_values = _draw_indexed_values(index, rng)
        draws[replicate] = _spearman(x_values, y_values)
        _notify_progress(progress_callback, replicate + 1, int(n_boot), progress_every)
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        raise ValueError("clustered Spearman bootstrap produced no finite draws")
    alpha = 1.0 - float(ci)
    return {
        "mean": estimate,
        "ci_low": float(np.quantile(finite, alpha / 2)),
        "ci_high": float(np.quantile(finite, 1 - alpha / 2)),
        "n_boot": int(n_boot),
        **_cluster_counts(data, columns),
    }


def paired_difference_frame(
    frame: pd.DataFrame,
    value_col: str,
    learned: str,
    baseline: str,
    *,
    method_col: str = "method",
) -> pd.DataFrame:
    selected = frame[frame[method_col].isin([learned, baseline])].dropna(subset=[value_col]).copy()
    if selected.empty:
        raise ValueError(f"paired methods absent: {learned}, {baseline}")
    learned_frame = collapse_exact_optimization_duplicates(selected[selected[method_col] == learned], (value_col,))
    random_baseline = baseline == "random_orthogonal"
    baseline_frame = collapse_exact_optimization_duplicates(
        selected[selected[method_col] == baseline],
        (value_col,),
        preserve_random_seed=random_baseline,
    )
    excluded = {method_col}
    if method_col == "intervention":
        excluded.add("evaluation_role")
    keys = [column for column in PAIR_CONTEXT_COLUMNS if column in selected.columns and column not in excluded]
    learned_values = learned_frame.groupby(keys, dropna=False, as_index=False)[value_col].mean()
    baseline_keys = keys + (["random_baseline_seed"] if random_baseline and "random_baseline_seed" in selected else [])
    baseline_values = baseline_frame.groupby(baseline_keys, dropna=False, as_index=False)[value_col].mean()
    merged = learned_values.merge(
        baseline_values,
        on=keys,
        how="inner",
        suffixes=("_learned", "_baseline"),
        validate="one_to_many" if random_baseline else "one_to_one",
    )
    if merged.empty:
        raise ValueError(f"no paired rows for methods: {learned}, {baseline}")
    merged["learned_minus_baseline"] = merged[f"{value_col}_learned"] - merged[f"{value_col}_baseline"]
    return merged


def paired_baseline_ci(
    frame: pd.DataFrame,
    value_col: str,
    learned: str,
    baseline: str,
    *,
    method_col: str = "method",
    n_boot: int = 2000,
    seed: int = 42,
    cluster_columns=None,
    progress_callback=None,
    progress_every: int = 200,
) -> dict:
    paired = paired_difference_frame(frame, value_col, learned, baseline, method_col=method_col)
    summary = hierarchical_mean_ci(
        paired,
        "learned_minus_baseline",
        n_boot=n_boot,
        seed=seed,
        cluster_columns=cluster_columns,
        collapse_optimization_duplicates=False,
        preserve_random_seed=baseline == "random_orthogonal",
        progress_callback=progress_callback,
        progress_every=progress_every,
    )
    summary.update({
        "learned_estimate": float(paired[f"{value_col}_learned"].mean()),
        "baseline_estimate": float(paired[f"{value_col}_baseline"].mean()),
        "n_pairs": int(len(paired)),
    })
    return summary
