"""Inference, frozen gates, and stable artifacts for AFS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bsg.experiments.main_outputs import write_figure_pair
from bsg.experiments.runtime_state import atomic_write_frame, atomic_write_json
from bsg.statistics.hierarchical import hierarchical_mean_ci


FIGURES = (
    "generator_unseen_composites", "d4_generator_words", "algebra_registration_null",
    "correspondence_scramble", "embedding_vs_internal",
)


def _paired(frame, learned: str, baseline: str, value: str, *, method_col="method") -> pd.DataFrame:
    keys = [
        column for column in (
            "model", "model_key", "group", "depth_slot", "layer", "split_seed", "data_split",
            "template_id", "target_element", "word_length", "step", "evaluation_role",
            "scramble_variant", "scramble_seed",
        ) if column in frame and column not in {method_col}
    ]
    left = frame[frame[method_col] == learned].copy()
    right = frame[frame[method_col] == baseline].copy()
    if baseline == "scrambled_ridge" and "scramble_seed" in right:
        keys = [key for key in keys if key not in {"scramble_seed", "scramble_variant"}]
    if "word" in left and baseline == "matched_wrong_word":
        keys = [key for key in keys if key != "word"]
    left = left.groupby(keys, dropna=False, as_index=False)[value].mean()
    right_keys = keys + (["scramble_seed", "scramble_variant"] if baseline == "scrambled_ridge" else [])
    right = right.groupby(right_keys, dropna=False, as_index=False)[value].mean()
    paired = left.merge(right, on=keys, suffixes=("_learned", "_baseline"), validate="one_to_many" if baseline == "scrambled_ridge" else "one_to_one")
    if paired.empty:
        raise ValueError(f"no AFS pairs for {learned} versus {baseline}")
    paired["difference"] = paired[f"{value}_learned"] - paired[f"{value}_baseline"]
    return paired


def generator_inference(frame: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    primary = frame[(frame.data_split == "test") & (frame.evaluation_role == "unseen_composite")]
    comparisons = {
        "generator_vs_identity": "identity",
        "generator_vs_random": "matched_random_generator",
        "generator_vs_direct": "direct_ridge",
    }
    rows = []
    for group in ("C7", "D4"):
        group_frame = primary[primary.group == group]
        group_comparisons = dict(comparisons)
        if group == "D4":
            group_comparisons["correct_vs_wrong_word"] = "matched_wrong_word"
        for name, baseline in group_comparisons.items():
            paired = _paired(group_frame, "generator_composed", baseline, "E_gen_eq")
            for model in ["ALL_MODELS", *sorted(paired.model.unique())]:
                selected = paired if model == "ALL_MODELS" else paired[paired.model == model]
                summary = hierarchical_mean_ci(
                    selected, "difference", n_boot=n_boot, seed=seed,
                    cluster_columns=("model", "depth_slot", "split_seed", "template_id"),
                    collapse_optimization_duplicates=False,
                )
                rows.append({"group": group, "contrast": name, "model": model, **summary})
    return pd.DataFrame(rows)


def scramble_inference(frame: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    primary = frame[frame.data_split == "test"]
    rows = []
    for group in ("C7", "D4"):
        for variant in ("train_only_state_relabel", "template_independent_target_scramble"):
            selected = pd.concat([
                primary[(primary.group == group) & (primary.method == "real_ridge")],
                primary[(primary.group == group) & (primary.method == "scrambled_ridge") & (primary.scramble_variant == variant)],
            ], ignore_index=True)
            for metric in ("E_eq", "E_rel_data"):
                paired = _paired(selected, "real_ridge", "scrambled_ridge", metric)
                for model in ["ALL_MODELS", *sorted(paired.model.unique())]:
                    subset = paired if model == "ALL_MODELS" else paired[paired.model == model]
                    summary = hierarchical_mean_ci(
                        subset, "difference", n_boot=n_boot, seed=seed,
                        cluster_columns=("model", "depth_slot", "split_seed", "scramble_seed", "template_id"),
                        collapse_optimization_duplicates=False,
                    )
                    rows.append({"group": group, "variant": variant, "metric": metric, "model": model, **summary})
    return pd.DataFrame(rows)


def evaluate_gates(generator_raw: pd.DataFrame, generator_ci: pd.DataFrame, permutation: pd.DataFrame, scramble_ci: pd.DataFrame, config: dict) -> dict:
    if config["tier"] != "full":
        return {
            "A1_C7": "NOT_EVALUABLE", "A2_D4": "NOT_EVALUABLE", "B": "NOT_EVALUABLE",
            "C": "NOT_EVALUABLE", "D": "DIAGNOSTIC_ONLY", "overall": "NOT_EVALUABLE",
            "reason": "SMOKE/PRE_FULL does not contain the frozen six-model, four-depth, five-split matrix",
        }
    gen_test = generator_raw[(generator_raw.data_split == "test") & (generator_raw.evaluation_role == "unseen_composite")]
    def model_depth_pass(group, baselines):
        flags = []
        for model in sorted(gen_test.model.unique()):
            depth_passes = 0
            for depth in sorted(gen_test.depth_slot.unique()):
                cell = gen_test[(gen_test.group == group) & (gen_test.model == model) & (gen_test.depth_slot == depth)]
                checks = []
                for baseline in baselines:
                    checks.append(float(_paired(cell, "generator_composed", baseline, "E_gen_eq").difference.mean()) < 0)
                depth_passes += int(all(checks))
            flags.append(depth_passes >= 3)
        return sum(flags) >= 5
    c7_ci = generator_ci[(generator_ci.group == "C7") & (generator_ci.contrast == "generator_vs_random") & (generator_ci.model == "ALL_MODELS")].iloc[0]
    d4_random_ci = generator_ci[(generator_ci.group == "D4") & (generator_ci.contrast == "generator_vs_random") & (generator_ci.model == "ALL_MODELS")].iloc[0]
    d4_wrong_ci = generator_ci[(generator_ci.group == "D4") & (generator_ci.contrast == "correct_vs_wrong_word") & (generator_ci.model == "ALL_MODELS")].iloc[0]
    a1 = model_depth_pass("C7", ["matched_random_generator"]) and float(c7_ci.ci_high) < 0
    a2 = model_depth_pass("D4", ["matched_random_generator", "matched_wrong_word"]) and float(d4_random_ci.ci_high) < 0 and float(d4_wrong_ci.ci_high) < 0
    ptest = permutation[(permutation.data_split == "test")]
    b_checks = []
    for group in ("C7", "D4"):
        for metric in ("E_hom_op", "E_hom_data"):
            cells = ptest[(ptest.group == group) & (ptest.metric == metric)]
            extreme = float((cells.true_relation_defect < cells.null_q025).mean()) >= 0.80
            no_reversal = all(
                float((family.true_relation_defect >= family.null_median).mean()) < 0.50
                for _, family in cells.groupby("model_family")
            )
            b_checks.append(extreme and no_reversal)
    b_pass = all(b_checks)
    c_checks = []
    for group in ("C7", "D4"):
        for variant in ("train_only_state_relabel", "template_independent_target_scramble"):
            cells = scramble_ci[(scramble_ci.group == group) & (scramble_ci.variant == variant) & (scramble_ci.metric == "E_eq")]
            pooled = cells[cells.model == "ALL_MODELS"].iloc[0]
            model_rows = cells[cells.model != "ALL_MODELS"]
            c_checks.append(int((model_rows["mean"] < 0).sum()) >= 5 and float(pooled.ci_high) < 0)
    c_pass = all(c_checks)
    a_pass = a1 and a2
    overall = "STRONG_PASS" if a_pass and b_pass and c_pass else "PARTIAL_PASS" if a_pass and (b_pass or c_pass) else "FAIL"
    return {
        "A1_C7": "PASS" if a1 else "FAIL", "A2_D4": "PASS" if a2 else "FAIL",
        "B": "PASS" if b_pass else "FAIL", "C": "PASS" if c_pass else "FAIL",
        "D": "DIAGNOSTIC_ONLY", "overall": overall,
    }


def write_stable_outputs(output_dir: Path, generator, permutation_raw, permutation_summary, scramble, embedding, generator_bootstrap, scramble_bootstrap) -> list[Path]:
    output_dir = Path(output_dir)
    afs = output_dir / "afs"
    tables = output_dir / "results" / "tables"
    figures = output_dir / "results" / "figures"
    for directory in (afs, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)
    outputs = [
        atomic_write_frame(generator, afs / "generator_composition_metrics.csv"),
        atomic_write_frame(permutation_raw, afs / "algebra_permutation_null.csv"),
        atomic_write_frame(scramble, afs / "correspondence_scramble_metrics.csv"),
        atomic_write_frame(generator_bootstrap, afs / "bootstrap_generator_contrasts.csv"),
        atomic_write_frame(scramble_bootstrap, afs / "bootstrap_scramble_contrasts.csv"),
        atomic_write_frame(permutation_summary, afs / "permutation_summary.csv"),
    ]
    generator_table = generator.groupby(
        ["model", "group", "depth_slot", "method", "evaluation_role", "target_element", "step"],
        dropna=False, as_index=False,
    )[["E_gen_eq", "generator_generalization_gap"]].mean()
    scramble_table = scramble.groupby(
        ["model", "group", "depth_slot", "method", "scramble_variant"], dropna=False, as_index=False,
    )[["E_eq", "E_rel_data", "E_rel_op", "E_comp_data", "E_comp_op", "E_inverse_data", "E_inverse_op"]].mean()
    table_frames = {
        "afs_generator_c7.csv": generator_table[generator_table.group == "C7"],
        "afs_generator_d4.csv": generator_table[generator_table.group == "D4"],
        "afs_permutation_null.csv": permutation_summary,
        "afs_scramble_null.csv": scramble_table,
    }
    if embedding is not None:
        outputs.append(atomic_write_frame(embedding, afs / "embedding_baseline_metrics.csv"))
        embedding_for_table = embedding.copy()
        if "evaluation_role" not in embedding_for_table:
            embedding_for_table["evaluation_role"] = "operator_evaluation"
        else:
            embedding_for_table["evaluation_role"] = embedding_for_table["evaluation_role"].fillna("operator_evaluation")
        embedding_table = embedding_for_table.groupby(
            ["model", "group", "method", "evaluation_role"], dropna=False, as_index=False,
        )[[column for column in ("E_eq", "E_rel_data", "E_rel_op", "E_gen_eq") if column in embedding]].mean()
        table_frames["afs_embedding_baseline.csv"] = embedding_table
    for name, frame in table_frames.items():
        outputs.append(atomic_write_frame(frame, tables / name))
    figure_specs = {
        "generator_unseen_composites": (generator[(generator.group == "C7") & (generator.evaluation_role == "unseen_composite")], "step", "E_gen_eq"),
        "d4_generator_words": (generator[(generator.group == "D4") & (generator.evaluation_role == "unseen_composite")].assign(word_index=lambda x: np.arange(len(x))), "word_index", "E_gen_eq"),
        "algebra_registration_null": (permutation_raw.assign(null_index=lambda x: np.arange(len(x))), "null_index", "null_E_hom_data"),
        "correspondence_scramble": (scramble.assign(row_index=lambda x: np.arange(len(x))), "row_index", "E_eq"),
    }
    if embedding is not None:
        figure_specs["embedding_vs_internal"] = (pd.concat([
            embedding.assign(representation_location="embedding"),
            scramble[(scramble.method == "real_ridge")].assign(representation_location="internal"),
        ], ignore_index=True).assign(row_index=lambda x: np.arange(len(x))), "row_index", "E_eq")
    for stem, (frame, x, y) in figure_specs.items():
        outputs.extend(write_figure_pair(frame, x, y, figures, stem, stem.replace("_", " ").title()))
    return outputs
