#!/usr/bin/env python3
"""Visualize existing reviewed budget means; no fitting or new aggregation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = ((0.5, "0p5"), (1.0, "1"), (2.0, "2"), (4.0, "4"), (8.0, "8"))
METHODS = (
    ("limited_support", "Limited support", "#666666", "s"),
    ("rff", "RFF", "#0072B2", "o"),
    ("nystrom", "Nyström", "#D55E00", "^"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path = ROOT) -> dict:
    root = root.resolve()
    source = root / "results/reproduction_v1_review/derived_summary.json"
    review_path = root / "release/reproduction_review.json"
    summary = json.loads(source.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    # Read the existing aggregate: do not re-average checkpoints or choose budgets.
    eligible = [
        row for row in summary["compact"]
        if row["registration"] == "semantic"
        and row["track"] == "all_element"
        and row["split"] == "test"
    ]

    def select(domain: str, method: str) -> dict:
        matches = [r for r in eligible if r["domain"] == domain and r["method"] == method]
        if len(matches) != 1 or matches[0]["cells"] != 24:
            raise ValueError(f"Expected one existing 24-cell aggregate: {domain}/{method}")
        return matches[0]

    rows = []
    for domain in ("C7", "D4_corner"):
        ridge = select(domain, "ridge")
        rows.append({"domain": domain, "method": "ridge", "nominal_budget": 1.0,
                     "mean": ridge["mean"], "cells": ridge["cells"],
                     "mean_parameter_bytes": ridge["mean_parameter_bytes"],
                     "display_role": "horizontal_reference"})
        for method, _, _, _ in METHODS:
            for budget, suffix in BUDGETS:
                item = select(domain, f"{method}_budget_{suffix}")
                rows.append({"domain": domain, "method": method, "nominal_budget": budget,
                             "mean": item["mean"], "cells": item["cells"],
                             "mean_parameter_bytes": item["mean_parameter_bytes"],
                             "display_role": "observed_curve_point"})

    output = root / "paper/figures"
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "budget_frontier"
    csv_path = stem.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.45), sharey=True)
    for ax, domain, title in zip(axes, ("C7", "D4_corner"), (r"$C_7$: weekdays", r"$D_4$: corners")):
        for method, label, color, marker in METHODS:
            data = [r for r in rows if r["domain"] == domain and r["method"] == method]
            ax.plot([r["nominal_budget"] for r in data], [r["mean"] for r in data],
                    color=color, marker=marker, markersize=4.2, linewidth=1.35,
                    markeredgewidth=0.55, label=label)
        ridge = select(domain, "ridge")["mean"]
        ax.axhline(ridge, color="#222222", linewidth=1.15, linestyle=(0, (4, 2.5)),
                   label=r"Ridge ($1\times$ reference)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(0.42, 9.5)
        ax.set_ylim(0.0047, 1.15)
        ax.xaxis.set_major_locator(FixedLocator([v for v, _ in BUDGETS]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", which="major", length=3, pad=3)
        ax.tick_params(axis="y", which="minor", length=1.6, width=0.4)
        ax.grid(axis="y", which="major", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, pad=5)
        ax.set_xlabel("Nominal storage budget / ridge", labelpad=4)
    axes[0].set_ylabel("Prediction loss (log scale)", labelpad=5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=2.1, columnspacing=1.3,
               handletextpad=0.5)
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.275, top=0.9, wspace=0.14)
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "plot_budget_frontier.py",
                "CreationDate": None, "ModDate": None, "Title": "Predictor rankings depend on storage"})
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)

    provenance = {
        "role": "visualization_of_existing_reviewed_means",
        "source_run": review["source_run"],
        "source_archive_sha256": review["source_archive_sha256"],
        "archive_source_path": "parent-from-zero-v1/compact_context/checkpoints/*.json",
        "inputs": {str(p.relative_to(root)): sha(p) for p in (source, review_path)},
        "script": "scripts/plot_budget_frontier.py", "script_sha256": sha(Path(__file__)),
        "selector": {"registration": "semantic", "track": "all_element", "split": "test",
                     "domains": ["C7", "D4_corner"], "budgets": [b for b, _ in BUDGETS],
                     "methods": [m for m, _, _, _ in METHODS], "reference": "ridge"},
        "endpoint": "all-element changed-state compact-coordinate prediction",
        "existing_aggregation": "Equal action means followed by equal means over 24 fixed model/readout cells per domain",
        "x_axis": "Nominal incremental parameter-storage budget relative to identity-pruned ridge; not exact realized bytes or total RAM",
        "transformation": "Select existing scalar means and plot all five declared budgets; copy scalar values to CSV. No recomputation of statistical summaries.",
        "new_statistics": False, "model_execution": False, "refitting": False,
        "resampling": False, "confidence_intervals": False, "error_bands": False,
        "scope": "Fixed-panel descriptive means, not a test-selected budget policy or unseen-element endpoint",
        "outputs": {str(p.relative_to(root)): sha(p) for p in
                    (csv_path, stem.with_suffix(".pdf"), stem.with_suffix(".png"))},
    }
    stem.with_suffix(".source.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"figure": str(stem.with_suffix(".pdf")), "points": len(rows), "new_statistics": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(build(parser.parse_args().root), indent=2))
