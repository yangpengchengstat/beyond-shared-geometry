#!/usr/bin/env python3
"""Render the evidence-chain table from hash-verified, already reviewed results.

Uses only the Python standard library. No models, predictor fitting, calibration,
resampling, new significance decisions, or modifications to input evidence.
The accompanying JSON records every numerical source and selector. This renderer
does not edit release manifests; the release assembler must register its outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = "results/reproduction_v1_review/derived_summary.json"
FULL = "results/derived/full/primary_structural_template_metrics.csv"
GENERATOR = "results/derived/afs_abc/primary_generator_composition_metrics.csv"
PERMUTATION = "results/frozen/afs_abc/permutation_summary.csv"
REVIEW = "release/reproduction_review.json"
OUTPUT = "paper/generated/identification_summary"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build(root=ROOT, check=False):
    root = Path(root).resolve()
    provenance = {
        "schema_version": 1,
        "renderer": "scripts/build_identification_table.py",
        "renderer_sha256": digest(Path(__file__)),
        "operation": "Select recorded aggregates and compute descriptive means/counts only",
        "model_execution": False,
        "fitting": False,
        "resampling": False,
        "new_significance_or_equivalence_decisions": False,
        "independent_confirmation": False,
        "scope": "Different historical and replay experiments; compare losses only within their stated endpoint and study",
        "inputs": {},
        "values": {},
    }

    def file(name):
        p = root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Non-relative publication path: " + name)
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root),
                "Missing or unsafe input: " + name)
        return p

    def read_json(name):
        return json.loads(file(name).read_text())

    registry = read_json("results/registry.json")
    artifacts = read_json("release/artifact_manifest.json")
    review = read_json(REVIEW)
    for key in ("automatic_promotion", "original_flags_changed", "independent_confirmation",
                "new_statistical_significance_decisions"):
        require(review[key] is False, "Reviewed interpretation boundary changed: " + key)

    def verify(name, entries, index_path, selector):
        require(len(entries) == 1, "Expected one verification entry: " + name)
        actual = digest(file(name))
        require(actual == entries[0]["sha256"], "Evidence hash mismatch: " + name)
        provenance["inputs"][name] = {
            "sha256": actual, "verified_against": index_path,
            "verification_selector": selector,
        }

    for name in (FULL, GENERATOR, PERMUTATION):
        verify(name, [e for e in registry["entries"] if e["path"] == name],
               "results/registry.json", {"entries.path": name})
    verify(SUMMARY, [e for e in artifacts["artifacts"] if e["path"] == SUMMARY],
           "release/artifact_manifest.json", {"artifacts.path": SUMMARY})
    raw = {}
    for component in ("compact", "context"):
        name = "results/reproduction_v1_review/" + component + "_rows.json"
        verify(name, [review["inputs"][name]], REVIEW, {"inputs": name})
        raw[component] = read_json(name)
    # Do not embed artifact_manifest's own digest: it will register these outputs.
    for name in ("results/registry.json", REVIEW):
        provenance["inputs"][name] = {"sha256": digest(file(name)), "role": "verification contract"}
    summary = read_json(SUMMARY)

    def record(key, value, name, selector, operation, **extra):
        require(key not in provenance["values"], "Duplicate provenance key: " + key)
        require(isinstance(value, (int, float)) and math.isfinite(value), "Nonfinite value: " + key)
        provenance["values"][key] = {
            "value": value, "source_path": name,
            "source_sha256": provenance["inputs"][name]["sha256"],
            "selector": selector, "operation": operation, **extra,
        }
        return value

    def csv_rows(name):
        with file(name).open(newline="") as handle:
            return list(csv.DictReader(handle))

    historical = {name: csv_rows(name) for name in (FULL, GENERATOR, PERMUTATION)}

    def historical_mean(key, name, group, method, metric, expected_rows):
        where = {"group": group, "method": method, "latent_dim": "16"}
        rows = [r for r in historical[name] if all(r[k] == v for k, v in where.items())]
        require(len(rows) == expected_rows, "Historical panel size changed: " + key)
        keys = [tuple(r[k] for k in ("model", "depth_slot", "split_seed", "template_id")) +
                ((r["target_element"],) if name == GENERATOR else ()) for r in rows]
        require(len(set(keys)) == len(rows), "Duplicate historical panel rows: " + key)
        require(len({r["model"] for r in rows}) == 6, "Historical fixed-model panel changed")
        require({float(r["depth_slot"]) for r in rows} == {.2, .4, .6, .8}, "Depth panel changed")
        return record(key, statistics.fmean(float(r[metric]) for r in rows), name,
                      {"where": where, "field": metric}, "unweighted arithmetic mean of retained rows",
                      selected_rows=len(rows))

    def aggregate_value(key, component, where, field="mean"):
        matches = [(i, r) for i, r in enumerate(summary[component])
                   if all(r[k] == v for k, v in where.items())]
        require(len(matches) == 1, "Nonunique reviewed aggregate: " + key)
        i, selected = matches[0]
        require(selected["cells"] == 24, "Reviewed panel size changed: " + key)
        if component in raw:
            rows = [r for r in raw[component] if all(r[k] == v for k, v in where.items())]
            require(len(rows) == len({r["cell"] for r in rows}) == 24, "Incomplete raw fixed panel: " + key)
            require(all(r.get("invalid", 0) == 0 and math.isfinite(r["nmse"]) for r in rows),
                    "Invalid prediction cannot be silently excluded: " + key)
            require(math.isclose(selected["mean"], statistics.fmean(r["nmse"] for r in rows),
                                 rel_tol=1e-13, abs_tol=1e-15), "Stored mean does not match raw cells: " + key)
        return record(key, selected[field], SUMMARY,
                      {"json_pointer": f"/{component}/{i}/{field}", "where": where},
                      ("read recorded aggregate; mean independently checked against reviewed cell rows"
                       if component in raw else
                       "read recorded paired count; independently checked against paired reviewed cell losses"),
                      cells=selected["cells"])

    observed = {}
    for group, domain in (("C7", "C7"), ("D4", "D4_corner")):
        o = observed[group] = {}
        o["full"] = historical_mean(group + ".full", FULL, group, "independent_ridge", "E_eq", 720)
        for method in ("generator_composed", "direct_ridge"):
            o[method] = historical_mean(group + "." + method, GENERATOR, group, method, "E_gen_eq", 3600)
        for metric in ("E_hom_data", "E_hom_op"):
            where = {"group": group, "data_split": "test", "metric": metric, "latent_dim": "16"}
            cells = [r for r in historical[PERMUTATION] if all(r[k] == v for k, v in where.items())]
            require(len(cells) == 120, "AFS-B panel size changed")
            ids = [tuple(r[k] for k in ("model", "depth_slot", "split_seed")) for r in cells]
            require(len(set(ids)) == 120, "Duplicated AFS-B cell")
            n = sum(float(r["true_relation_defect"]) < float(r["null_q025"]) for r in cells)
            require(n == 120, "Recorded restricted-reference result changed")
            record(group + "." + metric + ".below_reference", n, PERMUTATION,
                   {"where": where, "predicate": "true_relation_defect < null_q025"},
                   "count below recorded restricted-reference quantile", denominator=len(cells))
        where = {"domain": domain, "registration": "semantic", "track": "all_element", "split": "test"}
        for method in ("ridge", "decoded_centroid"):
            o[method] = aggregate_value(group + ".compact." + method, "compact", {**where, "method": method})
        paired_where = {k: v for k, v in where.items() if k != "split"}
        paired_where["method"] = "decoded_centroid"
        o["centroid_wins"] = aggregate_value(group + ".centroid_wins", "paired_compact",
                                               paired_where, "cells_lower_than_ridge")
        pool = [r for r in raw["compact"] if all(r[k] == v for k, v in where.items())]
        maps = {m: {r["cell"]: r["nmse"] for r in pool if r["method"] == m}
                for m in ("ridge", "decoded_centroid")}
        require(maps["ridge"].keys() == maps["decoded_centroid"].keys(), "Unpaired centroid cells")
        require(o["centroid_wins"] == sum(v < maps["ridge"][k] for k, v in maps["decoded_centroid"].items()),
                "Recorded centroid win count mismatch")
        for registration, method, short in (("semantic", "coordinate_fitted_operator_context", "coordinate"),
                                             ("semantic", "linear_transport_context", "direct_context"),
                                             ("consistent_swap01", "linear_transport_context", "coherent")):
            o[short] = aggregate_value(group + ".context." + short, "context",
                {"domain": domain, "registration": registration, "method": method, "split": "test"})
        for method, crossover in (("nystrom", 2), ("rff", 4)):
            first = None
            for tag, budget in (("0p5", .5), ("1", 1), ("2", 2), ("4", 4), ("8", 8)):
                value = aggregate_value(group + ".compact." + method + "." + tag, "compact",
                    {**where, "method": method + "_budget_" + tag})
                if value < o["ridge"] and first is None:
                    first = budget
            require(first == crossover, "First tested mean crossover changed: " + group + "." + method)
            record(group + ".compact." + method + ".first_tested_crossover", first, SUMMARY,
                   {"array": "compact", "where": where,
                    "candidate_methods": [method + "_budget_" + t for t in ("0p5", "1", "2", "4", "8")],
                    "reference_method": "ridge", "field": "mean"},
                   "first declared nominal budget whose recorded mean is below the recorded ridge mean",
                   nominal_budget_grid=[.5, 1, 2, 4, 8], interpretation="descriptive; not a selected deployment policy")
    require((observed["C7"]["centroid_wins"], observed["D4"]["centroid_wins"]) == (9, 2),
            "Centroid heterogeneity changed")

    c, d = observed["C7"], observed["D4"]
    rows = [
        (r"Independent fits\newline (FULL, $D$)",
         f"Ridge: {c['full']:.5f} / {d['full']:.5f}.",
         "Shared linear maps predict activations in held-out templates; stable category coding remains compatible."),
        (r"Generator transfer\newline (AFS-A, $D$)",
         f"Composed: {c['generator_composed']:.6f} / {d['generator_composed']:.6f}. " + r"\newline " +
         f"Direct: {c['direct_ridge']:.6f} / {d['direct_ridge']:.6f}.",
         "Transfer to five excluded action elements; state labels remain seen. Direct fits receive additional element supervision."),
        (r"Restricted registration\newline (AFS-B)",
         r"120/120 cells in each group, separately for data and operator defects, below the reference 2.5th percentile.",
         "Separation from invalid registrations; the restricted ensemble is not a calibrated semantic null."),
        (r"Category control\newline (replay prediction)",
         f"Centroid: {c['decoded_centroid']:.6f} / {d['decoded_centroid']:.6f}. " + r"\newline " +
         f"Ridge: {c['ridge']:.6f} / {d['ridge']:.6f}. " + r"\newline " +
         f"Centroid lower in {c['centroid_wins']}/24 / {d['centroid_wins']}/24 cells.",
         "Category coding is competitive for the C7 mean, not most cells; centroids receive extra training labels and the action table."),
        (r"Context differences\newline (replay transport)",
         f"Coordinate fit: {c['coordinate']:.6f} / {d['coordinate']:.6f}. " + r"\newline " +
         f"Direct fit: {c['direct_context']:.6f} / {d['direct_context']:.6f}. " + r"\newline " +
         f"Coherent fit: {c['coherent']:.6f} / {d['coherent']:.6f}.",
         "Direct fitting helps; a coherent alternative has close means. Targets and masks are registration-specific; closeness is not equivalence."),
        (r"Compact nonlinear\newline (replay prediction)",
         r"First tested budget with lower mean than ridge, in both groups:\newline Nystr\"om $2\times$; RFF $4\times$.",
         "Predictor ranking depends on storage. Parameter bytes do not match statistical capacity or runtime."),
    ]
    rows = [rows[i] for i in (0, 1, 4, 5, 3, 2)]
    caption = (
        r"Evidence and identification boundaries. Paired numbers are $C_7$ / $D_4$ (corners); smaller losses are better. "
        r"Rows concern different experiments, not one matched comparison or independent confirmations. "
        r"Historical $D$ averages per-row target-normalized squared errors; replay prediction and context transport use "
        r"distinct training-energy denominators $E_z$ and $E_u$. Compare losses only within the stated study and endpoint. "
        r"Replay means cover 24 fixed model/readout cells per group. Budget crossovers are descriptive across the full recorded grid, "
        r"not a test-selected deployment policy. D4 fixed-point identities, and hence scored rows, can differ across registrations. "
        r"Statistical scope is defined in Section~\ref{sec:setup}."
    )
    lines = [r"% Generated by scripts/build_identification_table.py; see identification_summary.source.json.",
             r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\renewcommand{\arraystretch}{1.10}",
             r"\begin{tabular}{@{}p{.16\textwidth}p{.40\textwidth}p{.40\textwidth}@{}}",
             r"\toprule", r"Test and endpoint & Recorded observation ($C_7$ / $D_4$) & Supported conclusion and remaining alternative \\",
             r"\midrule"]
    for index, row in enumerate(rows):
        lines.append(" & ".join(row) + r" \\")
        if index < len(rows) - 1:
            lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{" + caption + "}",
              r"\label{tab:identification-summary}", r"\end{table*}"]
    contents = {
        OUTPUT + ".tex": "\n".join(lines) + "\n",
        OUTPUT + ".source.json": json.dumps(provenance, indent=2, allow_nan=False) + "\n",
    }
    for name, content in contents.items():
        p = root / name
        if check:
            require(p.is_file() and p.read_text() == content, "Generated artifact stale: " + name)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
    return {"status": "verified" if check else "rendered", "rows": len(rows),
            "values_with_provenance": len(provenance["values"]), "outputs": list(contents),
            "model_execution": False, "fitting": False, "resampling": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify evidence and generated bytes without writing.")
    args = parser.parse_args()
    print(json.dumps(build(check=args.check), indent=2))
