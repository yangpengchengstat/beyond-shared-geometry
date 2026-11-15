#!/usr/bin/env python3
"""Reproduce Tables 3–6 from saved endpoints using Python's standard library."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
DOMAINS = ("C7", "D4")
INPUT_NAMES = (
    "spectral_cells.csv", "witness_units.csv", "image_endpoints.csv",
    "canonical_images.json", "cell_risk_reference.csv", "registration_margins.csv",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tolerance(a, b=0.0):
    """Frozen symmetric, pairwise scalar tolerance; never rounds input values."""
    return 1e-8 + 1e-7 * max(abs(a), abs(b))


def number(value):
    result = float(value)
    require(math.isfinite(result), "Nonfinite numerical input")
    return result


def flag(value):
    require(value in ("True", "False"), "Invalid saved Boolean")
    return value == "True"


def canonical_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_inputs(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_names = {"data/" + name for name in INPUT_NAMES} | {"expected/table_values.json"}
    require(manifest.get("schema") == 1, "Unsupported manifest schema")
    require(set(manifest["files"]) == expected_names, "Unexpected manifest file set")
    for name, digest in manifest["files"].items():
        path = root / name
        require(path.is_file() and not path.is_symlink(), "Missing or symbolic-link input: " + name)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                "Input checksum mismatch: " + name)
    data = {}
    for name in INPUT_NAMES:
        path = root / "data" / name
        if name.endswith(".json"):
            data[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            with path.open(encoding="utf-8", newline="") as handle:
                data[name] = list(csv.DictReader(handle))
    expected = json.loads((root / "expected/table_values.json").read_text(encoding="utf-8"))
    return data, expected, manifest


def cell_index(rows, name):
    require(len(rows) == 48, name + ": expected 48 cells")
    require(len({r["cell"] for r in rows}) == 48, name + ": duplicate cell")
    require(Counter(r["domain"] for r in rows) == {"C7": 24, "D4": 24},
            name + ": incomplete domain coverage")
    return {r["cell"]: r for r in rows}


def validate_panel(data):
    indexes = {name: cell_index(data[name], name) for name in (
        "spectral_cells.csv", "witness_units.csv", "cell_risk_reference.csv", "registration_margins.csv")}
    spectral = indexes["spectral_cells.csv"]
    for name, index in indexes.items():
        require(set(index) == set(spectral), name + ": mismatched cell identities")
        require(all(index[c]["domain"] == spectral[c]["domain"] for c in spectral),
                name + ": mismatched cell domains")
    for c, row in spectral.items():
        risk = indexes["cell_risk_reference.csv"][c]
        margin = indexes["registration_margins.csv"][c]
        require(row["used_dimension"] == "16", "Unexpected retained dimension")
        require(all(row[k] == risk[k] == margin[k] for k in ("model_id", "layer")),
                "Mismatched model/readout identity")
        require(risk["join_key"] == margin["join_key"], "Mismatched saved bank join key")
        require(indexes["witness_units.csv"][c]["status"] == "complete", "Incomplete witness endpoint")

    canonical = data["canonical_images.json"]
    require(len(canonical) == 123, "Expected 123 canonical images")
    require(Counter(r["domain"] for r in canonical) == {"C7": 120, "D4": 3},
            "Incomplete canonical image coverage")
    image_map = {(r["domain"], r["image_sha256"]): r for r in canonical}
    require(len(image_map) == 123, "Duplicate canonical image")
    for row in canonical:
        require(type(row["semantic"]) is bool, "Invalid canonical semantic flag")
        require(canonical_digest(row["image_permutations"]) == row["image_sha256"],
                "Canonical permutation-image hash mismatch")
    for domain in DOMAINS:
        require(sum(r["semantic"] for r in canonical if r["domain"] == domain) == 1,
                "Expected one semantic image per domain")

    images = data["image_endpoints.csv"]
    require(len(images) == 2952, "Expected 2952 image endpoints")
    by_cell = defaultdict(list)
    for row in images:
        require(row["cell"] in spectral, "Unknown image endpoint cell")
        source = spectral[row["cell"]]
        require(source["domain"] == row["domain"], "Image endpoint domain mismatch")
        key = row["domain"], row["image_sha256"]
        require(key in image_map, "Unknown canonical image")
        require(flag(row["semantic_registration"]) == image_map[key]["semantic"],
                "Semantic image flag mismatch")
        require(row["join_key"] == indexes["cell_risk_reference.csv"][row["cell"]]["join_key"],
                "Image endpoint bank join key mismatch")
        number(row["gamma"])
        by_cell[row["cell"]].append(row)
    require(set(by_cell) == set(spectral), "Missing image endpoint cell")
    for cell, rows in by_cell.items():
        domain = spectral[cell]["domain"]
        wanted = {key[1] for key in image_map if key[0] == domain}
        require(len(rows) == len(wanted) and {r["image_sha256"] for r in rows} == wanted,
                "Missing or duplicate image within cell")
    return indexes, by_cell


def image_statistics(rows):
    semantic = [r for r in rows if flag(r["semantic_registration"])]
    require(len(semantic) == 1, "Expected exactly one semantic image in a cell")
    semantic = semantic[0]
    gamma = number(semantic["gamma"])
    others = [r for r in rows if r["image_sha256"] != semantic["image_sha256"]]
    require(len(others) == len(rows) - 1 and others, "Duplicate or absent alternative image")
    values = [number(r["gamma"]) for r in others]
    greater = sum(v > gamma + tolerance(v, gamma) for v in values)
    tied = sum(abs(v - gamma) <= tolerance(v, gamma) for v in values)
    lower_rank, upper_rank = greater + 1, greater + tied + 1
    return {
        "semantic_image_sha256": semantic["image_sha256"], "semantic_gamma": gamma,
        "alternatives": len(values), "d_top": gamma - max(values),
        "d_median": gamma - statistics.median(values),
        "spread": max(values + [gamma]) - min(values + [gamma]),
        "rank_low": lower_rank, "rank_high": upper_rank,
        "standing": "unique_leading" if upper_rank == 1 else "tied_leading" if lower_rank == 1 else "behind",
        "alternative_ties": tied,
        "best_alternative_images": ";".join(sorted(r["image_sha256"] for r in others if number(r["gamma"]) == max(values))),
        "pairwise_tolerance_min": min(tolerance(v, gamma) for v in values),
        "pairwise_tolerance_max": max(tolerance(v, gamma) for v in values),
    }


def compare_saved(row, values, context):
    for key, value in values.items():
        stored = row[key]
        if isinstance(value, bool):
            require(flag(stored) == value, context + ": " + key)
        elif isinstance(value, (int, float)):
            require(number(stored) == value, context + ": " + key)
        else:
            require(stored == value, context + ": " + key)


def derive(data):
    indexes, images = validate_panel(data)
    risks, margins = [], []
    for cell in sorted(indexes["spectral_cells.csv"]):
        row = indexes["spectral_cells.csv"][cell]
        saved = indexes["cell_risk_reference.csv"][cell]
        gamma, eta, total, selected = (number(row[k]) for k in ("gamma", "eta_LS", "trivial_e", "trivial_b"))
        lower, upper = 2 * (1 - gamma), 2 * (1 - gamma * gamma)
        values = {
            "gamma": gamma, "eta": eta, "e_triv": total, "b_triv": selected,
            "total_nontrivial_energy": 1 - total, "selected_nontrivial_energy": gamma - selected,
            "a": lower, "u": upper, "identity_risk": 2 * (1 - total),
            "premium_lower": max(0.0, lower - eta), "premium_upper": upper - eta,
        }
        values["positive_lower_beyond_tolerance"] = values["premium_lower"] > tolerance(values["premium_lower"])
        compare_saved(saved, values, "Saved per-cell risk differs for " + cell)
        require(abs(lower - number(row["L_lower"])) <= 1e-14 and abs(upper - number(row["L_upper"])) <= 1e-14,
                "Saved class bounds disagree with the spectral endpoints")
        identity = {k: saved[k] for k in ("cell", "domain", "model_id", "layer", "join_key")}
        risks.append(dict(identity, **values))
        margin = image_statistics(images[cell])
        require(abs(margin["semantic_gamma"] - gamma) <= tolerance(margin["semantic_gamma"], gamma),
                "Semantic image and spectral-cell endpoint disagree")
        compare_saved(indexes["registration_margins.csv"][cell], margin,
                      "Saved per-cell image summary differs for " + cell)
        margins.append(dict(identity, **margin))

    results, summaries = {}, []
    for domain in DOMAINS:
        rr = [r for r in risks if r["domain"] == domain]
        mm = [r for r in margins if r["domain"] == domain]
        ww = [r for r in data["witness_units.csv"] if r["domain"] == domain]
        counts = [sum(r["standing"] == label for r in mm) for label in ("unique_leading", "tied_leading", "behind")]
        means = lambda rows, keys: [f"{statistics.mean(number(r[k]) for r in rows):.5f}" for k in keys]
        lower_values = sorted((r["premium_lower"] for r in rr), reverse=True)
        summary = {
            "n": len(rr), "mean": statistics.mean(lower_values), "median": statistics.median(lower_values),
            "min": min(lower_values), "max": max(lower_values),
            "top_two_sum": math.fsum(lower_values[:2]), "all_cells_sum": math.fsum(lower_values),
            "top_two_share_percent": 100 * math.fsum(lower_values[:2]) / math.fsum(lower_values),
            "positive_beyond_original_tolerance": sum(r["positive_lower_beyond_tolerance"] for r in rr),
        }
        results[domain] = {
            "table3": means(rr, ("gamma", "a", "u", "eta", "b_triv")) + [f"{counts[0]}/24"],
            "table4": means(ww, ("train_witness_risk", "test_witness_risk", "test_ols_risk", "test_ridge_risk"))
            + [f"{sum(number(r['test_witness_risk']) > number(r['test_ols_risk']) for r in ww)}/24"],
            "table5": means(rr, ("eta", "identity_risk", "premium_lower", "premium_upper")),
            "table6": {key: [f"{func(r[key] for r in mm):.2e}" for func in (statistics.median, min, max)]
                       for key in ("d_top", "d_median")},
            "standing_counts": counts, "lower_endpoint_summary": summary,
        }
        summaries.append(dict(domain=domain, **summary))
    return results, risks, margins, summaries


def compare_expected(actual, expected, location="expected"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), location + ": key mismatch")
        for key in expected:
            compare_expected(actual[key], expected[key], location + "/" + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), location + ": length mismatch")
        for i, item in enumerate(expected):
            compare_expected(actual[i], item, location + "/" + str(i))
    elif isinstance(expected, float):
        require(math.isclose(actual, expected, rel_tol=5e-14, abs_tol=5e-15), location + ": aggregate mismatch")
    else:
        require(actual == expected, location + ": displayed value mismatch")


def scientific_tex(value):
    mantissa, exponent = value.split("e")
    return "$" + mantissa + r"\!\times\!10^{" + str(int(exponent)) + "}$"


def render_table(number_, results):
    headers = {
        3: r"Domain & Mean $\gamma_r$ & Mean bound endpoints & Mean $\eta$ & Mean $b_{\rm triv}$ & Image rank 1",
        4: r"Domain & Construction witness & Test witness & Test OLS & Test ridge & Witness worse than OLS",
        5: r"Domain & Mean $\eta$ & Mean $L(I)$ & Mean $p^-$ & Mean $p^+$",
        6: r"Domain & Margin & Median & Min. & Max. & U/T/B",
    }
    body = []
    for domain in DOMAINS:
        values = results[domain]["table" + str(number_)]
        label = "$" + domain[0] + "_" + domain[1] + "$" if number_ in (3, 4) else domain
        if number_ == 3:
            body.append([label, values[0], "[" + values[1] + ", " + values[2] + "]", *values[3:]])
        elif number_ in (4, 5):
            body.append([label, *values])
        else:
            for key, name in (("d_top", r"$d_{\rm top}$"), ("d_median", r"$d_{\rm median}$")):
                first = key == "d_top"
                body.append([label if first else "", name, *[scientific_tex(v) for v in values[key]],
                             "/".join(map(str, results[domain]["standing_counts"])) if first else ""])
    columns = {3: "lrrrrr", 4: "lrrrrr", 5: "lrrrr", 6: "llrrrc"}[number_]
    lines = ["% Reproduced from saved full-precision endpoints.", "\\begin{tabular}{" + columns + "}",
             r"\toprule", headers[number_] + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in body]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reproduce(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), "Output directory must be new; existing files are never overwritten")
    require(root != output and root not in output.parents, "Choose an output directory outside the evidence package")
    data, expected, manifest = load_inputs(root)
    results, risks, margins, summaries = derive(data)
    compare_expected(results, expected)
    output.mkdir(parents=True, exist_ok=False)
    for table in range(3, 7):
        (output / f"table{table}.tex").write_text(render_table(table, results), encoding="utf-8")
    write_csv(output / "cell_risk_recomputed.csv", risks)
    write_csv(output / "cell_margins_recomputed.csv", margins)
    write_csv(output / "lower_endpoint_summary.csv", summaries)
    # Retain unrounded domain means and full image-margin summaries as ordinary CSVs.
    domain_risk = []
    domain_margin = []
    for domain in DOMAINS:
        rr = [r for r in risks if r["domain"] == domain]
        mm = [r for r in margins if r["domain"] == domain]
        domain_risk.append(dict(domain=domain, n_cells=len(rr), **{
            "mean_" + key: statistics.mean(r[key] for r in rr)
            for key in ("gamma", "eta", "a", "u", "b_triv", "identity_risk", "premium_lower", "premium_upper")}))
        summary = dict(domain=domain, n_cells=len(mm), **dict(zip(
            ("unique_leading", "tied_leading", "behind"), results[domain]["standing_counts"])))
        for key in ("d_top", "d_median", "spread"):
            for label, func in (("median", statistics.median), ("min", min), ("max", max)):
                summary[key + "_" + label] = func(r[key] for r in mm)
        domain_margin.append(summary)
    write_csv(output / "domain_risk_summary.csv", domain_risk)
    write_csv(output / "domain_margin_summary.csv", domain_margin)
    (output / "reproduced_values.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    # Check again after writing: output generation cannot silently alter evidence.
    load_inputs(root)
    report = {
        "status": "PASS", "input_hashes_verified": len(manifest["files"]),
        "spectral_cells": 48, "witness_cells": 48, "image_endpoints": 2952, "canonical_images": 123,
        "per_cell_risk_and_margin_checks": "exact equality with saved float values",
        "displayed_tables": "exact equality with frozen expected numerical strings",
        "c7_top_two_lower_endpoint_share_percent": results["C7"]["lower_endpoint_summary"]["top_two_share_percent"],
        "original_spectral_computation_repeated": False, "coordinate_identity_residual_check_repeated": False,
        "model_execution": False, "family_deletion_or_resampling": False,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Fresh output directory outside this package")
    args = parser.parse_args()
    try:
        report = reproduce(args.output)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.exit(1, "Reproduction failed: " + str(exc) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
