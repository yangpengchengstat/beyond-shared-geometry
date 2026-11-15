"""Summarize retained p-minus CSV rows; no original analysis is rerun."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "results/main30_bounded_v1/cell_risk_reference.csv"
SOURCE_SHA256 = "ec52b77b536d8241edf5229b548d559ec09b8845454807946e03a4bf6171ae61"


def summarize(root=ROOT):
    source = root / SOURCE
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Retained full-precision input changed")
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 48 or len({row["cell"] for row in rows}) != 48:
        raise ValueError("Expected all 48 unique cells")
    if {row["domain"] for row in rows} != {"C7", "D4"}:
        raise ValueError("Unexpected domain coverage")
    domains = {}
    with localcontext() as ctx:
        ctx.prec = 50
        for domain in ("C7", "D4"):
            selected = [row for row in rows if row["domain"] == domain]
            if len(selected) != 24:
                raise ValueError("Expected 24 equally weighted cells per domain")
            values = [Decimal(row["premium_lower"]) for row in selected]
            if not all(value.is_finite() and value >= 0 for value in values):
                raise ValueError("Invalid saved lower endpoint")
            domains[domain] = {
                "n": len(values), "median": str(median(values)),
                "min": str(min(values)), "max": str(max(values)),
                "mean": str(sum(values) / len(values)), "sum": str(sum(values)),
            }
            if domain == "C7":
                largest = sorted(selected, key=lambda row: Decimal(row["premium_lower"]), reverse=True)[:2]
                top_sum = sum(Decimal(row["premium_lower"]) for row in largest)
                domains[domain]["largest_two"] = [
                    {key: row[key] for key in ("cell", "model_id", "layer", "premium_lower")}
                    for row in largest
                ]
                domains[domain]["largest_two_share_percent"] = str(100 * top_sum / sum(values))
    return {
        "input": SOURCE, "input_sha256": SOURCE_SHA256,
        "field": "premium_lower = p_minus = max(0, 2(1-gamma)-eta)",
        "decimal_policy": "Parse saved CSV decimals exactly; 50-digit arithmetic for sums and division",
        "domains": domains, "excluded_cells": 0,
        "weighting": "unchanged equal weight for all 24 cells within each domain",
        "interpretation": "Lower bounds on the joint exact-group-law and orthogonality class premium, not exact premiums",
        "original_analyses_rerun": False, "new_model_execution": False,
        "lofo_status": "PREPARED_NOT_RUN",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = summarize()
    if args.check:
        if json.loads(args.output.read_text()) != report:
            raise ValueError("Saved heterogeneity summary differs")
        print("RETAINED_CSV_SUMMARY_VERIFIED: 48 cells; no exclusions")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("SAVED_CSV_SUMMARY_ONLY: no models, fitting or LOFO")
