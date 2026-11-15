"""Format selected saved aggregate fields; no matrix or statistical analysis."""
import argparse
import hashlib
import json
from pathlib import Path


def render(coverage):
    if coverage.get("status") != "completed":
        raise ValueError("Expected the completed saved sensitivity coverage")
    rows = {(r["domain"], r["dimension"]): r for r in coverage["domain_summaries"]}
    lines = [
        "% Format-only selection from the completed sensitivity coverage.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Domain & $r$ & Mean $\gamma_r$ & Median $p^-$ & Unique leading image \\",
        r"\midrule",
    ]
    for domain in ("C7", "D4"):
        for dimension in (16, 8):
            row = rows[domain, dimension]
            if row["aggregate_status"] != "complete_descriptive" or row["fully_evaluable_cells"] != 24:
                raise ValueError("Incomplete saved domain aggregate")
            group = domain[0] + "_" + domain[1]
            lines.append(
                f"${group}$ & {dimension} & {row['means']['gamma']:.6f}"
                f" & {row['medians']['p_minus']:.6f}"
                f" & {row['classification_counts']['unique_leading']}/24 " + r"\\"
            )
        if domain == "C7":
            lines.append(r"\addlinespace")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Retained-dimension sensitivity on the same complete Study~2 banks. Each row summarizes all 24 fixed conditions; $p^-$ is the lower endpoint for joint exact-law and orthogonality cost. The last column counts a uniquely highest budget for the specified semantic image among all 120 C7 or three D4 images, using the original tolerance. Eight coordinates are the leading eight saved PCA coordinates. The dimensions have different whitened metrics and normalizations; these are not optimal-dimension comparisons or significance tests. Full values and checks remain in \cref{app:spectral-dimension-sensitivity}.}",
        r"\label{tab:sensitivity-main}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.coverage.read_bytes()
    result = render(json.loads(source))
    with args.output.open("x", encoding="utf-8") as f:
        f.write(result)
    provenance = {
        "operation": "select_and_format_saved_aggregates_only",
        "source_name": args.coverage.name,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "fields": ["means.gamma", "medians.p_minus", "classification_counts.unique_leading"],
        "rows": "C7/16,C7/8,D4/16,D4/8; all 24 recorded cells per row",
        "rounding": "six decimal places, matching existing sensitivity table",
        "output_sha256": hashlib.sha256(result.encode()).hexdigest(),
        "matrix_analysis": False,
        "statistical_reanalysis": False,
        "experiment_execution": False,
    }
    args.output.with_suffix(".source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
