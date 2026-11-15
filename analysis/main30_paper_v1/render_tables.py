"""Relabel existing tables without changing their numerical bodies."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
CAPTIONS = {
    "followup_spectral": r"Finite-bank exact-action bounds in the test-bank metric. Each domain has 24 model/readout cells at $r=16$. The bounds concern cell-specific $L^\star$ over the entire exact orthogonal-action class in each test bank's own second-moment-whitened metric; displayed endpoints average those bounds, not confidence limits. $b_{\rm triv}$ is selected trivial-type energy divided by $r$. The last column counts unique highest-$\gamma_r$ ranks of the specified image among 120 C7 or three D4 images; it is not a rank of exact $L^\star$ or a significance test.",
    "followup_witness": r"Transfer of frozen predictors in the training metric. These means use training whitening for both construction and old test banks, including identity and fixed points. OLS and ridge need not obey group laws. The witness is feasible, not claimed optimal. Its transfer loss is not checked against Table~\ref{tab:followup-spectral}'s test-metric exact-orthogonal bounds: changing whitening can also change admissibility of the same raw-coordinate function (\cref{app:metric-conjugation}).",
}


def render(root=ROOT):
    records = []
    for stem, caption in CAPTIONS.items():
        src = root / "paper/generated" / (stem + ".tex")
        dst = src.with_name(stem + "_bounded.tex")
        text = src.read_text()
        start = text.index("\\caption{")
        end = text.index("\n\\label{", start)
        replacement = text[:start] + "\\caption{" + caption + "}" + text[end:]
        # Existing rows, numbers and label remain byte-identical.
        assert replacement.split("\\caption{")[0] == text.split("\\caption{")[0]
        assert replacement[replacement.index("\n\\label{"):] == text[end:]
        dst.write_text(replacement)
        records.append({"input": str(src.relative_to(root)),
                        "input_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                        "output": str(dst.relative_to(root)),
                        "output_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
                        "operation": "Caption replacement only; numerical body and label unchanged"})
    for name in ["bounded_macros", "risk_reference_table", "registration_margin_table",
                 "cell_risk_C7", "cell_risk_D4", "cell_margins_C7", "cell_margins_D4",
                 "cell_type_energy_C7", "cell_type_energy_D4"]:
        src = root / "results/main30_bounded_v1" / (name + ".tex")
        stem = name if name.startswith("bounded_") else "bounded_" + name
        dst = root / "paper/generated" / (stem + ".tex")
        dst.write_bytes(src.read_bytes())
        records.append({"input": str(src.relative_to(root)),
                        "input_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                        "output": str(dst.relative_to(root)),
                        "output_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
                        "operation": "Byte-identical import of generated full-precision-data summary"})
    return records


if __name__ == "__main__":
    records = render()
    path = ROOT / "analysis/main30_paper_v1/render_manifest.json"
    path.write_text(json.dumps({"records": records}, indent=2) + "\n")
    print("Rendered two caption-only table variants.")
