"""Draw the conceptual evaluation diagram; consumes no experimental data."""
from pathlib import Path
import argparse
import hashlib
import json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor


def make(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(650, 232), invariant=1)
    c.setTitle("Three evaluation targets in fixed representations")
    c.setAuthor("Anonymous authors")
    dark = HexColor("#233341")
    c.setStrokeColor(dark)

    def box(x, y, w, h, fill):
        c.setFillColor(HexColor(fill))
        c.setStrokeColor(HexColor("#82939e"))
        c.roundRect(x, y, w, h, 5, fill=1, stroke=1)

    def line(x1, y1, x2, y2, arrow=False):
        c.setStrokeColor(dark)
        c.setLineWidth(0.9)
        c.line(x1, y1, x2, y2)
        if arrow:
            import math
            angle = math.atan2(y2-y1, x2-x1)
            for off in (-0.48, 0.48):
                c.line(x2, y2, x2-5*math.cos(angle+off), y2-5*math.sin(angle+off))

    def txt(x, y, value, size=10, bold=False):
        c.setFillColor(dark)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawCentredString(x, y, value)

    top = [(17, "Frozen language model"), (177, "Selected readout"),
           (337, "Training-fitted PCA"), (497, "Fixed coordinates")]
    for x, label in top:
        box(x, 189, 136, 29, "#edf1f4")
        txt(x+68, 200, label, size=9.5, bold=True)
    for (x, _), (nx, _) in zip(top, top[1:]):
        line(x+136, 203.5, nx-4, 203.5, arrow=True)
    # All evaluations branch from the same fixed-coordinate observation stage.
    line(565, 189, 565, 169)
    line(113, 169, 565, 169)
    for center in (113, 325, 537):
        line(center, 169, center, 150, arrow=True)
    branches = [
        (18, "#edf4fa", "Fitted prediction", [
            "Predict transformed activations",
            "Maps fitted on training templates",
            "Score on held-out contexts",
            "Prediction / consistency losses"]),
        (230, "#eaf4f1", "Exact-action class risk", [
            "Optimize over a declared map class",
            "Complete labeled state orbits",
            "Compare linear, orthogonal, exact",
            "Same-bank whitened metric"]),
        (442, "#faf2e7", "Registered alternatives", [
            "Compare coherent state actions",
            "Same source bank and dimension",
            "Different transformed targets",
            "Spectral-budget preference"]),
    ]
    for x, fill, heading, lines in branches:
        box(x, 45, 190, 101, fill)
        txt(x+95, 128, heading, size=10.5, bold=True)
        for j, value in enumerate(lines):
            txt(x+95, 109-15*j, value, size=9)
    txt(325, 20, "Native causal use requires separate behavioral and intervention evidence.", size=9)
    c.showPage()
    c.save()
    receipt = {"operation": "conceptual_vector_diagram", "experimental_inputs": [],
               "new_empirical_claims": False, "source": Path(__file__).name,
               "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".source.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    make(parser.parse_args().output)
