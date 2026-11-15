"""Readable vector layout of the existing three-target diagram; no empirical data."""
from pathlib import Path
import argparse
import hashlib
import json
import math
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor


def make(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(468, 180), invariant=1)
    c.setTitle("Three evaluation targets in fixed representations")
    c.setAuthor("Anonymous authors")
    dark = HexColor("#233341")

    def box(x, y, w, h, fill):
        c.setFillColor(HexColor(fill)); c.setStrokeColor(HexColor("#82939e"))
        c.roundRect(x, y, w, h, 5, fill=1, stroke=1)

    def line(x1, y1, x2, y2, arrow=False):
        c.setStrokeColor(dark); c.setLineWidth(.9); c.line(x1, y1, x2, y2)
        if arrow:
            a = math.atan2(y2-y1, x2-x1)
            for d in (-.48, .48):
                c.line(x2, y2, x2-5*math.cos(a+d), y2-5*math.sin(a+d))

    def text(x, y, s, size=9, bold=False):
        c.setFillColor(dark)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawCentredString(x, y, s)

    stages = [(6, ["Frozen", "language model"]), (122, ["Selected", "readout"]),
              (238, ["Training-fitted", "PCA"]), (354, ["Fixed", "coordinates"])]
    for x, labels in stages:
        box(x, 142, 108, 32, "#edf1f4")
        for j, label in enumerate(labels): text(x+54, 162-11*j, label, 9, True)
    for (x, _), (nx, _) in zip(stages, stages[1:]):
        line(x+108, 158, nx-1, 158, True)
    line(408, 142, 408, 126); line(80, 126, 408, 126)
    for center in (80, 234, 388): line(center, 126, center, 110, True)
    branches = [
        (8, "#edf4fa", ["Fitted", "prediction"], [
            "Predict transformed activations", "Fit maps on training templates",
            "Evaluate held-out contexts", "Prediction / consistency losses"]),
        (162, "#eaf4f1", ["Exact-action", "class risk"], [
            "Optimize over a declared class", "Complete labeled state orbits",
            "Linear, orthogonal, exact classes", "Same-bank whitened metric"]),
        (316, "#faf2e7", ["Registered", "alternatives"], [
            "Compare coherent state actions", "Same bank and dimension",
            "Different transformed targets", "Spectral-budget preference"]),
    ]
    for x, fill, headings, lines in branches:
        box(x, 22, 144, 88, fill)
        for j, label in enumerate(headings): text(x+72, 96-12*j, label, 10, True)
        for j, label in enumerate(lines): text(x+72, 66-12*j, label)
    text(234, 6, "Native causal use requires separate behavioral and intervention evidence.")
    c.showPage(); c.save()
    receipt = {"operation": "conceptual_vector_diagram_readability_revision",
               "experimental_inputs": [], "new_empirical_claims": False,
               "source": "supplementary/submission_presentation_v3/render_concept.py",
               "prior_source": "supplementary/maintext_revision/generate_concept_figure.py",
               "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".source.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    make(parser.parse_args().output)
