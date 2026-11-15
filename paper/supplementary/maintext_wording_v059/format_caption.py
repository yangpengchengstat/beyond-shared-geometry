#!/usr/bin/env python3
"""Make the v0.5.9 caption-only table derivative; performs no scientific analysis."""
from pathlib import Path
import hashlib
import json
import re
import sys

def main():
    paper = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "paper"
    source = paper / "generated/core_risk_decomposition.tex"
    target = paper / "generated/core_risk_decomposition_v059.tex"
    text = source.read_text()
    replacements = [
        ("(a) Original joint-constraint comparison", "(a) Joint constraint cost"),
        ("Panel (a) retains the primary joint-cost comparison. In (b),", "In panel (b),"),
    ]
    revised = text
    for before, after in replacements:
        if revised.count(before) != 1:
            raise ValueError("Caption source does not match the declared baseline")
        revised = revised.replace(before, after)
    # This check preserves every numeric token, all mathematical expressions,
    # tabular contents, labels and scientific caveats; the two edits affect prose only.
    numbers = lambda value: re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", value)
    assert numbers(text) == numbers(revised)
    target.write_text(revised)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = {
        "operation": "caption_only_wording_revision_no_scientific_computation",
        "source": str(source.relative_to(paper)), "source_sha256": sha(source),
        "output": str(target.relative_to(paper)), "output_sha256": sha(target),
        "numeric_tokens_unchanged": True,
        "replacements": [{"before":a,"after":b} for a,b in replacements],
    }
    (target.with_suffix(".provenance.json")).write_text(json.dumps(receipt, indent=2)+"\n")
    print("CAPTION_DERIVATIVE_VERIFIED; numerical cells and source retained")

if __name__ == "__main__":
    main()
