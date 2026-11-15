"""Verify saved inputs, or explicitly reproduce the archived centering diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))


def verify():
    from retained_inputs import Study2Inputs, sha256
    source = json.loads((HERE / "PUBLICATION_DERIVATIVES.json").read_bytes())
    for name, item in source.items():
        original = (HERE / item["original_source_path"]).read_bytes()
        derivative = (HERE / name).read_bytes()
        if hashlib.sha256(original).hexdigest() != item["original_source_sha256"]:
            raise ValueError("Original scientific script checksum differs")
        if hashlib.sha256(derivative).hexdigest() != item["publication_source_sha256"]:
            raise ValueError("Publication path adapter checksum differs")
        restored = derivative.decode()
        for change in reversed(item["changes"]):
            if restored.count(change["after"]) != 1:
                raise ValueError("Unexpected publication adaptation")
            restored = restored.replace(change["after"], change["before"])
        if restored.encode() != original:
            raise ValueError("Changes extend beyond the recorded path/import adaptations")
    kernel = REPO / "studies/spectral_dimension_sensitivity_v1/core/_vendor/gamma_r_frozen.py"
    if sha256(kernel) != "e91c735890bdf888a25db07d1207d3f44cdce8ce1bf6ee7e2efcf7bf40a50f41":
        raise ValueError("Frozen mathematical kernel checksum differs")
    for form in ("rewhitened", "three_way"):
        directory = REPO / "results/template_centering_v1" / form
        receipt = json.loads((directory / "archive_record.json").read_bytes())
        for filename, expected in receipt["archived_files_sha256"].items():
            path = directory / filename
            if path.is_file() and sha256(path) != expected:
                raise ValueError("Saved historical diagnostic result changed")
    protocol = json.loads((HERE / "historical_protocol.json").read_bytes())
    with Study2Inputs(REPO, protocol) as inputs:
        answer = inputs.preflight()
    print(json.dumps(answer, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Standard-library hash and row-binding verification only")
    run = sub.add_parser("run", help="Explicit offline mathematical reproduction; never loads a language model")
    run.add_argument("--stage", choices=("rewhitened", "fixed"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--execute-offline-analysis", action="store_true", required=True)
    args = parser.parse_args()
    verify()
    if args.command == "verify":
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Numerical reproduction requires a fresh output directory")
    # No numerical module is imported before the explicit run command and gate.
    import numpy as np
    if np.__version__ != "2.2.6":
        raise ValueError("The archived diagnosis requires NumPy 2.2.6")
    output.mkdir(parents=True, exist_ok=False)
    os.environ["BSG_CENTERING_OUTPUT"] = str(output)
    filename = "diagnose_rewhitened.py" if args.stage == "rewhitened" else "diagnose_fixed_metric.py"
    runpy.run_path(str(HERE / filename), run_name="__main__")


if __name__ == "__main__":
    main()
