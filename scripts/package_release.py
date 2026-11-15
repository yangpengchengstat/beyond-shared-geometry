#!/usr/bin/env python3
"""Current packaging entry point. Earlier recipes are preserved in release/history."""
from pathlib import Path
import argparse
import json
from package_submission import package

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--output-dir", dest="output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(package(Path(__file__).resolve().parents[1], args.output.resolve()), indent=2))
