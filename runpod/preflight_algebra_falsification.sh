#!/usr/bin/env bash
set -uo pipefail

cpu_contract=false
if [[ "${1:-}" == "--cpu-contract" ]]; then cpu_contract=true; elif [[ -n "${1:-}" ]]; then echo "usage: bash runpod/preflight_algebra_falsification.sh [--cpu-contract]" >&2; exit 2; fi
pass=true

PYTHONPATH=src python3 -m bsg.experiments.algebra_falsification --smoke --validate-only || pass=false
[[ -f configs/algebra_falsification.yaml ]] || pass=false
[[ -f docs/ALGEBRA_FALSIFICATION_PROTOCOL.md ]] || pass=false
[[ -f runpod/run_algebra_falsification.sh ]] || pass=false
[[ -f runpod/package_afs_results.sh ]] || pass=false
bash -n runpod/run_algebra_falsification.sh || pass=false
bash -n runpod/package_afs_results.sh || pass=false

if [[ "${cpu_contract}" == false ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || pass=false
  PYTHONPATH=src python3 - <<'PY' || pass=false
import os
from pathlib import Path
from bsg.experiments.algebra_falsification_units import HistoricalCheckpointSource
from bsg.experiments.algebra_falsification_config import resolve_afs_config
root = Path(os.environ.get("BSG_MAIN_FULL_003_ROOT", "/workspace/run_outputs/main_experiment/main_full_003"))
config = resolve_afs_config("configs/algebra_falsification.yaml", "smoke")
HistoricalCheckpointSource(root, config["historical_source"]).assert_unchanged()
if not os.environ.get("HF_TOKEN"):
    raise SystemExit("HF_TOKEN is required for embedding-layer extraction")
print("historical main_full_003 and HF token: ok")
PY
fi

PYTHONPATH=src python3 tests/run_runtime_checks.py || pass=false
if [[ "${pass}" == true ]]; then echo "AFS PRE-RUNPOD CHECK: PASS"; exit 0; fi
echo "AFS PRE-RUNPOD CHECK: FAIL"; exit 1
