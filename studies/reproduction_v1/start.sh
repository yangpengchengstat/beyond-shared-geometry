#!/usr/bin/env bash
# Explicit paid-machine execution; bootstrap and authentication happen beforehand.
set +x
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project="$(cd -- "$here/../.." && pwd -P)"
if [[ $# -ne 1 ]]; then echo 'Usage: bash studies/reproduction_v1/start.sh /workspace/bsg-runs/from-zero-v1'; exit 2; fi
run="$1"
venv="${BSG_REPRO_VENV:-/workspace/bsg-reproduction-venv}"
py="$venv/bin/python"
export HF_HOME="${HF_HOME:-/workspace/bsg-model-cache}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$py" "$here/run.py" verify
# Validate the destination before copying provenance or downloading anything.
PYTHONPATH="$here" "$py" - "$run" <<'PY'
from receipt import safe_output
import sys
safe_output(sys.argv[1])
PY
mkdir -p "$run/environment"
exec 9>"$run/.launch.lock"
if ! flock -n 9; then echo 'Another launcher is already active for this output.' >&2; exit 2; fi
"$py" - "$venv/reproduction-provenance" "$run/environment/resolved_environment" <<'PY'
from pathlib import Path
import hashlib, shutil, sys
src,dst=map(Path,sys.argv[1:]);dst.mkdir(parents=True,exist_ok=True)
if not (src/'environment.requirements.lock').exists():raise SystemExit('Run bootstrap.sh before start.sh')
for name in ('environment.requirements.lock','resolution-report.json','installation-report.json','environment.freeze.txt','runtime_identity.json'):
    p=src/name
    if not p.exists():continue
    q=dst/name
    if q.exists() and q.read_bytes()!=p.read_bytes():raise SystemExit('Environment provenance changed; choose a fresh run')
    if not q.exists():shutil.copyfile(p,q)
PY
# A full replay retains several independent experiment cache trees. Reserve 180GiB
# before initial download, and 100GiB after weights; on resumes use a 15GiB floor.
minimum=180
if [[ -f "$run/identity.json" ]]; then minimum=15; fi
"$py" "$here/environment.py" --output "$run/environment" --hardware \
  --minimum-free-gib "$minimum" --minimum-ram-gib 32 --minimum-vram-gib 44 \
  --config "$project/configs/prospective/bf16_revision.json" \
  --config "$project/configs/prospective/peer_anchors.json" \
  --config "$project/configs/main_experiment.yaml" \
  --download-models
minimum=100
if [[ -f "$run/identity.json" ]]; then minimum=15; fi
"$py" "$here/environment.py" --output "$run/environment" --hardware \
  --minimum-free-gib "$minimum" --minimum-ram-gib 32 --minimum-vram-gib 44 \
  --config "$project/configs/prospective/bf16_revision.json" \
  --config "$project/configs/prospective/peer_anchors.json" \
  --config "$project/configs/main_experiment.yaml"
exec "$py" -u "$here/run.py" run --output "$run" --execute-models
