#!/usr/bin/env bash
# Explicit continuation after the observed original q16 embedding rank failure.
set +x
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project="$(cd -- "$here/../.." && pwd -P)"
if [[ $# -ne 2 ]]; then echo 'Usage: bash start.sh /workspace/bsg-runs/from-zero-v1 /workspace/bsg-runs/from-zero-v1-rank-recovery'; exit 2; fi
parent="$1"
run="$2"
py="${BSG_REPRO_VENV:-/workspace/bsg-reproduction-venv}/bin/python"
export HF_HOME=/workspace/bsg-model-cache
export HF_HUB_ENABLE_HF_TRANSFER=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$py" "$here/continue.py" verify --parent "$parent" --output "$run"
mkdir -p "$run/environment"
exec 9>"$run/.launch.lock"
if ! flock -n 9; then echo 'Another recovery launcher owns this output.' >&2; exit 2; fi
"$py" - "$parent/environment" "$run/environment" <<'PY'
from pathlib import Path
import shutil,sys
source,target=map(Path,sys.argv[1:])
for relative in ('model_inventory.json','runtime_identity.json','model_inventory','resolved_environment'):
    p=source/relative
    if not p.exists():raise SystemExit('Missing parent environment provenance: '+relative)
    for original in (p.rglob('*') if p.is_dir() else [p]):
        if not original.is_file():continue
        if original.is_symlink():raise SystemExit('Unexpected provenance symlink')
        destination=target/original.relative_to(source)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if destination.read_bytes()!=original.read_bytes():raise SystemExit('Existing environment provenance differs')
        else:shutil.copyfile(original,destination)
PY
"$py" "$project/studies/reproduction_v1/environment.py" --output "$run/environment" --hardware \
  --minimum-free-gib 15 --minimum-ram-gib 32 --minimum-vram-gib 44 \
  --config "$project/configs/prospective/bf16_revision.json" \
  --config "$project/configs/prospective/peer_anchors.json" \
  --config "$project/configs/main_experiment.yaml" --verify-models
exec "$py" -u "$here/continue.py" run --parent "$parent" --output "$run" --execute-models
