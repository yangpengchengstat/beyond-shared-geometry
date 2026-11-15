#!/usr/bin/env bash
set -uo pipefail
cpu_contract=false
if [[ "${1:-}" == "--cpu-contract" ]]; then
  cpu_contract=true
elif [[ -n "${1:-}" ]]; then
  echo "usage: bash runpod/preflight.sh [--cpu-contract]" >&2
  exit 2
fi
if [[ "${cpu_contract}" == true ]]; then
  export BSG_CPU_CONTRACT=1
else
  export BSG_CPU_CONTRACT=0
fi
pass=true

check() {
  local label="$1"
  shift
  echo "== $label =="
  if "$@"; then
    echo "$label: ok"
  else
    echo "$label: failed"
    pass=false
  fi
}

check "python" python3 --version
if [[ "${cpu_contract}" == false ]]; then
  check "nvidia-smi" command -v nvidia-smi
fi

echo "== torch/cuda/bsg/config =="
PYTHONPATH=src python3 - <<'PY' || pass=false
import os, sys
print('python', sys.version.split()[0])
try:
    import torch
    print('torch', torch.__version__)
    cuda_available = torch.cuda.is_available()
    print('cuda_available', cuda_available)
    if not cuda_available and os.environ.get('BSG_CPU_CONTRACT') != '1':
        raise SystemExit(2)
    if cuda_available:
        print('gpu', torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print('vram_gib', round(props.total_memory / 1024**3, 2))
except ModuleNotFoundError as exc:
    if os.environ.get('BSG_CPU_CONTRACT') == '1':
        print('torch unavailable in CPU contract mode:', exc)
    else:
        raise
except Exception as exc:
    print('torch/cuda check failed:', exc)
    raise SystemExit(2)
import bsg
print('bsg', bsg.__version__)
from bsg.experiments.main_config import resolve_main_config
cfg = resolve_main_config('configs/main_experiment.yaml', 'smoke')
print('canonical smoke config ok', [m['id'] for m in cfg['selected_models']])
if os.environ.get('HF_TOKEN'):
    print('HF_TOKEN present')
elif os.environ.get('BSG_CPU_CONTRACT') == '1':
    print('HF_TOKEN not required in CPU contract mode')
else:
    print('HF_TOKEN missing')
    raise SystemExit(3)
PY

echo "== required files =="
[[ -f configs/main_experiment.yaml ]] || pass=false
[[ -f runpod/run_main_experiment.sh ]] || pass=false
[[ -f docs/EXPERIMENT_PROTOCOL.md ]] || pass=false
PYTHONPATH=src python3 -m bsg.experiments.main_experiment --smoke --validate-only || pass=false

echo "== runtime-critical checks =="
if [[ "${cpu_contract}" == true ]]; then
  BSG_CPU_CONTRACT=1 PYTHONPATH=src python3 tests/run_runtime_checks.py || pass=false
else
  PYTHONPATH=src python3 tests/run_runtime_checks.py || pass=false
fi
if [[ "$pass" == true ]]; then
  echo "RUNPOD RUNTIME CHECKS: PASS"
fi

if [[ "$pass" == true ]]; then
  if [[ "${cpu_contract}" == true ]]; then
    echo "RUNPOD CPU CONTRACT: PASS"
  fi
  echo "PRE-RUNPOD CHECK: PASS"
  exit 0
else
  echo "PRE-RUNPOD CHECK: FAIL"
  exit 1
fi
