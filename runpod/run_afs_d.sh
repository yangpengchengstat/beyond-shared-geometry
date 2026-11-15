#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "${mode}" in
  --validate-only|--preflight|--run) shift ;;
  *) echo "usage: bash runpod/run_afs_d.sh --validate-only|--preflight|--run [AFS-D args]" >&2; exit 2 ;;
esac

for argument in "$@"; do
  if [[ "${argument}" == "--output-root" || "${argument}" == "--run-id" ]]; then
    echo "set BSG_AFS_D_OUTPUT_ROOT/BSG_AFS_D_RUN_ID instead of passing ${argument}" >&2
    exit 2
  fi
done

full_root="${BSG_MAIN_FULL_003_RESULTS_ROOT:-/workspace/immutable/main_full_003_results}"
afs_root="${BSG_AFS_ABC_001_ROOT:-/workspace/immutable/afs_full_abc_001}"
output_root="${BSG_AFS_D_OUTPUT_ROOT:-/workspace/run_outputs/algebra_falsification}"
run_id="${BSG_AFS_D_RUN_ID:-afs_d_rank_aware_$(date -u +%Y%m%dT%H%M%SZ)}"
internal_activation_source="${BSG_AFS_D_INTERNAL_ACTIVATION_SOURCE:-/workspace/immutable/run_outputs_9_4_2026_full3.zip}"

if [[ "${mode}" == "--validate-only" ]]; then
  PYTHONPATH=src python3 -m bsg.experiments.afs_d --validate-only --config configs/afs_d.yaml "$@"
  exit 0
fi

if [[ "${mode}" == "--preflight" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo "AFS-D preflight requires nvidia-smi" >&2; exit 1; }
  [[ -n "${HF_TOKEN:-}" ]] || { echo "AFS-D preflight requires HF_TOKEN" >&2; exit 1; }
  PYTHONPATH=src python3 - <<'PY'
import torch
import transformers
if not torch.cuda.is_available():
    raise SystemExit("AFS-D preflight requires CUDA")
print(f"AFS-D CUDA: {torch.cuda.get_device_name(0)}")
print(f"AFS-D transformers: {transformers.__version__}")
PY
fi

args=(
  "${mode}"
  --config configs/afs_d.yaml
  --full-results-root "${full_root}"
  --afs-abc-root "${afs_root}"
  --internal-activation-source "${internal_activation_source}"
)
if [[ "${mode}" == "--run" ]]; then
  args+=(--output-root "${output_root}" --run-id "${run_id}")
fi
args+=("$@")
PYTHONPATH=src python3 -m bsg.experiments.afs_d "${args[@]}"

if [[ "${mode}" == "--run" ]]; then
  run_dir="${output_root}/${run_id}"
  bash runpod/package_afs_d_results.sh "${run_dir}"
fi
