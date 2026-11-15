#!/usr/bin/env bash
# Ordered prospective stage launcher; legacy runpod launchers are unchanged.
set +x
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash runpod/next_run.sh PHASE SAME_RUN_DIRECTORY [bsg-next run options]" >&2
  echo "PHASE: cpu, retrospective, pilot, confirm, full, peers, crossmodel, extensions" >&2
  echo "CPU work on a detected GPU requires --allow-cpu-on-gpu. full additionally requires --allow-combined-full." >&2
  exit 2
fi
phase="$1"
run_dir="$2"
shift 2
case "$phase" in cpu|retrospective|pilot|confirm|full|peers|crossmodel|extensions) ;; *) echo "Unknown phase: $phase" >&2; exit 2 ;; esac
allow_cpu_on_gpu=false
allow_combined_full=false
arguments=()
for argument in "$@"; do
  case "$argument" in
    --allow-cpu-on-gpu) allow_cpu_on_gpu=true ;;
    --allow-combined-full) allow_combined_full=true ;;
    --phase|--phase=*|--output|--output=*|--config|--config=*)
      echo "Do not override --phase/--output/--config in trailing options. Use the positional phase/run and BSG_CONFIG." >&2
      exit 2 ;;
    *) arguments+=("$argument") ;;
  esac
done
if [[ "$phase" == "full" && "$allow_combined_full" != true ]]; then
  echo "full combines substantial CPU calibration with model execution. Run the separate CPU/GPU phases; an intentional combined run requires --allow-combined-full." >&2
  exit 2
fi
if [[ "$phase" != "cpu" && "$phase" != "retrospective" ]]; then
  permitted=false
  for argument in ${arguments[@]+"${arguments[@]}"}; do
    if [[ "$argument" == "--execute-models" ]]; then permitted=true; fi
  done
  if [[ "$permitted" != true ]]; then
    echo "This phase loads model checkpoints. Add --execute-models explicitly." >&2
    exit 2
  fi
fi
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$phase" == "cpu" || "$phase" == "retrospective" ]]; then runtime_mode=cpu; else runtime_mode=gpu; fi
venv_dir="${BSG_VENV:-/workspace/bsg-${runtime_mode}-venv}"
python_bin="${BSG_PYTHON:-$venv_dir/bin/python}"
config_path="${BSG_CONFIG:-$project_dir/configs/prospective/bf16_revision.json}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${BSG_HF_TRANSFER:-0}"
if [[ ! -x "$python_bin" ]]; then
  echo "Missing runtime: $python_bin. Run next_bootstrap.sh first or set BSG_PYTHON." >&2
  exit 2
fi
if [[ "$phase" == "cpu" || "$phase" == "retrospective" || "$phase" == "full" ]]; then
  gpu_detected=false
  if command -v nvidia-smi >/dev/null 2>&1; then
    if gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)" && [[ -n "$gpu_names" ]]; then
      gpu_detected=true
    fi
  fi
  if [[ "$gpu_detected" != true ]] && "$python_bin" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    gpu_detected=true
  fi
  if [[ "$gpu_detected" == true && "$allow_cpu_on_gpu" != true ]]; then
    echo "A GPU is visible. CPU calibration will leave it idle while the pod can keep charging. Use a CPU machine, or explicitly add --allow-cpu-on-gpu after reviewing the measured cost." >&2
    exit 2
  fi
fi
if [[ ! -f "$config_path" ]]; then
  echo "Missing study profile: $config_path. Set BSG_CONFIG to the frozen profile used for every phase." >&2
  exit 2
fi
cd -- "$project_dir"
echo "Phase: $phase; profile: $config_path; run: $run_dir" >&2
exec "$python_bin" -m bsg_next.cli run --phase "$phase" --config "$config_path" --output "$run_dir" ${arguments[@]+"${arguments[@]}"}
