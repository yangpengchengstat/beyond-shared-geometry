#!/usr/bin/env bash
set -euo pipefail
study_release_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/workspace/bsg-gpu-venv/bin/python -u "$study_release_dir/run_study.py" --self-test
exec timeout --signal=TERM --kill-after=60s 6h \
  /workspace/bsg-gpu-venv/bin/python -u "$study_release_dir/run_study.py" --execute-models "$@"
