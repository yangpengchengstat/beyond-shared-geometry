#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HOME="${HF_HOME:-/workspace/bsg-model-cache}"
BSG_BATCH_PYTHON="${BSG_BATCH_PYTHON:-/workspace/bsg-tmlr-venv/bin/python}"
exec "$BSG_BATCH_PYTHON" -u "$(dirname "$0")/run.py" "$@"
