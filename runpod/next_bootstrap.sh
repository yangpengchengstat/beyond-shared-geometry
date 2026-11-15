#!/usr/bin/env bash
# Install the prospective runtime; this never downloads model checkpoints.
set +x
set -euo pipefail

mode="${1:-gpu}"
if [[ "$mode" != "cpu" && "$mode" != "gpu" ]]; then
  echo "Usage: bash runpod/next_bootstrap.sh [cpu|gpu]" >&2
  exit 2
fi
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
venv_dir="${BSG_VENV:-/workspace/bsg-${mode}-venv}"
base_python="${BSG_BASE_PYTHON:-python3}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# Some container images enable an optional transfer backend without installing
# it. Standard Hub downloads work without that backend and reuse the same cache.
export HF_HUB_ENABLE_HF_TRANSFER="${BSG_HF_TRANSFER:-0}"
mkdir -p -- "$(dirname -- "$venv_dir")" "$HF_HOME"
"$base_python" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,14), "Use Python 3.10–3.13 for these pinned wheels (3.11 recommended)"'
if [[ ! -x "$venv_dir/bin/python" ]]; then
  "$base_python" -m venv "$venv_dir"
fi
python_bin="$venv_dir/bin/python"
BSG_BOOTSTRAP_MODE="$mode" "$python_bin" - "$venv_dir/prospective-environment.json" <<'PY'
import json
import os
import sys
from pathlib import Path
manifest = Path(sys.argv[1])
if manifest.is_file():
    previous = json.loads(manifest.read_text())
    if previous.get("mode") not in (None, os.environ["BSG_BOOTSTRAP_MODE"]):
        raise SystemExit("Use separate CPU and GPU virtual environments; set BSG_VENV to a fresh path.")
PY
"$python_bin" -m pip install --upgrade pip setuptools wheel
if [[ "$mode" == "cpu" ]]; then
  torch_index="${BSG_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
else
  torch_index="${BSG_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
fi
"$python_bin" -m pip install 'torch==2.8.0' --index-url "$torch_index"
BSG_BOOTSTRAP_MODE="$mode" "$python_bin" - <<'PY'
import os
import torch
mode = os.environ["BSG_BOOTSTRAP_MODE"]
if (mode == "gpu") != (torch.version.cuda is not None):
    raise SystemExit("The installed Torch wheel belongs to the other CPU/GPU mode. Use a fresh BSG_VENV path; do not copy a CPU environment to a GPU pod.")
PY
"$python_bin" -m pip install -r "$project_dir/runpod/next_requirements_${mode}.txt"
"$python_bin" -m pip install -e "$project_dir[prospective,gpu,test]" -c "$project_dir/runpod/next_requirements_cpu.txt"
"$python_bin" -m pip check
"$python_bin" -m pip freeze > "$venv_dir/prospective-freeze.txt"
BSG_BOOTSTRAP_MODE="$mode" "$python_bin" - "$venv_dir/prospective-environment.json" <<'PY'
import importlib.metadata as metadata
import json
import os
import platform
import sys
from pathlib import Path
import torch
packages = ("numpy", "scipy", "torch", "transformers", "accelerate", "huggingface_hub", "tokenizers", "safetensors")
data = {"python": sys.version, "platform": platform.platform(), "mode": os.environ["BSG_BOOTSTRAP_MODE"],
        "packages": {name: metadata.version(name) for name in packages}, "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "models_downloaded_by_bootstrap": False}
Path(sys.argv[1]).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(json.dumps(data, indent=2))
PY
echo "Prospective runtime installed. Checkpoint downloads require a later --execute-models command."
echo "Use: source $venv_dir/bin/activate"
