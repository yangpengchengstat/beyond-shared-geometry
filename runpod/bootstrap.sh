#!/usr/bin/env bash
set -euo pipefail
echo "== OS =="
uname -a
echo "== Python =="
python3 --version
echo "== NVIDIA =="
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi not found"; fi
echo "== Torch CUDA =="
python3 - <<'PY'
try:
    import torch
    print('torch', torch.__version__)
    print('cuda', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('gpu', torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print('vram_gib', round(props.total_memory / 1024**3, 2))
except Exception as exc:
    print('torch check failed:', exc)
PY
python3 -m pip install -e '.[gpu,analysis]'
python3 - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

path = Path("DEPLOYMENT_PROVENANCE.json")
if not path.exists():
    print("FULL LOCAL CI STATUS FROM PACKAGE: unavailable")
else:
    data = json.loads(path.read_text(encoding="utf-8"))
    status = data.get("full_cpu_tests_status", "unknown")
    count = data.get("full_cpu_test_count", "unknown")
    timestamp = data.get("full_cpu_test_timestamp_utc", "unknown")
    print(f"FULL LOCAL CI STATUS FROM PACKAGE: {status} ({count} checks, {timestamp})")
PY
echo "Bootstrap complete. No Llama weights were downloaded by bootstrap."
