#!/usr/bin/env bash
# Install a fresh Linux/Python 3.11 GPU environment; never download model weights.
set +x
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
venv="${BSG_REPRO_VENV:-/workspace/bsg-reproduction-venv}"
base_python="${BSG_BASE_PYTHON:-python3.11}"
replay_lock="${1:-}"
if [[ $# -gt 1 ]]; then echo 'Usage: bash bootstrap.sh [previous-environment.requirements.lock]' >&2; exit 2; fi
if [[ -L "$venv" ]]; then echo 'Virtual environment must not be a symlink.' >&2; exit 2; fi
"$base_python" - <<'PY'
import platform, sys
if sys.version_info[:2] != (3, 11) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('Requires Linux x86_64 with Python 3.11. Set BSG_BASE_PYTHON to the appropriate executable.')
PY
if [[ ! -x "$venv/bin/python" ]]; then "$base_python" -m venv "$venv"; fi
py="$venv/bin/python"
"$py" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use a fresh Python 3.11 environment"'
provenance="$venv/reproduction-provenance"
mkdir -p "$provenance"
lock="$provenance/environment.requirements.lock"
if [[ ! -f "$lock" ]]; then
  "$py" - <<'PY_CHECK'
import importlib.metadata as m
unexpected = {d.metadata['Name'].lower().replace('_','-') for d in m.distributions()} - {'pip','setuptools','wheel'}
if unexpected:
    raise SystemExit('Use an empty fresh BSG_REPRO_VENV; unsealed installed packages were found.')
PY_CHECK
fi
if [[ -n "$replay_lock" ]]; then
  "$py" - "$replay_lock" "$lock" <<'PY'
import pathlib, sys
source, target = map(pathlib.Path, sys.argv[1:])
raw = source.read_bytes()
if target.exists() and target.read_bytes() != raw:
    raise SystemExit('Existing environment has a different lock; use a fresh BSG_REPRO_VENV.')
if not target.exists(): target.write_bytes(raw)
PY
elif [[ ! -f "$lock" ]]; then
  "$py" -m pip install --dry-run --ignore-installed --only-binary=:all: \
    --report "$provenance/resolution-report.json" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r "$here/requirements.in"
  "$py" "$here/environment.py" --pip-report "$provenance/resolution-report.json" --lock-output "$lock"
fi
"$py" -m pip install --only-binary=:all: --require-hashes -r "$lock" \
  --report "$provenance/installation-report.json"
"$py" -m pip check
"$py" -m pip freeze --all > "$provenance/environment.freeze.txt"
env PYTHONHASHSEED=0 "$py" "$here/environment.py" --output "$provenance" --hardware \
  --minimum-free-gib 5 --minimum-ram-gib 1 --minimum-vram-gib 44
printf '\nRuntime ready: %s\nNo model weights were downloaded. Retain reproduction-provenance for exact wheel replay.\n' "$py"
