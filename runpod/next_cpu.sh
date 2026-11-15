#!/usr/bin/env bash
# CPU calibration only. Never invokes a model stage or provisions a cloud pod.
set +x
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash runpod/next_cpu.sh SAME_RUN_DIRECTORY [CPU run options]" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$script_dir/next_run.sh" cpu "$@"
