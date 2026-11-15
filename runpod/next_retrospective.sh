#!/usr/bin/env bash
# Explicit CPU reanalysis of historical archives; never part of default calibration.
set +x
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash runpod/next_retrospective.sh SAME_RUN_DIRECTORY --archive inputs/full.zip" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$script_dir/next_run.sh" retrospective "$@"
