#!/usr/bin/env bash
# Run one chosen model phase. Does not automatically start CPU calibration.
set +x
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash runpod/next_gpu.sh PHASE SAME_RUN_DIRECTORY --execute-models [run options]" >&2
  echo "PHASE: pilot, confirm, peers, crossmodel, extensions" >&2
  exit 2
fi
case "$1" in
  pilot|confirm|peers|crossmodel|extensions) ;;
  *) echo "Choose one GPU phase: pilot, confirm, peers, crossmodel, extensions. CPU calibration and combined full runs are not accepted here." >&2; exit 2 ;;
esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$script_dir/next_run.sh" "$@"
