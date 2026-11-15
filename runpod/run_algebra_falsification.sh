#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "${mode}" in
  --smoke|--pre-full|--full) shift ;;
  *) echo "usage: bash runpod/run_algebra_falsification.sh --smoke|--pre-full|--full [AFS args]" >&2; exit 2 ;;
esac

run_id="${BSG_AFS_RUN_ID:-afs_${mode#--}_$(date -u +%Y%m%dT%H%M%SZ)}"
output_root="${BSG_AFS_OUTPUT_ROOT:-/workspace/run_outputs/algebra_falsification}"
historical_root="${BSG_MAIN_FULL_003_ROOT:-/workspace/run_outputs/main_experiment/main_full_003}"

python3 -m bsg.experiments.algebra_falsification "${mode}" \
  --config configs/algebra_falsification.yaml \
  --historical-root "${historical_root}" \
  --output-root "${output_root}" \
  --run-id "${run_id}" \
  "$@"

bash runpod/package_afs_results.sh "${output_root}/${run_id}"
