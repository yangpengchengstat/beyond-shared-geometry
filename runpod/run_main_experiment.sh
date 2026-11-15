#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "${mode}" in
  --smoke|--core|--pre-full|--full) ;;
  *) echo "usage: bash runpod/run_main_experiment.sh --smoke|--core|--pre-full|--full [main_experiment args]" >&2; exit 2 ;;
esac
shift
tier="${mode#--}"
tier="${tier//-/_}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${BSG_RUN_ID:-${tier}_${timestamp}}"
output_root="${BSG_OUTPUT_ROOT:-/workspace/run_outputs/main_experiment}"
cache_dir="${BSG_CACHE_DIR:-/workspace/run_outputs/activation_cache}"
temp_log="/workspace/${run_id}_console.log"

python3 -m bsg.experiments.main_experiment "${mode}" \
  --config configs/main_experiment.yaml \
  --run-id "${run_id}" \
  --output-root "${output_root}" \
  --cache-dir "${cache_dir}" "$@" 2>&1 | tee "${temp_log}"

run_dir="${output_root}/${run_id}"
mv "${temp_log}" "${run_dir}/logs/console.log"
bash runpod/package_results.sh "${run_dir}"
echo "MAIN EXPERIMENT ${tier^^}: PASS"
