#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo 'Usage: bash run_cached.sh ORIGINAL_CHARACTERIZATION_RESULTS.zip NEW_OUTPUT_DIRECTORY [WORKERS]'
  exit 2
fi
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
archive_path="$1"
output_path="$2"
workers="${3:-2}"
python_cmd="${BSG_CPU_PYTHON:-python3}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# No timeout wrapper, GPU imports, old calibration or old-run continuation.
"$python_cmd" "$package_dir/run.py" --mode pilot --archive "$archive_path" --output "${output_path}.pilot" --workers "$workers"
"$python_cmd" "$package_dir/run.py" --mode full --archive "$archive_path" --output "$output_path" --workers "$workers"
