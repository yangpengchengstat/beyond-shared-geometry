#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${1:?Pass the original repository root}"
OUT="${2:?Pass a NEW output directory}"
PY="${AUDIT_PYTHON:-python3}"
if [[ -e "$OUT" ]]; then
  printf '%s\n' 'Refusing to overwrite an existing output directory.' >&2
  exit 2
fi
"$PY" -c 'import sys; from pathlib import Path; r=Path(sys.argv[1]).resolve(); o=Path(sys.argv[2]).resolve(); assert o!=r and r not in o.parents, "Use an output outside the evidence repository"' "$REPO" "$OUT"
mkdir -p "$OUT/logs" "$OUT/provenance" "$OUT/components/history" "$OUT/components/lexical" "$OUT/components/sensitivity" "$OUT/scripts"
cp "$BASE/audit_protocol.json" "$OUT/audit_protocol.json"
cp "$BASE/components/lexical/protocol.json" "$OUT/components/lexical/protocol.json"
cp "$BASE/components/lexical/full_input_protocol.json" "$OUT/components/lexical/full_input_protocol.json"
cp -R "$BASE/paper" "$OUT/paper"
cp "$BASE/scripts/build_audit_tables.py" "$OUT/scripts/build_audit_tables.py"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$PY" "$BASE/scripts/audit_saved_matrices.py" --repo "$REPO" --output "$OUT" > "$OUT/logs/matrix_audit.log" 2>&1
"$PY" "$BASE/scripts/enrich_rank_records.py" --repo "$REPO" --output "$OUT" > "$OUT/logs/enrich_rank_records.log" 2>&1
"$PY" "$BASE/components/history/audit_dimension_history.py" --repo "$REPO" --out "$OUT/components/history" > "$OUT/components/history/execution.log" 2>&1
"$PY" "$BASE/components/history/verify_coordinate_bindings.py" --repo "$REPO" --out "$OUT/components/history" > "$OUT/components/history/coordinate_binding_execution.log" 2>&1
"$PY" "$BASE/components/lexical/prepare_lexical_input.py" --repo "$REPO" --output "$OUT/_input_cache/historical_full_activation_archive.zip" > "$OUT/components/lexical/prepare.log" 2>&1
"$PY" "$BASE/components/lexical/audit_lexical_rank.py" --repo "$REPO" --output "$OUT/components/lexical" --internal-archive "$OUT/_input_cache/historical_full_activation_archive.zip" > "$OUT/components/lexical/execution.log" 2>&1
"$PY" "$BASE/components/lexical/audit_original_full_inputs.py" --archive "$OUT/_input_cache/historical_full_activation_archive.zip" --output "$OUT/components/lexical" > "$OUT/components/lexical/full_input_execution.log" 2>&1
"$PY" "$BASE/components/sensitivity/audit_existing_sensitivity.py" --repo "$REPO" --output "$OUT/components/sensitivity/output" > "$OUT/components/sensitivity/execution.log" 2>&1
"$PY" "$OUT/scripts/build_audit_tables.py" > "$OUT/logs/build_audit_tables.log" 2>&1
printf '%s\n' "Read-only audit complete: $OUT"
