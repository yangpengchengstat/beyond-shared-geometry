Spectral dimension sensitivity: compact, full-precision evidence supplement

This folder accompanies Appendix K. The primary dimension remains16. All data
are saved outputs of the completed fixed8/16 analysis, not a new analysis here.

Files:
  coverage.json: completed6000-unit coverage and equal-condition summaries.
  cell_summary.csv:96 dimension/condition entries.
  image_summary.csv:6000 canonical-semantic and permutation-image entries.
  registrations.json: complete image ranks, margins and spreads at both dimensions.
  paired_dimension_changes.csv:48 paired comparisons to original16-dimensional values.
  canonical_diagnostics.csv:96 canonical-condition checks and type corrections.
  registration_transitions.csv:48 paired registration-rank transitions.
  interpretation_summary.json: descriptive heterogeneity/type summaries.
  input_mapping.json: source archive/member/PCA and row identity mappings.
  SOURCE_BINDINGS.json: original result hashes, relative to the revision project.
  EVIDENCE_SHA256.json: hashes of the files distributed here.

To regenerate ONLY the manuscript table from these stored summaries:
  python3 generate_table.py --coverage coverage.json --output table_check.tex
The output must be a new file. Compare its bytes with
../../generated/spectral_dimension_summary.tex. This command performs no matrix
analysis, model execution, operator fitting or inference.

The main paper and tables compile without external result archives. Full numeric
re-execution requires the two checksum-identified original archives plus the
separately preserved offline_dimension_sensitivity_v1 project. That code uses
Python3.11/NumPy2.2.6 and gates8-dimensional computation on full16-dimensional
reference checks. Its source snapshot and run receipts are retained under
offline_dimension_execution_v1. Raw inputs and the complete per-unit numeric
artifacts remain in the project and are not duplicated in this compact Overleaf
source archive. No new dimension selection, significance decision,32-dimensional
claim, witness transfer or context-transport result follows from this supplement.
