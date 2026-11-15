Main-text presentation supplement, v0.5.8

The coverage file is copied byte for byte from the completed spectral_dimension_sensitivity_v1 results.
The formatter selects existing gamma means, joint-premium medians and unique-leading counts; it performs no averaging or matrix calculation.
Run from this directory:
python3 format_sensitivity.py --coverage spectral_dimension_coverage.json --output /tmp/spectral_sensitivity_main.tex
Compare that output with paper/generated/spectral_sensitivity_main.tex in the manuscript project. The output must not already exist.
The full original table and all original cell records remain unchanged. Input mapping records exact public source paths and hashes.

Conceptual figure (no data analysis):
python3 generate_concept_figure.py --output /tmp/evaluation_targets.pdf
This requires ReportLab. It draws the parallel evaluation targets without empirical inputs.
The witness_transfer_main table preserves followup_witness_bounded.tex verbatim except for one caption reference redirected to the new main core table; its source JSON records that exact change.
