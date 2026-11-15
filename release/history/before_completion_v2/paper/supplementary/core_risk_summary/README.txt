Core same-metric risk summary (Table 1)

This is a formatting-only extraction of previously reviewed full-precision domain summaries, not a new matrix analysis or new aggregation. All48originalconditions remain included.

From this directory run:
  python3 generate_table.py --source domain_risk_reference.csv --output /tmp/core-risk-table-check.tex
Compare its bytes with ../../generated/core_risk_summary.tex. The source CSV is byte-identical to results/main30_bounded_v1/domain_risk_reference.csv in the public code project.

Mean bounds are deterministic bounds on the equal-cell mean optimum/premium under each cell-specific test-bank whitening. They are not confidence limits, measured minima, frozen-predictor transfer losses, or a decomposition of orthogonality versus group-law costs. Median/concentration statements repeat the saved Section5.1 summary; no newstatistics arecomputed.
