Orthogonal-comparator table supplement

The numerical study and 8 software tests completed previously; this manuscript revision did not run them again.
Run python3 generate_table.py --inputs . --output /tmp/core-decomposition-check.tex from this folder to format the existing endpoints.
All inputs use full saved precision. The program selects/rounds saved means, not a new fit or aggregation.
Panel A reuses the original r16 joint bounds. Panel B uses coverage.json saved means after clipping each condition lower endpoint.
Inputs/data identity are in INPUT_MAPPING.json, input_binding.json and EVIDENCE_SHA256.json.
All 96 cell rows are included in cell_summary.csv; domain_diagnostics.json contains previous medians/ranges/concentration and threshold proximity.
Theoretical costs decompose exactly, but clipped lower bounds need not add. These are finite-bank bounds, not confidence intervals.
Complete executable study source is in the separately supplied public project; large input archives and diagnostics remain in the preserved research/result archives.
