# Frozen results of the completed template-centering diagnosis

`rewhitened/` contains the original-versus-centered/rewhitened comparison:
96 cell rows and 5,904 cell/image rows. `three_way/` adds centering while retaining
the original metric: 144 cell rows and 8,856 cell/image rows.

The CSVs, summaries and `archive_record.json` receipts retain their original
2026-11-10 bytes and SHA-256 identities. Receipts additionally identify the
original scripts and execution logs; the script bytes are in
`studies/template_centering_v1/historical_sources/`. Execution logs are not
needed by this publication adapter and are not included here. Relative historical
`work/` locators in receipts are provenance, not runtime dependencies.

No numbers, rank tolerances, cases or metric definitions were changed for export.
No diagnostic was rerun during packaging. The public runner and input mapping
are described in `studies/template_centering_v1/README.md`.
