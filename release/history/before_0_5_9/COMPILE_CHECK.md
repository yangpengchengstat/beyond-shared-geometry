# Compilation and presentation check — v0.5.8

Entry: paper/main.tex. Baseline: v0.5.7 (71 pages; main end page 14).

- Built-in compiler: success on the final saved sources; no editor tab was opened or replaced.
- Export: existing Tectonic, --only-cached --keep-logs --keep-intermediates. Final bibliography and reference passes succeeded without downloads.
- Final PDF: **71 pages**, main text and Impact Statement end on **page 13**.
- PDF SHA256: `a89e64ec995ea64b92a17a8ba63fcaabccaf300e99d0285092766c449922889d`.
- No undefined cross-references/citations, rerun requests, multiply-defined references, overfull boxes, missing-character warnings or PDF `??` placeholders.
- Nonfatal retained diagnostics: **13 underfull horizontal boxes**, the existing OT1/lmss font substitution, and an h-to-ht float preference change. They were not hidden through warning suppression.

| Object | Number | Page |
|---|---:|---:|
| `tab:metric-contract` | 1 | 5 |
| `tab:core-risk-summary` | 2 | 9 |
| `tab:followup-witness` | 3 | 10 |
| `tab:bounded-margins` | 4 | 11 |
| `tab:sensitivity-main` | 5 | 11 |
| `tab:followup-context` | 6 | 12 |
| `fig:evaluation-targets` | 1 | 2 |
| `fig:context-followup` | 2 | 12 |
| `sp:main` | 7 | 25 |
| `app:orthogonal-results` | M | 69 |
| `app:retained-summaries` | N | 70 |
| `sec:main-end` | 7 | 13 |

Table numbers change automatically with the new organization; preserved data are tracked by stable labels and source hashes. Theorem 7 remains Theorem 7. Old standalone spectral/premium summaries remain in Appendix N instead of duplicating the main core table.

Main pages 1–13 and Appendix N pages 70–71 were rendered and inspected individually; final page 10 was rechecked after its caption reference was redirected to Table 2. All 71 pages were also rendered for a contact-sheet layout overview. No clipping, overlapping content, unintended blank pages or unreadable new figure/table was observed. The overview does not claim a fresh line-by-line scientific audit of unchanged appendices.

The core table, compact sensitivity table and conceptual vector diagram reproduce byte for byte from their saved presentation inputs/scripts. No experiment, matrix calculation, numerical fixture or resampling was executed. Existing study logs are retained prior evidence.

An initial export failed because the local cached font bundle lacked msbm7 for a new blackboard-bold real-matrix domain in an infimum subscript. The domain was moved to the preceding prose with identical mathematical meaning; final built-in/export compilation succeeded. The failure log is retained in the private audit. No font or model download was used.

Final package/source-closure and independent unpacked Overleaf compilation results are recorded in release/ and the delivery audit/release_engineering/. Exact commands, text diff, build responses, logs and rendering records are retained in the audit. This is a local build and package verification, not an upload or submission.

## Portable package verification

The exported Overleaf source was independently unpacked and compiled using the
existing offline TeX cache. It produced 71 pages with pagewise text identical to
the verified manuscript (main text through page 13). There were no undefined
references or citations, missing glyphs, or overfull boxes; retained underfull
and font/float substitution warnings are recorded in the compilation logs.
The unpacked public source passed its integrity and publication-closure checks.
All portable metadata report references resolve to included review notes.
These are packaging and compilation checks, not numerical tests or experimental
re-execution. Final package checksums accompany the ZIP files.
