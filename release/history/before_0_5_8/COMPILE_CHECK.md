# Manuscript compilation and presentation check — v0.5.7

Entry: `paper/main.tex`. Baseline: preserved 68-page v0.5.6 PDF.

- Built-in document compiler: success on the saved new main source; no editor tab was opened or replaced.
- Export compiler: existing Tectonic with `--only-cached --keep-logs --keep-intermediates`; successful bibliography and cross-reference passes. No resources downloaded.
- Final PDF: **71 pages**; main text ends on **page 14**.
- PDF SHA256: `78dbf89823d8bea52686f3a2658988f45434c621e517b9eb2958e437147e3ebc`.
- No unresolved references/citations, `??`, overfull boxes or missing-character warnings.
- Retained nonfatal diagnostics: **11 underfull horizontal-box warnings**, an OT1/lmss font substitution, and an `h` float preference changed to `ht`. These are disclosed rather than suppressed.

| Object | Number | Page |
|---|---:|---:|
| `tab:core-risk-summary` | 1 | 3 |
| `tab:followup-spectral` | 3 | 9 |
| `tab:followup-witness` | 4 | 10 |
| `tab:bounded-risk` | 5 | 10 |
| `tab:bounded-margins` | 6 | 11 |
| `tab:spectral-dimension-sensitivity` | 32 | 66 |
| `tab:main-findings` | 33 | 67 |
| `sp:main` | 7 | 26 |
| `sec:main-end` | 8 | 14 |
| `app:orthogonal-results` | M | 70 |

Tables 3–6, Table 32, Table 33 and Theorem 7 retain their numbers. Table 1 now has two panels; Appendix M documents the completed independent-orthogonal comparison.

Rendered pages 1–14, 26–28 and 70–71 were visually inspected. The main table, new result text, added metric row, theory reference and final appendix show no clipping or overlap. Unchanged appendix pages were not all visually re-reviewed; the entire PDF was scanned for unresolved placeholders.

The portable table formatter reproduced `core_risk_decomposition.tex` byte for byte from saved aggregate inputs. This formats existing endpoints only; no cell reanalysis, tests or experiments were run. Exact input paths and checksums accompany the full-precision supplement. Independent unpacked Overleaf compilation and source/archive closure checks are recorded separately under `release/` and the delivery's `audit/release_engineering/`. The PDF hash above identifies this final manuscript; a packaged recompilation is checked against it.

Actual commands, final compiler responses, logs, rendered pages, manuscript diff and machine-readable checks are retained under the delivery's `audit/`. Original v0.5.6 build records and historical failure logs remain preserved.
