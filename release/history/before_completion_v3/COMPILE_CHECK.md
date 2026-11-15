# Compilation and presentation check — v0.5.9

Entry: `paper/main.tex`. Baseline: v0.5.8 (71 pages; main text and Impact through page 13).

- Built-in editor compiler: final saved source compiled successfully; no tab was opened or replaced.
- Export: existing cached Tectonic with `--only-cached --keep-logs --keep-intermediates`. No font/model download.
- Final PDF: **71 pages**, main text and Impact through **page 13**.
- PDF SHA256: `c0557aee2a9ef14a49a4d8715cec515137a1826a95afe93552e2f9fd72ab1652`.
- No undefined reference/citation, duplicate source label, overfull box, missing-character warning or PDF `??` placeholder.
- Retained nonfatal diagnostics: 13 underfull horizontal boxes, the existing OT1/lmss font substitution and h-to-ht float preference change. Warnings were not suppressed.

| Object | Number | PDF page |
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

All main pages 1–13 were rendered for visual review. The revised page 2 was re-rendered after the final cost-bound wording clarification. No clipping, overlapping content, missing figures or unreadable revised table/caption was observed. References and appendices (pages 14–71) have pagewise text identical to the baseline; their source files and inherited assets are unchanged. This is not a new line-by-line scientific review of the appendices.

The caption-only derivative preserves all numeric tokens and the original table, with a hash receipt. Compilation, source/hash checks, rendering and packaging are editorial validation; no numerical fixture, matrix analysis or experimental job was run.

Source/package checks and independent unpacked Overleaf compilation are recorded under the delivery's `audit/release_engineering/` and the current `release/` metadata. No upload, publication or submission is performed by this build.

## Portable archive verification

The unpacked Overleaf archive compiled independently using cached Tectonic. It produced 71 pages with text identical on every page to the delivered manuscript, with the same 13 underfull warnings and known font substitution; no undefined reference, citation or overfull box appeared. The archived original caption source, derivative and JSON receipt pass their hash checks, and the two recorded prose substitutions reproduce the derivative exactly with every numeric token unchanged. This was a static presentation check, not a scientific calculation.

All three archives pass checksum, CRC, source-identity and dependency-closure checks. Current revision/style/preservation reports and the bounded wording diff resolve inside `review_notes/` in the Overleaf and anonymous-review packages; the GitHub package retains project-relative report links. The final package refresh changes report and provenance records only; its TeX compilation closure matches the independently compiled archive byte for byte.
