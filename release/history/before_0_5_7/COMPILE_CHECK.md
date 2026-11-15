# Final manuscript compilation and presentation check

Revision: 0.5.6, independent copy of the 65-page 0.5.5 manuscript. Entry: `paper/main.tex`.

- Built-in document compiler: **success** on the saved main source, without opening or replacing the user's editor.
- Export compiler: existing Tectonic with `--only-cached --keep-logs --keep-intermediates`; **success**, bibliography and cross-reference passes complete.
- Revised PDF: **68 pages**, main text through page **13** (baseline main text through page 15). Preserved historical controls and added derivations increase the complete manuscript length.
- SHA256: `a75432cab48957ee4d02aa95b6ab3c18349c3a3f78feb70cc6596c12cb79f4a4`.
- No undefined references/citations, unresolved `??`, overfull boxes or missing-character warnings were found in the final log/PDF text.
- The final log retains 8 underfull horizontal-box warnings in existing appendix/table text and an `OT1/lmss/m/n` font substitution to Computer Modern Roman. These are formatting warnings, not clean-log claims; no global warning suppression or spacing patch was applied.

## Numbering and placement

| Object | Final number / page |
|---|---|
| Same-bank core risk summary | Table 1 / page 2 |
| Existing spectral / witness / premium / margin tables | Tables 3/4/5/6 / pages 9/9/10/10 |
| Main risk theorem | Theorem 7 / page 25 |
| Positivity consequence in main text | Equation (6) / page 7 |
| Positivity / independent-orthogonal / invariant-energy derivations | Equations (17)/(18)/(19) / pages 26/26/28 |
| Existing 8/16 sensitivity | Table 32 / page 65 |
| Preserved original core findings | Table 33 / page 66, Appendix L |
| Original budget frontier | Figure 7 / page 68, Appendix L |

The old core-table and budget-figure numbering change because their full contents moved to the appendix. Their labels continue to resolve. Existing Tables 3–6, Table 32, and Theorem 7 keep their numbers.

## Visual and source checks

Rendered pages 1–13, 25–29 and 65–68 were visually inspected, covering all main text, the new core table, new mathematical derivations and moved appendix material. No clipping or overlap was found. The entire PDF was also scanned for reference/overflow warnings and unresolved placeholders. This is not a claim that every unchanged appendix page received a new visual review.

The portable core-table generator was run against the copied saved aggregate CSV and reproduced the final TeX byte for byte. It only formats existing rows; no scientific analysis was repeated. The original tables, figures and endpoint records remain unchanged. New independent-orthogonal code received syntax/help/source-hash checks and independent static mathematical review; its numerical tests and data analysis were not run.

## Retained operational records

The independent delivery's `audit/` contains the built-in compiler response, final Tectonic log, PDF-reference and final-check JSON, source diff, render snapshots and actual command record. A first export attempt encountered an uncached font requested by an apostrophe macro in the new caption. That macro was replaced by an ordinary apostrophe; the failed log is retained, and no compiler resources were downloaded. `PRESERVATION_REPORT.md` identifies the separate frozen-input checks.
