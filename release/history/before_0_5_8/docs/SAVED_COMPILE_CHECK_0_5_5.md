# Compile and presentation check

Current update: integration of completed8/16-dimensional spectral sensitivity,2026-11-06UTC.

- Entry remains `paper/main.tex`; the current editor stays open on that source.
- Built-in desktop LaTeX compiler: **success**, recorded in `logs/builtin_compile.spectral_dimension.json`.
- The same source was exported in place with existing Tectonic0.17.0, `--only-cached --keep-logs --keep-intermediates`, exit0. No alternate manuscript or separate PDF was compiled.
- PDF: **65 pages**, with new **AppendixK and Table32**. The prior64-page PDF and its sources are preserved in `revision_history/before_spectral_dimension_integration_20261106/`.
- All previously defined label numbers remain unchanged, including Tables3/4/5/6, AppendicesI.7/I.8, AppendixJ and Table31. Pagination may change.
- No LaTeX errors, undefined references/citations, multiply defined labels or overfull boxes occur. Existing underfull-spacing and font-substitution warnings remain; none is a new overflow in the added material.
- Rendered and visually inspected PDF pages1,2,3,11,12,14,30,54,62,63,64,65. The new appendix/table, changed paragraphs and existing table captions are legible without clipping or overlap. Page65 was also inspected at full render resolution.
- The new table is reproduced byte-for-byte from packaged full-precision `coverage.json` using `generate_table.py`; this is formatting only, not a numerical rerun.
- The updated Overleaf source bundle includes all paper dependencies plus full-precision new summaries, input mapping, hashes and the table generator. Its members are checked against saved sources. It excludes compiled main.pdf, runtime environments, private absolute paths and large raw arrays; those inputs/evidence remain preserved in the project.
- Original data,16-dimensional results, source code, analysis outputs, failure records and existing numeric table bodies remain unchanged. Details: `logs/spectral_dimension_integration_preservation.json`.

The previous compilation/audit records remain in revision_history and their original logs. This editorial integration performed no new matrix diagnostics, model forwards, spectral calculations or experiments. LOFO remains prepared and unexecuted.
