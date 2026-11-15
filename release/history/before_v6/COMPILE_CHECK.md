# Current manuscript compilation status

The only active entry is `paper/main.tex`. Read `release/current_release_status.json`
for this delivery's actual compile and visual-review status. A fresh exported PDF,
if supplied at `paper/main.pdf`, must be bound to the current source hashes and PDF
hash in `release/current_compile.json`; the static verifier rejects an unbound PDF.

No successful build of a historical version is evidence that this version compiled.
The former v0.5.9 report is preserved unchanged at
`release/history/before_completion_v3/COMPILE_CHECK.md`. All other historical build
reports retain their original revision scope.

The v5 PDF was built using Tectonic 0.17.0 (XeTeX + BibTeX). Its source and PDF
hashes are in `release/current_compile.json`. The complete owner project
contains the build log and layout check in `verification/v5/`. The owner PDF
is delivered separately.

For an author-run Overleaf build, upload the current `overleaf_source.zip`, select
`main.tex` with pdfLaTeX, and choose **Recompile from scratch**. Keep the downloaded
PDF, complete log and bibliography build output. Inspect all body pages, changed
appendices and references for missing glyphs, clipped figures/tables, unresolved or
duplicated references, anonymity and readability.

If pdfLaTeX and BibTeX are already installed, use a fresh scratch copy of `paper/`:

```sh
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

These are manual compilation instructions, not installation commands. The default
Makefile and CI do not compile, download packages, or run scientific analysis.
