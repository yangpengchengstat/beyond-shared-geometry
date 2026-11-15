# Current compilation

Delivery **20261113-v6**; active entry `paper/main.tex`. The matching 105-page `paper/main.pdf` was built in a fresh output directory with Tectonic 0.17.0 (XeTeX + BibTeX). Official TMLR style files were not modified.

`release/current_compile.json` binds the PDF to its active source closure. `release/PDF_QA_V6.json` records page and cross-reference checks. Complete logs are in the owner delivery. Historical build reports describe their own revisions.

The abstract reflects the centering diagnostics and evaluation distinctions. Manual title breaks remove title-word hyphenation. A float barrier before Appendix D prevents its tables from interrupting C's final paragraph.

## Recompile

With Tectonic installed, create a fresh output directory and run from `paper/`:

```sh
tectonic --untrusted --keep-logs --keep-intermediates --outdir /path/to/fresh-build main.tex
```

A first build may fetch TeX resources. The delivery used an existing compiler without installing a new compiler. The source archive was also extracted and compiled independently.

For Overleaf, import `overleaf_source.zip`, select `main.tex` and XeLaTeX to match the local engine family, then recompile from scratch. No Overleaf server or pdfLaTeX-specific build was run here. The official template supports pdfLaTeX as an alternative.

Default verification and CI do not compile or execute scientific workers. A successful build is not independent replication of all measurements.
