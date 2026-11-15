## Current revision — 24 October 2026

Version 0.4.0 integrates the completed returned replay as reviewed descriptive evidence. Compact/context and secondary extension results are now available. Native competence still failed, structural power remains limited, and independent confirmation is blocked. Original frozen results and source capsules retain their identities.

For current instructions use the root README, `docs/REPRODUCTION_REVIEW_20261024.txt`, and `release/reproduction_review.json`. Rebuild tables offline with `python scripts/build_reproduction_assets.py`; no new calibration or full model replay is needed. Maintenance v2 defaults to checking only and requires a separate explicit choice before any execution. Prior dated instructions below describe earlier project states and should not be used to launch another run automatically.

## Preserved earlier documentation

# Getting started

**Current status (21 October 2026):** Calibration and excluded development analyses have completed; structural power is limited and the original native branch stopped at clean competence. Continue with [independent characterization v1](CHARACTERIZATION_WORKFLOW.md). The complete return has passed compact-data integrity, analytic-coordinate and interval audits, with raw/PCA and neural-optimizer limitations explicit. Reviewed descriptive results are included in the manuscript. Read [the current re-audit](PROJECT_REAUDIT_2026_10_21_V2.md) before planning another experiment. The earlier protocol below is retained as scope/history, not an instruction to rerun the full pipeline.


This standalone publication project brings together the research source, configurations, selected frozen evidence, reproducible figures and tables, an English manuscript, and an anonymous Overleaf bundle. The original project's historical files, frozen outputs, and published repositories remain separate and unchanged.

## Files to read first

- `paper/main.pdf`: the compiled manuscript, including the main text, references, and full appendix.
- `paper/main.tex` and `paper/sections/`: editable manuscript sources.
- `dist/2026-10-21-project-reaudit-v2/overleaf.zip`: upload to Overleaf, select `main.tex`, and compile with pdfLaTeX.
- `dist/2026-10-21-project-reaudit-v2/github-source.zip`: the complete source and selected evidence prepared for a future public release, retaining the original copyright attribution.
- `results/registry.json`: original identities, hashes, and selection/aggregation records for the included evidence.
- `release/pdf_qa.json`: the compiled PDF's binding to its source files, page-count assessment, and automated checks.

Paths above are relative to the repository root. The anonymous manuscript and the future public source archive have different audiences. The manuscript bundle excludes author identities, identifying repository links, and private filesystem paths. The public code retains the MIT copyright statement. A local build does not publish a GitHub repository or submit a paper to a conference.

## Scientific scope

The study examines approximate finite-group actions on fixed weekday and corner labels under held-out template variation. Independent operator recovery and generator transfer are positive findings. Stable category coding, the dependence of composition error on prediction error, finite-orbit realizability at lexical rank, and the stronger additive behavioral comparisons delimit those findings. They do not establish that the language model internally executes a group algorithm.

The main text presents the principal methods and evidence. The appendix records metric definitions, fitting losses, optimization settings, activation/PCA/intervention procedures, statistical design, sensitivity coverage, AFS definitions, and proofs. New analytical discussions are not presented as newly executed model experiments. Consult [Scientific scope](SCIENTIFIC_SCOPE.md) for the remaining identification questions.

## Check, rebuild, and package

Create a separate Python environment as described in the root [README](../README.md). Then:

```bash
make check
make assets
make paper
python scripts/check_pdf.py
make release
```

`make check` checks numerical-source integrity, CPU contracts, frozen evidence, and publication dependencies. `make assets` rebuilds figures and tables from the bundled evidence. `make paper` requires TeX Live with latexmk or Tectonic. Creating the PDF QA record with `scripts/check_pdf.py` additionally requires the optional paper environment, installed with `python -m pip install -e '.[paper]'`. `make release` updates the two ZIPs and their checksum files. See [Reproducibility](REPRODUCIBILITY.md) for the full environment and command distinctions.

After editing the manuscript or its assets, compile again, run PDF QA, and package again. The saved QA record rejects a stale PDF paired with changed source files. Visually inspect tables, equations, legends, citations, and pagination as well; automated checks do not replace reading the rendered document.

GPU experiments are a separate optional workflow requiring model access and suitable hardware. CPU checks do not download weights or launch cloud resources. The public archive omits large activation tensors and some detailed raw outputs, while recording their identities and the scope of what is included. Bundled evidence can rebuild the paper's assets; that is distinct from reproducing model forward passes.

## Submission status

The bundle uses the unmodified official ICML 2026 style, including its fonts, layout, and bibliography style. The official ICML 2027 template and policies remain unverified. This project therefore does not claim confirmed 2027 submission compliance. When those rules become available, update the template, compile again, and check anonymity, page limits, supplementary-material rules, and submission metadata.

Scientific conclusions and engineering checks remain separate. A successful build, a frozen PASS, or FULL's historical GO is not an acceptance guarantee. Additional experiments should receive new run identities, configurations, and output directories while preserving the current frozen evidence.
