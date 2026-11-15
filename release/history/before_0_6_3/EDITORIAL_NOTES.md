# v0.6.2: integrate the completed C7 family-influence analysis

This independent revision starts from v0.6.1 and its actual `paper/main.tex` entry point. It incorporates the previously completed `bsg_c7_family_influence_20261107_v1` records. No numerical test, scientific analysis, model forward pass, fit, dimension search or resampling was run for this editorial revision. Table formatting uses the saved full-precision scalar results only.

## Manuscript changes

- **Section 5.3, page 11:** a short paragraph reports all 1,536 complete-family deletions across the 24 C7, r=16 conditions. The same 23 conditions remain above the original tolerance after every deletion. The dominant condition, Llama-3.1-8B/layer 13, retains a further-law lower bound of 0.02435–0.02696 and supplies 99.38%–99.48% of the lower-endpoint sum. The original cross-condition heterogeneity remains explicit.
- **Appendix M.3, pages 75–76, Table 33:** all 24 conditions are listed with their complete-bank lower endpoint, minimum and maximum over the 64 deletions, and count exceeding the original tolerance. The appendix states the paired-state removal rule, numerical validation recorded in the completed run, baseline reproduction, input conditioning, and limits of interpretation. A zero lower endpoint is inconclusive about a strictly positive true cost.
- **Discussion and scope references:** one sentence summarizes the limited stability finding. The separately prepared all-image registration LOFO remains unexecuted; the completed canonical-action risk analysis did not rerank alternative registrations.

## Interpretation retained

Each reduced bank uses its own uncentered second-moment whitening while the original PCA, training mean, dimension, state correspondence and thresholds remain fixed. Deleting a family therefore changes the empirical distribution and metric, including the orthogonal-action class in original coordinates. The observed changes are not additive family contributions under a fixed loss. The dominant share concerns the sum of lower bounds, not the unknown true costs. This checks individual-family deletions; it does not resolve joint deletions, shared construction bias, population generalization, or native computation. The new authored-context transfer study remains a separate completed result.

## Reproducibility and preservation

The public project includes the unchanged executed source in `studies/c7_family_influence_v1/` and all saved per-deletion diagnostics in `results/c7_family_influence_v1/`. Their input/source hashes and execution/test receipts retain the original run identity. `paper/supplementary/c7_family_influence_v1/` contains full-precision scalar tables, input mapping and the table formatter for the Overleaf and review-source packages. Only a private executable path is omitted from the portable runtime record, with the original hash and transformation recorded explicitly.

The v0.6.1 source and packages, original family-analysis archive, prior scientific results and failure records are preserved. The 16-dimensional main results and completed 8/16 analyses are unchanged. Older dated reports are retained as historical records; current status is in `release/current_delivery_provenance.json` and `release/revision_validation_0_6_2.json`.

## Validation

The saved document passed the desktop compiler and offline cached TeX export. The full manuscript has 77 pages; the main text ends on page 16. The 21-page editorial reading copy contains a cover plus manuscript pages 1–20, including references. Changed layouts and the 24-row table were visually checked. All 72 displayed endpoints and tolerance counts were independently checked against saved results. Existing appendix spacing/font warnings remain recorded; no undefined references, missing citations or overfull boxes were introduced. Release checks and package compilation are recorded alongside the actual manuscript patch.

Final author review is still required. No submission, GitHub publication or cloud action is performed.

Portable package verification passed: an offline build of the unpacked Overleaf source reproduces all 77 pages of manuscript text, and its saved-summary formatter reproduces the 24-row family table and receipt byte-for-byte. The public source passes frozen-evidence, dependency, anonymity/privacy and artifact-integrity checks. These are publication checks; no numerical tests or analysis were rerun.
