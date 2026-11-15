# v0.6.1: targeted theory and presentation revision

Based on the actual v0.6.0 working project and its `paper/main.tex` entry point. The prior manuscript, packages, metadata and all completed scientific evidence are preserved. This revision adds analytical exposition and a plot of existing endpoints; it runs no numerical experiment, matrix analysis, model forward pass, fit, resampling, dimension search or C7 family-influence analysis.

## What changed

- **Closest literature (Section 2; Appendix H).** Added Wang et al. (2023), Elesedy and Zaidi (2021), Neural Fourier Transform (2024), and Ordonez-Apraez et al. (2026). The comparison distinguishes optimized predictors, learned reconstruction, conditional-operator approximation and fixed-observation exact-action risk. HAE is brought forward from the existing appendix. Standard spectral, representation-theoretic and Procrustes tools are credited; no priority certification is claimed. An approximation cost is not a general claim of worse finite-sample generalization.
- **Procrustes compatibility (Section 4.3; Proposition 8 in Appendix B.2).** The further group-law cost vanishes exactly when the elementwise orthogonal optimizer sets admit a common exact action. Nonsingular cross-moments give a unique polar family. With singular matrices, an arbitrary incompatible SVD completion is inconclusive. The proof uses compactness and nonnegative component excess risks; approximate numerical relations do not establish exact equality.
- **Analytical example (Section 4.3; Appendix B.3).** The existing A/B systems now explicitly separate independent-orthogonality cost and further-law cost. A has costs `(1/2, 0)`; B has `(3/2 - 2 a_7, 2 a_7)`, where `a_7` is the stated exact finite cosine sum. This is a hand-derived extension of an existing example, not a newly executed experiment or a new main-theorem claim.
- **Same-metric presentation (Section 5.3; Figure 2).** All 48 primary conditions are plotted from saved full-precision endpoints: independent linear optimum, independent orthogonal optimum, exact-class risk bounds and identity-family risk. The existing Table 2 is unchanged. The plot retains every condition and displays absolute risks on a log scale; additive costs are interpreted in the table and text. Training-metric witness transfer remains separate.

## Preserved evidence and scope

All previously completed r=8/16 analyses, 96 orthogonality units, registration comparisons, rank-audit qualifications, original numerical thresholds, normalization rules, and failure records remain intact. The C7 90.7% joint-bound and 99.44% further-law-bound concentration statements are preserved. They describe different lower-bound sums across model/readout conditions, not true-cost shares or context-family influence. LOFO remains prepared but unexecuted.

The added plot's `paper/supplementary/risk_presentation_v061/` directory includes exact copied CSVs, source hashes, input mapping, numeric strings for all plotted points and a portable plotting script. It uses no activation tensors and computes no new scientific endpoints. The same directory is included in the Overleaf and review-source packages.

## Review and delivery

The saved main document passed the desktop editor compiler and an offline cached TeX export. The complete manuscript has 76 pages; Section 7 and the Impact Statement end on page 16. Added arguments, citations, labels and changed layouts were checked, with a separate mathematical review of the compatibility statement and analytic decomposition. Remaining appendix spacing/font warnings are recorded rather than suppressed. Final author review remains pending.

Current records: `release/revision_validation_0_6_1.json`, `release/manuscript_revision_0_6_1.patch`, and `release/current_delivery_provenance.json`. Older reports retain their original dated scope. No submission, GitHub publication or cloud action is performed.
