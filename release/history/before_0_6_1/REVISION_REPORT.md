# Limited pre-submission wording revision — v0.5.9

Date: 2026-11-07. Baseline: the actual 71-page v0.5.8 project, whose main text and Impact Statement end on page 13. Work was performed in an independent copy. The open v0.5.8 editor document and all original files were left in place.

## What changed

1. **Separate the evidence questions.** Abstract and Introduction now distinguish exhaustive registration comparisons plus retained-dimension sensitivity from the separate new-context transport comparison. Section 5.6 explicitly says the new bank does not reevaluate spectral risk or registration rankings. The third contribution no longer groups registration, dimension and context sensitivity into one potentially misleading claim.
2. **Name the measured object.** “Semantic rankings” becomes spectral-budget rankings of the prescribed registration. The Introduction describes variation of cost bounds rather than measured unknown exact costs. “Rank-16 panel” becomes “dimension-16 panel”; actual mathematical rank terminology is retained.
3. **Make first-use terms readable.** PCA, ordinary least squares (OLS), and random Fourier features (RFF) are expanded; the constructed-action witness is introduced directly. No method or estimator changes.
4. **Trim local repetition.** Three repeated caveat sentences in Section 5.3 were removed where the theory and adjacent table already supply the same distinction. The C7 magnitude/concentration explanation and wide-upper-bound limitation remain next to the numbers. The core table has a neutral heading and a shorter caption, using a new prose-only derivative; the original generated table is untouched.
5. **Use examples for writing, not scientific scoring.** All 18 supplied PDFs (17 distinct manuscripts) were examined for section structure, introduction/result functions, captions and appendix allocation. STYLE_AND_STRUCTURE_NOTES.md ties specific examples to these bounded edits. It also corrects the earlier mistaken statement that MoEFL was absent: it is the Global Motif Embedding paper. The old note remains in version history.

The v0.5.8 seven-section organization, concept figure, all results, tables, theorem statements, proofs, bibliography and appendices remain. This is a wording pass, not another restructuring or a claim of improved experimental evidence.

## Evidence kept explicit

- r=16 is primary and fixed; r=8 is a separate retained-coordinate comparison with its own whitening and normalization. Neither chooses an optimal dimension.
- C7 joint-premium lower endpoints have a two-condition 90.7% concentration; further-law lower endpoints have a one-condition 99.44% concentration. These refer to computed bounds, not the distribution of unknown true costs.
- All 96 orthogonal units, nine zero further-law endpoints and the r=8 near-tolerance case remain.
- Constructed witnesses' held-out ordering is empirical and uses the frozen training metric. Registration rankings are budget rankings, not exact-risk rankings, p-values or semantic probabilities.
- The new-context difference-fitting advantage reverses across domains under the recorded selection contracts. It is a transport result, not independent contextual validation of spectral or registration results.
- Native failures, infeasible historical settings and blocked confirmations remain. LOFO and C7 family influence remain unexecuted.

## Main-text independence

| Reading question | Main-text answer | What the reader can establish |
|---|---|---|
| State, context and shared operator | Section 3.1–3.2; eq:projection, eq:prediction, eq:convention | The state is visible in a prescribed domain; one family acts across contexts within fixed coordinates. |
| Training and held-out units | Sections 3.1–3.2 and 5.1; sec:setup, sec:design | Training-only mean/PCA and operator fitting; complete-template splits, paired-family banks and reused partitions are explicit. |
| eta, gamma and exact-class risk | Sections 3.3 and 4.2; eq:class-loss-main, eq:orbit-gram-main, eq:budget-main | Three optima, the observable orbit Gram matrix and invariant projection budget are directly defined. |
| Risk bound and assumptions | Section 4.3; eq:main-risk-bound | Fixed real dimension, exact O(r) row action, full-group averaging and positive-definite second moment are specified; the guarantee is on a class optimum. |
| Why witness loss cannot be mixed with bank bounds | Sections 3.3, 4.3 and 5.4; sec:witness-transfer | Transfer retains training whitening; bank bounds use their own metric; conjugacy need not preserve Euclidean orthogonality. |
| Risk scale and heterogeneity | Section 5.3 and Table 2; sec:observed-risk | Actual intervals, medians, ranges, the two distinct C7 concentration percentages, zero endpoints and near-tolerance case remain. |
| Registration and dimension dependence | Section 5.5, Tables 4–5; sec:registration-sensitivity | All image subgroups, target changes, signed budget margins, unique-leading counts and dimension-specific metrics are given. |
| Meaning and limits of authored contexts | Section 5.6, Table 6 and Figure 2; sec:new-context | A separate transport comparison with matched validation selection; domain reversal, identity comparator, authored construction and old-selection distinction remain. |
| Increment over related work | Sections 1–2; sec:related-main | The class-risk bound in fixed coordinates and its application are separated from established tools, learned-coordinate identification and native-mechanism evidence. |

## Verification

The revised PDF remains 71 pages, with main text and Impact Statement through page 13. Main Tables 1–6, Figures 1–2 and Theorem 7 keep their numbers and pages. References and appendices (pages 14–71) have pagewise text identical to v0.5.8. Both built-in and cached export compilation succeeded. No undefined reference/citation, overfull box or missing PDF placeholder remains; retained nonfatal warnings are recorded in COMPILE_CHECK.md.

645 inherited scientific/evidence/presentation files are byte-identical in the new copy. All 1,559 baseline publication files remain unchanged at their original location. Numeric literals in all five edited main-text files are conserved. The current APPENDIX_DEPENDENCY_AUDIT.csv records 25 reference-bearing blocks; recursive main-text appendix-reference calls remain 26 and academic citation calls remain 12. These counts describe the edit; there was no citation-reduction quota. The full previous dependency audit remains in release history.

The exact paper diff is `release/manuscript_wording_0_5_9.patch` and the independent delivery also contains `changes.patch`. Source identity, checks, rendering and packaging records accompany the project.

## Further work

No experiment or additional analysis is required to finish this bounded wording task. A separately specified C7 context-family influence analysis could address whether the large lower-bound condition depends on particular families; it would not turn the present fixed-bank result into a population claim. It is optional future work and was not started or made a submission-completion condition. The manuscript still requires the author's final reading and submission decision; this report gives no acceptance probability or guarantee.
