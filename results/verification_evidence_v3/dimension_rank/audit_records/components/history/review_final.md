# Final targeted manuscript review

Reviewed the revision under `paper/` against `baseline_paper/`, limited to dimension selection, numerical rank, preserved metric contracts and experiment scope. Read the new audit appendix, modified main/methods/results/spectral/protocol sections, generated lexical table and the text embedded in the unchanged lexical figure. Also cross-checked the new numerical statements against `dimension_rank_audit.csv`.

## Result

**No material scientific inconsistency found in this bounded revision.** The chosen case-B treatment of the historical lexical comparison is appropriately conservative: raw lexical ranks6/3 and internal-input feasibility are verified, but missing historical projected arrays are not presented as directly verified. The prose uses “dimension-matched,” keeps the actual historical losses, and confines the finite-orbit conclusion to the verified raw lexical vectors.

## Specific checks

- q16 is described as a fixed setting, not intrinsic dimension, variance-selection threshold, externally preregistered optimum, or data-selected optimum. The design/implementation evidence and earlier-development uncertainty are both preserved.
- PCA centering and theoretical uncentered second-moment whitening remain distinct. The training mean is still `hat{mu}`, the column basis is `U_PCA`, scores remain z, and the reconstruction equation is unchanged.
- The raw unique lexical upper bound is correctly stated as `min(d,n−1)`; equality requires affine independence. No averaging across contexts is used to manufacture four/seven unique inputs.
- The changed main, methods and figure/table captions consistently use matched retained dimensions. Whole-paper source search found no remaining “rank-matched” or “entire lexical/centered span retained” claim. The embedded lexical figure itself says “matched q=6/3,” which is compatible with the revised wording and needs no regeneration.
- The original q16 lexical failure remains in the reproduction appendix. The separate6/3 repair is not converted into a q16 pass.
- 300 original FULL raw input ranks are now available: all C7 are195 and all D4 are111. This new evidence supersedes the earlier interim caveat that only the lower bound rank(H_c)>=16 was known. The revised appendix correctly limits the direct raw-input rank statement to the original run, while treating original/replay saved-coordinate findings separately.
- Both original and replay have300 saved PCA coordinate families. All their train/validation/test banks have observed numerical rank16. Their training singular ratios0.06729–0.41053 and the minimum among9,000Ridge/Joint matrices9.81e−5 agree with the CSV.
- The48later cells' raw full spectra and saved coordinate ranks are kept distinct. Training/test Sigma ratio minima0.003266/0.001625 and maximum saved-whitening residuals5.44e−15/7.61e−15 agree with the CSV and support applicability under the recorded numerical tests. The text does not present them as interval-arithmetic certificates or population guarantees.
- C7/D4 internal q6/3 EVR ranges54.42–69.83% and45.81–73.16% agree with the raw training-spectrum rows. Later q16EVR71.19–97.67% agrees with the full saved spectra. No retained-coordinate-only denominator is substituted.
- The dimensional sensitivity table uses matching models/depths/splits/endpoints at q8/16/32/64 rather than comparing the small subset against full-panel q16. It retains original normalization and states q16 is not uniformly best. No stability claim is extended to gamma, premiums, semantic rankings, witnesses or new-context transport.
- Main risk-premium formulas and interpretation boundaries are unchanged: premium jointly concerns exact group laws and orthogonality; p-minus remains a lower bound. The risk and registration generated tables, bounded macros and `sections/planned_studies.tex` are byte-identical to baseline.
- Appendix I.8 still says optional leave-one-family-out is prepared with no real family deletion/bootstrap executed. The new appendix repeats that it remains unexecuted. No new experiment is implied.

## Minor presentation observations

The explanatory rank paragraph now lengthens Section2; this is justified by the user's explicit request. Any further shortening should preserve the distinctions rather than removing the missing-coordinate caveat. There is no reason to rename r/q globally or revise the existing mathematical statements during this bounded pass.

The final CSV presently has13,784rows:13,484verified_supported and300unavailable. The latter are the unsaved historical lexical/internal training/validation/test coordinate entries, not300failed experiments. Ensure this distinction remains explicit in the user-facing summary.

## Unchanged checks verified by byte comparison

- `generated/bounded_risk_reference_table.tex`
- `generated/bounded_registration_margin_table.tex`
- `generated/followup_spectral_bounded.tex`
- `generated/followup_witness_bounded.tex`
- `generated/bounded_macros.tex`
- `sections/planned_studies.tex`

This review did not edit paper sources or run fitting, SVD, models, or experiments.
