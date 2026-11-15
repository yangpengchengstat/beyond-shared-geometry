# Independent static review of the saved-matrix audit

Reviewed `scripts/audit_saved_matrices.py` and `audit_protocol.json` on 2026-11-05. Reviewed the corrected version after the first witness-receipt lookup failed. No model code, fitted estimators, PCA fitting, or SVD diagnostics were executed by this review. A separate supplementary script performs only SHA256, indexing, exact-array-equality, and source/AST checks.

## Findings resolved

1. **Committed witness receipt path (execution bug):** the initial script used `witness/units/<cell>.json`; actual returned receipts are `witness/units/<cell>/receipt.json`. The corrected script follows `unit['attempt']`, verifies `diagnostics.npz` and `summary.json` against the receipt's `files` hashes, and requires complete status. This is now correct. Preserve the first failed log; it is an audit-loader failure, not a scientific failure of the saved witness.
2. **Training/validation coordinate identity:** initially the audit matched test coordinates to the parent but only compared training coordinates between two prepared copies. The supplementary `verify_coordinate_bindings.py` now verifies all48cells directly against authenticated parent `coordinates.npz` and `rows.json`. Prepared train/test banks match the saved row-index maps exactly. Context train/validation arrays match the complete expected families, para0/1, states0..n−1 order exactly. Prepared-file and parent-source SHA256 bindings all pass.
3. **FULL row-order source identity:** the original run manifest records a clean Git commit `30f3c6369a185f43e94b17ca03b0499cd5d65a78`. At that commit, both dataset builders enumerate 40templates first and states second. They are byte-identical to the later replay source archive, whose SHA256 sidecar and source manifest match the saved replay identity. Hence the audit's `np.repeat(np.arange(40),n_states)` ordering is supported by the actual executed-source identities, not just today's builder. Original and replay PCA/serializer source bytes also match. The replay extraction file differs only by an added embedding extraction function; ASTs of the existing extraction functions are unchanged.

## Numerical-rule review

- The posthoc SVD rank rule in `diag` agrees with `audit_protocol.json`: float64 computation, threshold `max(shape)*eps64*s_max`, strict `>` count; no threshold tuning to obtain6/3/16 is present.
- The posthoc ill-conditioning flag uses the declared retained singular-value ratio threshold `1e−8`. It is distinct from the historical spectral **eigenvalue** ratio threshold `1e−10`; these thresholds must not be described as the same rule.
- Original saved PCA feasibility has two different historical implementations: Study1 uses the original centered matrix-rank guard; Study2 uses `s>s_max*1e−12`. The script reports the Study2 full saved spectrum and its historical rank, without reconstructing/retraining PCA. Where historical raw ranks are absent, rank(Z)>=q only supports rank(H_centered)>=q, not an exact rank(H_centered) value.
- Computing the existing uncentered second moment from unchanged saved Z is within scope. The symmetrization immediately before eigendecomposition only removes floating-point antisymmetry in the newly computed diagnostic; it does not change coordinates, original weights, saved whitening, or frozen outputs.
- Study2's original whitening is checked by applying the **saved** whitening matrix, not estimating a replacement. Spectral test-bank whitening and witness training whitening remain separate.
- Operator rank is diagnostic of invertibility. No refitting or replacement occurs; a deficient learned matrix would not by itself invalidate its predictive loss. The CSV correctly distinguishes Ridge/Joint, raw-coordinate exact-action, and whitened exact-action matrices by `matrix_kind`.

## Explained-variance review

The existing Study1 `pca_explained` array is safe to sum: `src/bsg/experiments/canonical_units.py:162` saves `explained_variance_ratio_`, and `src/bsg/representations/pca.py:24–29` divides retained squared singular values by the complete original training SVD energy. The supplementary provenance check confirms these two source files are byte-identical at original FULL and replay identities. This is reliable saved total-variance information; it is not the prohibited use of truncated-coordinate variance as total variance.

Study2 saves the full original training singular spectrum in each legacy record. Its EVR computation is likewise legitimate. When EVR is repeated on validation/test coordinate rows, the `EVR_basis` field must remain explicit that it is **training-PCA explained variance**, not explained variance measured on that validation/test bank.

## Remaining reporting cautions (not new experiments)

- `historical_rank_pass` on validation rows is an audit application of the historical threshold, not evidence that the historical program actually tested validation whitening. The original recorded test check is separately bound to `summary['checks']`. The final report should distinguish “passes the recorded rule on rechecking” from “recorded original pass.”
- `historical_whitening_pass='see preserved witness summary'` is a provenance pointer, not a Boolean pass. Use the independently recomputed residual/threshold and source summary as the factual basis; do not count a nonempty string as a successful Boolean test.
- A missing checkpoint is stored in `matrix_missing.json`. If any are present, the combined CSV should add an `unavailable` row; do not count omitted objects as supported. Parent reported none for the completed main numerical panel.
- The minimum eigenvalues emitted for operator/PCA-basis rows are generic diagnostic Gram eigenvalues. They are not observation-bank Sigma inputs to the spectral theorem; distinguish by `matrix_kind` and weights/notes.
- Original FULL/replay FULL exact raw rank(H_centered) is not recovered merely from their16-column coordinates. Preserve NA for that exact quantity unless a complete raw spectrum is retained; the supported lower bound rank(H_centered)>=16 is sufficient for primary projection feasibility.
- A full-rank numerical audit supports the stated finite-bank positive-definiteness and invertibility conditions at the declared thresholds. It is not an interval-arithmetic certificate, proof of optimal q, intrinsic dimension, semantic identification, or robustness at other r.

## Supplementary executable verification

Files: `verify_coordinate_bindings.py`, `coordinate_binding_checks.json`, `coordinate_binding_execution.log`. All48coordinate-cell bindings and five source-file identities passed; no SVD, fitting, model imports, model forwards, or input writes occurred.

Executed command:

```sh
audit_python_environment/bin/python3 audit_workdir/agents/history/verify_coordinate_bindings.py --repo original_research --out audit_workdir/agents/history
```
