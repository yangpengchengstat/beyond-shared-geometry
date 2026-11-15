# Dimension selection and numerical rank — targeted audit

Initial audit date: 2026-11-05. The initial audit is a new, independent revision of the live `paper_editing` project. A separately authorized offline spectral sensitivity was completed later and is integrated in the update below (2026-11-06 UTC); statements about the initial audit do not describe that later computation. The original manuscript, original archives, calibration failures, q16 lexical failure, replay stops and all frozen measurements are preserved. The initial audit performed no LLM forward, model download, RunPod action, FULL/AFS rerun, operator fit, PCA refit, dimension search, LOFO execution or new simulation. The later sensitivity computes closed-form independent OLS risks at fixed8/16 dimensions without refitting PCA or historical operators; it performs none of those model, RunPod, calibration or LOFO operations.

## Answers and scientific consequences

1. **Why q=16?** It is a documented fixed primary setting. The initial Git configuration had a development grid `[8,16,32]`; the first recovered primary-16 declaration is commit `30ae942c510db1ed622c16cdbc78f2ce17b556b1`. The record does not recover why this particular number was initially preferred. Compute control is documented for restricting the causal branch to one dimension, not as a demonstrated explanation for choosing structural q16. The protocol prohibits validation/test dimension selection; no dimension selector occurs in the audited execution. This does not certify absence of earlier development influence or external preregistration. Study 2 explicitly retains16 to match development comparisons, fits its own training PCA, and passes these coordinates to the later extensions. The manuscript now states fixed retained dimension, not intrinsic/optimal dimension.
2. **Are 6/3 real ranks?** Yes: the four saved unique lexical matrices have directly verified centered ranks 6/3. Every state's saved embedding is bitwise identical across its 40 templates; no averaging across contexts was used. All 20 historical split-level rank checks agree. The 80 original internal training input ranks are 195 (C7) and 111 (D4), with nonzero leading 6/3 singular values. Together with the recorded leading-q SVD procedure, these spectra imply full-q training scores at both readouts. The paper therefore retains **rank-matched**, explicitly meaning **the ranks of the training projected representations**. This is an input-and-algorithm-supported conclusion, not direct remeasurement of historical PCA/Z outputs. Those arrays and the fitted operators were not archived; all 300 train/validation/test coordinate entries remain `unavailable`. Training rank does not by itself establish every held-out internal bank's rank. No rank-5 weekday example occurs in these archived lexical inputs, although rank ≤ 6 need not equal 6 in general.
3. **Are the 16-dimensional assumptions supported?** Yes for the audited fixed banks. All300 original FULL q16 raw input matrices support16 directions (240primary+60terminal), and their saved PCA transformations reproduce their archived coordinates. All saved original/replay training, validation and test Z banks have numerical rank16. The48 later train and48 spectral-test banks pass the original second-moment rule `lambda_min > 1e-10*lambda_max`; saved whitening residuals pass1e-8. This supports the declared finite-bank r16 analysis, not population coverage or results at other dimensions.
4. **What changes scientifically?** The main changes concern dimension-choice justification, the lexical comparison's training-rank and verification scope, and dimensional-sensitivity interpretation. No audited primary/spectral input fails its declared feasibility condition; none of the frozen tables requires numerical replacement. Missing AFS-D Z prevents a claim of directly remeasured historical projected-rank equality; it does not negate the training-rank implication of the audited spectra and recorded PCA algorithm. This is an archival/verification limitation of the auxiliary lexical comparison, not evidence that its saved prediction scores are wrong, nor a failure of the separate 48-cell spectral input checks. The historical q16 lexical failure remains a real failure, not a pass converted by this audit. No corrective refit is needed for the claims retained in this revision.

## Current version established from source and history

- Compile entry: `paper_editing/main.tex`, with local `tmlr.sty`, bibliography, generated tables and included `sections/*.tex`. It is an untracked working paper, not the older tracked `paper/main.tex`.
- Metadata reports base release0.5.4, but the working text contains later edits. Its exact starting SHA256 is `878265781ab0177988ac0988cf0763324e12d996c05beec5b5650471527c58a8`; all55 baseline files are hashed in `provenance/paper_editing.before.sha256.json` and preserved in `baseline_paper/`.
- Repository HEAD: `1b7adfe717c6a7663c90e0a7c31d8a612f22afe9` (rank-aware AFS-D repair). It is provenance for historical code, not the current manuscript version. Tracked modifications and untracked material were present at the start; `git_status.before.txt` and `tracked_changes.before.patch` preserve that state. No reset, clean, checkout, commit or stash was used.
- The new paper is `paper/main.tex`. Its revision is identified by source hashes and a manuscript diff, not by assuming a numbered filename denotes the latest text.

## Dimension choice: exact evidence

Paths below are relative to the original repository. `R/` means `submission/tmlr_20261101_final_edit_v1/beyond-shared-geometry/`; its numerical source identities were checked against returned archives. The manuscript baseline is nevertheless the actual `paper_editing`, not that release's paper.

| Record | Supported conclusion |
|---|---|
| Git `ef5c7aa8464def6bf7acb9cd67cefae0f928402d`, `configs/main_experiment.yaml`, `latent_dimensions` | Initial grid8/16/32, without a sole primary dimension. |
| Git `30ae942c510db1ed622c16cdbc78f2ce17b556b1`, design at lines332–344 | Primary16 first explicitly recorded; causal restriction is described as controlling compute. Original structural motivation remains unknown. |
| Same design lines317–321; current `docs/FINAL_EXPERIMENT_DESIGN.md:423–427` | Training-only PCA; validation may check integrity but may not select dimension. |
| Git `90cce9681d8a29758fc248073e5969b97cfcdd73`; current design97–112 | Post-CORE amendment restricts sensitivity coverage for runtime, retaining primary16. Earlier development inspection is documented. |
| `configs/main_experiment.yaml:31–34,144–159`; `src/bsg/experiments/main_config.py:150–161` | `primary_latent_dimension:16`; sensitivities8/32/64 at depths.4/.8, splits101/303. |
| `docs/EXPERIMENT_PROTOCOL.md:44–59` | No result-based dimension selection in the specified protocol. |
| `R/studies/characterization_v1/frozen_run/study_config.json:202–203,265–272` | Study2 fixed16, no secondary-q sweep, explicitly matching development comparisons. |
| `R/studies/reproduction_v1/characterization.py:293–318` | Training-mask-only PCA at16 and saved PCA/coordinates. |
| `R/studies/compact_context_v1/protocol.json:4–8,56–77` | Reuses parent16-dimensional coordinates; validation chooses predictor parameters, not dimension. |
| `R/studies/spectral_finite_bank_v1/protocol.json:95–118,185–190`; `gamma_r.py:264–298` | Fixedr16, no extra centering/rank reduction/regularization; uncentered second-moment threshold and saved whitening. |
| `R/studies/spectral_extensions_v1/protocol.json:32–38,67–81` | Training-whitened witnesses versus separate test-bank spectral target, bothr16. |

`components/history/report.md` supplies the detailed timeline; `history_evidence.json` includes source hashes, Git objects and equality checks. This audit supplies no95%-variance rule, validation optimum, intrinsic dimension estimate, group-theoretic requirement for16 or externally registered selection claim.

## Numerical contract and object meanings

- `q`: retained coordinate count.
- `H_centered`: original training activations after the training mean is subtracted; many rows/columns may remain outside the retained span.
- `Z`: an identified train/validation/test coordinate bank under saved PCA. Test banks are **not** additionally centered for the spectral analysis.
- `R_g`: a saved predictor matrix; regularization makes its fitting system solvable, not necessarily the resulting matrix invertible.
- `r`: declared theoretical action-space dimension. For the primary spectral application r=16. The separate sensitivity also declares r=8 and checks that input explicitly; changing dimension changes the target class, whitening and normalization, and is not an automatic rank-deficiency repair.

Before new SVD diagnostics, `audit_protocol.json` fixed float64 computation with `r_num(A)=count(s_j > max(shape(A))*eps64*s_1)`, absolute threshold0. The same default rule is recovered in the original lexical rank guard. Study2's historical PCA uses `s_j > 1e-12*s_1`, retained alongside the posthoc diagnostic. The audit's additional condition-warning threshold is `s_q/s_1 <=1e-8`, fixed before the new results; it is not an old scientific gate. Inputs' stored float32/float64 precision and float64 calculation precision are recorded separately.

`verified_supported` means the available matrix supports the declared q under the specified rule, not that its ambient feature columns are all independent, that q is intrinsic or that an operator is an exact group action. `unavailable` never counts as a pass. New diagnostics apply old rules where stated; an audit recheck of validation is distinguished from an originally recorded validation check. Operator/basis Gram spectra are not treated as the observation Sigma for the theorem. EVR entries attached to validation/test records describe their **original training PCA**, not newly estimated validation/test variance.

## Coverage and results

### Lexical comparison (original AFS-D)

`src/bsg/representations/extract.py:13–22,94–125` implements unique target-span selection, `hidden_states[0]` and token-span averaging. Dataset builders supply template-major/state-minor order. The original source at clean commit1b7adfe and input member hashes are verified. Historical token-ID/offset arrays were not saved, so the audit does not claim to have re-tokenized those prompts. The decisive unique-vector check uses the actual saved raw embeddings.

| Unique lexical matrix | Centered rank | smallest retained / largest singular value |
|---|---:|---:|
| Llama-3.2-1B C7 |6|0.4311007151720606|
| Qwen2.5-1.5B C7 |6|0.5501007075516324|
| Llama-3.2-1B D4 |3|0.7833253339996674|
| Qwen2.5-1.5B D4 |3|0.7629608385308416|

Coverage is two models, two domains, five splits and four internal depths:20 lexical fits and80 internal fits, from4 lexical and16 internal extraction arrays. Reused embedding arrays are not additional observations. Internal centered input rank is195/111; the minimum qth/first singular ratio is.4799/.5512. The raw lexical vectors are not deficient or near-degenerate under the fixed diagnostics.

The executed PCA procedure supplies a stronger conclusion than column count or input feasibility alone. At repository commit `1b7adfe717c6a7663c90e0a7c31d8a612f22afe9`, `src/bsg/representations/pca.py:16–36` centers training activations, computes the full SVD, selects the first q right singular vectors and applies that basis. `src/bsg/experiments/canonical_units.py:27–38` checks the training input rank and uses that procedure. For these fits the implementation's row/feature cap does not reduce q. These two current source files were also checked byte-for-byte against that commit during this wording revision; their SHA256 values are `4836f1762b9b51a58fb0ac44106ffdeae55adcadcccbb3dd1ad82e98af412799` and `a2c845fa5fbbeace0157324bf1f629c9c808adbfaebbd70142826756102ce732`, respectively.

For the centered training matrix, write

\[
H_c=L\Sigma V^\top,\qquad Z_{\mathrm{train}}=H_cV_q=L_q\Sigma_q.
\]

Thus `sigma_q(H_c) > 0` implies `rank(Z_train) = q`. The audited retained singular values are well separated from the fixed numerical rank threshold at both readouts. This supports **rank matching of training projected representations** at 6/3; it does not match the raw internal rank 195/111 to the lexical rank 6/3. No new SVD, PCA or coordinates were computed for this clarification; the argument uses the audit's existing input diagnostics and verified source.

AFS-D did not save PCA/Z/operator parameters (`afs_d.py:663–669,832–837,875–885`); filenames ending `operator.csv` contain metrics. All 300 expected coordinate-bank records remain `unavailable`: that status describes access to the actual historical output arrays, not the validity of the training-rank implication. The input-and-algorithm argument is not a direct floating-point remeasurement or bitwise reproduction of those outputs, and training full rank does not imply full rank for every internal validation/test bank. No fitted operator rank is inferred from this argument. No missing PCA or coordinates have been reconstructed.

### Primary q16 and later finite banks

- Original FULL:300 original centered training matrices, all rank195(C7) or111(D4);240 belong to the primary four-depth panel and60 to terminal-layer secondary fits. Saved PCA/coordinate bindings all pass; largest relative raw-to-saved-coordinate residual is1.908e-15. The fitted mean matches exactly.
- Original FULL and replay separately:300 train,300 validation and300 test coordinate banks per run, all numerical rank16; Joint copies match base coordinates/PCA exactly. Training singular ratios range.0672881–.4105285. Replays are not independent datasets.
- Saved Ridge/Joint operators:9,000 element matrices across the two records, all numerical rank16; smallest ratio9.80967e-5. This is an invertibility diagnostic, not exact group-law evidence. Operators outside this accessible coverage are not inferred from performance metrics.
- Study2:48 complete saved original training spectra supportq16 under the historical1e-12 SVD rule;48 training,48 validation and48 test coordinate banks have rank16. Saved witness/OLS/ridge matrices are audited without refitting.
- Spectral inputs:all48 test banks are exactly mapped to original coordinate rows. The supplementary verifier also binds training and validation to authenticated parent row IDs and preparation hashes, not just to matching copies.

| Bank | min/max eigenvalue ratio range | largest rechecked saved-whitening residual |
|---|---:|---:|
|48 training banks|0.0032662651–0.1493289837|5.43654e-15|
|48 spectral-test banks|0.0016253867–0.0870490173|7.60614e-15|

Every ratio exceeds the original strict1e-10 threshold; every whitening residual is below1e-8. Each object's full-precision eigenvalue extrema are in the CSV. All48 original recorded test checks and the witness training checks are retained separately from this recheck. The bank weights and fixed coordinates were unchanged. No affected spectral cell or theorem-application failure was discovered in this audit; this is not a rigorous rounding certificate.

### Explained variance with valid denominators

For original FULL primaryq16, original full-training energy gives EVR ranges70.52–92.45%(C7) and76.32–97.53%(D4). Study2's complete saved training spectra give71.19–97.67% across48cells. AFS-D internal original inputs give54.42–69.83% atq6 and45.81–73.16% atq3. Original `pca_explained` entries normalize by the complete SVD energy, verified against actual executed source; they are not normalized within the retained coordinates. These observed values do not explain the original choice of16. High EVR also does not guarantee retention of every action-relevant feature. Missing denominators remain NA.

## Existing sensitivity: matched, not expanded

Original `main_full_003` and replay `legacy_full` each have384 complete base/Joint units in the existing q8/16/32/64 sensitivity comparison. The matched subset fixes six models, domainsC7/D4, depths.4/.8, splits101/303, endpoint, templates, fitting method and control seeds. Each run contributes18,432 original per-template records; the delivered CSV has6,144 per-condition aggregates across the two runs, kept separate. Hashes and per-checkpoint values were checked. There are no missing/failed units **in this subset**; historical failures elsewhere remain preserved.

The pre-existing aggregate `latent_dim_sensitivity.csv` did not restrict its q16 row to this subset. The audit does so without overwriting that table or the primary results.

| Original test E_eq |q8|q16 matched|q32|q64|
|---|---:|---:|---:|---:|
|C7 Ridge|.115638296|.051285323|.050219966|.136808271|
|C7 Joint|.116178801|.052251168|.050733031|.135502090|
|D4 Ridge|.055581346|.060366010|.070745070|.135896093|
|D4 Joint|.056168531|.061449638|.071429361|.135591392|

The original rowwise-relative normalization is preserved. Because q changes the target coordinates, these are the recorded structural endpoints, not a common ambient reconstruction loss or a new dimension-selection objective. They establish neither a uniformly bestq16 nor robustness of later gamma_r, premium, registration rankings, witness transfer or authored-context results across dimensions. That initial audit did not evaluate new spectral dimensions. A separate fixed8/16 comparison has since completed, as documented below; witness transfer and authored-context transport were not re-evaluated. LOFO remains unexecuted.

## Manuscript changes and preserved content

Changes are limited to Section2's dimension/rank definitions; Section4.3's lexical interpretation; AppendixC.2/C.6, D.2 and lexical table/figure wording; the spectral positive-definite-input explanation; and a new AppendixJ with the audit and existing matched sensitivity table. Existing labels are preserved. Generated numeric cells in existing tables, the abstract's reported numbers, theorem statements, risks/metrics, Table3/4 distinction, Table5 premium, Table6 margins, AppendicesI.7/I.8, and LOFO's prepared/unexecuted status remain unchanged.

The original48-cell premium means/medians/ranges and registration comparisons are neither recomputed nor filtered here. No result was dropped, no numerical tolerance changed and no favorable dimension substituted. The new sensitivity table reports existing measurements with explicit matched coverage. `manuscript_changes.diff` contains the full English text diff.

The unresolved archival item is absent AFS-D projected arrays. Restoring original arrays from an independently preserved cache would allow direct remeasurement. Their absence does not prevent the narrower, input-and-algorithm-supported training-rank statement. The previous recommendation to replace "rank-matched" categorically with "dimension-matched" was overly restrictive and is superseded by this explicit training-support scope. The component report records the same clarification while preserving all measurements and unavailable statuses. A refit would create new evidence, not recover a lost historical array; it is neither executed nor required. No new experiment is needed to support the claims retained here. The independent 48-cell r16 spectral checks and results remain unchanged.

## Deliverables and reproducibility

- `paper/`: complete revised LaTeX project and revised PDF.
- `baseline_paper/`: byte-preserved starting working project, including the existing PDF as found (not asserted to have been freshly compiled from the working source).
- `dimension_rank_audit.csv`: combined per-object audit, including unavailable objects, full-precision statistics and source identities.
- `dimension_sensitivity_existing.csv`, `dimension_sensitivity_summary.csv`, `dimension_sensitivity_coverage.csv`: only verified existing sensitivity measurements and matching summaries.
- `scripts/`, `components/`: read-only numerical/history/verification programs, frozen rules, component reports and actual command logs.
- `provenance/`, `logs/`: original and revised source hashes, archive/member identities, recorded original Git changes, checks and compile diagnostics.
- `COMMANDS.md`, `COMPILE_CHECK.md`, `PRESERVATION_CHECK.json`: execution and final verification records.

The archived input matrices remain in their original repository locations; multi-GB source archives are referenced by relative path and SHA256 rather than duplicated in this revision. Reproduction of the matrix diagnostics requires those retained archives. The new tables and manuscript can be inspected/compiled directly from this folder. Compiler layout warnings are reported separately from errors; successful compilation is not a scientific pass.

## Subsequent completed spectral dimension sensitivity (2026-11-06 UTC)

This is additional, authorized offline analysis, separate from the initial rank audit and historical structural sensitivity. It uses the original Study2 coordinates and full complete-state test banks for the same48 model/readout/domain conditions. The primary16-dimensional results are retained. The8-dimensional observations are the first8 saved training-PCA scores; each dimension has its own unchanged-rule, uncentered second moment and whitening. Historical PCA and operators are not refitted; independent OLS risk is computed as a new spectral endpoint in each dimension.

Six numerical software tests passed. All3000 original16-dimensional canonical/image results and48 registration rankings matched under the original rules before8-dimensional execution. All6000 planned units completed with no dropped conditions. The8-dimensional minimum second-moment eigenvalue ratios are0.03673(C7) and0.007785(D4), above1e-10. See `offline_dimension_execution_v1/analysis/input_preflight.json`, `analysis/registration_anchor_check-6a9f40b8dfd64602ac52471465074499.json`, the final report directory and preserved run receipts.

| Endpoint | C7:16→8 | D4:16→8 |
| --- | --- | --- |
| Positive premium lower bound beyond fixed tolerance |24/24→24/24|24/24→24/24|
| Mean lower endpoint |0.004164→0.064969|0.026237→0.010235|
| Semantic image uniquely leading |7/24→4/24|16/24→19/24|
| Type correction above original comparison tolerance |1/24→8/24|0/24→0/24|

The lower bound and absence of a uniformly preferred semantic image persist in this downward truncation. Magnitudes, typed effects, registration ranks and heterogeneity change: the two largest C7 lower endpoints account for90.7% of the sum at16 dimensions but21.8% at8. These are different coordinate/whitening metrics, so smaller errors are not a selection criterion. The result does not identify16 as optimal, recover its original motivation, certify a population, or establish stability at32/64 dimensions.

The local cache inventory found only16-column original Study2 bases/scores. Complete saved singular spectra cannot recover missing directions. Higher-dimensional arrays in the follow-up belong to the new-context bank and cannot replace original training/test inputs.32 dimensions remain unavailable rather than silently counted as passed.

The current paper adds AppendixK and its compact Table32, qualifies the original primary results asr16, and links the separate r8 input checks. The old Table3/4/5/6 numbers, original rank audit CSVs, source thresholds and failed records are unchanged. The manuscript does not extend this check to witness transfer, nonlinear predictor rankings, new-context transport or LOFO. Full-precision summaries and table-generation code are included under `paper/supplementary/spectral_dimension_sensitivity/`; the complete numerical source/receipts remain in the standalone execution directory.

Editorial backup: `revision_history/before_spectral_dimension_integration_20261106/`. The original64-page paper and old project documents there preserve the earlier audit's exact wording. `SPECTRAL_DIMENSION_UPDATE.md`, the new diff and compilation/preservation records document this integration. No numerical analysis was rerun during the editorial integration.
