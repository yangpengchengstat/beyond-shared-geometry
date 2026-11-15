# AFS-D lexical-dimension and rank audit

> **Training-rank scope clarification, 2026-11-05:** The numerical diagnostics and all `unavailable` statuses below remain unchanged. The earlier editorial recommendation below to replace "rank-matched" with "dimension-matched" is superseded. The verified input spectra and recorded leading-q SVD procedure imply full-q **training projected** ranks at both readouts; "rank-matched training representations" is therefore supported. This is not direct remeasurement of the unsaved historical PCA/Z outputs and does not establish every held-out internal bank's rank. See the current `DIMENSION_RANK_AUDIT.md` and manuscript Appendix J for the derivation, source references and archival limit. No new matrix diagnostics or experiments were run for this clarification.

This is a post hoc read-only audit of existing inputs. It does not execute model extraction, refit PCA, reconstruct missing projected coordinates, refit operators, or change frozen results. All results below are from the explicitly identified historical run, not a new independent replication.

## Source and experiment identity

- The current paper's `generated/lexical_control.tex` and `figures/lexical_internal.pdf` trace to `afs_d_rank_aware_001`, as specified in `submission/tmlr_20261101_final_edit_v1/beyond-shared-geometry/scripts/build_paper_assets.py:24,107,158,216`.
- Original run archive: `AFS_D_rank_aware_001_full_runpod_output.tar.gz`; its `run_outputs/algebra_falsification/afs_d_rank_aware_001/manifest.json` records clean source commit `1b7adfe717c6a7663c90e0a7c31d8a612f22afe9`, status `SUCCESS`, NumPy 2.1.2, extraction dtype float16, cached array dtype float32. The inspected repository source files below are unchanged relative to that HEAD.
- Frozen internal input archive: `runpod_packages/Beyond_Shared_Geometry_afs_d_rank_aware_immutable_inputs.zip!immutable/run_outputs_9_4_2026_full3.zip`, inner SHA256 `5fe7234572120073d8dfab2e8dc0d3373a741f8d48abadc92288f98ab16721f4`. Its 16 specific activation members match every hash in the run's `provenance/internal_activation_availability.csv`.
- The replay archive `incoming/rank_recovery_v1/from-zero-v1-rank-recovery.results.zip` contains the same four embedding checkpoints and the same 20-row rank diagnostic CSV byte for byte. They are reused evidence, not four new vectors or twenty independent rank experiments. The replay has no additional saved AFS-D PCA or projected arrays.
- Exact hashes of consumed members and preservation checks are in `input_sha256_before.json`, `replay_archive_check.json`, and `preservation_check.json`.

## Extraction and the unique-vector premise

`src/bsg/representations/extract.py:13-22` rejects missing or repeated target substrings. Lines 94-125 extract Hugging Face `hidden_states[0]`, select token positions whose tokenizer offsets overlap that unique target span, and average the token vectors within that span. This averaging is over target **tokens**, not over contexts. The original configuration has a fast tokenizer, right padding, special tokens, and tokenizer revision equal to the pinned model revision. Lines 57-82 give the corresponding block-output extraction procedure. `src/bsg/experiments/afs_d.py:127-132` locks these readout descriptions; `src/bsg/experiments/algebra_falsification_units.py:436-444` checks architecture and layer count, but does not by itself prove state-wise equality.

The decisive check uses the retained raw matrices. `src/bsg/data/c7.py:34-39` and `src/bsg/data/d4.py:32-37` establish template-major, state-minor row order. In each of the four saved embedding matrices, the 40 observed rows belonging to a given state are **bitwise identical**, and different states provide exactly 7 (C7) or 4 (D4) unique rows. The audit chooses the first observed row for each state only after this check. It never averages different contexts to manufacture a seven-/four-row matrix. See `lexical_identity_checks.json`.

The historical per-prompt token IDs/offset arrays are not retained, and this audit does not rerun tokenization or download tokenizers. Therefore extraction-source and config checks are separate from direct checks of the saved embedding arrays; it does not claim a fresh token-ID audit.

## Actual coverage

Two models only: Llama-3.2-1B and Qwen2.5-1.5B. Two domains, five split seeds 101/303/909/2701/4243, 40 complete templates (28 train, 6 validation, 6 test).

- Four raw lexical matrices, shapes 280x2048 and 160x2048 for Llama, 280x1536 and 160x1536 for Qwen.
- Twenty lexical training-rank diagnostics, using the same four matrices with repeated template partitions.
- Eighty internal fits: two models x two domains x four depths x five splits. Llama blocks 3/6/10/13; Qwen blocks 6/11/17/22. The raw internal arrays are 16 matrices shared across split choices, not 80 independently collected matrices.
- One hundred completed lexical/internal fit units; each has a train, validation and test coordinate object in the audit availability table. All **300 saved-coordinate entries are unavailable** because the run retained metrics but not these arrays or their fitted PCA bases.

## Numerical rule and results

The historical rank diagnostic is `np.linalg.matrix_rank(np.asarray(matrix, dtype=float))` after arithmetic training-row centering (`src/bsg/experiments/afs_d.py:554-586`; `src/bsg/experiments/fitting.py:55-63`). The default rule is a strict singular-value comparison with `max(shape) * eps(float64) * sigma_max`. The audit reproduces that rule in float64 from float32 saved inputs (NumPy 2.3.5 in the audit runtime). `protocol.json` was written before the new diagnostics. Its additional post hoc condition warning, also fixed before seeing results, is `sigma_q/sigma_1 <= 1e-8` for a matrix otherwise supporting q directions. This warning is not represented as a historical gate.

| Raw unique lexical input | Centered rank | sigma_q / sigma_1 |
|---|---:|---:|
| Llama-3.2-1B, C7 | 6 | 0.4311007151720606 |
| Qwen2.5-1.5B, C7 | 6 | 0.5501007075516324 |
| Llama-3.2-1B, D4 | 3 | 0.7833253339996674 |
| Qwen2.5-1.5B, D4 | 3 | 0.7629608385308416 |

Thus the theoretical bound `rank(E_c) <= min(d,n-1)` is attained by these four archived lexical matrices. The statement is directly verified; it is not inferred from the number of states. All twenty historical training ranks are reproduced: 6 for C7 and 3 for D4. There is no low-rank or near-degenerate retained lexical direction under the declared diagnostics.

For the original internal training matrices, all 40 C7 fits have centered input rank 195; all 40 D4 fits have rank 111. These are ranks of the pooled **original** training activations, not ranks of their q-column coordinates. They greatly exceed q=6/3, as permitted when contextual variation is present. At the declared retained cutoff, `sigma_q/sigma_1` ranges 0.4799346303--0.6371863413 for C7 and 0.5511959569--0.7342619142 for D4. Every input supports the specified q; none triggers the additional condition warning.

The full original training matrices are available for these 80 internal cells, so a legitimate SVD-based EVR(q) diagnostic can be computed without re-fitting or replacing PCA. The top-q energy divided by the **full original centered training energy** ranges 0.5441559059--0.6983097579 (C7, q=6) and 0.4580771039--0.7315736806 (D4, q=3). Lexical EVR is one up to floating-point rounding. This concretely illustrates that equal retained dimensions do not imply equal retained information. Raw CSV values are not clipped (a lexical value such as 1.0000000000000004 is ordinary rounding, not an additional variance claim).

## What remains unverified

`src/bsg/experiments/canonical_units.py:27-38` checks the observed input rank and computes a full-SVD PCA in memory; `src/bsg/representations/pca.py:16-33` takes its leading right-singular vectors. `src/bsg/experiments/afs_d.py:663-669,832-837,875-885` persists only operator/generator **metric CSVs**, not matrix parameters, PCA or Z. Inspection of both the original and replay archive members confirms this limitation. The term `operator.csv` denotes evaluation metrics, not a saved operator matrix.

The verified input spectra and implementation support the feasibility of the intended q-dimensional fits. They also imply full q-dimensional training scores in exact SVD arithmetic. They do **not** constitute a direct post hoc measurement of the actual historical projected training/validation/test arrays. The audit therefore leaves every retained-Z rank blank with `status=unavailable` and does not reconstruct it. No operator ranks are inferred from metrics or Ridge regularization.

Conservative manuscript treatment: call this a **dimension-matched lexical comparison**, state the directly verified lexical ranks and source-input feasibility, and explicitly note the absence of independently retained projected arrays. It is valid to retain the finite-orbit argument for the full-rank archived lexical vectors. Avoid claiming that both locations' saved coordinate ranks were measured in this audit. The theoretical possibility that seven states could have centered rank five remains real in general; it simply does not occur in these four archived raw lexical inputs.

## Recommended manuscript locations

- Section 2 retained-dimension paragraph: explain n-1 is an upper bound, equality requires affine independence, and the actual four lexical inputs attain the bound under the documented numerical rule.
- Section 4.3: change rank-matched to dimension-matched; distinguish lexical full-span feasibility from incomplete direct saved-coordinate validation. Keep the warning about information retention, strengthened by the internal EVR ranges if desired.
- Appendix C.2/C.6: identify readout, source hashes, numerical rank rule, 20 split diagnoses and 80 internal input checks. Explicitly separate raw inputs from unavailable projected arrays.
- D.2, lexical table and figure captions: use dimension-matched retained coordinates and the exact two-model coverage. The metric values and confidence intervals are unchanged.

## Deliverables and actual execution

`lexical_dimension_rank_audit.csv` has 404 rows: 4 unique lexical E diagnostics, 100 centered original training-input diagnostics, 300 unavailable projected-coordinate entries. `audit_lexical_rank.py` is a self-contained NumPy-only read-only script. `execution.log` records all executed matrix diagnoses and completion. `protocol.json`, input hashes and `preservation_check.json` record the unchanged inputs. No PCA/operator fitting or model forward was performed.
