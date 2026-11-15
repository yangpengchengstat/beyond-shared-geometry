# Input map: MAIN30 bounded tasks B/C

This map concerns new bounded derivations, not recertification of original experiments. The release manuscript reports 48 spectral cells and 2,952 image conditions; those full-precision saved rows are locally available and were read in this run. Only the algebraic B/C summaries and identity-action residual check are newly computed. Original gamma, eta, spectra, budgets, witnesses, and image results were not rerun.

## Scientific identity and normalization

- Bank: original Study 2 test bank, 24 fixed model/readout cells per domain, 64 structural-test families per cell, two paired variants per family, 128 complete state blocks. C7 has seven states and seven group elements; four-corner D4 has four states and eight group elements. This is neither Study 1 nor the new context bank nor a split mixture.
- Frozen rank is 16, readout is span_mean, and context/state/group weights are uniform. The complete group grid includes the identity and fixed points. PCA is the saved original PCA. Whitening uses each complete test bank's uncentered second moment; no extra centering, regularization, jitter, direction removal, rank reduction, or element deletion is applied.
- `cell_identities.json` retains model/tokenizer revisions, original coordinate/PCA/row member hashes, PCA-array hashes, cache/contract/source-manifest bindings, complete-test-context identity hash, streamed bank-member hash, weights, dimension, metric, and semantic action hash. A canonical JSON hash of this complete identity is the `join_key`, included in each risk and image row; model/layer alone is not the join key.
- `gamma_r.py` lines 363–397 computes `e=trace(restricted)/r` and `b=d*sum(selected multiplicity eigenvalues)/r`; both saved fields already are normalized dimensionless energy fractions. They are not divided by rank again. The original type-spectra CSV repeats e/b on each multiplicity row; validation deduplicates by (cell,type), obtaining 216 distinct type rows.
- The trivial type is `trivial` in C7 and `chi_pp` in D4. The saved compact CSV aliases D4 `chi_pp_e/b` to `trivial_e/b`. The identity action reference uses total e, never selected b.

## Endpoint mapping

| Manuscript endpoint | Saved source/field | New output and transformation |
|---|---|---|
| Table 3 gamma and eta | `spectral_cells.csv`: gamma, eta_LS; original `cell_summary.csv` and per-cell summary JSON: metrics.gamma, metrics.eta_LS; eta_spec is independently saved and equal within frozen tolerance | `cell_risk_reference.csv`: gamma, eta. Full-precision values checked exactly against saved summary/original CSV; no refitting |
| Table 3 class bounds | same per-cell metrics.L_lower, metrics.L_upper | a=2(1-gamma), u=2(1-gamma^2), checked against saved bounds |
| Same-metric identity reference | per-cell summary.types[].e, compact trivial_e | identity_risk=2(1-e_triv), plus new direct complete-grid check from archived saved test coordinates |
| Class premium in §5.1 | same gamma and eta_LS, same bank/PCA/action/weights/metric | premium_lower=max(0,a-eta), premium_upper=u-eta, computed per cell before equal-weight domain mean |
| Type energy supplement | per-cell summary.types[].e,b and original type_spectra.csv | `type_energy.csv`, complete 216 cell/type rows; e sum≈1, b sum≈gamma, 0≤b≤e under original tolerance |
| Registration margins in §5.2 | `alternatives_units.csv`: test_bank_gamma, semantic_registration, source_summary/hash; original saved plan canonical image_permutations | `image_endpoints.csv` (all 2,952), `canonical_images.json` (all 123 images), `registration_margins.csv` (48); semantic image excluded by canonical hash, not ordinal |
| Existing rank counts/ranges | `original_descriptive_ranks.json`; original numerics.registration_ranks | exact reproduction of rank_low/high; unique-leading/tied-leading/behind categories; no p-values |
| Table 4 frozen witness losses | `witness_units.csv`: test_witness_risk, train_witness_risk, test_ols_risk | unchanged source only; no witness is fitted, no training metric is changed, and these losses are not compared with test-bank orthogonal-class bounds |

## Exact archive and member binding

Archive basename: `tmlr-all-v2.20261030T203725Z-5489fd.results.zip`. SHA-256: `ef87e9aaeb8f093ba0c2438590d6a30b5dffcd256993e2366bd600cca3cb5bbb`. The original checksum sidecar was checked.

Only required metadata, per-cell saved summaries, 48 saved test-bank members, and row metadata were streamed; no bulk archive extraction was used. Selected raw JSON metadata is retained in `results/main30_bounded_v1/input_metadata/`. No coordinates are copied into the release. All read member hashes are in validation_report.json; original summary paths/hashes accompany risk and image CSV rows. The unchanged earlier provenance records the prior full receipt verification; this run does not claim to have repeated that earlier full-stage audit.

Key metadata hashes:

- `alternatives/plan.json`: `309419eadaaa34f244bf65546c1961d9591b6aabc30103af16b5dcda1587e664`
- `alternatives/run_manifest.json`: `4254bdc1a4be2998a4818e655213a53fd4a487e3ae87e6a7d36ba02eec67430a`
- `alternatives/input_snapshot/prepared_manifest.json`: `60010bf09c7327f9e01c5912fc935bd8830dc4434a9c962b55d2c92363dd14ae`
- `alternatives/input_snapshot/inputs.json`: `792e6955b64138f6961fd6a28c9e8e4765f0c51984013537222726f4768ed6b0`
- `alternatives/input_snapshot/parent_input_manifest.json`: `fdb499c80c34f01367dc946af1568a547e647598c2bfea45bc9c503f7c5a3b7e`
- `spectral-results-436df4be8558423f8e84aea0e4632408/input_manifest.json`: `fdb499c80c34f01367dc946af1568a547e647598c2bfea45bc9c503f7c5a3b7e`

Compact source SHA-256 values:

- `results/tmlr_followup_v2/spectral/spectral_cells.csv`: `4c8ff80dd0656008feb3cc9ba7820a72f19e2b66a6d3beb14b1801e93ff414e4`
- `results/tmlr_followup_v2/spectral/alternatives_units.csv`: `f0e70d8c8f6a96d629708f9c83c3bb226b09e5d95e6b4e53d334d51f488c963a`
- `results/tmlr_followup_v2/spectral/original_spectral_cell_summary.csv`: `f00d816093a2ed960f2f8b6daa961a9bbfbc5e51352d5744cb4da21d361e93b9`
- `results/tmlr_followup_v2/spectral/original_spectral_type_spectra.csv`: `0aa002a6564382da9cd4e325cf558ef1dc3a1b8d3db0c2eca36bfc3766fc09f6`
- `results/tmlr_followup_v2/spectral/original_descriptive_ranks.json`: `8ad085bea1f1ab5de27fcc21d193bd43f240e1f699e80dc8d541ce3538b4d004`
- `results/tmlr_followup_v2/spectral/provenance.json`: `3fcbfc1707d46e4bf5d97e939c0dd3ec0a8a38b95f1ef387c52a9300c827e8db`
- `results/tmlr_followup_v2/spectral/file_hashes.json`: `86151edcbbd7f7eaa5c3abdc7c94e437e1af264f7af9dacffbf216ad80a009b8`

## Tolerance, canonical identity, and output precision

The original dimensionless scalar tolerance is `1e-8 + 1e-7 * max(abs(a), abs(b))`, symmetric absolute-plus-relative, defined in `studies/spectral_extensions_v1/numerics.py:_tol` and inherited from the finite-bank constants. The original `registration_ranks` counts candidates strictly above semantic+tau and candidates within abs(delta)<=tau; both operations are reused/cross-checked without changing the tie policy. Per-cell min/max actual pairwise tolerance is retained in registration_margins.csv. These numerical ties are distinct from unrun finite-family sensitivity.

The saved `alternatives/plan.json` supplies all 120 C7 and three D4 image subgroups as sorted lists of actual permutations, their consistent action-grid representative, and semantic flags. The script hashes canonical JSON of each saved image_permutations list and validates image/action agreement, group direction, identity inclusion and distinctness. It never calls subgroup enumeration. Ordinal subgroup IDs are provenance labels only; they do not identify an image across arbitrary runs.

All numeric CSV values retain Python float round-trip precision. No clipping or tolerance adjustment is performed; `gamma_above_one` is reported explicitly (none in these 48 cells). TeX formatting uses ordinary decimal or three-significant-digit scientific rounding only; there is no tolerance-based zeroing. The human-readable tables are generated from the same numeric output objects.

## Verification levels and status

- `DONE`: B/C new saved-result derivations, full 48-cell and 2,952-image coverage, 216 distinct type-energy rows, complete input identity bindings, invariant checks, and all 48 new direct identity residual checks. Maximum absolute identity formula/direct residual difference is 3.774758283725532e-15.
- `VERIFIED_ALREADY_PRESENT`: original compact-result hashes, metadata bindings, saved full-precision summary agreement, published gamma rounding and rank counts. This does not independently confirm historical model extraction or re-execute original experiments.
- `PREPARED_NOT_RUN`: family sensitivity is a separate task; this analysis performs no deletion/bootstrap run.
- `NOT_IN_SCOPE`: new model forwards, GPU/cloud work, original calibration/full-suite reruns, new witness or exact-action optimization.
- No B/C input blocker or numerical failure remains. Selected source manifests before and after are identical.
