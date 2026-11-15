# Optional family deletion sensitivity — MAIN30 task D

Status: **PREPARED_NOT_RUN**. This revision prepared code, checked retained input identities and ran synthetic engineering tests. It did not execute any family deletion on LLM coordinates. Completion of the manuscript revision does not depend on this optional analysis. This protocol was prepared after the original results were available; it is not a preregistration or independent confirmation.

## Target and fixed panel

The target is the change in the original Study 2 test-bank spectral margins and semantic-image rank when one of its 64 families is omitted. The panel remains all 48 original cells (24 per domain), fixed saved PCA coordinates at q = r = 16, and the original C7 seven-state and D4 four-corner state actions. This is not Study 1's 40-template bank, the new-context bank, a mixture of splits, or a resampling of saved scalar gamma values.

Each family consists of variants 0 and 1. Delete the two complete context blocks together, preserving every state of each retained block. Every deletion has 63 equally weighted families, 126 equally weighted contexts and all 7 or 4 states per context. Use the same named deletion in every model/readout cell. The available IDs are `structural_test_000` through `structural_test_063`. Their repeated strings alone do not establish semantic matching across domains; this implementation makes no cross-domain paired interpretation.

For each deletion, recompute the uncentered empirical second moment, its full-rank inverse square root and every image's K and typed integer spectral budget gamma. The saved original PCA projection and its center remain fixed; no new PCA, additional centering, direction deletion, new forward pass or predictor training occurs. All image comparisons share the deletion's complete bank, weights and newly computed metric. Identity elements and fixed points remain present. Changing whitening is required by the new finite-bank target and is not predictor refitting.

## Retained input and availability

The inspected results archive is `tmlr-all-v2.20261030T203725Z-5489fd.results.zip`, expected archive SHA-256 `ef87e9aaeb8f093ba0c2438590d6a30b5dffcd256993e2366bd600cca3cb5bbb`. Whole-archive hashing is handled by the parent revision's provenance work; this entrypoint does not repeat it. It verifies pinned metadata hashes and each bank and rows member against the pinned prepared manifest.

The metadata-only inspection in `preparation_audit/availability.json` found all 48 required cells, both paired variants, all state IDs, the fixed 16-column coordinate schema and every expected bank/row member hash. The NPY headers report test arrays of shape `(128, 7, 16)` for C7 and `(128, 4, 16)` for D4. This inspection read compressed bytes and array headers but did not load coordinate values or compute a deletion. Finite values, conditioning and evaluability after deletion are not certified by the header check.

Input schema, relative to the archive:

| Member | Required meaning |
|---|---|
| `alternatives/input_snapshot/inputs.json` | All 48 cells, model/layer/domain identity, bank path, preparation status, original source-member hashes, `binding`, saved PCA basis/center identities and dimensions. |
| `alternatives/input_snapshot/prepared_manifest.json` | SHA-256 map for input members, itself pinned in `config.json`. |
| `alternatives/input_snapshot/cells/<cell>/banks.npz` | `test` is the original saved PCA coordinate array, float64, with dimensions `(128,p,16)`. `train` is retained in the file but unused by this analysis. No object arrays/pickle. |
| `alternatives/input_snapshot/cells/<cell>/rows.json` | `test_contexts` and `test_indices` associate each coordinate block with split, family, variant, orbit, all state IDs, canonical row IDs and original source indices. D4 rows retain their original `D4_corner` name. |
| `alternatives/plan.json` | Saved complete image list: 120 C7 and 3 D4 images with actual `image_permutations`, semantic flag and frozen action/group specification. |

Canonical image identity is SHA-256 of the sorted actual permutation list. Display IDs such as `C7_image_000` are retained for traceability but never used as the scientific join identity. The execution uses the saved image plan; it does not introduce a different enumeration or deduplication rule. Synthetic tests alone call the original enumeration function to exercise all comparators.

The code also hashes the unchanged frozen numerical core, extension numerics and parent protocol before a real input audit. It verifies the complete model/layer/domain panel, not merely a count. No subset is silently substituted when an input is missing. Missing or invalid metadata is recorded as `MISSING_INPUT_OR_INVALID_INPUT` with overall `PREPARED_NOT_RUN`; dependent execution stops.

## Sufficient statistics and independent synthetic check

For each cell, read its test coordinates once. With coordinate indices a,b, states s,t and variants v, store

`F[f,s,t,a,b] = (1/2) sum_v z[f,v,s,a] z[f,v,t,b]`.

For deletion f, form `T = (sum_h F[h] - F[f]) / 63`. Then

`Sigma[a,b] = (1/p) sum_s T[s,s,a,b]`,

`Twhite[s,t] = W.T @ T[s,t] @ W`, where `W = Sigma^(-1/2)`, and

`K[g,a,h,b] = (1/(|G| p)) sum_s Twhite[g.s,h.s,a,b]`.

Thus F contains every family second moment and the state-pair moments needed for every complete orbit Gram matrix. This optimization sums original coordinate contributions; it never resamples gamma, averages prewhitened K matrices or fabricates family data from aggregate summaries. Group-only regular-action matrices and type bases are reused across the images, whose multiplication tables and type contracts are checked against the frozen core. Whitening is performed once per deletion, shared by its images.

`test_sensitivity.py` compares this route with direct deletion of synthetic coordinates and `gamma_r._analyze_spec`. It checks the full K, total type energies and gamma across all 120/3 images at small synthetic rank, and additionally tests q=16 synthetic banks. This is an engineering equivalence test, not a real-data sensitivity result. No jitter, eigenvalue clipping or coordinate-rank reduction is used by either route.

## Numerical rules and output

Reuse the frozen core's strict rank rule: the smallest second-moment eigenvalue must exceed `1e-10` times the largest, which must be positive. Preserve the original matrix normalized-Frobenius tolerance `1e-8`, scalar absolute tolerance `1e-8`, scalar relative tolerance `1e-7`, and K spectral tolerance `1e-8`. Failed checks remain failures; do not replace a failed deletion, lower r, add regularization or select a different tolerance. Raw gamma is retained, including any tolerance-sized excursion beyond [0,1].

Ranks call the unchanged `studies/spectral_extensions_v1/numerics.py::registration_ranks`, which uses `_tol(a,b) = 1e-8 + 1e-7 max(|a|,|b|)` on dimensionless gamma. It counts candidates strictly greater than semantic gamma plus this symmetric tolerance as better and candidates within the tolerance as tied, including the semantic image itself. The rank interval is `[better+1, better+tied]` when all images are evaluable. The optimized path uses the unchanged `gamma_r.integer_budget` for whole irreducible-copy allocation and preserves its raw-value maximization/exact-tie rule.

For each deletion and cell, report `d_top = gamma_semantic - max(gamma_alternative)`, `d_median = gamma_semantic - median(gamma_alternative)`, all-image spread, the original rank interval, numerical diagnostics, failures and actual image identities. The strongest/median alternative excludes the semantic image. If any comparator fails, the complete-set margins and complete rank interval are null; the original rank function's evaluable/possible rank bounds and failed count remain available.

A future authorized execution creates:

- `deletions.jsonl`: all 3,072 cell/deletion records, each with the 120 or 3 image results, numerical checks, conditioning, source binding and PCA identity. Rank failures retain an explicit failed entry for every planned image.
- `execution_summary.json`: each cell's valid and failed deletion counts, valid-deletion minima/maxima of both margins and rank endpoints, and unique-leading count with denominator 64. Failed deletions remain in that denominator and are reported separately. The code labels this a count rather than a probability or a claim of baseline-rank retention.
- `execution_config.json` and `availability.json`: fixed parameters and input receipts. Existing output directories are never overwritten; interrupted output remains an incomplete receipt without a completion summary.

These summaries describe only finite-family composition sensitivity. They are not population confidence intervals, permutation p-values, probability of semantic truth, exact-action risk rankings, recovery of an untouched test set or confirmation of the mechanisms. No bootstrap is implemented. A future bootstrap would need its own fixed sample count, seed, shared family draws, failure policy and aggregation rule, decided explicitly rather than after viewing results.

## Resources and controlled execution

The entrypoint uses Python and NumPy on CPU, one cell/deletion/image loop at a time. It sets common BLAS thread limits to one before importing NumPy; if `threadpoolctl` is installed, it also limits the loaded pool. The inspected audit environment has NumPy and lacks the optional `threadpoolctl`; the fallback is covered by a synthetic test. Invoke as the standalone CLI so environment limits precede NumPy loading. No GPU, model libraries, download, network or new model forward is required.

The full fixed workload is 64 deletions × 48 cells with domain-specific image counts, namely `64 × (24 × 120 + 24 × 3) = 188,928` image evaluations and 3,072 fresh whitenings. Each image works with a `(m*16) × (m*16)` K: 112 × 112 for C7 and 128 × 128 for D4. Dense eigendecompositions/type-restricted spectra dominate remaining computation; the family moment optimization removes repeated coordinate IO and repeated construction over all 126 context blocks, but does not remove this image/deletion workload.

Float64 family moment storage is `64*p*p*16*16*8` bytes per active cell: 6,422,528 bytes for C7 and 2,097,152 bytes for D4, plus coordinate arrays, group matrices and temporary eigensolver storage. Passed/failing numerical checks are retained in JSON and can make the result file much larger than these arrays. Allocate a fresh writable output destination with sufficient disk space for all detailed records. No wall-clock estimate is claimed: no real-family benchmark or full execution was run.

Default metadata-only invocation from the revision project:

```sh
python analysis/rank_sensitivity_v1/sensitivity.py \
  --archive /path/to/tmlr-all-v2.20261030T203725Z-5489fd.results.zip \
  --output /path/to/new-metadata-audit
```

The shipped `config.json` has `full_run: false`. Even supplying `--execute-family-sensitivity` alone leaves the run at preparation. A future explicit authorization is required before creating a separate configuration with `full_run: true` and passing it with `--config`, together with `--execute-family-sensitivity` and a new output directory. The other frozen fields must stay unchanged. The current revision does not perform either execution opt-in.

Synthetic checks:

```sh
python analysis/rank_sensitivity_v1/test_sensitivity.py
```

No frozen study file, coordinate bank, release metadata or manuscript was modified by task D.
