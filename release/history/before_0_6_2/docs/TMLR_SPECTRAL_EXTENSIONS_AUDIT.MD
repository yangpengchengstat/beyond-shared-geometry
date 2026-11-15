# Audit of completed finite-bank spectral and extension results

Audit date: 2026-10-30. Evidence: `tmlr-all-v2.20261030T203725Z-5489fd.results.zip`. This report summarizes saved outputs only. No experiment, refit, rewhitening, bootstrap, or new null was run. The original archive and staged study sources were not modified.

## Completion and provenance

| Stage | Planned | Committed/evaluable | Not evaluable | Uncommitted attempts |
|---|---:|---:|---:|---:|
| Original-test spectral panel | 48 cells | 48 | 0 | Not an attempt-based stage |
| Train-fitted exact-action witnesses | 48 | 48 | 0 | 0 |
| Controlled simulation cases | 36 | 36 | 0 | 0 |
| Complete alternative image subgroups | 2,952 | 2,952 | 0 | 0 |

The audit verified every file listed in the four stage receipts (105 spectral, 317 witness, 166 simulation, and 11,934 alternatives files). It independently checked all 3,036 extension unit receipts, their plan/run-manifest bindings, terminal status, and committed output hashes. Aggregate numeric fields agree with the corresponding committed summaries, and saved semantic rank intervals agree with a descriptive recomputation using the frozen tolerance. The semantic alternative reproduces each original spectral gamma and witness/OLS/ridge result to within 1e-12. The staged source code and source manifests match all 16 source-lock files recorded by the witness run. Full-archive transfer verification is handled separately by the parent audit.

These are completed numerical endpoints, not scientific acceptance gates. Saved states explicitly retain `scientific_pass_assessed=false`, `independent_confirmation=false`, and `population_certificate=false`; `scientific_pass=false` is therefore not an assessed negative scientific decision. Source protocol/README phrases such as “prepared” describe the source release before this run, whereas committed receipts establish that this archived run executed.

## Numerical target and aggregation

For each complete bank, the core forms the **uncentered** second moment Sigma, whitens the rank-16 coordinates, and constructs the uniformly weighted complete state/action orbit Gram matrix K. Identity actions and fixed points are retained. C7 uses seven states and seven actions; the four-corner D4 action uses four states and eight actions. The typed budget C_r selects whole real irreducible copies with total dimension at most r=16; gamma=C_r/r. The reported correction is Delta_type=C_untyped-C_r, with normalized correction Delta_type/r. All 48 cells use all 16 selected dimensions.

The finite-bank optimal exact orthogonal-action risk is bounded by L_lower=2(1-gamma) and L_upper=2(1-gamma^2) in that same bank's empirical metric. Eta_spec=1-tr(K^2)/r equals same-bank, independently fitted action-wise OLS risk eta_LS. This OLS identity is verified numerically; it is not a separate successful prediction experiment. These quantities do not equal historical row-normalized Ridge/Joint scores.

The witness fits on all 96 original training context blocks and freezes the training whitening and action before evaluation on all 128 original test blocks. Its risk is squared prediction error averaged uniformly over context, state and action, divided by r. OLS and ridge(alpha=0.001) use the same frozen metric. The row-operator convention correctly checks rho(g)rho(h)=rho(hg); the reversed multiplication order is deliberate, not a group-law error. Bounds are checked on training risk only. The witness is a constructed feasible exact action, not a proved globally risk-minimizing action.

All means below are unweighted descriptive cell means within domain (six models times four layers = 24 cells). Ranges are cell minima/maxima. Cells and context blocks are not claimed independent replicates. No standard error, confidence interval, population statement, or significance test is inferred.

## Original-test spectral panel

| Domain | gamma mean [range] | L_lower mean [range] | L_upper mean [range] | Same-bank OLS eta mean [range] |
|---|---|---|---|---|
| C7 | 0.985102 [0.923826, 0.996749] | 0.029796 [0.006501, 0.152349] | 0.058633 [0.012982, 0.293092] | 0.025632 [0.006467, 0.089805] |
| D4 | 0.935278 [0.858516, 0.975848] | 0.129443 [0.048304, 0.282968] | 0.248946 [0.095442, 0.525901] | 0.103206 [0.045766, 0.195781] |

Median gamma is 0.989010 for C7 and 0.938058 for D4. The nonzero lower bounds exclude zero-risk exactness in these declared finite-bank metrics, subject to the numerical assumptions/checks; the upper bounds express finite-bank approximability, not native or semantic mechanism identification.

**The real-type correction is resolved in only one of 48 cells.** For Llama-3.1-8B, C7, layer 13, C_untyped=14.9135627171 and C_r=14.7812117345, giving Delta_type=0.1323509825 and Delta_type/r=0.0082719364. Its gamma is 0.9238257334 and risk interval [0.1523485332, 0.2930920286]. The selected allocation is ten trivial copies and one copy of each of the three real C7 planes. The other 47 corrections have absolute magnitude at most 5.33e-15 and must be described as numerically unresolved, preserving small negative roundoff values. It would be incorrect to claim a broadly active integrality correction across the panel.

Trivial-type total energy averages 0.619938 in C7 (range 0.608013–0.671395) and 0.735339 in D4 (0.596299–0.797349). Trivial selected energy averages 0.615404 and 0.727424, respectively. Thus much of the high budget is in the trivial component; this is a descriptive decomposition, not a post hoc removal or alternative endpoint.

All 37,848 saved core numerical checks pass. The maximum absolute eta_spec−eta_LS discrepancy is 5.00e-15. Second-moment condition numbers range from 11.49–228.41 (C7) and 16.06–615.24 (D4). Passing float64 residual checks is not rigorous interval arithmetic.

## Constructed witnesses versus predictive controls

| Domain | Mean train witness risk | Mean old-test witness risk [range] | Mean old-test OLS | Mean old-test ridge | Mean witness−OLS [median] |
|---|---:|---|---:|---:|---|
| C7 | 0.029063 | 0.028059 [0.005644, 0.185982] | 0.021408 | 0.021406 | 0.006651 [0.000248] |
| D4 | 0.119962 | 0.111172 [0.041731, 0.266695] | 0.088779 | 0.088777 | 0.022393 [0.011672] |

The constructed exact-action witness has greater old-test risk than both OLS and ridge in **all 48 cells**, beyond the frozen scalar tolerance. This is not evidence that every exact action must lose out of sample: only this construction was evaluated, and training-bank bounds do not certify test risk.

The C7 mean is affected strongly by Llama-3.1-8B layer 13: old-test witness 0.185982 versus OLS 0.080412 (gap 0.105571). Across C7 cells the absolute gap ranges 0.0000203–0.105571. D4 gaps range 0.001711–0.094385; the largest is Qwen2.5-1.5B layer 11 (witness 0.266695; OLS 0.172310). Medians are preferable alongside means to expose this heterogeneity.

All 48 training-bound checks pass. Minimum lower-bound slack (witness−L_lower) is 4.63e-8 in C7 and 4.13e-7 in D4; minimum upper-bound slack (L_upper−witness) is 0.005097 and 0.047234. All 4,296 saved witness construction checks and 37,848 corresponding training-spectral checks pass. Maximum normalized witness group-law residual is 3.80e-15 and maximum orthogonality residual 7.20e-15, compared with the declared matrix tolerance 1e-8. These are numerical construction checks, not statistical generalization guarantees.

## Semantic registration among complete alternative subgroups

Every cell is evaluated over the complete distinct image-subgroup panel: 120 C7 subgroups for each of 24 cells and three D4 subgroups for each of 24 cells. All 2,952 train-frozen witness endpoints and separately whitened old-test spectra are evaluable. All targets change consistently on the complete grid. Ranks sort old-test-bank gamma from largest to smallest; ties use atol=1e-8 and rtol=1e-7. No subgroup is removed for a failed outcome, and no p-values are computed.

| Domain | Semantic uniquely first | Other semantic ranks | Within-cell gamma spread, median [range] | Best gamma minus semantic, maximum |
|---|---:|---|---|---:|
| C7 | 7/24 | Remaining rank intervals range from 3–3 through 102–105 out of 120 | 3.88182e-5 [3.04620e-6, 4.06585e-4] | 3.02872e-4 |
| D4 | 16/24 | Six rank 2; one tied ranks 2–3; one rank 3 | 1.35637e-4 [1.47808e-6, 0.0398400] | 0.0123304 |

No cell has all alternatives tied. C7 high gamma is nonetheless broadly shared by alternative image subgroups; rank can vary considerably while absolute gamma differences remain very small. D4 shows more frequent semantic first place, but not uniformly, and there are only three candidates. These are descriptive comparisons between full alternative actions, not exchangeable null draws or calibrated evidence of semantic specificity. Counts of first places must not be turned into binomial or permutation significance tests. The full 48-cell tie-aware rank table is retained in `spectral_evidence/alternative_cells.csv` and the original rank JSON.

## Controlled illustrations

Each displayed row summarizes exactly three fixed seeds, 32 train and 32 test contexts, at rank 16. These are construction-specific demonstrations; they do not estimate power or false-positive rates.

| Domain | Mechanism | Noise | Mean test-bank gamma | Mean frozen-test witness risk | Mean frozen-test OLS risk |
|---|---|---:|---:|---:|---:|
| C7 | M1 | 0 | approximately 1 | approximately 0 | approximately 0 |
| C7 | M1 | 0.05 | 0.997196 | 0.006004 | 0.006019 |
| C7 | M2 | 0 | 0.575313 | 1.233137 | 0.874962 |
| C7 | M2 | 0.05 | 0.573704 | 1.236250 | 0.877212 |
| C7 | M3 | 0 | approximately 1 | approximately 0 | approximately 0 |
| C7 | M3 | 0.05 | 0.988587 | 0.026931 | 0.026526 |
| D4 | M1 | 0 | approximately 1 | approximately 0 | approximately 0 |
| D4 | M1 | 0.05 | 0.992392 | 0.021704 | 0.021616 |
| D4 | M2 | 0 | 0.453304 | 1.872778 | 1.097438 |
| D4 | M2 | 0.05 | 0.453971 | 1.879581 | 1.096580 |
| D4 | M3 | 0 | approximately 1 | approximately 0 | approximately 0 |
| D4 | M3 | 0.05 | 0.995132 | 0.013044 | 0.013039 |

M1 has a shared action before noise. M2 applies context-specific random orthogonal conjugations; its construction label is not itself a certified negative-null distribution. M3 is an oblique state/context direct sum with an exact shared action before noise and **must not be labeled a negative control**. Its near-unit noiseless gamma illustrates that state/context coding can satisfy the mathematical target. All 36 units preserve `mechanism_is_valid_negative_null=false`. The noiseless values are raw floating-point approximations and should not be interpreted as literally exact numerical zeros/ones.

## Compact evidence and manuscript-safe claims

`audit/summarize_spectral_extensions.py` is an executable extraction/verification recipe. It operates directly on the immutable ZIP, writes only the specified output directory, and calls no study execution code. `spectral_evidence/` includes:

- Byte-preserved original core cell/type CSVs, three extension report CSVs (including all 2,952 alternative rows), and descriptive-rank JSON.
- Compact derived per-cell and per-unit tables with exact archive-relative committed-summary paths and SHA-256 values; all 3,036 unit receipt/summary identities are also in `committed_unit_provenance.csv`.
- Domain tables, simulation tables, `metrics.json`, detailed verification `provenance.json`, and output `file_hashes.json`.
- The staged-versus-run source comparison is retained separately as `audit/spectral_source_verification.json`.

The original source-member hashes are in `provenance.json.aggregate_source_sha256`; all report and unit hashes are linked to verified stage receipts. These compact artifacts suffice to regenerate the reported descriptive aggregates without raw tensors or repeat fitting. They remain derived audit artifacts until selected and registered by the publication pipeline.

Suggested substantive interpretation: the finite-bank diagnostic is evaluable for all prespecified cells; C7 admits a smaller declared-metric exact-action risk interval than D4 on average; whole-real-type constraints change the ordinary top-r spectral budget materially in one cell; explicit training witnesses satisfy their numerical algebra/bounds but lose to OLS/ridge on this retrospective test comparison; high spectral budgets do not generally single out semantic registration, especially in C7. None of these results supplies new independent-context confirmation, a population exactness certificate, a causal mechanism claim, or proof of predictive optimality of the constructed witness.
