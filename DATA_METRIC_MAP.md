# Data and metric map — v0.5.8

All numbers below are inherited saved aggregates; no scientific calculation was performed in this editorial revision. The full v0.5.7 map is preserved in release/history/before_0_5_8/. Labels are the stable navigation keys. Current main tables: 1 metric contracts, 2 same-bank class costs, 3 frozen-witness transfer, 4 registration margins, 5 compact existing dimension sensitivity, 6 context transport. The original spectral and joint-premium summaries are retained in Appendix N. Full dimension and orthogonal records remain in Appendices K and M.

## Three distinct contracts

| Contract | Coordinates and bank | Whitening / weights | Object reported | Safe comparison and source |
|---|---|---|---|---|
| A: descriptive finite-bank exact-class analysis | Original Study 2 old test bank, fixed training-derived PCA coordinates, 48 model/readout/domain cells at r=16; each bank has 128 complete state blocks from 64 paired families | The evaluated bank's own **uncentered second moment**. Uniform contexts, states, and all group elements, including identity and fixed points. No new mean subtraction | Same-bank independent linear optimum eta, exact orthogonal-class lower/upper bounds, joint class-premium bounds, identity risk, type energy | Compare quantities within this row only. `results/main30_bounded_v1/domain_risk_reference.csv:2`; per-cell provenance in `cell_risk_reference.csv`; original spectral exports under `results/tmlr_followup_v2/spectral/`. Paper labels `tab:followup-spectral`, `tab:bounded-risk`, `app:bounded-risk-details` |
| B: construction-bank witness check | Original Study 2 training bank, same saved training PCA; 48 fixed conditions | Training-bank second moment; same full state/group averaging | Constructed exact action's attained training risk; training-bank lower/upper bounds; recorded law, orthogonality and bound checks | The implementation checks its specific theorem-based constructor on its construction bank. Mere feasibility does not imply every feasible action lies below the theorem's existential upper bound. `studies/spectral_extensions_v1/numerics.py:224`; saved `witness_units.csv` and `witness_domain_summary.csv` |
| C: frozen fitted-predictor transfer | Original old test bank; maps fitted on training bank | **Training whitening remains fixed**, as do witness/OLS/ridge maps | Test loss of each particular predictor | Empirical held-out comparison, not a class-optimum comparison. Test second moment in this metric is not assumed identity. `results/tmlr_followup_v2/spectral/witness_units.csv`, `witness_domain_summary.csv`; label `tab:followup-witness` |
| Registration comparison | Same bank, coordinates and metric as A; each candidate supplies its own transformed targets | Full grid, original pairwise tolerance | Budget margin/rank among all 120 C7 or three D4 permutation-image subgroups | Two alternative D4 images after excluding the specified image. Not p-values or L-star rankings. `registration_margin_summary.csv`; label `tab:bounded-margins` |
| Existing 8/16 sensitivity | Same original Study 2 banks; eight coordinates are the first eight recorded training-PCA coordinates | Each dimension has **its own** second-moment whitening | Descriptive changes in budget, premium endpoints, type effect and image rank | No cross-dimension common-loss superiority claim. Original r=16 remains primary. No witness transfer/context transfer rerun. `results/spectral_dimension_sensitivity_v1/`; label `tab:spectral-dimension-sensitivity` |

Training-PCA centering and bank-specific second-moment whitening are separate operations. A general change of whitening preserves group laws by conjugacy but can destroy orthogonality in the destination Euclidean metric; it does not make B/C numerically comparable with A. Study 1's rowwise target-energy-normalized D scores and Study 2's training-energy-normalized coordinate/context errors are separate endpoints, not entries in an A/B/C risk ranking.

## Core Table 2(a): retained saved aggregate values

The first panel of the core table places the scale of A's quantities together. Each entry is an equal-cell mean over the same 24 conditions in its domain. Brackets are means of deterministic cell-wise bound endpoints, **not confidence intervals**, exact optimal risks, or achieved predictor losses.

| Domain | Mean eta | Mean bounds [LB, UB] for L-star | Mean bounds [p-minus, p-plus] for L-star minus eta | Mean L(I) |
|---|---:|---:|---:|---:|
| C7 | 0.02563 | [0.02980, 0.05863] | [0.00416, 0.03300] | 0.76012 |
| D4 | 0.10321 | [0.12944, 0.24895] | [0.02624, 0.14574] | 0.52932 |

These strings already occur in `paper/generated/followup_spectral_bounded.tex:6`, `paper/generated/bounded_risk_reference_table.tex:5`, and `paper/generated/bounded_macros.tex:2`. If new convenience macros are introduced, copy the saved aggregates and preserve the existing display policy; do not derive a new formal table from rounded PDF values.

Exact saved values from `results/main30_bounded_v1/domain_risk_reference.csv:2` and `:3`:

| Domain | eta | LB | UB | p-minus | p-plus | L(I) |
|---|---:|---:|---:|---:|---:|---:|
| C7 | 0.025631850779367624 | 0.02979627746026327 | 0.058633073953772485 | 0.004164426680895648 | 0.03300122317440486 | 0.7601247570330282 |
| D4 | 0.10320627367664041 | 0.12944321597922318 | 0.248945749051769 | 0.026236942302582777 | 0.1457394753751286 | 0.5293214369253398 |

The joint premium constrains exact group laws **and** orthogonality. A small lower endpoint does not establish a small true premium. This revision should not add a ratio-of-means, mean-of-ratios, practical-equivalence margin, or relative-risk claim absent a saved, explicitly defined endpoint.

## Existing heterogeneity summary

Source: `results/final_edit_v1/premium_lower_summary.json:19` (C7) and `:27` (D4). These are the earlier exact-decimal summaries of the original cell CSV, not newly recomputed values.

| Domain | Saved median p-minus | Saved minimum | Saved maximum |
|---|---:|---:|---:|
| C7 | 0.00038929152944697925 | 0.000034867499308447944 | 0.06254392518998934 |
| D4 | 0.0187559524517441415 | 0.002538358344475869 | 0.08718769943343879 |

C7's two largest cells are Llama-3.1-8B/layer 13 and Llama-3.2-3B/layer 11. Their previously saved share of the sum across all 24 lower endpoints is 90.68064875787752% (display 90.7%). All conditions and the original equal-cell mean remain. The original threshold is `1e-8 + 1e-7*abs(p_minus)`; exceeding it is a software comparison, not a rigorous propagated error certificate or a significance statement.

## Construction and transfer evidence already available

`results/tmlr_followup_v2/spectral/witness_units.csv:1` explicitly names `train_L_lower`, `train_L_upper`, `train_witness_risk`, `train_bound_passed`, test losses, and source summary/hash. `witness_domain_summary.csv` already supplies the following summaries:

| Saved endpoint | C7 | D4 |
|---|---:|---:|
| Mean construction-bank witness risk | 0.029062747092058493 | 0.11996171673744503 |
| Minimum witness risk minus construction lower bound | 4.627452179772895e-08 | 4.1325225012822653e-07 |
| Minimum construction upper bound minus witness risk | 0.005097231688347054 | 0.04723449680932735 |
| Mean frozen-witness old-test risk | 0.02805853636113307 | 0.11117161935740612 |
| Mean frozen-OLS old-test risk | 0.021407731153662048 | 0.0887786707411905 |
| Mean frozen-ridge old-test risk | 0.021406334337141272 | 0.0887771220061726 |

The lower/upper-slack rows establish that a same-bank construction check was already recorded. They do not provide an independent replay or interval-arithmetic proof. The 24/24 test-loss comparisons against OLS are empirical properties of these frozen fitted predictors. Population/function-class inclusion alone does not force this held-out ordering, and this constructor's ordering does not rank the whole exact-action class on test data.

## Existing registration and dimension records

`results/main30_bounded_v1/registration_margin_summary.csv:2` records C7 median strongest-alternative margin `-5.5201760282841406e-06`, range `[-0.00030287169320897256, 1.9671060621395675e-05]`; D4 (`:3`) median `5.955768246357351e-06`, range `[-0.012330377109351254, 0.038213640997174836]`. The original pairwise rule is `1e-8 + 1e-7*max(abs(a),abs(b))`. Keep signed margins and rank intervals; no permutation-null calibration exists.

The unchanged full dimension table (`paper/generated/spectral_dimension_summary.tex`, label `tab:spectral-dimension-sensitivity`) reports:

| Domain/r | Mean gamma | Mean p-minus | Median p-minus | Unique-leading image |
|---|---:|---:|---:|---:|
| C7/16 | 0.985102 | 0.004164 | 0.000389 | 7/24 |
| C7/8 | 0.933943 | 0.064969 | 0.041288 | 4/24 |
| D4/16 | 0.935278 | 0.026237 | 0.018756 | 16/24 |
| D4/8 | 0.959581 | 0.010235 | 0.005865 | 19/24 |

`results/spectral_dimension_sensitivity_v1/interpretation_summary.json` records C7 type corrections above the original tolerance in one rank-16 condition and eight rank-8 conditions; D4 has none at either rank. The corresponding maximum normalized corrections are 0.008271936407181824/0.05172833603314386 for C7. This is a dimension-dependent empirical effect, not a reason to select a favorable dimension or change the primary analysis.

## Publication implementation and preserved scope

All original generated numerical files, I.7/I.8 and Appendix L remain unchanged. Automatic table numbers change with the reorganization. The completed independent-orthogonal analysis appears in Table 2(b), Section 5.3 and Appendix M. No old result is replaced and no experiment is rerun. LOFO remains prepared and unexecuted. The q=32 same-bank input gap is unchanged.

## Completed orthogonality comparator: Table 2(b) and Appendix M

`results/orthogonal_comparator_v1/coverage.json` supplies the four displayed mean rows directly. `review/domain_diagnostics.json` supplies existing medians, ranges and concentration; `cell_summary.csv` and 96 unit summaries retain full precision. The copied execution and numerical-test logs are `.txt` for release inclusion, with original `.log` paths and byte hashes in `SOURCE_BINDINGS.json`. No log is presented as a fresh run.

| r/domain | Mean L_orth | Mean beta | Mean further-law endpoints | Above original tolerance |
|---|---:|---:|---:|---:|
| 16/C7 | 0.028706138539504867 | 0.0030742877601372524 | [0.0010972473075670668, 0.029926935414265932] | 23/24 |
| 16/D4 | 0.12773711539307211 | 0.02453084171643166 | [0.0018417862444482702, 0.12120863365869687] | 23/24 |
| 8/C7 | 0.09671736976195285 | 0.029571351169869475 | [0.03539741217900925, 0.15184218588614895] | 22/24 |
| 8/D4 | 0.08081945437417146 | 0.010216729667878904 | [0.000052338847710833684, 0.07693009390044526] | 19/24 |

Each lower endpoint is max(0, L_lower - L_orth) **before** averaging, not a difference of domain means. True costs have an additive decomposition; clipped lower bounds generally do not. L_orth is the optimum over independently chosen O(r) maps on the same bank, not a frozen predictor's transfer loss. The full uniform group average includes identity and fixed points. Each dimension has its own uncentered second-moment whitening and normalization; r=8 uses the first eight saved PCA columns and is not a preferred replacement for r=16.

For r=16 C7, the further-law lower median is 5.345539812429481e-6 and range [0, 0.026187662085921326]; Llama-3.1-8B/layer 13 contributes 99.44454448767871% of the sum of all 24 lower endpoints. This does not estimate the concentration of unknown true costs and differs from the older joint-premium 90.7% two-cell concentration. The near-tolerance positive case at r=8 (Qwen2.5-3B/C7/layer 7, lower 1.5462046788761175e-8) and all nine zero cases are retained. Software tolerance is not a propagated roundoff certificate or significance threshold.

The portable supplement gives exact project input paths and hashes, all 96 rows, saved summaries and a formatter that only selects/rounds existing aggregates. It introduces no matrix computation or inference.
