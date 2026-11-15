# Review of completed TMLR follow-up results

Date: 30 October 2026  
Release: 0.5.1  
Evidence status: reviewed descriptive integration; final author review pending

This revision reviews and integrates the returned completed suite. It does not
rerun model extraction, fitting, evaluation, simulations or bootstrap sampling.
Compact public outcomes and their derivation records are in
`results/tmlr_followup_v2`; `scripts/build_tmlr_results.py` generates manuscript
assets from those recorded results. Original scientific source/protocol locks,
historical measurements, stopped branches and raw output archives remain
preserved.

The source result archive is `tmlr-all-v2.20261030T203725Z-5489fd.results.zip`,
SHA-256 `ef87e9aaeb8f093ba0c2438590d6a30b5dffcd256993e2366bd600cca3cb5bbb`.
It is a private evidence archive, not part of the public code or Overleaf package.

## Completion and findings

| Analysis | Complete declared scope | Reviewed interpretation |
| --- | --- | --- |
| Finite-bank spectra | 48 model/readout/domain cells | Exact-class risk is bounded in the bank's second-moment metric; large trivial-type contributions limit semantic interpretation. |
| Constructed exact actions | 48 witnesses | Numerical action-law and same-training-bank bound checks pass; frozen old-test witness losses exceed OLS in every cell. |
| Alternative image subgroups | 2,952 units: 120 C7 or three D4 images in each cell | Semantic-image rank varies and alternative actions remain competitive; ranks are descriptive, not p-values. |
| Controlled mechanisms | 36 settings | Shared-action and stable-state-code constructions both yield near-exact noiseless recovery. This is not a semantic classifier or power calibration. |
| New-context transport | 48 cells over 64 paired authored families | The primary average is close to zero with opposing C7/D4 contrasts; general superiority of difference fitting is not supported. |

### Spectral and witness results

Across the 24 cells in each domain, mean gamma is 0.985102 for C7 and 0.935278
for D4. Mean selected trivial energy is 0.615404 and 0.727424, respectively;
these are contributions to total normalized energy, not fractions of gamma.
The mean analytic lower/upper endpoints are [0.029796, 0.058633] for C7 and
[0.129443, 0.248946] for D4. These averages summarize separate cell-specific
bounds; they do not describe one pooled action or a confidence interval.

The normalized type-budget correction is at most 0.008272 on C7 and is at
floating-point scale on D4. Raw values, including tiny signed rounding residuals,
are retained. Thus the theoretical type-sensitive separation does not imply a
large observed type correction at rank 16. A high gamma establishes compatibility
with a low-risk exact orthogonal action on the declared bank, not semantic
specificity or native causal computation.

Constructed actions use training whitening and frozen operators on the original
test bank. Mean test losses are 0.028059 versus OLS 0.021408 for C7, and 0.111172
versus OLS 0.088779 for D4. The witnesses lose to OLS in all 48 cells. Their losses
are attained feasible risks, not the exact optimum of the constrained class.
The frozen-metric test losses must not be compared with test-bank bounds computed
using a newly defined test whitening metric. The extension's fixed-alpha ridge
is OLS shrunk by approximately 1/1.001 in the training-whitened coordinates; it is
not a broad regularization comparison.

The semantic image ranks uniquely first by gamma in 7/24 C7 and 16/24 D4 cells.
Mean best-minus-semantic gamma gaps are 0.0000316 and 0.0008668. Ranks, gaps and
ties must be read together: each image defines different target correspondences,
and a gamma ranking is not necessarily a ranking of exact optimum risk or frozen
test prediction. No p-value or semantic uniqueness claim is made.

The three mechanism constructions are illustrative. M1 has a shared action; M2
uses varying contextual orientations; M3 combines stable state coding with
context features. M3 admits an exact shared action before noise, so it is not a
negative null. The sampled M2 construction, rather than its name, determines
whether sharing is possible. All 36 settings are retained without claiming
false-positive calibration or population power.

### New-context transport results

The primary effect is difference-fitted minus coordinate-fitted transport loss;
negative favors difference fitting. Both arms select action-wise regularization
using the same old-validation difference endpoint and fixed grid. New contexts
are used for neither fitting nor selection. Losses use the old-training context
difference energy and are distinct from the spectral metric.

| Aggregate | Effect | Recorded 95% family-resampling sensitivity range |
| --- | ---: | ---: |
| Equal-domain primary panel | +0.000178 | [-0.001035, +0.001362] |
| C7 | -0.005833 | [-0.006680, -0.005008] |
| D4 | +0.006190 | [+0.003970, +0.008406] |

The primary result does not support a general advantage or equivalence claim.
C7 favors difference fitting on average, while D4 favors coordinate fitting.
On C7, the identity reference has mean loss 0.043571, below both fitted means
(difference 0.044479; coordinate 0.050312); the favorable pairwise contrast must
not be described as improvement over every baseline. Fixed cell counts are
descriptive and are not independent Bernoulli trials.

The 2,000 recorded bootstrap draws resample 64 paired authored family units while
keeping variants, states, actions, models, readouts and domains together. These
percentile ranges describe sensitivity conditional on the authored bank and
fixed panel. They do not establish population coverage, account for refitting
uncertainty, or implement the theoretical finite-context certificate.

## Context wording and provenance limits

The frozen protocol describes an outer-context change around a fixed factual
clause. The executed new bank fixes its clauses internally, but the old and new
wordings differ:

| Domain | Historical clause | New clause |
| --- | --- | --- |
| C7 | “the designated weekday is [state]” | “The scheduled weekday is [state]” |
| D4 | “the marked corner is [state]” | “The marked square corner is [state]” |

These strings are recorded in `src/bsg_next/data.py` (also the retained
characterization source) and `studies/context_generalization_v1/bank.py`.
The revision corrects the reporting, preserves the frozen protocol, and does not
alter prompts or rerun the study. The evidence jointly tests new outer frames
and limited factual wording changes; it does not isolate their separate effects.
State vocabulary is unchanged. At a causal state-token readout, suffix text
after the target cannot affect the representation. This is not a test of arbitrary
new grammars or independently sampled human prose.

The source archive preserves an operator attestation for review of all 64 paired
frames per domain before extraction. It is an unauthenticated attestation, not
proof of independent human authorship or final manuscript approval. The review
verified source/model/PCA/row bindings and recorded outcome consistency; it did
not infer independent confirmation from the execution receipts.

The prior input compatibility repair recovers full prompt text by exact row-ID
join from the archived datasets and explicitly handles the `D4_corner` row
domain. Its before-copies, change record and original failure inventory remain
under `repair_v2`. Those files describe repair-time validation, not the later
scientific results; their historical false/pending flags remain unchanged.

## Remaining scope

All five analyses are completed and integrated as descriptive evidence. The
historical calibration/native failures are not overturned. No semantic mechanism,
population exact-action certificate or general transport superiority is claimed.
Final author review of the full paper, attribution, anonymity, bibliography and
submission metadata remains outstanding; `submission_ready` stays false.
