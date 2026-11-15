# C7 complete-family deletion influence, v1

This separate, user-authorized offline analysis extends the v0.6.1 project without altering its manuscript, original results, source locks, or failure records. The protocol is fixed after inspecting the original complete-bank results and before any real deletion result. It is post hoc sensitivity analysis, not preregistration or confirmation.

## Question and complete panel

Does the further common-group-law risk lower bound in the original Study 2 C7 test bank depend strongly on a single context family? The known 99.44% concentration is across model/readout cells, not across context families and not a share of the unknown true cost.

Only canonical C7, r=16, all six original models and four original readouts (24 cells) are included. Every cell undergoes every one of the same 64 family deletions: 1,536 units. Each omitted family removes both variants and all seven states. The remaining 63 families, 126 contexts, seven states, and seven actions have uniform weights, including identity. No alternative-image enumeration, D4, r=8, bootstrap, new models, or hyperparameter selection is included.

## Target and numerical rules

For every reduced bank, use its own uncentered second moment and whitening, with original saved training PCA coordinates and mean unchanged. Consequently, deletion changes both the empirical distribution and its induced metric (and orthogonal class in the original coordinates). These are not additive family contributions under a fixed common metric. No supplementary centering is performed.

The original frozen spectral kernel and independent-orthogonal kernel are called directly on each reduced array; there is no sufficient-statistic subtraction shortcut. Their full diagnostics are retained. Report eta, gamma, L_orth, beta=L_orth-eta, class bounds, p-minus/plus, ell-minus/plus, identity reference, type correction and moment conditioning. The only clipping is the mathematical max(0, ...) definition of lower endpoints; no spectra, eta, gamma, upper endpoints or failed banks are repaired. Zero endpoints are inconclusive, not evidence of zero true cost.

The historical checks remain: strict smallest/largest second-moment eigenvalue ratio greater than 1e-10; normalized Frobenius tolerance 1e-8; scalar tolerance 1e-8 + 1e-7 max(abs(a), abs(b)); frozen spectral tolerance 1e-8. The positivity count uses ell-minus > 1e-8 + 1e-7 abs(ell-minus), exactly as the completed comparator. These are software tolerances, not rigorous floating-point certificates. No jitter, pseudoinverse, rank reduction, PCA refit or historical predictor/witness retraining occurs. The kernels compute the independent linear and independent orthogonal optima and the spectral budget by their declared formulas; L-star remains bounded, not measured exactly. This is not model training.

## Gates, identity and failures

1. Freeze new source, protocol and dependencies. Synthetic tests must pass without skips, in the same Python/NumPy runtime as execution. Their receipt binds source hashes; changing tested code invalidates the gate.
2. Verify both complete original archives and their pinned members; bind original PCA, rows, model revisions, family IDs and follow-up snapshot. Snapshot test coordinates must be bitwise equal to the original archive selection.
3. Compute all 24 full-bank baselines and compare with saved spectral and orthogonal results under historical tolerances before any deletion starts. These are reproduction checks, not replacements for the saved main results.
4. Execute every planned deletion. Preserve failures with the planned denominator. Failed/incomplete cells are not substituted, and a family-panel mean is withheld unless all 24 cells succeed. Outputs are create-only; no overwrite/resume or success-only filtering.
5. Verify original archives and source hashes again. Retain original source project and current Git changes unchanged.

The independent synthetic checks include analytic scalar C7 rotation, exact r16 actions, a Fourier-based budget oracle, direct least-squares/Procrustes risks, family pairing and invalid metadata, input preservation, uncentered moments, invariance, and deficient/rank-threshold cases. Numerical fixtures are not new model experiments.

## Predeclared summaries

For every cell and every metric, record full-bank value and deletion minimum/maximum, their family IDs and absolute changes; include all 64 cases. Report per-cell positive-lower counts (denominator 64), failure counts and smallest moment ratio. For each common family deletion, aggregate only the complete 24-cell panel with equal cell weight: mean risks/bounds, median and range of lower endpoints, positive count, largest contributor identity/share and top-two share. Summaries of lower bounds do not measure the distribution of true class costs. Do not label family counts as probabilities or uncertainty intervals.

## Commands

From the copied project, with an existing CPU Python providing NumPy:

```sh
PYTHONDONTWRITEBYTECODE=1 python studies/c7_family_influence_v1/run.py test --output /path/to/fresh-tests
PYTHONDONTWRITEBYTECODE=1 python studies/c7_family_influence_v1/run.py execute \
  --repo /path/to/original-repository-with-incoming-archives \
  --tests /path/to/fresh-tests --output /path/to/fresh-results --execute-offline
```

Output contains detailed per-unit JSON, baseline/deletion CSV, cell and paired-family summaries, original input bindings, source and protocol identities, runtime/test receipts, and before/after preservation records. Review outcomes before deciding on any manuscript integration. The older registration LOFO remains unexecuted and distinct from this analysis.
