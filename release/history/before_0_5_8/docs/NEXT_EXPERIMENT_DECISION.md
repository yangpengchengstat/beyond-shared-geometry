## Current revision — 24 October 2026

Version 0.4.0 integrates the completed returned replay as reviewed descriptive evidence. Compact/context and secondary extension results are now available. Native competence still failed, structural power remains limited, and independent confirmation is blocked. Original frozen results and source capsules retain their identities.

For current instructions use the root README, `docs/REPRODUCTION_REVIEW_20261024.txt`, and `release/reproduction_review.json`. Rebuild tables offline with `python scripts/build_reproduction_assets.py`; no new calibration or full model replay is needed. Maintenance v2 defaults to checking only and requires a separate explicit choice before any execution. Prior dated instructions below describe earlier project states and should not be used to launch another run automatically.

## Preserved earlier documentation

# Next experiment decision after characterization v1

Status: design recommendation, not executed, not a frozen confirmation protocol. Existing characterization test outcomes have been inspected. Initial cached-coordinate analyses below must be labelled exploratory/development. A new frozen holdout is needed for later confirmatory claims. Preserve all old source, failures, results and eligibility flags.


Implementation update (21 October 2026): the cached compact/context code is now prepared in [compact_context_v1](../studies/compact_context_v1/README.md), with [engineering validation](COMPACT_CONTEXT_VALIDATION_2026_10_21.md). Full exploratory execution remains pending. The broader fresh-holdout/native/peer proposals below are not automatically authorized or completed.

## 1. Resolve the compactness comparison before expanding GPUs

Use the same existing q16 coordinates, train/validation roles, six-model/four-readout panel, changed-state masks and normalization. Fit every method only on construction data and select hyperparameters only on validation. Existing inspected tests may illustrate exploratory patterns but must not select models, budgets, grids or headline endpoints.

Compare: independent ridge; low-rank/affine ridge with charged biases; decoded-centroid displacement with an explicitly supplied action table; the existing limited-support kernel; and **at least one nonlinear predictor fitted on all target pairs at each storage budget**. A random-Fourier-feature residual regressor and a Nyström feature regressor are concrete first choices. For each primitive, fit `f_g(z)=z+phi(z) W_g` to all construction displacement pairs. Keep action supervision primitive-only in the generator track. Validation selects bandwidth and regularization from a declared grid. A fixed-seed random map can be reconstructed from a seed, but report both serialized bytes and fully materialized inference coordinates; charge normalizers, biases, landmarks/frequencies, coefficients and action tables. No free hidden test labels or uncharged target-model test activations.

Report a fixed budget curve around the identity-pruned ridge baseline (for example 0.5, 1, 2, 4 and 8 times incremental coordinates), with parameter counts calculated before fitting. Some exact budgets may be infeasible because minimum feature/landmark overhead exceeds them; mark those unavailable instead of silently exceeding the budget. Match exposure to every training target pair in the full-data compressed arm. Report all methods at all feasible budgets rather than a test-selected winner.

Score direct changed-state error and unseen-composite error by repeatedly applying each primitive to its own predicted output. The decoded-centroid method must re-decode each intermediate output in this track; a one-decode known-composite lookup belongs in a separate additional-information track. Include cycles, discriminating noncommuting D4 endpoints, output norms, invalid/nonfinite outputs and complete fixed-panel heterogeneity. Use the original training energy, no cross-study normalization mixing.

Start with a deterministic CPU fixture, then one excluded development cell to measure time/memory. Set a measured wall-time budget before the full cached panel. This does not require a new model forward pass or automatic reuse/repetition of the old calibration.

## 2. Test the narrower contextual benefit

The reported nonlinear prediction advantage does not answer the context-difference question. Fit competitors to the same paired paraphrase-difference endpoint and use its own training energy. Include identity, validation-selected scalar retention, linear transport, category-conditioned retention and a full-data nonlinear context predictor. Define exactly which context/state information is available at prediction. Preserve S3 joint/context/entity and zero-shot/support separation.

Add consistently registered alternative actions under the same training/evaluation supervision. The state-code account also learns consistent alternatives, so an inconsistent relabeling null alone is insufficient. Match direct-prediction error and operator/output norms when investigating whether composition defects carry evidence beyond error propagation. These are targeted falsification analyses, not a universal semantic-identification certificate.

Stop broad expansion if the supposed benefit vanishes under these controls; report the boundary result. If a useful contextual benefit survives, freeze the selected estimator and narrow claim before collecting fresh independently authored context families, new lexical realizations or a third model lineage. Expand only the factor needed for that generalization claim.

## 3. Require a separate decision for native or peer studies

Native D4 competence failures and the C7 intervention diagnostics are preserved. Do not select clean-correct test items, change scoring after test inspection, or call natural donor patches learned-operator success. A causal branch first needs a competent interface on excluded development data, frozen sites/ranks/strength selection, zero-delta checks, natural donor controls and complete-answer/collateral metrics on fresh cases.

For peer methods, separate original-setting reproduction from matched-benchmark adaptations. Match intervention layer/token site, preserved complement, rank and tuning budget. Conditional means are not automatically a full factorial ANOVA reproduction. Source-model before/after activations in cross-model transport must be declared. RunPod execution should follow a short smoke test and measured throughput, not an unconditional all-branch launch.

## 4. Evidence needed before making a stronger claim

Retain raw/PCA caches and model/environment pins. Preselect a small subset for raw-to-coordinate/forward replay; do not choose it from favorable effect sizes. Independently replay selected potential/Joint optimizers only if their outputs are central. Keep an execution ledger recording source identity, split access, failed attempts and all selection decisions.

For a new confirmatory estimand, define sampling units, practical effect, multiplicity family and stopping rules before fresh outcomes. Use development variance and an appropriate targeted simulation/power assessment. The long old calibration should not be rerun automatically, and passing it would not validate a changed estimator/design. Synthetic data may test known truths or code; it must remain labelled synthetic and cannot substitute for real-model results.

Deliverables for the next authorized stage: a frozen protocol/config, tested cached-coordinate runner, all-method budget table, validation-only selection ledger, and exploratory results clearly separated from any later fresh evaluation. This document itself launches nothing.
