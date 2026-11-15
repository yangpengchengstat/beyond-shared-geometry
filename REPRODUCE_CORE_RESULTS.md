# Locate and inspect the retained results

This index uses LaTeX labels and endpoint names because table numbers may change
after compilation. The prior v0.5.8 instructions are preserved at
`release/history/before_completion_v3/REPRODUCE_CORE_RESULTS.md`; their numbering,
local paths and captions are historical.

| Current result | Portable evidence and explanation |
|---|---|
| Same-metric class risk and orthogonality decomposition (`tab:core-risk-summary`) | `paper/supplementary/core_risk_summary/`, `paper/supplementary/orthogonal_comparator/`, `results/orthogonal_comparator_v1/` |
| Witness transfer (`tab:followup-witness`) | `review_reproduction/data/`, `results/main30_bounded_v1/`; training-metric held-out losses, not test-bank class-risk bounds |
| Registration margins (`tab:bounded-margins`) | `review_reproduction/`, `results/main30_bounded_v1/`; signed spectral-budget comparisons, not p-values or exact-risk ranks |
| Existing 8/16-dimensional sensitivity (`tab:sensitivity-main`) | `paper/supplementary/spectral_dimension_sensitivity/`, `results/spectral_dimension_sensitivity_v1/` |
| New-context transport (`tab:followup-context`) | `results/tmlr_followup_v2/`, with its saved paired-family and validation-selection records |
| C7 canonical-action family-risk influence (`tab:c7-family-influence`) | `paper/supplementary/c7_family_influence_v1/`, `results/c7_family_influence_v1/`; re-whitening changes the metric after deletion |
| Training-rank audit and historical dimension subset | `results/verification_evidence_v3/dimension_rank/` and its parent README/manifest |
| Historical calibration and power decisions | `results/verification_evidence_v3/calibration/`; original failures and stops retained |
| Earlier structural, lexical, predictor and behavior comparisons | `results/derived/`, `results/frozen/`, `results/characterization_v1/`, `results/reproduction_action_review/`, `results/reproduction_v1_review/` |

## Safe verification

From the project root:

```sh
python3 -B scripts/verify_delivery.py
```

This checks file identities, source syntax and static manuscript references. It
does not regenerate tables, refit predictors, compute spectra or execute experiments.
The source and result manifests specify what each frozen record binds.

## Optional saved-endpoint arithmetic

`review_reproduction/README.md` documents a separate, standard-library-only capsule
that can reproduce its saved-endpoint tables in a fresh output directory. Its
`table3`-`table6` filenames refer to the capsule's original numbering. Identify
current manuscript tables by endpoints and labels, not those filenames. The capsule
keeps the original full-precision inputs and tolerances, and does not rerun models,
whitening, spectral estimation, simulation, bootstrap or LOFO.

The existing supplement formatters likewise operate on specified saved records.
This delivery revises manuscript definitions, proofs, interpretation and presentation; byte-for-byte
comparison of an old formatter's entire TeX/PDF output with the new presentation is
therefore not an appropriate scientific equivalence test. Compare the preserved
numerical bodies and source bindings, and use the v3 presentation renderer for the
new figures. No saved formatter or original result is silently overwritten.

Canonical-action C7 family-risk LOFO and the independent orthogonality and 8/16
analyses are completed evidence. All-image registration LOFO is still unexecuted.
Newly supplied scalar audit records do not restore missing historical PCA arrays.

## Inputs needed for full computational reproduction

The `studies/` protocols and manifests specify original coordinates, model revisions,
normalizations and required archives. Large activation caches and model weights are
not included. `docs/DATA_AVAILABILITY_CURRENT.md` states the access limits; a checksum
identifies an input but does not provide it. Full reproduction must use the recorded
inputs and cannot be claimed from the static verification alone.

## Saved-coordinate centering diagnostic (v6)

Section 5.5 and Appendix I.9 compare original coordinates, within-template centering with original whitening, and centering with rewhitening. Commands: `studies/template_centering_v1/README.md`. Saved results: `results/template_centering_v1/`. Inputs and original member identities: `data/retained_coordinates/`. These inputs support this diagnostic without model downloads or activation extraction; they do not restore discarded hidden-space directions or historical missing arrays.
