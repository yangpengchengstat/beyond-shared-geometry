# Reproducing the revised core presentation

This revision changes exposition and formats previously reviewed endpoints. It does not rerun model inference, spectral analyses, the 8/16 sensitivity, witness construction, historical operators, simulations, resampling or LOFO. The independent-orthogonal comparison is **PREPARED_NOT_RUN**; its numerical tests also remain unexecuted.

## 1. Latest core table: formatting existing aggregates

From the public project root:

```sh
python3 paper/supplementary/core_risk_summary/generate_table.py \
  --source paper/supplementary/core_risk_summary/domain_risk_reference.csv \
  --output /tmp/core-risk-table-check.tex
cmp paper/generated/core_risk_summary.tex /tmp/core-risk-table-check.tex
```

This standard-library program selects the already-saved domain rows and formats them to five decimal places. It performs no averaging, fitting, matrix operations or new inference. The CSV is byte-identical to `results/main30_bounded_v1/domain_risk_reference.csv`; hashes and the input mapping accompany the supplement. This formatting-and-comparison command was executed in the revision, and the generated table matched byte for byte.

The table reports equal-cell means of finite-bank class-risk bounds, premium bounds and identity references. Its intervals are deterministic bounds on the mean of cell-specific optima; they are not confidence intervals, frozen-estimator transfer losses, or measured optimal errors. Median and concentration statements are retained from `results/final_edit_v1/premium_lower_summary.json`, not newly estimated.

## 2. Existing Tables 3–6

The immutable `review_reproduction/` capsule retains its original input hashes and implementation. Its documented optional reproduction command is:

```sh
python3 review_reproduction/reproduce.py --output /tmp/fresh-review-tables
```

It reproduces saved-endpoint arithmetic and numerical table contents. Its implementation, inputs and expected outputs have not changed. This revision verified their preservation; it did not rerun that endpoint analysis. Table 4 uses frozen training whitening and training-fitted predictors. Tables 3, 5 and 6 describe the test bank in its own whitened metric. See `DATA_METRIC_MAP.md` before comparing their values.

## 3. Existing Appendix K / Table 32

The saved 8/16 analysis is complete and unchanged. To format its retained summary without repeating that analysis:

```sh
python3 paper/supplementary/spectral_dimension_sensitivity/generate_table.py \
  --coverage paper/supplementary/spectral_dimension_sensitivity/coverage.json \
  --output /tmp/fresh-dimension-table.tex
cmp paper/generated/spectral_dimension_summary.tex /tmp/fresh-dimension-table.tex
```

Use a new output path. The supplementary input map and checksums bind the 96 canonical-condition rows, 6000 complete units and 48 dimension comparisons. The prior numerical test/execution records remain part of the research archive; no repeated 8/16 analysis is needed to build this manuscript.

## 4. Independent-orthogonal comparator: preparation only

`studies/orthogonal_comparator_v1/` contains the new protocol, implementation, source/dependency manifest and planned test fixtures. It declares all 48 canonical conditions at dimensions 16 and 8, with the original state/group weights, whitening and tolerances. It requires explicit opt-in and a fresh output directory, and retains failed units. No comparator values, matrix files or test-pass claims are supplied as empirical evidence in this revision.

Read its README before any future execution. The complete local delivery retains the original archives in the sibling `bsg_latest_complete_20261106/research_archive/`. From this revised project root, the corresponding data-root argument is `../../bsg_latest_complete_20261106/research_archive`. Older source-locked dimension-study documentation describes the previous delivery layout; its numerical code accepts an explicit data root. The original archives and prior code identities remain intact.

## 5. Building the manuscript and source bundle

The LaTeX entry is `paper/main.tex`; the unchanged official TMLR style is included. `bash scripts/build_paper.sh` uses an installed LaTeX compiler. The current revision was compiled with the desktop LaTeX compiler and exported from the same main file using the existing Tectonic installation with `--only-cached --keep-logs --keep-intermediates`; no model or compiler-resource download was performed.

`scripts/package_release.py` packages the paper dependency closure, both compact evidence supplements and the public source. The anonymous saved-endpoint capsule remains separately identified. Building an artifact does not grant author approval or assert that optional analyses have been executed. Actual commands, compilation records, source diff and preservation checks accompany the independent revision directory.
