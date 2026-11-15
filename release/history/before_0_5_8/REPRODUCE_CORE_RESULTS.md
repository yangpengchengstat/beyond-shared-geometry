# Reproduce the 0.5.7 presentation from saved results

The orthogonal comparator previously completed 8 numerical fixtures and 96 bank/dimension conditions. This integration does not repeat those tests or calculations, the existing spectral analyses, model inference, witness training, historical fitting, resampling or LOFO.

## Current Table 1: two panels, existing aggregates only

From the project root:

```sh
python3 paper/supplementary/orthogonal_comparator/generate_table.py \
  --inputs paper/supplementary/orthogonal_comparator \
  --output /tmp/core-decomposition-check.tex
cmp paper/generated/core_risk_decomposition.tex /tmp/core-decomposition-check.tex
```

This standard-library program selects and rounds already saved rows. Panel (a) reads the unchanged original domain-risk CSV. Panel (b) reads the completed coverage record, including means computed from per-cell clipped endpoints. It does not subtract domain means, refit matrices, recompute spectra or average cell CSV rows. The formatter was executed for this revision; it is not a numerical experiment. Input mappings, full-precision rows and checksums accompany it. The prior core-table source and its generator are preserved, but `paper/main.tex` now inputs `core_risk_decomposition.tex`.

## Unchanged tables and numerical source

Tables 3–6 retain the original saved-endpoint capsule at `review_reproduction/`; Appendix K/Table 32 retains `paper/supplementary/spectral_dimension_sensitivity/`. Their original source and outcomes are preserved; this revision did not rerun either endpoint pipeline. The new Appendix M describes the completed comparator contract and limitations.

`results/orthogonal_comparator_v1/` contains the copied coverage, 96 full-precision rows, all 96 unit summaries, saved heterogeneity and zero cases, input bindings and recorded checks. `SOURCE_BINDINGS.json` maps each imported file to the completed run with its original hash. `original_output_manifest.json` describes the full external execution archive, including diagnostic NPZ files not duplicated in this public subset; it is not a manifest of this selected subset.

## Execution state versus frozen preparation snapshots

The immutable `studies/orthogonal_comparator_v1/README.md`, protocol and `status` subcommand describe the original preparation snapshot. They were not edited after execution because their identities are source-locked. The active `README.md`, release metadata and copied `coverage.json` record completed execution. No new run is needed to compile or inspect the current paper.

Original archives remain at `../../bsg_latest_complete_20261106/research_archive` relative to this project root. The complete comparator result delivery remains at `../../bsg_orthogonal_comparator_20261106_v1`. Keep these external preservation directories. The public source includes code and selected evidence, not model weights or full historical activation archives.

## Build and package

The manuscript entry is `paper/main.tex`. The built-in compiler and existing Tectonic installation are used for this revision; the official TMLR style remains unchanged. Final checks and retained warnings are in `COMPILE_CHECK.md`. The source package and Overleaf package include the new saved-result supplement. Building them does not execute studies or constitute author approval or submission.
