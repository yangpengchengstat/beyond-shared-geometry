# Reproduce the v0.5.8 presentation from saved results

This editorial release executes no scientific experiment. The completed orthogonal run's eight numerical fixtures and 96 units remain prior evidence, not new replications. Raw coordinates and original outcomes are unchanged.

## Main Table 2: saved same-metric aggregates

From the project root, choose a fresh output filename:

```sh
python3 paper/supplementary/orthogonal_comparator/generate_table.py \
  --inputs paper/supplementary/orthogonal_comparator \
  --output /tmp/core-decomposition-v058-check.tex
cmp paper/generated/core_risk_decomposition.tex /tmp/core-decomposition-v058-check.tex
```

The formatter selects and rounds already saved aggregates. It does not refit matrices, recompute spectra or average cell rows. Per-cell clipping and original equal-cell means are inherited from the completed run. The full-precision source records, 96 rows, hashes and mappings remain in the supplement and results/orthogonal_comparator_v1/.

## Main Table 5: compact view of the completed 8/16 analysis

```sh
python3 paper/supplementary/maintext_revision/format_sensitivity.py \
  --coverage paper/supplementary/maintext_revision/spectral_dimension_coverage.json \
  --output /tmp/spectral-sensitivity-v058-check.tex
cmp paper/generated/spectral_sensitivity_main.tex /tmp/spectral-sensitivity-v058-check.tex
```

This selects saved gamma means, joint-premium medians and unique-leading counts. It performs no averaging or matrix calculation. The original full sensitivity table, cell records and metric distinctions remain unchanged. Table 3's new source file witness_transfer_main.tex differs from followup_witness_bounded.tex only by redirecting one caption reference to the current main core table; its source JSON records the exact transformation.

## Concept figure

```sh
python3 paper/supplementary/maintext_revision/generate_concept_figure.py \
  --output /tmp/evaluation-targets-v058-check.pdf
```

ReportLab is required. The vector diagram consumes no empirical input. INPUT_MAPPING.json and EVIDENCE_SHA256.json identify the portable scripts, their project mirrors and copied inputs. These formatting checks were run for this revision; no numerical test suite or analysis pipeline was run.

## Build, preserved records and packaging

The manuscript entry is paper/main.tex. Its source closure includes all sections, generated tables, figures, references and the TMLR style. COMPILE_CHECK.md records the actual export and independent Overleaf-package compilation checks. Existing review_reproduction/ and the spectral-dimension/orthogonal supplements retain their full-precision results and original reproduction tools; their presence does not imply rerunning them in this task.

The immutable studies/ preparation protocols keep their original source identities. Current README, release metadata and saved coverage records distinguish prepared snapshots from completed executions. LOFO is still prepared and unexecuted.

Original large archives remain at ../../bsg_latest_complete_20261106/research_archive, and the full orthogonal delivery remains at ../../bsg_orthogonal_comparator_20261106_v1, relative to this public project. The public/GitHub package contains code and selected traceable evidence, not model weights or every historical activation array. No GitHub, Overleaf or journal upload was performed.
