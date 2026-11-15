# Project map

This is the independent 0.6.2 TMLR delivery, based on the retained 0.6.1 manuscript.
Previous papers, release metadata and packages are preserved under
`release/history/before_0_6_2`; original research archives remain unchanged.

## Active manuscript and studies

- `paper/main.tex`: anonymous TMLR manuscript entry point.
- `paper/sections/introduction_main.tex`, `related_work_main.tex`, `setup_main.tex`,
  `risk_main.tex`, `experiments_restructured.tex`, `discussion_main.tex`: current
  main-text structure; the current validation records retained evidence references.
- `paper/sections/retained_summaries.tex`: complete relocated summary tables.
- `paper/figures/evaluation_targets.pdf`: conceptual illustration, not new data.
- `paper/generated/spectral_sensitivity_main.tex`: compact display of existing
  sensitivity endpoints; its source mapping records the preserved inputs.
- `paper/supplementary/maintext_revision/`: presentation-only generators and
  input mappings, without numerical re-execution.
- `paper/sections/orthogonal_comparator.tex`: independent-orthogonality comparison
  and its numerical and interpretive limits.
- `paper/supplementary/orthogonal_comparator/`: saved full-precision inputs,
  explicit mappings and table-formatting code.
- `EDITORIAL_NOTES.md`: current completed family-sensitivity integration and preserved evidence.
- `paper/sections/spectral_risk.tex`: definitions, risk bound, proof and scope.
- `studies/spectral_finite_bank_v1/`: frozen-cache spectral analysis, CPU only.
- `studies/context_generalization_v1/`: new paired-context extraction and transfer
  evaluation, explicit opt-in model execution and human bank review.

- `studies/spectral_extensions_v1/`: exact-action witnesses, complete image-subgroup
  comparisons and controlled simulations; CPU-only, explicit execution.

- `studies/spectral_dimension_sensitivity_v1/`: preserved 8/16 offline sensitivity.
- `studies/orthogonal_comparator_v1/`: executed comparator source; its frozen
  preparation-time documentation is preserved. Current status is in the receipts.

- `studies/c7_family_influence_v1/`: source of the already completed C7 family-cost sensitivity; this release does not rerun it.
- The original registration LOFO remains prepared and unexecuted.

## Retained evidence

- `results/c7_family_influence_v1/`: preserved scalar outputs and source mappings for all 1,536 completed family-cost sensitivity units.

- `results/orthogonal_comparator_v1/`: all 96 completed 8/16 summaries, coverage,
  original input bindings and previously passed test records.
- `results/spectral_dimension_sensitivity_v1/`: completed offline sensitivity.
- `results/tmlr_followup_v2/`: all five completed follow-up studies.
- `results/frozen/`: immutable historical evidence.
- `results/characterization_review/` (when present): audited fixed-bank summaries.
- `results/reproduction_v1_review/`: returned study summaries and scientific stops.
- `results/reproduction_action_review/`: per-action endpoints and identities.
- `paper/generated/`, `paper/figures/`: existing traceable numerical assets.
- `release/artifact_manifest.json`: source identities for published evidence.

The records retain original calibration failure, infeasible rank-16 lexical
control, native clean-competence stops and limited structural power. A technical
completion is not a scientific pass or independent confirmation.

## Historical reproducibility, not the default new-run path

`src/bsg`, `src/bsg_next`, `configs`, the earlier studies, older RunPod launchers
and dated documentation are retained. Read their own scope before using them.
They intentionally preserve historical names, dates and gate semantics. A TMLR
packaging revision does not rewrite an old study's identity or retrospectively
convert inspected observations into an untouched test set.

For an exact old-source reconstruction, use `scripts/restore_frozen_source.py`
in a new directory before following its own reproduction README. Do not launch a
legacy source-locked study directly from modified publication metadata. This
source-only command never launches the original full queue.

The complete private coordinate archive and all raw activations are not copied
into the public project. The cache-based study checks exact input identities
before use. Rebuilding these inputs would be a separate experiment request.
