# Beyond Shared Geometry: a TMLR research project

**Version 0.5.6 - integrated manuscript revision, 6 November 2026 (UTC).**

This project studies what recovered finite-group transformations establish about
language-model representations. It combines fixed-dimensional exact-action risk
bounds with preserved prediction, category-control, storage-budget and
context-transport results. The current manuscript is the integrated revision built from the retained dimension/rank evidence.

The manuscript uses the official anonymous TMLR template. **All five follow-up
analyses are complete and their returned results are integrated as descriptive
evidence. Final author review remains pending.** There are no pending analysis
slots; `submission_ready` remains false. Historical measurements, source locks
and scientific stops remain unchanged. The prior ICML project is preserved.

The results show high finite-bank exact-action compatibility with substantial
trivial-type contributions, competitive alternative registrations, and constructed
witnesses with higher old-test loss than OLS. New-context transport has opposing
C7/D4 effects and no overall advantage for difference fitting. See the
[results review](docs/TMLR_RESULTS_REVIEW.md) for numbers and interpretation limits.

## Latest integrated revision

The revised manuscript foregrounds the same-metric prediction/exact-action
comparison and the domain-dependent new-context transport result. A compact core
risk table reformats already saved full-precision endpoints; the previous broad
findings table and storage frontier remain in the appendix. The existing
16-dimensional primary results and completed 8/16 sensitivity are preserved.

The interpretation now separates independent linear prediction, independent
orthogonal prediction, exact orthogonal actions and preferential semantic
alignment. Positive regression error already forces an orthogonality cost; a
positive total premium alone does not isolate an additional group-law cost. No
new empirical independent-orthogonality result is claimed. Its complete optional
96-unit protocol, implementation and planned software tests are under
`studies/orthogonal_comparator_v1`, explicitly **PREPARED_NOT_RUN**. They are not
a requirement for completing this publication revision. LOFO remains unexecuted.

The root review records are `REVIEW_DISPOSITION.md`, `THEORY_CHECK.md`,
`DATA_METRIC_MAP.md`, `REPRODUCE_CORE_RESULTS.md`, `REVISION_REPORT.md` and
`PRESERVATION_REPORT.md`. They distinguish completed evidence, manuscript
changes, preparation-only code and the retained limitations. Final author review
remains pending; no journal or GitHub upload is performed by a local package build.

## Local data layout

This new project is placed beside the earlier complete delivery. From this
project root the immutable archived data root is:

```text
../../bsg_latest_complete_20261106/research_archive
```

For any optional reproduction, use this path as `--repo`. The byte-frozen
8/16-study README retains commands for the older delivery layout; its source
manifest must not be edited just to change that example. The new orthogonality
README uses the current layout explicitly. The original archives, failures and
run receipts remain external to this public source and are not replaced.

## Start here

| Purpose | Entry point | Runs models? |
| --- | --- | --- |
| Review revision and evidence scope | `REVISION_REPORT.md`, `DATA_METRIC_MAP.md` | No |
| Reproduce core Table 1 from saved endpoints | `REPRODUCE_CORE_RESULTS.md` | No |
| Inspect optional orthogonal-comparator code (not run) | `studies/orthogonal_comparator_v1/README.md` | No, unless explicitly executed |
| Read the TMLR manuscript | `paper/main.pdf`, `paper/main.tex` | No |
| Reproduce Tables 3-6 anonymously from saved endpoints | `review_reproduction/README.md`; `dist/anonymous-review-source.zip` | No |
| Reproduce Table 32 from saved summaries | `paper/supplementary/spectral_dimension_sensitivity/README.txt` | No |
| Inspect completed 8/16 sensitivity | `studies/spectral_dimension_sensitivity_v1/README.md` | No |
| Review completed follow-up evidence | `docs/TMLR_RESULTS_REVIEW.md`, `results/tmlr_followup_v2` | No |
| Review new bounded derivations | `analysis/main30_bounded_v1/input_map.md`, `results/main30_bounded_v1` | No |
| Check optional family-sensitivity inputs (no execution by default) | `analysis/rank_sensitivity_v1/rank_sensitivity_protocol.md` | No |
| Rebuild follow-up publication assets | `scripts/build_tmlr_results.py` | No |
| Verify software and preserved evidence | `make check` | No |
| Analyze existing coordinates spectrally | `studies/spectral_finite_bank_v1/README.md` | No |
| Build exact-action witnesses, enumerate image subgroups, and run controlled simulations | `studies/spectral_extensions_v1/README.md` | No |
| Evaluate newly authored context families | `studies/context_generalization_v1/README.md` | Extraction only, with explicit opt-in |
| Run the new workflow on a rented GPU | `docs/TMLR_RUNPOD.md` | Only the explicit extraction step |
| Build public source, Overleaf and anonymous review packages | `make release` | No |
| Locate historical evidence and failure records | `docs/TMLR_PROJECT_MAP.md` | No |
| See the complete supplementary execution plan | `docs/TMLR_EXPERIMENT_PLAN.md` | Explicit study commands only |
| Review what still prevents submission | `docs/TMLR_SUBMISSION_STATUS.md` | No |

A paper build, unit test, source check, installation without the GPU extra, or
package build never starts RunPod or downloads pretrained model weights.

## Local installation and software validation

Use Python 3.11 for the prospective study environment. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,analysis,prospective,paper]'
make check PYTHON=python
```

The legacy runtime remains available for reproducibility. Its objectives,
thresholds and failure meanings have not been changed by the TMLR migration.
Software fixtures check implementation contracts; they are not empirical
experiments, power studies or statistical calibration.

## Completed follow-up suite and reproduction routes

The reviewed suite contains 48 spectral cells, 48 witnesses, 2,952 complete
image-subgroup comparisons, 36 mechanism settings and 48 new-context cells. The
following source entry points document how those studies were run. A rerun is
not required to inspect or package the returned results.

### A. Spectral analysis of existing compact coordinates

The frozen protocol covers six models x four readouts x C7/D4 = **48 cells**, with
fixed rank 16 and each cell's original test partition. Complete state/action grids
include identity and fixed points. Each bank contains 128 blocks from 64 families
and two paraphrases; these are not 128 independent natural-language samples.

Follow `studies/spectral_finite_bank_v1/README.md`. Supply the private, previously
verified rank-recovery result archive and its recorded SHA-256 when reproducing
the analysis. Preparation checks
its source, PCA and row identities. Analysis is an explicit separate command.
No model forward pass, new GPU, historical calibration or full-suite replay is
needed. A high gamma establishes compatibility with a low-risk exact orthogonal
action in this finite bank and metric; it does not identify semantic or causal
mechanisms or certify new-context performance.

### B. New-context transfer evaluation

Follow `studies/context_generalization_v1/README.md` for the complete workflow:
prepare, inspect and attest to the new text bank, fit from old training/validation
data, extract with explicitly enabled model execution, analyze, and collect results. Reuse the archived training
coordinates and training-only PCA. Keep model revisions and readouts fixed, and
freeze fitting/selection before observing new-context outcomes.

The completed primary comparison is dedicated context-difference fitting versus
coordinate-fitted operators on the same new paired context differences, with an
identity control. Both primary fitted methods select hyperparameters on the same
old difference-validation loss. The earlier coordinate-validation-selected
comparator is retained separately as a secondary reference. The newly authored text bank has explicit authoring provenance;
it is not an independently sampled natural-language population. The archive
contains an operator review attestation made before model extraction; it is not
an authenticated signature or independent human authorship. The new bank changes
both outer narratives and limited factual clause wording. The historical
“designated weekday”/“marked corner” clauses become “scheduled weekday”/“marked
square corner”; therefore the study does not isolate outer-narrative changes.
Any reproduction must retain the protocol's review requirement before execution.

No complete replacement of FULL, AFS, synthetic calibration, native program
studies or failed scientific gates is required. Optional older workflows are
retained for provenance and should not be used as the default TMLR entry point.

### C. Exact-action witnesses, complete image subgroups, and simulations

The complete requested extension suite is in
`studies/spectral_extensions_v1/README.md`. It adds proof-based exact orthogonal
action witnesses, all 120 C7 and 3 D4 alternative image subgroups, and fixed
controlled mechanism simulations. All declared outputs are now reviewed and
retained. No model loading or GPU rental is required by this suite.

A witness's same-bank loss must use exactly the theorem's coordinates and metric.
Prediction on other contexts is scored separately using a witness frozen before
those outcomes are inspected. A different registered action changes targets; even
an exhaustive descriptive ranking is not an automatically valid permutation p
value. State coding and shared actions can coexist, so simulation mechanisms are
not treated as mutually exclusive semantic classes or as a calibrated classifier.
The registered simulations do not restart the original calibration pipeline.

The source README specifies exact commands and committed outputs for each phase.
Run the whole prespecified grid and retain failed cells and negative results.
Do not tune the primary dimension or choose a registration from final test scores.

## Results and manuscript integration

All real runs write to new directories outside the public source surface. Do not
replace `results/frozen/`, old source manifests, inspected results or failure
records. Keep input hashes, frozen protocols, complete per-cell outcomes,
environment details and checksums with each result archive. A failed or missing
cell must not be silently dropped or replaced with a different rank or layer.

This publication revision manually integrates the returned outputs using compact
public tables and traceable generated assets under `results/tmlr_followup_v2`.
It does not rerun extraction, fitting, evaluation or bootstrap sampling. The
recorded intervals remain descriptive resampling sensitivity summaries; no
population coverage, equivalence or general superiority claim follows from them.
See `docs/TMLR_RESULTS_REVIEW.md` for the full review and the recorded context
wording discrepancy. The prior input compatibility repair and its unchanged
before-copies are retained in [repair_v2](repair_v2/README.txt).

The private input archive, model weights and raw activations are deliberately not
in GitHub or Overleaf packages. See `inputs/README.txt` for the exact audited
archive identity and the preserved-source reconstruction route. To run from existing caches, obtain the exact
archive documented in the study README. To regenerate historical caches instead,
read the preserved reproduction documentation first; that is a separate, costly
workflow rather than a dependency of the new CPU analysis.

## Paper and packages

With TeX Live/latexmk or Tectonic installed:

```bash
make paper
python scripts/check_pdf.py
make verify PYTHON=python
make release PYTHON=python
```

`dist/github-source.zip` is the public code package, retaining the original MIT
license and author metadata. `dist/overleaf.zip` contains the anonymous manuscript
and its dependency closure, bibliography, figures and official TMLR support files.
Upload the latter to Overleaf and set `main.tex` as the main document.

The public code package is **not anonymous supplemental material**: it contains
license and citation attribution. Do not attach it to anonymous peer review or
link an identifying public repository from the blinded manuscript without
checking the venue's anonymity requirements. An anonymous review-code deposit
requires a separate attribution/anonymity review; never strip third-party license
notices to disguise authorship.

The package builder preserves the declared review status. A strict release
attempt with `python scripts/package_release.py --require-submission-ready`
refuses while final author review is pending. Completed experiments and successful
compilation alone do not establish author approval or acceptance.

## Provenance and contribution

- `release/source_manifest.json`: historical numerical source/configuration lock.
- `release/prospective_source_manifest.json`: preserved runtime plus explicitly
  versioned publication tooling; numerical runtime remains unchanged.
- `studies/spectral_finite_bank_v1/source_manifest.json`: cached-analysis code lock.
- `studies/context_generalization_v1/source_manifest.json` and
  `studies/spectral_extensions_v1/source_manifest.json`: new studies and inherited
  dependency identities.
- `release/artifact_manifest.json`: traced historical evidence and paper assets.
- `results/tmlr_followup_v2`: compact reviewed follow-up outcomes and provenance.
- `repair_v2/CHANGE_RECORD.json`: preserved input-adapter repair and predecessor hashes.
- `release/history/before_0_5_0/`: parent publication metadata before conversion.
- `release/current_release_status.json`: completed analysis scope and final review status.

Keep private inputs and results out of commits. Code review, CI and package builds
must remain CPU-only and require no Hugging Face credentials. No Git repository
initialization, remote publication, push or cloud launch is performed by these
commands. See `LICENSE` and `CITATION.cff` for code attribution. Official template
licensing is retained separately in `paper/TMLR_TEMPLATE_LICENSE.txt`.

The exact executed batch controller is retained at `runpod/tmlr_batch_v2/`; it remains opt-in. To rebuild the new manuscript assets without experiments, run `python scripts/build_tmlr_results.py` after installing the analysis extra. The compact evidence and archived-output import recipes are documented in `docs/TMLR_EVIDENCE_REBUILD.md`.
