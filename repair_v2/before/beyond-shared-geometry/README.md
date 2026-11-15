# Beyond Shared Geometry: a TMLR research project

**Version 0.5.0 - TMLR preparation draft, 29 October 2026.**

This project studies what recovered finite-group transformations establish about
language-model representations. It combines fixed-dimensional exact-action risk
bounds with preserved prediction, category-control, storage-budget and
context-transport results. All manuscript prose and project instructions are in
English.

The manuscript uses the official anonymous TMLR template. **The supplementary
analyses are prepared but have not been run. This is not yet a completed submission.**
Pending result slots are labeled explicitly; they contain no simulated substitute
for real observations. Historical measurements, source locks and scientific stops
remain unchanged. The prior ICML project is a separate preserved project.

## Start here

| Purpose | Entry point | Runs models? |
| --- | --- | --- |
| Read the TMLR manuscript | `paper/main.pdf`, `paper/main.tex` | No |
| Verify software and preserved evidence | `make check` | No |
| Analyze existing coordinates spectrally | `studies/spectral_finite_bank_v1/README.md` | No |
| Build exact-action witnesses, enumerate image subgroups, and run controlled simulations | `studies/spectral_extensions_v1/README.md` | No |
| Evaluate newly authored context families | `studies/context_generalization_v1/README.md` | Extraction only, with explicit opt-in |
| Run the new workflow on a rented GPU | `docs/TMLR_RUNPOD.md` | Only the explicit extraction step |
| Build source and Overleaf packages | `make release` | No |
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

## Prepared supplementary evidence suite

### A. Spectral analysis of existing compact coordinates

The frozen protocol covers six models x four readouts x C7/D4 = **48 cells**, with
fixed rank 16 and each cell's original test partition. Complete state/action grids
include identity and fixed points. Each bank contains 128 blocks from 64 families
and two paraphrases; these are not 128 independent natural-language samples.

Follow `studies/spectral_finite_bank_v1/README.md`. Supply the private, previously
verified rank-recovery result archive and its recorded SHA-256. Preparation checks
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

The primary comparison is dedicated context-difference fitting versus
coordinate-fitted operators on the same new paired context differences, with an
identity control. Both primary fitted methods select hyperparameters on the same
old difference-validation loss. The earlier coordinate-validation-selected
comparator is retained separately as a secondary reference. The newly authored text bank has explicit authoring provenance;
it is not claimed to be an independently sampled natural-language population.
Human semantic review is required before model execution. The code does not
attest that the user performed this review.

No complete replacement of FULL, AFS, synthetic calibration, native program
studies or failed scientific gates is required. Optional older workflows are
retained for provenance and should not be used as the default TMLR entry point.

### C. Exact-action witnesses, complete image subgroups, and simulations

The complete requested extension suite is in
`studies/spectral_extensions_v1/README.md`. It adds proof-based exact orthogonal
action witnesses, all 120 C7 and 3 D4 alternative image subgroups, and fixed
controlled mechanism simulations. These are prepared studies, not measured paper
results. No model loading or GPU rental is required by this suite.

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

New outputs do not automatically alter the paper. Inspect them, preserve every
planned outcome, then update the manuscript's pending tables with traceable
aggregates and a source record. The stored historical bootstrap intervals remain
descriptive sensitivity summaries; no new population coverage or significance
claim follows from this migration.

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

The package builder preserves draft status. A strict release attempt with
`python scripts/package_release.py --require-submission-ready` refuses while the
supplementary studies remain pending. Compilation success alone does not establish
scientific completeness, author approval or acceptance.

## Provenance and contribution

- `release/source_manifest.json`: historical numerical source/configuration lock.
- `release/prospective_source_manifest.json`: preserved runtime plus explicitly
  versioned publication tooling; numerical runtime remains unchanged.
- `studies/spectral_finite_bank_v1/source_manifest.json`: cached-analysis code lock.
- `studies/context_generalization_v1/source_manifest.json` and
  `studies/spectral_extensions_v1/source_manifest.json`: new studies and inherited
  dependency identities.
- `release/artifact_manifest.json`: traced historical evidence and paper assets.
- `release/history/before_0_5_0/`: parent publication metadata before conversion.
- `release/current_release_status.json`: current completion and pending status.

Keep private inputs and results out of commits. Code review, CI and package builds
must remain CPU-only and require no Hugging Face credentials. No Git repository
initialization, remote publication, push or cloud launch is performed by these
commands. See `LICENSE` and `CITATION.cff` for code attribution. Official template
licensing is retained separately in `paper/TMLR_TEMPLATE_LICENSE.txt`.
