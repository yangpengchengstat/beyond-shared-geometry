# MAIN30 bounded saved-result derivations

This directory implements authorized tasks B/C as **new analysis** of immutable saved results. It does not rerun the original spectral, witness, simulation, alternative-image, or context experiments. It does not perform model execution, calibration, optimization, or family sensitivity.

From the release root, using Python with NumPy:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analysis/main30_bounded_v1/derive.py --archive /path/to/tmlr-all-v2.20261030T203725Z-5489fd.results.zip --output /path/to/fresh-bounded-derivation
python -m unittest discover -s analysis/main30_bounded_v1/tests -v
```

The archive must have its original `.sha256` sidecar. The script checks the archive checksum, compact saved-result checksums, saved metadata bindings, all 48 cells and all 2,952 image conditions. It reads required members directly without extracting coordinate files. It saves metadata and hashes, full-precision CSV outputs, generated TeX fragments, before/after input manifests, and `validation_report.json` under the selected fresh output directory. The published derivation is retained in `results/main30_bounded_v1/`.

The original total type energy `e` and selected energy `b` are already normalized by rank. No additional division is applied. The risk premium is calculated per cell before equal-weight aggregation. The identity reference uses **total** trivial energy; an additional CPU check whitens each saved original test bank, computes its context-mean energy and complete-grid identity residual, and compares both with the saved-energy formula. This is a newly performed identity check, not recertification of the original experiments.

Image ordinals are retained only as provenance. Canonical sorted permutation-image lists from the saved plan are hashed and used for semantic exclusion and identity. No image set is re-enumerated. Ranks reuse the original symmetric absolute-plus-relative tolerance and are cross-checked against the original rank function and saved rank report.

All raw floats are retained. TeX display uses ordinary decimal rounding, with no clipping or tolerance-based numerical zeroing. Full CSVs remain authoritative. Fixed-panel counts and margins are descriptive; they do not provide significance, semantic correctness probabilities, risk rankings, or finite-family sensitivity results.

The tests are engineering/arithmetic fixtures. The average-before-max test uses a generic mixed-sign vector to distinguish operation order and is not asserted to be a realizable exact spectral bank. The 2-by-2 metric test and small complete-grid identity test are mathematical checks, not new scientific experiments.
