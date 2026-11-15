# Working on this release

This is a publication project derived from an existing research repository. Preserve the original MIT license and author metadata in the public code release. The manuscript uses anonymous submission mode.

- Treat `results/frozen/` and original source archives as immutable evidence. Do not overwrite historical measurements, frozen decisions, or source identities. Write derived artifacts separately and retain their input hashes and transformation records.
- `release/source_manifest.json` locks the exported numerical source and configurations. Do not silently edit scientific objectives, splits, metrics, null constructions, bootstrap units, or model revisions to make checks pass. Document prospective scientific changes and regenerate provenance only when intentionally changing that release contract.
- Keep model execution optional. CPU tests, paper builds, checks, and packaging must not load/download model weights, launch GPU/cloud jobs, or spend money. An actual experiment needs an explicit user request and a fresh output location.
- Preserve historical pipelines and their gate meanings. A completed FULL run's `overall: GO` is not proof that all scientific gates passed or that learned edits outperform the additive control.
- Keep publication paths relative. Exclude credentials, private host paths, caches, checkpoints, weights, raw activation tensors, and Git history from release bundles.
- Every published figure/table or selected evidence file must appear in `release/artifact_manifest.json` with its checksum, source run, archive-relative source path, and role. Derived figures and TeX must retain transformation provenance.
- Use the unchanged official TMLR style and its retained license. Preserve anonymous manuscript mode. Pending experiment slots prevent submission readiness; software or layout checks do not convert them into completed results.
- Use `PYTHONPATH=src python tests/run_cpu_checks.py`, `python scripts/verify_frozen.py`, and `python scripts/check_release.py` for CPU validation. Rebuild paper assets and the manuscript only when the corresponding inputs change, then run `python scripts/package_release.py`.
- No Git initialization, publication, push, or cloud launch is implied by a local build request.

- The active TMLR entry points are the source-locked finite-bank spectral, spectral-extension and context-generalization studies. Do not invoke historical calibration or full-suite launchers as prerequisites.
- Preserve the prior ICML project and old release metadata. New study outputs belong in new private run directories and must not auto-promote manuscript claims.
