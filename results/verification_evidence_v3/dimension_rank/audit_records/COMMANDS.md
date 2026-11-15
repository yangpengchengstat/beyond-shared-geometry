# Actual execution and reproduction

All numerical work in this revision was local, read-only analysis of retained matrices or aggregation of recorded CSV values. No study launcher, training routine or model API was invoked.

The actual runtime was the bundled Python3.12 environment (NumPy2.3.5), with float64 SVD/eigendecomposition; environment details are in `logs/matrix_runtime.json`. Historical extraction and rank-check precisions remain separately recorded.

Actual primary commands (staging directory `audit_workdir`):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 audit_python_environment/bin/python3 -u audit_workdir/scripts/audit_saved_matrices.py --repo original_research --output audit_workdir
audit_python_environment/bin/python3 audit_workdir/scripts/enrich_rank_records.py --repo original_research --output audit_workdir
audit_python_environment/bin/python3 audit_workdir/scripts/build_audit_tables.py
```

Component execution commands are preserved verbatim in `components/lexical/commands.txt`, `full_input_commands.txt`, `components/sensitivity/commands.txt`, and the history component reports. They include the original temporary output paths. No temporary raw archive copy is packaged; its original remains in the repository at the hash-identified location.

The first primary audit-loader attempt failed because a witness receipt path was assumed to end in `<cell>.json` instead of `<cell>/receipt.json`. Its exact script/log remain in `logs/matrix_audit.attempt1.*`. The corrected script follows and verifies the committed receipt. Sensitivity-component initial loader/serialization logs are also preserved. These are audit-software corrections, not failed or rerun scientific experiments. The applied rank/conditioning thresholds were not changed. Final checks pass.

The derived table builder normalizes component column names and labels originally recorded checks separately from new rechecks; it never changes a raw measurement. Its presentation table rounds to six decimals while all CSV values retain Python float round-trip precision. The final export changes only staging component paths from `agents/` to `components/`.

## Optional rerun of this read-only audit

Use `scripts/run_readonly_audit.sh REPOSITORY_ROOT NEW_EMPTY_OUTPUT_DIRECTORY` with `AUDIT_PYTHON` set to a Python environment containing NumPy. It operates on the same existing archives, does not import model or fitting code, refuses to reuse an output directory, and writes only new derived files. No rerun is necessary to read the delivered results. Inputs are referenced relative to the repository, with hashes in `provenance/` and the component records.

The helper's role is reproducibility of this audit, not further experiments. The finite-bank gamma computation, LOFO, simulation grid and original experiments are not invoked.

## Paper compilation

The existing project Tectonic0.17.0 binary was used with `--only-cached --keep-logs --keep-intermediates` to export the revised PDF; the built-in LaTeX compiler was separately called on the final saved main file. `COMPILE_CHECK.md` records the outcomes, page count, cross-reference checks and visual review. No model/network download was required.

## Training-rank wording clarification (2026-11-05)

This follow-up changes the interpretation and wording only. Existing diagnostic values and thresholds are reused; no audit script or experiment was rerun.

- Compared `pca.py` and `canonical_units.py` byte-for-byte against Git commit `1b7adfe717c6a7663c90e0a7c31d8a612f22afe9`.
- Applied the bounded editorial changes with `python3 temporary_workdir/bsg_rank_scope_revision.py --apply`; the script is retained as `provenance/editorial_scripts/clarify_training_rank.py` and the final diff as `provenance/training_rank_clarification.diff`. A subsequent one-sentence correction specifies that attaining n−1, rather than attaining min(d,n−1) in all dimensions, requires affine independence; it is included in the retained script and final diff.
- Called the built-in `compile_latex_document` on the saved `paper/main.tex` with the existing project root; final status success.
- Exported the same paper in place by running `original_research/submission/icml2027/.tooling/tectonic/tectonic --only-cached --keep-logs --keep-intermediates main.tex` in `paper/`.
- Rendered only the changed/relevant PDF pages with `pdftoppm` into `temporary_workdir/bsg_rank_scope_pdf_review`; performed visual checks.
- Ran the preserved editorial finalization helper to compare checksums, maintain diffs and refresh the existing source archive. This reads PDF metadata and file bytes; it does not load model/matrix-fitting code.

The editorial helper is an exact historical patch, not a general rerun command for the already modified paper; its original-content assertions intentionally prevent reapplication.

## Completed8/16 sensitivity and subsequent editorial integration

The offline numerical execution is recorded separately in `offline_dimension_execution_v1/analysis_launch.json`, `analysis_exit.json`, `numerical_tests.json` and `runtime.json`. It used Python3.11.8/NumPy2.2.6, unlike the initial matrix-audit runtime above. Six numerical software tests passed, all16-dimensional anchors and rankings matched, then all8-dimensional units completed. This computation is additional descriptive analysis, not part of the original rank audit or independent confirmation.

The later editorial integration used `python3 temporary_workdir/bsg-dimension-integration-20261106/integrate_paper.py` and `update_project_docs.py`. Exact helpers are retained under `provenance/editorial_scripts/`; they are historical patches with assertions, not scripts to reapply to the already revised paper. They copy saved full-precision outputs and generate documentation, not numerical results.

The built-in LaTeX compiler was called on the current saved `paper/main.tex`; the same source was exported in place with the existing Tectonic binary, `--only-cached --keep-logs --keep-intermediates`. The table reproduction command and current compile/visual checks are recorded in `COMPILE_CHECK.md`. Original16-dimensional results, old calibration/rank failures, audit CSVs and analysis source snapshots remain unchanged. No experiment was started during this editorial step.
