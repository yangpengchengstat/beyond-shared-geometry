# Completed offline spectral dimension sensitivity

This portable export accompanies release 0.5.5 and Appendix K / Table 32. The
completed analysis contains 6,000 canonical/image computations: 3,000 at dimension
16 followed by 3,000 at dimension 8. All 48 original conditions were retained.
Results are already available under `results/spectral_dimension_sensitivity_v1`
and `paper/supplementary/spectral_dimension_sensitivity`. No numerical computation
is required to read, compile or package this project.

The original 16-dimensional PCA was not re-estimated: dimension 8 uses its first
eight coordinates. The original 16-dimensional results remain primary. The
32-dimensional inputs from the same original Study 2 bank were unavailable.
Cross-dimension losses use different observation spaces and whitening metrics;
this is sensitivity analysis, not selection of a superior dimension. Witness
transfer and context transport were not recomputed. LOFO remains prepared and
unexecuted.

## Evidence and identity

`EXPORT_PROVENANCE.json` records the original manifest identity and byte-for-byte
identity of all copied code, tests, frozen kernel and numerical protocols. This
export has a new source manifest because its documentation is portable and
reflects completed outcomes. It does not replace the archived original source or
its receipts. Existing run resumption must use that original source snapshot.
The full original snapshot, run receipts and failures are retained separately in
the local research archive.

## Optional reproduction, never needed for packaging

Use the recorded runtime: Python 3.11 and NumPy 2.2.6. Install dependencies only
when deliberately reproducing the study. The input reader takes a data root
containing the following original files, with their identities checked by code:

- `incoming/rank_recovery_v1/from-zero-v1-rank-recovery.results.zip`
- `incoming/tmlr_all_v2/tmlr-all-v2.20261030T203725Z-5489fd.results.zip`

The local complete-project delivery places these beneath `research_archive`.
From the GitHub project root, the metadata-only preflight is:

```bash
python3 studies/spectral_dimension_sensitivity_v1/run.py preflight \
  --repo ../../research_archive --output ../../dimension-preflight-fresh
```

This command verifies archives and array headers without numerical analysis.
The complete optional numerical reproduction requires a fresh output directory:

```bash
python3.11 -m unittest discover -s studies/spectral_dimension_sensitivity_v1/core/tests -v
python3.11 studies/spectral_dimension_sensitivity_v1/run.py run \
  --repo ../../research_archive --output ../../dimension-reproduction-fresh \
  --execute-offline-analysis
```

These reproduction commands were not executed when assembling release 0.5.5.
They do not load a language model, start RunPod, train historical operators,
change source archives or substitute a different bank. The archived completed
run, rather than this packaging operation, supplies the empirical evidence.
