# Rebuild the completed follow-up evidence

The current paper uses received experimental results. These commands format and
verify recorded results; they do not run models, estimate new spectra, fit models,
or rerun bootstrap draws. Use Python 3.11+ and the `analysis` extra for plotting.

## Public compact evidence

`results/tmlr_followup_v2/evidence_manifest.json` locks every compact input.
`context_metrics.json`, `context_cell_metrics.csv`, and
`context_family_contrasts.csv` preserve the primary effect, all 48 cells and all
64 family effects. The `spectral/` subdirectory retains all 48 spectra and witnesses,
36 mechanism settings, all 2,952 image-subgroup rows, and committed member hashes.

```bash
python scripts/build_tmlr_results.py
bash scripts/build_paper.sh
python scripts/check_pdf.py
python scripts/check_release.py
python scripts/package_release.py
```

Rendering changes no evidence files. Main tables and the context plot are linked
to the compact evidence in `paper/generated/followup_results.source.json`.

## Verify the private archive and reconstruct compact summaries

Use the received archive with SHA-256
`ef87e9aaeb8f093ba0c2438590d6a30b5dffcd256993e2366bd600cca3cb5bbb`.
Supply its path explicitly; do not overwrite the reviewed compact evidence.

```bash
python scripts/extract_context_summary.py RESULTS_ZIP /tmp/bsg-context-review
python scripts/summarize_spectral_extensions.py RESULTS_ZIP /tmp/bsg-spectral-review
```

These archive-specific recipes inspect receipt and member hashes and summarize
already saved values. The context recipe copies the recorded bootstrap endpoints;
it does not generate new resamples. The complete original coordinate ZIP and raw
run archive remain private inputs outside the public GitHub and Overleaf bundles.
Their public provenance does not claim that omitted raw tensors are distributed.

## Experimental reproduction, only if separately requested

The byte-identical batch controller used for the completed run is preserved at
`runpod/tmlr_batch_v2/`. Its `start.sh` requires an explicit command, source path,
original rank-recovery archive and a fresh run location. It enforces source/model
identities and bank review before new extraction. See its README and the three
study READMEs. This release does not authorize or start another cloud experiment.
Historical calibration/native stops remain as recorded; they are not prerequisites
for regenerating the current paper's reviewed tables.

The public source bundle retains the author's license/metadata and is not an
anonymous review supplement. The separate Overleaf bundle is anonymous. Final
submission and any external raw-data release require author review.
