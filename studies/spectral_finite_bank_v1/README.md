# Finite-bank spectral diagnostic v1

This CPU-only implementation prepares the bounded analysis specified in
`docs/SPECTRAL_FINITE_BANK_PROTOCOL_0_4_8.txt`. Release 0.4.9 adds hardened
software and small software fixtures; no real-coordinate spectral results have
been computed by this revision. The original prototype is preserved separately.
No command imports a model runtime, downloads weights, starts RunPod or runs
calibration. Real-data analysis is an explicitly selected later step.

## Fixed scientific target

The primary panel is six pinned models, four saved layers per model and C7/D4:
48 cells, original `test` partition, rank 16, original semantic registration.
Each cell must contain all 128 declared context blocks with all seven or four
states. These blocks are not independent natural-language contexts. All group
elements, including the identity and fixed points, receive equal weight.
`protocol.json` freezes model revisions, layers, row/context identities, source
contract hashes, metric, tolerances and interpretation. No alternative ranks,
registrations, partitions or cell selection are exposed by the CLI.

The endpoint is the real-type-constrained invariant-subspace budget in a new
uncentered empirical second-moment metric. Its risk bounds concern the entire
exact orthogonal action class in that same finite bank. They do not bound the
historical row-normalized Ridge/Joint scores, certify new contexts, distinguish
state coding from every shared action, or establish a native causal mechanism.
No witness action, resampling, p-value or population certificate is implemented.

## Software checks

From the release root, using a Python environment with NumPy:

```bash
python -m unittest discover -s studies/spectral_finite_bank_v1/tests -v
```

These fixtures validate algebra and safe input handling. They are not a
statistical calibration study or new LLM evidence. The source lock must cover
all study files before production preparation/analysis; changes require a new
reviewed source identity. Test-only private helpers cannot be selected by CLI.

## Later execution on the existing private archive

The following commands are instructions, not a record of completed analysis.
They work on a local CPU; no new cloud rental is required. Keep the private ZIP
outside this public source release. Copy its already-verified checksum into
`BSG_ARCHIVE_SHA256`; do not calculate a fresh checksum merely to trust an
unknown or changed file.

```bash
BSG_ARCHIVE=/path/to/from-zero-v1-rank-recovery.results.zip
BSG_ARCHIVE_SHA256=PASTE_THE_VERIFIED_64_CHARACTER_SHA256
python studies/spectral_finite_bank_v1/runner.py prepare \
  --archive "$BSG_ARCHIVE" \
  --expected-sha256 "$BSG_ARCHIVE_SHA256" \
  --output local_results/spectral-v1-prepared
```

`prepare` checks archive and member hashes, pinned model/PCA/extraction
identities, NPZ headers, and every selected row/context/state. It does not load
coordinate values into the spectral calculation or compute any eigenvalues.
It writes a manifest before any scientific endpoint is seen. It reports all
planned cells, including invalid or missing cells. A metadata-ready cell has
not yet passed numerical rank, conditioning or residual checks.

After reviewing preparation status, the explicit analysis command is:

```bash
python studies/spectral_finite_bank_v1/runner.py analyze \
  --prepared local_results/spectral-v1-prepared \
  --archive "$BSG_ARCHIVE" \
  --output local_results/spectral-v1-results \
  --execute-analysis
```

`--archive` is optional when the archive has not moved. Relocation is allowed
only for a file with the exact recorded checksum. The frozen source, protocol,
all selected input hashes and metadata are checked again before computation.
Both commands reject existing output directories; use fresh names for a
reviewed rerun. Never delete failure records to reuse a name. Preparation and
analysis write only to their new output directories and never modify the ZIP.
The archive is hashed before and after inspection, so wall time includes reading
the full approximately 3 GB archive even though only compact coordinate members
are needed. No completion-time guarantee is made.

## Outputs and failures

- `input_manifest.json`: exact selected row/context IDs, source and PCA hashes,
  array schemas, full group/action tables, protocol and code identities.
- `runtime.json`, `protocol.json`, `state.started.json`: saved before any result.
- `cell_summary.csv`: one row for every planned cell, including failures.
- `type_spectra.csv`: complete multiplicity spectra, type energies and selections.
- `cells/<cell>/summary.json`: metrics, all type spectra/energies and residuals.
- `cells/<cell>/diagnostics.npz`: raw matrices, whitening and numerical spectra
  when available. Loading requires `allow_pickle=False`.
- `validation.json`, `interpretation.txt`, `output_manifest.json`: coverage,
  scope and checksums. Source coordinate arrays are not copied into this output.

`not_evaluable_input` means the declared input contract was not met;
`not_evaluable_rank` and `not_evaluable_numerics` retain diagnostic values without
substantive gamma/risk claims. A global input mismatch prevents analysis rather
than selecting an apparently usable subset. Archive mutation during analysis
marks the output ineligible. Unexpected execution interruption leaves the started
record and partial results; it does not constitute a completed panel.

A passed float64 residual check is not a rigorous floating-point interval
certificate. Keep raw values; near-zero effects are numerically unresolved.
Do not clip eigenvalues, lower rank, add jitter, delete trivial components or
choose more favorable cells to obtain a desired result. No result automatically
updates the manuscript. Review all 48 rows before deciding whether the endpoint
adds information beyond the existing predictor errors and ordinary spectrum.
