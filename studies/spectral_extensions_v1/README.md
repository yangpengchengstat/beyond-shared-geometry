# Spectral extensions v1: prepared TMLR follow-up

This is executable CPU-only research software, prepared but **not run on the
research coordinate archive** in this release. Its software tests are small
fixtures, not the planned simulation study or new LLM evidence. The parent
finite-bank implementation, all historical measurements and failures remain
unchanged. No command starts a model, downloads weights, provisions a GPU,
repeats historical calibration or updates manuscript results automatically.

## Fixed scope and distinct endpoints

1. **Exact-action witnesses:** all 48 original cells. Fit an exact action and
   its whitening on all 96 old training context blocks; freeze it before scoring
   all 128 original test blocks. Compare with OLS and ridge at alpha 0.001 using
   the same training-frozen metric. Verify the theorem's bound on the training
   bank only. Old-test prediction is a separate retrospective comparison, not
   a theorem-certified generalization result or a historical row-normalized score.
2. **Complete alternative image subgroups:** all 120 C7 and three four-corner
   D4 image subgroups for each relevant cell: 2,952 cell/group units in total.
   Each changes its targets consistently on the complete state/action grid,
   including identity and fixed points. Save test-bank gamma using that bank's
   own second moment separately from the train-frozen prediction results.
3. **Controlled illustrations:** 36 fixed cases, two domains, mechanisms
   M1/M2/M3, noise 0 or 0.05, and seeds 2026102900--2026102902. Each has 32 train
   and 32 test contexts at rank 16. M3 is an oblique state/context direct-sum
   construction with a shared action before noise. It is not a negative null.
   M2's construction label also does not certify absence of every shared action.
   Its `shared_action_before_noise` is therefore null; the separate
   `context_specific_conjugations` field records what was constructed.
   These cases are illustrations, not false-positive or power calibration.

`protocol.json` fixes all choices. There is no CLI selector for best cells,
subsets, dimensions, noise levels, ridge penalties or subgroups. Any changed
scientific scope or code identity requires a new protocol and a new run.
The original test bank has already been inspected; none of these comparisons
is new independent-context confirmation or evidence of a native causal mechanism.

## Install and test

Use the project's numerical environment with NumPy. No GPU dependencies are
required for this study. From the release root:

```bash
python -m unittest discover -s studies/spectral_extensions_v1/tests -v
```

Both this study and `spectral_finite_bank_v1` must pass their checked-in source
manifests before production work. Do not edit a manifest merely to bypass a
source mismatch. Private fixture overrides are available only to software tests,
not through CLI. Importing the modules or requesting `--help` performs no research.

## Prepare from the verified coordinate archive

First follow `studies/spectral_finite_bank_v1/README.md` to create the original
metadata-only spectral preparation. Its original-test spectral analysis need
not already have run. Set the actual private archive path:

```bash
BSG_ARCHIVE=/path/to/from-zero-v1-rank-recovery.results.zip
python studies/spectral_extensions_v1/runner.py prepare \
  --spectral-prepared local_results/spectral-v1-prepared \
  --archive "$BSG_ARCHIVE" \
  --output local_results/spectral-extensions-prepared
```

Preparation rechecks the original archive checksum and parent source/metadata
manifest, then verifies the old training blocks using canonical IDs. It copies
only the selected train/test compact coordinates into new `banks.npz` files and
retains their row selection and PCA/source identity. It computes no spectrum,
fits no predictor and makes no numerical-rank eligibility claim. The original
archive is read only. Missing states, duplicate rows, wrong hashes or identities
produce retained invalid-cell records; no favorable subset replaces the panel.
The prepared result is sealed with file checksums before analysis.

## Run the stages explicitly

Use a distinct fresh directory for each stage:

```bash
python -u studies/spectral_extensions_v1/runner.py witness \
  --prepared local_results/spectral-extensions-prepared \
  --output local_results/spectral-witness-v1 --execute-analysis

python -u studies/spectral_extensions_v1/runner.py alternatives \
  --prepared local_results/spectral-extensions-prepared \
  --output local_results/spectral-alternatives-v1 --execute-analysis

python -u studies/spectral_extensions_v1/runner.py simulate \
  --output local_results/spectral-mechanisms-v1 --execute-analysis
```

The analyses are CPU-only. Complete subgroup enumeration entails 2,952 separate
fits/diagnostics and can take substantial CPU time and disk space; no one-minute
or total-runtime promise is made. Each `START_UNIT`/`DONE_UNIT` line identifies
progress. `state.json` records the active unit and completed/planned count.
Run one controller per directory. The advisory lock prevents simultaneous runs
and is released when a process exits; the lock file itself need not be deleted.
For long execution, redirect stdout/stderr to a separate log and use the shell's
usual background process facilities. No controller automatically stops a Pod.

## Resume without repeating verified units

A completed unit is reusable only when its input, source, protocol, numerical
runtime and output checksums match. Existing directories require `--resume`:

```bash
python -u studies/spectral_extensions_v1/runner.py alternatives \
  --output local_results/spectral-alternatives-v1 \
  --execute-analysis --resume
```

The corresponding `witness` and `simulate` commands accept the same resume flag.
Each real-coordinate run contains a verified `input_snapshot`, so the external
preparation/archive is not needed for its continuation. If `--prepared` is
supplied during resume, it must match the stored snapshot. A relocated run can
resume under the same released code and Python/NumPy versions. The release also
saves source snapshots for audit; using a matching source release remains the
simplest way to continue. The snapshots retain both study sources if restoration
is necessary.

Immutable scientific checkpoints live in `attempts/<unit>/<attempt-id>/` with
an exclusive completion receipt under `units/<unit>/receipt.json`. A receipt is
written only after all outputs have been saved and hashed. Completed and
scientifically not-evaluable units are terminal and are not repeated on resume.
An interrupted or technically failed attempt stays in place; a retry gets a new
attempt ID. Resume refuses corrupted completed output rather than deleting it.
Source/protocol changes require a new run instead of mixing scientific versions.
`state.json` and `events.jsonl` are execution records, not replacements for receipts.

## Interpret and collect complete or partial results

Every report snapshot contains all planned units, marking unfinished ones as
`not_completed`. `reports/<snapshot>/unit_summary.csv` gives the principal
metrics; complete per-unit summaries and NPZ diagnostics retain group laws,
orthogonality, support checks, training bounds and separately evaluated test
spectra. Test-bank spectra can be evaluable even when a particular witness
construction is not; both statuses remain visible. An alternative unit is
`complete` only when both its witness and test-bank spectrum are evaluable.
Its `witness_status` and `test_spectral_status` identify the distinct endpoints.

`descriptive_ranks.json` is never a significance test. A fully evaluated panel
gets a semantic rank interval accounting for numerical ties. If groups fail,
its main rank fields remain null; it separately reports ranks among evaluable
groups and the possible range across all planned groups. Missing groups cannot
improve the apparent semantic rank by being silently excluded.

Witness construction checks selected-support eigenvalues against the declared
absolute 1e-12 and relative 1e-10 thresholds. A selected support direction may
be unresolved even when the scalar spectral budget passes its own tests; this
returns `not_evaluable_witness`. No eigenvalue is silently dropped, clipped or
regularized to force a witness. The implementation does not promise successful
construction on every numerically marginal input. Keep raw diagnostics and avoid
interpreting near-zero differences as strictly positive effects.

Collect a completed or interrupted run into a new ZIP outside the run directory:

```bash
python studies/spectral_extensions_v1/runner.py collect \
  --run local_results/spectral-alternatives-v1 \
  --output local_results/spectral-alternatives-v1.results.zip
```

Collection verifies checkpoint hashes, refuses an actively running controller
and existing output names, and includes input/source snapshots, original
failures, partial attempts, plan, runtime, reports and a collection manifest.
Its `partial` flag reports execution completeness, not scientific success. An
executed panel containing not-evaluable units has `execution_status: complete`
and `status: completed_with_not_evaluable`; its collection is not a partial run.
The `successful`, `not_evaluable_units` and `not_completed` counts distinguish
evaluable output, retained numerical/input failures and unfinished execution.
A technical interruption or source-identity failure still produces a partial
collection even if its terminal count happens to equal its planned count.
`scientific_pass` remains false and `scientific_pass_assessed` false: these stages
do not implement a scientific acceptance gate. The command also writes a
`.sha256` sidecar. Verify both after
transfer before deleting any machine's working data. The source archive is not
included because each run carries its required compact input snapshot; no model
weights or raw activation cache are needed for these three CPU studies.

Neither completed execution nor passing numerical checks promotes a result into
the paper. Review all coverage and failures, confirm metric distinctions, and
register selected figures/tables with their exact source identity before filling
any pending manuscript slot.
