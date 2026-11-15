# Maintenance v2: selective future execution

The 24 October 2026 returned results have already been audited. No new cloud
run, calibration, or full-study rerun is needed to incorporate that evidence.
This entry point is supplied for a future, explicitly authorized selective run.
It does not launch anything during installation, checking, paper building, or
packaging. The existing scientific failures remain failures.

This source is a new version. The original reproduction-v1 and rank-recovery-v1
source capsules and results retain their original identities. Their launchers
are historical scripts and must run from their matching original capsules;
they are not automatically valid against this modified source tree. Do not
extract this revision over a running or frozen old source directory.

## Changes

- The interface creates its `fits` directory before its first checkpoint. It
  records total fit time including PCA, initialization, training, and evaluation;
  serialization time is separately included in `fit_and_checkpoint_seconds`.
- Interface checkpoints bind their metadata and parameter arrays. Resume checks
  checksums, descriptor fields, array names, float64 shapes, and finite values.
  A complete fit is reused; a missing incomplete fit is retried. The selection
  objective, training steps, seeds, and validation-only choice are unchanged.
- Factorial evaluation atomically commits all eight scores for each recipient.
  After an interruption, complete recipients are verified and reused. Only an
  interrupted recipient (at most eight scores) is rescored. JSONL exports retain
  their scientific row structure and are rebuilt from completed checkpoints.
- Both tasks print completed/total, reused/new counts and elapsed session time.
  `progress.json` describes the latest session. Every invocation has a separate
  durable record under `sessions/`, so resume does not erase prior timings or
  failure records. A forcibly killed process may leave a `running` session;
  that label alone never certifies completion. No ETA is inferred from an early
  fast fit or from verification-only resume time.
- Identity or checksum mismatch stops execution. Old v1 result directories are
  deliberately incompatible with v2 and are never silently imported or rewritten.

## Offline check (no model or GPU)

From the new release root, using its installed Python environment:

```bash
python studies/maintenance_v2/run.py --task interface
python studies/maintenance_v2/run.py --task peer_factorial
```

These commands verify the new release source manifest and print a plan. They
create no run output and schedule neither models nor experiments. Failure of
source verification means the release must first be deliberately rebuilt and
versioned; do not edit the old frozen manifests to bypass it.

## Selective future run — only if separately requested

An interface rerun needs the complete, verified raw activation caches, not just
the returned results ZIP. It does not recreate missing caches. A factorial run
requires the exact pinned model files already present locally; this launcher
sets `local_files_only=true` and will not download missing models. Use the
release's pinned environment and a new empty output path. The two examples
are alternatives, not an instruction to execute both now:

```bash
python studies/maintenance_v2/run.py --task interface \
  --config configs/prospective/bf16_revision.json \
  --cache-root /workspace/verified-activation-caches \
  --output /workspace/bsg-runs/interface-maintenance-v2 \
  --execute --execute-models
```

```bash
python studies/maintenance_v2/run.py --task peer_factorial \
  --config configs/prospective/bf16_revision.json \
  --output /workspace/bsg-runs/factorial-maintenance-v2 \
  --execute --execute-models
```

Reissuing the exact same command resumes only matching v2 checkpoints. Do not
change configuration, model revisions, software/source identity, cache content,
or task mode inside an existing output namespace. Default runtime controls and
scientific settings come from the supplied configuration. Task completion is
operational: it does not waive competence/power gates or promote an observation
to an independently confirmed result.

## Validation scope

`tests/prospective/test_maintenance_v2.py` uses artificial features and mocked
fits. It checks fresh directories, interruption and exact resume, parameter
corruption, nonfinite fits, immutable prior-session records, single-writer
exclusion, legacy-output rejection before model creation, and byte-equivalent
factorial exports after interruption. This is software regression validation,
not a rerun of research observations. No model weights or GPU are used.
