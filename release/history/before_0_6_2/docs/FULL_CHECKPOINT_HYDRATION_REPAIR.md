> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# FULL Checkpoint Hydration Repair

Repair date: 2026-09-05

Affected run: `main_full_002`

Frozen scientific baseline: `b420cc2b37172cfbf6619217beed0a6dbaaaccd1`

## Scope

This is a runtime persistence and orchestration repair. It does not change the
frozen FULL matrix, fitting procedures, operators, metrics, estimands,
bootstrap hierarchy, seeds, controls, model revisions, or success rules.

The scientific computation status and runtime persistence status are distinct:

- The completed structural base and joint cells are valid scientific
  computations with checksum-verified per-unit checkpoints.
- The failed process did not reconstruct all downstream in-memory indexes and
  aggregate input lists from those checkpoints. Cross-model execution and
  final aggregation therefore failed despite completed upstream science.

## Root Cause

Fresh execution populated the cross-model `artifact_index` and the structural
and irrep CSV path lists while traversing model loading, activation extraction,
and structural fitting loops. The local COMPLETE branch in each structural
unit could deserialize its NPZ, but there was no process-level branch that
reconstructed the complete index directly from the ledger. Recovery therefore
depended on re-entering all upstream loops. If that traversal was skipped or
did not reach every completed cell, cross-model lookup failed for keys of the
form:

`(model_key, group, depth_slot, split_seed, latent_dim, optimization_seed)`.

Final aggregation had the same coupling: it concatenated process-local path
lists rather than enumerating verified completed checkpoints. Empty lists
produced a one-byte structural CSV and later caused the missing `layer` and
`E_eq` schema error.

Model-load markers recorded the resolved revision but not the tokenizer
identity. Tokenizer identity lived only in the in-memory manifest until the
manifest was flushed, so an interrupted invocation could leave null provenance.

## Representative Structural Joint Checkpoint

One `structural_joint` unit has three checksum-bound files:

1. `<digest>.csv` contains held-out validation/test template metrics and cell
   identifiers: model, group, layer/depth, latent dimension, split seed,
   deterministic optimization seed, random-control seed, method, template,
   PCA variance, algebraic metrics, and joint-objective terms.
2. `<digest>.irrep.csv` contains the same cell identifiers plus learned
   representation/irrep diagnostics for the joint operator.
3. `<digest>.npz` contains exact numerical state: PCA mean, PCA components,
   explained-variance ratios, PCA fit-row count, every latent coordinate `z`,
   and all learned operator matrices stacked in the canonical group-element
   order.

The NPZ therefore contains everything required by cross-model fitting:
held-out latent coordinates, PCA state, and learned operators. Hydration uses
`load_artifact`; it does not refit PCA, ridge operators, or joint operators.
Structural hidden-state activation caches are separately revision- and
metadata-bound NPZ files and provide the held-out activations used for CKA.

## Repair

When model load, both extraction modules, structural base, structural joint,
and causal units are all COMPLETE and checksum-valid, runtime now:

1. reconstructs configured group specifications;
2. derives and verifies every frozen structural checkpoint key;
3. validates every serialized joint NPZ and rebuilds the exact six-field
   cross-model artifact index;
4. reconstructs structural activation-cache keys from frozen metadata;
5. retries only incomplete, FAILED, or interrupted/RUNNING cross-model units;
6. reuses valid COMPLETE cross-model checkpoints;
7. rebuilds structural, irrep, causal, and cross-model promoted CSVs from
   verified per-unit checkpoints;
8. rejects empty or malformed aggregate schemas before postprocessing; and
9. continues through bootstrap, tables, figures, gates, and completeness
   validation.

`RunLedger.begin` preserves and increments attempt counts for FAILED or RUNNING
units. COMPLETE upstream units are only verified and hydrated; their status,
attempt count, checkpoint bytes, and provenance are unchanged.

New model-load checkpoints persist exact tokenizer identity alongside the
resolved revision. Existing legacy markers can exactly recover the immutable
revision and configured tokenizer settings, but not tokenizer class or token
IDs that were never serialized. Such legacy recovery is explicitly marked
`PARTIAL_LEGACY_CHECKPOINT` rather than fabricating unavailable provenance.

## Verification

A CPU subprocess fixture starts a brand-new Python process and demonstrates:

- exact restoration of PCA means/components, latent coordinates, and all
  operator matrices;
- exact restoration of serialized structural metrics;
- reconstruction of
  `('model_a', 'C7', 0.5, 101, 2, 42)`;
- successful cross-model computation and checkpoint creation;
- retry of a FAILED cross-model unit with its attempt history preserved;
- no invocation of structural base or joint fitting during recovery; and
- non-empty promoted structural output with the normal `layer` and `E_eq`
  schema.

The same tests cover fresh structural computation feeding cross-model, exact
and legacy manifest hydration, and rejection of empty promoted CSVs.

## Recovery Contract

For `main_full_002`, the repaired package must be unpacked over a clean code
directory while preserving the existing `run_outputs` and activation cache.
Running the same FULL command and run ID automatically selects checkpoint-only
cross-model recovery when all expensive prerequisites verify. No separate
scientific protocol or recovery estimand is introduced.
