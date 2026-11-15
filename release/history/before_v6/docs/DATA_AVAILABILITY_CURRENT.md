# Included data and access limits

The public repository and the anonymous support package provide saved scientific
evidence for inspection. They do not contain every input needed to rerun every
historical measurement. The distinction below is part of the release's scope.

## Included

- Full-precision scalar outcomes, aggregate and per-condition records, frozen
  historical tables, scientific failure/stopping records, code and configurations.
- Source/checksum mappings in `results/registry.json`, study input manifests and
  `paper/supplementary/`; these identify the inputs and recorded outputs.
- A self-contained saved-endpoint arithmetic capsule in `review_reproduction/`.
  Its documented table numbering is historical; `REPRODUCE_CORE_RESULTS.md` maps
  current endpoints and labels to retained sources.
- Restored portable evidence in `results/verification_evidence_v3/`: the four
  rank/dimension audit CSVs previously omitted from the public bundle, and five
  existing calibration/power JSON records. Its README and `source_manifest.json`
  document source aliases, original/derivative hashes and path-only sanitization.
  Numerical values, NA fields, rank statuses and stopping decisions are retained.
  The same portable evidence is mirrored in `paper/supplementary/verification_evidence_v3/`
  so the Overleaf and anonymous bundles contain the promised audit records.

## Not included or not recoverable here

Model weights, raw activation tensors and multi-gigabyte original result archives
are not bundled. Historical missing PCA/retained-coordinate arrays remain missing.
The added audit tables document their unavailable status; they do not recreate
arrays or establish new independent measurements.

Full computational reproduction of stages requiring these inputs needs access to
the original archives and the gated model revisions listed by the relevant study.
An archive checksum identifies a file but is not a download link. This release
asserts no public archive URL, unrestricted model license, or automatic data access.
Before making a full-data-availability claim, the author must designate an actual
permitted distribution location or describe access restrictions in the submission.

## Reproduction levels

| Operation | What it establishes | Required inputs |
|---|---|---|
| `make check` | Integrity and static source consistency | This repository; Python standard library |
| Saved-endpoint capsule/formatters | Specified arithmetic and presentation from retained records | Capsule/supplement records; dependencies documented by each formatter |
| Offline matrix or spectrum analysis | Results from the declared coordinate banks and frozen protocol | Original coordinate/cache archives; numerical dependencies |
| New representation extraction | Hidden activations for the declared frozen model revisions | Model access, matching environment and prompts; compute resources |

Only packaging/static checks and presentation rendering were performed for this
release. No new matrix analysis, refit, simulation, bootstrap or LOFO was run.
Historical reports describe their own executions. The original scientific source
locks, frozen data and failure records are preserved.

The public project may contain attribution; the anonymous package contains a
separately selected and checked evidence subset. Source-location aliases replace
personal paths in portable audit copies. Raw byte originals and private path maps
are owner-only and must not be published with the anonymous package.
