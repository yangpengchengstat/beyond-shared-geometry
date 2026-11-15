# Saved 16-dimensional coordinate inputs

This directory makes the retained Study 2 coordinates available without the
multi-gigabyte historical archives. The numerical arrays are **original saved
bytes**, not new model activations, PCA fits, whitenings or experimental results.

`manifest.json` covers 72 fixed model/readout/domain cells: 24 C7, 24 D4 and
24 S3. `manifest_c7_d4.json` is the self-contained 48-cell subset required for
the spectral and three-treatment centering diagnosis. Each cell maps the
coordinates, PCA and row-metadata roles to hash-addressed files in `objects/`.
Identical metadata are stored once. NPZ files retain their original member bytes;
no numeric array has been rounded, recompressed, recentered or sliced for export.

The manifests retain the source-archive aliases and SHA-256 identities, original
ZIP member names and member hashes, model/tokenizer pins, readout, array headers
and row roles. The C7/D4 records also bind the 128 complete test blocks to the
original saved coordinate rows. Coordinates include other recorded partitions;
the numerical diagnosis selects only those bound test rows.

`centering_anchors.json` is a **publication derivative**, not a byte copy of each
large historical summary. It selects the existing scalar, type-spectrum and
diagnostic fields required by the unchanged centering logic: 48 canonical and
2,952 image anchors, original ranks, and subgroup definitions. It records every
source summary member and its original SHA-256. The export does not recompute
these values. The full original summaries remain in the authenticated historical
follow-up archive identified by the manifest.

The source archives are identified as:

| Alias | Original file | SHA-256 |
| --- | --- | --- |
| parent | `from-zero-v1-rank-recovery.results.zip` | `6a4c38237bf1368d4c77c97504f666d0d933bfebed9417d173fbf7441c0fecef` |
| followup | `tmlr-all-v2.20261030T203725Z-5489fd.results.zip` | `ef87e9aaeb8f093ba0c2438590d6a30b5dffcd256993e2366bd600cca3cb5bbb` |

The archives themselves are not needed for the public centering commands in
`studies/template_centering_v1/README.md`. A source hash is an identity record,
not a claim of unrestricted access to a different archive or to model weights.

## Verification and anonymous selection

From the repository root, using the Python standard library only:

```sh
python3 -B studies/template_centering_v1/run.py verify
```

This verifies the C7/D4 data closure and saved results; it performs no matrix
computation. All-cell object identities can also be checked directly against
`manifest.json`. `anonymous_files_c7_d4.json` gives the exact root-relative data
allowlist and hashes for the anonymous package. That subset excludes S3 objects
and the all-cell manifest, while retaining the C7/D4 PCA, rows and anchor inputs.

## What this export does not restore

- Original full-dimensional hidden activations, pretrained model weights and
  original extraction environments are not in this directory. Saved PCA bases
  and retained coordinates cannot independently establish the original forward
  pass or recover discarded dimensions, full-space within-template variance,
  or a new PCA fitted to within-template variation.
- S3 coordinates/PCA/rows are included, but S3 source-embedding potential inputs,
  potential neural fits, full-hidden interface training inputs and interface
  checkpoints are not part of this coordinate capsule. Coordinate-only analytic
  procedures and those additional-information/neural procedures have different
  input requirements.
- The separate authored-context coordinates and all historical model-forward
  caches are not supplied by this capsule. Existing outcome tables elsewhere
  in `results/` are evidence for their saved endpoints, not replacements for
  the absent inputs needed to rerun those measurements.
- Missing historical Study 1 lexical PCA outputs and operator arrays remain
  missing; this Study 2 export does not reconstruct them. Original raw/cache
  archives and access restrictions are documented separately by the release.

Consequently this input export enables reproducible retained-bank analysis; it
does not claim full raw-to-paper computational reproduction of every experiment.
