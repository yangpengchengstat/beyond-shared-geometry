# Data availability

The release includes scientific code, saved outcomes and selected original coordinate inputs. It does not claim every historical experiment can be rerun without further inputs.

## Included

- Frozen results, full-precision endpoints, source/configuration identities and historical failures.
- Study 2 saved 16-dimensional coordinate banks for 72 cells: 24 C7, 24 D4 and 24 S3, with available PCA/row metadata. `data/retained_coordinates/` records exact archive-member hashes and deduplicated metadata identities.
- C7/D4 input anchors, portable centering code and original saved cell/image results in `results/template_centering_v1/`.
- Rank/dimension and calibration/power audits in `results/verification_evidence_v3/`.
- The `review_reproduction/` saved-endpoint capsule and paper supplements.

The anonymous support ZIP includes the 48-cell C7/D4 subset needed for the centering comparison. The public tree includes all 72 retained banks. Its manifest gives exact coverage.

## Limits

Model weights, full hidden-state tensors and complete multi-gigabyte parent archives are not supplied. Historical missing PCA/projected arrays remain missing. The saved 16-dimensional banks cannot recover discarded directions, measure state-variance retention in full hidden space, or implement a new PCA fitted to full-dimensional within-template differences.

Some historical and end-to-end workflows require the original archives and recorded model revisions. An archive hash is not a download URL. No public location or unrestricted model access is invented.

| Operation | Scope |
|---|---|
| Default checks | Standard-library integrity, syntax and manuscript consistency |
| Saved-endpoint capsule | Included scalar arithmetic; no matrix or model recomputation |
| Centering diagnostic | Included C7/D4 banks plus NumPy; original, fixed-metric centered and rewhitened treatments |
| Other coordinate analyses | Consult individual input manifests; a coordinate bank does not guarantee all auxiliary inputs |
| New activation extraction | Requires model access, recorded prompts/environment and compute; not done for v6 |

Centering diagnoses the retained PCA subspace. Fixed-metric centering preserves the original whitening; rewhitening changes the metric. Spectral ranks are not exact-action optimum ranks or evidence of native model computation.

Public attribution is preserved. Owner path maps, editorial backups and build logs are outside the anonymous bundles.

The coordinate capsule also excludes separately authored-context coordinates, S3 source-embedding potential inputs, neural potential fits and full-hidden interface checkpoints. Their retained outcome tables remain available, but these particular stages require additional original inputs. See data/retained_coordinates/README.md for the detailed distinction.
