LEGACY ARCHIVE INPUTS AND THE NEW PUBLIC COORDINATE CAPSULE

For the v6 template-centering diagnostic, the required C7/D4 inputs are already supplied in data/retained_coordinates/. Use studies/template_centering_v1/run.py; this entry does not need the private parent archives.

The instructions below concern legacy archive readers and broader historical workflows. The new capsule does not supply full hidden activations, model weights, separately authored-context coordinates or all auxiliary S3/interface inputs.

PRIVATE INPUTS - NOT INCLUDED IN THE PUBLIC SOURCE PACKAGE

The prepared cache-based workflows require the existing compact-evidence archive:

  from-zero-v1-rank-recovery.results.zip
  from-zero-v1-rank-recovery.results.zip.sha256

The audited archive identity recorded in docs/REPRODUCTION_REVIEW_20261024.txt is:
  6a4c38237bf1368d4c77c97504f666d0d933bfebed9417d173fbf7441c0fecef

Place your private copy here, or keep it outside the project and provide its path
to the study CLI. Verify the original sidecar before use. The code also checks
archive member, PCA, model, context and frozen-source identities. Creating a new
checksum for an unknown file does not establish that it is the audited archive.
Never edit source manifests merely to make a different input pass.

The public GitHub bundle contains code, audited aggregate evidence and historical
failure records. It does not contain this approximately 3 GB private ZIP, all raw
activations, pretrained weights or Hugging Face credentials. It is not a zero-input
one-command recreation of the original private run.

The full historical reproduction source is preserved. If caches must be rebuilt,
first reconstruct its original locked source tree into a NEW directory:

  python scripts/restore_frozen_source.py --output ../bsg-preserved-reproduction

This command copies verified source only; it does not run an experiment. Read that
restored tree's studies/reproduction_v1/README.md and, where applicable, the
rank-recovery workflow before deciding to execute it. Historical calibration and
FULL/AFS queues are costly optional reconstruction paths, not prerequisites for
the new TMLR follow-up when the audited cache is available. Known original failures
are deliberately preserved. A regenerated archive is a new run, not replacement
provenance for the original evidence; its compatibility must be reviewed.

All private files in inputs/ are excluded from the publication ZIP and ignored by
Git. Only this README is part of the public package. Result archives go into new
private run/output locations, never results/frozen/.
