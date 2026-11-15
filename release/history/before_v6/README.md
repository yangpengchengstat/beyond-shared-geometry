# Beyond Shared Geometry

Research code, the current TMLR manuscript, and preserved evidence for predictive
recovery, exact orthogonal-action risk, and semantic registration in language-model
representations. Delivery revision **20261108-v5** retains the scientific software baseline **0.6.3**.
This revision adjusts spacing and the placement of Table 7 and Figures 3 and 9.
The owner delivery pairs the source project with a separately compiled PDF.

## Start here

| Purpose | Entry |
|---|---|
| Read or edit the manuscript | [`paper/main.tex`](paper/main.tex), the only active manuscript entry |
| Inspect release/build status | [`release/current_release_status.json`](release/current_release_status.json) and [`COMPILE_CHECK.md`](COMPILE_CHECK.md) |
| Verify the delivered files | `python3 -B scripts/verify_delivery.py` or `make check` |
| Find data for paper results | [`REPRODUCE_CORE_RESULTS.md`](REPRODUCE_CORE_RESULTS.md) |
| Understand included and unavailable data | [`docs/DATA_AVAILABILITY_CURRENT.md`](docs/DATA_AVAILABILITY_CURRENT.md) |
| Prepare an anonymous submission | [`docs/SUBMISSION_CURRENT.md`](docs/SUBMISSION_CURRENT.md) |
| Publish the public repository | [`docs/GITHUB_CURRENT.md`](docs/GITHUB_CURRENT.md) |

The default verification uses only the Python standard library (Python 3.10+),
reads local files, and runs no scientific workers. It requires no GPU, network,
model weights or package installation. A PASS means the recorded integrity and
static-source checks passed; it is not a successful manuscript compilation or an
independent replication of measurements. CI runs this same restricted check.

## Project layout

- `paper/`: LaTeX sources, figures, generated tables and portable supplements.
- `results/`: full-precision scalar outcomes, per-condition records, historical
  failures and source mappings. `results/registry.json` locks the inherited evidence.
- `results/verification_evidence_v3/`: restored rank/dimension audit CSVs and
  calibration/power records, with original-to-portable-copy hashes and path aliases.
- `review_reproduction/`: a self-contained capsule for inspecting and reproducing
  specified saved-endpoint arithmetic; it does not redo spectral measurements.
- `src/`, `studies/`, `analysis/`, `configs/`: retained research implementations and
  frozen protocols. Their launchers are opt-in, separate from release verification.
- `release/`: current status, artifact identities and preserved historical records.
  Older validation files describe their own revisions, not the present build.

## Evidence status

The five follow-up studies, 8/16-dimensional spectral comparison, independent
orthogonality comparison and canonical-action C7 family-risk analysis are complete
in the retained records. All-image registration LOFO remains prepared but
unexecuted. Historical scientific stops remain stops. The manuscript retains C7
heterogeneity, the re-whitening limitation, and the distinction between training-
metric transfer loss and test-bank class-risk bounds.

The package contains saved numerical evidence, not model weights, every raw
activation tensor, or multi-gigabyte original archives. Historical missing arrays
remain unavailable; new files do not reconstruct them. Some computations require
undistributed archives listed in the input maps. No public data URL or unrestricted
access is asserted. See the data-availability document before planning reproduction.

## Submission and publication

Use the separately supplied `overleaf_source.zip` for manuscript compilation and
`anonymous-review-source.zip` for anonymous supporting code/data. The complete
project contains author and license metadata and must not be uploaded as an
anonymous review attachment. The package directory's `dist/` contains the ready
source bundles and their checksums; these archives are separate from this public
repository tree.

The current status file records the v5 Tectonic compilation and layout review.
Author review and disclosures are pending.
No GitHub repository, Overleaf project or journal submission has been created.

## License

Project software retains the MIT license in [`LICENSE`](LICENSE), with its existing
author attribution. The TMLR template license is in
[`paper/TMLR_TEMPLATE_LICENSE.txt`](paper/TMLR_TEMPLATE_LICENSE.txt).
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) identifies separate third-party
materials. Model weights and reference papers are not relicensed as project code.
The software license does not replace author approval of the manuscript's TMLR
license. `CITATION.cff` describes the software baseline without an invented DOI,
public repository URL or acceptance claim.
