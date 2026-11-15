# Beyond Shared Geometry

Research code, manuscript and retained evidence for **Beyond Shared Geometry: Predictive Recovery, Exact Actions, and Registration Preference in Language-Model Representations**. Delivery **20261113-v6** integrates the revised main text, Appendices A–N and abstract. The inherited scientific software baseline remains **0.6.3**.

## Published preprint

Pengcheng Yang. **Beyond Shared Geometry: Predictive Recovery, Exact Actions, and Registration Preference in Language-Model Representations**. Zenodo, version 1.0, November 15, 2026 (publication date displayed by Zenodo).

[Read the paper](https://doi.org/10.5281/zenodo.24238471) · [Download the included PDF](publication/bsg_arxiv_paper.pdf)

The preprint is released under CC BY 4.0. The software retains its MIT license. The paper DOI identifies the manuscript, not a separately archived software release.

## Start here

| Purpose | Entry |
|---|---|
| Read the public paper | [publication/bsg_arxiv_paper.pdf](publication/bsg_arxiv_paper.pdf) |
| Inspect the preserved submission snapshot | [paper/main.pdf](paper/main.pdf) (anonymous v6) |
| Edit or compile | [paper/main.tex](paper/main.tex), [COMPILE_CHECK.md](COMPILE_CHECK.md) |
| Verify files | `python3 -B scripts/verify_delivery.py` and `python3 -B scripts/verify_submission_surface.py` |
| Reproduce the centering diagnostic | [studies/template_centering_v1/README.md](studies/template_centering_v1/README.md) |
| Locate results | [REPRODUCE_CORE_RESULTS.md](REPRODUCE_CORE_RESULTS.md) |
| Input availability | [docs/DATA_AVAILABILITY_CURRENT.md](docs/DATA_AVAILABILITY_CURRENT.md) |
| Submission / GitHub | [docs/SUBMISSION_CURRENT.md](docs/SUBMISSION_CURRENT.md), [docs/GITHUB_CURRENT.md](docs/GITHUB_CURRENT.md) |

Default checks use standard-library Python, need no model or GPU, and execute no scientific workers. Numerical reproduction is a separate explicit command.

## Contents

- `paper/`: LaTeX, bibliography, nine figures, 37 tables, current PDF and portable supplements.
- `src/`, `studies/`, `configs/`, `analysis/`: retained implementations and protocols.
- `results/`: full-precision outcomes, condition records and historical failures.
- `data/retained_coordinates/`: 72 saved 16-dimensional banks and available PCA/row metadata, with original archive-member hashes.
- `studies/template_centering_v1/` and `results/template_centering_v1/`: portable code and saved results for original, fixed-whitening centered and rewhitened centered comparisons in Section 5.5 and Appendix I.9.
- `review_reproduction/`: saved-endpoint arithmetic capsule.
- `release/`: current status and preserved historical provenance.

No new activation extraction was performed. Saved coordinates do not recover full hidden states or historically missing projected arrays. Model weights and complete parent archives are not bundled. Read the data-availability document before claiming end-to-end reproduction.

Original scientific source locks, frozen outcomes and failures are preserved. Editorial derivatives are explicitly recorded. All-image registration LOFO remains prepared but unexecuted.

## Delivery and licensing

The enclosing delivery contains separate `dist/overleaf_source.zip`, `dist/anonymous-review-source.zip` and `dist/github-source.zip`. Upload only this public repository tree to GitHub; use dedicated anonymous materials for review. The anonymous diagnostic inputs cover C7/D4; this public tree also contains S3 banks.

The v6 paper was freshly compiled and its pages checked. The named preprint has since been published on Zenodo. The original anonymous submission snapshot and its build receipts are preserved under `paper/` and `release/`; they do not represent a journal submission or acceptance. This public repository contains author identities and must not be uploaded wholesale as an anonymous review supplement.

Software retains the [MIT license](LICENSE) and author attribution. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [template license](paper/TMLR_TEMPLATE_LICENSE.txt). Model weights and cited papers are not relicensed. The software license does not replace the manuscript's publication-license agreement. `CITATION.cff` retains software baseline 0.6.3 and supplies the published paper as its preferred citation. No software DOI has been assigned.
