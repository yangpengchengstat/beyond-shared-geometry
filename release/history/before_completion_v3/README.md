# Beyond Shared Geometry

Complete research-source candidate with the verified editorial manuscript,
preserved results and MIT-licensed research code. Delivery revision: 20261108-v2;
scientific software baseline: 0.6.3. This is not a journal acceptance or a new
scientific release. No experiments were run in preparing this delivery.

## Start here

- Current manuscript: `paper/main.tex` (the only active paper entry).
- Safe, dependency-free verification: `python3 -B scripts/verify_delivery.py`.
- Saved-endpoint reproduction: `REPRODUCE_CORE_RESULTS.md`,
  `review_reproduction/README.md`, and `paper/supplementary/`.
- Research implementations: `src/`, `studies/`, `analysis/`, `configs/`.
- Retained measurements and failures: `results/`, with identities in
  `results/registry.json` and the source bindings next to each study.
- What is and is not included: `docs/DATA_AVAILABILITY_CURRENT.md`.
- Author submission steps: `docs/SUBMISSION_CURRENT.md`.
- Publication instructions: `docs/GITHUB_CURRENT.md`.

## Completed evidence

The five follow-up studies, 8/16 dimensional sensitivity (6,000 image units),
independent orthogonality comparison (96 units), and canonical-action C7
family-risk analysis (1,536 deletions) are retained. All-image registration
LOFO remains prepared and unexecuted. Historical failures are not passes.
C7 heterogeneity and the re-whitening limitation remain in the manuscript.

## Validation and reproduction

`make check` performs only checksums, source syntax and static source checks;
it does not run numerical experiments, tests requiring scientific dependencies,
model forwards, or downloads. CI uses the same restricted check. Prior test and
build reports are historical evidence, not current execution claims.

The existing scientific tests and opt-in study launchers are included unchanged.
Read their protocols before invoking them; they are not required for reviewing
saved results. Dependencies and exact model revisions remain in the existing
project and study manifests. No installation was performed for this delivery.

No newly compiled PDF is supplied. Local TeX and final ZIP recompilation are
BLOCKED because tools are unavailable and installation was declined. Compile
the separate anonymous Overleaf package from scratch, with pdfLaTeX and BibTeX,
and review the generated PDF and logs. Old PDFs under release/history are
references only. `submission_ready=false`; author review remains pending.

The full GitHub project contains author/license metadata. It is not anonymous
review material. Upload only the dedicated anonymous bundle to reviewers.
Publication, repository creation and submission require the author's action.

## License

Existing software remains under the MIT license in LICENSE. TMLR template
license is retained at paper/TMLR_TEMPLATE_LICENSE.txt. Third-party model
weights and research papers are not relicensed or bundled as public software.
See THIRD_PARTY_NOTICES.md. No DOI or public repository URL is invented.
