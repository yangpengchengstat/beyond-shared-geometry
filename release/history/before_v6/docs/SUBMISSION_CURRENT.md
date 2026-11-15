# TMLR submission workflow

Current source: `paper/main.tex`. This is a source submission candidate, not a
completed journal submission. The actual build and PDF review results are in
`release/current_release_status.json`; author review remains pending.

## Which delivery file to use

| Delivery archive | Use |
|---|---|
| `overleaf_source.zip` | Import as a new Overleaf project; `main.tex` is the root entry |
| `anonymous-review-source.zip` | Anonymous manuscript source plus supporting saved-result evidence and reproduction code |
| `github-source.zip` | Public, attributed research repository; never use it as an anonymous review attachment |
| Full project archive | Private owner backup, including verification and original material; do not upload this archive to reviewers or GitHub |

1. In Overleaf select `main.tex`, choose pdfLaTeX, and **Recompile from scratch**
   so bibliography output is regenerated. Local pdfLaTeX + BibTeX commands, if
   those tools are already installed, are in `COMPILE_CHECK.md`.
2. Download the fresh PDF and complete compilation log. Inspect the body,
   changed appendices and references: missing glyphs, unreadable plot labels,
   clipped tables, unresolved references/citations, float placement and anonymity.
   Table captions belong above their tables; figure captions belong below figures.
3. Confirm the final author list, affiliations, conflicts, funding, relevant ethics
   disclosures, rights to publish, prior/overlapping submissions and scientific
   claims. Complete the necessary OpenReview profiles and form fields.
4. Submit the anonymous PDF and the dedicated anonymous supporting ZIP. Do not
   submit personal audit folders, original handoff files or public author metadata.
   Do not link the identified public repository from the anonymous manuscript.
5. Retain the exact submitted PDF/ZIP/log and checksums as the submission snapshot.
   Mark readiness only after those steps are checked; there is no automatic
   promotion from a static-verification PASS.

## Applicable requirements

Official pages reviewed for this delivery on 2026-11-08:
[TMLR submissions](https://jmlr.org/tmlr/submissions.html),
[author guide](https://jmlr.org/tmlr/author-guide.html), and the
[official template](https://github.com/JmlrOrg/tmlr-style-file).
Use the current official template without layout overrides. The manuscript is
anonymous; supplementary PDF/ZIP material must also be anonymous and within the
100 MB limit. Confirm the actual archive size in the delivery manifest. There is
no fixed body-page limit, but the manuscript should justify its length.

TMLR's manuscript license requires the authors' CC BY 4.0 agreement. The software's
MIT license is separate and does not provide that author confirmation. Check the
current official submission form at submission time; this checklist is not an
OpenReview form submission or a guarantee of editorial acceptance.

## Scientific status retained

Completed: independent orthogonality comparison, 8/16-dimensional analysis and C7
canonical-action family-risk LOFO. Prepared but not run: all-image registration
LOFO. Historical scientific stops remain recorded. No new experiment is a release
gate for these editorial/package repairs.
