# Delivery validation commands — 0.5.5

Run from the public project root. These are source/evidence/packaging checks only;
none starts a model forward or a numerical study. The full Python source tree was
also parsed with the standard-library AST parser.

```bash
python3 scripts/verify_frozen.py
python3 scripts/verify_bounded_revision.py
python3 scripts/check_saved_delivery_pdf.py
python3 scripts/check_release.py
python3 scripts/package_release.py
```

The PDF checker uses pdfplumber and pypdf in the existing document runtime.
The original `scripts/check_pdf.py` requires PyMuPDF, which was unavailable in
that runtime; its attempted invocation stopped before reading or changing files.
The alternative checker verified the unchanged saved 65-page PDF, author
metadata, text bounds, main-table position, references and current source hashes.
Saved prior compile and rendered-page inspection are quoted separately; no fresh
LaTeX compilation is asserted for this packaging copy.

The first frozen verification identified supporting publication assets missing
from the lean manuscript copy; these were restored byte-for-byte from the 0.5.4
public baseline. The first release checks identified old engineering manifest
hashes for packaging/version metadata and three already edited paper assets.
The old manifests and those original assets were retained under
`release/history/before_0_5_5`, and current metadata distinguishes the newer
editorial assets. Historical numerical source, frozen evidence, results and
scientific inputs were neither replaced nor recomputed.

Table 32 was regenerated only as TeX formatting from saved coverage.json and
compared byte-for-byte to the saved manuscript table. This is a publication
check, not repetition of the 6,000-unit matrix analysis. Existing anonymous
Tables 3-6 capsule files remain byte-identical to the prior release.

The optional full CPU test suite and numerical study fixtures were not rerun
in this packaging task: numerical code was preserved by hashes. The existing
completed study and its numerical-test evidence remain in the local archive.
Source/ZIP checks do not establish scientific validity or final author approval.
