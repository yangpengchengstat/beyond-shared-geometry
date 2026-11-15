# Historical rank-audit records (2026-11-05/06)

This is a bounded, portable copy of the source and evidence behind the saved rank
audit CSVs in the parent directory. It preserves the original audit report, the
commands that were recorded at that time, the numerical rules, the numerical
source programs, component reports, input/member hashes, preservation checks,
and numerical execution/failure logs. **These programs were not executed during
the present packaging revision.**

The report's references to its then-current manuscript, compiled PDF, temporary
directories, or an unexecuted LOFO analysis describe that historical revision.
They are not current release checks. In the current project, the independent
orthogonal comparison, 8/16 spectral comparison, and C7 r=16 canonical-action
family-risk LOFO are completed; all-image registration LOFO remains unexecuted.
Current build status is maintained separately in the project's release records.

## Reading order

1. `DIMENSION_RANK_AUDIT.md`: dimension-choice evidence, training-rank conclusion,
   available inputs, missing arrays, and the later 8/16 update.
2. `audit_protocol.json`: the recorded numerical rules and their scope.
3. `components/lexical/report.md` and `full_input_addendum.md`: saved lexical and
   internal training-input evidence, including limitations of absent PCA scores.
4. `components/history/report.md` and `history_evidence.json`: recorded source/Git
   history evidence. The linked source identifiers refer to preserved originals.
5. `COMMANDS.md`, component command files, `logs/`, and `provenance/`: historical
   execution records and input identity/preservation evidence.

Every copied file is listed in `source_manifest.json`, with the original and
packaged SHA256. Personal/local path prefixes in 13 records have been replaced
with declared logical aliases. Other text, numerical values, missingness, source
hashes, and statuses are preserved. All 46 other records are byte-identical.
Original byte copies and absolute-path mappings are retained only in the enclosing
project's private evidence directory, not in public or anonymous material.

## Reproduction boundaries

The `scripts/` and `components/*/*.py` files retain the historical computations
and their parameter interfaces. They inspect existing arrays; they do not obtain
missing arrays or download models. A numerical rerun would require the original
hash-identified archives, NumPy, and a new output directory. Those large archives
and raw activation arrays are not included here. No rerun is required to inspect
the delivered full-precision CSVs, and no rerun is part of a packaging check.

The preserved `scripts/run_readonly_audit.sh` is an archival launcher for the
**original audit directory layout**, including its then-current `paper/` tree.
It is not an entry point for this bounded evidence copy. Likewise, the recorded
paper compilation and editorial commands are provenance, not instructions to
reapply old manuscript patches. Their original paths are now logical identifiers.
The current project's static evidence verifier is
`results/verification_evidence_v3/verify_evidence.py`; it performs no numerical
analysis and requires no external arrays.

The 300 historical retained-coordinate entries marked `unavailable` remain
unavailable. The retained report explains why the separately verified training
input spectra and recorded leading-component PCA algorithm support the narrower
training-rank statement. Copying this evidence does not recover historical arrays,
establish rank on every held-out bank, or certify population generalization.
