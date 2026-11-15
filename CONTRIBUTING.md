# Contributions

Keep scientific objectives, normalization, splits, tolerances and source locks
unchanged in editorial changes. Place new authorized analyses in fresh outputs.
Never replace failed cells or infer successful runs from available code alone.

Run `python3 -B scripts/verify_delivery.py` for packaging changes. It is read-only
and uses no scientific dependencies. Packaging-only tests live in
`tests/test_delivery_verifier.py`; do not run the whole scientific suite merely
to validate an editorial change. Scientific execution is a separate explicit
action. Retain provenance and license notices.

Keep edits to presentation scripts distinct from their historical renderer source.
Record original and derivative hashes when changing captions or layouts; preserve
numerical table bodies and full-precision inputs. Never replace an old script hash
with a new one to imply that an old artifact was regenerated.

Build and publication status belongs in `release/current_release_status.json`.
Do not copy a historical PASS to a new manuscript. Any current exported PDF needs
a successful fresh compilation and source/PDF hash binding. The author decides
when final scientific review and submission disclosures are complete.
