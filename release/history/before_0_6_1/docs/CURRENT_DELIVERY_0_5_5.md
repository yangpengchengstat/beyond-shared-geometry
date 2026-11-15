# Current delivery 0.5.5

The manuscript is the saved 65-page dimension/rank revision. It has Appendix K
and Table 32 in addition to the existing audit appendix. The previous 0.5.4
metadata is preserved under `release/history/before_0_5_5`.

The 8/16 study was completed before this packaging task. All 6,000 declared
canonical/image computations are represented by compact full-precision results;
the original 16-dimensional comparisons were checked first and remain primary.
Read `results/spectral_dimension_sensitivity_v1/coverage.json` and the
interpretation summary. Do not infer that historical q=8/32/64 tests established
dimension stability of current spectral endpoints. The same-bank 32-dimensional
inputs were unavailable. No missing result is replaced by a zero, different
coordinate construction or unrecorded experiment.

The full local delivery retains original archives, code snapshots, historical
failures and complete run receipts outside the public source. The public project
contains compact evidence and source; it is not an archive of model weights or
raw activations. The data-root layout expected by the portable analysis code is
described in its README.

Submission preparation consists of the anonymous PDF, Overleaf source and
anonymous review source capsule. Tables 3-6 can be regenerated through
`review_reproduction/reproduce.py`; Table 32 through the supplementary
`generate_table.py`. These format saved endpoints and do not rerun a study.
The GitHub source intentionally retains original author and MIT license metadata
and must not be substituted for the anonymous review package.

`submission_ready` remains false pending final author review. This is a local
delivery, not an upload to GitHub or a journal submission. No new model forward,
RunPod task, numerical study, optimizer run or LOFO execution occurred during
assembly. Saved compile evidence and structural PDF checks are distinguished in
`release/pdf_qa.json` and `release/current_delivery_provenance.json`.
