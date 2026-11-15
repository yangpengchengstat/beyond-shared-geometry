# Preservation check for v0.5.7

The complete v0.5.6 public-project baseline was freshly compared with the pre-edit SHA256 snapshot: **1023 files, zero mismatches in the original directory**. The independent v0.5.7 copy retains **538 inherited scientific files byte for byte**, with zero mismatches. New result imports and newly formatted publication artifacts are separate additions.

Baseline: `incoming/bsg_tmlr_revision_20261106_v056/beyond-shared-geometry`, relative to the original repository. The original 68-page paper remains there and is also copied under this delivery's `baseline/paper/`. Prior release documentation and packages are preserved in `release/history/before_0_5_7/`.

| Frozen inherited scope | Files checked | Mismatches |
|---|---:|---:|
| `results/` | 120 | 0 |
| `src/` | 78 | 0 |
| `configs/` | 10 | 0 |
| `studies/` | 148 | 0 |
| `analysis/` | 14 | 0 |
| `tests/` | 41 | 0 |
| `runpod/` | 25 | 0 |
| `paper/generated/` | 39 | 0 |
| `paper/figures/` | 14 | 0 |
| `paper/supplementary/` | 18 | 0 |
| `review_reproduction/` | 13 | 0 |
| `repair_v2/` | 17 | 0 |
| `inputs/` | 1 | 0 |

The checks cover original numerical results, code, configurations, study protocols, tests, generated tables, figures and supplements. Tables 3–6 and the prior 8/16 sensitivity table retain their original numerical sources. Original results are neither overwritten nor re-fitted. The old generated core table remains, while the manuscript now includes a separately named two-panel core table.

The completed comparator's 112 selected original evidence files are copied without byte changes; `results/orthogonal_comparator_v1/SOURCE_BINDINGS.json` records each original path and checksum. Copied log extensions are `.txt` to ensure release inclusion; their bytes and original `.log` provenance are unchanged. Original full result arrays, runtime snapshots, logs and all failure records stay in the prior completed-run delivery. The public import is an explicit subset, not a claim to reproduce every diagnostic array.

The prior large private research archive at `incoming/bsg_latest_complete_20261106/research_archive` remains in place. Its earlier preservation verification is retained; this editorial round does not claim a new complete hash pass over its 14.9 GB contents. Existing historical rank, calibration, native-program and preparation/authentication stops remain in the prior evidence tree. No original history, code protocol or failed result was repaired or relabeled.

Machine-readable preservation checks, old-delivery verification logs, the paper diff and final delivery checks reside under `audit/` in the independent delivery. New evidence subsets have their own `EVIDENCE_SHA256.json`. This revision runs formatting, compilation and integrity checks only: no numerical tests, matrix analysis, model forward, download, RunPod job, retraining, or experiment rerun.
