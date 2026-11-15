# Supplementary saved evidence

This directory closes evidence-packaging gaps identified in the manuscript audit.
It contains copies of **previously completed** rank audits, historical dimension
sensitivity summaries, and calibration/power records. No scientific computation
was rerun to produce these copies.

## Contents

| Path | Saved evidence | Scope |
|---|---|---|
| `dimension_rank/dimension_rank_audit.csv` | 13,784 per-object records | Matrix identities, ranks, singular values, thresholds, missingness, and original audit qualifications. |
| `dimension_rank/dimension_sensitivity_existing.csv` | 6,144 historical matched-condition records | Existing structural prediction/consistency endpoints; original normalization and template identities. |
| `dimension_rank/dimension_sensitivity_summary.csv` | 128 summaries | Saved matched comparisons, including the historical dimension table. |
| `dimension_rank/dimension_sensitivity_coverage.csv` | 8 coverage records | Executed scope and excluded conditions, rather than configurations alone. |
| `dimension_rank/audit_records/` | 59 historical source/provenance records | Original audit report, commands, rules, numerical programs, input hashes, checks, and retained failure logs; not rerun. |
| `calibration/original_recipe/outputs/synthetic_positive/aggregate_summary.json` | Original calibration aggregate | Includes the original sensitivity outcome; it is not replaced by later calibration. |
| `calibration/original_recipe/stage_state/positive_sentinel_gate.json` | Original gate state | Preserves the scientific stopping decision and its error record. |
| `calibration/revised_recipe/outputs/calibration_gate/calibration_gate.json` | Revised calibration gate | Separate registered recipe and its saved status. |
| `calibration/revised_recipe/outputs/structural_power/power.json` | Structural power record | Saved scenario results and design decision; no bootstrap or simulation rerun. |
| `calibration/revised_recipe/outputs/s3_power/s3_power.json` | S3 power record | Saved family/orbit design decision and limitations. |

## What the rank files establish

These are retained audit **results**, not replacements for historical arrays.
The training rank-matched lexical comparison uses evidence from the unique lexical
vectors, retained raw-training spectra, and documented linear projection procedure.
The audit preserves 300 unavailable historical retained-coordinate entries; their
status remains `unavailable`. Inclusion here does not recover those coordinate
arrays or turn them into directly rechecked observations. Rank matching concerns
training projected representations, not equality of information content or equality
of the ranks of complete contextual hidden-state matrices.

The files preserve distinctions among retained dimension, centered activation
rank, coordinate-bank rank, fitted-operator rank, and declared theoretical action
dimension. Numerical thresholds and original input/compute precision are unchanged.
Entries without a valid original total variance retain their original `NA` values.
The historical 8/16/32/64 comparisons concern their recorded structural endpoints;
they do not establish dimension stability of spectral risk or registration ranks.
The separately completed 8/16 spectral analyses remain in the existing result
directories. No additional dimensions were searched here.

## Provenance and privacy

`source_manifest.json` lists each original source identifier, original byte count
and SHA256, packaged byte count and SHA256, and the precise copy transformation.
Eight of the nine scientific files are byte-identical copies. In the rank audit
CSV only, 404 `evidence_source` cells had a local absolute prefix replaced: 324
repository locators now use `original_research/`, and 80 historical temporary
extraction locators use `temporary_audit/`. Every other cell, including numbers, thresholds,
status, empty fields, and `NA`, is unchanged. This logical prefix identifies the
preserved original repository or historical audit working directory; it is **not**
a claim that the referenced raw archives or temporary extraction archive are
included here. Existing relative source/archive/member identifiers
are also provenance locators, not promises of bundled arrays.

The five JSONs were read as individual members of the preserved rank-recovery ZIP,
with member CRC and SHA256 checks. Packaging did not recompute the whole archive's
SHA256 or extract its model caches. Original byte copies and the private absolute
path mapping are retained in the enclosing project's private area, outside the
public repository and anonymous supplementary package.

The additional `dimension_rank/audit_records/source_manifest.json` covers the
historical report and its supporting records. Its README distinguishes that
historical snapshot's manuscript/LOFO/build status from the current release.

Raw model weights, credentials, and full activation caches are not distributed in
this directory. Missing historical arrays remain explicitly missing. This evidence
addition does not change the main results, failure records, tolerances, fits, or
the status of the still-unexecuted all-image registration LOFO analysis.

## Static verification

From the repository root:

```sh
python3 results/verification_evidence_v3/verify_evidence.py
```

This command checks packaged file identities, declared row counts, and portable
path hygiene using only the standard library. It does not execute experiments,
rank diagnostics, simulations, bootstrap, model loading, or network requests.
