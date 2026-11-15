# Source and scientific-result preservation report

Checked on 2026-11-06 at 19:52 UTC. Status: **all declared preservation checks passed; zero mismatches**.

This is a bounded integrity check, not a claim that the edited working project is unchanged. Manuscript prose, bibliography, documentation, release metadata, and newly added files are intentionally allowed to change. No experiment or model execution was performed by this check.

## Baseline and original workspace

The recorded baseline is the 967-file public source project at:

`incoming/bsg_latest_complete_20261106/github/beyond-shared-geometry` (relative to the original repository)

The independent editable copy is:

the separate `beyond-shared-geometry` working directory of this revision; its local path is recorded in the private audit

Every baseline file in the **original public source project** was freshly SHA256-checked against `audit/baseline_source_manifest.json`: all **967 files / 90,700,273 bytes** match. The baseline manifest SHA256 is `4a13015d34d3852756801e9e8c72ea4dca863c9246f0c1fff52c627e75186c2b`.

The actual containing Git root is the original `beyond-shared-geometry` repository, verified by Git rather than inferred by counting parent directories; its exact local path is retained in the private machine-readable audit. HEAD remains `1b7adfe717c6a7663c90e0a7c31d8a612f22afe9`. Current HEAD output, short status and tracked binary diff are byte-identical to the captured `git_head.before.txt`, `git_status.before.txt` and `git_diff.before.patch`. Existing user modifications remain present; no reset, clean, commit or push was performed. Matching Git status alone does not verify untracked file contents; the separate file checks above cover the baseline public files explicitly.

## Frozen scientific materials in the editable copy

Every pre-existing file in these scopes was freshly compared with the baseline SHA256:

| Scope | Files | Mismatches |
|---|---:|---:|
| Saved numeric results, registries and result provenance (`results/`) | 120 | 0 |
| Scientific implementation (`src/`) | 78 | 0 |
| Original configurations | 10 | 0 |
| Pre-existing study implementations and protocols | 138 | 0 |
| Pre-existing analysis code | 13 | 0 |
| Original tests | 41 | 0 |
| Historical RunPod source and launcher records | 25 | 0 |
| Entire existing anonymous saved-endpoint capsule | 13 | 0 |
| Original generated TeX tables/macros | 34 | 0 |
| Original figure assets | 14 | 0 |
| Original full-precision manuscript supplement | 13 | 0 |

These scopes contain **499 distinct files / 80,093,070 bytes**, all unchanged. This covers original Table 3/4/5/6 numerical bodies, the completed 8/16 Table 32 evidence, the saved endpoints underlying their summaries, numerical protocols, and the full existing anonymous reproduction capsule.

The new `paper/generated/core_risk_summary.tex` and its source mapping are separate publication artifacts. They do not replace the original generated tables. The newly added `studies/orthogonal_comparator_v1/` is prepared code/protocol, not an inherited frozen result or a completed real-bank experiment; it is explicitly listed as new rather than falsely compared with a baseline entry. Release and editorial updates are excluded from the unchanged-science claim.

## Original raw results and failure archives

This revision does not rewrite the complete private archive at:

`incoming/bsg_latest_complete_20261106/research_archive` (relative to the original repository)

The existing `manifests/private_snapshot_verification.json` records the earlier complete preservation check: **34,447 files / 14,899,667,345 logical bytes**, independent APFS clones, matching source/destination hashes, independent destination inodes, and unchanged source metadata. The previous final delivery verification also records a full recheck of those 34,447 frozen files. This report references those existing checks; **it does not claim to have freshly rehashed the full 14.9 GB archive**.

The large file manifest was freshly hashed and still matches its recorded SHA256, `468d8279fee8f3ff3c2f0150557d335f3642917fdae4f405e3a38a3c524d6d4a`. A current existence/byte-size check also passes for the six principal archive objects: rank recovery, TMLR follow-up, follow-up audit logs, the BF16 failure return, the BF16 backup, and the original completed FULL return. Their previously verified hashes are retained in the machine-readable report. Existence/size checks are metadata checks, not new content-hash verification of those large payloads.

Original rank infeasibility, calibration/native/pilot scientific stops, preparation/authentication failures, and their existing summaries remain in that preserved evidence tree. The earlier documented gaps in some early C7/D4 original uploads and in same-bank wider Study 2 coordinates are not repaired or relabeled by this packaging operation. A preserved later audit or replay is not asserted to be the missing original upload.

## Verification artifacts and limits

Machine-readable results are saved as `audit/preservation_scientific.json` alongside the revision audit. It records each checked scope, source and destination roots, mismatch lists, Git comparisons, the earlier archive-verification records and current large-archive metadata checks.

The manuscript and release files receive intentional editorial/build updates. Their compiler/package checks are recorded separately from preservation of the scientific inputs. The present successful preservation result neither asserts new experimental validity nor substitutes for final author review.
