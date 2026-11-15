# Frozen evidence included in this submission

This directory contains historical measurements from completed runs. It contains no new model execution, refitting, synthetic observations, or changed confidence intervals. All publication assets are rebuilt from these included files by `scripts/build_paper_assets.py`.

`frozen/` contains byte-for-byte copies of selected original tables and configurations. `derived/` contains new reporting subsets or descriptive aggregates. Every derivative is declared in `registry.json` with its original source-file SHA256, exact filter/aggregation, retained columns and row count. `provenance/*.public.json` are explicitly sanitized metadata derivatives: they are not the original run manifests. The original manifest hashes are retained.

## Evidence families

| Family | Run | Execution source commit | Included material |
|---|---|---|---|
| FULL | `main_full_003` | `30f3c6369a185f43e94b17ca03b0499cd5d65a78` | Primary structural template-level subset; all registered confidence intervals; cross-model metrics; descriptive causal transformation means and recorded continuation counts; frozen configuration |
| AFS A/B/C | `afs_full_abc_001` | `9a491f89d5adf48c3119a89ec8ce1b215ec7802d` | Test unseen-composite and correspondence-scramble subsets; registered generator/scramble intervals; all permutation summaries; frozen configuration |
| Rank-aware AFS D | `afs_d_rank_aware_001` | `1b7adfe717c6a7663c90e0a7c31d8a612f22afe9` | All 4,000 location observations, 150 paired depth-contrast intervals, 20 measured embedding-rank diagnostics, frozen configuration |
| Global additive field | `additive_field_falsification_001` | Execution script identified by SHA256 in the registry | Matched operator/action aggregates; five split-seed aggregates/contrasts; all 12 registered intervals; D4 order diagnostics; implementation configuration |

The source-conditioned additive behavior comparator in FULL and the global additive field are different estimators. They must not be combined or referred to as one baseline.

## What can be reproduced from this directory

The clean standalone project can regenerate the manuscript's summary tables, descriptive figures, original-interval plots and figure-source CSVs without network access or models. Included template-level structural and AFS subsets permit direct verification of their reported means. Included AFS-D observations permit paired location analyses; the original intervals remain frozen. Included global-additive seed contrasts permit independent reproduction of its registered five-seed bootstrap.

The tables do not permit independent recovery of omitted forward passes, activation caches, operator fits, raw behavioral pairings, or AFS-B's 2.4 million permutation draws. The public package therefore does not by itself reproduce model execution or rederive every registered interval. Cross-model metrics describe fitted in-sample commutation compatibility; they do not establish invertible full-representation equivalence.

## Omitted historical archives and availability

`registry.json` identifies six known historical archives by filename, byte count, SHA256, file-entry count and inventory category. These include the FULL raw activation/checkpoint archive, full scientific results, AFS ABC results, rank-aware AFS D results/raw output, and the reissued global-additive results. They are omitted to keep the submission compact. Their existence was checked in the local scientific audit. No public download URL, permission to redistribute model weights, or public availability of these archives is asserted. A public repository release must supply a real repository URL and any intended raw-data release separately.

Historical failed AFS-D at dimension 16 is not the successful rank-aware run included here. Its failed-run checkpoint directory was unavailable during audit. Rank-aware AFS-D uses C7 dimension 6 and D4 dimension 3 at both embedding and internal locations; it must not be plotted as FULL dimension 16.

## Verification and interpretation

Run `python scripts/verify_frozen.py` after building the assets, or add `--registry-only` to check evidence before building. Checksums verify correspondence to this portable registry. They do not prove the truth of an experiment or independently authenticate the omitted execution environment.

The historical FULL ledger is complete, but its overall `GO` flag was unconditional after execution; scientific component gates and numerical contrasts must be assessed separately. AFS's `STRONG_PASS` is a frozen decision under its specified controls, not a test against every categorical-code explanation. The global additive estimator averages complete finite orbits, so it cancels to zero by construction and supplies the identity prediction. AFS-D's lexical result and the strong source-conditioned additive comparator are essential limits on stronger mechanistic interpretations.

Field meanings and measurement conventions are documented in `DATA_DICTIONARY.md`. Row-level and generated-file checksums are tracked separately in `registry.json` and `release/artifact_manifest.json`.
