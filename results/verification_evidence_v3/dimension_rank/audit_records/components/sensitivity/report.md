# Existing dimension-sensitivity audit

This audit summarizes existing measurements only. It performs no model forward, operator fit, PCA fit, bootstrap, new dimension search or change of archived values. Original and replay runs remain separate throughout.

## Evidence and coverage

The original authoritative structural run is `main_full_003`, with source commit `30f3c6369a185f43e94b17ca03b0499cd5d65a78` and effective configuration hash `910d430f18c5f865d88948c3284e0c9eaea531323aaf9c697ad2399cbeb372dd`. Its complete raw archive is available within this repository at:

- `runpod_output_by_date/result_9_6_2026/run_outputs_9_4_2026_full3.zip`
- Archive root: `run_outputs/main_experiment/main_full_003/`.
- Detailed measurements: `structural/template_metrics.csv`.
- Completion and checkpoint hashes: `run_ledger.json`.
- Aggregate hash and scientific run metadata: `manifest.json`.
- Recorded design: `config_snapshot.yaml`.

The retained computational replay is separately available at:

- `submission/archive/runpod-from-zero-v1/2026-10-22-full-completed/full-results-v1.tar.gz`
- Archive root: `historical/runs/legacy_full/`.
- Same relative detailed measurements and ledger; effective snapshot is `config_snapshot.json`.

The archive checksums, member checksums and pre/post preservation checks are in `output/input_preservation.json` and `output/member_hashes.json`. `output/verification_summary.json` gives exact cross-check counts. Files under `runpod_output_by_date` are ignored by ordinary `rg --files`; they were located using `rg --files --hidden --no-ignore` within the authorized project, not by searching outside the repository.

The configuration explicitly sets `sensitivity_latent_dimensions: [8,32,64]`, `sensitivity_depth_slots: [0.40,0.80]` and `sensitivity_split_seeds: [101,303]` (`configs/main_experiment.yaml:153–157`). The runtime cell selector in `src/bsg/experiments/main_experiment.py:226–238` enforces this reduced grid. These are **executed, complete and traceable**, not merely configuration declarations, in both original and replay records:

- Six base models: Llama-3.2-1B/3B, Llama-3.1-8B, Qwen2.5-1.5B/3B/7B.
- Both C7 and D4 domains.
- Two nonterminal depth slots, 0.4 and 0.8; corresponding block layers are 6/13, 11/22, 13/26, 11/22, 14/29 and 11/22, respectively.
- Split seeds 101 and 303; validation and test banks remain separate, each with six templates.
- Independent Ridge, Joint, identity and five random orthogonal controls (seeds 104729, 130363, 155921, 196613, 262147); Joint's optimization seed remains 42.
- At each q in 8/16/32/64, the matched grid contains 48 base units and 48 Joint units per run. All are marked COMPLETE and all corresponding CSV, NPZ and irrep-CSV hashes agree with the retained ledger.
- Each run contributes 18,432 selected per-template rows and 3,072 aggregates of six original template values. Every selected aggregate value was checked against its checksum-verified per-checkpoint CSV; only CSV serialization differences up to 4.44e-16 were observed. The resulting CSV has 6,144 rows across the two runs.
- No missing or failed unit occurs **within this declared matched sensitivity grid**. This does not remove unrelated historical failures elsewhere in the project.

## Why the existing ready-made sensitivity table is not a matched comparison

`results/tables/latent_dim_sensitivity.csv` groups all numeric-depth test rows by model, domain, method and q. The table generator (`src/bsg/experiments/main_experiment.py:904–905`) does not restrict the q=16 rows to the two-depth/two-split sensitivity subset. Its q=16 mean therefore uses broader coverage. It also averages metadata columns such as layer and split seed, which should not be interpreted as individual conditions.

This audit instead matches q=16 to the exact model/domain/layer/depth/split/bank/method/control-seed/template IDs used at q=8/32/64. Terminal readouts and the extra q=16 depths/splits are excluded from **this matched comparison only**, and remain preserved in the original evidence and primary manuscript results. `dimension_sensitivity_existing.csv` retains per-condition values, matched q=16 values and arithmetic differences. `dimension_sensitivity_summary.csv` averages these matched conditions, with equal template and control-seed weights. There are 24 model/depth/split conditions per domain.

## Findings and limitations

The original-run matched Joint test means for E_eq are:

| Domain | q=8 | q=16 matched | q=32 | q=64 |
|---|---:|---:|---:|---:|
| C7 | 0.116178801 | 0.052251168 | 0.050733031 | 0.135502090 |
| D4 | 0.056168531 | 0.061449638 | 0.071429361 | 0.135591392 |

The corresponding original-run matched Ridge test means are:

| Domain | q=8 | q=16 matched | q=32 | q=64 |
|---|---:|---:|---:|---:|
| C7 | 0.115638296 | 0.051285323 | 0.050219966 | 0.136808271 |
| D4 | 0.055581346 | 0.060366010 | 0.070745070 | 0.135896093 |

These values do not support the claim that q=16 is uniformly best. C7's recorded direct-prediction mean is slightly lower at q=32; D4's is lower at q=8. q=64 worsens direct prediction in both domains. Group-law diagnostics vary by endpoint as well. These are descriptive comparisons, not statistically calibrated dimensional-optimality tests or a rule used to revise the selected primary dimension.

The original error normalizations are retained: E_eq is the nonidentity-action average of rowwise squared error divided by target squared norm plus 1e-12 (`src/bsg/metrics/core.py:7–19`, `equivariance.py:8–15`); consistency/relation endpoints retain their original targets, action sets and domain-specific aggregands. Operator defects use the original unsquared relative Frobenius norm. The audit neither compares their magnitudes as if they were the same metric nor adds a new common denominator. Stored `pca_variance` is retained as a recorded endpoint: `src/bsg/representations/pca.py:25–31` uses the full original training SVD spectrum as its denominator. This file is byte-identical to commit `30f3c6369a185f43e94b17ca03b0499cd5d65a78` (SHA256 `4836f1762b9b51a58fb0ac44106ffdeae55adcadcccbb3dd1ad82e98af412799`). It is not recomputed from truncated-coordinate variance.

The replay has the same matching grid and nearly identical means but is a computational replay of an inspected design, not an independent dataset. Both records are retained separately; they are never pooled to increase apparent replication.

The historical dimension sweep addresses structural prediction, group-consistency/relation/inverse diagnostics, and related recorded endpoints. The causal and cross-model settings are fixed at q=16 (`configs/main_experiment.yaml:158–159`). Later finite-bank and witness protocols declare `rank: 16` (`studies/spectral_finite_bank_v1/protocol.json:104–106`, `studies/spectral_extensions_v1/protocol.json:36,58` within the current release). The historical sweep **does not establish dimension stability of gamma_r, the exact-action risk premium, registration rankings, witness transfer or authored-context transport**. The optional family-deletion/LOFO analysis also remains unexecuted; it is not a dimension sweep.

## Suggested concise English manuscript text

> The primary retained dimension is fixed at q=16. Existing structural sensitivity runs use q=8,32,64 on the six-model subset at depth slots 0.4 and 0.8 and split seeds 101 and 303. We compare them with q=16 restricted to the same model, depth, split, template and control-seed conditions. The matched results show endpoint- and domain-dependent dimension effects rather than a uniformly optimal q: for example, Joint prediction errors at q=8/16/32/64 are 0.11618/0.05225/0.05073/0.13550 for C7 and 0.05617/0.06145/0.07143/0.13559 for D4. These checks concern the historical structural losses; the later spectral and registration results remain restricted to r=16.

The existing `paper_editing/sections/methods.tex:51` sensitivity sentence has the correct reduced-grid settings but currently supplies no matched numerical comparison or endpoint restriction. An appendix table can report the evidence above, with a brief forward reference from that paragraph. Avoid describing all later conclusions as dimension-robust, and do not replace the primary q=16 panel's existing figures or values with these smaller-subset means.

## Reproduction outputs

- `audit_existing_sensitivity.py`: read-only standard-library aggregation and verification script.
- `commands.txt`, `execution.log`: actual successful command and verification output.
- `output/dimension_sensitivity_existing.csv`: 6,144 traceable condition aggregates with exact matched-q16 values and differences; all existing validation/test values retained separately.
- `output/dimension_sensitivity_summary.csv`: 128 matched descriptive summaries; original and replay separate.
- `output/dimension_sensitivity_coverage.csv`: eight run-by-q coverage records.
- `output/input_preservation.json`: both source archive hashes unchanged before/after.
- `output/member_hashes.json`: the precise archive members read and validated.
- `output/metric_definitions.json`: endpoint normalization scope.
- `output/verification_summary.json`: complete-unit/row/hash-check counts.
- `output/missing_and_failure_records.json`: empty because no failures/missing cells occur in this matched grid; not a statement about other experiments.

Two earlier audit-script execution logs are preserved: exact-serialization equality was relaxed solely to accommodate historical CSV decimal round trips, and an inefficient archive-read pass was interrupted in favor of sequential hashing. Neither changed source inputs, scientific thresholds or measurements.
