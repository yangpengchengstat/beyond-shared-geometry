# Dimension-selection history audit (read-only)

Audit date: 2026-11-05. Scope: the provenance and motivation of the primary retained dimension 16; no numerical matrix diagnostics, fits, model imports, or source edits were performed by this audit component.

Paths below are relative to the repository. `R/` abbreviates `submission/tmlr_20261101_final_edit_v1/beyond-shared-geometry/`. This is a verified source mirror for the cited returned results, not a claim that its paper is the currently edited manuscript. The parent's current `paper_editing` snapshot remains the manuscript baseline.

## Answer supported by the record

**16 is a recorded fixed primary analysis setting. The original reason for choosing the particular number 16 rather than another feasible value is not recoverable from the inspected record.** The earliest available Git commit has a development grid containing 8, 16, and 32, but no primary dimension designation. A later committed design specifies 16 as primary before the final large run. It records compute control as the reason for restricting the *causal branch* to this single primary dimension. That narrower statement is not sufficient evidence that compute alone determined the structural dimension, or that 16 was optimal.

There is no recovered explained-variance rule, statistical intrinsic-dimension estimator, validation/test optimizer over dimension, or group-theoretic argument selecting 16. No claim of external preregistration is established. Local documents use “frozen” and sometimes “preregistered,” but the available evidence is a versioned local protocol, not an externally registered timestamped protocol. Earlier development experiments were already present when the first repository commit was made, so the Git record cannot certify that no prior result influenced the original choice.

## Evidence timeline and exact locations

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Git `ef5c7aa8464def6bf7acb9cd67cefae0f928402d`, 2026-09-02, `configs/main_experiment.yaml`, key `latent_dimensions: [8,16,32]` | The initial committed configuration already included 16 within a development grid. | No sole primary dimension, variance criterion, or reason for preferring 16. |
| Git `30ae942c510db1ed622c16cdbc78f2ce17b556b1`, 2026-09-02, `docs/FINAL_EXPERIMENT_DESIGN.md` at that commit, lines 332–344 (especially 336–344) | Earliest recovered explicit declaration of 16 as the primary structural and causal dimension; 8/32/64 named sensitivity settings. The causal-only restriction says “to control compute.” | Why the particular structural dimension 16 was originally chosen. |
| Same commit, design lines 317–321; current root file lines 423–427 | Train-only PCA/operator fitting; validation only for numerical/integrity diagnostics, explicitly prohibited from selecting dimension; test for final estimates. | Proof of every historical external action or pre-repository development choice. |
| Git `90c575d980723c4b1a9797fd6061038564b3dee9`, 2026-09-02, implementation commit | Implements the canonical large-experiment design, including explicit primary-dimension configuration and guard. | New evidence that 16 is statistically optimal. |
| Git `90cce9681d8a29758fc248073e5969b97cfcdd73`, 2026-09-03, post-CORE amendment; current `docs/FINAL_EXPERIMENT_DESIGN.md:97–112` | Retains full primary q16 grid; restricts q8/32/64 sensitivity to depths .40/.80 and splits101/303; explicit runtime-reduction discussion and earlier CORE inspection. | General stability of every later endpoint across dimension. |
| `configs/main_experiment.yaml:31–34,144–159` | `primary_latent_dimension:16`, full primary `[16]`, sensitivities `[8,32,64]`, their restricted depth/split axes. | Selection motivation by itself. |
| `src/bsg/experiments/main_config.py:150–161` | Runtime configuration validation enforces primary16 and the frozen sensitivity specification. | An intrinsic-dimension estimator. |
| `docs/EXPERIMENT_PROTOCOL.md:44–59` | Prohibits selecting dimension from results; primary16 and separate sensitivity grid. | Evidence of an external preregistration. |
| `runpod_output_by_date/result_9_6_2026/Beyond_Shared_Geometry_main_experiment_results/manifest.json:62,290–292,317–323` | Actual retained run `main_full_003`, `status:SUCCESS`, primary16, source commit `30f3c6369a185f43e94b17ca03b0499cd5d65a78`. | This success marker alone is not a numerical-rank test. |
| `docs/ALGEBRA_FALSIFICATION_PROTOCOL.md:23,56` and `configs/algebra_falsification.yaml:12` | AFS-A/B/C inherit q16 and import the same primary PCA/coordinates read-only from main_full_003. | AFS-D feasibility: that separate historical failure and repaired6/3 are not a q16 lexical pass. |

The current root design has append-only amendments and some older superseded grid descriptions. The final revised subset is given by the post-CORE amendment and the executable `tiers.full` configuration, not by reading the old all-dimension cross-model sentence in isolation.

## Study 2 and follow-up inheritance

1. **Characterization (Study 2): fixed q16, trained on its own training bank.** `R/studies/characterization_v1/frozen_run/study_config.json:202–203` declares `structural.q_primary=16` and an empty secondary list. Lines 265–272 explicitly describe the study as pilot-informed and say “q=16 only, matching the audited development comparisons; no new q sensitivity claims.” `README.txt:25–28,45–63` documents retaining six models, four readouts, training-only q16, excluding a new dimension sweep. This is evidence for continuity with development comparisons, not proof that q16 was selected by minimizing validation risk. Study 2's PCA is newly fitted on its own training bank, not the identical PCA from Study 1.
2. **Returned same-design replay verifies that implementation.** `R/studies/reproduction_v1/characterization.py:293–318` calls `fit_train_pca(...,train,16)`, writes 16-column coordinates, and retains PCA metadata. `R/studies/characterization_v1/frozen_run/src/bsg_next/structural.py:18–28` uses only training rows, observes the centered rank via `s > s[0]*1e-12`, rejects insufficient rank, and saves the complete singular spectrum and explained variance. No dimension-selection loop is used by the replay driver. These are source facts; numerical ranks must come from returned saved evidence, handled by the main audit.
3. **Compact nonlinear/context extension:** `R/studies/compact_context_v1/protocol.json:4–8,56–77` fixes the parent archive and `coordinate_dimension=16`, describes fixed q16 PCA as a shared common cost, and excludes raw forwards/PCA refits. Validation is explicitly used to select predictor hyperparameters (e.g., lambda/bandwidth and context objective settings); the coordinate dimension is fixed, not a validation-selected variable.
4. **Finite-bank spectral analysis:** `R/studies/spectral_finite_bank_v1/protocol.json:95–118,185–190` fixes `rank=16`, `rank_reduction=false`, `extra_centering=false` (line14), `regularization=false`, test-bank uncentered second-moment whitening and all48cells. It points to the exact characterization replay contract and archive prefix. This tests the retained q16 coordinates in a different metric; it does not select q or establish q-invariant gamma/risk/rank findings.
5. **Witness/alternative extensions:** `R/studies/spectral_extensions_v1/protocol.json:32–38,67–81` inherits the spectral protocol and `rank=16`, uses all declared train/test blocks, training-fixed whitening for witness prediction, and no outcome-based cell filtering. Finite-bank test whitening and training-whitened witness transfer are distinct targets. A documented r16-only panel does not establish robustness to any alternative r.

### Verified returned-source identity

The read-only audit directly opened the two returned ZIPs using Python `zipfile` and verified these bytes, without extracting or modifying frozen inputs:

- `incoming/rank_recovery_v1/from-zero-v1-rank-recovery.results.zip::parent-from-zero-v1/characterization/study_contract.json` SHA256 `d0e601a7d2563ec1a468db6d6f6bade1776e6b5f167bb0741592122482e53bfb`.
- The contract's `driver_sha256` equals the release replay driver hash `dd3d661c12d8f6279675775e4075a3e28d3c3c7df7b2577143965c636bb7fe01`.
- Its `reference_configuration_sha256` equals the release fixed configuration hash `25654bb404fea71201a2225baf2f7f5603569e165b3732c852ddca1a88f4bd04`.
- Its `frozen_source_manifest_sha256` equals the release vendored numerical manifest `5832afe9f3d31b4962ab30ba6c717f151bcb99a14b23ae25f146108a3396f739`.
- Returned TMLR archive `source_snapshot/studies/spectral_finite_bank_v1/protocol.json` is byte-identical to the cited release, SHA256 `fd6211393971bb1e402426a6309432a70c1761630a6d5a42c91022159e02ee90`.
- Returned `source_snapshot/studies/spectral_extensions_v1/protocol.json` likewise matches, SHA256 `a939b4f07c16e8f6c232186c04760636d36848c572cb95f98e9e8c418dbd0d88`.

The replay contract itself states `prior_outcomes_inspected:true`, `independent_holdout:false`, and `scope: Complete same-design descriptive characterization replay; not old-run continuation or new confirmation.` Thus a replay is not a fresh independent dimension-choice confirmation.

## Recommended manuscript wording (evidence-backed)

> The primary retained dimension, q=16, is a fixed analysis setting carried from the structural protocol into the later predictor and spectral analyses. It is not an estimate of intrinsic dimension, and the available records do not establish that it is optimal. The structural protocol fixes this value and reserves other dimensions for specified sensitivity checks; validation and test scores do not select q in the audited implementation. Study 2 fits its own training-only PCA at the same dimension, and subsequent coordinate-only analyses reuse those coordinates.

If space is tight, put the implementation/selection statement and lineage details in the methods appendix. Keep “available records do not establish” rather than asserting universal absence of earlier development selection. Avoid “95% retained variance,” “validation-optimal,” “group theory requires16,” externally “preregistered,” or “all conclusions robust across dimensions.” Report actual explained variance separately if the main matrix audit can recover its complete training spectra.

## Reproduction and limits

`audit_dimension_history.py` only reads named source/config/manifest files, Git objects, and named archive members. It records source hashes and equality checks in `history_evidence.json`; `execution.log` contains the actual read-only verification results. This component does not modify the repository. Missing original motivation remains explicitly unresolved. It does not independently establish complete absence of inaccessible historical tests or author decision records.

Executed command:

```sh
/usr/bin/python3 audit_workdir/agents/history/audit_dimension_history.py --repo original_research --out audit_workdir/agents/history
```
