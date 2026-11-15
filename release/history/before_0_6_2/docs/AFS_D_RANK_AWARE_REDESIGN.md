> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# AFS-D Rank-Aware Redesign

**Status:** preregistered repair design; not implemented; not executed
**Trigger:** structural rank failure documented in `docs/AFS_D_RANK_FAILURE_AUDIT.md`
**Scope:** AFS-D only. FULL and AFS-A/B/C remain immutable and untouched.

## 1. Purpose

This document preregisters a mathematically valid repair of AFS-D after the frozen 16-dimensional embedding analysis failed:

```text
ValueError: requested latent_dim=16 exceeds observed centered training rank 6
```

The redesign preserves the original scientific question as closely as possible:

**Is the finite-group structure observed in internal representations already present in the target-span embedding representation, or does it improve through internal contextualization?**

The repair separates two questions that the original frozen design accidentally conflated:

1. **Representation-capacity result:** the target-span `hidden_states[0]` embedding representation has insufficient centered rank to support the original 16-dimensional PCA/operator comparison.
2. **Algebraic-quality question:** within a common mathematically supported dimension, do internal representations show stronger finite-group structure than the embedding representation?

No AFS-D scientific metric values were inspected to choose this redesign. The proposed dimensions are determined from rank constraints only.

## 2. Preserved Frozen Design

The following original AFS-D choices remain frozen wherever mathematically possible:

- model panel: `meta-llama/Llama-3.2-1B` and `Qwen/Qwen2.5-1.5B`
- model revisions
- groups: C7 and D4
- embedding convention: Hugging Face `hidden_states[0]`, before transformer block 1, mean over the unique target-token span
- internal layers: Llama blocks 3, 6, 10, 13; Qwen blocks 6, 11, 17, 22
- depth slots: 0.20, 0.40, 0.60, 0.80
- 40 templates per group
- 28/6/6 train/validation/test template split
- split seeds: 101, 303, 909, 2701, 4243
- ridge coefficient: 0.01
- group operations, words, and relations
- direct independent-ridge operator fit
- generator-composition fit/evaluation
- metrics: `E_gen_eq`, `E_rel_data`, `E_hom_data`, `E_comp_data`, `E_eq`, `E_hom_op`, plus D4 relation components
- aggregation rules: unseen generator elements averaged within template; operator-only quantities collapsed to split-level units
- paired internal-minus-embedding contrasts
- fixed-depth average over the four preregistered internal depths
- bootstrap seed 314159, 5,000 replicates, 95% percentile intervals
- original A/B/C/D/MIXED interpretation logic
- no favorable-result pass/fail gate

## 3. Necessary Changes To The Estimand

The original estimand compared embedding and internal representations at `latent_dim=16`. That estimand is not mathematically defined for target-span embeddings because the embedding training matrix has centered rank at most 6 for C7 and at most 3 for D4.

The repaired primary estimand is:

```text
Delta_q(model, group, depth; d_group)
  = mean_test E_q(internal depth, latent_dim=d_group)
    - mean_test E_q(embedding, latent_dim=d_group)
```

with fixed group dimensions:

| Group | Repaired primary latent dimension | Mathematical basis |
|---|---:|---|
| C7 | 6 | 7 unique target vectors imply centered rank at most 6 |
| D4 | 3 | 4 unique target vectors imply centered rank at most 3 |

Both sides of each embedding-vs-internal comparison must use the same latent dimension. The repaired primary comparison must not compare embedding C7 at 6 or D4 at 3 against internal representations at 16.

What is preserved:

- the same embedding-to-depth scientific contrast;
- the same model, group, depth, split, template, fitting, metric, and aggregation structure;
- the same direction of interpretation: negative internal-minus-embedding values mean stronger internal algebraic fidelity;
- the same fixed-depth average and model-specific inference.

What necessarily changes:

- the primary latent dimension becomes group-specific: C7 uses 6, D4 uses 3;
- imported internal 16-dimensional metric tables are no longer valid primary comparators;
- rank diagnostics become a preregistered first-stage output;
- any comparison to FULL `latent_dim=16` becomes contextual only, not the repaired primary AFS-D contrast.

## 4. Intrinsic-Rank Diagnostic

Before any rank-matched algebraic metric is interpreted, AFS-D must report intrinsic ranks of the target-span embedding representation. This diagnostic is not a performance metric and must be computed before downstream operator metrics are used.

For every planned embedding extraction set and split:

- model key
- model id
- group
- split seed
- number of rows
- number of unique embedding vectors, using exact or tolerance-stable duplicate accounting specified before execution
- raw matrix rank
- centered training rank
- maximum admissible PCA dimension
- proposed repaired dimension
- pass/fail for feasibility

Expected structural results:

| Group | Training rows | Unique target states | Maximum centered rank | Required observed rank for repaired run |
|---|---:|---:|---:|---:|
| C7 | 196 | 7 | 6 | 6 |
| D4 | 112 | 4 | 3 | 3 |

The C7 rank 6 value is already evidenced by the reported failure. D4 was not reached in the failed run, so the repaired run must verify D4 rank from embedding artifacts before fitting D4 algebraic metrics.

If observed rank is lower than the proposed dimension for any model/group/split, the repaired AFS-D must stop before downstream metric interpretation. It must not select a new dimension after seeing metric outcomes.

## 5. Rank-Matched Comparison

The primary repaired comparison is group-wise rank matched:

- C7 embedding PCA and C7 internal PCA use `latent_dim=6`.
- D4 embedding PCA and D4 internal PCA use `latent_dim=3`.
- Pairing is only valid when embedding and internal rows have the same `model`, `group`, `split_seed`, `template_id`, metric, and `latent_dim`.

Internal metrics at dimensions 6 and 3 are not present in the original AFS-D imported metric sources, which select `latent_dim == 16`. Therefore the repaired run must produce AFS-D-local rank-matched internal metrics from immutable internal activation evidence, or from a newly frozen immutable source that already contains exactly those rank-matched internal metrics. In either case, FULL and AFS-A/B/C themselves must not be modified.

The repaired run may not use the existing imported internal `latent_dim=16` metrics as primary comparators. Those rows may be cited only as prior-context evidence from FULL/AFS-A/B/C.

## 6. Template-Collapse Interpretation

At `hidden_states[0]`, the target-span representation can collapse different templates with the same target token to identical vectors. For C7, all templates containing the same weekday may share one target vector. For D4, all templates containing the same corner label may share one target vector.

The original template-held-out split remains frozen because changing the split after failure would introduce a new design decision. However, interpretation changes:

- For internal contextual states, held-out templates test generalization across distinct prompt contexts.
- For target-span embeddings, held-out templates may not be representationally distinct after conditioning on the target state.
- Therefore embedding held-out-template performance should be interpreted as algebraic recoverability from lexical target-state vectors, not as template-context generalization.

This distinction must be reported as part of the rank-capacity result. It must not be used to change splits, templates, or outcomes based on scientific metric values.

## 7. Outcome Criteria

AFS-D remains a diagnostic control with no favorable-result PASS gate. The original A/B/C/D/MIXED interpretation criteria are reused with one mathematical substitution: all primary rows must come from the group-specific rank-matched dimension.

Let the fixed-depth contrast be internal minus embedding, so negative is better internally.

For each group and model, compute fixed-depth average contrasts for:

- co-primary metrics: `E_gen_eq`, `E_rel_data`
- supporting metrics: `E_hom_data`, `E_comp_data`
- secondary diagnostics: `E_eq`, `E_hom_op`

The repaired interpretation criteria are:

| Outcome | Criterion |
|---|---|
| A: embedding substantially worse | For the group, both models have upper 95% CI bounds below 0 for both co-primary metrics, and both supporting algebraic point estimates are negative. |
| B: no systematic difference detected | For the group, both models' co-primary intervals include 0, and no supporting metric has a concordant nonzero interval in the same direction in both models. |
| C: embedding stronger | For the group, both models have lower 95% CI bounds above 0 for both co-primary metrics, and both supporting algebraic point estimates are positive. |
| D: C7 embedding-competitive, D4 internally strengthened | C7 meets B or C while D4 meets A. |
| MIXED | Any other model, metric, or depth pattern. |

These criteria are not weakened. The only change is that the tested rows use C7 `latent_dim=6` and D4 `latent_dim=3`, because `latent_dim=16` is infeasible for embedding.

Execution validity criteria:

- all expected rank diagnostics are present before algebraic interpretation;
- C7 observed centered training rank is 6 for every planned embedding split;
- D4 observed centered training rank is 3 for every planned embedding split;
- embedding and internal rows are paired only at equal group-specific latent dimension;
- all four internal depths are present for every paired unit;
- no AFS-D metric values are used to alter dimensions, models, depths, splits, metrics, or outcome rules.

If any execution validity criterion fails, the repaired AFS-D run is invalid rather than negative.

## 8. Provenance And Bias Controls

This redesign is triggered solely by the structural rank failure documented in `AFS_D_RANK_FAILURE_AUDIT.md`.

Provenance commitments:

- no AFS-D scientific metric values were inspected for repair selection;
- dimensions are determined from rank constraints, not observed performance;
- the C7 dimension 6 follows from 7 unique target vectors and the observed centered rank 6 failure;
- the D4 dimension 3 follows from 4 unique target vectors and must be verified before D4 metric interpretation;
- FULL and AFS-A/B/C remain untouched;
- existing FULL and AFS-A/B/C `latent_dim=16` conclusions are not revised by this repair;
- the repaired AFS-D run receives its own run id, manifest, config hash, ledger, rank diagnostics, metric tables, and report.

Post-hoc restrictions:

- do not change dimensions after rank-matched metrics are computed;
- do not choose global dimension 3 instead of group-specific 6/3 based on outcomes;
- do not drop D4 if D4 is harder or less favorable;
- do not compare rank-matched embedding against internal dimension 16 as the primary result;
- do not alter the A/B/C/D criteria to obtain a desired interpretation.

## 9. Implementation Plan, Not Yet Authorized

No implementation is performed by this document. The following files or config values would need changes for execution:

1. `configs/afs_d.yaml`
   - Replace the single primary `primary_latent_dimension: 16` with a frozen group-specific rank-matched specification, for example `group_latent_dimensions: {C7: 6, D4: 3}`.
   - Add explicit rank-diagnostic expectations, for example expected centered ranks `{C7: 6, D4: 3}`.
   - Add immutable provenance for any internal activation source or newly frozen rank-matched internal metric source used by repaired AFS-D.

2. `src/bsg/experiments/afs_d.py`
   - Update config validation to require the rank-aware design rather than rejecting non-16 AFS-D dimensions.
   - Update `_embedding_metadata(...)` so `latent_dim` is group-specific.
   - Update `_run_dataframe_analysis(...)` to pass the group-specific dimension to `fit_base_artifact(...)`.
   - Add a rank-diagnostic stage after embedding extraction and before algebraic analysis.
   - Update source import: do not select `latent_dim == 16` as the primary internal comparator.
   - Add AFS-D-local internal rank-matched fitting from immutable internal activations, or import a separately frozen rank-matched internal source.
   - Update manifest fields from one `primary_latent_dimension` to group dimensions and rank diagnostics.
   - Ensure the ledger distinguishes embedding extraction, embedding rank diagnostics, embedding analysis, internal rank-matched analysis/import, and postprocessing.

3. `src/bsg/experiments/afs_d_outputs.py`
   - Include `latent_dim` in location identity or enforce that paired embedding/internal rows have identical group-specific dimension before pairing.
   - Update report text from `Latent dimension: 16` to C7=6 and D4=3.
   - Preserve the original bootstrap, contrast, and classification logic otherwise.

4. `src/bsg/experiments/canonical_units.py`
   - No mathematical change is required if callers pass the desired dimension. `fit_base_artifact(...)` already accepts `latent_dim` and preserves the rank check.

5. `src/bsg/experiments/fitting.py`
   - No weakening or removal of `ensure_observed_pca_rank` is allowed.
   - Optional only: expose a helper for rank diagnostics so execution and reporting use one audited calculation.

6. `runpod/run_afs_d.sh`, `runpod/package_afs_d_results.sh`, and AFS-D packaging scripts
   - Include rank diagnostics and any rank-matched internal-analysis artifacts in the packaged outputs.
   - Preserve isolation from immutable FULL and AFS-A/B/C roots.

7. `tests/test_afs_d.py` and runtime-contract tests
   - Update frozen-dimension assertions to the group-specific rank-aware specification.
   - Add tests that pairing rejects dimension-mismatched embedding/internal rows.
   - Add tests that C7=6 and D4=3 are selected from rank constraints, not metric values.
   - Add tests that rank-diagnostic failure stops the run before scientific metric interpretation.

The implementation must not edit prior FULL, AFS-A, AFS-B, or AFS-C result files, ledgers, manifests, or configs.

## 10. Inputs Required Before Execution

The repaired run is valid only if these inputs are available and checksum-frozen before execution:

- target-span embedding extraction artifacts or authorization to regenerate them under the same pinned model revisions;
- immutable internal activation artifacts for the two AFS-D models, two groups, four depths, and five split seeds, sufficient to fit C7 dimension 6 and D4 dimension 3 inside AFS-D; or a separately frozen internal rank-matched metric source produced without inspecting AFS-D embedding outcomes;
- source manifests and checksums for any new immutable input bundle;
- a repaired AFS-D config hash frozen before running metric analysis.

If rank-matched internal inputs cannot be obtained without modifying FULL or AFS-A/B/C, the repaired primary comparison is not executable as designed.

## 11. Reporting Requirements

The repaired AFS-D report must have two separate result sections:

1. **Intrinsic rank and template collapse**
   - observed embedding ranks by model/group/split;
   - unique-vector counts;
   - confirmation that 16D embedding PCA is infeasible;
   - interpretation of template collapse.

2. **Rank-matched algebraic fidelity**
   - embedding and internal means at C7=6 and D4=3;
   - depth-specific paired contrasts;
   - fixed-depth average contrasts;
   - split-level effects;
   - original A/B/C/D/MIXED interpretation using rank-matched rows only.

The report must not present `latent_dim=16` internal results as the primary comparator for rank-matched AFS-D.

## 12. Recommendation

**CONDITIONAL_GO**

Run the repaired AFS-D only after implementation has been updated to enforce group-specific rank matching and after rank-matched internal inputs are checksum-frozen without modifying FULL or AFS-A/B/C.

The condition is mathematical and provenance-based, not outcome-based:

- C7 primary dimension is fixed at 6.
- D4 primary dimension is fixed at 3.
- Both sides of each group comparison must use the same dimension.
- The original A/B/C/D interpretation criteria remain unchanged except for replacing infeasible 16D rows with rank-matched rows.
