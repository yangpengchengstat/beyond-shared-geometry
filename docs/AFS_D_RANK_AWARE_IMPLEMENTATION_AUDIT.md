> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# AFS-D Rank-Aware Implementation Audit

## Scope

This audit covers implementation and structural validation only. The repaired AFS-D scientific run was not launched, and no repaired AFS-D scientific metric outcomes were inspected or summarized.

Implemented preregistered dimensions:

| Group | Frozen repaired latent dimension | Expected embedding centered rank |
|---|---:|---:|
| C7 | 6 | 6 |
| D4 | 3 | 3 |

## Files Modified

- `configs/afs_d.yaml`
- `src/bsg/experiments/afs_d.py`
- `src/bsg/experiments/afs_d_outputs.py`
- `runpod/run_afs_d.sh`
- `runpod/package_afs_d_results.sh`
- `tests/test_afs_d.py`
- `tests/run_runtime_checks.py`

Files created for documentation:

- `docs/AFS_D_RANK_AWARE_IMPLEMENTATION_AUDIT.md`

Existing prior-result artifacts for FULL, AFS-A, AFS-B, and AFS-C were not modified.

## Implementation Summary

`configs/afs_d.yaml` now declares the repaired rank-aware experiment identity:

- `experiment: beyond_shared_geometry_afs_d_rank_aware`
- `group_latent_dimensions: {C7: 6, D4: 3}`
- `expected_embedding_centered_ranks: {C7: 6, D4: 3}`
- immutable internal activation source provenance for `main_full_003_activation_cache`

`src/bsg/experiments/afs_d.py` now:

- validates the rank-aware config and rejects any dimension drift;
- uses group-specific latent dimensions for embedding cells;
- runs an `embedding_rank_diagnostic` ledger stage after embedding extraction and before embedding algebraic analysis;
- aborts before downstream metric interpretation if a rank diagnostic fails;
- preserves `ensure_observed_pca_rank()` through `fit_base_artifact(...)`;
- recomputes rank-matched internal C7/D4 analyses from immutable internal activation caches;
- no longer imports internal `latent_dim=16` metrics as the repaired primary comparator;
- records rank-aware group dimensions and internal activation source provenance in the manifest;
- creates a new default run id prefix: `afs_d_rank_aware_...`.

`src/bsg/experiments/afs_d_outputs.py` now:

- includes `latent_dim` in location identity;
- rejects dimension-mismatched embedding/internal pairing loudly;
- writes `afs_d/embedding_rank_diagnostics.csv`;
- reports C7=6 and D4=3 in the scientific report text;
- preserves the original bootstrap and A/B/C/D/MIXED outcome logic.

RunPod scripts now:

- pass `--internal-activation-source`;
- use the rank-aware run-id prefix;
- require `afs_d/embedding_rank_diagnostics.csv` in packaged results;
- package only the repaired experiment identity.

## Tests Added Or Changed

Updated or added tests cover:

- C7 selects exactly `d=6`;
- D4 selects exactly `d=3`;
- dimensions come from frozen config/rank constraints, not scientific metrics;
- dimension-mismatched pairing is rejected;
- insufficient observed rank aborts before metric interpretation;
- `ensure_observed_pca_rank()` remains active;
- packaging includes rank diagnostics;
- template splits, seeds, model panel, and layers remain frozen via existing AFS-D config/matrix tests;
- FULL/AFS-A/B/C runtime contracts remain covered by existing runtime checks.

Updated test files:

- `tests/test_afs_d.py`
- `tests/run_runtime_checks.py`

## Validation Results

Commands run:

```text
PYTHONPATH=src python -m bsg.experiments.afs_d --validate-only --config configs/afs_d.yaml
```

Result: PASS. The command returned `status: VALID` and the repaired matrix with 20 embedding rank diagnostics and 80 rank-matched internal cells.

```text
PYTHONPATH=src python -m bsg.experiments.afs_d --preflight --config configs/afs_d.yaml --full-results-root runpod_output_by_date/result_9_6_2026/Beyond_Shared_Geometry_main_experiment_results --afs-abc-root run_outputs/algebra_falsification/afs_full_abc_001 --internal-activation-source runpod_output_by_date/result_9_6_2026/run_outputs_9_4_2026_full3.zip
```

Result: PASS. The structural preflight verified immutable source manifests and all required rank-matched internal activation cache entries. It did not launch AFS-D and did not inspect AFS-D scientific metric outcomes.

```text
python - <<'PY'
from pathlib import Path
for path in ['src/bsg/experiments/afs_d.py','src/bsg/experiments/afs_d_outputs.py','tests/test_afs_d.py']:
    compile(Path(path).read_text(), path, 'exec')
    print('ok', path)
PY
```

Result: PASS.

Direct invocation of the repaired AFS-D unit-test functions passed for:

- `test_afs_d_config_and_matrix_are_frozen`
- `test_afs_d_generator_and_operator_units_are_aggregated_without_pseudoreplication`
- `test_afs_d_pairing_keeps_depth_fixed_and_bootstrap_is_deterministic`
- `test_afs_d_dimension_mismatched_pairing_is_rejected`
- `test_afs_d_outcome_rules_cover_a_b_c_d_and_mixed`
- `test_afs_d_rank_diagnostics_select_frozen_dimensions_without_metrics`
- `test_afs_d_insufficient_rank_aborts_before_analysis`
- `test_afs_d_observed_pca_rank_guard_remains_active`
- `test_afs_d_internal_metadata_uses_rank_matched_dimensions`
- `test_afs_d_embedding_pca_and_operators_are_training_only`
- `test_afs_d_reuses_exact_c7_and_d4_generator_composition`
- `test_afs_d_result_packaging_is_namespaced_and_excludes_checkpoints`

Full `pytest tests/test_afs_d.py -q` did not complete in the local environment: the process exited with code 139. This appears to be an environment-level Python/NumPy binary incompatibility; pandas imports print NumPy 2.4 compatibility warnings for optional compiled modules `numexpr` and `bottleneck`. The direct test-function invocations above ran through the repaired structural checks despite those warnings.

## Rank-Diagnostic Implementation

Rank diagnostics are computed per model/group/split immediately after embedding extraction and before any embedding or internal algebraic metric interpretation.

Each diagnostic row records:

- model key and model id;
- group;
- split seed;
- row counts;
- exact unique-row count;
- raw matrix rank;
- centered training rank;
- maximum admissible PCA dimension;
- frozen group latent dimension;
- expected centered rank;
- rank status.

The repaired run aborts if the observed centered rank is below the frozen dimension or differs from the preregistered expected rank.

No automatic dimension reduction or smaller-dimension fallback is implemented.

## Internal Activation Source And Hashes

Rank-matched internal analyses are executable from the immutable FULL activation cache archive:

```text
runpod_output_by_date/result_9_6_2026/run_outputs_9_4_2026_full3.zip
sha256: 5fe7234572120073d8dfab2e8dc0d3373a741f8d48abadc92288f98ab16721f4
cache_subdir: run_outputs/activation_cache
activation convention: mean over the unique target-token span at transformer block output
```

Required activation cache entries verified by preflight:

| model_key | group | depth | layer | cache_key | cache_sha256 |
|---|---|---:|---:|---|---|
| llama_1b | C7 | 0.20 | 3 | `61c8736d6e1e25c1a182984f79a4c74e83b4c7c2461861542d9cc1d5da58bbaf` | `930fe58df3350acf6b828ec8b70fa014b0c72b2a409b953bfe6593867d9b8b29` |
| llama_1b | C7 | 0.40 | 6 | `ec4467afd3f1bb58853b9e64d0fcaac4e9b2a8debcb9cd260689cf118ded8ab0` | `e2d0b1fc136da571560e7b441617d6e32e679ed9b7c82f33f8ecd26b79d7e4da` |
| llama_1b | C7 | 0.60 | 10 | `7e7e37ad67648e29003833c73b276ca1e7237b14e9ab514baaeff3d34ef1b17c` | `4b1f1f5d6100b212342ca85c842b6b8875c71537ee035686c2d66e27e9f41320` |
| llama_1b | C7 | 0.80 | 13 | `be1466e7bcf0e97ccdfa13ed07916c4296f4732a724f6497c6309e7b5bd226f1` | `1b4e71f3093619569ac0a6b9064da39d209782c69204fe254acf2db70729dd02` |
| llama_1b | D4 | 0.20 | 3 | `bd3b9069cd4ceef3cb747f53aa3a3e07fa735c85083f6abaa9768a1f6c634b08` | `04d292e90ef59bb98e37c861e144303c9bb4260867f785a87d14f4303de172fc` |
| llama_1b | D4 | 0.40 | 6 | `1c7642d228841a66a41a904c76dbfb6c164db411cc7bb0f7876fa06b60de2cb9` | `1bd03e7265f7104fecfbe8233e8631fc989cb68c7faf3939232b8960861cdb29` |
| llama_1b | D4 | 0.60 | 10 | `3645445cc8000e6149349c160d5710a8f82f4ee81afa28856ce1a676ef8b7de2` | `ec9ae144d9a44b0a34a6b880bf43b713fcf01edd9b3b4134845a1b10d4fff8b9` |
| llama_1b | D4 | 0.80 | 13 | `2bb2159df7a5e4e41a05589061ff8beae93055f722a386b14ce3ab7890a24935` | `9cef94d2a27c990a796f48cac4166e1d0f72c3012c0882d857cebd661ed45d00` |
| qwen_1_5b | C7 | 0.20 | 6 | `4f097b7f73cf7749d04f9e7f439215fad86b723058f41cab2c7976d95d9a2c10` | `50ab104b133d2611f99deee0f5024d0357f92909193cea88e0886b8f84266d2b` |
| qwen_1_5b | C7 | 0.40 | 11 | `f1b1225c2a63551ba103e85dc8c19bdf4f7421805645b03092e6dcce65b998e6` | `9c76acf784c0430a610b525e7a0bd4cc4dd195217cced4bccc3bb537139609bf` |
| qwen_1_5b | C7 | 0.60 | 17 | `75255057ce8a331f84e48fee194991cfffda833f1a77f3a9587b1e8329a319f6` | `ab63235620ccf5d20426b81f5bbdeaa62c466069d4a5f1842140cfaf7a81b498` |
| qwen_1_5b | C7 | 0.80 | 22 | `e181c8ad444ae6bb461ce5ebc0dd2d499fcbcbba00e422df209529db1786fbeb` | `6126c8b2f46c8cc28bd0332d6fd570a0c80a47b6a0131613a934e3f532f86cdc` |
| qwen_1_5b | D4 | 0.20 | 6 | `a9b392615920c895bcd9f43979b74aa4d1fbf6d8c73310863dd052c205d3b342` | `fe9a9d7ceeaccf65791f9b8ded14aceb68d8c126c3fb78735c9b82aa4e0ac5c1` |
| qwen_1_5b | D4 | 0.40 | 11 | `7fc8179c84a6f674ba287a646700a12281cc5e3fad49ce6f96278dce4d96d378` | `e101026ac0b88292999b0e0e8fc4404128b9e88475ce9866a4d2279f3f56c9bd` |
| qwen_1_5b | D4 | 0.60 | 17 | `0c0178bd761824ed2ae60095aae2b738f383f4f09b5d9d7a77aa890de7ad63a1` | `4faac99b9c19c8d552623bd79bed5cb87ddd6375e20d07dea7b4c1649862c962` |
| qwen_1_5b | D4 | 0.80 | 22 | `c592df11e9e843dab5f2cec4b37b20c331955edadfac82fb8c921d4d966aca1d` | `163f4c8079edb98c0528af2108307d250a6ea120c9bcaf665179dab05d841967` |

All required rank-matched internal analyses are executable from these caches.

## Scientific Metric Inspection

No repaired AFS-D scientific metric outcomes were inspected.

The implementation preflight verifies source manifests, archive hashes, cache-key availability, and cache file hashes. It does not compute or summarize embedding-vs-internal metric outcomes.

## FULL / AFS-A/B/C Status

FULL and AFS-A/B/C result artifacts, manifests, ledgers, and configs remain unchanged.

The repaired AFS-D no longer uses imported internal `latent_dim=16` metrics as primary comparators. Instead, it recomputes rank-matched internal analyses inside the repaired AFS-D run from immutable activation caches.

## Remaining Blockers

The only local validation blocker is environmental: full pytest invocation exits with code 139 under the current local Python environment. The error is consistent with NumPy 2.4 incompatibility in optional compiled pandas dependencies (`numexpr`, `bottleneck`). Direct structural/unit test-function execution for the repaired AFS-D checks passed.

No scientific-run blocker remains: immutable internal activation inputs are available and preflight validates all 16 required caches.

## Recommendation

GO_FOR_REPAIRED_AFS_D_RUN
