> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# AFS-D Implementation Plan

**Status:** design-only implementation audit; no code changes or execution authorized
**Audit date:** 2026-09-28
**Repository state inspected:** `9a491f89d5adf48c3119a89ec8ce1b215ec7802d`

## 1. Existing relevant code paths

| Path | Existing capability | AFS-D assessment |
|---|---|---|
| `src/bsg/representations/extract.py` | `extract_embedding_representations` obtains exact Hugging Face `hidden_states[0]` and averages the canonical unique target-token span. | Reuse unchanged. It performs a model forward and returns only the embedding vectors. |
| `src/bsg/experiments/algebra_falsification.py` | `_run_embedding_modules` loads pinned models, verifies architecture/block count/revision, checkpoints one embedding tensor per model/group, and fits split-specific embedding artifacts. | Scientifically useful reference, but tied to the monolithic AFS runner and fixed tiers. Do not use it to append to the immutable AFS-A/B/C run. |
| `src/bsg/experiments/algebra_falsification_config.py` | Validates the six-model FULL AFS panel and fixed smoke/pre-full/full matrices. | It has no two-model/all-depth/all-split D-only tier. Changing its frozen matrices would risk reinterpreting AFS-A/B/C. |
| `src/bsg/experiments/canonical_units.py` | `fit_base_artifact` performs training-only PCA and independent-ridge fitting; `evaluate_operators` emits all requested direct and algebraic metrics, including D4 relation components. | Reuse unchanged. |
| `src/bsg/experiments/algebra_falsification_units.py` | `fit_generators` and `generator_composition_rows` implement generator-only unseen-element evaluation and its controls. | Reuse unchanged for `E_gen_eq`. |
| `src/bsg/experiments/algebra_falsification_outputs.py` | Existing AFS paired bootstrap and stable-output helpers. | Reuse low-level helpers where exact, but add D-specific paired depth inference. Current `embedding_vs_internal` plot is not a prespecified paired depth analysis. |
| `src/bsg/statistics/hierarchical.py` | Deterministic pre-indexed hierarchical bootstrap implementation. | Reuse for split/template resampling, with a D-specific wrapper that preserves paired location indices. |
| `src/bsg/representations/cache.py` | Revision- and metadata-aware activation cache. | Reuse its provenance pattern or equivalent metadata-bearing NPZ contract. |
| `src/bsg/experiments/runtime_state.py` | Atomic outputs, run ledger, deterministic keys, and runtime telemetry. | Reuse unchanged. |
| `src/bsg/data/c7.py`, `src/bsg/data/d4.py`, `src/bsg/data/splits.py` | Canonical datasets, exact group actions, and split construction. | Reuse unchanged. |

### Existing D-path limitation

The dormant D path proves that embedding extraction and fitting are supported, but it is not sufficient as the final AFS-D analysis:

1. the `full` tier selects all six models, whereas the recommended design is a separately frozen two-model panel;
2. a new run invokes A/B/C before D, causing redundant CPU recomputation;
3. the current embedding output mixes direct-operator and generator rows in one schema and maps generator `E_gen_eq` into `E_eq` for compatibility;
4. it produces no paired embedding-to-depth confidence intervals or preregistered outcome classification;
5. its comparison figure uses internal `real_ridge` rows from the scramble table rather than a dedicated immutable FULL/AFS source join.

The minimal correct implementation is therefore an isolated D-only runner that reuses existing functions and imports immutable internal metrics by checksum.

## 2. Existing relevant artifacts

### Immutable FULL

- Final results: `runpod_output_by_date/result_9_6_2026/Beyond_Shared_Geometry_main_experiment_results/`
- Final bundle: `runpod_output_by_date/result_9_6_2026/Beyond_Shared_Geometry_main_experiment_results_9_4_2026_full3.zip`
- Raw/checkpoint bundle: `runpod_output_by_date/result_9_6_2026/run_outputs_9_4_2026_full3.zip`
- Direct structural metrics: `runpod_output_by_date/result_9_6_2026/Beyond_Shared_Geometry_main_experiment_results/structural/template_metrics.csv`
- Raw source commit: `30f3c6369a185f43e94b17ca03b0499cd5d65a78`
- Raw ZIP SHA256: `5fe7234572120073d8dfab2e8dc0d3373a741f8d48abadc92288f98ab16721f4`

### Immutable AFS-A/B/C

- Results root: `run_outputs/algebra_falsification/afs_full_abc_001/`
- Generator metrics: `run_outputs/algebra_falsification/afs_full_abc_001/afs/generator_composition_metrics.csv`
- Manifest: `run_outputs/algebra_falsification/afs_full_abc_001/manifest.json`
- Run ledger: `run_outputs/algebra_falsification/afs_full_abc_001/run_ledger.json`
- Source commit: `9a491f89d5adf48c3119a89ec8ce1b215ec7802d`
- Completed modules: A/B/C; `D=NOT_RUN`

The internal direct-ridge and generator metrics already cover both recommended models, both groups, depth slots 0.20/0.40/0.60/0.80, all five split seeds, latent dimension 16, and held-out test templates.

## 3. Whether embeddings already exist

**Finding: NO.**

The raw FULL ZIP contains 300 activation-cache NPZ files. Their metadata contains only block-output layers `3, 6, 7, 10, 11, 13, 14, 16, 17, 19, 22, 26, 28, 29, 32, 36`; none has `layer=0`. Cache metadata identifies the representation convention as the target-span mean at a transformer-block output.

The AFS manifest plans 60 full-panel embedding cells but records `executed_embedding_units: 0`, and the result tree contains no `embedding_extraction` or `embedding_analysis` checkpoints. Structural NPZ checkpoints contain PCA means/components, latent coordinates, and operators, not the original hidden-state-0 tensors. Exact embeddings cannot be back-calculated from them.

Internal representations and metrics are available; the pre-transformer arm is absent. The answer is therefore **NO**, not PARTIAL, for the exact embedding representation required by AFS-D.

## 4. Whether model inference is required

**Finding: YES, limited to new embedding extraction.**

The supported exact convention calls each pinned causal LM with `output_hidden_states=True` and selects `hidden_states[0]`. This requires loading each recommended model and running the canonical prompts. No internal transformer activation extraction, structural fitting, causal forward, cross-model analysis, or AFS-A/B/C recomputation is required.

An input-embedding-table lookup might be computationally cheaper, but it should not replace the already tested `hidden_states[0]` convention without an architecture-equivalence proof. The minimal scientifically conservative path is to reuse `extract_embedding_representations` unchanged.

## 5. Exact minimal files that would need modification later

No implementation change is authorized by this audit. If AFS-D is approved, the only existing file that should require modification is:

- `tests/run_runtime_checks.py` - register focused AFS-D runtime-contract tests.

The following existing scientific files should remain unchanged:

- `configs/main_experiment.yaml`
- `configs/algebra_falsification.yaml`
- `docs/ALGEBRA_FALSIFICATION_PROTOCOL.md`
- all files under the immutable FULL and AFS-A/B/C result roots
- all group, metric, PCA, fitting, and extraction implementations listed in Section 1

Keeping the old AFS configuration untouched prevents the new two-model diagnostic from silently changing the frozen AFS-A/B/C contract.

## 6. Exact new files that would be created later

The recommended isolated implementation would add:

- `configs/afs_d.yaml` - frozen two-model D-only matrix and source checksums;
- `src/bsg/experiments/afs_d.py` - D-only extraction, source import, checkpointing, and orchestration;
- `src/bsg/experiments/afs_d_outputs.py` - paired depth contrasts, bootstrap summaries, tables, figures, and interpretation labels;
- `tests/test_afs_d.py` - configuration, extraction, coverage, pairing, deduplication, bootstrap, and immutability tests;
- `runpod/run_afs_d.sh` - one-shot smoke/preflight/full entry point without A/B/C execution;
- `runpod/package_afs_d_results.sh` - D-only result packaging with checksums;
- `scripts/build_afs_d_runpod_package.sh` - deployment package builder, if remote execution is approved.

The design document should be treated as the scientific specification. Approval should be recorded before implementation by commit/tag or an explicit authorization record; do not rewrite the outcome rules after data exist.

## 7. Estimated computational requirements

### Recommended Option B matrix

```text
2 models x 2 groups x 5 split seeds = 20 new embedding fit/evaluation cells
2 models x 2 groups = 4 new embedding extraction sets
2 models x 2 groups x 4 depths x 5 splits = 80 immutable internal cells reused
5 representation locations x 2 models x 2 groups x 5 splits = 100 paired cells analyzed
```

The two group datasets contain 440 canonical prompt rows per model (280 C7 and 160 D4), so the new extraction evaluates 880 prompt rows total. Models can be loaded sequentially.

Observed FULL telemetry for these models reached approximately 2.6 GB reserved GPU memory for Llama-3.2-1B and 3.2 GB for Qwen2.5-1.5B. An 8 GB or larger CUDA GPU is a conservative execution target. Warm-cache FULL model loads took about 19-21 seconds, and structural extraction took about 1-3 seconds per group, but AFS-D wall time should be budgeted more conservatively because environment setup, model availability, output-hidden-state retention, and storage speed vary.

Expected practical budget after models are locally available:

- GPU extraction: under 10 minutes;
- 20 PCA/ridge/generator cells: a few minutes on CPU;
- 5,000-replicate paired bootstrap and output generation: approximately 5-15 minutes;
- total active runtime: approximately 15-30 minutes, excluding model download and environment installation.

Model download/cache storage is several GB and environment-dependent. New embedding arrays and tabular outputs should remain well below 100 MB. These are planning estimates, not a cost guarantee.

## 8. Expected output artifacts

All outputs should be created under a new root such as:

`run_outputs/algebra_falsification/afs_d_embedding_depth_001/`

Expected contents:

- `manifest.json`
- `config_snapshot.yaml`
- `run_ledger.json`
- `runtime_profile.csv`
- `interpretation.json`
- `artifact_checksums.json`
- `provenance/full_source.json`
- `provenance/afs_abc_source.json`
- `checkpoints/embedding_extraction/*.npz` (4 extraction sets)
- `checkpoints/embedding_analysis/*.csv` (20 fit/evaluation cells)
- `afs_d/embedding_operator_metrics.csv`
- `afs_d/embedding_generator_metrics.csv`
- `afs_d/internal_metrics_import.csv`
- `afs_d/depth_contrasts.csv`
- `afs_d/bootstrap_depth_contrasts.csv`
- `results/tables/afs_d_by_model_group_location.csv`
- `results/tables/afs_d_depth_contrasts.csv`
- `results/tables/afs_d_d4_relation_components.csv`
- `results/figures/afs_d_operator_algebra_by_depth.{pdf,svg,source.csv}`
- `results/figures/afs_d_generator_transfer_by_depth.{pdf,svg,source.csv}`
- `AFS_D_SCIENTIFIC_REPORT.md`

The final package should be a new AFS-D bundle. It must not write into either immutable source run.

## 9. Validation checks

### Provenance and immutability

- verify repository commit and clean/dirty state before execution;
- verify pinned model and tokenizer revisions exactly;
- verify FULL and AFS-A/B/C source identities and SHA256 values before import;
- snapshot source-tree checksums before and after AFS-D and require no change;
- reject any output path inside an immutable source root.

### Matrix and representation contracts

- require exactly two models, two groups, five split seeds, latent dimension 16, and four internal depth slots;
- require four embedding extraction checkpoints and 20 embedding analysis cells;
- require `layer=0`, `realized_depth=0.0`, and `representation_location=embedding` for every embedding row;
- validate Llama/Qwen architecture type, exact block count, public input embeddings, and one vector per canonical row;
- verify target-token positions use the unique canonical character span and match the existing tokenizer settings;
- fail on missing, duplicate, nonfinite, or extra scientific keys.

### Leakage and metric contracts

- prove PCA and every learned operator use training templates only;
- perturb validation/test vectors in a fixture and verify fitted PCA/operators do not change;
- verify test-template metrics and generator unseen-element indexing against exact C7/D4 actions;
- verify D4 `r^4`, `s^2`, `srs=r^-1`, and noncommutative word order;
- compare reused metric functions with exact synthetic representations;
- aggregate unseen generator elements within template before inference;
- collapse repeated `E_hom_op` values to one observation per fitted split cell.

### Statistical contracts

- pair embedding/internal rows on model, group, split, template, and word/target where applicable;
- reuse the same bootstrap draws across locations;
- resample split then template clusters, never depth or metric rows as IID units;
- use 5,000 replicates, seed `314159`, and two-sided 95% percentile intervals;
- verify deterministic reruns are byte-stable for tabular summaries;
- test a synthetic known-direction fixture for Outcomes A, B, C, D, and mixed;
- report model-specific results before any fixed-panel summary.

### Runtime and packaging

- record model load, embedding extraction, each fit/evaluation cell, bootstrap, table, figure, and package stages in `runtime_profile.csv`;
- verify resume skips completed extraction and analysis keys without loading a model when all embedding caches exist;
- run the full CPU suite and selected RunPod runtime checks before execution;
- extract the result ZIP and validate imports, manifests, row counts, checksums, and required artifacts.

## 10. Immutability confirmation

`main_full_003` and `afs_full_abc_001` remain read-only evidence sources. AFS-D must neither append to their ledgers nor regenerate their tables, figures, checkpoints, manifests, or gates. Imported rows retain source-run identifiers and checksums. The new run receives its own namespace, ledger, checkpoints, report, and package.

This plan does not alter FULL, AFS-A, AFS-B, or AFS-C. No experiment was executed during this audit.
