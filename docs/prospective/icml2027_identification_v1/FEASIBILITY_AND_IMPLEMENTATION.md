# Feasibility and implementation specification

Date: 2026-10-16. This is an implementation plan based on a read-only inspection of saved artifacts and source code. No model was loaded, no operator or PCA was fitted, and no scientific experiment was executed for this audit. All new analyses and reruns must write into a new prospective experiment directory. Frozen FULL, AFS, source repositories, and published releases remain immutable.

## 1. Findings that determine the plan

1. **The raw FULL activations needed for offline reanalysis exist.** They cover all six pinned models, both groups, the four primary depths, and an exploratory terminal readout. They retain all 40 templates, not just the test templates. Training-only PCA, state centroids, residual analyses, new regularization fits, and fixed-feature action reassignment can therefore be implemented without another language-model pass.
2. **The saved representation is a pooled source-token readout.** It is neither a full sequence nor a per-token cache. New prompts, new lexical items, new readout positions, new layers, and behavioral effects of a new intervention require fresh model forwards.
3. **The primary structural grid is 240 cells, not 444 independent primary replications.** The 444 base checkpoints include 240 primary cells, 60 terminal exploratory cells, and 144 dimension-sensitivity cells. A corresponding 444 joint checkpoints are available. Split seeds reuse the same templates and activations.
4. **A category-centroid control is essential.** In matched coordinates the source-conditioned mean additive field equals a difference of training state centroids. With a true discrete state tracker it composes exactly by construction. Group closure alone therefore cannot distinguish contextual transformation structure from a stable category code.
5. **The previously suggested consistent-action null needs a correction.** A simultaneous feature/action renaming is a tautology. Reassigning the action while holding features fixed is informative, but the natural four-corner D4 task has only three distinct conjugate action-image families. There are only two alternative families to the original, so repeated sampling cannot create a high-resolution D4 family-level null distribution.
6. **The proposed prospective S3 crossing is substantially larger than FULL.** The primary pooled float32 arrays alone total 15,049,162,752 bytes (15.049 GB decimal; 14.015625 GiB). The existing extraction function accumulates every returned layer for every row; it should not be used unchanged for this larger design.

The checks below establish availability, array structure, and file correspondence. They do not establish that a scientific claim is true or that a new experiment will succeed.

## 2. Source of truth and audit products

Original repository, read-only: the original `beyond-shared-geometry/` repository. All `src/` and `runpod_output_by_date/` paths in this report are relative to that repository root. Protocol documents and `evidence/` links are relative to this prospective plan's directory; the original repository location is supplied by the reader's checkout, not a personal filesystem path.

Canonical FULL raw archive, relative to that repository:

`runpod_output_by_date/result_9_6_2026/run_outputs_9_4_2026_full3.zip`

The archive is 690,220,115 bytes, contains 3,691 members, and reports 923,639,878 total uncompressed member bytes. Its previously verified expected SHA256 is `5fe7234572120073d8dfab2e8dc0d3373a741f8d48abadc92288f98ab16721f4`. This task did not recompute the whole archive digest. It did inspect every NPZ header, verify all 888 structural NPZ byte hashes against the run ledger, and verify all 300 activation-cache filename keys against their canonical metadata hashes. There were no mismatches in those checks.

Archive metadata members:

- `run_outputs/main_experiment/main_full_003/manifest.json`
- `run_outputs/main_experiment/main_full_003/run_ledger.json`
- `run_outputs/main_experiment/main_full_003/config_snapshot.yaml`
- `run_outputs/main_experiment/main_full_003/runtime_profile.csv`

The historical FULL source commit is `30f3c6369a185f43e94b17ca03b0499cd5d65a78`. The frozen archive, its ledger, and its configuration identify the historical execution. A later source checkout is not evidence that a changed function ran historically. The current data/group/split files inspected here match their historical versions; other current code references below describe implementation contracts and must be versioned explicitly in a rerun.

Machine-readable audit products in this plan's `evidence/` directory:

| File | Purpose |
|---|---|
| [FULL_CACHE_AND_CHECKPOINT_HEADERS.json](evidence/FULL_CACHE_AND_CHECKPOINT_HEADERS.json) | All 1,188 NPZ members: array keys, shapes, dtypes, byte sizes, metadata, NPZ hashes, ledger correspondence |
| [CELL_INPUT_INDEX.json](evidence/CELL_INPUT_INDEX.json) | Every one of the 444 structural cells, with base/joint member, raw activation member, cell identity, and exact saved split IDs |
| [CACHE_COVERAGE.csv](evidence/CACHE_COVERAGE.csv) | All 300 raw activation caches, including model, task, layer, row count, dimension, and archive member |
| [EMBEDDING_CACHE_HEADERS.json](evidence/EMBEDDING_CACHE_HEADERS.json) | Repository-relative paths and headers for the four separately saved AFS-D embedding caches |
| [CONSISTENT_ACTION_DESIGN.json](evidence/CONSISTENT_ACTION_DESIGN.json) | Exhaustive C7/D4 action-image families and representative fixed-feature reassignment tables |
| [enumerate_consistent_actions.py](evidence/enumerate_consistent_actions.py) | Standard-library-only construction of those finite design tables; no activations or scientific fitting |

These files use archive-member or repository-relative provenance references. They can accompany the plan without exposing a personal checkout location. The governing prospective statistical definitions are in [STATISTICAL_AND_SYNTHETIC_PROTOCOL.md](STATISTICAL_AND_SYNTHETIC_PROTOCOL.md); the historical replay remains a separate analysis.

## 3. FULL checkpoint and raw-cache contract

### 3.1 Structural checkpoints

Archive prefixes:

`run_outputs/main_experiment/main_full_003/checkpoints/structural_base/`

`run_outputs/main_experiment/main_full_003/checkpoints/structural_joint/`

There are 444 NPZ files under each prefix. The NPZ itself has no complete embedded cell identity; the ledger and [CELL_INPUT_INDEX.json](evidence/CELL_INPUT_INDEX.json) supply model, group, layer, dimension, split seed, and method. Never infer a cell from filename ordering.

| NPZ key | Shape and dtype | Meaning |
|---|---|---|
| `pca_mean` | `(D,)`, float64 | Training mean in raw hidden coordinates |
| `pca_components` | `(q,D)`, float64 | PCA decoder rows; the encoder is its transpose |
| `pca_explained` | `(q,)`, float64 | Retained training explained-variance ratios, not the complete spectrum |
| `pca_fit_rows` | scalar int64 | 196 for C7; 112 for D4 |
| `z` | C7 `(280,q)`; D4 `(160,q)`, float64 | Encoded rows for all 40 templates |
| `operators` | C7 `(7,q,q)`; D4 `(8,q,q)`, float64 | One row-vector operator per canonical group element |

The implementation saves group elements in the order supplied by the canonical group specification (`src/bsg/experiments/canonical_units.py:152`). The independent fitting path first fits PCA on training rows and then transforms every row (`canonical_units.py:27`). The joint path reuses that PCA and those encoded rows (`canonical_units.py:39`). This audit verified headers and checkpoint hashes; it did not compare every base/joint array elementwise.

Coverage **for each** of base and joint:

| Grid | Cells | Interpretation |
|---|---:|---|
| q=16; six models; C7/D4; depths 0.2/0.4/0.6/0.8; five saved splits | 240 | Primary historical grid |
| q=16; six models; C7/D4; terminal; five saved splits | 60 | Exploratory terminal grid |
| q=8/32/64; six models; C7/D4; depths 0.4/0.8; splits 101/303 | 144 | Restricted dimension sensitivity |
| Total | 444 | Not 444 independent datasets |

The five saved split seeds are 101, 303, 909, 2701, and 4243. Their literal partitions should be loaded from the ledger/cache metadata, rather than regenerated using a possibly different library version. Every split is 28 training, 6 validation, and 6 test templates. [CELL_INPUT_INDEX.json](evidence/CELL_INPUT_INDEX.json) contains the exact IDs.

The archived configuration explicitly records Ridge alpha 0.01; Joint 1,200 steps, learning rate 0.002, weight decay 1e-5, data coefficient 1 with homomorphism/relation/identity weights 1/1/1 and orthogonality weight 0.05; deterministic learned-fit seed 42; float16 extraction, batch size 8, and device_map auto; bootstrap 5,000 with seed 314159. Random-control seeds are 104729, 130363, 155921, 196613, and 262147. These are the historical settings to reproduce, not a recommendation that unchanged settings are optimal for the larger prospective task. The inherited objective averages squared entries/predictions, so scale and q matter when comparing newly tuned conditions.

### 3.2 Raw FULL activation caches

All 300 are under `run_outputs/activation_cache/<metadata_sha256>.npz`. Keys:

- `activations`: float32 `(N,D)`.
- `metadata`: scalar Unicode JSON, including pinned model/tokenizer revisions, task, layer, tokenizer settings, prompt construction version, and template/split identity.

| Cache task | Count | Rows | Coverage |
|---|---:|---|---|
| `structural_C7` / `structural_D4` | 60 | 280 / 160 | Six models × two groups × five readouts; all 40 templates |
| `causal_C7` / `causal_D4` | 240 | 42 / 24 | Six models × two groups × four primary readouts × five splits; six test templates per split |

The structural float32 activation payload is 144,179,200 bytes. The causal payload is 86,507,520 bytes. The approximately 220 MiB sum excludes metadata, compression overhead, and checkpoint copies. The summed uncompressed ndarray payload across all 1,188 NPZ files is 752,655,568 bytes, partly because PCA and `z` are repeated in base/joint artifacts.

Exact examples, useful for a first read-only pilot:

- Llama-3.2-1B, C7, hidden-state index 6: `run_outputs/activation_cache/ec4467afd3f1bb58853b9e64d0fcaac4e9b2a8debcb9cd260689cf118ded8ab0.npz`.
- Qwen2.5-1.5B, D4, hidden-state index 11: `run_outputs/activation_cache/7fc8179c84a6f674ba287a646700a12281cc5e3fad49ce6f96278dce4d96d378.npz`.

The full exact per-cell mapping is in the generated index, including a raw structural cache shared by multiple split/dimension fits. Causal cache metadata also records explicit train/validation/test IDs; it can establish the split identity even for structural caches whose own metadata describes the whole template pool.

### 3.3 Model and readout axes

Here a layer means the saved Hugging Face `hidden_states` index, with embedding at index 0. The four first listed indices are primary; the last is terminal.

| Model | D | Saved indices |
|---|---:|---|
| `meta-llama/Llama-3.2-1B` | 2,048 | 3, 6, 10, 13; terminal 16 |
| `meta-llama/Llama-3.2-3B` | 3,072 | 6, 11, 17, 22; terminal 28 |
| `meta-llama/Llama-3.1-8B` | 4,096 | 6, 13, 19, 26; terminal 32 |
| `Qwen/Qwen2.5-1.5B` | 1,536 | 6, 11, 17, 22; terminal 28 |
| `Qwen/Qwen2.5-3B` | 2,048 | 7, 14, 22, 29; terminal 36 |
| `Qwen/Qwen2.5-7B` | 3,584 | 6, 11, 17, 22; terminal 28 |

Pinned revisions from the archived configuration, in the same row order: `4e20de362430cd3b72f300e6b0f18e50e7166e08`; `13afe5124825b4f3751f836b40dafda64c1ed062`; `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`; `8faed761d45a263340a0528343f099c05c9a4323`; `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`; `d149729398750b98c0af14eb82c78cfe92750796`. The historical tokenizer revision follows the model revision; it uses a fast tokenizer, special tokens, and right padding.

`src/bsg/representations/extract.py:59` requests HF hidden states and averages every token overlapping the unique character span of the source label. It returns stacked pooled states. Cache metadata describes these as block outputs, but the terminal HF hidden state can include the model's final normalization. A new extractor must name raw last-block output and normalized final readout separately. Do not silently reinterpret the old terminal cache as the raw last block.

### 3.4 Separately saved embedding caches

Only two small models × two groups were found in the inspected embedding extraction directory:

`runpod_output_by_date/result_10_10_2026/run_outputs/algebra_falsification/afs_d_rank_aware_001/checkpoints/embedding_extraction/`

| Filename | Model / group | `hidden` shape | Metadata latent dimension |
|---|---|---|---:|
| `02195ec7bb9e5cefcb3258f2d04c3c0288e576ef262b1956de54ab2854c7befd.npz` | Qwen-1.5B / C7 | `(280,1536)` float32 | 6 |
| `51ac10997b811b6a1581af9be012d846e58150aa60d5c481fea86246de45a6b5.npz` | Llama-1B / D4 | `(160,2048)` float32 | 3 |
| `a3a7afaba329cc634f0e40c5d8da485455decdd3e4370325c039bf0c4903c613.npz` | Qwen-1.5B / D4 | `(160,1536)` float32 | 3 |
| `a5803b441c55548b100aa7fea892e9bfe40dad1dce20a4770a5e2cc256eff517.npz` | Llama-1B / C7 | `(280,2048)` float32 | 6 |

Each also has scalar JSON `metadata`. These are raw hidden vectors despite the q=6/3 metadata; they are not restricted to a q-dimensional array. Llama revision is `4e20de362430cd3b72f300e6b0f18e50e7166e08`, Qwen revision is `8faed761d45a263340a0528343f099c05c9a4323`. The other four model embeddings require new extraction. Internal hidden states cannot reconstruct them.

## 4. Row identities, action direction, and leakage rules

The historical arrays lack explicit row text/IDs. Reconstruct rows from the frozen dataset builders, verify row counts, and record the reconstructed row manifest as a new derived artifact. C7 is template-major, then Monday through Sunday (`src/bsg/data/c7.py:31`). D4 is template-major, then NW, NE, SE, SW (`src/bsg/data/d4.py:32`). All group-expanded rows from a template stay in one split (`src/bsg/data/splits.py:30`).

Conventions that every new implementation must test before fitting:

- C7 states are 0..6 in weekday order; `a_k(i)=(i+k) mod 7`.
- D4 states are 0:NW, 1:NE, 2:SE, 3:SW. `r(i)=(i+1) mod 4`, and `s=[1,0,3,2]`. Elements are `(k,b)` in `((0,0),(0,1),(1,0),(1,1),...,(3,1))`. Their action applies `s` first if b=1, then `r` k times (`src/bsg/groups/d4.py:13`).
- `compose(g1,g2)` applies g2 first, then g1 (`src/bsg/groups/d4.py:32`). With row vectors, sequential application is `z @ R[g2] @ R[g1]`, compared with `R[compose(g1,g2)]`.
- Pair construction is within the same template: source `(t,i)` targets `(t,a_g(i))`; no pair crosses templates (`src/bsg/operators/independent.py:8`).
- PCA is training-centered and unwhitened: `z=(H-mean_train) @ C.T`. Its current implementation uses reduced SVD (`src/bsg/representations/pca.py:17`).

Future row manifests must contain dataset version and SHA, row ID, full prompt SHA, split/family ID, state/configuration ID, group-action target row IDs, target character/token offsets where applicable, readout position, and an explicit row ordering. Store prompt text in a separately versioned dataset file. Cache keys must include its digest, tokenizer/model revision, extraction source digest, precision, readout definition, and truncation/padding policy.

The original 40 templates have already been analyzed. New family labels assigned to these templates permit a useful **retrospective family-grouped stress test**, but cannot turn them into a fresh confirmatory holdout. Never describe repeated resplitting of these 40 templates as independent new evidence.

## 5. What can be done offline and what requires new inference

| Proposed operation | Saved input | New model forwards? | Qualification |
|---|---|---|---|
| Recompute metrics for a saved operator | Checkpoint `z`, `operators`, split IDs | No | Historical definitions and output count must remain unchanged |
| Norms, singular values, effective rank, condition summaries | `operators` | No | Planned diagnostic; not computed in this audit |
| Training state centroids / within-state residuals | Checkpoint `z` plus training IDs | No | Fit centroids only on training rows |
| New q, training-only whitening, residual PCA, family resplit | Structural `activations` | No | New analysis, not a historical replay |
| Different alpha, generator-only fits, matched baselines, action reassignment | `z` or raw `activations` | No | Refit into new outputs; lock validation rules first |
| Previously discarded PCA directions | Raw structural `activations` | No | Cannot recover them from only `z` and the retained decoder |
| Input embeddings for the four unsaved models | None | Yes | Four inspected embedding caches do not cover them |
| New lexical/prompt families or role-binding tasks | None | Yes | Dataset must be frozen before extraction |
| Tokenwise states, other positions/layers, attention, untrained models | None | Yes | Existing NPZ cannot supply them |
| New intervention accuracy/logits / natural-target controls | No sufficient cache | Yes | Scalar historical CSV metrics cannot evaluate a new replacement |

The causal prompts append ` Report the transformed state:` to structural text (`src/bsg/experiments/main_experiment.py:345`). Causal masking suggests a suffix cannot affect earlier source-token activations, but byte equality across these two cache types was not checked here. Verify it if substituting one cache for another. Changed prefixes, task instructions, answer positions, or state order can change the representation and must get a distinct cache identity.

## 6. Exact offline analysis protocol

### 6.1 Stages and fit boundaries

1. **Read-only pilot:** 16 primary cells = Llama-1B/Qwen-1.5B × C7/D4 × depths 0.4/0.8 × splits 101/303. Resolve cells through [CELL_INPUT_INDEX.json](evidence/CELL_INPUT_INDEX.json). Validate schemas, finite values, reconstructed row order, group table, split disjointness, PCA dimensions, and a small deterministic forward/reverse orientation contract. Do not write to the source archive or the original cache directory.
2. **Historical replay track:** load saved q=16 operators and evaluate unchanged historical metrics; do not refit them and call it the same result. Record discrepancies separately. Rerunning the historical fitting pipeline is a distinct track with its own code/environment identity.
3. **Retrospective diagnostic track:** fixed q=16, original five splits, 240 primary cells. Fit only training PCA/centroids/classifiers/operators; use only validation rows for declared tuning. Terminal and q=8/32/64 remain secondary strata, never a mechanism for choosing the best test layer or dimension.
4. **Prospective track:** use the new split described below, with frozen code, manifest, hypotheses, metrics, tuning budget, and decisions before revealing held-out outcomes.

The primary q=16 diagnostic comparison uses the saved PCA to isolate estimator changes. A separate raw-H PCA refit should reproduce its training projection up to the usual basis/sign ambiguities; comparisons should use projections and prediction quantities rather than requiring arbitrary SVD vector signs to match. New whitening/residual-PCA variants are separate representation conditions.

### 6.2 Common baseline inputs and effective capacity

Within each cell all primary estimators must receive the same training rows, group targets, PCA/readout, and validation budget. Report two distinct information tracks:

**Label-blind track:** inference receives only `(z,g)`. Independent Ridge and Joint belong here. A centroid decoder must infer the source state using a training-only classifier, for example nearest training centroid with a fixed metric. Its source-state error is reported, not hidden. If a richer classifier is used, predeclare its architecture and tuning budget.

**Oracle-label track:** inference also receives true state/configuration i. Compare with a source-conditioned category baseline and a symbolic control. This is a useful upper-bound explanation, but its metadata access differs from the ordinary operator and must be named explicitly.

Training centroids in q-space are `mu[i]=mean_{training contexts} z[t,i]`. The tied residual-copy control is:

`B(g,z,i) = z - mu[i] + mu[a_g(i)]`.

For complete balanced training orbits this is exactly the train-pair source-conditioned mean additive field:

`mean_t(z[t,a_g(i)]-z[t,i]) = mu[a_g(i)]-mu[i]`.

Its effective learned table has at most `(n_states-1)*q` identifiable mean parameters, although one could redundantly store `|G|*n_states*q` deltas. At q=16 this is 96 for C7 and 48 for D4, versus 1,792 / 2,048 unconstrained matrix entries for independent all-element operators. Joint has the same operator parameter count as Ridge. Report these counts and supplied labels; equal parameter count is not itself equal information or equal inductive bias.

The historical causal additive control is fitted in full D-dimensional hidden space (`src/bsg/interventions/causal.py:78`). Retain it as a historical comparator, and add a **matched-q additive** control. A q-dimensional operator and D-dimensional additive field do not operate in the same feature space. For S3 define whether the oracle has the full role assignment and entity IDs; a parser-based symbolic oracle is a separate information track.

To evaluate multi-step category baselines, update the current state after each action. Telescoping gives `mu[a_g(a_h(i))]-mu[i]` exactly, so algebraic closure may hold even if all contextual residuals are merely copied. Leaving the state index fixed during composition would implement the baseline incorrectly.

### 6.3 Residual test that discriminates category substitution

Use **training** centroids to form every split's residual, `r[t,i]=z[t,i]-mu[i]`. Fit independent or generator-only residual operators using training pairs `(r[t,i],r[t,a_g(i)])`. Evaluate:

- centroid-only prediction `mu[a_g(i)]`;
- residual copy `mu[a_g(i)] + r[t,i]`;
- fitted residual transformation `mu[a_g(i)] + r[t,i] @ A[g]`;
- ordinary full-z operator `z[t,i] @ R[g]`.

Run oracle-label and predicted-label versions where applicable. Compare absolute target MSE, pooled relative error, and residual prediction improvement against residual copy. Retain the unchanged historical mean-of-row relative metric as a separately named endpoint. Small target-residual norms can destabilize rowwise ratios; a fixed training-derived scale and absolute MSE prevent a near-zero denominator from dominating. Do not remove hard or small-norm test rows after viewing them.

Never subtract a test-template mean over its complete state orbit in the ordinary rowwise track. That supplies extra test-batch information. If a batch-transductive control is desired, name it and give the same complete test orbit to every competing estimator.

### 6.4 Selection, generator-only fitting, and normalization

**Historical replay:** keep fixed-alpha Ridge (`alpha=0.01`) and the historical Joint objective intact. The historical unscaled alpha is not the prospective tuning parameter. There is no additional raw-alpha sweep in the prospective protocol.

**Prospective and new validation-tuned comparisons:** follow [STATISTICAL_AND_SYNTHETIC_PROTOCOL.md](STATISTICAL_AND_SYNTHETIC_PROTOCOL.md). For training design matrix X with d input features, define the actual Ridge penalty as `rho = lambda * trace(X.T @ X) / d`, with dimensionless `lambda` in `{0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1}`. Fit `argmin_R ||X R-Y||_F^2 + rho ||R||_F^2`; compute the scale from training X only. Save lambda, actual rho, design rank, and numerical precision for each fit. Do not reuse the historical alpha grid under the name lambda or estimate the scale from validation/test rows.

For the zero-penalty candidate use float64 `pinv(X) @ Y`, with fixed relative singular-value cutoff `rcond=1e-12` on X itself: invert only singular values strictly greater than `1e-12 * s_max`. Do not apply this cutoff to `X.T @ X`, which squares the spectrum. A zero design matrix yields the zero minimum-norm predictor. Record the realized cutoff and retained rank; unsupported PCA rank remains a separately reported condition, not a reason to alter the cutoff after viewing results.

Select the strongest category predictor and one operator variant separately for each model/depth/group using mean validation prediction loss, not test algebra. Use the same declared tuning budget for relevant comparators. Ties within `1e-8` prefer the simpler predictor, then the smaller-norm operator, with a deterministic final candidate ordering. The category context-retention coefficient uses its separate grid `{0,0.25,0.5,0.75,1,1.25}`; it is not the Ridge penalty. Freeze the selected choices and publish every candidate's validation and test loss, including an independent estimator that does not win selection. A context-difference estimator may be selected separately on its declared validation difference target. Report loss normalization and Ridge effective degrees of freedom in addition to parameter counts.

For generator-only claims, train and select only on declared generator transition pairs. C7 uses generator 1; D4 uses r and s. Construct composite operators by the published row convention, without fitting free matrices for held-out elements. Do not select hyperparameters using validation outcomes of a supposedly unseen element, and do not provide a held-out relation as a Joint penalty. An all-element Ridge fit does not become relation-held-out merely because its code never consumes the multiplication table.

For new Joint analyses, retain the historical objective as one named baseline; do not assume a default value in current code equals the frozen run's actual steps. Extract actual weights, number of steps, seed, optimizer, and scaling from the frozen configuration. Any changed objective or early stopping is a new condition. Validation selection over closure regularization must be disclosed; small training closure error is expected when that closure is optimized.

PCA, whitening, centroids, feature scaling, state classifiers, norm caps, and numerical-rank tolerances are fitted or set on training data. Whitening must use a training spectrum and a predeclared floor. A similarity transform changes ambient Frobenius/orthogonality scores, so compare coordinate-invariant predictions and activation-supported errors alongside coordinate-sensitive operator metrics.

## 7. Consistent-action null: corrected finite design

Let the original action be `a_g(i)`. There are two different operations:

1. Reorder feature rows, state labels, and action tables together. This is a change of names. It must preserve the result and is an implementation invariance check, not a falsification.
2. Keep every hidden feature, lexical identity, template, split, and state label fixed, but replace targets by `a_g^pi(i)=pi^{-1}(a_g(pi(i)))`. Fit all estimators afresh using that one globally fixed action across every template and split. This tests whether the observed features favor the intended semantic action over other equally coherent actions.

Do not use an independent permutation for each template in operation 2; that destroys global action consistency and answers a different question. A family-specific inconsistent scramble remains a useful negative control, separately labeled.

Independent combinatorial enumeration, without any activation fitting, gives:

| State domain | Distinct labelled conjugate tables | Distinct action-image families | Nonoriginal image families |
|---|---:|---:|---:|
| C7 on seven weekday states | 720 | 120 | 119 |
| D4 on four corner states | 12 | 3 | 2 |

For C7, the six alternative generator choices inside a cyclic subgroup are group-element relabelings of the same action-image family. For D4, four labelled conjugates occur inside each of three image families. For all-element averages invariant to group-element reindexing, these labelled versions do not create new action families. Generator-specific semantic endpoints may distinguish them, but must not count them as independent sampled families.

Consequences:

- Exhaust the 120 C7 families and all three D4 families when feasible; fix a canonical representative and retain its permutation/action table. Report intended-action ranks and effect sizes, including ties.
- Under a justified uniform family-rank test, the smallest achievable rank probabilities including the original are 1/120 and 1/3 respectively. The labelled D4 table set has at most 1/12 granularity even for a generator-specific statistic. Sampling those same actions 5,000 times cannot improve the design's information. A semantic-action null also needs an explicit exchangeability rationale; a permutation rank is not automatically a valid causal p-value.
- Compare held-out target prediction at the same model/feature/label access and regularization budget. Do not match nulls to the test error by tuning on the test labels. If fitting-error matching is required, choose a validation-error/norm/rank band before unsealing test results and report any nulls excluded by that rule.
- A faithful C8 action on four states does not exist. Merely matching abstract group order eight is insufficient to match the natural D4 action. State count, faithfulness, stabilizers, cycle types, orbits, and supervision must be declared for any alternative group/action.
- Full-rank category codes can realize every finite permutation action on their states. Success on the intended action is not sufficient; the discriminating question is intended-action preference, contextual residual transformation, or new semantic/behavioral generalization beyond the category control.

## 8. Prospective extraction design and exact tensor axes

This section records the agreed prospective design. Dataset construction, family assignment, entity sets, and all held-out prompts must be frozen before extraction or model-based screening.

### 8.1 C7 and D4

Each domain has 96 context families × 2 paraphrases = 192 contexts. Split by family: 48 training, 16 validation, 32 test. All states of a context stay together. Primary readout is the mean of the source state span. A common final-position readout for C7/D4 is a separate secondary control, not a substitution for their primary readout.

| Domain | Full row axis | Training / validation / test rows |
|---|---|---|
| C7 | 192 contexts × 7 states = 1,344 | 672 / 224 / 448 |
| D4 | 192 contexts × 4 states = 768 | 384 / 128 / 256 |

Repeated paraphrases are nested within a family. If lexical variants are used, split their identity at the planned lexical level as well; a paraphrase split cannot claim lexical holdout when the same labels appear on both sides.

### 8.2 S3 role binding

S3 crosses the same 192 contexts with 48 entity triples and all six role assignments: **55,296 prompts**, not 288. Every row represents the whole prompt at **one final-token readout**. Do not concatenate token vectors for the three roles or permute readout coordinates: that would build a permutation representation into the measurement.

Context-family split is 48/16/32 (96/32/64 contexts); entity-triple split is 24/8/16. Define the ordered reference triple, role assignment label, S3 action table, source/target row map, and the final token before extraction. Make every action preserve the same context and same triple. If entities recur across nominally different triples, report and control that overlap; truly entity-disjoint generalization requires disjoint individual entity identities, not just distinct unordered triples.

The full cache includes nine crossed strata:

| Context partition × entity-triple partition | Train triples (24) | Validation triples (8) | Test triples (16) |
|---|---:|---:|---:|
| Train contexts (96) | 13,824 | 4,608 | 9,216 |
| Validation contexts (32) | 4,608 | 1,536 | 3,072 |
| Test contexts (64) | 9,216 | 3,072 | 6,144 |

Fit only the train×train block; tune only validation×validation. Primary confirmatory test is test×test. The test-context/train-entity and train-context/test-entity blocks isolate the two generalization demands as predeclared secondary tests. Other cross blocks are reserved diagnostics; they do not enter PCA, centroids, operator fitting, early stopping, or selection. Do not accidentally fit PCA on all 55,296 cached rows.

### 8.3 Storage and extraction implementation

Six pinned models × four primary depths × three domains gives **72 primary arrays** of shape `(row,D)` float32. The model dimensions sum to 16,384.

- C7 payload: `1,344 × 16,384 × 4 depths × 4 bytes`.
- D4 payload: `768 × 16,384 × 4 depths × 4 bytes`.
- S3 payload: `55,296 × 16,384 × 4 depths × 4 bytes = 14,495,514,624 bytes` (13.5 GiB).
- Primary total: `57,408 × 16,384 × 4 × 4 = 15,049,162,752 bytes` (14.015625 GiB), before manifests, checksums, compression, PCA outputs, and experiment results.

The C7/D4 secondary final-position readout adds 48 arrays and 553,648,128 bytes (528 MiB); primary plus these secondary arrays is 15,602,810,880 bytes (14.53125 GiB). Multiple readouts can be pooled in the same forward pass. A changed causal prompt still requires its own prompt/cache identity.

The largest primary array is S3 at D=4,096: `55,296×4,096×4 = 905,969,664 bytes` (864 MiB). Its train×train rows in float64 alone use 452,984,832 bytes (432 MiB). These are payload sizes, **not peak RAM or GPU requirements**. SVD, copies, BLAS workspace, tokenizer/model objects, logits, and batch outputs add memory. The q=16 encoded arrays for all 72 primary cells would total 176,357,376 bytes in float64 if saved once per cell without split duplicates.

Implement a bounded-batch extractor that pools only the declared readouts/depths and writes into new `.npy` memory maps or explicitly chunked arrays, plus an atomic completion manifest. Avoid retaining `N×all_layers×D` in a Python list: `extract_hidden_representations` currently does so (`src/bsg/representations/extract.py:78`). For S3 this would multiply the payload by the full number of layers. Do not write all-token caches by default. Store token offsets/IDs and selected diagnostic token states only for a small, predeclared QA subset.

At extraction, assert no truncation of any relevant token; record padding, attention mask, precision, and actual readout position. Validate a fixed sample against an unbatched forward to detect padding/position errors. A fixed final marker's **input embedding** may be identical on every S3 row, producing zero rank; report that fact rather than forcing q=16 on a rank-zero control. Embedding at a different informative token is a different readout control and must be named accordingly.

## 9. Behavior rerun: patch controls required by the implementation

The historical hook maps hidden-state index l to block l−1, then replaces **each** token overlapping the source label by the same replacement vector (`src/bsg/interventions/causal.py:254`). The identity condition skips patching (`src/bsg/experiments/main_experiment.py:499`). For multi-token labels, a mean-overwrite changes within-span variation, even for an identity transformation. This audit identifies the implementation fact; it does not measure its behavioral impact.

The new rerun must distinguish:

1. Unpatched original prompt.
2. Sham mean overwrite at the same layer/span, using the live original span mean.
3. Historical mean-overwrite intervention, retained as a named reproduction condition.
4. Token-residual-preserving intervention: at the live hook compute span mean h, then add `replacement_mean−h` to **each** source-span token. This changes the mean while preserving the tokens' deviations from it.
5. Matched-q learned, matched-q additive, and full-hidden additive replacements, all with the same patch rule, layer, prompt, precision, and site.
6. Natural-target activation patch from a matched prompt differing only by the transformed state, and a natural-target prompt forward. These are useful upper bounds and patch-validity controls, not trainable competitor results.

The natural-target state is oracle information; do not include it in a label-blind model comparison. For S3 patch the declared single final-token readout and preserve one common site across conditions. If task instructions or operation requests appear before that readout, use activations extracted from those exact task prompts rather than from a structurally different prefix.

Use complete candidate answer sequence log probability and constrained-choice accuracy, along with the historical first-token/logit endpoints as separate diagnostics. Source/target answers may be multi-token or share a first token. Distances of full vocabulary logits are not accuracy. Score both correct and matched-length wrong-order operations where they reach different targets. Controls with coincident targets cannot test noncommutativity.

Natural target references, patching, logits, and answer likelihoods require new forwards. Cached mean activations are enough to calculate replacement vectors, but cannot reveal downstream model responses. Start the causal feasibility pilot on the designated 8B/7B models; do not extrapolate feasibility from only 1B extraction.

## 10. Norm, SVD, and synthetic calibration outputs

Saved operators permit, without model inference: Frobenius norm, spectral norm, singular spectrum, effective rank, condition summaries with declared tolerances, per-cell prediction error, and activation-supported composition error. Report ambient and data-supported quantities separately. An operator can behave differently outside the observed activation span; small supported error is not a full ambient representation theorem.

Before applying a new estimator to LLMs, its implementation should be calibrated on frozen synthetic datasets containing:

- exact finite permutation codes with stable category means and context noise;
- a category-centroid model with copied contextual residuals;
- an exact contextual equivariant latent model with controlled non-isotropic noise;
- context-inconsistent target mappings;
- a known rank-deficient action;
- an invertible coordinate transformation of the same latent model.

The category and centroid conditions are **positive controls for recoverability but negative controls for a stronger contextual-mechanism claim**. Exact synthetic matrices should test the row convention and group table. Coordinate transformations test which reported metrics depend on Euclidean coordinates. Fit/selection/test boundaries in synthetic runs must match the declared LLM comparison; sample size, signal/noise, and effective rank are explicit factors, not tuned after observing which condition favors the proposed method.

Store known truth, training/validation/test sample IDs, generation seed, action table, signal/noise scales, and exact intended failure criterion. No synthetic outcomes have been produced in this audit.

## 11. RAM and timing pilot without fabricated compute estimates

For the existing FULL arrays, the largest structural matrix is `280×4096` float32 = 4,587,520 bytes; the largest D4 matrix is 2,621,440 bytes. A q=64 decoder at D=4096 is 2,097,152 float64 bytes. Process one cache and its relevant splits at a time rather than loading all repeated checkpoints. Historical centered training rank is at most 195 (C7) or 111 (D4); observed numerical rank still needs measuring on training data using a frozen tolerance.

For prospective S3, select a reduced/economy or chunked covariance PCA implementation deliberately. The training matrix can have more rows than D, unlike FULL. A covariance implementation must control numerical precision and be verified against direct reduced SVD on a small fixture; an approximate/randomized PCA is a new numerical method and requires an error tolerance and seed. Choose worker count from measured RSS; avoid unrestricted BLAS threads multiplied by process workers.

Historical `runtime_profile.csv` contains 2,168 rows and fields including `experiment_module`, model/group/layer/q/split, wall-clock seconds, peak allocated/reserved GPU bytes, cache bytes, events, and timestamps. These are environment-specific records, not a reliable price or new-runtime forecast.

The executable pilot must record, on the actual proposed machine:

- GPU model/count, driver/CUDA, CPU/RAM, package versions, model revision, precision, device mapping, sequence length, actual padded tokens, batch size, and cache status;
- cold model load and warm extraction separately;
- per-domain extraction, readout pooling, cache I/O, PCA, Ridge, Joint, and causal forward phases separately;
- synchronized GPU timing, PyTorch peak allocated/reserved bytes, and driver/device memory where available; allocated tensor memory alone is not total device use;
- process peak RSS and disk bytes, plus prompts/tokens per second;
- a repeated warmed sample with median/range, and failures/OOMs including the attempted batch size;
- numerical checksums and output checks so that a faster configuration has not changed the protocol.

Use the 16 cached cells for a CPU-memory/throughput pilot. For new inference, use a prespecified prompt subset spanning token-length and multi-token-label cases, then pilot the designated larger causal models. Scale by measured rows, tokens, layers, and patched-forward counts for each model/domain. Extrapolation from one model size to another is provisional until measured. No GPU hours, prices, or minimum VRAM are asserted by this document.

## 12. Output contract and stopping rules

Proposed components are a specification, not a claim that scripts already exist:

| Component | Required outputs and invariants |
|---|---|
| Read-only artifact importer | Source archive/member hashes; row and split manifest; no calls to cache `store`/`discard`; reject incomplete or mismatched identities |
| Offline diagnostic runner | One cell ID per model/domain/readout/q/split/condition; fit provenance; hyperparameter selection table; per-template/per-family metrics; operators and scales |
| Prospective extractor | Chunk checksums; dataset/revision/readout identities; explicit incomplete/complete state; bounded writes into new storage |
| Baseline runner | Label-access track, parameterization, effective capacity, target/action table, common input dimension and training set |
| Behavioral runner | Prompt/candidate answer IDs, source/target/site, patch rule, unpatched/sham/natural-target controls, sequence scores, timing |
| Analysis | Locked estimands, family/entity clustering, paired method differences, multiplicity policy, all failed cells and exclusions |

The historical `ActivationCache` constructor creates a directory and its class includes mutating store/discard methods (`src/bsg/representations/cache.py:25`). Do not instantiate it on an immutable source path merely to read arrays. Read from the ZIP using an explicit read-only wrapper. A current example of the pattern is `InternalActivationSource` in `src/bsg/experiments/afs_d.py:324`; reuse its provenance discipline, not its run identity.

Stop a phase for a schema/provenance mismatch, cross-split leakage, nonfinite fit, unsupported rank, wrong action orientation, missing sham behavior, or unplanned holdout access. Record the failure and fix implementation before unsealing outcomes. Do not silently omit a cell or switch q/alpha/readout to make it pass. Resource failure may change batching or storage format without changing the scientific sample or readout; any scientific design change requires a new frozen manifest.

Uncertainty is over the declared contexts/families/entities, not millions of repeated method rows. Six selected models, four within-model depths, repeated seeds, and every group element are not independent populations. For crossed S3 tests, account for both context family and entity-triple clusters; two paraphrases and six configurations are nested repeated measurements. Report model-specific results and the finite selected-model aggregate separately. A preregistered gate is a decision rule, not evidence that all stronger interpretations have passed.

The minimal implementation order is: freeze manifests → validate exact cached cells → calibrate synthetic estimators → run the 16-cell cached pilot → run all 240 cached primary cells → freeze prospective data and tuning → measure new extraction/causal pilots → execute the authorized prospective reruns → unseal the held-out evaluation once. This document authorizes no execution or cloud spending.
