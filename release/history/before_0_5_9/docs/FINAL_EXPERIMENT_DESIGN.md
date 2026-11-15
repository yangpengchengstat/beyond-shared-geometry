> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# Final Large-Experiment Design

Status: methodology frozen after completed PRE-FULL audit; FULL authorized
Audit date: 2026-09-02  
Canonical repository: `.`  
Audited source commit: `48bd5f17a13def9838879ef29b4982816c1194f8`

METHOD STATUS: FROZEN FOR FULL
FULL STATUS: AUTHORIZED

Authorization marker (2026-09-04): PRE-FULL run `pre_full_001` completed
successfully. The source-conditioned additive audit at commit
`06981aaf7e3645785b64c7dc5628ca8cfd89c1e5` returned
`PASS WITH REPORTING LIMITATION` and `AUTHORIZE FULL`. This marker changes
authorization status only; it does not revise any scientific setting below.

This document is the pre-run specification for the final large experiment. It
records the current implementation exactly, then defines the proposed final
protocol. It does not authorize a GPU run. The canonical config, experiment
code, datasets, protocol, and paper mapping must not change until this design
is approved. After approval, the changes in the implementation checklist must
be made in place and validated before any scientific GPU result is inspected.

## Post-freeze implementation clarification (approved 2026-09-02)

The user approved one implementation-level correction after the standalone
freeze commit `30ae942c510db1ed622c16cdbc78f2ce17b556b1`. This note is append-only
and supersedes only the SMOKE entries below; CORE and FULL scientific settings
are unchanged.

- SMOKE uses latent dimension 16, the canonical primary dimension.
- SMOKE uses seven templates per group with a deterministic 5/1/1
  train/validation/test split.
- D4 therefore has 20 training rows and centered training rank at most 19, so
  dimension 16 is feasible. Configuration validation must reject any latent
  dimension greater than centered training rank.

The required D4 behavioral comparison also executes a direct `r^-1`
intervention in addition to `srs`. This makes the D4 causal grid eight paths,
not seven. With base-prompt reuse, the corrected planned causal forward counts
are 254 SMOKE, 105,768 CORE, and 876,780 FULL. This corrects workload
accounting only; it does not alter a scientific endpoint or FULL setting.

## Post-CORE prospective amendment (approved 2026-09-03)

This prospective amendment governs PRE-FULL and the revised FULL run. It does
not rewrite, reinterpret, or modify the completed dated CORE snapshot. CORE
retains its historical `historical_seed_replicated` configuration so its
ledger and rows remain interpretable exactly as run.

The joint optimizer is scientifically deterministic with respect to its seed:
it starts from deterministic independent-ridge matrices, performs full-batch
AdamW updates, and contains no random initialization, dropout, minibatch
sampling, or shuffled update order. The seed setter therefore does not alter
the learned fit, as confirmed by exact CORE equality across seeds 42, 202, and
777. PRE-FULL/FULL use one fit with technical seed 42 per independent
model x group x layer x latent-dimension x split cell. Optimization seed is
retained as provenance/API metadata, not a replication axis. Historical CORE
inference collapses learned rows that are exact copies differing only in that
technical seed label. Genuine random-control seeds are never collapsed.

The revised inference hierarchy is `model -> depth -> split -> template` for
an aggregate cross-model estimand. A model-specific claim conditions on one
model and resamples `depth -> split -> template`; it does not use models as
replicates. Templates are clusters carrying all group-expanded states. A
random-control seed level is added only to a contrast involving the random
control. Deterministic learned-fit uncertainty, data/split uncertainty, and
random-control Monte Carlo variation are reported separately. Pairing occurs
at the lowest shared scientifically valid cell before hierarchical resampling.

The primary additive comparator is now the stronger training-only,
source-conditioned baseline

`v_(g,s) = mean_(t in T_train) [h(g s,t) - h(s,t)]`,

applied on a held-out template as `h'(s,t_test) = h(s,t_test) + v_(g,s)`.
It may depend on both element and source state; the learned operator instead
represents one shared transformation law. No validation/test activation enters
fitting. The former orbit average `mean_x[h(gx)-h(x)]` cancels exactly on a
balanced complete orbit and is historical CORE-only, labeled `mean_difference`.

For D4, row-vector order gives `s @ r @ s = r^-1`. The matched-length
wrong-order control is `r @ s @ r`, whose exact group product is `s`, not
`r^-1`. Its downstream output is nevertheless compared with the same direct
`r^-1` reference. The confirmatory paired contrast is
`distance(srs,r^-1) - distance(rsr,r^-1) < 0`. This is order-sensitive
activation-intervention evidence, not by itself proof of noncommutativity or a
complete causal mechanism.

PRE-FULL uses Llama-3.2-1B and Qwen2.5-1.5B, C7 and D4, depth slots 0.40 and
0.80, split seeds 101 and 303, latent dimension 16, one deterministic learned
fit per cell, random-control seeds 104729 and 130363, and 2,000 hierarchical
bootstrap replicates. It is calibration only and is not yet authorized to run.
It subsequently completed as `pre_full_001`; the 2026-09-04 authorization
marker above supersedes only this prospective status sentence.

Revised FULL keeps all six models, both groups, all four primary depth slots,
five split seeds, and latent dimension 16 over the full confirmatory structural
grid; terminal structural diagnostics are retained. Dimensions 8, 32, and 64
are sensitivity analyses only at depths 0.40/0.80 and split seeds 101/303.
Cross-model and causal analyses use dimension 16 only. FULL uses five
independent random-control seeds and 5,000 bootstrap replicates.

Expected revised scientific units are 6 model loads, 12 structural extraction
units, 60 held-out causal extraction units, 444 base fits, 444 joint fits, 240
causal units, and 360 cross-model units. The corresponding PRE-FULL counts are
2, 4, 8, 16, 16, 16, and 8. The frozen FULL projection was 113.66 serial A40
hours. Complete FULL post-processing expects 926 bootstrap analysis rows, 9
stable tables, and 9 SVG/PDF figures with source CSVs; PRE-FULL expects 398
bootstrap rows with the same stable artifact contract. Direct CORE mean-time
scaling gives 9.62 hours for revised scientific units before increased control
cost. Accounting for five random controls, the additive comparator, the extra
D4 path, and repeated cross-model controls gives 42,513 seconds, or about
11.81 hours. Adding roughly 0.2-1.0 hour for post-processing gives a planning
range of 12.0-12.8 serial A40 hours, an estimated 88.7%-89.4% reduction.
PRE-FULL telemetry is the authoritative calibration before any FULL launch.
Completed PRE-FULL telemetry projects 37,471.07 serial-equivalent A40 seconds,
or 10.41 hours; this estimate supersedes the earlier CORE-derived planning
estimate without changing the scientific workload.

## 1. Current canonical audit

### 1.1 Models, revisions, and layers

The six configured identifiers are the official Hugging Face repositories.
Every current revision is the mutable branch name `main`.

| Key | Official model ID | Current revision | Blocks | Structural layers | Causal layers |
|---|---|---:|---:|---|---|
| `llama_1b` | `meta-llama/Llama-3.2-1B` | `main` | 16 | 2, 4, 6, 8, 10, 12, 14, 16 | 4, 8, 12, 14 |
| `llama_3b` | `meta-llama/Llama-3.2-3B` | `main` | 28 | 4, 7, 10, 14, 18, 21, 24, 28 | 7, 14, 21, 24 |
| `llama_8b` | `meta-llama/Llama-3.1-8B` | `main` | 32 | 4, 8, 12, 16, 20, 24, 28, 32 | 8, 16, 24, 28 |
| `qwen_1_5b` | `Qwen/Qwen2.5-1.5B` | `main` | 28 | 4, 7, 10, 14, 18, 21, 24, 28 | 7, 14, 21, 24 |
| `qwen_3b` | `Qwen/Qwen2.5-3B` | `main` | 36 | 4, 9, 14, 18, 22, 27, 32, 36 | 9, 18, 27, 32 |
| `qwen_7b` | `Qwen/Qwen2.5-7B` | `main` | 28 | 4, 7, 10, 14, 18, 21, 24, 28 | 7, 14, 21, 24 |

Layer `l >= 1` is the output of transformer block `l`. Embedding hidden state
0 is excluded. Runtime validates the configured block count. Terminal blocks
are excluded from source-token intervention but currently included in the
structural grid.

Official source pages:

- [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)
- [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)

### 1.2 Tasks and template splits

Current C7 contains 18 templates crossed with seven weekdays: 126 rows.
Current D4 contains 16 templates crossed with four square corners: 64 rows.
The group action expands a template only after the template is assigned to a
split, so all states of one template remain together.

`split_templates` shuffles template IDs using a split seed and takes
`round(template_count * 0.70)` for fitting. There is no validation split.

| Tier | Templates/group | C7 train/test templates | C7 train/test rows | D4 train/test templates | D4 train/test rows |
|---|---:|---:|---:|---:|---:|
| SMOKE | first 4 | 3 / 1 | 21 / 7 | 3 / 1 | 12 / 4 |
| CORE | first 10 | 7 / 3 | 49 / 21 | 7 / 3 | 28 / 12 |
| FULL | all canonical | 13 / 5 | 91 / 35 | 11 / 5 | 44 / 20 |

PCA is fit only on fitting rows and then applied to held-out rows. Causal
evaluation uses the first `causal_prompt_count` held-out template IDs. The
current FULL value of five therefore uses every held-out template.

### 1.3 Fitting and regularization

Current estimators are `independent_ridge` and `group_consistent_joint`.
Independent ridge also initializes every joint fit.

| Setting | Current value |
|---|---:|
| Latent dimensions | 8, 16, 32 |
| Ridge alpha | 0.01 |
| Joint learning rate | 0.002 |
| Joint steps, FULL | 1,200 |
| Joint steps, CORE | 500 |
| Joint steps, SMOKE | 20 |
| AdamW weight decay | 0.00001 |
| Homomorphism weight | 1.0 |
| Relation weight | 1.0 |
| Identity weight | 1.0 |
| Orthogonality weight | 0.05 |
| Activation dtype | model `float16`, cached `float32` |
| Batch size | 8 |

The joint objective contains data fit, homomorphism, defining relation,
identity, and orthogonality terms. The code recomputes PCA and independent
ridge inside every method and optimization-seed loop. Consequently,
deterministic ridge work is duplicated across methods and optimization seeds.

### 1.4 Current seeds and tier values

- FULL split seeds: `42, 202, 777, 1001, 2026`.
- FULL optimization seeds: `42, 202, 777, 1001, 2026`.
- CORE uses the first two values for both seed roles.
- SMOKE uses `42` for both seed roles.
- Bootstrap seed: `314159`.
- Bootstrap replicates: 100 SMOKE, 1,000 CORE, 5,000 FULL.
- A causal random-operator seed is derived from split seed, optimization seed,
  and layer. It is not a separate configured random-baseline seed.
- PCA receives the split seed, although its exact SVD implementation is
  deterministic and does not consume it.

### 1.5 Current structural and representation metrics

The structural CSV reports held-out-template values for:

- normalized equivariance, `E_eq`;
- identity transformation baseline, `E_identity_baseline`;
- data-conditioned and operator-space homomorphism;
- data-conditioned and operator-space defining relations;
- data-conditioned and operator-space composition;
- data-conditioned and operator-space inverse consistency;
- generator-only word generalization;
- PCA explained variance and joint-objective terms.

There is no matched random-orthogonal structural baseline. Identity is useful
as a transformation baseline but is not a meaningful negative control for
group-law metrics because identity operators satisfy those laws exactly.

C7 diagnostics contain generator eigenvalues, unit-circle RMSE, nearest
seventh-root angle RMSE, conjugate-pair count, `R^7` operator error, and
learned-versus-exact normalized character RMSE. D4 diagnostics contain
character values, approximate irrep multiplicities, `r^4`, `s^2`, and
learned-versus-exact character errors. D4 generator spectra are not currently
reported.

### 1.6 Current cross-model analysis

Artifacts are matched on group, rounded numeric normalized depth, split seed,
optimization seed, and latent dimension. The code then computes all model
combinations present at an exactly equal rounded depth:

- linear CKA on held-out full hidden states;
- orthogonal Procrustes residual on held-out PCA latents;
- an intertwiner fit and residual using the same complete operator set;
- `intertwiner_error - procrustes_error`.

The current layer grids only align all six models at depths 0.25, 0.50, 0.75,
and 1.0. Other depths produce within-block-count comparisons or incorrectly
labeled self-smoke rows. There is no explicit model-pair matrix, shuffled-label
control, random-map control, or comparison of algebraic commutation residual
for the Procrustes map. Procrustes reconstruction error and intertwiner
commutation error have different meanings, so subtracting them is not a valid
test that algebra adds information beyond geometry.

### 1.7 Current causal analysis

The intervention is the residual-preserving row-vector operation

`h' = h - z C + (z R) C`, where `z = E(h)`.

C7 uses `R^k` for `k=1,...,7`; steps 1 through 6 are transformations and step
7 is cycle recovery. D4 uses `r`, `s`, `r^2`, `r^3`, `r^4`, `s^2`, and `srs`,
with the last three assigned recovery/consistency roles. Methods are learned
joint operator, identity, one dimension-matched random orthogonal operator,
and train-only mean-difference steering.

Current behavioral outputs are target first-token logit change, probability
change, top-1 success, target-versus-source logit margin, recovery-only hidden
displacement, recovery probability error, and learned relation-operator error.
The implementation does not yet report general hidden displacement,
downstream-state displacement, final-logit displacement, or direct behavioral
agreement between D4 `srs` and learned `r^-1`. Multi-token targets are reduced
to their first context-conditioned continuation token.

Base logits and tokenization are repeated for every optimization seed,
dimension, and layer. All detailed rows remain in memory until the entire
six-model run finishes. There is no resume/checkpoint contract.

### 1.8 Current bootstrap and outputs

The percentile bootstrap resamples split seeds, optimization seeds within
split, and held-out template IDs within optimization seed. Group-expanded
states remain together only because each template row already contains a
metric averaged over group operations. Structural intervals pool all layers
and dimensions within model/group/method. Causal intervals only compare
learned against random orthogonal, not learned against identity.

Current outputs are detailed source CSVs, broad summaries, two truncated TeX
tables, two simple SVGs, a manifest, environment snapshot, and config snapshot.
They do not implement the final stable paper-artifact names in Section 12.

### 1.9 Current activation cache

On a miss, extraction requests all transformer hidden states in one model
forward sweep, then stores each requested layer separately as compressed
`float32` NPZ. Keys include model ID, resolved model revision, task string,
layer, template IDs, the full configured split-seed list, and tokenizer
settings. Structural activations are reused across fits. Causal activation
tasks are keyed separately per split seed, so the same prompt can be stored
again when it appears in multiple held-out splits. Cache metadata is encoded
in the key but is not independently checked on load. Cache files are excluded
from result bundles.

### 1.10 Current execution size

Counts below describe the current runner, not the proposed final protocol.
An extraction pass is one full sweep over a task frame, independent of the
number of requested layers.

| Tier | Model loads | Extraction passes | `_fit_setting` calls | Ridge solves | Joint fits | Causal forwards | Cross rows | Uncompressed activation cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SMOKE | 1 | 4 | 6 | 6 | 4 | 242 | 2 self diagnostics | about 0.8 MB |
| CORE | 2 | 12 | 256 | 256 | 128 | 23,232 | 32 cross + 64 self rows | about 8 MB |
| FULL | 6 | 72 | 14,400 | 14,400 | 7,200 | about 2,178,000 | 11,400 cross + 600 self rows | about 172 MB |

These counts make the current FULL implementation a no-go even before the
scientific changes below.

## 2. Frozen final model matrix

FULL includes all six models. The official API revisions observed at freeze
time are pinned below. A mutable `main` revision is forbidden in a scientific
run.

| Family | Model ID | Immutable revision | Blocks | Hidden size |
|---|---|---|---:|---:|
| Llama | `meta-llama/Llama-3.2-1B` | `4e20de362430cd3b72f300e6b0f18e50e7166e08` | 16 | 2,048 |
| Llama | `meta-llama/Llama-3.2-3B` | `13afe5124825b4f3751f836b40dafda64c1ed062` | 28 | 3,072 |
| Llama | `meta-llama/Llama-3.1-8B` | `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` | 32 | 4,096 |
| Qwen | `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` | 28 | 1,536 |
| Qwen | `Qwen/Qwen2.5-3B` | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | 36 | 2,048 |
| Qwen | `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` | 28 | 3,584 |

Preflight must resolve each requested revision to the identical full hash and
fail otherwise.

## 3. Frozen layer design

Primary depth slots are `0.20, 0.40, 0.60, 0.80`. For a model with `N` blocks,
map each slot to the nearest integer block `round(slot * N)`; there are no
half-integer ties in this matrix. Cross-model matching uses the declared slot
label, never equality of the realized numeric fraction.

| Model | 0.20 | 0.40 | 0.60 | 0.80 | Realized primary depths | Terminal structural diagnostic |
|---|---:|---:|---:|---:|---|---:|
| Llama 1B | 3 | 6 | 10 | 13 | .1875, .3750, .6250, .8125 | 16 |
| Llama 3B | 6 | 11 | 17 | 22 | .2143, .3929, .6071, .7857 | 28 |
| Llama 8B | 6 | 13 | 19 | 26 | .1875, .4063, .5938, .8125 | 32 |
| Qwen 1.5B | 6 | 11 | 17 | 22 | .2143, .3929, .6071, .7857 | 28 |
| Qwen 3B | 7 | 14 | 22 | 29 | .1944, .3889, .6111, .8056 | 36 |
| Qwen 7B | 6 | 11 | 17 | 22 | .2143, .3929, .6071, .7857 | 28 |

The four primary slots are used for structural, cross-model, and causal
analysis. The terminal block is retained only as an explicitly exploratory
structural/irrep endpoint for continuity with prior structural experiments.
It is excluded from primary depth-robustness decisions and all source-token
causal interventions.

## 4. Frozen templates and splits

The current template pools are too small for a disjoint validation set and a
64-dimensional D4 PCA. Even using all 16 current D4 templates gives only 64
rows and centered PCA rank at most 63. The final pool is therefore fixed at 40
templates per group before any GPU run.

The existing C7 templates 0-17 and D4 templates 0-15 remain unchanged. Add the
following exact templates in the approved implementation.

### 4.1 C7 additions, IDs 18-39

1. `The noted day of the week is {}.`
2. `The planned weekday reads {}.`
3. `The log identifies the day as {}.`
4. `The appointment falls on {}.`
5. `The timetable points to {}.`
6. `The stated weekday is {}.`
7. `The named day is {}.`
8. `The workweek marker is {}.`
9. `The calendar entry names {}.`
10. `The day assigned to the task is {}.`
11. `The listed day of the week is {}.`
12. `The chosen weekday reads {}.`
13. `The day's label is {}.`
14. `The weekly position is labeled {}.`
15. `The noted calendar day is {}.`
16. `The schedule's weekday is {}.`
17. `The current weekday label is {}.`
18. `The recorded day of the week is {}.`
19. `The week's selected day is {}.`
20. `The day shown in the entry is {}.`
21. `The specified day is {}.`
22. `The weekday named in the note is {}.`

### 4.2 D4 additions, IDs 16-39

1. `The marked square location is {}.`
2. `The active vertex is {}.`
3. `The diagram's corner label is {}.`
4. `The noted square position is {}.`
5. `The indicated vertex is {}.`
6. `The square's selected vertex is {}.`
7. `The labeled corner reads {}.`
8. `The active square position is {}.`
9. `The diagram marks {}.`
10. `The location label is {}.`
11. `The square entry identifies {}.`
12. `The vertex under discussion is {}.`
13. `The specified corner is {}.`
14. `The recorded square position is {}.`
15. `The chosen vertex is {}.`
16. `The diagram's active position is {}.`
17. `The noted corner reads {}.`
18. `The square label points to {}.`
19. `The selected square location is {}.`
20. `The record names the corner as {}.`
21. `The highlighted square position is {}.`
22. `The displayed vertex is {}.`
23. `The corner label in the diagram is {}.`
24. `The square's indicated location is {}.`

Every template must contain exactly one target placeholder and pass unique
target-span/tokenization checks for all six tokenizers. Duplicate normalized
text and target leakage are fatal.

For CORE and FULL, each split seed independently shuffles 40 template IDs and
assigns exactly 28 train, 6 validation, and 6 test templates. PCA and operators
use train only. Validation is used only for predeclared numerical/integrity
diagnostics; it cannot select layers, dimensions, metrics, or hyperparameters.
Final structural and behavioral estimates use test only.

| Group | Templates | Expanded rows | Train | Validation | Test | Train rows | Validation rows | Test rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C7 | 40 | 280 | 28 | 6 | 6 | 196 | 42 | 42 |
| D4 | 40 | 160 | 28 | 6 | 6 | 112 | 24 | 24 |

D4 train PCA rank is at most 111, so dimensions through 64 are feasible.
SMOKE uses a fixed five-template whitelist per group with a 3/1/1 split,
dimension 8, and no inferential interpretation.

## 5. Frozen dimensions, seeds, and fitting

### 5.1 Latent dimensions

- Structural and representation-theoretic FULL: `8, 16, 32, 64`.
- Cross-model FULL: all four dimensions; dimension 16 is primary.
- Causal FULL: dimension 16 only, preregistered as primary to control compute.
- CORE: dimension 16 only.
- SMOKE: dimension 8 only.

The primary paper structural result is dimension 16. Dimensions 8, 32, and 64
are prespecified sensitivity analyses, not alternate opportunities to choose a
favorable result.

### 5.2 Separate seed namespaces

| Role | FULL values |
|---|---|
| Template split | `101, 303, 909, 2701, 4243` |
| Operator optimization | `42, 202, 777, 2026, 31415` |
| Random baseline | `104729, 130363, 155921, 196613, 262147` |
| Bootstrap | `314159` |

Random baseline seeds are paired one-to-one, in listed order, with optimization
seeds. They are never derived arithmetically from split, optimization seed, or
layer. CORE uses the first three values in each list. SMOKE uses the first one.

### 5.3 Frozen optimizer settings

Keep ridge alpha 0.01, joint learning rate 0.002, 1,200 FULL and CORE joint
steps, weight decay 0.00001, and objective weights 1.0/1.0/1.0/0.05. SMOKE may
use 20 steps. Deterministic PCA and ridge are computed once per
model/group/split/dimension/layer and reused across optimization seeds and
methods. No test metric may tune these values.

## 6. Frozen structural and representation analysis

### 6.1 Estimators and controls

- Estimators: independent ridge and group-consistent joint.
- Identity baseline: transformation/equivariance only.
- Matched random orthogonal: identity element fixed to `I`; every nonidentity
  group element receives a separate Haar orthogonal matrix from its configured
  random seed. It is evaluated on exactly the same PCA/test rows.
- Learned-label shuffle: diagnostic structural control that permutes
  nonidentity group labels within a fitted operator set.

Identity is not compared on homomorphism/relation/inverse claims because it is
an exact but behaviorally useless trivial representation.

### 6.2 Metrics

Primary structural endpoints, separately for C7 and D4:

1. `E_eq` on held-out test templates.
2. `E_rel_data` on held-out test activations.

Secondary prespecified endpoints are `E_hom_data`, `E_comp_data`,
`E_inverse_data`, generator-word test error, and operator-space counterparts.
Identity consistency is reported separately. All results retain model, depth
slot, realized layer, latent dimension, split, optimization seed, random seed,
and test template.

### 6.3 Representation-theoretic outputs

C7 primary representation metrics are normalized `R^7` data/operator error,
nearest seventh-root angle RMSE, and unit-circle RMSE. Full eigenvalue spectra,
conjugate pairs, candidate real 2D rotation blocks, and character discrepancy
are diagnostic figures.

D4 primary representation metrics are separate `r^4`, `s^2`, and `srs-r^-1`
data/operator errors. Generator spectra for both `r` and `s`, character values,
approximate A1/A2/B1/B2/E multiplicities, block stability, and learned-versus-
exact character discrepancy are diagnostic. Approximate multiplicities remain
real-valued and are never silently rounded.

## 7. Frozen cross-model analysis

### 7.1 Pair matrix

Within-family scale pairs:

- Llama 1B - Llama 3B
- Llama 3B - Llama 8B
- Llama 1B - Llama 8B
- Qwen 1.5B - Qwen 3B
- Qwen 3B - Qwen 7B
- Qwen 1.5B - Qwen 7B

Cross-family matched-scale pairs:

- Llama 1B - Qwen 1.5B
- Llama 3B - Qwen 3B
- Llama 8B - Qwen 7B

No other model pair is primary. Matching uses group, declared depth slot,
split, latent dimension, and exact test-row order.

### 7.2 Metrics and hypothesis tests

Report linear CKA on full hidden states, latent orthogonal Procrustes residual,
and fitted intertwiner commutation residual. Also compute the operator
commutation residual produced by the Procrustes map itself, a random orthogonal
map, and shuffled group labels.

The primary algebra-beyond-geometry contrast is

`E_comm(fitted intertwiner) - E_comm(Procrustes map)`.

The second primary control contrast replaces the Procrustes map with shuffled
group labels. Negative values favor algebra-specific alignment. Do not subtract
Procrustes sample-reconstruction error from intertwiner commutation error.

Association between geometry and algebra is summarized by Spearman correlation
of CKA with negative fitted commutation error, with a clustered interval. The
claim that they are related but not identical requires a positive point
association and a strictly negative 95% interval for the fitted-minus-
Procrustes commutation contrast in both within-family and cross-family strata.

## 8. Frozen causal intervention analysis

This remains an activation-intervention causal pilot. It is not proof of a
complete causal mechanism.

### 8.1 C7

- Transformation: `R^k`, `k=1,...,6`.
- Cycle recovery: `R^7`, reported separately and never counted as a
  transformation success.
- Primary latent dimension: 16.
- Layers: all four primary depth slots, never terminal.
- Prompts: all six test templates and all seven source states.

### 8.2 D4

- Transformations: `r`, `s`, `r^2`, `r^3`.
- Recovery: `r^4` versus identity and `s^2` versus identity.
- Noncommutative consistency: behavioral output from `srs` versus output from
  the directly learned `r^-1` operator.
- Prompts: all six test templates and all four source states.

### 8.3 Methods and metrics

Methods are learned joint operator, identity, paired-seed matched random
orthogonal, and train-only mean-difference steering. Mean-difference remains
only if tests verify that it uses no validation/test activation.

For every intervention report:

- target top-1 success;
- target first-token logit gain;
- target first-token probability gain;
- target-versus-original first-token logit margin;
- normalized displacement at the patched hidden state;
- normalized downstream displacement at the final residual stream and final
  prompt position;
- normalized L2 displacement of final prompt-position logits.

Sequence-level target log probability is a required secondary metric so that
first-token reduction cannot hide multi-token behavior. For D4 `srs`, directly
compare final-logit vectors, downstream states, target probabilities, and
margins against direct `r^-1` under every method.

Base logits and unpatched hidden states are computed once per
model/group/split/example and reused. Patched calls may be batched only after a
batch-versus-single equivalence test. Detailed rows are checkpointed by stable
resume key after each model/group/split.

## 9. Frozen statistical plan

FULL uses 5,000 percentile bootstrap replicates; CORE uses 2,000; SMOKE uses
100 only as a runtime check. Confidence level is two-sided 95%.

The independent source units are prompt templates and optimization
initializations. Group-expanded states, layers, dimensions, and interventions
are matched repeated observations, not IID samples. Each replicate:

1. resamples source template IDs as clusters, carrying every state and matched
   method row for a selected template;
2. resamples optimization seeds independently, carrying their paired random
   baseline seeds;
3. resamples split seeds as partition-sensitivity blocks without treating
   repeated appearances of the same source template as new templates;
4. computes paired method contrasts before aggregation.

Primary estimates aggregate first within template over group states, then
within optimization seed, then over optimization seeds and split blocks.
Models and four primary depth slots remain matched. Terminal layers and
nonprimary dimensions cannot enter a primary estimate.

Every primary table reports estimate, 95% CI, learned-minus-random difference
and CI, and learned-minus-identity difference and CI where identity is
meaningful. Ties do not count as improvement. CORE is descriptive and cannot
replace FULL. No significance stars are used.

## 10. Frozen execution tiers

| Property | SMOKE | CORE | FULL |
|---|---|---|---|
| Purpose | runtime only | scientific pipeline validation | paper-grade evaluation |
| Models | Llama 1B | Llama 1B/3B, Qwen 1.5B/3B | all six |
| Primary depth slots | 0.40 only | 0.40, 0.80 | 0.20, 0.40, 0.60, 0.80 |
| Terminal structural | no | no | yes, diagnostic only |
| Dimensions structural | 8 | 16 | 8, 16, 32, 64 |
| Dimensions causal | 8 | 16 | 16 |
| Split/optimization/random seeds | 1 / 1 / 1 | 3 / 3 / 3 paired | 5 / 5 / 5 paired |
| Templates | fixed 5/group | all 40/group | all 40/group |
| Joint steps | 20 | 1,200 | 1,200 |
| Bootstrap | 100 diagnostic | 2,000 | 5,000 |

SMOKE emits self-alignment diagnostics only. CORE uses four model pairs:
Llama 1B-3B, Qwen 1.5B-3B, Llama 1B-Qwen 1.5B, and Llama 3B-Qwen 3B.

## 11. Compute and storage estimate

These are conservative planning estimates, not benchmarks. Forward calls are
short-context batch-one calls unless causal batching is validated. Runtime may
vary by at least 3x with GPU, host CPU, storage, and implementation overhead.

| Tier | Model loads | Extraction passes | Ridge fits | Joint fits | Causal forwards after reuse | Intertwiner comparisons | CKA / Procrustes comparisons | Cache, uncompressed | Rough GPU-hours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SMOKE | 1 | 4 | 2 | 2 | 242 | 2 self diagnostics | 2 / 2 self | about 0.6 MB | 0.05-0.25 |
| CORE | 4 | 32 | 48 | 144 | about 100,584 | 144 | 48 / 48 | about 44 MB | 4-15 |
| FULL | 6 | 72 | 1,200 | 6,000 | about 833,580 | 7,200 | 360 / 1,440 | about 231 MB | 40-120 |

FULL causal detail is about 1.11 million method rows. Allow 1-3 GB for CSV,
logs, checkpoints, and packaged outputs, in addition to model downloads and
the activation cache. The counts assume deterministic ridge/PCA reuse, base
forward reuse, dimension-16-only causal evaluation, the fixed pair matrix, and
one paired random baseline per optimization seed. Without these changes, FULL
is not approved.

Approximate inference-only peak GPU memory in float16 is 5-7 GB for 1B/1.5B,
8-12 GB for 3B, and 19-26 GB for 7B/8B. These ranges include model weights,
short-context activations, logits, allocator overhead, and modest safety
margin, but must be checked by CORE telemetry.

RunPod currently lists 24 GB RTX 4090/A5000/L4-class devices, 48 GB
A40/L40S/A6000-class devices, and 80 GB A100/H100-class devices in its
[GPU reference](https://docs.runpod.io/references/gpu-types). Recommended:

- CORE: one 24 GB RTX 4090 or A5000-class GPU; 48 GB is acceptable for more
  allocator headroom.
- FULL 7B/8B: one A100 80 GB as the conservative default; H100 80 GB for lower
  wall time. A 48 GB L40S/A6000 is memory-feasible but receives no runtime
  recommendation until CORE calibrates causal throughput.

## 12. Frozen output contract

The successful run root must contain manifest, environment, config snapshot,
progress/resume ledger, console log, detailed source artifacts, and checksums.
The final stable paper artifacts are:

```text
tables/
  structural_main.csv
  c7_causal_main.csv
  d4_causal_main.csv
  cross_model_main.csv
  latent_dim_sensitivity.csv
  model_scale_summary.csv

figures/
  structural_by_layer.svg
  structural_by_model.svg
  c7_causal_by_layer.svg
  c7_causal_by_step.svg
  d4_relation_intervention.svg
  irrep_spectra.svg
  intertwiner_vs_cka.svg
  scale_analysis.svg
```

Each figure also has a PDF copy generated from the same summary data. Detailed
machine-readable sources remain at:

```text
structural/template_metrics.csv
irrep/representation_diagnostics.csv
intertwiner/cross_model_metrics.csv
causal/intervention_metrics.csv
bootstrap/confidence_intervals.csv
```

No table may silently truncate rows. Every paper artifact records source run
ID and config hash in metadata or an adjacent manifest.

## 13. Go/no-go and claim rules

### 13.1 Execution no-go conditions

FULL must not start unless all of the following pass:

- six immutable model revisions and block counts match Section 2;
- exact 40-template pools and 28/6/6 splits pass leakage and tokenizer tests;
- PCA rank 64 succeeds on every train split;
- seed namespaces are separate and recorded;
- normalized depth slots match Section 3;
- structural random controls and cross-model controls are implemented;
- all causal metrics and D4 direct `srs`-versus-`r^-1` comparison exist;
- base-forward reuse, checkpointing, duplicate-key detection, and resume are
  tested;
- SMOKE and CORE complete with finite required metrics, complete manifests,
  no duplicate/missing keys, and valid result packages;
- CORE peak memory leaves at least 20% device headroom;
- projected FULL wall time and disk remain within the approved budget.

A null or adverse CORE scientific effect is reported but is not an engineering
reason to alter this frozen protocol. Any scientific redesign requires a new
explicit preregistration before inspecting FULL results.

### 13.2 Prespecified scientific support rules

Structural support, separately for C7 and D4, requires at dimension 16:

- aggregate joint-minus-random `E_eq < 0` with 95% CI upper bound below zero;
- aggregate joint-minus-random `E_rel_data < 0` with 95% CI upper bound below
  zero;
- joint-minus-identity `E_eq < 0` with 95% CI upper bound below zero; and
- negative point estimates at at least three of four primary depth slots for
  both `E_eq` and `E_rel_data`.

Latent-dimension robustness requires negative joint-minus-random point
estimates at at least three of four dimensions, including dimension 16. A
dimension with a CI wholly favoring random is reported as heterogeneity and
cannot be hidden by pooling.

C7 causal support requires transformation-only learned-minus-random and
learned-minus-identity target-logit-gain contrasts above zero with 95% CI lower
bounds above zero, plus positive point estimates at at least three of four
depth slots. `k=7` recovery is a separate secondary claim.

D4 causal relation support requires learned `srs` to be closer to direct
learned `r^-1` in final-logit space than its paired random control, with the
learned-minus-random distance CI wholly below zero. Transformation and
identity-recovery findings are reported separately.

Cross-model algebra-beyond-geometry support requires fitted-intertwiner
commutation error below both Procrustes-map and shuffled-label controls, with
95% contrast CIs wholly below zero in both within-family and cross-family
strata. The CKA/algebra Spearman point association must be positive to use the
phrase "related but not identical." Otherwise geometry and algebra are
reported separately without that relational claim.

Failure of a support rule is a scientific result, not a failed run. No model,
layer, dimension, seed, template, or group may be removed after results are
seen unless an independently documented integrity failure invalidates it.

## 14. Required implementation after approval

The following are approved-design changes, not changes made by this audit:

1. Pin model revisions and replace layer lists with Section 3 slots.
2. Add and test the exact templates and three-way split.
3. Add dimension 64 with explicit rank checks and dimension-16 causal scope.
4. Separate all seed namespaces and pair random seeds explicitly.
5. Deduplicate PCA/ridge work and cache base causal forwards.
6. Add structural random/shuffled controls and correct cross-model tests.
7. Add missing causal displacement, sequence, and D4 relation outputs.
8. Replace pooled bootstrap aggregation with the Section 9 clustered plan.
9. Add stable resume keys, per-split checkpoints, and final duplicate/missing
   key validation.
10. Implement the exact artifact contract and update protocol/paper mapping.
11. Update CPU/runtime tests and rebuild the single canonical RunPod package.

## 15. Final decision

Current canonical FULL: **NO-GO**.  
Frozen proposed design: **GO FOR IMPLEMENTATION AFTER USER APPROVAL**.  
GPU execution: **NOT AUTHORIZED**.

There are no unresolved scientific choices in this specification. Runtime is
the only material uncertainty; CORE telemetry must calibrate the broad timing
range without changing any scientific setting.
