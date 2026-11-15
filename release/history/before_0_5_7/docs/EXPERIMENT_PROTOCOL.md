> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# Main Experiment Protocol

`docs/FINAL_EXPERIMENT_DESIGN.md` is the frozen pre-run specification. This
document describes its canonical implementation. Config and code must fail
rather than silently deviate.

## Scientific scope

The experiment tests held-out C7 and D4 equivariance and group laws,
representation-theoretic signatures, cross-model algebraic transport, and
activation-intervention effects. Interventions are causal pilots, not proof of
a complete causal mechanism.

## Models and immutable revisions

FULL contains exactly the three base Llama and three base Qwen models in
`configs/main_experiment.yaml`. Every model is pinned to the immutable Hugging
Face revision recorded in the frozen design. Runtime must match both revision
and block count.

Layer `l >= 1` is transformer-block output `l`; embedding state 0 is excluded.
Primary slots and realized layers are:

| Model | 0.20 | 0.40 | 0.60 | 0.80 | Terminal structural only |
|---|---:|---:|---:|---:|---:|
| Llama 1B | 3 | 6 | 10 | 13 | 16 |
| Llama 3B | 6 | 11 | 17 | 22 | 28 |
| Llama 8B | 6 | 13 | 19 | 26 | 32 |
| Qwen 1.5B | 6 | 11 | 17 | 22 | 28 |
| Qwen 3B | 7 | 14 | 22 | 29 | 36 |
| Qwen 7B | 6 | 11 | 17 | 22 | 28 |

Cross-model matching uses the declared slot, not equality of realized depth.
Terminal blocks are diagnostic structural endpoints and are never causal.

## Tasks and splits

C7 has 40 templates crossed with seven weekdays (280 rows). D4 has 40
templates crossed with four corners (160 rows). For each split seed, template
IDs are shuffled deterministically into 28 train, 6 validation, and 6 test.
This yields C7 196/42/42 and D4 112/24/24 rows. Every state expansion of a
template remains in one split.

PCA and operator fitting use train only. Validation is integrity/diagnostic
only. Final metrics and causal prompts use test only. No result can select a
layer, dimension, seed, metric, or hyperparameter.

The approved SMOKE clarification uses seven templates with a 5/1/1 split and
latent dimension 16. D4 then has centered training rank 19; rank is checked
before fitting.

## Dimensions and seeds

- Structural FULL primary: dimension 16 at all four depths, all five splits,
  and terminal structural diagnostics.
- Structural sensitivity: dimensions 8, 32, and 64 at depths 0.40/0.80 and
  split seeds 101/303 only.
- Cross-model FULL: dimension 16 only.
- Causal: dimension 16 only.
- Split seeds: 101, 303, 909, 2701, 4243.
- Deterministic learned-fit technical seed: 42; it is not a replication axis.
- Independent random-control seeds: 104729, 130363, 155921, 196613, 262147.
- Bootstrap seed: 314159.

Seed namespaces are disjoint. Historical CORE retains its original three
paired optimization/random labels. Exact learned duplicates from that axis are
collapsed before inference, while genuine random-control replications remain.

## Fitting and structural evaluation

PCA and independent ridge are computed once per model/group/layer/dimension/
split. One deterministic joint fit is computed per such cell. The optimizer
uses deterministic ridge initialization and full-batch updates with no random
initialization, dropout, minibatch sampling, or shuffled order. Joint fits use
AdamW learning rate 0.002, weight
decay 0.00001, 1,200 CORE/FULL steps, and frozen objective weights. Ridge alpha
is 0.01.

Primary test endpoints are `E_eq` and `E_rel_data`. Secondary endpoints are
homomorphism, composition, inverse, identity, generator-word, and operator-
space metrics. Methods are independent ridge, group-consistent joint, identity
where scientifically meaningful, and matched random orthogonal. Identity is
not a negative control for algebraic laws because it is a trivial exact
representation.

## Representation theory

C7 records the generator eigenspectrum, unit-circle and nearest-seventh-root
errors, conjugate/real-rotation-block diagnostics, `R^7`, and character
comparison. D4 records `r` and `s` spectra, separate `r^4`, `s^2`, and
`srs-r^-1` defects, approximate irrep multiplicities, block diagnostics, and
learned-versus-exact characters. Multiplicities are never silently rounded.

## Cross-model analysis

SMOKE runs a Llama-1B self-alignment diagnostic to exercise the complete
cross-model implementation path. This row is labeled
`self_smoke_diagnostic`, is runtime validation only, and is not part of the
nine frozen scientific comparisons used by CORE and FULL.

The only primary pairs are the six within-family scale pairs and three
matched-scale cross-family pairs explicitly listed in config. At matched group,
depth slot, split, and primary dimension 16, report CKA, latent
Procrustes reconstruction, fitted-intertwiner commutation, Procrustes-map
commutation, random-map commutation, and shuffled-label commutation. The
primary algebra-beyond-geometry contrasts compare fitted-intertwiner
commutation with Procrustes-map and shuffled-label commutation.

## Activation intervention

The row-vector intervention is fixed:

`h_intervened = h - z C + (z R^k) C`.

C7 `k=1,...,6` are transformations; `k=7` is cycle recovery and is never
pooled with transformation success. D4 evaluates `r`, `s`, `r^2`, `r^3`,
`r^4`, `s^2`, `srs`, and a direct `r^-1` behavioral reference. Correct
row-vector order for the relation is `z @ R(s) @ R(r) @ R(s)`.

Methods are learned, identity, five independently seeded random orthogonal
controls, and the train-only source-conditioned additive comparator
`v_(g,s)=mean_(t in train)[h(gs,t)-h(s,t)]`. The vector is fit separately by
element and source state, then applied only to held-out source activations. It
is a stronger comparator because it may depend on source state; the learned
operator represents one shared transformation law. The balanced-orbit global
mean difference cancels and remains historical CORE-only. Outputs include
top-1, target logit/probability gain, target-source
margin, patched/downstream displacement, final-logit displacement, single-
token sequence log probability where defined, cycle recovery, and direct D4
`srs`-versus-`r^-1` behavior distances. Revised tiers also apply the
matched-length word `rsr`, whose exact D4 product is `s`, and compare its
downstream output with direct `r^-1`. The paired true-minus-wrong distance is
negative when the true relation is more consistent.

## Statistics

FULL uses 5,000 and PRE-FULL/CORE use 2,000 deterministic percentile bootstrap
replicates. For within-model claims the hierarchy is depth -> split ->
template. For a declared aggregate cross-model estimand, model is an outer
level. Models are not used as replicates for model-specific claims. Templates
carry all group-expanded states. Random-control seed is inserted between split
and template only for random-control contrasts because one sampled random
operator is shared across its template rows. Pairing occurs at the lowest
shared valid cell. Exact historical learned duplicates are collapsed and
receive no extra weight. Reports include joint-minus-independent-ridge paired
intervals; an interval containing zero is "no detected difference," not
equivalence.

## Tiers and gates

- SMOKE: Llama 1B, layer 6, seven templates, 5/1/1 split, dimension 16, one
  seed tuple, 20 joint steps, every major module; runtime evidence only.
- CORE: Llama 1B/3B and Qwen 1.5B/3B, slots 0.40/0.80, all templates,
  dimension 16, three historical seed tuples, 1,200 steps, 2,000 bootstraps.
- PRE-FULL: Llama 1B and Qwen 1.5B, slots 0.40/0.80, splits 101/303,
  dimension 16, one deterministic fit, two random controls, 2,000 bootstraps.
- FULL: all six, dimension-16 confirmatory coverage plus the reduced
  sensitivity grid above, one deterministic fit per cell, five random
  controls, and 5,000 bootstraps.

FULL requires a clean source tree and an explicitly reviewed successful
PRE-FULL manifest, `go_no_go.json`, and `FULL_COST_PROJECTION.md`. No tier
automatically starts the next tier.

## Cache, resume, and failure safety

Raw activation keys include immutable model/tokenizer revisions, tokenizer
settings, group/task, template/split identity, layer, prompt version, and
extraction convention. Embedded metadata is checked before reuse. Latent
dimension is excluded because PCA is downstream.

Every scientific unit has a deterministic key in `run_ledger.json`. Completion
is atomic and includes output checksums. Verified units are skipped on resume;
missing/corrupt outputs become stale and are recomputed. Failures record key,
type, message, and timestamp. Other independent units may continue, but final
success is impossible while any required unit is incomplete.

## Outputs

Each run contains manifest, environment, config snapshot, ledger, telemetry,
cost projection, gate summary, detailed source CSVs, stable tables under
`results/tables/`, and SVG/PDF figures under `results/figures/`. Checkpoints and
activation caches are excluded from downloadable result bundles. Raw outputs
are never committed automatically.
