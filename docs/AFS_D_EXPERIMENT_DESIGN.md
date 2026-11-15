> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# AFS-D Experiment Design: Embedding-to-Depth Algebra Control

**Status:** read-only design audit; not authorized or executed
**Audit date:** 2026-09-28
**Repository state inspected:** `9a491f89d5adf48c3119a89ec8ce1b215ec7802d`
**Immutable source runs:** `main_full_003` and `afs_full_abc_001`

## 1. Scientific motivation

The completed FULL experiment establishes that approximate finite-group operators can be recovered from internal transformer activations across six models, four normalized depths, five split seeds, and both C7 and D4. AFS-A/B/C further show generator-only transfer to never-fitted elements, failure under non-automorphic operator relabeling, and failure under scrambled training correspondences.

The main unresolved question is where this recoverable algebra originates. FULL starts at transformer-block outputs and excludes Hugging Face hidden state 0. AFS-D therefore asks whether the same operator algebra is already recoverable before the first transformer block or whether its fidelity changes through depth.

AFS-D is an origin/localization control. It is not a rerun of FULL, a new representation-learning method, or a search for a favorable layer.

## 2. Literature-driven alternative explanation

Recent work cited in the project memorandum shows that calendar-like geometry can arise from language co-occurrence statistics or symmetry in target distributions. This creates a concrete alternative explanation: the project may be recovering structure already encoded in lexical embeddings and merely preserving it through the network.

Circular visualization is not enough to resolve this alternative. The discriminating evidence is whether pre-transformer representations support the same fitted transformation operators, group-law consistency, defining relations, and generator-only composition tested at internal layers. D4 is especially informative because its noncommutative laws cannot be reduced to a single cyclic ordering.

## 3. Exact hypothesis

**Question:** Is approximate C7/D4 group-representation structure already recoverable from the pre-transformer representation, or does its algebraic fidelity change systematically across the frozen transformer depths?

For representation location `l` in `{embedding, 0.20, 0.40, 0.60, 0.80}`, fit the same training-only PCA and operators and evaluate the same held-out templates. For every error metric, lower is better.

The primary paired contrast is

```text
Delta_q(model, group, depth)
  = mean_test E_q(internal depth) - mean_test E_q(embedding).
```

Thus, a negative contrast means the internal representation has higher algebraic fidelity; a positive contrast means the embedding representation has higher fidelity.

The fixed-depth summary is the equal-weight mean of the four preregistered depth contrasts. Depths are fixed design points, not IID replicates.

## 4. Null and alternative interpretations

AFS-D has no favorable-result gate. It supports a directional interpretation, not a binary success claim.

- **Lexical-inheritance/null interpretation:** embedding and internal errors show no systematic directional difference under this design. Approximate algebra may already be inherited from lexical or language-statistical structure and preserved through the network.
- **Internal-strengthening interpretation:** internal errors are consistently lower than embedding errors. Transformer computation appears to strengthen or refine recoverable algebra beyond the lexical baseline.
- **Embedding-dominant interpretation:** embedding errors are consistently lower. Contextual processing weakens or transforms structure that is strongest at the lexical level.
- **Group-specific interpretation:** C7 is embedding-competitive while D4 strengthens internally. Simple cyclic structure may be more directly inherited from lexical statistics than noncommutative D4 structure.

Failure to reject a zero contrast is not formal proof of equivalence. The words "similar" or "no systematic difference detected" must be accompanied by the confidence intervals and this limitation.

## 5. Experimental units

The independent fitted cell is:

```text
model x group x representation location x split seed
```

Each cell has one deterministic PCA fit and one deterministic learned fit. Training uses 28 templates. Validation uses 6 templates for diagnostics only. The primary evaluation uses the 6 held-out test templates. Every template contains its complete group orbit and remains wholly within one split.

Data-supported metrics are evaluated at the test-template level. For `E_gen_eq`, errors across the preregistered unseen composite elements are averaged within template before inference. Operator-only `E_hom_op` is one value per fitted split cell; repeated copies in template-level tables must be collapsed before inference. Group elements, words, metric terms, and depths are not treated as independent replicates.

## 6. Models

### Options audited

| Option | Models | New embedding extraction sets | New embedding analysis cells | Scientific assessment |
|---|---|---:|---:|---|
| A | All six frozen FULL models | 12 | 60 | Strongest scale coverage, but repeats robustness already established internally and triples the new extraction work. |
| B | `meta-llama/Llama-3.2-1B` and `Qwen/Qwen2.5-1.5B` | 4 | 20 | Smallest panel preserving both model families, both groups, all depths, and all splits. These models were the predeclared PRE-FULL panel, so selection is not outcome-driven. |
| C | `meta-llama/Llama-3.2-3B` and `Qwen/Qwen2.5-3B` | 4 | 20 | Nominally size-matched cross-family pair, but more expensive and not more informative for the narrow embedding-origin question. |

### Recommendation

Use **Option B**. AFS-D is a targeted control, not a new scale-replication experiment. The completed six-model FULL and AFS-A results already establish internal-layer robustness. One predeclared representative from each family is sufficient to test whether an embedding-to-depth change reproduces across architectures. Claims must remain restricted to this fixed two-model panel.

Pinned revisions remain unchanged:

| Model | Revision |
|---|---|
| `meta-llama/Llama-3.2-1B` | `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` |

## 7. Groups

- **C7 weekdays:** canonical Abelian/calendar case.
- **D4 square corners:** noncommutative case, retaining `r^4=e`, `s^2=e`, `srs=r^-1`, and the distinction `rs != sr`.

Both groups are mandatory. C7 alone would not distinguish cyclic lexical geometry from richer operator algebra. D4 is the stronger control against that reduction, while C7 anchors the comparison to the literature-driven calendar concern.

## 8. Layers and representation convention

The representation locations are fixed before execution:

| Model | Embedding | Depth 0.20 | Depth 0.40 | Depth 0.60 | Depth 0.80 |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | hidden state 0 | block 3 | block 6 | block 10 | block 13 |
| Qwen2.5-1.5B | hidden state 0 | block 6 | block 11 | block 17 | block 22 |

The embedding representation is exactly Hugging Face `hidden_states[0]`, before transformer block 1, averaged over the same unique target-token span as FULL. It is labeled `representation_location=embedding`, `layer=0`, and `realized_depth=0.0`. It must not be duplicated over synthetic depth slots or described as a transformer-block output.

All four frozen internal depths are retained. Using only one early, middle, or late block would introduce an avoidable layer-selection decision; the four existing slots already provide early, two middle, and late observations without adding new locations. Terminal diagnostic blocks and causal intervention sites are excluded.

## 9. Existing frozen data reused

AFS-D reuses without modification:

- prompt construction version `final_large_experiment_v1`;
- the canonical 40 templates per group and complete group orbits;
- 28/6/6 train/validation/test split logic;
- split seeds `101, 303, 909, 2701, 4243`;
- primary latent dimension 16;
- training-only centered PCA and rank checks;
- ridge coefficient `0.01`;
- row-vector group-action conventions;
- target-span averaging and metric normalization;
- deterministic generator fitting and random-generator seed namespace;
- bootstrap seed `314159`, 5,000 replicates, and 95% percentile intervals.

Internal direct-ridge metrics are reused from immutable `main_full_003`. Internal generator-composition metrics are reused from immutable `afs_full_abc_001`. No internal activations, PCA bases, or operators are refit for the primary comparison.

The raw FULL archive contains 300 activation-cache files, all with `layer >= 3`; no cache has `layer=0`. The completed AFS manifest records 60 planned embedding cells and zero executed embedding units. Exact embedding representations therefore do not already exist and cannot be reconstructed from the stored latent PCA artifacts.

## 10. Metrics

No new headline metric is introduced. Independent ridge remains the direct-operator estimator, and generator-only ridge remains the generator-transfer estimator.

### Co-primary directional metrics

- `E_gen_eq`: held-out state-prediction error for generator-composed operators on never-directly-fitted elements.
- `E_rel_data`: held-out data-supported defining-relation defect.

### Supporting algebraic metrics

- `E_hom_data`: held-out data-supported homomorphism defect.
- `E_comp_data`: held-out data-supported composition defect.

### Secondary diagnostics

- `E_eq`: held-out direct equivariance error.
- `E_hom_op`: operator-space homomorphism defect, evaluated once per fitted split cell.

For D4, the existing data-supported relation components `r^4=e`, `s^2=e`, and `srs=r^-1` are reported separately beneath `E_rel_data`. The existing generator-only D4 unseen-element set and correct-word construction remain unchanged. Identity, matched-random-generator, direct-ridge, and matched wrong-word rows are retained as calibration controls for `E_gen_eq`, but they do not replace the embedding-to-depth contrast.

## 11. Statistical comparisons

All primary comparisons are paired on the same model, group, split seed, held-out template, and, where applicable, target element/word. Before inference, unseen-element rows are averaged within template and operator-only duplicates are collapsed.

For each model and group, report:

1. embedding and each depth's mean for all six metrics;
2. four depth-specific paired contrasts against embedding;
3. the equal-weight fixed-depth average contrast;
4. split-level estimates for all five split seeds;
5. the fraction of depth-specific point estimates favoring internal representations.

Inference is **within model**. The hierarchy is split seed -> test-template cluster, with paired resampling indices reused across embedding and every internal depth. Depth is a fixed repeated condition and is not resampled as an IID level. `E_hom_op` resamples split seeds only. Models are a fixed robustness panel and are not treated as population replicates.

An equal-weight two-model fixed-panel summary may be shown descriptively, but it cannot replace model-specific intervals or support a population claim about model families.

## 12. Confidence intervals and uncertainty

Use 5,000 two-sided 95% percentile bootstrap replicates with seed `314159`. Resample split seeds with replacement; within each sampled split, resample held-out template clusters with replacement. Use the same sampled split/template indices for every representation location in a paired replicate.

Report point estimates, interval bounds, and all five split-level effects. No p-value threshold, multiple-comparison search, or best-depth selection determines the conclusion. The co-primary interpretation uses only the preregistered fixed-depth average; depth-specific results describe trajectory and heterogeneity.

## 13. Preregistered outcome interpretation

Let the fixed-depth contrast be internal minus embedding, so negative is better internally.

Classification is performed separately for C7 and D4. A model satisfies an internal-strengthening pattern only when both co-primary contrast intervals have upper bounds below zero and both supporting algebraic point estimates are negative. It satisfies an embedding-dominant pattern only when both co-primary interval lower bounds exceed zero and both supporting point estimates are positive.

- **Outcome A: embedding substantially worse.** Both models satisfy the internal-strengthening pattern for the group. Interpret as evidence that transformer computation strengthens or refines recoverable algebra beyond this lexical baseline. Do not call it emergence, creation, or a causal mechanism.
- **Outcome B: embedding and internal representations show no systematic difference.** In both models, both co-primary intervals include zero, and there is no supporting metric with a concordant nonzero interval in the same direction in both models. Interpret as compatibility with strong lexical/language-statistical inheritance and preservation. This is not a formal equivalence claim.
- **Outcome C: embedding stronger.** Both models satisfy the embedding-dominant pattern for the group. Interpret as evidence that recoverable algebra is strongest pre-transformer and is weakened or transformed by contextual processing.
- **Outcome D: C7 embedding-competitive, D4 internally strengthened.** C7 meets Outcome B or C while D4 meets Outcome A. Interpret as evidence that cyclic lexical/statistical organization explains C7 more readily than noncommutative D4 structure.
- **Mixed outcome.** Any other combination, including model disagreement, metric disagreement, or depth reversals, is reported as heterogeneous without forcing A-D.

Magnitude tables and depth trajectories must be reported regardless of classification. No result changes the completed FULL or AFS-A/B/C gate decisions.

## 14. Stop rule

AFS-D is a one-shot control. Once this design and its implementation are approved and frozen:

- do not add models, seeds, dimensions, layers, templates, or metrics in response to results;
- do not redesign prompts or tune PCA, ridge, bootstrap, or operator hyperparameters;
- do not select a favorable depth or group as the headline;
- do not rerun FULL or AFS-A/B/C;
- do not add AFS-E behavioral composition to this experiment.

After the one authorized AFS-D run and prespecified audit, freeze the experimental program unless a documented implementation error invalidates the run.

## 15. Claims AFS-D can support

AFS-D can support bounded statements that, in the two fixed models and two tested groups:

- operator algebra is or is not recoverable before transformer block 1;
- algebraic fidelity strengthens, weakens, or remains unresolved across the four frozen internal depths;
- generator-only transfer changes relative to the embedding baseline;
- C7 and D4 show concordant or different embedding-to-depth patterns;
- the existing internal-layer result is compatible with refinement or inheritance under the preregistered interpretation.

## 16. Claims AFS-D cannot support

AFS-D cannot establish:

- that algebra "emerges" at a particular layer;
- a causal mechanism for transformer computation;
- that language statistics are the unique source of embedding structure;
- universality across architectures, model scales, groups, domains, or tokenizers;
- superiority of one model family;
- behavioral composition or causal steering;
- that PCA is uniquely necessary or that the learned representation is exact;
- any revision of the immutable FULL or AFS-A/B/C conclusions.

The scientifically valid conclusion is about recoverability and depth dependence in a fixed panel, not ontological origin or mechanistic proof.
