> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# PRE-FULL Source-Conditioned Additive Baseline Audit

Audit date: 2026-09-04

Scope: completed PRE-FULL run `pre_full_001` only. The immutable input snapshot
is `runpod_output_by_date/result_9_4_2026`. No CORE result was used as a
substitute, no GPU inference was run, and no result artifact or scientific
configuration was modified.

## Decision

The source-conditioned additive baseline is **PASS WITH REPORTING
LIMITATION**. Its implementation is valid, train-only, nondegenerate for every
non-identity transformation, and statistically paired correctly against the
deterministic learned operator. It is a strong direct-steering comparator.

The reporting limitation is important: the additive rows for multi-step and
relation labels are direct vectors for the final target element. They do not
apply a reusable additive law repeatedly. Identity and relation recoveries for
this method are therefore tautological direct-target controls, not evidence of
algebraic composition. This limitation does not block FULL because the learned
operator is separately evaluated by genuine repeated C7 powers and D4 word
order.

## Integrity And Provenance

| Check | Result |
|---|---:|
| Final result ZIP SHA256 | `ca363823e6ee4d6fb973bb3fa5533c3d2c5e7696f33e2036c0ec59ead3b303dd` |
| Raw `run_outputs.zip` SHA256 | `df2e35aacc98acb3f1cfc18b426af904944cfdbea58199d88d5bf3ebbf1d27ec` |
| Run ID / tier / status | `pre_full_001` / `pre_full` / `SUCCESS` |
| Source commit | `90cce9681d8a29758fc248073e5969b97cfcdd73` |
| Source tree | clean |
| Learned-fit semantics | `deterministic_single` |
| Manifest artifact hashes | 47 checked, 0 mismatches |
| Run ledger | 70/70 required units complete, 0 failed |
| Causal output | 20,400 rows |
| Execution completeness gate | PASS |

The run's overall `CONDITIONAL_GO` is expected for PRE-FULL: only two depth
slots were evaluated, so the three-depth structural, causal, and cross-model
FULL gates are `NOT_EVALUABLE`, not failed. The D4 relation gate passed.

## Definition And Reconstruction

The implementation in `src/bsg/interventions/causal.py` computes, for every
tested element and source state,

```text
v_(g,s) = mean over t in T_train of [h(g s,t) - h(s,t)]
h'(s,t_test) = h(s,t_test) + v_(g,s).
```

`src/bsg/experiments/main_experiment.py` passes the full structural activation
frame, the split's training mask, the tested target elements, and the exact
C7/D4 `transform_state` function to that fit. At held-out evaluation it looks
up the vector by `(target_element, source_id)` and adds it to the held-out
source activation. The vector is in full hidden space; it is not restricted to
the 16-dimensional PCA subspace.

The raw archive contained 8 full structural activation caches and 16
split-specific causal caches. Reconstructing all vectors from those caches and
the recorded split identities gave:

- 16 causal cells: 2 models x 2 groups x 2 depths x 2 splits;
- exactly 28 training, 6 validation, and 6 test templates in every cell;
- zero train/validation, train/test, or validation/test overlaps;
- zero held-out template mismatches in the output rows;
- zero target-state mismatches across all 20,400 output rows;
- maximum error between reconstructed vector displacement and recorded
  `patched_hidden_displacement`: `2.22e-16`.

The C7 and D4 source and target indices therefore follow the exact committed
group actions. For the deliberate D4 `rsr_wrong_order` control, the word
product is recorded as `s`, while the requested target remains `r^-1`; this is
the preregistered comparison against the intended relation target.

## Non-Degeneracy

The unit below is one fitted vector per model, group, depth, split, unique
target element, and source state. Near zero means nonzero norm at most `1e-8`.

| Population | N | Min | Median | Mean | Max | Exactly zero | Near-zero nonzero |
|---|---:|---:|---:|---:|---:|---:|---:|
| All fitted vectors | 552 | 0 | 5.3429 | 18.6607 | 70.0895 | 15.9420% | 0% |
| Non-identity only | 464 | 1.3284 | 14.5063 | 22.1998 | 70.0895 | 0% | 0% |

All 88 exact zeroes are required identity vectors: 56 C7 `r^0` vectors and 32
D4 `e` vectors. No unexpected zero or near-zero vector exists.

### Non-Identity Norms By Primary Axis

| Axis level | N | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 232 | 1.3284 | 2.9093 | 3.7494 | 8.6333 |
| Qwen2.5-1.5B | 232 | 20.3792 | 38.6951 | 40.6502 | 70.0895 |
| C7 | 336 | 1.3284 | 13.0078 | 21.3967 | 58.9903 |
| D4 | 128 | 1.8119 | 17.4018 | 24.3081 | 70.0895 |
| Depth 0.4 | 232 | 1.3284 | 11.5233 | 15.2084 | 35.0991 |
| Depth 0.8 | 232 | 3.1512 | 25.4622 | 29.1912 | 70.0895 |
| Split 101 | 232 | 1.3502 | 14.5970 | 22.3498 | 70.0895 |
| Split 303 | 232 | 1.3284 | 14.4847 | 22.0498 | 69.5169 |

Absolute hidden-state norms are model-scale dependent; the large Qwen/Llama
difference is not interpreted as a cross-model effect. The exact displacement
reconstruction above confirms that both scales were applied as intended.

### Norms By Transformation

| Group / element | N | Min | Median | Mean | Max | Zero fraction |
|---|---:|---:|---:|---:|---:|---:|
| C7 `r^0` | 56 | 0 | 0 | 0 | 0 | 100% |
| C7 `r^1` | 56 | 1.3284 | 12.6408 | 19.0777 | 50.9650 | 0% |
| C7 `r^2` | 56 | 1.6975 | 14.5186 | 21.8484 | 56.5013 | 0% |
| C7 `r^3` | 56 | 2.1820 | 16.7461 | 23.2639 | 58.9903 | 0% |
| C7 `r^4` | 56 | 2.1820 | 16.7461 | 23.2639 | 58.9903 | 0% |
| C7 `r^5` | 56 | 1.6975 | 14.5186 | 21.8484 | 56.5013 | 0% |
| C7 `r^6` | 56 | 1.3284 | 12.6408 | 19.0777 | 50.9650 | 0% |
| D4 `e` | 32 | 0 | 0 | 0 | 0 | 100% |
| D4 `r` | 32 | 1.8119 | 16.9288 | 22.7858 | 56.2507 | 0% |
| D4 `r^2` | 32 | 2.2591 | 21.0634 | 28.2696 | 70.0895 | 0% |
| D4 `r^3` | 32 | 1.8119 | 16.9288 | 22.7858 | 56.2507 | 0% |
| D4 `s` | 32 | 1.9118 | 17.9257 | 23.3912 | 56.2507 | 0% |

### Non-Identity Norms By Source State

| Group / source | N | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|---:|
| C7 Monday | 48 | 1.5881 | 13.9322 | 21.3433 | 54.7214 |
| C7 Tuesday | 48 | 1.3284 | 12.8683 | 20.6180 | 56.5013 |
| C7 Wednesday | 48 | 1.3284 | 12.7619 | 20.4031 | 55.9752 |
| C7 Thursday | 48 | 1.4007 | 12.9071 | 20.7542 | 58.9903 |
| C7 Friday | 48 | 1.5367 | 13.9124 | 21.5273 | 56.1639 |
| C7 Saturday | 48 | 1.8118 | 15.8274 | 21.9219 | 53.6656 |
| C7 Sunday | 48 | 1.8784 | 15.8263 | 23.2089 | 58.9903 |
| D4 NW | 32 | 1.9459 | 18.3986 | 24.2217 | 67.6431 |
| D4 NE | 32 | 1.8119 | 17.0727 | 24.3506 | 70.0895 |
| D4 SE | 32 | 1.8119 | 17.4018 | 24.2567 | 67.6431 |
| D4 SW | 32 | 1.9118 | 18.2141 | 24.4033 | 70.0895 |

For every non-identity transformation cell, source-conditioned vectors differ:
the minimum pairwise source-vector distance is `1.7231`, and the maximum is
`140.1791`. Conversely, averaging these source-specific vectors over a complete
orbit produces the old cancellation (maximum orbit-average norm
`1.81e-15`). The implementation does not use that orbit average. This directly
confirms that source conditioning eliminates the previous degeneracy.

## Held-Out Single-Step Comparison

The following rows use only evaluation role `transformation`: C7 `R^1` through
`R^6`, and D4 `r`, `s`, `r2`, and `r3`. `L-A` means learned operator minus
source-conditioned additive. Positive favors learned only for target logit,
probability, margin, top-1, and sequence log probability. Displacement
differences are descriptive and have no preregistered preferred direction.

| Group | Metric | Learned | Additive | L-A |
|---|---:|---:|---:|---:|
| C7 | target logit change | 0.731186 | 0.795170 | -0.063984 |
| C7 | target probability change | 0.006783 | 0.007069 | -0.000286 |
| C7 | target-vs-original margin | 0.589693 | 0.692247 | -0.102555 |
| C7 | top-1 success | 0.000000 | 0.000000 | 0.000000 |
| C7 | downstream hidden displacement | 0.052783 | 0.054926 | -0.002142 |
| C7 | final-logit displacement | 0.049265 | 0.050323 | -0.001058 |
| C7 | target sequence log probability | -4.780010 | -4.715110 | -0.064902 |
| D4 | target logit change | 2.427280 | 2.409110 | 0.018168 |
| D4 | target probability change | 0.020959 | 0.020526 | 0.000433 |
| D4 | target-vs-original margin | 2.151370 | 2.134640 | 0.016724 |
| D4 | top-1 success | 0.071615 | 0.063802 | 0.007812 |
| D4 | downstream hidden displacement | 0.077303 | 0.077779 | -0.000475 |
| D4 | final-logit displacement | 0.065093 | 0.065601 | -0.000508 |
| D4 | target sequence log probability | -4.095730 | -4.112030 | 0.016295 |

C7 has a top-1 floor: neither method produced a top-1 target on these direct
transformations. Both methods nevertheless increased mean target logits,
probabilities, and margins. D4 produced nonzero top-1 success for both.

### Model-Specific L-A Differences

| Group / model | Logit | Probability | Margin | Top-1 | Downstream | Final logit | Sequence log P |
|---|---:|---:|---:|---:|---:|---:|---:|
| C7 Qwen | 0.006518 | 0.000124 | 0.006534 | 0 | -0.000283 | -0.000359 | 0.005260 |
| C7 Llama | -0.134487 | -0.000696 | -0.211643 | 0 | -0.004002 | -0.001757 | -0.135063 |
| D4 Qwen | 0.048401 | 0.000896 | 0.047099 | 0.013021 | -0.000840 | -0.000992 | 0.044762 |
| D4 Llama | -0.012065 | -0.000030 | -0.013652 | 0.002604 | -0.000110 | -0.000024 | -0.012173 |

### Depth-Specific L-A Differences

| Group / depth | Logit | Probability | Margin | Top-1 | Downstream | Final logit | Sequence log P |
|---|---:|---:|---:|---:|---:|---:|---:|
| C7 0.4 | -0.118459 | -0.000740 | -0.187360 | 0 | -0.003236 | -0.001580 | -0.119341 |
| C7 0.8 | -0.009510 | 0.000168 | -0.017749 | 0 | -0.001049 | -0.000536 | -0.010462 |
| D4 0.4 | 0.015544 | 0.000423 | -0.001038 | 0.002604 | -0.000324 | -0.000143 | 0.013045 |
| D4 0.8 | 0.020793 | 0.000443 | 0.034485 | 0.013021 | -0.000626 | -0.000873 | 0.019544 |

### Transformation-Specific L-A Differences

| Transformation | Logit | Probability | Margin | Top-1 | Downstream | Final logit | Sequence log P |
|---|---:|---:|---:|---:|---:|---:|---:|
| C7 `R^1` | -0.058106 | -0.000306 | -0.123442 | 0 | -0.003180 | -0.001556 | -0.059803 |
| C7 `R^2` | -0.055943 | -0.000215 | -0.075149 | 0 | -0.001277 | -0.000656 | -0.056443 |
| C7 `R^3` | -0.070568 | -0.000307 | -0.107585 | 0 | -0.001883 | -0.000919 | -0.071299 |
| C7 `R^4` | -0.063081 | -0.000286 | -0.101423 | 0 | -0.001854 | -0.000940 | -0.064669 |
| C7 `R^5` | -0.065034 | -0.000224 | -0.080659 | 0 | -0.001334 | -0.000708 | -0.066217 |
| C7 `R^6` | -0.071173 | -0.000376 | -0.127069 | 0 | -0.003324 | -0.001569 | -0.070978 |
| D4 `r` | 0.006836 | 0.000325 | 0.007568 | 0.005208 | -0.000654 | -0.000691 | 0.005036 |
| D4 `r2` | 0.026571 | 0.000596 | 0.024292 | 0.005208 | -0.000535 | -0.000673 | 0.024408 |
| D4 `r3` | 0.022095 | 0.000451 | 0.010905 | 0.010417 | -0.000481 | -0.000421 | 0.020491 |
| D4 `s` | 0.017171 | 0.000359 | 0.024129 | 0.010417 | -0.000230 | -0.000247 | 0.015245 |

### Preregistered Paired Inference

The stable bootstrap output provides learned-minus-additive 95% hierarchical
intervals for target logit, target probability, top-1, and margin by model and
for the declared aggregate. The aggregate hierarchy is model -> depth -> split
-> template; within-model inference is depth -> split -> template.

| Group / model | Metric | Difference | 95% CI |
|---|---|---:|---:|
| C7 ALL | target logit | -0.063984 | [-0.196327, 0.014387] |
| C7 ALL | target probability | -0.000286 | [-0.001224, 0.000445] |
| C7 ALL | margin | -0.102555 | [-0.305808, 0.036880] |
| C7 Llama | target logit | -0.134487 | [-0.268061, -0.005970] |
| C7 Llama | margin | -0.211643 | [-0.397368, -0.006740] |
| C7 Qwen | target logit | 0.006518 | [-0.009814, 0.026214] |
| C7 Qwen | margin | 0.006534 | [-0.046976, 0.065421] |
| D4 ALL | target logit | 0.018168 | [-0.043247, 0.073906] |
| D4 ALL | target probability | 0.000433 | [-0.000312, 0.001421] |
| D4 ALL | margin | 0.016724 | [-0.085726, 0.120957] |
| D4 ALL | top-1 | 0.007812 | [0.000000, 0.022135] |

The C7 Llama result favors additive steering for target logit and margin. The
aggregate C7 intervals and Qwen intervals do not establish a difference. D4 is
similarly competitive, with no clear aggregate separation. This is acceptable:
the protocol did not require learned operators to dominate every direct
single-step metric. A confidence interval containing zero is not evidence of
equivalence.

Formal artifact CIs were not generated for downstream displacement,
final-logit displacement, sequence log probability, depth-specific contrasts,
or individual transformations. Those results above are descriptive paired
differences and are not silently presented as inferential findings.

## Composition And Reusable Transformation Laws

### C7

For `R^1` through `R^6`, the learned method applies repeated powers of one
learned generator. The additive method instead looks up a separately fitted
direct vector for each final element `r^k` and source state. Its apparent
multi-step behavior therefore tests direct target steering, not repeated
application of a shared additive transformation.

At `k=7`, the learned result is a genuine repeated-operator cycle. The additive
method selects `v_(e,s)=0`, making hidden recovery true by construction.

| Method | Hidden cycle error | Latent cycle error | Probability recovery error |
|---|---:|---:|---:|
| learned operator | 0.020728 | 0.052993 | 0.000247 |
| random orthogonal | 0.557252 | 1.412970 | 0.009786 |
| source-conditioned additive | 0.000000 | 0.000000 | 0.000095 |
| identity | 0.000000 | 0.000000 | 0.000000 |

The tiny nonzero additive probability error comes from executing a second
patched forward pass with an unchanged hidden vector; it is not a nonzero
fitted identity vector. No formal learned-versus-additive CI was emitted for
the cycle-error metrics.

### D4

The learned operator evaluates actual matrix words. Its relation behavior is:

| Test / method | Hidden error or distance | Latent error | Probability error or difference |
|---|---:|---:|---:|
| `r^4=e`, learned | 0.042864 | 0.099680 | 0.001335 |
| `r^4=e`, random | 0.584097 | 1.367950 | 0.018360 |
| `s^2=e`, learned | 0.031729 | 0.073730 | 0.000955 |
| `s^2=e`, random | 0.584748 | 1.371650 | 0.018213 |
| `srs` vs direct `r^-1`, learned | 0.002038 | n/a | 0.000507 |
| `srs` vs direct `r^-1`, random | 0.059678 | n/a | 0.005868 |
| wrong `rsr` vs intended `r^-1`, learned | 0.044350 | n/a | 0.010914 |
| wrong `rsr` vs intended `r^-1`, random | 0.062585 | n/a | 0.005968 |

For the learned method, true `srs` is substantially closer to the direct
inverse than wrong-order `rsr` on every registered relation measure:

| Measure, true minus wrong | Difference | 95% CI |
|---|---:|---:|
| final-logit distance | -0.035737 | [-0.050283, -0.022357] |
| downstream-hidden distance | -0.042312 | [-0.052195, -0.033095] |
| target-probability difference | -0.010407 | [-0.015728, -0.005709] |
| logit-margin difference | -1.354530 | [-1.714110, -0.979518] |
| sequence-log-probability difference | -1.462130 | [-1.776630, -1.124940] |

The additive comparator does not execute `r^4`, `s^2`, `srs`, or `rsr` as
composed operations. It receives `e` for the first two and `r^-1` as the
intended target for the latter three. Consequently, its identity hidden errors
and its true/wrong/direct relation distances are exactly zero by construction.
These zeroes cannot be interpreted as reproducing D4 composition or order
sensitivity. No unmeasured additive composition experiment is invented here.

The evidence must therefore remain separated:

1. **Single-step steering:** additive is competitive and sometimes stronger.
2. **Reusable transformation law:** only the learned operator is evaluated by
   repeated use of shared operators; additive uses `(g,s)`-specific vectors.
3. **Algebraic composition:** learned operators show strong C7 cycle and D4
   relation/order behavior relative to random controls. The present additive
   implementation does not furnish a compositional comparator.

## Leakage And Fairness

| Method | Training information | Held-out intervention information |
|---|---|---|
| Learned operator | training activations, train-fitted PCA encoder/decoder, one shared latent operator per group element | held-out hidden activation and desired element `g`; source state is represented through the activation, not used as a vector lookup key |
| Source-conditioned additive | training activations and one full-hidden-space vector for every `(g,s)` | held-out hidden activation, desired element `g`, and explicit source state `s` |
| Random orthogonal | latent dimension and independent declared random seed | held-out hidden activation and requested operator word |
| Identity | none | unchanged held-out hidden activation |

The additive baseline has more source-specific information and more parameters
than the learned shared law. This makes it a deliberately strong comparator
for direct behavior, not an element-only baseline. The comparison is fair if
reported as direct steering versus a stronger source-conditioned control; it
would be unfair to treat its direct target lookup as evidence of reusable
composition. No validation or test template enters its fit, and no train/test
leakage was found.

## Statistical Audit

- Learned rows: 4,080; additive rows: 4,080.
- Both use only technical optimization seed 42.
- Both have zero non-null random-control seed labels.
- Neither method has a duplicate scientific pairing key.
- Their pairing-key symmetric difference is zero.
- All 90 formal learned-versus-additive rows report zero random-seed levels.
- Formal cluster columns are exactly
  `model,depth_slot,split_seed,template_id` for aggregate claims, with the
  model level absent in effect for within-model subsets.
- Paired comparisons retain group, depth, layer, split, template,
  intervention, role, source, and target context before aggregation.
- Random-control replications remain separate elsewhere and are not injected
  into learned-versus-additive inference.

This supports valid split/template uncertainty without optimization-seed
pseudo-replication. The reporting limitation is inferential coverage, not an
invalid calculation: stable formal CIs cover four direct behavioral metrics
and the registered relation distance, but not all requested displacement and
sequence metrics or every depth/transformation breakdown.

## Computational Audit

All 70 units completed. Summed instrumented completed-stage time was 1,511.28
seconds (25.19 minutes): deterministic joint fitting used 974.48 seconds,
causal evaluation 301.18 seconds (16 units, mean 18.82, maximum 27.38), and
2,000-replicate bootstrap statistics 128.52 seconds. The PRE-FULL-derived FULL
projection is finite at 37,471.07 seconds (10.41 serial-equivalent hours).
There was no failed or invisible long-running stage. Runtime does not isolate
the additive comparator from the other methods inside each causal unit, so no
method-specific speed claim is made.

## Recommended Interpretation

PRE-FULL establishes that the corrected additive baseline is executable,
train-only, nondegenerate, and strong enough to challenge direct learned
steering. It does not establish learned superiority on every direct metric.
The learned operator's differentiated evidence is that one shared operator can
be reused in powers and noncommutative words, with substantially better cycle
and D4 relation/order behavior than random operators. This remains an
activation-intervention result, not proof of a complete causal mechanism.

PRE-FULL ADDITIVE VERDICT:

Implementation validity: PASS; exact `(g,s)` lookup and held-out addition were reconstructed from raw caches, with maximum displacement error `2.22e-16`.
Training-only verification: PASS; every cell used 28 training templates, all 6 validation and 6 test templates were excluded, and no split overlap or held-out mismatch was found.
Non-degeneracy: PASS; all 464 non-identity vectors were nonzero, minimum norm `1.3284`, and no nonzero norm was at most `1e-8`.
Single-step learned vs additive: Additive is competitive; it is stronger for C7 Llama target-logit and margin intervals, while aggregate C7 and D4 comparisons do not establish general dominance.
Composition comparison: Learned operators support genuine repeated C7 and D4 word tests; additive composition is NOT EVALUATED because current additive rows are direct final-target lookups, and their identity/relation zeroes are tautological.
Statistical validity: PASS WITH REPORTING LIMITATION; pairing and hierarchy are valid with no pseudo-replication, but formal CIs are unavailable for several displacement/sequence and depth/transformation-specific contrasts.
Scientific interpretation: The additive method is a stronger source-aware direct-steering comparator; the learned method's distinct claim is reusable shared transformation-law and algebraic-order behavior.
Blocks FULL?: NO.
Recommended paper wording: "On held-out templates, source-conditioned additive steering was competitive with learned operators for direct steering. It used train-only vectors conditioned on transformation and source state and was not evaluated as a reusable compositional law. Learned operators were separately evaluated under repeated C7 powers and D4 relation/order tests."
Final decision: AUTHORIZE FULL
