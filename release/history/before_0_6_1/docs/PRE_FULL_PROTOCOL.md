> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# PRE-FULL Calibration Protocol

Status: completed and audited as `pre_full_001`
Approval date: 2026-09-03

METHOD STATUS: FROZEN FOR FULL
FULL STATUS: AUTHORIZED

PRE-FULL is a small prospective GPU calibration for the implementation changes
made after the completed CORE audit. It is not confirmatory evidence and does
not replace FULL. No PRE-FULL or FULL result was inspected while defining this
protocol.

## Frozen matrix

| Axis | PRE-FULL values |
|---|---|
| Models | `meta-llama/Llama-3.2-1B`, `Qwen/Qwen2.5-1.5B` |
| Groups | C7, D4 |
| Depth slots | 0.40, 0.80 |
| Split seeds | 101, 303 |
| Latent dimensions | primary 16 only |
| Learned fits | one deterministic fit per independent cell, technical seed 42 |
| Random controls | seeds 104729, 130363 |
| Bootstrap | 2,000 replicates, seed 314159 |
| Joint optimizer | frozen 1,200 steps and existing hyperparameters |

Expected scientific units are 2 model loads, 4 structural extraction units,
8 causal extraction units, 16 independent-ridge/base fits, 16 deterministic
joint fits, 16 causal units, and 8 cross-model units. Each structural/joint
unit is model x group x depth x split at dimension 16. Each causal unit uses
the same axes. Cross-model uses the one available fixed Llama-1B/Qwen-1.5B
pair, both groups, both depths, and both splits.

The complete-data post-processing contract expects 398 bootstrap analysis
rows, 9 stable tables, and 9 figures, each in SVG/PDF form with a sibling
source CSV.

Historical clean CORE telemetry projected the frozen FULL grid at 113.66
serial A40 hours. Applied to the revised unit matrix with explicit causal and
cross-model control-workload multipliers, it projects 42,513 seconds (11.81
hours) for scientific stages, before roughly 0.2-1.0 hour of post-processing.
This 12.0-12.8 hour planning range is preliminary; telemetry from this
PRE-FULL run is the authoritative calibration.

## Required checks

PRE-FULL must verify all of the following before FULL authorization:

1. Learned output has one deterministic fit per cell and no optimization-seed
   pseudo-replication.
2. Random-control seeds remain independent replications and are not collapsed.
3. The hierarchical bootstrap records the intended cluster hierarchy and
   emits model-specific and explicitly aggregated cross-model estimands.
4. `joint_vs_ridge_inference.csv` contains paired E_eq, E_rel_data,
   E_hom_data, E_comp_data, and E_inverse_data estimates and intervals.
5. `source_conditioned_additive` is fit from training templates only and is
   evaluated on held-out templates.
6. D4 records both `srs_r_inverse` and `rsr_wrong_order`; `rsr` has exact
   group product `s`, and true-minus-wrong-order distance is paired.
7. C7 transformation figures include only k=1,...,6; k=7 has a separate cycle
   recovery figure.
8. Gate states distinguish PASS, FAIL, and NOT_EVALUABLE. A CKA association
   interval containing zero is reported as no reliable detected association.
9. Every stable table has `source_csv`, and every figure has a sibling
   `.source.csv` file.
10. Runtime telemetry and the revised FULL cost projection are finite.

## Inference semantics

Within one model, the hierarchy is depth -> split -> template. For the
predeclared aggregate across the two models, model is an outer level. Models
are not interchangeable IID replicates for model-specific claims. Templates
remain clusters containing all state/intervention rows. Random-control seed is
an additional level only for random-control contrasts. Learned-fit uncertainty
is not manufactured from repeated deterministic optimization labels.

Joint-minus-ridge reports a paired difference. A 95% interval containing zero
means no detected difference, not equivalence. No equivalence margin is
preregistered here.

## Activation controls

The additive comparator is

`v_(g,s) = mean_(t in T_train)[h(g s,t) - h(s,t)]`.

It is fit separately for every tested element and source state, then applied as
`h'(s,t_test) = h(s,t_test) + v_(g,s)`. This stronger comparator can depend on
source state; the learned group operator is shared across source states.
Identity/no intervention and random orthogonal controls remain separate.

The D4 wrong-order word is `rsr`. Under the exact action table its product is
`s`, whereas `srs=r^-1`. Both composed outputs are compared against the direct
`r^-1` intervention for the order-sensitive contrast.

## Commands

After deployment and preflight, the intended command is:

```bash
BSG_RUN_ID=main_pre_full_calibration bash runpod/run_main_experiment.sh --pre-full
```

That PRE-FULL run is complete. Its manifest, gates, source tables, cost
projection, and additive baseline were reviewed in
`docs/PRE_FULL_ADDITIVE_AUDIT.md`, which records `AUTHORIZE FULL`. The command
above is retained only as the historical reproduction command and must not be
rerun for FULL deployment.
