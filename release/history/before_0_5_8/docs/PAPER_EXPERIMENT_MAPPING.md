## Current revision — 24 October 2026

Version 0.4.0 integrates the completed returned replay as reviewed descriptive evidence. Compact/context and secondary extension results are now available. Native competence still failed, structural power remains limited, and independent confirmation is blocked. Original frozen results and source capsules retain their identities.

For current instructions use the root README, `docs/REPRODUCTION_REVIEW_20261024.txt`, and `release/reproduction_review.json`. Rebuild tables offline with `python scripts/build_reproduction_assets.py`; no new calibration or full model replay is needed. Maintenance v2 defaults to checking only and requires a separate explicit choice before any execution. Prior dated instructions below describe earlier project states and should not be used to launch another run automatically.

## Preserved earlier documentation

> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# Paper Experiment Mapping

The canonical source is `run_outputs/main_experiment/<full_run_id>/`. Smoke and
CORE outputs are runtime/preliminary evidence and cannot replace FULL.
PRE-FULL run `pre_full_001` authorized the frozen FULL protocol but remains
calibration evidence and likewise cannot replace FULL paper results.

| Paper item | Stable generated artifact | Required tier |
|---|---|---|
| Primary structural table | `results/tables/structural_main.csv` | FULL |
| C7 causal table | `results/tables/c7_causal_main.csv` | FULL |
| D4 causal/relation table | `results/tables/d4_causal_main.csv` | FULL |
| Cross-model table | `results/tables/cross_model_main.csv` | FULL |
| Dimension sensitivity | `results/tables/latent_dim_sensitivity.csv` | FULL |
| Scale summary | `results/tables/model_scale_summary.csv` | FULL |
| Structural figures | `results/figures/structural_by_layer.*`, `structural_by_model.*` | FULL |
| Joint versus ridge inference | `results/tables/joint_vs_ridge_inference.csv` | FULL |
| Additive comparator inference | `results/tables/causal_additive_baseline_summary.csv` | FULL |
| D4 true versus wrong-order inference | `results/tables/d4_order_control_summary.csv` | FULL |
| C7 causal figures | `results/figures/c7_causal_by_layer.*`, `c7_causal_by_step.*`, `c7_cycle_recovery.*` | FULL |
| D4 relation figure | `results/figures/d4_relation_intervention.*` | FULL |
| Irrep spectra | `results/figures/irrep_spectra.*` | FULL |
| Geometry/algebra figure | `results/figures/intertwiner_vs_cka.*` | FULL |
| Scale figure | `results/figures/scale_analysis.*` | FULL |
| Detailed structural source | `structural/template_metrics.csv` | FULL |
| Detailed causal source | `causal/intervention_metrics.csv` | FULL |
| Cross-model source | `intertwiner/cross_model_metrics.csv` | FULL |
| Representation source | `irrep/representation_diagnostics.csv` | FULL |
| Intervals | `bootstrap/confidence_intervals.csv` | FULL |
| Provenance | manifest, environment, config, ledger, telemetry, checksums | FULL |

Every figure has SVG and PDF forms plus a sibling `.source.csv` containing the
exact plotted data. Every stable table has a `source_csv` column. A
joint-minus-ridge interval containing zero is described as no detected
difference, never equivalence. A CKA/algebra interval containing zero is no
reliable detected association, never independence. Historical
private-workspace evidence remains separately labeled and is never silently
replaced.
