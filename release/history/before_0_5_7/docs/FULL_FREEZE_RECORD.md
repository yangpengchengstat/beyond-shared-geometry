> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# FULL Freeze Record

METHOD STATUS: FROZEN FOR FULL
FULL STATUS: AUTHORIZED

- Methodology frozen: 2026-09-04
- PRE-FULL run: `pre_full_001`
- PRE-FULL execution: `SUCCESS`, 70/70 units complete
- PRE-FULL overall gate: `CONDITIONAL_GO` because two-depth calibration cannot
  evaluate three-depth FULL gates; no execution failure
- Additive audit: `PASS WITH REPORTING LIMITATION`
- Additive audit decision: `AUTHORIZE FULL`
- Approved source commit before freeze:
  `06981aaf7e3645785b64c7dc5628ca8cfd89c1e5`
- Final deployment source commit: the commit containing this record, resolved
  exactly by `BUILD_INFO.json` and the companion package manifest
- Package: `Beyond_Shared_Geometry_main_experiment_runpod.zip`
- Package SHA256: generated after the source commit and recorded in
  `runpod_packages/Beyond_Shared_Geometry_main_experiment_runpod.manifest.txt`
  and its `.zip.sha256` companion; it cannot be embedded in its own committed
  source record without creating a circular package/commit hash dependency

## Frozen Scientific Matrix

| Axis | Frozen FULL value |
|---|---|
| Models | Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B |
| Groups | C7, D4 |
| Primary depth slots | 0.20, 0.40, 0.60, 0.80 |
| Primary latent dimension | 16 at full planned coverage |
| Primary split seeds | 101, 303, 909, 2701, 4243 |
| Sensitivity dimensions | 8, 32, 64 |
| Sensitivity coverage | depths 0.40/0.80 and splits 101/303 only |
| Learned fits | one deterministic fit per independent cell; technical seed 42 |
| Random controls | 104729, 130363, 155921, 196613, 262147 |
| Templates | 28 train, 6 validation, 6 test |
| Bootstrap | 5,000 replicates; seed 314159 |

Within-model inference uses `depth -> split -> template`. A declared aggregate
cross-model estimand uses `model -> depth -> split -> template`. Random-control
seed is an additional level only for contrasts involving random controls.
Optimization seed is not a replication axis.

The training-only additive comparator is

`v_(g,s) = mean_(t in T_train)[h(g s,t) - h(s,t)]`,

applied to held-out activations as
`h'(s,t_test) = h(s,t_test) + v_(g,s)`. It is source- and
transformation-conditioned and is not interpreted as a compositional
representation.

Under the repository's row-vector convention, the true D4 word is
`s @ r @ s = r^-1`, with product `(3,0)`. The matched-length wrong-order word
is `r @ s @ r`, with product `s=(0,1)`, which is not `r^-1`.

Expected scientific units are 6 model loads, 12 structural extractions, 60
causal extractions, 444 structural base/ridge fits, 444 structural joint fits,
240 causal units, and 360 cross-model units: 1,566 total. Complete
post-processing expects 926 bootstrap analyses, 9 stable tables, and 9 figure
families with source CSVs.

FULL checkpoint keys bind model, group, layer/depth, latent dimension, split,
technical fit seed, and the complete five-seed random-control set. Causal keys
also bind every intervention name, exact target/word element, evaluation role,
and method, including `source_conditioned_additive`. Random realizations remain
separately identified by `random_baseline_seed` inside the atomic cell
checkpoint. Thus all five controls are recoverable on resume without treating
them as learned-fit replications, while the scientific unit count remains
unchanged. This key enrichment is FULL-only; historical CORE and completed
PRE-FULL checkpoint identities are untouched.

PRE-FULL telemetry projects 37,471.07 serial-equivalent A40 seconds, or 10.41
hours. This is an estimate, not guaranteed wall-clock runtime.

FULL requires an explicit successful PRE-FULL authorization manifest. The
deployment package contains the reviewed record at
`runpod/full_authorization/manifest.json`; it does not depend on local dated
result directories or the additive audit document at runtime.

No scientific protocol changes are permitted after this freeze except to
correct a demonstrated implementation defect.
