# Beyond Shared Geometry

Version 0.4.9, spectral implementation and manuscript revision dated 29 October 2026. A research release about finite-group predictor families in language-model activations and the limits of identifying semantics or native computation from them.

The returned full replay and rank-aware continuation are complete **with preserved scientific stops**. The manuscript now includes reviewed compact nonlinear, context-transport, trained-interface, peer-adaptation, cross-model and product-action results. This is descriptive integration of inspected evidence, not independent confirmation. Native clean competence still failed; structural power remained limited; the original rank-16 embedding control remains infeasible.

The 0.4.9 revision adds a separate, opt-in implementation of the real-type-constrained spectral budget and a verified cached-coordinate input adapter. It strengthens input, projector, whitening and numerical checks, preserves the supplied prototype unchanged, and retains the 0.4.8 fixed 48-cell test-bank protocol. The manuscript distinguishes the positive finite-bank risk guarantee from semantic/causal identification and from prediction on new contexts. **No real-coordinate spectral analysis, model execution, calibration or experiment was performed in this revision.** See the [implementation workflow](studies/spectral_finite_bank_v1/README.md), [review dispositions](docs/GAMMA_IMPLEMENTATION_REVIEW_0_4_9.txt) and [next-evidence decision](docs/NEXT_EVIDENCE_DECISION_0_4_9.txt). Deterministic software checks are not empirical research results.

The 0.4.8 revision reconciles the supplied theory and review reports with the current manuscript. It clarifies the prediction target on context differences, distinguishes total representation-type energy from energy selected by the spectral budget, and separates state invariance from semantic or causal interpretation. The main findings table, complete budget figure, all numerical results and scientific runtime are unchanged. See the [reference disposition and revision record](docs/INTEGRATED_REVIEW_0_4_8.txt), [proposed finite-bank protocol](docs/SPECTRAL_FINITE_BANK_PROTOCOL_0_4_8.txt), and [next-evidence decision](docs/NEXT_EVIDENCE_DECISION_0_4_8.txt). The coordinate inventory establishes feasibility only: no spectral endpoint, new fit, model forward, calibration or resampling was computed in this revision.

The 0.4.7 revision brings the fixed-group, same-regression/same-spectrum counterexample and exact-action risk bound into a short main-text theory sequence. A three-finding table and full storage-budget figure replace the dense main summary; its original six-row evidence table remains in the appendix. The figure copies existing recorded means, including unfavorable budgets, without new fitting or statistical computation. An elementary appendix observation limits the normalized C7/D4 type-budget correction to at most 1/16 at dimension 16; no empirical type correction is claimed. See [the revision and experiment decision](docs/MAIN28_REVISION_0_4_7.txt) and [the conditional follow-up specification](docs/NEXT_EVIDENCE_DECISION_0_4_7.txt). No new experiment, model forward, calibration, cached spectral analysis or resampling was performed. Current fixed-panel claims do not require additional experiments for their stated scope; an independent context evaluation is recommended if broader empirical robustness becomes a core claim.

The 0.4.6 revision adds a concise theorem and proof connecting best invariant-subspace approximation to optimal prediction risk over the entire fixed-dimensional exact orthogonal action class. It includes a type-sensitive analytical counterexample, precise literature positioning, and a conservative finite-context certificate whose bound is unusable at the present context count. This population-whitened risk is not the historical experiment metric; no existing result receives certification. See [the integration record](docs/SPECTRAL_RISK_REVISION_0_4_6.txt). All experimental code, configurations, measurements and failure records are unchanged; no model execution, calibration, fitting or empirical resampling was performed.

The 0.4.5 revision distinguishes measured predictor performance, class-optimal risk, supported-space action uniqueness and native mechanism identification. It clarifies fixed-coordinate support restrictions and precisely positions the paper relative to Sun and Equivariance by Contrast. These are standard interpretation boundaries, not a new general theory. See [the concise integration record](docs/IDENTIFICATION_SCOPE_REVISION_0_4_5.txt). Existing measurements, experiment code, failed gates and prior releases are unchanged. No experiment was added or rerun.

The 0.4.4 revision leads with concrete predictive, storage and context-transport findings, consolidates repeated limitations, and corrects registration-specific scoring masks in the paper and captions. All numerical evidence and scientific code are unchanged. See [the revision and independent-context assessment](docs/FINDINGS_REVISION_0_4_4.txt). No new experiment is required for the retained fixed-panel findings; independent context validation is the recommended next scientific extension if broader generalization is pursued. It has not been run.

The 0.4.3 revision clarifies AFS-C training-pair corruption, Ridge refitting and scoring against unchanged true held-out targets. It distinguishes local state conjugacy from element-wise target scrambling and explains relation-score normalization and shared-seed dependencies. All numerical results and scientific runtime files are unchanged. See [the local revision record](docs/AFS_C_CLARIFICATION_0_4_3.txt). No experiment, calibration, fitting or empirical resampling was run in this publication revision.

The 0.4.2 revision organizes the paper around what recovery tests identify, places six existing contrasts in one traceable main table, and separates four interpretation questions: predictive recovery, action consistency, preferential semantic alignment and native causal use. Original curves and detailed estimates remain in the appendices. See the [restructuring record and next-step assessment](docs/NARRATIVE_REVISION_0_4_2.txt); the [0.4.1 classified checklist](docs/FINAL_REVISION_CHECKLIST_20261025.txt) remains a dated record with its original source-line references. No experiment, calibration, fitting or statistical resampling was added. Scientific runtime remains 0.4.0. Independent-context validation is optional for broader generalization, not necessary for the retained fixed-panel claim.

## Current workflow: offline review and paper build

No GPU, model download, calibration sweep or full experiment rerun is needed for the current work. The primary entry points are:

- `paper/main.pdf` and `paper/main.tex`: anonymous manuscript with proofs and limitations.
- `results/reproduction_v1_review/`: fixed-panel rows, preserved stop records and reviewed summaries.
- `results/reproduction_action_review/`: verified per-action records and separate all-action and unseen-only endpoints.
- `release/reproduction_review.json`: manual integration decision and source archive/input hashes.
- `docs/REPRODUCTION_REVIEW_20261024.txt`: claims, metric scopes and evidence boundaries.
- `docs/theory_revision_20261024.txt`: disposition of the supplied theoretical report.
- `studies/maintenance_v2/`: opt-in future maintenance workflow; read its scope before use.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,analysis,prospective,paper]'
PYTHONPATH=src python tests/run_cpu_checks.py
python -m pytest -q studies/spectral_finite_bank_v1/tests
python scripts/verify_frozen.py
python scripts/verify_prospective.py
bash scripts/build_paper.sh
python scripts/check_pdf.py
python scripts/check_release.py
python scripts/package_release.py --output-dir dist/reviewed-0.4.9
```

The new figure and concise table are rendered from existing recorded values by scripts/plot_budget_frontier.py and scripts/build_main_findings.py. They do not fit predictors, recompute statistical summaries or inspect raw activation coordinates. Other existing tables and figures are reused. The optional paper extra is needed for PDF checks; a local TeX distribution is needed for compilation. To use Overleaf, upload `dist/reviewed-0.4.9/overleaf.zip`, select `main.tex`, and use pdfLaTeX. It contains its complete dependency closure and bibliography.

The target is ICML 2027. The unchanged official ICML 2026 style is a fallback; the official 2027 format and submission rules remain unverified. Successful compilation and eight-page main-text compliance do not certify acceptance or submission readiness.

## Preservation and reproducibility

The prior source release, returned ZIP, original results and failed runs remain unchanged outside this new version. Checked-in historical frozen evidence and frozen characterization workers retain their original checksums. Runtime maintenance has a new source identity and is not retroactively attributed to the completed run. See `release/runtime_revision.json` for exact differences.

The original from-zero and rank-recovery scripts under `studies/reproduction_v1/` and `studies/reproduction_rank_recovery_v1/` retain their frozen manifests as historical references. Their old manifests describe the old source capsule, so they intentionally cannot verify the revised active runtime. Do not launch those wrappers from this release or edit their manifests to make them pass. The original capsule is needed for exact old-source replay.

The result export contains compact coordinates, row banks and interface parameters, but omits regenerable raw activation caches and model weights. The public source ZIP includes reviewed numeric rows and selected frozen evidence, not that full private 3 GB export. Raw-forward/PCA or complete neural retraining requires rebuilding inputs. The audit independently repeated five representative coordinate analyses, not all model forwards or all 1,728 neural fits.

## Status of earlier workflows

The characterization and compact/context studies have completed. Their original protocols remain available for provenance; earlier labels such as “prepared” or “awaiting audit” in dated documents describe their historical state. The current status is `release/reproduction_review.json` and the manuscript's returned-replay appendix. Original `paper_eligible=false` and `independent_confirmation=false` fields are retained. Reviewed descriptive tables have a separate explicit integration decision.

## What the evidence supports

The tasks cross seven weekday or four corner labels with controlled templates. PCA and operator fits use training templates; held-out templates assess transfer of those fits. Joint fitting explicitly includes algebraic constraints. Its relation and composition residuals therefore measure consistency, with additional falsification controls needed to distinguish stronger algebraic claims from category encoding.

Activation edits test downstream effects of externally applied operators. Cross-model intertwiners test commutation compatibility, including possible shared subspaces. Neither result alone establishes an internal reasoning algorithm or equivalence of full model representations. The source-conditioned additive comparisons are part of the evidence and must be read alongside identity/random controls. See [Scientific scope](docs/SCIENTIFIC_SCOPE.md).

Frozen FULL `overall: GO` means the execution completed under that runner's logic. Scientific component gates remain separate, and the FULL causal gate does **not** require superiority over the additive baseline. No release check rewrites those historical definitions or results.

## Repository map

| Location | Purpose |
| --- | --- |
| `src/bsg/`, legacy `configs/` | Preserved numerical implementation and frozen experiment matrices |
| `src/bsg_next/`, `configs/prospective/` | New prospective CPU/GPU runtime and frozen pre-execution design |
| `tests/` | Inherited CPU checks and publication-release regression tests |
| `studies/spectral_finite_bank_v1/` | Opt-in finite-bank spectral implementation; real-coordinate analysis not executed |
| `studies/characterization_v1/` | Byte-verified frozen study, private-result importer, independent coordinate refits and model-free tests |
| `results/frozen/`, `results/registry.json` | Selected frozen evidence and source identities |
| `results/characterization_v1/` | Manually reviewed, descriptive aggregate tables; original private coordinates excluded |
| `results/derived/` | Selected portable derivatives of frozen evidence |
| `paper/generated/`, `paper/figures/` | Traceable publication tables, figures, and figure source CSVs |
| `paper/` | Anonymous manuscript, bibliography, official style, referenced assets |
| `release/`, `provenance/` | Source hashes, artifact traceability, readiness and template metadata |
| `docs/` | Reproduction, scientific scope, and preserved historical protocols |
| `runpod/` | New prospective stage launchers and preserved historical GPU scripts |
| `scripts/` | Offline verification, paper assets/build, and deterministic packaging |

Historical protocol documents describe their recorded stage of the project; they are not automatically current release instructions. [Reproducibility](docs/REPRODUCIBILITY.md) distinguishes source-only checks, frozen evidence verification, and optional new model execution. [Artifacts](docs/ARTIFACTS.md) explains hashes and derivative provenance.

The [literature audit](docs/LITERATURE_AUDIT.md) documents primary-source comparisons and corrected attribution. The [scientific audit](docs/SCIENTIFIC_AUDIT.md) separates supported findings from identification gaps. [Review coverage](docs/REVIEW_COVERAGE.md) records which files, archives, and accessible conversations were checked and the limits of that access. The [research roadmap](docs/RESEARCH_ROADMAP.md) lists prospective experiments; those proposals are not completed results.

## Active runtime and earlier maintenance

Version 0.4.0 adds verified atomic interface/factorial checkpoints, completed/total progress, durable session timing and a separate opt-in maintenance launcher. It retains the confirmation prerequisites, output receipts and extraction-index checks introduced in 0.3.2. Historical numerical source and the frozen characterization worker are unchanged. Earlier outputs remain bound to their recorded source version; do not restart a failed study under a changed identity. See `studies/maintenance_v2/README.md` for the current narrow workflow and [the earlier re-audit](docs/PROJECT_REAUDIT_2026_10_21_V2.md) for its predecessor.

## Optional model reproduction

CPU validation and paper rebuilding do not reproduce hidden states or GPU measurements. Model reproduction requires suitable hardware, model-provider access, the optional GPU environment, and explicit selection of a historical tier. Installing GPU dependencies does not start an experiment:

```bash
python -m pip install -r requirements-gpu.txt
python -m bsg.experiments.main_experiment --smoke --validate-only
```

Consult [Reproducibility](docs/REPRODUCIBILITY.md) before an actual run. Run scripts do not provision paid infrastructure. The preserved full-replay orchestrator requires the restored original source capsule; it is not an entry point for the revised active runtime. The current maintenance launcher exposes only explicitly selected interface/factorial tasks. Frozen outputs, model weights, activation caches, private historical workspaces, and original repositories must remain untouched.

## License and anonymity

The public source archive preserves the original [MIT license](LICENSE) and source author metadata. It is a public-release preparation artifact. The separate Overleaf archive is the anonymous manuscript bundle; it excludes repository identity documents. Model-provider and third-party template terms remain applicable. Do not publish an identifying repository link in a double-blind submission without checking the venue's rules.
