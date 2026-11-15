## Current revision — 24 October 2026

Version 0.4.0 integrates the completed returned replay as reviewed descriptive evidence. Compact/context and secondary extension results are now available. Native competence still failed, structural power remains limited, and independent confirmation is blocked. Original frozen results and source capsules retain their identities.

For current instructions use the root README, `docs/REPRODUCTION_REVIEW_20261024.txt`, and `release/reproduction_review.json`. Rebuild tables offline with `python scripts/build_reproduction_assets.py`; no new calibration or full model replay is needed. Maintenance v2 defaults to checking only and requires a separate explicit choice before any execution. Prior dated instructions below describe earlier project states and should not be used to launch another run automatically.

## Preserved earlier documentation

# Reproducibility

This release separates three operations: checking exported evidence, rebuilding the paper, and rerunning models. Only the third requires GPU/model access. None is launched automatically by another tier.

The 0.3.2 active runtime adds execution-integrity guards and intentionally requires a fresh run identity/output receipts. It does not replace historical numerical source or the frozen characterization worker. Consult [the current review](PROJECT_REAUDIT_2026_10_21_V2.md) before any model run. Completed descriptive aggregates are curated separately from private imports; rendering them does not change original eligibility or inference flags.

## CPU and publication checks

From the repository root, install `python -m pip install -e '.[test,analysis]'`, then run:

```bash
PYTHONPATH=src python tests/run_cpu_checks.py
python scripts/verify_frozen.py
python scripts/check_release.py
```

The inherited suite checks C7/D4 actions and row-vector order, template splitting, train-only PCA, observed rank, additive controls, checkpoint integrity/hydration, bootstrap determinism, and historical runtime contracts. Release tests additionally reject missing/untraced TeX assets, altered artifact hashes, unrecorded numerical-source edits, private paths, credentials, and nonportable dependencies. Small synthetic CPU fixtures exercise these contracts; passing them does not repeat GPU experiments or validate every scientific inference.

`release/source_manifest.json` records the exact exported numerical-source/configuration bytes. FULL's historical source commit is recorded separately in `release/release_manifest.json`. Later AFS-specific additions in the exported repository must not be described as FULL's historical execution state. `results/registry.json` and per-run public provenance connect selected outputs to their source identities; an exported-source hash and a frozen-run source identity serve different purposes.

## Rebuild the manuscript

Install the optional paper extra (`python -m pip install -e '.[paper]'`) for PDF QA, plus a TeX engine for compilation. Each asset builder retains its own provenance; the following commands rebuild all current manuscript assets.

```bash
python scripts/build_paper_assets.py
python scripts/build_characterization_status.py
python scripts/build_characterization_assets.py
bash scripts/build_paper.sh
python scripts/check_pdf.py
python scripts/check_release.py
python scripts/package_release.py
```

Asset rebuilding reads only bundled selected evidence and writes derived tables/figures. It does not recompute model activations. A local PDF build needs a TeX distribution; the Overleaf ZIP is an alternative with all local dependencies included. Standard distribution packages are not model dependencies. The packager includes the compiled PDF when present and always includes editable source.

The manuscript targets ICML 2027 using the unmodified official ICML 2026 fallback template. `submission_ready_for_2027` stays false until the actual 2027 rules and template are checked. A successful compile checks typesetting, not scientific acceptance or submission eligibility. The public source ZIP retains identity and licensing; the paper ZIP contains anonymous manuscript material only.

## Optional GPU reproduction

Create a separate environment and install `python -m pip install -r requirements-gpu.txt`. Model-provider access, exact pinned model revisions, sufficient GPU memory, disk space, and model licenses are prerequisites. Supply credentials through the environment; never save them in files or commands intended for publication.

Configuration validation is CPU-safe:

```bash
python -m bsg.experiments.main_experiment --smoke --validate-only
```

Actual SMOKE/CORE/PRE-FULL/FULL and AFS execution is documented by the preserved `runpod/` scripts and frozen protocol/configuration files. Read each script's help and select the tier/output location explicitly. The historical FULL runner requires a clean Git source tree and an explicitly supplied successful calibration manifest. The reviewed record under `runpod/full_authorization/` is historical calibration evidence, not permission to provision a machine or repeat a paid run. An exported ZIP is not a Git checkout; initialize/version a separate working copy intentionally before any runner that requires Git provenance.

The legacy `scripts/build_runpod_package.sh`, `scripts/build_afs_runpod_package.sh`, and `scripts/build_afs_d_runpod_package.sh` prepare optional slim runtime deployments after their clean-source checks. Those deployments deliberately omit the manuscript and selected paper evidence. They retain inherited numerical tests and use `tests/run_runtime_checks.py` for runtime preflight; the publication-only release test is omitted because it depends on the full paper/evidence bundle. Use `scripts/package_release.py` for the complete public source and anonymous paper archives. This publication reorganization changes packaging inputs, not numerical definitions.

New execution must use a new output directory and record its own code identity, dirty state, complete configuration, requested/resolved model revisions, tokenizer identity, environment, seeds, ledger, and artifact hashes. Preserve frozen outputs and published repositories unchanged. Floating optional dependency bounds describe supported installation ranges; they do not guarantee bitwise reproduction of a frozen GPU environment. Compare the per-run environment/provenance and preserve any numerical discrepancies.

## Statistical and gate scope

The historical bootstrap nests model/depth/split/template axes, adding random-control seed for its comparisons and collapsing exact technical optimizer duplicates. The implementation's resampling hierarchy does not automatically account for all crossed dependencies from reused templates or model pairs sharing a model. Report the actual estimand and model panel, and retain this limitation.

For completed FULL runs, `overall: GO` follows execution completion in the frozen runner; component scientific gates and additive comparisons remain separate. The C7 causal gate compares with identity/random operators. It does not require superiority over the source-conditioned additive baseline. Release verification preserves these facts and does not redefine historical PASS/FAIL outcomes.
