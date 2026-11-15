# Optional historical GPU execution

This directory preserves the main, AFS-A/B/C, and rank-aware AFS-D runtime launchers. It is separate from the CPU/paper release workflow in [Reproducibility](../docs/REPRODUCIBILITY.md). These scripts do not create or terminate RunPod resources. Historical calibration and authorization records describe the archived runs; they do not launch or authorize a new paid experiment.

## Prepare a fresh reproduction

Model execution requires an intentionally provisioned machine, suitable model-provider access, GPU dependencies, and fresh run/output/cache identities. Install `requirements-gpu.txt` in a separate environment. The legacy deployment builders require the relevant clean source/provenance checks; a publication ZIP is not a Git checkout. Choose the main or AFS builder for the desired experiment rather than assuming there is only one pipeline.

The main deployment can be built from a separately versioned clean working copy with:

```bash
bash scripts/build_runpod_package.sh main_experiment
```

Upload the generated main runtime archive to a fresh machine/location and extract it into an empty directory. Set `HF_TOKEN` through the environment. The following commands are for an explicitly chosen **new** smoke execution, not for updating frozen evidence:

```bash
cd /workspace/beyond-shared-geometry
bash runpod/bootstrap.sh
bash runpod/preflight.sh
export BSG_RUN_ID=reproduction_smoke_001
export BSG_OUTPUT_ROOT=/workspace/reproduction_outputs/main_experiment
export BSG_CACHE_DIR=/workspace/reproduction_cache
export RUNPOD_RESULT_BUNDLE_PATH=/workspace/reproduction_smoke_001_results.zip
bash runpod/run_main_experiment.sh --smoke
```

Use unused names and locations for each new experiment. Each tier is selected explicitly; the runner does not advance from SMOKE to CORE, PRE-FULL, or FULL. FULL additionally checks clean source provenance and an explicitly supplied successful calibration manifest. Existing historical records are not evidence that a new run satisfies those checks.

The slim runtime bundles omit publication assets and the publication-only release regression test. Use `tests/run_runtime_checks.py` or the runner's CPU-contract preflight there. Use the complete public source project for `scripts/check_release.py`, paper rebuilding, and source/Overleaf packaging.

## Resume versus new execution

Reuse a `BSG_RUN_ID` only to resume that same newly created computation under its matching source/configuration. The ledger skips verified complete units and recomputes corrupt/incomplete ones. Do not reuse identifiers such as `main_full_002` or point a new run at original frozen outputs. The historical repair procedure is recorded in [FULL_CHECKPOINT_HYDRATION_REPAIR.md](../docs/FULL_CHECKPOINT_HYDRATION_REPAIR.md); it is historical context, not the default reproduction recipe.

The result packager writes the chosen `RUNPOD_RESULT_BUNDLE_PATH` and its checksum. Its historical default is a fixed path that can be overwritten by later runs, so the example chooses a new result filename explicitly. Activation caches and scientific checkpoints are omitted from the lightweight result ZIP; preserve them separately when needed for a reproducible new run. Never overwrite the archived experiments or published repository to follow these instructions.
