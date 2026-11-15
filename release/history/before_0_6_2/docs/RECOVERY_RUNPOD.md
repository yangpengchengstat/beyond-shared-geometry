# Recover the failed calibration and plan the next RunPod session

The first BF16 CPU study stopped because two positive-control cells did not meet the registered 90% joint sensitivity criterion. The failed result is evidence to preserve. A completed stage or a successful archive checksum is not a scientific PASS.

## Preserve the old run

Keep the downloaded result ZIP, its SHA-256 file, the complete run-directory TAR archive, its SHA-256 file, and `cpu-bf16.log`. Verify the checksums on the computer holding the downloads. The default result ZIP excludes `.npy` and `.npz` arrays; it is an audit bundle, not a complete resume backup.

Do not edit or delete the failed run's identity/configuration records. Do not replace its source hashes with those of this revision. The revised program needs a **fresh run directory**. Existing model checkpoint files may be reused through the Hugging Face cache; scientific result reuse requires explicit matching provenance.

Once the backups are verified and contain the expected files, an idle temporary-storage Pod can be deleted. Do not delete a Pod before its local files are backed up. Network volumes are billed separately and survive Pod deletion.

## Diagnose locally before paying for another GPU

First inspect the existing seed-level results and run the bounded development diagnostics described in the revision report. These are CPU tasks. Their fixed seed lists, sample-size alternatives, limits, elapsed time, and status must be recorded. A reduced development experiment does not substitute for the registered calibration. Do not change the 90% criterion to make the failed run pass.

In an installed **CPU** environment, run this command once into a fresh local directory:

```bash
bsg-next diagnose --config configs/prospective/bf16_revision.json \
  --output /tmp/bsg-context-development-2026-10-19 \
  --replicates 12 --max-seconds 180
```

This compares noisy Y4 positive controls at 32, 48, and 64 test families using an independent development seed namespace. It remains permanently technical and ineligible for paper results. Review any budget-limited completion status; a small diagnostic is not a high-precision estimate of 90% sensitivity. Do not rent a GPU for this command.

Do **not** immediately repeat `next_run.sh full` or the old all-phases background script. Determine which implementation or design change is justified, freeze the revised profile, and measure a small technical run before authorizing a larger CPU job. A new BF16 runtime profile does not by itself fix sensitivity.

## Run CPU and GPU work separately

The launchers default to `configs/prospective/bf16_revision.json`. Its sample-size revision is a prospective candidate, not an independently validated fix. Set `BSG_CONFIG` explicitly for every terminal and keep it, the source tree, dependency versions, and scientific overrides identical for every phase in one study. Choose a fresh Linux path and preserve it across machines, because the study stores absolute paths.

Example setup on a CPU machine:

```bash
cd /workspace/beyond-shared-geometry
export BSG_CONFIG=/workspace/beyond-shared-geometry/configs/prospective/bf16_revision.json
export BSG_RUN=/workspace/bsg-runs/icml2027-bf16-revision-v1
export BSG_VENV=/workspace/bsg-cpu-venv
bash runpod/next_bootstrap.sh cpu
```

Only after the development diagnosis supports the registered revision, start its CPU stage:

```bash
bash runpod/next_cpu.sh "$BSG_RUN"
```

The CPU launcher refuses a detected GPU unless `--allow-cpu-on-gpu` is explicitly supplied. That flag acknowledges that CPU work may keep a billable GPU idle; it does not bypass any scientific check. There is no automatic cloud shutdown, spending cap, or guarantee of completion time. Program failure does not stop Pod billing.

The CPU phase now evaluates the 800 registered positive-control datasets first and stops immediately if their strict gate fails. A second positive check follows the complete calibration. Negative controls and retrospective diagnostics are not started after a positive-check failure. Preserve the outputs instead of deleting them and retrying a different seed.

The CPU phase no longer reruns the expensive historical archive diagnostics by default. Existing failed-run diagnostics remain immutable evidence. If the numerical method or historical analysis changes, request the separate CPU-only retrospective stage explicitly:

```bash
bash runpod/next_retrospective.sh "$BSG_RUN" \
  --archive /workspace/beyond-shared-geometry/inputs/full.zip
```

This is not a claim that outputs from an old source identity passed the revised study.

Before moving a completed study directory from a CPU machine to a GPU Pod, create and verify a complete backup, restore it under the **same absolute path**, and keep the code/configuration unchanged. Do not copy the CPU virtual environment to the GPU machine.

Example GPU setup:

```bash
cd /workspace/beyond-shared-geometry
export BSG_CONFIG=/workspace/beyond-shared-geometry/configs/prospective/bf16_revision.json
export BSG_RUN=/workspace/bsg-runs/icml2027-bf16-revision-v1
export BSG_VENV=/workspace/bsg-gpu-venv
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
bash runpod/next_bootstrap.sh gpu
source "$BSG_VENV/bin/activate"
bsg-next preflight --config "$BSG_CONFIG" --require-cuda --check-access \
  --output /workspace/bsg-preflight-revision.json
bsg-next model-smoke --config "$BSG_CONFIG" --execute-models \
  --output /workspace/bsg-model-smoke-revision
```

The separate CPU/GPU bootstrap defaults prevent a CPU Torch wheel from being reused accidentally on a GPU Pod. Standard Hub downloads are enabled by default; an optional transfer backend is not required. Tokens belong in private environment settings or the external Hugging Face cache, never in the study directory or result archives.

## Start one model phase at a time

After the required calibration is valid and the measured model checks succeed:

```bash
bash runpod/next_gpu.sh pilot "$BSG_RUN" --execute-models
```

Inspect the returned task-competence and power decisions before continuing. Only after the required frozen selections exist:

```bash
bash runpod/next_gpu.sh confirm "$BSG_RUN" --execute-models
```

Then schedule each separately, checking actual time, storage, failures, and remaining budget:

```bash
bash runpod/next_gpu.sh peers "$BSG_RUN" --execute-models
bash runpod/next_gpu.sh crossmodel "$BSG_RUN" --execute-models
bash runpod/next_gpu.sh extensions "$BSG_RUN" --execute-models
```

The GPU launcher never invokes the CPU phase automatically. Conditional extension outcomes can be ineligible or inconclusive. These commands do not guarantee scientific effects or conference acceptance.

Use the same stage command to resume only a matching run. Do not run two writers on the same study directory. A source or protocol change needs a new registered identity. A CPU-to-GPU transfer is not permission to relabel an old failed calibration as successful.

## Collect results without confusing silence with failure

```bash
bsg-next collect --run "$BSG_RUN" --output /workspace/bsg-revision-results.zip
```

Collection scans the records and computes checksums; a large seed archive takes time. Wait for the command to finish and return an archive summary. Verify the companion `.sha256` file, download both, and keep the complete study backup separately if resumption is needed. Never stop a temporary-storage Pod just because a partially written ZIP has appeared.

The original [prospective guide](PROSPECTIVE_RUNPOD.md) contains the broader experiment definitions. For revised scheduling and launcher defaults, use this recovery guide.
