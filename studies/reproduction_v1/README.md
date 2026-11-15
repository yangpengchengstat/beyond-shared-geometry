# From-zero A40 reproduction

This is a new replay namespace. It reconstructs raw model activations and reruns the current descriptive study **and all implemented historical FULL/AFS/calibration recipes**. It does not require an old RunPod disk, activation cache, or downloaded result ZIP. Preserve existing result archives: rerunning does not replace the provenance of the original observations.

A completed program is not a favorable scientific outcome. Previously inspected prompt populations remain previously inspected. Outputs are marked as replication/exploratory, not a fresh independent confirmation, and are not automatically copied into the manuscript.

## Exact scope

| Stage | Work performed | Scientific dependencies |
|---|---|---|
| Characterization | Regenerate exact C7/D4/S3 populations; six pinned base models, four readouts; BF16 extraction, float64 train-only PCA q16; 72 cells, 120 evaluation strata, 48 field cells; descriptive uncertainty and embedding floor | No old cache or calibration required |
| Compact/context | Frozen RFF and Nyström compact predictors, storage budgets, direct/context controls, semantic and fixed alternative actions; 240 jobs on newly generated coordinates | Verified fresh characterization |
| Historical PRE-FULL/FULL | Original FP16 six-model structural, causal, cross-model and dimension-sensitivity recipes | Original PRE-FULL execution/readiness gates retained |
| AFS-A/B/C and initial D | Generator composition, algebra permutation null, correspondence scramble, original q16 embedding diagnostic | New FULL bound by actual manifest/checkpoint hashes; verified PRE-FULL coverage |
| Repaired AFS-D | Original rank-aware C7 q6/D4 q3 embedding-versus-internal controls | New FULL raw activations and new AFS, not the old archive |
| Retrospective pilot/full | Original 16/240-cell saved-operator versus source-CSV replay, then retrospective controls using newly generated FULL | New FULL; descriptive paired templates, not new independent families |
| Additive-field falsification | Original 240-cell matched operator/additive comparisons and original bootstrap/outcome rules | New FULL checkpoints |
| Original and revised calibration | Both original and BF16/F64 revised design recipes | Current preserved corrected numerical implementation; old numerical bugs are not emulated |
| Prospective pilot/confirmation | Original context-eligibility, power and native gates; confirmation only when its original prerequisites pass | Revised calibration; original thresholds unchanged |
| Native intervention | Original clean competence, selection, power and intervention pipeline, including matched peer baselines when eligible | Stops on failed clean competence or power |
| Peer adaptations | Configured original-model anchors and factorial conditional-mean comparison | Independently scheduled; these are declared adaptations, not full author-code reproductions |
| Cross-model | Controlled whole-word displacement transport and descriptive structural transport | Word fields or fresh characterization |
| Extensions | C4×C4/product actions and supervised learned-interface training | Own fresh extraction or fresh characterization caches |

Not fabricated or silently omitted: GetImpl/behavior-preserving implementation families for Sun–Toneva and independent multilingual/authored banks are not supplied or implemented as a complete acquisition pipeline. They appear as missing external prerequisites in `plan.json`. The conditional third-lineage study is disabled in the registered six-model recipe. Past ad-hoc debugging scripts are not counted as independent formal experiments.

## Hardware and space

Use one NVIDIA A40 (48GB VRAM), at least about 40GB available host RAM, and **300GB container disk** for this broader replay. The initial guard requires 180GiB free before model downloads and 100GiB afterward. Current-mainline raw arrays alone are 28.97GiB; weights, historical/prospective cache trees, fitted interfaces and outputs are additional. No automatic deletion occurs. Check actual disk usage during the run. Smaller disks may be workable with a separately planned staged archive workflow, but that is not this all-at-once launcher.

Persistent storage is optional for a single uninterrupted run, but a container disk is temporary. Download and verify results before stopping, restarting or terminating the Pod: container-disk data does not survive these actions. See the [official RunPod storage lifecycle](https://docs.runpod.io/pods/storage/types). This program never stops or deletes the Pod. Do not assume closing the browser stops billing. There is **no three-hour timeout**. Whole-suite duration and cost are unknown until measured on the new machine; historical FULL and supervised-interface fits can be much longer than the characterization stage.

Use Linux x86_64 and Python 3.11. One published RunPod image with these components is `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` ([official image layers](https://hub.docker.com/layers/runpod/pytorch/2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04/images/sha256-cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853?tab=layers)). Confirm the selected image actually provides `python3.11`; the project installs its own sealed Python environment and does not rely on the template's preinstalled packages. The RunPod image needs an NVIDIA driver compatible with the pinned Torch 2.8.0 CUDA 12.8 wheel. The hardware preflight exercises BF16 and records the driver, GPU, CUDA build and exact Python patch version. A40 feasibility is based on earlier execution, not a new hardware test of this package.

## Upload and bootstrap

Upload `bsg-from-zero-v1.zip` and its `.sha256` to `/workspace` using JupyterLab. In a terminal:

```bash
cd /workspace
sha256sum -c bsg-from-zero-v1.zip.sha256
mkdir -p /workspace/bsg-reproduction-source
unzip -n bsg-from-zero-v1.zip -d /workspace/bsg-reproduction-source
cd /workspace/bsg-reproduction-source/beyond-shared-geometry
python3.11 --version
bash studies/reproduction_v1/bootstrap.sh
```

If `python3.11` is absent, choose an image providing Python 3.11 before proceeding. Do not replace the old virtual environment. Bootstrap creates `/workspace/bsg-reproduction-venv`, resolves the full wheel dependency closure, saves wheel URLs and SHA256 hashes, then installs with `--require-hashes`. The shipped `requirements.in` is only the top-level input; the actual complete lock is created before installation on Linux. To reproduce that exact resolution later, supply the saved `environment.requirements.lock` as the bootstrap argument. Exact Python patch/driver details are recorded; bitwise agreement across different hardware or BLAS implementations is not promised.

Authenticate to Hugging Face after accepting access terms for **all** configured gated models. Model download failures stop before the experiment queue. Never paste the token into a script or this chat:

```bash
export HF_HOME=/workspace/bsg-model-cache
/workspace/bsg-reproduction-venv/bin/hf auth login
```

## Start the entire queue

```bash
cd /workspace/bsg-reproduction-source/beyond-shared-geometry
mkdir -p /workspace/bsg-logs
nohup bash studies/reproduction_v1/start.sh /workspace/bsg-runs/from-zero-v1 \
  > /workspace/bsg-logs/from-zero-v1.log 2>&1 < /dev/null &
```

`start.sh` checks the source manifest, hardware/environment, disk and model access; downloads only exact pinned commits; records SHA256 hashes of weight/config/tokenizer files; then runs one GPU job at a time. Model loading subsequently runs offline. Bootstrap provenance is copied into the run. Historical extraction remains FP16; current extraction remains BF16; they are never silently shared as equivalent caches.

The characterization worker checks two identical first-batch model forwards before extraction, preserves batch8 boundaries, and commits each analyzed cell atomically. Every completed cell/checkpoint is verified before reuse. Incomplete work remains separate and is recomputed. This small engineering check is separate from the explicitly requested historical calibration.

## Progress, failure and resume

```bash
tail -n 30 /workspace/bsg-logs/from-zero-v1.log
/workspace/bsg-reproduction-venv/bin/python \
  /workspace/bsg-reproduction-source/beyond-shared-geometry/studies/reproduction_v1/run.py \
  status --output /workspace/bsg-runs/from-zero-v1
df -h /workspace
```

The outer log names the active stage and emits a heartbeat every 30 seconds. Detailed logs are under `/workspace/bsg-runs/from-zero-v1/logs/`; for example:

```bash
tail -n 30 /workspace/bsg-runs/from-zero-v1/logs/characterization.log
```

`scientific_gate_stopped` preserves a measured failure and skips dependent stages. Other independent branches continue. `technical_failure` stops the queue: inspect that stage's log before retrying. Never change thresholds to force a pass. To resume after fixing an environmental interruption, repeat the same launch command and same output path; verified completed work is reused. If code, dependencies or scientific configuration changed, use a new source package and new output directory. Do not launch two controllers on the same output.

## Download results

The final whole-suite archive and sidecar are:

```text
/workspace/bsg-runs/from-zero-v1.results.zip
/workspace/bsg-runs/from-zero-v1.results.zip.sha256
```

A final archive can be created even after a scientific stop. To collect partial evidence after a technical interruption, first ensure the experiment process has stopped, then:

```bash
/workspace/bsg-reproduction-venv/bin/python \
  /workspace/bsg-reproduction-source/beyond-shared-geometry/studies/reproduction_v1/run.py \
  collect --output /workspace/bsg-runs/from-zero-v1
cd /workspace/bsg-runs
sha256sum -c from-zero-v1.results.zip.sha256
```

Download both files and verify again locally. The archive includes numerical results, q16 coordinates, PCA/fitted predictors, generated task banks, configurations, input/model/source hashes, environment lock and logs. It excludes model weights and bulk high-dimensional raw caches. Those can be regenerated; the original run's compact evidence and provenance must still be retained. Review the complete `state.json`, `plan.json`, scientific gate records and `export_manifest.json` before describing the project as reproduced.

## Local validation and limits

These commands do not load/download model weights:

```bash
PYTHONPATH=src python -m pytest -q studies/reproduction_v1/test_environment.py studies/reproduction_v1/test_historical.py studies/reproduction_v1/test_orchestrator.py
python -m pytest -q studies/reproduction_v1/test_characterization.py
python studies/reproduction_v1/run.py plan
python studies/reproduction_v1/run.py verify
```

Run the characterization tests in a separate process because they intentionally load the preserved vendored numerical runtime. All reported fixtures are technical tests, never experimental evidence. The full new A40 workflow is prepared, but has not been executed locally. Its first cloud execution is still needed to establish complete GPU/runtime compatibility and measured duration.
