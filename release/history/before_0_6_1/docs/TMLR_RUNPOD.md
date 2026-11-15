# Running only the new TMLR follow-up on RunPod

The cached spectral analysis is CPU-only and can run locally. Do not rent a GPU
for that step, the spectral-extension studies, or the controlled simulations.
Do not repeat the earlier calibration. The new-context extraction step
is the only new workflow stage that loads pretrained language models.

Use a fresh workspace and one model at a time. The fixed panel's largest model is
8B parameters; a single A40 is the intended execution target. Actual memory use
must be checked during the first extraction; no runtime or disk-space guarantee
is implied. Model caches, private inputs and results consume disk in addition to
the source package. Check `df -h /workspace` before downloading models.

## 1. Upload and install

Upload the new GitHub source ZIP, its SHA-256 sidecar, and the existing verified
rank-recovery results archive with its original sidecar. Verify checksums before
extracting. Do not replace an older project directory or delete existing evidence.
In the new project directory:

```bash
python3.11 -m venv /workspace/bsg-tmlr-venv
source /workspace/bsg-tmlr-venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r studies/reproduction_v1/requirements.in
python -m pip install --no-deps -e .
export HF_HOME=/workspace/bsg-model-cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

If gated Llama access is needed, use `hf auth login` interactively. Never paste a
token into source files, logs, commands sent to another person, or the paper.
Read-only token access and approval for the exact gated model are separate
requirements. Existing approved model caches can be reused at the pinned revision.

The requirements file fixes the inherited top-level numerical/runtime versions,
including Torch 2.8.0+cu128. It is not a complete transitive lock. Keep the collected
environment record, and inspect any `environment_mismatch.json` rather than
weakening its checks or silently mixing extraction precisions.

## 2. Prepare and inspect before model execution

From the new project root, set the actual private archive path and its verified
original checksum. The run directory below must be new:

```bash
BSG_INPUT=/workspace/from-zero-v1-rank-recovery.results.zip
BSG_INPUT_SHA256=6a4c38237bf1368d4c77c97504f666d0d933bfebed9417d173fbf7441c0fecef
BSG_RUN=/workspace/bsg-runs/tmlr-context-v1
python studies/context_generalization_v1/run.py prepare \
  --archive "$BSG_INPUT" --expected-sha256 "$BSG_INPUT_SHA256" \
  --output "$BSG_RUN"
```

Preparation verifies the old archive, exports the allowed old training/validation
coordinates and PCA, and freezes the new text bank. It does not fit a predictor or
load a model. Read the produced bank and review instructions in the study README.
After actually inspecting all new families, copy the printed bank checksum:

```bash
BSG_BANK_SHA256=PASTE_THE_PREPARATION_BANK_SHA256
python studies/context_generalization_v1/run.py review \
  --run "$BSG_RUN" --bank-sha256 "$BSG_BANK_SHA256" --attest-reviewed
```

This is a record of actual human semantic review. Do not run it automatically as
part of a shell chain or use it to bypass unreviewed or incorrect text. No historical
FULL, AFS, calibration or native-program launcher is part of this path.

## 3. Fit, extract, analyze, collect

Freeze predictors from old training/validation data only:

```bash
python studies/context_generalization_v1/run.py fit \
  --run "$BSG_RUN" --execute-analysis
```

The primary coordinate and difference methods use the same old difference-loss
validation criterion. Old coordinate-loss selection is a separate secondary arm.
No test score chooses a hyperparameter, layer or registration.

Launch extraction explicitly. This example permits downloading the pinned model
revisions when absent; omit `--allow-download` to require an existing local cache:

```bash
mkdir -p /workspace/bsg-logs
nohup python -u studies/context_generalization_v1/run.py extract \
  --run "$BSG_RUN" --execute-models --allow-download \
  > /workspace/bsg-logs/tmlr-context-extract-v1.log 2>&1 < /dev/null &
```

Check progress in a separate terminal:

```bash
tail -n 40 /workspace/bsg-logs/tmlr-context-extract-v1.log
nvidia-smi
df -h /workspace
```

After extraction finishes successfully, analyze and collect:

```bash
python studies/context_generalization_v1/run.py analyze \
  --run "$BSG_RUN" --execute-analysis
python studies/context_generalization_v1/run.py collect \
  --run "$BSG_RUN" --output /workspace/tmlr-context-v1.results.zip
```

Follow the study README for the exact completion records and supported resume
behavior. Keep failures and partial records. Never start two controllers on the
same run. Changed input/source/protocol identities require a new run directory;
verified completed extraction work may be reused only under matching identities.

## 4. Before terminating the Pod

Collect the complete study result archive and its checksum, download both, and
verify the checksum on the receiving machine. Confirm the archive contains the
run manifest, bank/review identity, all required compact coordinate receipts,
analysis outputs, failures and logs needed for audit and continuation. The model
cache need not be downloaded if weights can be fetched again. Raw or compact data
needed for unfinished work must not be discarded without a verified backup.

A container disk may be erased when the Pod stops or is terminated. This project
does not stop the Pod automatically; inspect the provider's current lifecycle and
billing state yourself after verifying the downloaded results. Do not delete old
research archives merely because the new workflow finished.
