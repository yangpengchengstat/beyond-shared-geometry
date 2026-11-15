# One-Pod TMLR follow-up launcher

This additive execution package runs the five prepared TMLR follow-ups from the
unchanged 0.5.0 source release. It does not change the paper, scientific source,
protocols, model revisions, original evidence or failure records. It does not
launch or terminate a paid Pod. Research execution has not been performed by
building this package; controller tests use fixtures only.

## Scope and order

After preparation and actual human bank review, a single explicit run command
executes: (1) old-data predictor fitting, new-context extraction on six models,
and context analysis; (2) original-bank spectra; (3) exact-action witnesses;
(4) the 36 fixed mechanism simulations; (5) 2,952 alternative-image units. The
first stage loads one model at a time. Remaining stages use CPU. A GPU Pod can
execute both CPU and GPU work, but its GPU rental remains billable during CPU
work. The convenience of one Pod does not minimize cost.

"All experiments" means these five new supplements. No historical FULL/AFS,
old calibration, native-screening or reproduction launcher is invoked. Simulation
cases are the new declared mechanism illustrations, not old calibration.

There is no three-hour timeout. Finishing the program is distinct from a
scientifically favorable result. Numerically not-evaluable spectral/extension
units remain in the output; they do not become successful measurements. A
technical error stops the queue with logs and partial evidence retained.

The context bank remains the released assistant-authored outer-narrative bank
with fixed factual clauses and known state vocabulary. It is not independently
authored core syntax or a new semantic domain. Strengthening that protocol needs
a separate version before model outputs; this launcher does not silently do so.

## Upload four files

1. `bsg-tmlr-all-v1.zip`
2. `bsg-tmlr-all-v1.zip.sha256`
3. The preserved `from-zero-v1-rank-recovery.results.zip`
4. Its original `.sha256` sidecar.

Only the code bundle is supplied here. The private 3.18 GB input stays separate.
The six pinned model snapshots total approximately 49.51 GB. Allow room for the
Python/CUDA environment, downloads, new arrays, CPU analyses and export copies.
A 200 GB container disk is a practical planning choice for the combined run,
not a verified peak-space requirement. Check actual free space. The controller
checks for 10 GiB free before stages and enough uncompressed-file-size headroom
before collecting; disk consumption within a long stage can still exhaust space.
No automatic data/cache deletion is performed. Keep old private evidence locally.

## 1. Verify and unpack into a fresh directory

In the Pod terminal, after uploading to `/workspace`:

```bash
cd /workspace
sha256sum -c bsg-tmlr-all-v1.zip.sha256
sha256sum -c from-zero-v1-rank-recovery.results.zip.sha256
unzip -n bsg-tmlr-all-v1.zip -d /workspace/tmlr-all-v1
df -h /workspace
```

Both checks must say OK. If `/workspace/tmlr-all-v1` already contains a different
version, use a fresh directory and adjust paths consistently. Do not merge code.

## 2. Install the inherited numerical/GPU runtime

Use Python 3.11 and a CUDA-capable PyTorch Pod. In the extracted project:

```bash
cd /workspace/tmlr-all-v1/beyond-shared-geometry
python3.11 -m venv /workspace/bsg-tmlr-venv
source /workspace/bsg-tmlr-venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r studies/reproduction_v1/requirements.in
python -m pip install --no-deps -e .
export HF_HOME=/workspace/bsg-model-cache
export HF_HUB_ENABLE_HF_TRANSFER=0
hf auth login
nvidia-smi
```

The token is entered interactively. The exact three Llama base repositories need
access approval; token validity alone does not grant it. Do not install different
runtime versions after preparation. Source and recorded runtime mismatches stop
instead of silently changing extraction precision or scientific definitions.

Optional early access check downloads only six small configuration files:

```bash
python - <<'PY'
import json
from huggingface_hub import hf_hub_download
with open('studies/spectral_finite_bank_v1/protocol.json') as f:
    models = json.load(f)['models']
for m in models:
    hf_hub_download(m['id'], 'config.json', revision=m['revision'])
    print('ACCESS_OK', m['id'], flush=True)
PY
```

Stop and resolve any access failure before starting the long queue. Full weights
are downloaded as needed by extraction and stored in the shared HF cache.

## 3. Prepare the new bank without computing results

```bash
bash /workspace/tmlr-all-v1/tmlr_batch_v1/start.sh prepare \
  --source /workspace/tmlr-all-v1/beyond-shared-geometry \
  --archive /workspace/from-zero-v1-rank-recovery.results.zip \
  --run /workspace/bsg-runs/tmlr-all-v1
```

This reads and hashes the old archive, checks all inherited source locks, saves
a source snapshot, and prepares the new-context bank. It loads no pretrained
model and runs no spectrum/fit/simulation. It prints `REVIEW_FILE` and `BANK_SHA256`.
Open the exact printed `BANK_REVIEW.txt` (also available through Jupyter's file
browser). Actually review all 64 paired frames for each domain, their meaning,
target spans and absence of competing state statements. The full bank is in
the adjacent `bank.json`. The code does not perform or attest human review.

After completing that review, replace the placeholder with the printed hash:

```bash
bash /workspace/tmlr-all-v1/tmlr_batch_v1/start.sh review \
  --run /workspace/bsg-runs/tmlr-all-v1 \
  --bank-sha256 PASTE_THE_REVIEWED_BANK_SHA256 \
  --attest-reviewed
```

Do not run this command as an automatic installation step. A needed textual
revision requires a new reviewed study version; do not edit a frozen bank or
regenerate its manifest to bypass a failure.

## 4. Start the whole queue once

```bash
mkdir -p /workspace/bsg-logs
nohup bash /workspace/tmlr-all-v1/tmlr_batch_v1/start.sh run \
  --run /workspace/bsg-runs/tmlr-all-v1 \
  --execute-models --execute-analysis --allow-download \
  >> /workspace/bsg-logs/tmlr-all-v1.log 2>&1 < /dev/null &
```

No interactive input should be required after review and valid model access.
The background process survives closing the terminal, provided the Pod itself
continues to run. No fixed total-runtime promise is made. Do not stop/delete the
Pod before the result package is downloaded and verified.

## 5. Progress, continuation, and preserved failures

```bash
tail -n 35 /workspace/bsg-logs/tmlr-all-v1.log
bash /workspace/tmlr-all-v1/tmlr_batch_v1/start.sh status \
  --run /workspace/bsg-runs/tmlr-all-v1
```

Heartbeats are every 30 seconds. Status includes recent active-stage log lines
and extension completed/planned counts. Per-attempt stage logs are under the
batch directory's `logs/`; failures are under `failures/`.

After an interruption, inspect the error and resolve its cause, then use the
same run command. Only one controller can own the batch. Completed stages are
skipped only after saved output checksums match. Context fit/extraction/analysis
and extension runs delegate to their original resume code if incomplete.
Extension receipts retain completed and not-evaluable units. The original
finite-bank script has no per-cell resume: if interrupted, its 48-cell analysis
is retried in a NEW attempt directory and the old partial output is preserved.
Preparation interruption should be inspected and restarted with a fresh batch
namespace; do not erase the failed one or label it completed.

Source/protocol changes require a new run. A runtime mismatch requires restoring
the recorded environment, not weakening the check. The collection command can
preserve a stopped run without changing its status.

## 6. Collect and download

Successful execution automatically writes a timestamped results ZIP and SHA-256
sidecar next to the batch folder, and prints `RESULT_ARCHIVE` followed by
`ALL_STAGES_FINISHED`. The final state is `completed_pending_scientific_review`.
It never means that every scientific hypothesis passed or the paper is ready.

If export failed for space, free additional space without deleting evidence and
retry only collection. The same command also exports partial evidence:

```bash
bash /workspace/tmlr-all-v1/tmlr_batch_v1/start.sh collect \
  --run /workspace/bsg-runs/tmlr-all-v1
```

Download both printed files. From their containing directory use
`sha256sum -c <exact-results-filename>.sha256` on Linux, or
`shasum -a 256 -c <exact-results-filename>.sha256` on macOS. Only after the local
copy verifies should remote storage be released. Keep the original input ZIP
separately. The new export includes the batch's prepared inputs, fits, new raw
span-feature arrays, compact projections, per-stage spectra, input/source
identities, source snapshot, review receipt, attempts and failures. It excludes
the old 3.18 GB archive itself, model weights and external Hugging Face cache.
The export is private audit material, not a public GitHub upload.

The launcher does not stop the Pod or erase storage. Its final message explicitly
reminds the operator that the Pod remains running and billable.

## Validation and provenance

`python -m unittest discover -s tmlr_batch_v1 -p 'test_batch.py' -v` runs controller
fixtures only. The delivered validation record covers all-stage ordering,
completed-stage reuse, incomplete-stage continuation, corruption rejection,
review enforcement, runtime identity, process locking and export hashes. The
inherited 51 source/dependency files were checked against existing manifests.
No real model extraction or empirical five-study execution was performed.
