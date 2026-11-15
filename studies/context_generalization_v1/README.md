# Separately constructed context-transfer evaluation

This is a prepared, opt-in study, not an executed result. It compares frozen
predictors on a new fixed bank of narrative contexts. It does not repeat old
FULL/AFS experiments, calibration, native-task screening, or model training.
The current implementation has software tests using generated arrays and a
mock feature runtime; successful real A40 execution has not yet been verified.

## Scientific contract

- Six pinned models and four original readouts per model, C7 and D4: 48 cells.
- Original rank-16 PCA center and basis; no fitting or centering on new prompts.
- Old 48 training families fit each predictor. Old 16 validation families select
  penalties from `[0, 1e-5, .001, .1, 1]`, separately per nonidentity action.
- The two **primary** arms train on state coordinates or paired context
  differences. Both choose their penalty using the **same old-validation
  context-difference loss**. A coordinate-validation-selected arm and identity
  are secondary controls. Thus primary selection does not confound different
  validation objectives with training objectives.
- The new bank has 64 explicitly written paired frames. Each has all seven
  weekdays and all four corner states, for 1,408 prompts per model. The factual
  sentence is fixed within each domain; outer narratives are new. This is not
  unseen vocabulary, independent core grammar, a reasoning task, or a random
  sample of natural language.
- Frames were written with an assistant. A person must review their meaning
  before execution. The code never labels assistant authorship as independent
  human authorship and never signs a review automatically.
- Prediction loss is squared context-difference error divided by the old-train
  uncentered mean squared context difference. Each family averages changed
  states within each nonidentity action, then equally averages actions. The
  primary contrast is `difference - coordinate`: negative favors direct
  difference fitting. Average the 24 fixed model/layer cells per domain and
  then the two domains equally, leaving 64 paired family contrasts.
- The 2,000-draw paired-family bootstrap (seed 2026102901) is conditional
  sensitivity across this authored bank. It is not calibrated population
  coverage, a significance decision, or the spectral theorem's certificate.
- No test-based layer, rank, penalty, action or family selection. Missing cells
  stay visible and prevent a complete primary-panel aggregate.

These are related, nonidentical contexts constructed separately from the old
prompt generator. Exact prompt disjointness is verified against every row in
the archived C7/D4 coordinate banks. This does not prove semantic independence
or remove design-level dependence among the fixed frames.

## Requirements and input

Use the project's GPU environment and its pinned Hugging Face revisions. The
input is the previously verified private `from-zero-v1-rank-recovery.results.zip`
and its trusted SHA-256 sidecar. The source release does not contain that private
archive or model weights. An aggregate-results CSV alone is insufficient.

Preparation reuses the hardened spectral input adapter to verify the complete
archive, source contract, model revisions, extraction identities, PCA and row
hashes. It saves only the old train/validation coordinates and PCA needed by
this study. Old test coordinates never enter fitting or penalty selection.
Every production phase checks the study's source manifest and inherited
numerical dependencies. Editing a protocol requires a new version and output
namespace; do not regenerate a manifest merely to silence an unexplained error.

## Run from the project root

The following is an execution guide, not a record that these commands ran.
Use a persistent or backed-up location. Paths below are ordinary RunPod paths.
All commands use the Python in your activated project environment.

### 1. Prepare and inspect without models

```bash
export BSG_ARCHIVE=/workspace/from-zero-v1-rank-recovery.results.zip
export BSG_RUN=/workspace/bsg-runs/tmlr-context-v1
# Copy the digest from the already trusted .sha256 sidecar, not from an unknown ZIP.
export BSG_ARCHIVE_SHA256=PASTE_THE_VERIFIED_64_CHARACTER_SHA256

python studies/context_generalization_v1/run.py prepare \
  --archive "$BSG_ARCHIVE" --expected-sha256 "$BSG_ARCHIVE_SHA256" \
  --output "$BSG_RUN"
```

This creates a fresh directory, verifies all 48 cells, and writes `bank.json`
and `BANK_REVIEW.txt`. It does not fit predictors or run any model. Inspect the
review text: all 64 frames for each domain, both renderings, preserved factual
meaning, unique target spans, and no competing weekday/corner statement.
`bank.json` contains all state substitutions. Preparation prints
`PREPARED_BANK_SHA256` for the exact bank that was inspected.

### 2. Record your actual semantic review

Only after that review, copy the printed bank hash:

```bash
python studies/context_generalization_v1/run.py review \
  --run "$BSG_RUN" --bank-sha256 PASTE_THE_REVIEWED_BANK_SHA256 \
  --attest-reviewed
```

The receipt is an explicit operator attestation, not an authenticated digital
signature or proof of independent authorship. If a frame needs revision, do not
approve it: revise the protocol in a new source release before any extraction.

### 3. Fit and freeze using old development data only

```bash
python studies/context_generalization_v1/run.py fit \
  --run "$BSG_RUN" --execute-analysis
```

This CPU stage records every candidate validation loss, selected penalty, fitted
matrix and normalization energy. All 48 predictor sets must be frozen before
new feature extraction is permitted. No new test output is used. Completed
checksummed fit cells are reused on an identical rerun.

### 4. Extract new prompts once, sequentially on one GPU

```bash
export HF_HOME=/workspace/bsg-model-cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p /workspace/bsg-logs

nohup python -u studies/context_generalization_v1/run.py extract \
  --run "$BSG_RUN" --execute-models --allow-download \
  > /workspace/bsg-logs/tmlr-context-v1.log 2>&1 < /dev/null &
```

`--allow-download` permits downloads only for pinned model revisions and may
require your own Hugging Face access. Omit it to require an already complete
local cache. Never put a token in source files or these logs.

The runtime is required to match the archived extraction identity, including
precision, attention, tokenization, batch size, readout convention and dependency
base versions. A mismatch writes `environment_mismatch.json` and stops before
loading weights. Install the recorded versions rather than weakening the check.
The original backbone remains unchanged; only new prompts are encoded.

Monitor with:

```bash
tail -n 40 /workspace/bsg-logs/tmlr-context-v1.log
```

Each model is loaded in sequence. Existing verified feature chunks and completed
PCA projections are resumed by rerunning the **same extraction command**; there
is no three-hour timer. A cache identity or checksum mismatch stops execution.
Do not delete failure records to force a resume. The Pod stays billable while
running; this script never shuts it down or starts another Pod.

### 5. Score the frozen predictors

After extraction finishes:

```bash
python studies/context_generalization_v1/run.py analyze \
  --run "$BSG_RUN" --execute-analysis
```

The primary outputs are `summary.json`, `cell_summary.csv` and per-family records
in `evaluations/`. Inspect every cell status. `incomplete_panel` is not a pass.
Completed analysis is immutable and reused after checks of both results and
their test-coordinate receipts. If analysis encounters bad/missing test inputs,
the full incomplete attempt is preserved in `analysis_attempts/`. After the
same frozen extraction finishes, rerun `analyze` to evaluate the complete panel;
an early incomplete attempt does not lock the result permanently. Changed
source, input or protocol identities still require a new namespace. Results
are never copied into the paper by this command.

### 6. Collect and verify the private result package

```bash
python studies/context_generalization_v1/run.py collect \
  --run "$BSG_RUN" --output /workspace/tmlr-context-v1.results.zip
cd /workspace
sha256sum -c tmlr-context-v1.results.zip.sha256
```

Download both the ZIP and SHA-256 sidecar before removing remote storage. The
package includes compact new coordinates, frozen fitted matrices, source/input
identities, prompt bank, review receipt, failures and results. It also includes
the readable per-cell original runtime identities and per-model extraction
manifests. These manifests are metadata only: their referenced raw feature
chunks are excluded, so this is not a complete raw extraction cache.
Member hashes and artifact-validation diagnostics are in `export_manifest.json`.
The diagnostics check fitted predictors, new-coordinate receipts and completed
analysis receipts against their hashes and frozen bindings. `verified` refers
only to those checks, not to a scientific pass. Missing or damaged completed
artifacts are reported as `missing` or `invalid` while their available records
are still exported for inspection; collection does not repair or endorse them.
Prepared-input identity checks still apply before collection.
The package excludes raw high-dimensional hidden arrays,
old train coordinates, credentials and model caches. It is a **private audit
package**, not an automatic GitHub upload. Retain the original input archive
separately; a compact export is not a replacement for it.

## Software validation

```bash
python -m unittest discover -s studies/context_generalization_v1/tests -v
```

Tests use synthetic arrays and a mock feature provider. They check complete
paired state blocks, target spans, no test influence on fits, matched selection
objectives, D4 aggregation weights, paired-family sensitivity, review identity,
source/artifact protection, explicit execution flags, private result export, and
an end-to-end 48-cell mock fit/extract/resume/analyze sequence, recovery from
premature analysis, detection of post-analysis coordinate mutation, readable
runtime metadata in compact exports, and flagged preservation of damaged
completion receipts. They do not import
model weights, run calibration, estimate scientific effects, or certify a real
GPU runtime. Existing analytic spectral tests remain in the separate study.
