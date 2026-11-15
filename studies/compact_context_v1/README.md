# Compact nonlinear predictors and context transport v1

This is a **new CPU-only exploratory study** using the already completed characterization coordinates. It does not run any language model, repeat synthetic calibration, restart a failed gate, or edit historical results. Previously inspected test partitions remain exploratory. No result is automatically inserted into the manuscript or certified as confirmatory.

## Input and scope

The one required data file is **`icml2027-characterization-v1.results.zip`**, SHA-256:

```
79b1c53900cfcd91f0644b3db8405160213785487d0f7cea8b0320244a5aec0a
```

This is the 270,401,520-byte **results archive**, not the earlier `bsg-characterization-v1.zip` deployment package. The archive contains all 72 q16 coordinate cells and paired row metadata needed here. The runner reads it in place, verifies SHA-256, ZIP CRC, the exact panel, finite coordinates and partition/action integrity. No unpacked raw activation cache, Hugging Face token, Torch, GPU, model weights, or old calibration output is required.

Every cell keeps the original training/validation/test roles. The S3 reserved and support rows never enter fitting, selection or primary scoring. S3 test_joint, test_context and test_entity remain separate. The initial fixed development benchmark physically excludes these tests, support and reserved coordinates before fitting/evaluation; it evaluates only the excluded pilot split.

## Fixed comparisons

C7/D4 each have two information conditions: all nonidentity elements fitted directly, and primitive generators only. Generator composition repeatedly edits its own predicted coordinates. The centroid baseline re-decodes the current prediction at each step.

Methods are independent linear ridge, affine ridge with charged bias, decoded centroid displacement with its extra construction labels/action table declared, limited-support Gaussian displacement fields, and **RFF and Nyström residual regressors fitted to every construction target pair**. Fixed incremental storage budgets are 0.5, 1, 2, 4 and 8 times the identity-pruned linear maps. Each compact predictor includes normalizers, feature parameters, coefficients, biases and numeric lookup tables in its accounting; infeasible budgets are explicit. Reported logical serialized parameter bytes equal the float64/int64 payload, not compressed ZIP size, Python object overhead or peak RAM. The common q16 PCA is shared and excluded from incremental predictor storage.

Context comparisons use paired differences `u = z(p0)-z(p1)` and **uncentered training difference energy**. They include identity, per-action validation-selected scalar retention, the coordinate-fitted linear operator applied to differences, directly fitted linear difference transport, decoded-category and explicitly oracle-category retention, and full-data RFF/Nyström difference predictors. The decoded category arm receives the paired source center in addition to the difference; the oracle arm receives the true source category. These extra-information arms are labeled separately.

Both components compare the semantic action to a coherent alternative obtained by conjugating every action table by the fixed state swap `(0,1)`. This preserves group relations and fixed-point counts, and changes the represented action set for C7/D4/S3. It is a consistently supervised alternative, not an independent-label scramble or a null p-value.

All changed-state actions, model/readout cells, budgets, failures and invalid predictions are reported. Outputs include selection ledgers, storage counts, family/orbit summaries, cycles, and D4 endpoint-discriminating order comparisons. A validation-only comparability ledger requires error and output squared norm to agree within the fixed factor 1.25, and linear coefficient Frobenius norm when available. It retains unmatched records too. This is approximate descriptive comparability; it does not match nonlinear Jacobian norms or establish semantic identification.

## Run on RunPod

Upload this source ZIP and its SHA-256 file to `/workspace`. Reuse the original results ZIP at `/workspace/bsg-runs/icml2027-characterization-v1.results.zip`; if it is absent, upload the saved local copy to that exact location. Use a CPU instance for this stage, or the existing Pod's CPU if you intentionally keep that GPU Pod rented. The latter still incurs the Pod's normal charges.

First prepare a **separate environment**; leave the previous study's environment and source alone:

```bash
cd /workspace
sha256sum -c bsg-compact-context-v1.zip.sha256
unzip -n bsg-compact-context-v1.zip
python3 -m venv /workspace/bsg-compact-cpu-venv
/workspace/bsg-compact-cpu-venv/bin/python -m pip install -r /workspace/bsg-compact-context-v1/requirements.txt
```

Then run the development benchmark and complete fixed panel consecutively:

```bash
mkdir -p /workspace/bsg-logs
nohup env BSG_CPU_PYTHON=/workspace/bsg-compact-cpu-venv/bin/python \
  bash /workspace/bsg-compact-context-v1/run_cached.sh \
  /workspace/bsg-runs/icml2027-characterization-v1.results.zip \
  /workspace/bsg-runs/icml2027-compact-context-v1 2 \
  > /workspace/bsg-logs/compact-context-v1.log 2>&1 < /dev/null &
```

The final `2` means two CPU workers, each with one numerical-library thread. There is **no three-hour timeout**. The wrapper first runs five fixed excluded-development jobs, then all 240 component/registration/cell jobs. It does not use pilot outcomes to change grids or select a favorable model. A technical exception stops the wrapper and preserves completed checkpoints.

Check progress without entering an ongoing `tail -f` session:

```bash
tail -n 30 /workspace/bsg-logs/compact-context-v1.log
```

Logs show `START_JOB`, `DONE_JOB`, and a heartbeat every approximately 30 seconds while waiting. Full state is also available once the full phase starts:

```bash
/workspace/bsg-compact-cpu-venv/bin/python /workspace/bsg-compact-context-v1/run.py \
  --mode status --output /workspace/bsg-runs/icml2027-compact-context-v1
```

To resume after an interruption, repeat the same launch command. A lock prevents simultaneous writes. Matching atomic checkpoints are verified and reused; changed source/protocol/NumPy/Python/input identities require a fresh output path. A job interrupted before its checkpoint is committed is recomputed; completed jobs are not. This is new-study checkpoint reuse, not an old calibration restart.

If a scheduling budget is wanted, invoke `run.py` directly with `--max-seconds NUMBER`; this is a per-session budget, default zero (unlimited). It stops scheduling new jobs, lets in-flight jobs finish, and exits with code 75. It is not a hard billing limit and does not stop/delete a Pod.

Completion prints:

```
RESULT_ARCHIVE /workspace/bsg-runs/icml2027-compact-context-v1.results.zip
```

Download that archive and its `.sha256` file. Also retain the small `.pilot.results.zip` and checksum for the runtime ledger. Before collection, every checkpoint is reverified; the resulting ZIP is CRC-tested and hashed. Downloaded results still require scientific review. Do not rerun simply to obtain favorable scores.

## Run locally instead

The same source runs on macOS/Linux with Python 3.10+ and NumPy 2.2.6. Set `BSG_CPU_PYTHON` to a dedicated Python interpreter and invoke `run_cached.sh` with the local input and fresh output paths. It needs only CPUs. Avoid changing any old experiment environment. The public source does not contain private coordinates; transfer the original input separately if moving machines.

## Preserve data before ending a Pod lease

The compact archive is sufficient for **these two analyses**, so retaining an A40 solely for them is unnecessary. Closing a browser window leaves the Pod running and billable. Confirm that no wanted job is active, then end compute through the RunPod console when ready.

Your earlier container-only `/workspace` configuration loses data on stop/restart. Before stopping/terminating, retain any remote-only materials needed for future raw/forward verification. In particular, the characterization run's `merged/` tree contains full hidden activations absent from the compact archive. The parent development cache, datasets, original source/configs and logs are useful for replay. Pinned model weights can be downloaded again; do not put Hugging Face credentials in an archive.

RunPod documents the distinct lifetimes of [container, volume and network storage](https://docs.runpod.io/pods/storage/types). This program never changes a Pod or storage subscription.

## Verification and limitations

`python -m pytest tests -q` runs bounded model-free regression tests (pytest is a development dependency, not required to run the experiments). They exercise budget accounting, all-pair training exposure, repeated centroid decoding, metadata/split separation, group consistency, context energy, source/input hashes, checkpoint corruption and result collection.

The protocol is frozen in `protocol.json`; `source_manifest.json` identifies this package's exact source. `package.py` creates a source-only ZIP. The parent study is immutable. This stage does not reproduce high-dimensional model forwards/PCA fitting, native causal interventions, peer papers in their original setting, a fresh confirmatory holdout, or support-adapted S3 estimators. Synthetic fixtures are engineering checks and are never substituted for model results.
