# Prospective study on RunPod

**Recovery release 0.3.1:** start with [RECOVERY_RUNPOD.md](RECOVERY_RUNPOD.md). The default profile is now `configs/prospective/bf16_revision.json` (BF16,64 structural test families). The old `full.json` is immutable historical protocol material. Diagnose locally before paid work. The CPU pipeline checks 800 positive sentinel datasets first, stops on any positive failure, then retains all complete calibration requirements. A sentinel pass alone never authorizes a model pilot or confirmation. CPU jobs on detected GPU hardware require explicit `--allow-cpu-on-gpu`; combined `full` additionally requires `--allow-combined-full`.


This guide runs the new `bsg-next` study. The historical FULL/AFS files, locked results, and legacy `runpod/bootstrap.sh` and `runpod/run_*.sh` launchers remain separate. No checkpoint inference, paid cloud job, or new scientific result was produced while building this runtime. CPU mock and randomly initialized tiny-model checks verify execution contracts; they do not measure the pretrained models in the study.

## Prepare a persistent workspace

Upload or clone this complete submission project into `/workspace/beyond-shared-geometry`. Put the private historical archive at `/workspace/beyond-shared-geometry/inputs/full.zip` when running retrospective checks. It is an input archive, not a model checkpoint. Keep the whole run directory, Hugging Face cache, and virtual environment on persistent storage that survives the pod being stopped. Confirm that your selected pod actually mounts persistent storage at `/workspace`.

Use a Python 3.10–3.13 image with an NVIDIA driver compatible with the selected Torch CUDA wheel. Python 3.11 is the recommended initial choice. The bootstrap defaults to Torch2.8.0's `cu128` wheel; `BSG_TORCH_INDEX_URL` can select another official Torch2.8.0 wheel index appropriate to your image. A successful package installation alone does not establish that CUDA inference works: run the GPU preflight below. No GPU-hour or final memory claim follows from the prompt counts.

Set these paths in the terminal, adjusting them to the actual upload location:

```bash
cd /workspace/beyond-shared-geometry
export BSG_VENV=/workspace/bsg-gpu-venv
export BSG_CONFIG=/workspace/beyond-shared-geometry/configs/prospective/bf16_revision.json
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/workspace/.cache/huggingface
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export BSG_RUN=/workspace/bsg-runs/icml2027-v1
```

Set `HF_TOKEN` through RunPod's secret/environment mechanism if a pinned checkpoint requires access. Accept any required model terms in your own Hugging Face account first. The scripts read the environment and do not print or save the token. Do not put credentials in JSON configurations, command arguments, source files, or result archives.

```bash
bash runpod/next_bootstrap.sh gpu
source "$BSG_VENV/bin/activate"
bsg-next validate
bsg-next plan
bsg-next preflight --require-cuda --check-access --output /workspace/bsg-preflight.json
python -m pytest tests/prospective -q
```

The bootstrap installs pinned top-level scientific dependencies and the editable project using the `prospective,gpu,test` extras. It does not download model weights. Its environment manifest and complete installed-package snapshot are saved inside the virtual environment as `prospective-environment.json` and `prospective-freeze.txt`. These files record transitive versions; the requirements files are not a platform-independent lock of every dependency. `next_bootstrap.sh cpu` installs the same interfaces with a CPU Torch wheel for offline checks.

Review `configs/prospective/bf16_revision.json` before starting. Checkpoint and tokenizer revisions are exact commit hashes; layers use raw transformer-block outputs. The structural study has six named models. The native study initially uses the two declared Llama/Qwen checkpoints and requires clean competence on the actual task. A replacement instruction-tuned checkpoint needs a new pinned configuration and a fresh validation/freeze before test outcomes. It is not pooled with an incompetent original pair.

## Run the ordered study

Use the **same** `BSG_RUN`, immutable source tree, configuration, seed, and override arguments for every phase of one study. Stage identities reject configuration or source changes. Do not edit the configuration after a pilot and reuse the same output directory. Register a new study directory when the protocol changes.

Start with a technical timing run in a separate directory. It cannot populate the paper:

```bash
bsg-next synthetic --mode timing --technical --limit 2 --output /workspace/bsg-timing-technical
```

The default configuration uses four spawned CPU workers with deterministic seed shards. Adjust `execution.cpu_workers` before the study, keep that configuration for every phase, and choose a worker count appropriate to the measured RAM/CPU budget. Each worker has a disjoint seed assignment; aggregation and confirmation validate all expected seeds. Keep BLAS thread counts under control when increasing process counts.

On a CPU machine with its separate CPU environment, run the complete CPU stage only after the bounded development check. This seals the maximum prompt/orbit pool before pilot outcomes and runs synthetic calibration, negative controls, crossed calibration, bootstrap sensitivity, and the historical archive diagnostics. The full CPU calibration is substantial; measure its timing before committing the remaining budget. Do not substitute the truncated timing output for formal calibration.

```bash
bash runpod/next_run.sh cpu "$BSG_RUN" --archive /workspace/beyond-shared-geometry/inputs/full.zip
```

Before this pilot, run the short hardware/architecture check in its own permanently technical output directory:

```bash
bsg-next model-smoke --execute-models --output /workspace/bsg-model-smoke
```

This loads the pinned models sequentially, reads construction-only prompts, checks real activation hooks and zero-delta scoring, and records timing/memory. It does not supply scientific effect estimates or use confirmation outcomes.

Run development extraction and the excluded pilot. The launcher requires `--execute-models` explicitly for every phase that can load checkpoint weights:

```bash
bash runpod/next_run.sh pilot "$BSG_RUN" --execute-models
```

Inspect the generated power reports, clean task competence, and frozen selections. A scientific gate can stop the study. Preserve the failed or inconclusive outputs. The runtime must not extract test outcomes merely because an editor exists or a process exited successfully. Native and structural test-family sizes are selected from the registered32/48/64-family prefixes by their own full decision procedures. The native planning alternative is a true ten-percentage-point advantage, with the five-point alternative reported as sensitivity; a five-point observed-effect gate is included in the simulation.

Only after the registered gates pass, run confirmation:

```bash
bash runpod/next_run.sh confirm "$BSG_RUN" --execute-models
```

Confirmation reuses construction caches and frozen validation choices. Native prepare and confirm share `outputs/native_study`; separate pipeline stage records do not create a fresh selection. All24 planned recipient cases per family are scored, including clean errors. The **primary native contrast uses only the eight long/transfer continuations of length6–8, with two paraphrases each**. The four familiar continuations per family are secondary. The excluded pilot uses the same long/familiar mix; validation competence is checked separately on both lengths, while editor/site/baseline selection uses familiar validation cases only. A final-answer patch is a positive control for the earlier intermediate-state intervention, whose target includes the model's remaining native program.

Run the extra comparison panels separately:

```bash
bash runpod/next_run.sh peers "$BSG_RUN" --execute-models
bash runpod/next_run.sh crossmodel "$BSG_RUN" --execute-models
bash runpod/next_run.sh extensions "$BSG_RUN" --execute-models
```

The native sensitivity panel also fits q=2/4/8 carriers at the frozen primary sites, using separate construction/validation fits before test outcomes; q=16 remains primary. The structural and archived-cache diagnostics additionally report a validation-selected local-displacement field as a secondary comparator. These extra conditions add real compute beyond the old 69,120-condition planning reference.

The common native benchmark already contains calendar-plane, conditional-mean, spline, and other registered comparators. The peer phase adds bounded original-task calendar anchors on their declared checkpoints and the complete state×first-operation×second-operation factorial panel. These are labeled according to their actual reproduction/adaptation scope. The spline is registered for C7; an unsupported noncyclic task is `N_A`, not a failed spline experiment. Wurgaft path diagnostics score actual intermediate path points, include an “other token” probability category, and separate the original complement asymmetry from a matched-complement comparison.

Cross-model word-field transport observes source-model test displacements as Hu's equation12 requires. It does not observe destination test displacements when predicting them. Refitted nulls, Gaussian RMS-matched controls, held-out intertwiner tests, and rank diagnostics are separate outputs. A Congruity claim additionally requires actual behavior-validated implementation families; one alignment matrix is not accepted as a proxy. Conditional training/extension stages require their own configured eligibility and should be interpreted using their returned gates, not as guaranteed positive findings.

`run --phase full` is an ordered convenience command for the applicable CPU, pilot, confirmation, peer, and cross-model phases. Supply the historical archive and explicit model-execution flag. The separate extension phase is intentional. For troubleshooting or partial work, use the individual commands below without claiming a complete study:

```bash
bsg-next native --mode prepare --execute-models --output /workspace/native-standalone
bsg-next native --mode confirm --execute-models --output /workspace/native-standalone
bsg-next peers --mode factorial --execute-models --output /workspace/factorial-standalone
bsg-next crossmodel --extract --execute-models --output /workspace/wordfields-standalone
```

An anchor requires an explicit `peers.anchor`, original checkpoint, and selected layers. The pipeline peer phase consumes the supplied `original_model_config` anchor bank; that bank is not a standalone study configuration accepted by `--config`. For a standalone anchor, create a full study configuration with `peers.anchor`, `peers.model`, and `peers.layers` explicitly set. Do not fall back to the first model in the structural registry. The output records the exact rendered prompt-bank hash and our implementation hash; exact author-code, prompt, and layer-convention equivalence is not certified. Do not run two writers concurrently against the same output directory. Use separate registered directories for independent configurations and calibration workers, then follow the documented merge/analysis entry points.

## Resume, inspect, and collect

Reissue exactly the same phase command after a disconnection. Completed stages are marked `RESUME`; failed stages retain their error records and resume only under a matching run identity. Native feature caches are content-addressed and checksummed. A changed source tree, prompt pool, model revision, configuration, or validation selection requires a new identity; do not delete the identity markers to bypass the check. A partially executed model-scoring stage may repeat work when restarted; stage-level resumption is not a promise of per-token checkpointing.

```bash
bsg-next report --run "$BSG_RUN" --output /workspace/bsg-report
bsg-next collect --run "$BSG_RUN" --output /workspace/icml2027-v1-results.zip
```

The collector writes a checksum alongside the ZIP. Its default bundle omits large activation caches and model weights; add `--include-caches` only when you intend the larger archive. Keep the persistent run directory until the copied ZIP, checksum, and scientific provenance have been checked. Preserve `prospective-freeze.txt`, `prospective-environment.json`, the GPU preflight, and runtime telemetry with your private execution records. Tokens and model checkpoint files must not be placed inside the result bundle.

Paper export is a separate, explicit operation against the new submission project:

```bash
bsg-next export-paper --run "$BSG_RUN" --project /workspace/beyond-shared-geometry
```

Technical runs are rejected. Missing required production results are rejected by default; `--allow-partial` is an explicit incomplete-reporting workflow, not evidence that the full study succeeded. Inspect the emitted tables, figures, numeric values, and claim limits before compiling or releasing a manuscript. Existing historical frozen tables are not replaced by prospective placeholders or technical results.

## Resource and failure checks

Use the preflight and measured pilot telemetry to choose batch size and hardware before freezing the production configuration. Keep `allow_cpu_offload` false unless an explicit revised protocol records otherwise. An out-of-memory failure should be diagnosed using real memory/sequence-length reports; do not silently enable quantization, truncate inputs, move layers to CPU, or change readouts halfway through a study. A changed batch/runtime policy changes the run identity.

The plan reports the structural activation-array floor; total storage also includes all checkpoint snapshots, native counterfactual caches, temporary merge arrays, calibration outputs, and optional archives. Check both persistent free space and local scratch space. Prompt counts do not equal forward counts: complete-answer scoring teacher-forces each candidate, runs an explicit first-token diagnostic prefill, and optionally performs greedy generation. Pathway donors and sensitivity/path panels add further real forwards. Use the emitted telemetry and method ledgers for actual costs.

If CUDA access, checkpoint permissions, task competence, calibration, or power fails, retain the corresponding negative/inconclusive branch and resolve it before confirmation. The runtime implements experiments and falsification controls; it cannot guarantee an effect, a unique mechanism, or conference acceptance.

## Predeclared conditional configurations

`configs/prospective/instruct_fallback.json` pins the two instruction-tuned checkpoints and enables their actual chat templates. Use a fresh standalone native prepare/confirm directory after a documented base-model competence failure:

```bash
bsg-next native --config configs/prospective/instruct_fallback.json --mode prepare --execute-models --output /workspace/bsg-runs/native-instruct-v1
bsg-next native --config configs/prospective/instruct_fallback.json --mode confirm --execute-models --output /workspace/bsg-runs/native-instruct-v1
```

`configs/prospective/third_lineage.json` pins Gemma-2-9B as a separate structural replication. Set `BSG_CONFIG` to that file and a fresh `BSG_RUN`, then repeat CPU, pilot and confirmation. Its native, peer and word-field branches are disabled; it does not replace the six primary checkpoints. Recheck model-provider access for this configuration before inference.

The private upload bundle contains `inputs/full.zip` when supplied by its builder. The public GitHub and Overleaf archives never include it. Full historical diagnostics verify its exact SHA before reading numerical arrays. A result collector is a private execution archive, not automatically an anonymous/public supplement.

RunPod storage semantics should be checked against the [official storage guide](https://docs.runpod.io/pods/storage/types): container disk and pod volume have different lifetimes; a network volume persists independently of a pod. Back up completed result bundles outside the pod. The [official SSH guide](https://docs.runpod.io/pods/configuration/use-ssh) distinguishes basic SSH access from a full public-IP SSH connection supporting SCP/SFTP; use the host/port shown in your own Connect panel.
