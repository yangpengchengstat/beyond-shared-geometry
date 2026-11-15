Beyond Shared Geometry — independent characterization v1
Study ID: icml2027-characterization-v1

Purpose and scope
-----------------
This is a new, pilot-informed independent holdout characterization, conditional
on the documented access history. It is not a continuation that changes an old
failed gate. The old structural power target was not met, and the native clean
competence gate failed. Neither finding is changed or relabelled here.

The readiness archive ac5038f1a4931e6cd765f159920f9dafa9ea7980af03e8f30b54d394f472e0c8
matches all three pinned parent files, the numerical source and all 18 original
development caches. It contains no recorded confirmation stage/freeze/artifact
in the named parent run. This is bounded evidence, not proof of no external
access. If contrary test-access evidence is known, stop and amend the study
before executing. The new run rechecks parent identities and access indicators.

Model execution is explicit and optional. --self-test never loads/downloads
weights, invokes a GPU job or opens the real sealed research holdout. The
launcher uses the existing pinned environment and locally cached weights.
It does not install packages or select newer model revisions automatically.

What runs
---------
The fixed six base models, four pinned readouts each, and q=16 train-only PCA
are retained. Existing development activations are verified and reused. New
holdout features are extracted once with all four layers in the same forward.
Source caches and frozen results are never edited.

Fresh rows per model:
  C7:                  896
  D4_corner:           512
  S3 test_joint:    12,288
  S3 test_context:  18,432
  S3 test_entity:    9,216
  S3 support:          384
  Total:            41,728 / model; 250,368 across six models.

The support block supplies four explicit adaptation examples per state/test
entity triple. It is separately charged, never enters PCA/operator fitting,
and is labelled in the support estimators' results. New reserved validation
cross blocks are excluded before extraction. New families are scenario draws
within the same fixed grammar roots; this is not unseen authored-grammar OOD.

The unchanged q=16 structural estimators retain prediction and context
endpoints, centroid/retention/potential controls and the affine, Procrustes,
residual, joint, generator and oracle-local secondary fits. Coherent-action
references remain descriptive. For S3, joint/context/entity strata are reported
separately with the registered source-embedding potential and support arms.

C7/D4 additionally compare the audited input-only full kernel, decoded-local
controls, generator-only composition and fixed training-support storage curves.
Bandwidths are selected only on validation; generator selection uses primitive
operations only. Composed predictions use their own intermediate outputs.
Support selection is label-free training-only farthest-first. Full_N retains
the original training order and reproduces the audited full-field baseline.
Coordinate counts cover shared source coordinates and all action displacements.
Storage matching is not total RAM, training-exposure or statistical-capacity
matching. All bandwidth boundary selections and all budgets are retained.

No new S3 kernel, native causal screen, synthetic calibration, historical refit,
cross-model study, product task, interface training or q sensitivity sweep is
included. These exclusions are explicit scope changes in study_config.json.
Do not describe this as completion of every historical/proposed experiment.

Legacy margins and alpha families remain in the provenance, but this run emits
descriptive estimates/family-orbit records, not powered confirmation or new
nonlinear significance certificates. No p-value is manufactured from model/
layer cells. A prespecified 10,000-replicate percentile bootstrap (seed 314159)
reports sensitivity intervals after averaging the fixed model/layer panel
within family; S3 independently resamples context and entity clusters. Intervals
are not multiplicity-adjusted new tests or a coverage certificate. Interval
interpretation and manuscript use require a subsequent audit; paper_eligible
remains false. These outputs alone cannot
identify the model's native algorithm or guarantee ICML acceptance.

Execution on the existing Pod
----------------------------
Upload bsg-characterization-v1.zip and its SHA256 file into /workspace.
Extract and launch once:

cd /workspace
sha256sum -c bsg-characterization-v1.zip.sha256
unzip -n bsg-characterization-v1.zip
mkdir -p /workspace/bsg-logs
nohup bash /workspace/bsg-characterization-v1/run.sh \
  > /workspace/bsg-logs/characterization-v1.log 2>&1 < /dev/null &

Read progress without following indefinitely:
tail -n 30 /workspace/bsg-logs/characterization-v1.log

The launcher runs model-free engineering checks first, then the actual study.
There is a cumulative six-hour software wall budget and an external six-hour
timeout with a termination grace period. This is a ceiling, not a runtime
promise or a current-price quote. No automatic restart or threshold relaxation
occurs. At 60-second intervals a heartbeat reports elapsed time/checkpoints.

A nonblocking lock prevents two workers writing the same study. Completed
extractions and cells are checked and reused; incomplete extractions and
copy/merge batches resume from checksummed progress. The cumulative budget
persists across invocations. An unclean termination conservatively counts
unaccounted wall time, including time until resume, against that budget.
After an interruption with budget remaining, use the same launch command once.
If the budget is exhausted or identity checks fail, preserve outputs and review
the cause rather than editing the contract or repeatedly relaunching.

Expected outputs
----------------
New run directory:
/workspace/bsg-runs/icml2027-characterization-v1

Portable results and checksum:
/workspace/bsg-runs/icml2027-characterization-v1.results.zip
/workspace/bsg-runs/icml2027-characterization-v1.results.zip.sha256

The portable ZIP contains cell results, fitted PCA, q16 coordinates for all
three domains, and metadata for private independent refits. S3 includes the
unique orbit/state input-embedding vectors used by the legacy potential
preprocessing, plus row mappings; no repeated multi-gigabyte embedding table
is exported. It excludes weights and full
hidden-state caches. Keep it private unless data publication is separately
reviewed. Raw heldout/merged caches remain in the new run for reproducibility.
The compact export supports q16 estimator refits; independent verification of
raw model forwards or refitting PCA in hidden space still needs those raw
caches or new extraction. Do not call the portable ZIP a full raw-data backup.
The archive is rebuilt on ordinary completion or a caught analysis/budget
error. A hard kill can interrupt export; no archive existence implies a pass.
Inspect summary.json status and verify SHA256 before downloading/deleting any
remote copy. There is no automatic Pod termination or data deletion.

If a hard kill interrupted final export, collect existing files without model
execution (only after the worker has stopped):
/workspace/bsg-gpu-venv/bin/python /workspace/bsg-characterization-v1/run_study.py --collect-existing
Collection requires the same frozen bundle, refuses an active writer, and
labels the recorded completion state; packaging does not certify success.
Final sensitivity summaries and export may use the termination grace period;
the external timeout bounds an invocation. Budget accounting is saved again
after ordinary export.

Validation and limitations
--------------------------
CPU fixtures cover legacy estimator replay, operation order, test-outcome-blind
hyperparameter/support selection, family separation, storage arithmetic, all
planned block counts on a distinct toy seed, interrupted extraction and merge
resume, exact copying, immutable inputs and cache hits without model execution.
The new core also reproduced all 48 existing excluded-pilot generator/full-field
endpoints from exported coordinates. No real independent test, Torch fit on
the full new dataset or complete A40 run has yet been executed locally.

source_manifest.json locks this new code/config and unchanged numerical source.
The embedded bsg_next source retains its original license and is reused without
modifying the installed/published project. Official ICML 2027 formatting and
scientific novelty are not certified by this execution bundle.
