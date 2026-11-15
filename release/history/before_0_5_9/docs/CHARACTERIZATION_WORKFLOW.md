# Current study and result-return workflow

Updated 21 October 2026. This is the current entry point. All local paths below are relative to this publication project, `submission/icml2027`, unless explicitly shown as RunPod paths. No command on this page provisions a Pod or starts a new calibration.

## 1. Identify the study before acting

Current study: **`icml2027-characterization-v1`**, a pilot-informed independent holdout characterization with limited precision. Its source ZIP SHA-256 is `ebbc9915ee622bd19c26228c78d0167f89acb2f073553cb54c4ca2dec29c2c89`; the expanded, byte-identical source is under `studies/characterization_v1/frozen_run/`. Its source manifest SHA-256 is `5832afe9f3d31b4962ab30ba6c717f151bcb99a14b23ae25f146108a3396f739`.

The previous identification study's revised calibration and structural pilot completed. Structural power was limited and native preparation stopped at clean competence. The input-only and generator field comparisons were excluded development analyses. They informed this new contract; they cannot serve as its independent evaluation. Original FULL/AFS archives, earlier failures and published source are retained unchanged.

The fixed six base models, four layers/model and train-only q=16 PCA remain. The study extracts 41,728 fresh rows/model, 250,368 total, including separately charged S3 support. It evaluates C7/D4 all-element fields, primitive-only generator composition, cycles/order and coordinate-storage curves; S3 joint/context/entity strata are separate. This run does not include a new S3 kernel, native intervention confirmation, peer intervention comparisons, cross-model transport, product/interface/third-lineage extensions or q sensitivity. None is silently recorded as completed.

## 2. Completed return and historical progress commands

The returned archive has completed 72 cells, 120 legacy split records and 48 field records. Its SHA-256 is `79b1c53900cfcd91f0644b3db8405160213785487d0f7cea8b0320244a5aec0a`. Supported analytic coordinate calculations and all 56 frozen sensitivity intervals were independently checked. The manuscript now uses curated descriptive aggregates under `results/characterization_v1/`, traced by `release/characterization_review.json`. Raw forwards/PCA and joint/potential optimization are not independently reproduced. These completed local findings do not establish the current remote Pod state.

The following commands are retained for historical recovery and future inspection; they are not a request to restart the completed study.

Do not launch a duplicate worker. A checksum and `unzip` only verify and extract a source package; a trailing `&&` waits for another command. The current launch entry is `run.sh` inside the source package. If a worker is already active, do not reissue its launch.

Read progress without staying attached to a live follower:

```bash
tail -n 40 /workspace/bsg-logs/characterization-v1.log
ps -eo pid,etime,pcpu,stat,args --sort=-pcpu | head -n 12
```

A finished ZIP or successful checksum is insufficient to establish that all scientific cells finished. The private importer checks completion independently. Expected returned files:

```text
/workspace/bsg-runs/icml2027-characterization-v1.results.zip
/workspace/bsg-runs/icml2027-characterization-v1.results.zip.sha256
```

If interrupted, preserve the run directory, logs and source. The frozen launcher can resume identity-checked checkpoints only while its cumulative budget remains. Do not change the source/configuration, reset the budget or relax a gate to make it finish. For a stopped worker, the frozen `run_study.py --collect-existing` path can export existing output; consult `frozen_run/README.txt` for its exact options. An incomplete export remains incomplete.

The six-hour budget is a software ceiling, not a guaranteed runtime or billing cap. Final summary/export time is not fully represented by every archived timer snapshot. The Pod remains billable after a worker exits; keep it running only as intentionally chosen by the user. This workflow never terminates it automatically.

## 3. Return files inside the authorized local project

Create `incoming/characterization_v1/` in this publication project and place both returned files there. This is an explicit local project location; no tool needs to enumerate personal download folders. Keep originals and sidecars. Research-coordinate exports remain private and are excluded from Git and public source packages.

Use a CPU environment with NumPy; the project's existing installation is sufficient. For a fresh environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'numpy>=1.24,<3'
```

First validate without importing. Then create one new receipt:

```bash
.venv/bin/python studies/characterization_v1/tools/import_results.py \
  --archive incoming/characterization_v1/icml2027-characterization-v1.results.zip \
  --checksum incoming/characterization_v1/icml2027-characterization-v1.results.zip.sha256 \
  --numpy-checks --validate-only

.venv/bin/python studies/characterization_v1/tools/import_results.py \
  --archive incoming/characterization_v1/icml2027-characterization-v1.results.zip \
  --checksum incoming/characterization_v1/icml2027-characterization-v1.results.zip.sha256 \
  --numpy-checks --receipt-name characterization-v1-return-01
```

The importer validates the frozen expanded source by default, without needing `dist/` in a future GitHub checkout. It verifies the supplied checksum, ZIP paths/CRC, source/contract identities, required completion members, exact model/layer/domain/split panel and exported geometry. It preserves the original archive and a receipt under `studies/characterization_v1/imports/`. It never overwrites a prior receipt or imports model runtimes.

For an intentionally incomplete archive, `--allow-incomplete` preserves it with an incomplete status. This is evidence preservation, not an exemption for publication. Investigate failed checks rather than editing original results. A SHA checksum detects changed or truncated bytes relative to the sidecar; it does not prove correct execution or scientific validity.

## 4. Independently reconstruct supported numerical results

After a complete validated receipt:

```bash
.venv/bin/python studies/characterization_v1/tools/refit_coordinates.py \
  --raw studies/characterization_v1/imports/characterization-v1-return-01/raw \
  --output studies/characterization_v1/imports/characterization-v1-return-01/coordinate_refit_audit.json
```

This separate NumPy implementation refits direct ridge selection across exported domains/strata and C7/D4 generator/kernel/storage predictions, using independent action definitions. It checks primitive-only selection, predictions from the method's own intermediate outputs, family losses, cycles/order masks, training support indices and storage accounting. It does not import the frozen worker to certify its own output.

Independent reconstruction is intentionally bounded. The compact export cannot reproduce full hidden-state model forwards or independently recalculate PCA from raw activations. The basic refitter reports its omissions explicitly. A separate deeper audit has since checked analytic context-difference/coherent-action quantities and all 56 bootstrap intervals. Torch joint/potential optimization remains outside independent reconstruction. An audit report with matching supported values is still not automatic manuscript authorization.

## 5. Scientific review and manually curated manuscript revision

Keep prediction and context-difference endpoints separate. Keep S3 joint/context/entity and zero-shot/support information tracks separate. Holdout results must not affect fitting, bandwidth/support selection or the frozen panel. Report all boundary bandwidth selections, failed cells and negative outcomes. The state vocabulary is seen; unseen composites do not imply unseen states. Coordinate-storage matching does not equate total RAM, training exposure or statistical capacity.

The 10,000-replicate fixed-seed family/crossed percentile intervals are sensitivity summaries for a fixed model/layer panel. They are not new multiplicity-adjusted significance tests or a power certificate. Review interval arithmetic and clustering separately before quoting an interval. Native mechanism claims and an ICML acceptance prediction do not follow from these structural measurements.

The present manually reviewed revision follows those checks and explicitly records the unresolved scope. Future descriptive revisions must likewise be created with source archive hash, member paths, estimand, units, weighting and limitations. Keep `paper_eligible=false` in original outputs; record any later publication decision in a separate signed-off provenance record. Do not rerun old `bsg-next export --allow-partial` to force these results into an incompatible confirmation table. Do not alter original results or their old failure labels.

## 6. Source-only checks and packaging

```bash
PYTHONPATH=src python tests/run_cpu_checks.py
python scripts/verify_frozen.py
python scripts/verify_characterization.py
python -m pytest -q studies/characterization_v1/tests
python scripts/check_release.py
python scripts/package_release.py --output-dir dist/2026-10-21-project-reaudit-v2
```

These commands do not execute a model or start a paid job. The public source includes frozen study code, offline tools and tests, but excludes incoming archives and private imports. Overleaf contains anonymous sources and only a current verified PDF when available. Read the current validation and PDF QA records before using a package.
