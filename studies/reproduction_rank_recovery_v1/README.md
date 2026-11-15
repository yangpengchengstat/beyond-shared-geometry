# Rank-aware continuation after the original q16 embedding failure

This is an explicit recovery layer for the 21 October from-zero source release. It does not edit that release, its source manifest, the completed parent run or frozen historical evidence. It requires the observed `afs_pre_full` embedding-analysis failure: requested rank 16 versus observed centered training rank 6. Other failures are rejected.

The parent PRE-FULL and FULL are verified and imported read-only. Completed AFS-A/B/C checkpoints from the failed pilot are checksum-verified and copied under a fresh provenance receipt; their scientific configuration and FULL input binding must match exactly. The old failed D unit remains in the parent. This continuation reports it as infeasible, never a favorable gate or a q16 measurement. AFS-A/B/C then run without the infeasible old D block, matching the historical ABC-only route. The separate already frozen rank-aware AFS-D retains C7=6, D4=3, training-rank guards, matched internal dimensions, original seeds and original bootstrap rules. No scientific threshold is weakened.

The original wrapper's requirement that the AFS pilot execute the infeasible q16 D block was an integration error. PRE-FULL authorization in this branch checks the complete A/B/C matrix and retains its NOT_EVALUABLE scientific gate; the full historical replay is authorized by the user's existing request, not a fabricated scientific pass.

The two completed mainline stages, current characterization and compact/context, are imported from the parent and are not recomputed. Additive, retrospective and all remaining original root-queue branches continue in the new output directory. The original and revised calibration designs remain in that remaining queue as requested by the user; they were not completed before the historical failure. There is no three-hour timeout. Scientific stops preserve outcomes and skip dependent branches; technical errors stop the queue. No automatic manuscript promotion, deletion or Pod termination occurs.

## RunPod operation

Upload `bsg-rank-recovery-v1.zip` and its SHA256 sidecar to `/workspace`. Preserve the existing source directory and `/workspace/bsg-runs/from-zero-v1`. Stop neither the Pod nor delete its caches.

```bash
cd /workspace
sha256sum -c bsg-rank-recovery-v1.zip.sha256
unzip -n bsg-rank-recovery-v1.zip -d /workspace/bsg-reproduction-source/beyond-shared-geometry
cd /workspace/bsg-reproduction-source/beyond-shared-geometry
mkdir -p /workspace/bsg-logs
nohup bash studies/reproduction_rank_recovery_v1/start.sh \
  /workspace/bsg-runs/from-zero-v1 \
  /workspace/bsg-runs/from-zero-v1-rank-recovery \
  > /workspace/bsg-logs/rank-recovery-v1.log 2>&1 < /dev/null &
```

The launcher verifies parent evidence and installed environment, then verifies already downloaded pinned models offline. It does not reinstall dependencies or download weights. Parent writer locks prevent simultaneous modification by the original queue. The continuation writes to a separate output and never rewrites failed parent states or source identities. On an environmental interruption repeat the same command; verified outputs are reused.

```bash
tail -n 40 /workspace/bsg-logs/rank-recovery-v1.log
tail -n 40 /workspace/bsg-runs/from-zero-v1-rank-recovery/logs/historical.log
```

Final evidence ZIP and sidecar:

```text
/workspace/bsg-runs/from-zero-v1-rank-recovery.results.zip
/workspace/bsg-runs/from-zero-v1-rank-recovery.results.zip.sha256
```

The combined export includes compact evidence from the read-only parent, its original rank failure, and new continuation evidence, in separate archive roots. Bulk raw activations and model weights are excluded. Preserve and download this evidence before stopping, restarting or deleting a Pod backed only by a container disk.

This recovery has CPU/integration tests; full real-model recovery remains a cloud validation task. A program completing is not proof of favorable scientific outcomes or ICML acceptance.
