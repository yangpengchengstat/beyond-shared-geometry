TMLR RunPod input compatibility repair v2, 2026-10-30

Observed v1 failure, reproduced with the preserved rank-recovery archive:
24 C7 cells: Prompt text missing; overlap cannot be checked.
24 D4 cells: Row domain differs.

Cause: per-cell rows.json is a compact coordinate index. Full text is in
characterization/datasets/<domain>/rows.jsonl. The archive uses D4_corner
for the four-corner dataset, while analysis cell directories use D4.

Repair:
- Declare archive_row_domains C7 -> C7, D4 -> D4_corner. Never rename original
  row IDs or orbit IDs. The eight-state D4_free8 task remains incompatible.
- Read sealed dataset records, verify original file/dataset hashes, join by
  exact row ID, require all compact metadata to match, and verify restored
  extraction order using BOTH the dataset hash and serialized-row hash.
- Check exact old/new text overlap using recovered original text, not generated
  or guessed text. Save these source bindings in every prepared cell.
- Validate the explicit legacy domain in old train/validation block selection.
  New bank text, labels, coordinates, and source spans are unchanged.
- Make an incomplete prepare return nonzero and list failed cells; refuse
  review or model execution for incomplete preparation.

Scope: technical input adapter repair only. No estimator, action table,
normalization, hyperparameter grid, bootstrap, threshold, model revision, split,
rank, or new-context bank change. No empirical endpoint, fit, model execution,
calibration or FULL/AFS rerun during development of this repair. CPU software
fixtures and real-archive input integrity/preparation checks are distinguished
in VALIDATION.json. Real input preparation does not certify scientific success.

Preservation: original v1 ZIP and local release are unchanged. Modified files'
prior bytes are retained under before/ with hashes in CHANGE_RECORD.json.
The source locks were deliberately updated with predecessor identities to
record this implementation repair, not to waive a failed scientific gate.
The original archive is an external, separately hashed input. Do not include
private arrays, credentials or model caches in a public code upload.

Use a new /workspace/tmlr-all-v2 source and /workspace/bsg-runs/tmlr-all-v2 run.
Keep the failed v1 source, state, input manifest and logs. Reuse HF_HOME and
/workspace/bsg-tmlr-venv. Actual human semantic review is still required by
this frozen context-bank protocol before models run; no approval is fabricated.
