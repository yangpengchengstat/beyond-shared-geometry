# Preparation validation — 21 October 2026

Status: implemented and checked without model execution. The full new A40 queue has not yet run. No new scientific results or favorable gate decisions were produced during preparation.

- 16 characterization adapter tests passed, with 8 subtests: exact regenerated dataset identities; 72-cell plan; original batch boundaries; raw-cache reuse/tamper detection; atomic cell commit; interrupted-cell isolation; S3 compact export round-trip; source and output namespace checks.
- 29 environment/historical/orchestrator tests passed, with 7 subtests: full wheel-lock conversion, pinned model/shard inventory, no token disclosure, original config/API compatibility, real frozen AFS-A/B/C computations on a small synthetic fixture using a fresh FULL identity, saved-operator retrospective replay against newly generated FULL metrics, scientific prerequisite preservation, checksum-bound reuse, and real subprocess scheduling after a simulated scientific stop.
- 113 existing publication CPU checks passed.
- The source ZIP was extracted into an empty temporary directory. Its internal source manifest, complete queue plan, historical stage plan and from-zero C7/D4/S3 task-bank generation all passed without old caches, model downloads or model execution.
- Original evidence validation passed: 26 included evidence files and 53 release artifact checks. The unchanged 36-file characterization numerical dependency and 59-file prospective runtime also passed release validation.
- Existing manuscript PDF QA remains current. No manuscript result table or numerical claim was changed by this preparation.

The small synthetic fixture is a software integration test only. It is not model evidence. A pandas future-warning occurred in the preserved runtime telemetry concatenation path; the relevant operations passed under the pinned pandas version.

The historical source policy is explicitly new: immutable portable SHA256 source receipts with actual Git metadata retained. It does not fabricate a clean historical Git commit. Original PRE-FULL gates and AFS scientific outcome rules remain unchanged. AFS PRE-FULL authorization records verified coverage plus the user's explicit full-replay request; its NOT_EVALUABLE scientific label is preserved.

Exact GPU results and total duration remain unverified until the new cloud run. Linux wheel resolution, gated-model access, real-model repeatability checks, GPU/driver compatibility, and all downstream scientific gates are checked at runtime. Missing external implementation families and language banks remain missing prerequisites, not fabricated results.
