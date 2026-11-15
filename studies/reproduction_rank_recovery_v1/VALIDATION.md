# Recovery validation — 22 October 2026

The original all-history wrapper accidentally required the old q16 input-embedding diagnostic although its centered lexical state rank is six for C7 and three for D4. The existing separate repaired AFS-D design was already frozen at those respective dimensions. The recovery keeps this distinction explicit and preserves the q16 failure.

Eleven focused CPU regression/integration checks passed:

- Real C7/D4 categorical arrays reproduce the original rank guard, while the separately declared matched dimensions fit successfully; the guard remains active.
- Real small A/B/C computations are checkpointed, a real rank error is recorded by the original ledger implementation, and only the completed A/B/C artifacts are checksum-verified and reused.
- Changed scientific configurations and corrupt checkpoints are rejected.
- The original failed D ledger and every parent artifact remain unchanged after reuse and successful ABC postprocessing.
- Real two-domain AFS postprocessing, the NOT_EVALUABLE pilot gate, explicit coverage-based full-replay authorization and a second resume attempt pass.
- Verified parent FULL/PRE-FULL stages are imported read-only, while other failure types or changed FULL artifacts are refused.
- Parent/output overlap is rejected; cached extension stages use the real parent activation path.
- A real subprocess queue imports the two completed mainline stages, preserves a simulated scientific stop, skips its dependent stage and runs independent branches.
- Combined export preserves original failure evidence and excludes bulk activations.

The fixtures use synthetic arrays and are software tests only. No model was loaded or downloaded, no new scientific result was inferred and the full A40 recovery has not yet executed. The original pandas FutureWarning is unchanged. Exact installed runtime, source identities and existing checkpoint hashes are checked on the Pod before continuation.
