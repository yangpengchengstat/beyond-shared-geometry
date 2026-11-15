# Compact/context v1 engineering validation — 21 October 2026

The new standalone cached-coordinate study is prepared. Full exploratory execution has **not** been launched; no confirmatory result or manuscript promotion is implied.

## Implemented scope

- C7/D4 all-element and primitive-only comparisons: ridge, charged affine ridge, decoded-centroid displacement, limited-support fields, full-training-pair RFF and Nyström predictors at fixed storage budgets.
- Nyström absorbs the training-only whitening transform into inference coefficients. Its storage count includes retained landmarks, coefficients, bias, normalizer and bandwidth, without retaining an unnecessary square whitening matrix.
- C7/D4/S3 context differences: coordinate-trained operator reference, identity, scalar retention, context-trained linear transport, decoded and oracle category-retention controls, and full-data nonlinear predictors. S3 test strata remain separate; support and reserved rows are excluded.
- Semantic and consistently registered alternative actions; validation-only error/output-norm comparability with all unmatched methods retained. Linear coefficient norm is reported only as such, never substituted for nonlinear Jacobian norm.
- Input/source pins, atomic checksummed checkpoints, exclusive output locks, progress heartbeat, restart, result ZIP/sidecar generation and explicit exploratory labels.

## Verification

The new suite passed **48 tests plus 19 subtests**. The original 26 included evidence files and 53 frozen artifact checks passed. The release checker confirmed the frozen numerical implementation, the separately frozen characterization worker, and the existing manuscript QA remain valid.

The exact original results archive was SHA-256 and CRC verified, with all 72 finite float64 q16 cells and their action/split metadata accepted. The five predeclared development jobs used only train, validation and excluded pilot coordinates. No model was imported, no old calibration ran, and no inspected test output entered this benchmark.

| Development job | Seconds |
| --- | ---: |
| Qwen__Qwen2.5-1.5B__C7__6--semantic--compact | 0.297 |
| Qwen__Qwen2.5-1.5B__C7__6--semantic--context | 0.151 |
| Qwen__Qwen2.5-1.5B__D4__6--semantic--compact | 0.264 |
| Qwen__Qwen2.5-1.5B__D4__6--semantic--context | 0.115 |
| Qwen__Qwen2.5-1.5B__S3__6--semantic--context | 0.946 |

The final five-job benchmark took 1.67 seconds in the measured local two-worker computation session, **excluding archive/source preflight**. This is not a full-panel RunPod runtime estimate. Larger test strata add evaluation work; the full panel has 240 jobs.

A standalone unzipped source package was also tested against matching development checkpoints. The restart reported all five jobs already complete and did not refit them. The initial development receipt/source copy is preserved separately; no historical result was overwritten.

Final package SHA-256: `0ee532959262aa903e3ab964dcc4a2cdeed44cd296e4f340d64b8ef13b4bcd21`.

Final source-manifest SHA-256: `b4e247ca24284d858e6e64fed89c315339fb5f8dc2ee2baf39cc02c5f26b7167`.

Parent archive SHA-256: `79b1c53900cfcd91f0644b3db8405160213785487d0f7cea8b0320244a5aec0a`.

## Operation and remaining work

Use [the standalone run guide](../studies/compact_context_v1/README.md). It launches no cloud instance and has no three-hour timeout. The default wrapper runs the five short development jobs, then the complete fixed panel. Running it on an existing GPU Pod still incurs that Pod's charges even though the program uses CPUs only.

Full results still need to be collected and scientifically audited. Existing tests were inspected previously, so they cannot certify a fresh confirmatory claim. Raw hidden-state/forward replay, native intervention, peer-paper reproduction, explicit low-rank linear variants, and new holdouts are outside this package. Affine ridge is included; no claim is made that every option in the wider roadmap has been completed.
