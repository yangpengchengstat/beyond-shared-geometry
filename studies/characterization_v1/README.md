# Independent characterization v1

Study `icml2027-characterization-v1`, frozen 21 October 2026. This directory integrates the exact submitted RunPod source with separate post-run tools.

- `frozen_run/`: exact payload expanded from the pinned source ZIP; never edit these files for an active or completed run. Its own manifest and the release characterization manifest bind every byte.
- `tools/`: model-free result import, validation and independent coordinate reconstruction. These do not import the worker to derive its own answers.
- `tests/`: small synthetic engineering fixtures, never research evidence.
- `provenance/`: prior excluded-pilot and readiness audit notes.
- `imports/`: private locally imported archives, extracted coordinates and audit receipts; ignored by Git and explicitly excluded from public exports.

Read [the workflow](../../docs/CHARACTERIZATION_WORKFLOW.md) before using any result. The new contract is pilot-informed and limited in precision. A validated archive is not automatically paper eligible. Old failed gates and pilot outcomes are retained. No changes here require a new calibration or a GPU rerun.
