# Prepared output schema

No files described here contain results until an explicitly authorized run has
completed. The present status is PREPARED_NOT_RUN.

| Output | Content and interpretation |
| --- | --- |
| `plan.json` | Frozen protocol and all 96 cell/dimension units before numeric work. |
| `binding.json` | Local source manifest, protocol, saved dependency and archive hashes. |
| `source_snapshot/` | Exact executed local files and bound project dependencies. |
| `runtime.json` | Python, NumPy and platform identities; no model loading. |
| `input_preflight.json` | Archived row/PCA/model/member identities and header checks. |
| `units/<unit>/summary.json` | Unit/model/layer/domain/dimension, complete checks, moment eigenvalues, singular values, all raw endpoints, saved-bound comparison and any failure. |
| `units/<unit>/diagnostics.npz` | Optional saved Q/M/whitening/moment arrays; no pickle and no original-data replacement. |
| `units/<unit>/receipt.json` | Source binding, terminal status and hashes of committed unit outputs. |
| `cell_summary.csv` | All 96 declared units; full float precision, empty fields for unevaluable quantities and explicit failure status/reason. |
| `coverage.json` | Exact coverage/failure counts and equal-cell domain means only when all 24 are evaluable and inputs unchanged. |
| `input_preservation.json` | Both original archive hashes before and after; source/dependency hashes are also rechecked. |
| `failure.json`, `failure_units/` | Fatal/interrupted-run records and preserved partial units. |
| `output_manifest.json` | Every published run file's SHA-256, except this self-referential manifest. |

`eta` and `L_orth` are direct empirical residual optima for independent linear and
independent O(r) maps on the SAME complete bank. `eta_svd` and `L_orth_svd` are
checked singular-value identities. `beta=L_orth-eta` and `beta_svd` are the cost of
orthogonality alone. `beta_eta_floor` is the eta-forced lower floor, with an explicit
unavailable status if raw eta is outside the exact [0,1] interval. No endpoint is
silently clipped to make a mathematical formula evaluable.

`extra_law_lower=max(0,L_lower-L_orth)` and `extra_law_upper=L_upper-L_orth` bound the
additional optimum cost of exact group laws within the orthogonal class. They do
not equal that cost in general. The positive-lower-endpoint flag uses the original
scalar tolerance; the raw upper endpoint is retained even if a floating-point
check finds a tiny negative value. Failure statuses disallow completed-domain
means. A positive original total premium does not imply a positive extra-law lower
endpoint. All quantities are descriptive finite-bank targets, with no p-values or
population inference and no assertion that independently chosen Q matrices form
a group action.
