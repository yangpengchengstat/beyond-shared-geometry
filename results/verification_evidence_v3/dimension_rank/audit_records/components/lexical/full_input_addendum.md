# Original FULL q=16: raw-input and saved-PCA binding addendum

The read-only extension examines all 300 saved dimension-16 base fits in `main_full_003`: 240 primary model/domain/depth/split conditions and 60 terminal-layer secondary conditions. It uses the 60 corresponding original activation-cache matrices in the frozen FULL archive (SHA256 `5fe7234572120073d8dfab2e8dc0d3373a741f8d48abadc92288f98ab16721f4`). Original and later replay runs are not pooled as independent observations.

For every condition, the audit checks the model, domain, layer, complete 40-template cache identity, the fit artifact's ledger SHA256, and the training-row count. Training rows are the complement of the saved validation/test template IDs. The saved training mean exactly equals the arithmetic mean of those raw training rows. Evaluating the **already-saved** PCA mean and basis on the raw bank agrees with the already-saved coordinates in all 300 conditions: maximum relative Frobenius residual `1.908134660018014e-15`. No PCA basis is fitted or replaced. The separate `1e-10` binding threshold was fixed in `full_input_protocol.json` before these checks.

Read-only singular-value diagnostics find centered training rank 195 for every C7 condition (196 training rows) and 111 for every D4 condition (112 training rows), under `s > max(shape)*eps(float64)*s_max`. All 300 inputs support 16 retained directions. The input arrays are float32 and the diagnostics are float64. The smallest retained singular value is sigma_16; the expected centered null direction is not misclassified as a failure of the requested 16-dimensional retention.

| Panel | Domain | Conditions | rank(H_c) | sigma_16/sigma_1 range | EVR(16) range |
|---|---|---:|---:|---|---|
| Primary four depths | C7 | 120 | 195 | 0.183385812--0.410528535 | 0.705240960--0.924524189 |
| Primary four depths | D4 | 120 | 111 | 0.067288109--0.397734905 | 0.763246474--0.975279766 |
| Terminal secondary | C7 | 30 | 195 | 0.079009505--0.293937132 | 0.797053829--0.945089144 |
| Terminal secondary | D4 | 30 | 111 | 0.156685745--0.222717817 | 0.841796347--0.898565859 |

EVR uses the full **original centered training matrix** Frobenius energy, not the energy of the truncated coordinates. The new full-spectrum ratios agree with the saved `pca_explained` totals to at most `9.992007221626409e-16`. These are audit diagnostics, not recovered evidence that a 95% explained-variance rule originally selected q=16. They establish feasibility and provenance for the retained dimension; they do not establish optimality or stability of later gamma, premium or registration conclusions across other dimensions.

Every input receives `verified_supported`; none triggers the previously fixed `sigma_16/sigma_1 <= 1e-8` ill-conditioning warning. There are no missing raw inputs or failed bindings in this extension. This finding applies to the original FULL input matrices. It neither establishes the ranks of absent AFS-D projected arrays nor substitutes for the separate follow-up whitening checks.

Deliverables: `original_full_input_rank_audit.csv` (300 rows), `audit_original_full_inputs.py`, `full_input_protocol.json`, `full_input_execution.log`, `full_input_bindings.json`, `full_input_members.sha256.json`, and `full_input_preservation.json`. The input archive is byte-identical before and after. No operator or PCA training and no model forward was executed.
