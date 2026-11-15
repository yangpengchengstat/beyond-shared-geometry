# Anonymous saved-endpoint reproduction

This self-contained capsule reproduces the numerical contents of manuscript Tables 3, 4, 5 and 6 from saved full-precision endpoints. It also regenerates the 48 per-cell risk summaries, 48 registration-margin summaries and the descriptive lower-endpoint summaries. It needs Python 3.10 or later and only the Python standard library. No installation, network connection, model weights or private archive is required.

From this directory, run:

```sh
python reproduce.py --output /tmp/review-table-reproduction
python -m unittest discover -s tests -v
```

Choose a new output directory each time. The script refuses an existing output directory and refuses to write inside the evidence capsule. It first checks all seven evidence/expected-output SHA-256 hashes in `manifest.json`, validates the complete cell/image panel, recomputes the per-cell arithmetic, compares those values with the saved rows, and checks every displayed numerical value against the frozen targets. Successful execution prints `"status": "PASS"`.

The result directory contains `table3.tex` through `table6.tex`, full-precision `cell_risk_recomputed.csv`, `cell_margins_recomputed.csv`, `domain_risk_summary.csv`, `domain_margin_summary.csv`, `lower_endpoint_summary.csv`, `reproduced_values.json` and `verification.json`. The TeX files are table bodies requiring the usual `booktabs` commands; the manuscript supplies captions and table numbering. The numerical display is five decimal places in Tables 3–5 and three significant digits in Table 6, with no tolerance-based zeroing.

The tests cover hash tampering, missing and duplicate cells/images, the original symmetric pairwise tie tolerance, semantic-image exclusion, the all-24-cell concentration denominator, immutable inputs and fresh outputs. They also copy the capsule into an isolated temporary directory and run it with Python site packages disabled, then check the rendered table values and CSV float round trip.

## Interpretation and scope

All cells are retained: 24 C7 and 24 D4 model/readout cells, with 120 C7 or three D4 image conditions in each cell. Rank is 16. The finite-bank target uses the original Study 2 test bank, 64 paired families/128 context blocks, original coordinates/PCA, uniform context/state/action weights, and the complete action grid including identity and fixed points.

Tables 3, 5 and 6 use the test bank's own uncentered second-moment metric. Table 4 is a transfer comparison of frozen predictors in the training metric. Its witness, OLS and ridge losses must not be compared directly with Table 3's test-metric orthogonal-class bounds.

Table 5 uses total trivial-type energy `e_triv` for `L(I)=2(1-e_triv)`, not selected energy `b_triv`. The saved energies are already divided by rank. The class lower and upper endpoints are `a=2(1-gamma)` and `u=2(1-gamma*gamma)`. The premium bounds are `max(0,a-eta)` and `u-eta`, computed separately in every cell before taking equal-weight domain means. No cell or outlier is omitted.

The lower endpoint `p_minus` bounds the class-optimal risk premium from below; it is not the exact premium. The restricted class imposes exact group laws and orthogonality jointly, so this quantity does not isolate either constraint.

For C7, the lower-endpoint mean is 0.004164426680895648, the median is 0.00038929152944697926, and the range is [0.000034867499308447944, 0.06254392518998934]. The two largest lower endpoints contribute 90.68064875787752% of the sum over all 24 C7 cells. They are Llama-3.1-8B at layer 13 and Llama-3.2-3B at layer 11. This is concentration in a descriptive fixed panel; it is not a significance test, a uniform effect size or a conclusion about unseen contexts. The corresponding D4 mean/median/range and concentration are reproduced in `lower_endpoint_summary.csv`.

Table 6 excludes the semantic image itself by its canonical permutation-image hash. It reports semantic gamma minus the strongest alternative and semantic gamma minus the median alternative. The tolerance is exactly `1e-8 + 1e-7*max(abs(a),abs(b))`, applied pairwise against the semantic gamma. Unique-leading, tied-leading and behind counts retain that original rule. Margins keep their signs; they are neither permutation p-values nor ranks of the exact optimum risk.

This capsule reproduces saved-endpoint arithmetic and formatting. It does not repeat model extraction, spectral computation, witness fitting, optimization, image enumeration, coordinate whitening, or the historical direct coordinate residual check of the identity formula. The saved `identity_risk_direct` and associated diagnostic fields are retained as previously recorded evidence and are not newly verified here. No family deletion, LOFO analysis, bootstrap or new experiment runs. Hash checks establish agreement with the included evidence, not independent confirmation of the original measurements.

The full-precision numeric strings in the CSV exports are unchanged. Archive/run-path and source-summary locator fields were omitted for anonymous review; model identifiers, readout identifiers, canonical image identities and opaque saved bank join keys are retained. This capsule has newly written scripts and documentation and contains no copied repository source. Its license applies to those new files; no upstream repository license is altered.
