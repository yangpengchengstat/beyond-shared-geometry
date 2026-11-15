# Final limited edit 0.5.4

This revision starts from the 57-page TMLR 0.5.3 manuscript and keeps its existing analyses. The only manuscript edit is one sentence in Section 5.1 (page 9) describing the size and heterogeneity of the retained lower endpoints. The abstract, Tables 3-6, Appendix I.7/I.8 and all original numerical outputs remain unchanged. The revised PDF remains 57 pages.

## Saved-CSV verification

The source is `results/main30_bounded_v1/cell_risk_reference.csv`, SHA-256 `ec52b77b536d8241edf5229b548d559ec09b8845454807946e03a4bf6171ae61`. The column `premium_lower` is the lower endpoint `p_minus`. All 24 cells in each domain retain their original equal weights.

| Domain | Median | Minimum | Maximum | Original equal-cell mean |
|---|---:|---:|---:|---:|
| C7 | 0.00038929152944697925 | 0.000034867499308447944 | 0.06254392518998934 | 0.004164426680895648 |
| D4 | 0.0187559524517441415 | 0.002538358344475869 | 0.08718769943343879 | 0.026236942302582776 |

Two C7 cells have much larger lower endpoints: Llama-3.1-8B layer 13 (0.06254392518998934) and Llama-3.2-3B layer 11 (0.028087973960786328). Together they contribute 90.68064875787752% of the summed C7 lower endpoints, and hence the same fraction of the equal-cell mean. This explains the difference between the mean and median. Neither cell is removed, downweighted or treated as an erroneous measurement. No alternative mean excluding these cells is substituted.

`analysis/final_edit_v1/summarize_lower_bounds.py` independently reads the saved decimal strings and calculates this summary using the Python standard library. Its recorded output is `results/final_edit_v1/premium_lower_summary.json`. Exact-decimal summaries and the portable helper's binary-float summaries can differ in the final rounding bit; their manuscript displays agree. This is arithmetic summarization of retained values, not a rerun of spectral analysis, witness construction, model extraction or the original finite-bank derivation.

## Interpretation retained

- `p_minus` is a lower bound on the class-optimal risk premium; it is not the exact premium.
- The class restriction combines exact group laws and orthogonality. This revision does not separate their effects.
- Table 3 uses test-bank whitening; Table 4 evaluates frozen predictors in the training metric. Neither table is changed.
- Table 6 reports spectral registration margins, not p-values or ranks of the exact optimum `L*`.
- LOFO remains **PREPARED_NOT_RUN**. No family deletion, bootstrap, model forward pass, RunPod job or expanded simulation grid is executed.

## Anonymous reproducibility package

The previous anonymous Overleaf package contained manuscript sources and rendered tables, but no Python scripts or full-precision CSVs. The public GitHub package contained the retained code and data but also intentional author/license metadata, so it was not an anonymous supplement.

This revision supplies `dist/anonymous-review-source.zip` with two folders:

- `paper/`: anonymous TMLR source, its dependencies and the matching compiled PDF.
- `reproduction/`: new standard-library helper code, full-precision saved endpoints, canonical image identities, input mapping, expected table values, tests and checksums.

The reproduction script rebuilds the displayed values of Tables 3-6 and checks the Section 5.1 heterogeneity summary. It needs neither model weights nor the private coordinate archive. It does not recompute the original spectral budgets, fitted witnesses, or direct coordinate residual checks; those earlier results are treated as saved inputs. This scope distinction is explicit in the supplement.

After extracting the anonymous package, run `python reproduction/reproduce.py --output /tmp/fresh-review-tables` and `python -m unittest discover -s reproduction/tests -v`. Choose a fresh output directory. The full public source bundle retains the project's original MIT license and author metadata; it is supplied separately from the anonymous package.

## Validation and preservation

`release/final_manuscript_audit_0_5_4.json` records resolved table/appendix labels and unchanged source hashes. `release/pdf_qa.json` records the current PDF and source hashes. The source diff is `release/manuscript_changes_0_5_4.patch`; compilation, package and preservation checks are recorded in `release/software_validation_0_5_4.json` and `release/preservation_check_0_5_4.json`.

No existing study, tolerance, rank, split, original result, failure record or scientific eligibility flag is changed. This is a new local release copy, not an external submission or remote publication.
