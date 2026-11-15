# Bounded offline revision 0.5.3

Date: 1 November 2026. Base: TMLR 0.5.2, with its revised abstract retained and one evidence-supported sentence updated. This is a new publication copy; the MAIN30 PDF, earlier releases, frozen results and failure records remain preserved.

## What was newly computed

The original five follow-up studies were not rerun. This revision reads their full-precision saved outputs and derives same-bank identity references, the interval for the exact orthogonal-action class premium, and margins against all saved alternative images. An additional identity residual check uses the already saved test coordinates, under the same complete-grid whitening contract. It is new offline analysis, not a new model experiment or independent confirmation.

The input map is [analysis/main30_bounded_v1/input_map.md](../analysis/main30_bounded_v1/input_map.md). It binds the original Study 2 test bank, all 48 cells, all 2,952 image units, model/readout/PCA identities, weights, metrics and source hashes. Derived full-precision rows and checks are in [results/main30_bounded_v1](../results/main30_bounded_v1). No PDF-rounded mean is used as an analytical input.

## Revision map

| Prior location | Purpose and new location | Evidence/code | Execution and status |
|---|---|---|---|
| Table 2 and main Section 3.2 | Define identity risk and the joint exact-law/orthogonality premium in the same metric | Existing observation contract; Appendix B identity derivation | Algebra checked; DONE |
| Appendix B type-energy paragraph | Add the direct identity-risk consequence using total, not selected, trivial energy | `paper/sections/spectral_risk.tex`; direct residual tests | New proof explanation, no new theorem; DONE |
| Tables 3 and 4 | Make test-bank class bounds versus training-metric frozen transfer visible in captions | New caption variants retain byte-identical numerical table bodies | DONE |
| Appendix I.3 and main Section 5.1 | Explain row-vector metric conjugation and changing admissible raw-coordinate classes | `paper/sections/planned_protocols.tex`; 2-by-2 engineering test | DONE |
| Main Section 5.1 | Add compact same-metric identity/premium table, with cell-wise endpoints averaged only afterwards | `domain_risk_reference.csv`; `derive.py` | New offline derivation and saved-bank identity check; DONE |
| Main Section 5.2 | Add strongest/median alternative margins, retaining original ranks and every comparator | `registration_margin_summary.csv`; original tolerance/canonical images | New offline derivation; DONE |
| Appendix I.1 and I.4 | Link original quantities to bounded derivations and provenance | Input map and full cell identities | DONE |
| New Appendix I.7-I.8 | Include all cell risks, type energies and registration margins, and distinguish unrun sensitivity | Six generated cell/type tables; raw CSVs | DONE |
| Abstract and Discussion | Separate class optimum, one constructor's transfer, and spectral registration margins | All 48 lower endpoints exceed frozen tolerance; complete margin results | One new abstract claim, no causal/population claim; DONE |
| Existing five follow-up analyses | Retain 48/48/2,952/36/48 accounting, new-context domain reversal, simulations and original stops | Unchanged `results/tmlr_followup_v2` and historical evidence | VERIFIED_ALREADY_PRESENT |
| Optional family deletion | Prepare complete guarded CPU code and protocol; inspect retained metadata and array headers | `analysis/rank_sensitivity_v1` | Synthetic tests and input inspection only; PREPARED_NOT_RUN |
| Original calibration, FULL/AFS, native and model extraction | Preserve originals | Historical source/results and failures | NOT_IN_SCOPE |

LaTeX references renumber later tables automatically when the two compact main tables are inserted. Existing values are retained; manuscript references use labels rather than hardcoded new numbers. The original tables remain as source files beside their caption-only variants.

## Findings and their limits

The exact class premium is bounded above and below for the same finite bank and metric as independent least squares. All 48 lower endpoints are numerically positive under the original tolerance. This result concerns the combined group-law and orthogonality restrictions, not either constraint in isolation, the optimum over arbitrary invertible actions, a particular witness, or generalization to unseen contexts. The bounds do not determine the exact premium.

Identity risk uses total trivial energy. All 48 formulas were checked against saved-coordinate full-grid residuals. The theorem's existence upper bound is not required to upper-bound identity risk. Original type energies, raw floating-point values and tolerances were not changed.

Margins retain all 120 C7 and three D4 images per cell. C7's spectral margins are generally small; D4 contains both positive and negative margins of wider scale. The original unique-leading counts remain 7/24 and 16/24. These are descriptive comparator results, not p-values, semantic truth probabilities or statistical equivalence. Family composition sensitivity remains unmeasured.

No mathematical or input blocker remains for tasks A-C/E. Optional sensitivity inputs exist, but numerical success after each deletion and wall-clock runtime have not been established on real data. Its prepared protocol is not retrospective preregistration or a mandatory pre-submission experiment.

## Reproduction and verification

Run `make bounded-checks` for the engineering fixtures. The B/C README gives the archive-based derivation command; choose a separate output location when rerunning analysis. The parent result archive is needed for the direct coordinate check and full input audit, but model weights and GPU access are not needed. It is intentionally not embedded in the public source ZIP.

Run `make bounded-assets` to import already computed TeX fragments and relabel the two original table captions without recalculating results. Run `python scripts/verify_bounded_revision.py` to verify source/output bindings without executing a study. The optional sensitivity entrypoint defaults to metadata inspection; real deletion execution requires two explicit opt-ins and a fresh output destination. No full run was performed in this revision.

Publication validation and PDF review are recorded separately in `release/software_validation_0_5_3.json` and `release/pdf_visual_review_0_5_3.json`. Compilation, packaging and engineering tests do not constitute final author approval. No external submission, remote push or cloud job was performed.
