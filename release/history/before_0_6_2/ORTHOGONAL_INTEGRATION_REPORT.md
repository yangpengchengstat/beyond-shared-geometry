# Completed orthogonal comparison retained in v0.5.8

The numerical comparison was completed before v0.5.7. This v0.5.8 editorial reconstruction starts from that 71-page version, retains all existing results, and runs no numerical test or experiment. Its earlier integration report remains in release/history/before_0_5_8/ORTHOGONAL_INTEGRATION_REPORT.md.

## What is now supported

The completed same-bank comparator supplies the optimum over independently chosen O(r) matrices, L_orth. Its difference from the independent linear optimum eta is the orthogonality cost beta. The previously saved exact-action bounds then constrain the further cost of requiring the matrices to obey common group laws. These are comparisons on each finite bank's own whitened metric, not held-out prediction by training-fitted maps.

| Dimension/domain | Mean L_orth | Mean beta | Mean lower/upper further-law bounds | Above original tolerance |
|---|---:|---:|---:|---:|
| 16/C7 | 0.028706138539504867 | 0.0030742877601372524 | [0.0010972473075670668, 0.029926935414265932] | 23/24 |
| 16/D4 | 0.12773711539307211 | 0.02453084171643166 | [0.0018417862444482702, 0.12120863365869687] | 23/24 |
| 8/C7 | 0.09671736976195285 | 0.029571351169869475 | [0.03539741217900925, 0.15184218588614895] | 22/24 |
| 8/D4 | 0.08081945437417146 | 0.010216729667878904 | [0.000052338847710833684, 0.07693009390044526] | 19/24 |

Values above are copied from the completed coverage record, not recalculated from matrix data. Means retain all 24 conditions per domain/dimension and use per-cell clipped lower endpoints. The source map is `results/orthogonal_comparator_v1/SOURCE_BINDINGS.json`.

The primary 16-dimensional result rules out attributing the entire joint cost to orthogonality in 46/48 declared conditions, subject to the theoretical assumptions and recorded numerical checks. It does not identify the unknown exact-action optimum. It does not imply large practical costs, population significance, semantic uniqueness or native group computation.

## C7 heterogeneity is essential

For 16-dimensional C7, the further-law lower median is 5.345539812429481e-6 and its range is [0,0.026187662085921326]. Llama-3.1-8B at layer 13 contributes 99.44454448767871% of the sum over the 24 C7 conditions. This concerns the computed lower bounds, not the concentration of true costs. All conditions retain their original equal weight.

The older 90.7% concentration is a different observation: two C7 conditions dominate the joint lower bound p_minus. Both statements remain, with their quantities and denominators distinguished. The true cost decomposition is additive, but clipped lower bounds need not add: p_minus is not generally beta+ell_minus. In particular, beta/p_minus cannot estimate the fraction of true cost caused by orthogonality.

The two 16-dimensional zero lower endpoints and seven 8-dimensional zero endpoints remain included. Zero is an inconclusive lower bound, not proof of zero cost. The smallest positive 8-dimensional C7 endpoint is only about 1.55 times its fixed numerical tolerance; the paper retains this limitation instead of changing the threshold. Full values are in `review/domain_diagnostics.json` and `review/nonpositive_lower_cases.json` within the result directory.

## Current manuscript locations

- Section 3 defines the three classes in one common metric.
- Section 4.3 explains measured orthogonality cost and the bounded further common-law cost.
- Table 2 retains the original two panels, including all four 8/16 domain rows; Section 5.3 states C7 heterogeneity, fixed tolerance, zero endpoints and numerical limits.
- Section 5.5 keeps dimension sensitivity separate from dimension selection. Each dimension retains its own whitening and normalization.
- Appendix B formulas and Theorem 7 are unchanged. Appendix M retains complete evidence, provenance, runtime/test history and nine zero lower endpoints.
- Old generated tables remain byte-identical. Their automatic numbers may change; stable labels and the final compile record provide current navigation.

## Reproducibility and preservation

`paper/supplementary/orthogonal_comparator/` includes full-precision saved inputs, all 96 CSV rows, previous heterogeneity summaries, input mapping, hashes and a table-formatting-only generator. The public project additionally contains all 96 original unit summary JSON files and execution/test logs copied byte for byte (log extension changed to .txt for publication portability). Complete original NPZ diagnostics, receipts, source snapshots and the full execution archive remain in the preserved sibling result delivery; no private data are replaced.

The recorded execution passed 8 software fixtures before all 96 real-bank units, with 11,520 recorded checks passing and original archive hashes unchanged. This revision verifies those records and preservation; it does not claim a new independent numerical replication. The original numerical rules, q/r values, old coordinates, mean/PCA and primary outcomes are unchanged. LOFO remains prepared but unexecuted, and historical native/calibration/rank stops remain stops. Original q=16 selection motivation and missing wider same-bank caches remain unresolved.

Final compilation and package checks are recorded in `COMPILE_CHECK.md`. The original 0.5.7 metadata and documentation are retained in `release/history/before_0_5_8/`; earlier history remains within the project. The complete old manuscript is also preserved beside this independent revision. No journal, Overleaf or GitHub upload is performed.
