# Completed independent-orthogonality comparison

Status: **96/96 units complete**, after **8/8 numerical fixtures passed** in the
recorded Python 3.11 / NumPy 2.2.6 runtime. These computations occurred before
release 0.5.7 assembly. This editorial integration does not rerun them.

For each complete original Study 2 test bank, independent O(r) minimizers yield
`L_orth`; their excess over independent linear risk is `beta = L_orth - eta`.
The additional common-group-law cost is `L_star - L_orth`, bounded by
`[max(0, L_lower - L_orth), L_upper - L_orth]`. Domain summaries average the
**already-clipped per-condition lower endpoints**. They cannot be reconstructed
by subtracting rounded domain means. True costs decompose exactly; the displayed
clipped lower bounds do not necessarily add to the earlier joint lower bound.

| Dimension/domain | Mean orthogonality cost | Mean additional-law bounds | Positive lower count |
|---|---:|---:|---:|
| 16 / C7 | 0.00307429 | [0.00109725, 0.02992694] | 23/24 |
| 16 / D4 | 0.02453084 | [0.00184179, 0.12120863] | 23/24 |
| 8 / C7 | 0.02957135 | [0.03539741, 0.15184219] | 22/24 |
| 8 / D4 | 0.01021673 | [0.00005234, 0.07693009] | 19/24 |

The counts use the previously fixed software tolerance, not a significance test.
Intervals bound the mean of cell-specific finite-bank class optima; they are not
confidence intervals or measured exact costs. Zero lower endpoints are retained
and do not imply zero true cost. Each dimension has its own whitening and risk
normalization, so this is not a comparison for choosing an optimal dimension.

At dimension 16, the C7 additional-law lower-endpoint median is `5.34554e-6`.
Llama-3.1-8B at layer 13 contributes 99.44% of their sum. This describes concentration
of the **computed lower bounds**, not concentration of unknown true costs. All
conditions and equal weights are retained. It is distinct from the earlier joint
premium's 90.7% concentration across two C7 conditions. One 8-dimensional C7
positive endpoint is only about 1.55 times the fixed software threshold; no
certified floating-point error enclosure is claimed.

The new comparison narrows a possible explanation: positive additional-law lower
bounds cannot be attributed entirely to orthogonality under the stated common-bank
contract. It does not establish semantic uniqueness, native causal use, universal
large effects or generalization to new contexts. Saved witness-transfer and
registration results remain separate targets and are not rerun here.

Read the full-precision result summaries and `ORTHOGONAL_INTEGRATION_REPORT.md`.
The original complete execution directory retains all receipts, diagnostic arrays,
logs, input/source bindings and nonpositive-lower cases. No condition was dropped.
