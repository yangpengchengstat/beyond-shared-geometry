# Independent orthogonality comparator — PREPARED, NOT RUN

This directory prepares one bounded follow-up to separate the cost of independent
orthogonality from the additional cost of exact group laws. No numerical tests,
real-bank calculations or model forwards have been executed for this protocol.
It supplies no new empirical finding and is not a completed manuscript result.

## Fixed scope

All 48 canonical semantic Study 2 conditions are declared at retained dimensions
16 and 8: 96 units, with no new model, state, context, registration or parameter
search. Original coordinates and PCA remain unchanged. Dimension 8 uses the first
eight original PCA coordinates. Each dimension is whitened independently using
its own uncentered empirical second moment, with the original fixed threshold
and uniform context/state/action weights. Identity and fixed points are retained.
This comparison is computed on each complete existing test bank itself: it is a
same-bank class-optimal comparator, not a train-fitted predictor's held-out score.

The old spectral calculations are not repeated. The new cross-moment/SVD risks
are checked against saved eta, identity risk and second-moment diagnostics, then
combined with the authenticated existing lower and upper exact-action bounds.
Registration rankings, trained witnesses and new-context transport are not rerun.
Original 16-dimensional results remain primary. LOFO remains unexecuted.

## Quantities

For whitened row observations, M_g = mean(x^T x_g). An SVD M_g = U_g diag(s_g) V_g^T
gives one independent O(r) minimizer Q_g = U_g V_g^T. There is no determinant
correction. A singular M_g can have multiple minimizing completions. Nothing
requires the independently selected Q_g matrices to obey group laws.

Direct residual risks are the canonical reported eta and L_orth; singular-value
identities are independently checked:

- eta = mean_g sum_j (1-s_gj)(1+s_gj) / r;
- L_orth = mean_g 2 sum_j (1-s_gj) / r;
- beta = L_orth - eta = mean_g sum_j (1-s_gj)^2 / r.

Each singular value is checked against [0,1] with the original tolerance and
retained without clipping. The code checks eta <= L_orth <= 2 eta and
L_orth <= L_identity. It also records the eta-forced orthogonality floor using
eta^2 / (1 + sqrt(1-eta))^2 to avoid cancellation. If a raw tolerance-passing eta
lies outside [0,1], that square-root expression is marked unavailable rather than
clipping eta. Positive eta alone already forces positive orthogonality cost in
the exact identities. A positive old total-premium lower bound therefore does
not isolate a group-law cost.

The separately reported interval for L_star - L_orth is
[max(0, L_lower - L_orth), L_upper - L_orth]. Only the maximum defining this lower
endpoint is applied; negative raw numerical residuals, singular values, eta and
upper endpoints are not silently clipped. All failed checks and raw diagnostics
are retained. This interval need not have a positive lower endpoint. It is not a
measurement of L_star, a p-value, a population certificate or a semantic ranking.

## Inputs and immutable identities

`source_manifest.json` binds every local file plus the byte-unchanged historical
reader, group specification/kernel and the existing 96-row endpoint/diagnostic
CSVs. The reader checks both original archives, their member hashes, model/tokenizer
revisions, PCA and source-row identities, and the declared complete 48-cell panel.
Its historical metadata verification also checks old alternative-image receipts;
this does not recompute any alternative image. New calculations only use canonical
actions. `historical_protocol.json` is copied unchanged from the completed dimension
study. Neither this protocol nor its execution modifies the original inputs.

## Execution is not authorized by preparation

From the current project root, the following only displays prepared metadata and
hash-verifies source/endpoint files; it opens no original archive and imports no
numeric library:

```bash
python3 studies/orthogonal_comparator_v1/run.py --help
python3 studies/orthogonal_comparator_v1/run.py status
```

The exact future runtime is Python 3.11 / NumPy 2.2.6. No dependency was installed
in preparation. After separate execution authorization, first run the PLANNED
software checks, then the fixed 96-unit study:

```bash
python3.11 -m unittest discover -s studies/orthogonal_comparator_v1/tests -v
python3.11 studies/orthogonal_comparator_v1/run.py run \
  --repo ../../bsg_latest_complete_20261106/research_archive \
  --output ../orthogonal-comparator-results-v1 \
  --execute-offline-analysis --save-operators
```

The data-root path is relative to the new complete-project layout. It must contain
`incoming/rank_recovery_v1/from-zero-v1-rank-recovery.results.zip` and
`incoming/tmlr_all_v2/tmlr-all-v2.20261030T203725Z-5489fd.results.zip`. Archive identities
are frozen in the unchanged reader. A missing archive is a failure, not a reason
to use another run. Output must be fresh and outside both source and input roots.
Existing outputs are refused, including `--resume`: preserve failures and start a
separately identified run if a later correction is required.

## Outputs and reporting

The initial plan lists every unit. Per-unit summary/receipt files retain complete
input/source identities, whitening eigenvalues, singular values, direct and SVD
risk checks, old-bound comparisons and failure reasons. Optional diagnostic NPZs
retain whitening, cross-moments and Q matrices, including finite partial diagnostics
on a subsequent check failure. These arrays are new outputs only.

`cell_summary.csv` has all 96 declared rows and full Python float precision.
`coverage.json` includes failures and uses equal 24-cell domain means only when
that dimension/domain is complete and original archive hashes remain unchanged.
Partial interrupted files are preserved; unexpected process termination may leave
an incomplete plan and receipts, never a false complete result. A complete run
records source/dependency snapshots, runtime, before/after archive hashes and an
output manifest. Review results before any paper integration. No automatic
scientific promotion or replacement of existing results is implemented.
