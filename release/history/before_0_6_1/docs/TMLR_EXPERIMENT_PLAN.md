# Original TMLR supplementary execution plan

The five studies below have completed, and their results are integrated in the
current manuscript. These commands are retained as historical reproduction
instructions, not pending work or authorization to rerun them. Current status is
in `CURRENT_DELIVERY_0_5_8.md` and `TMLR_RESULTS_REVIEW.md`. The separately completed
8/16 independent-orthogonality comparison is documented in
`ORTHOGONAL_COMPARATOR_RESULTS.md`. LOFO alone remains prepared and unexecuted.
No new experiment or numerical test was run for publication revision 0.5.8.

| Study | Primary purpose | Input | Model execution |
| --- | --- | --- | --- |
| Finite-bank spectra | Compute gamma, type energies and exact-action risk bounds on all 48 fixed cells | Audited compact-coordinate archive | None |
| Exact-action witnesses | Construct proof-based witnesses; distinguish same-bank theorem checks from frozen train-to-test prediction | Prepared, identity-verified coordinates | None |
| Complete image subgroups | Descriptive comparison of all 120 C7 and 3 D4 image subgroups | The same preserved coordinate banks | None |
| Controlled mechanisms | Explain behavior under declared shared, context-conjugated and state-code constructions | Fixed simulated construction and seeds | None |
| New contexts | Compare frozen coordinate-trained and difference-trained predictors under matched validation objectives | Old train/validation plus reviewed new text bank | One sequential extraction pass per pinned model |

## Local CPU sequence

Read both study READMEs before executing. Replace paths with the actual private
archive, its already verified checksum, and new output directories. These commands
are a guide, not an execution record.

```bash
BSG_ARCHIVE=/path/to/from-zero-v1-rank-recovery.results.zip
BSG_ARCHIVE_SHA256=PASTE_THE_VERIFIED_64_CHARACTER_SHA256

python studies/spectral_finite_bank_v1/runner.py prepare \
  --archive "$BSG_ARCHIVE" --expected-sha256 "$BSG_ARCHIVE_SHA256" \
  --output local_results/spectral-v1-prepared
python studies/spectral_finite_bank_v1/runner.py analyze \
  --prepared local_results/spectral-v1-prepared --archive "$BSG_ARCHIVE" \
  --output local_results/spectral-v1-results --execute-analysis

python studies/spectral_extensions_v1/runner.py prepare \
  --spectral-prepared local_results/spectral-v1-prepared \
  --archive "$BSG_ARCHIVE" --output local_results/spectral-extensions-prepared
python studies/spectral_extensions_v1/runner.py witness \
  --prepared local_results/spectral-extensions-prepared \
  --output local_results/spectral-witness-v1 --execute-analysis
python studies/spectral_extensions_v1/runner.py alternatives \
  --prepared local_results/spectral-extensions-prepared \
  --output local_results/spectral-alternatives-v1 --execute-analysis
python studies/spectral_extensions_v1/runner.py simulate \
  --output local_results/spectral-mechanisms-v1 --execute-analysis
```

Use the extension README for supported resume and collection behavior. Its
collection command is `collect --run <completed-or-partial-run> --output <new.zip>`.
Do not delete partial outcomes or weaken a numerical check to continue. Same-bank
whitening and theorem losses must not be silently identified with historical
row-normalized Ridge or Joint endpoints. A witness frozen on old training data
must not be selected from the final test bank.

## New-context workflow

Use the exact prepare -> human review -> fit -> extract -> analyze -> collect
sequence in `docs/TMLR_RUNPOD.md` and
`studies/context_generalization_v1/README.md`. New text families are assistant-authored
and require actual human semantic review before extraction. They are a separately
constructed fixed bank, not an independently sampled natural-language population.

## How to decide what enters the manuscript

- Report complete planned coverage and every not-evaluable outcome.
- Preserve gamma, ordinary spectral budget, type correction, selected type energy,
  and numerical residuals jointly. A zero type correction is a valid finding.
- Check exact group laws, orthogonality and the declared same-bank witness bound.
  A held-out error has a separate role and is not automatically bounded by the
  finite-bank theorem.
- Treat alternate-subgroup rankings as descriptive. Complete enumeration does
  not supply exchangeability or an automatically valid p value.
- Shared state coding can support an exact action. Simulation mechanism labels
  are not mutually exclusive semantic classes, nor is a high gamma a false
  positive merely because the construction uses state labels.
- Review paired-family effects on new contexts with their conditional sensitivity
  intervals; do not turn selected models/layers into independent population units.
- Add verified result aggregates, their exact source identity and the transformation
  record before removing a pending slot. Do not modify original measurements.

No old FULL/AFS replay, broad historical calibration, native competence screen,
cloud launch or automatic manuscript promotion is part of this plan.

The controlled simulation plan has 36 fixed cases: two domains, three mechanisms, noise levels 0 and 0.05, and three seeds. Each case uses 32 training and 32 test contexts with rank 16. These are mechanism illustrations, not calibration or power estimates. The exact-action witness freezes its metric on training data and compares held-out predictions with least squares and ridge at alpha 0.001. No case has been executed for this release.
