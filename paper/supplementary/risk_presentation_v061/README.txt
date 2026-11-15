Same-bank risk presentation, v0.6.1

Run: python plot_saved_risks.py
Requires reportlab. This joins full-precision previously completed endpoints,
filters the original orthogonal table to r=16, and plots all 48 conditions.
It never reads activations, computes a new risk, refits, resamples, or runs models.
INPUT_MAPPING.json records exact original paths and checksums. The copied CSVs
retain the original bytes. plotted_endpoints.csv records plotted numeric strings.

Each condition uses the same complete Study 2 test bank, its original r=16
coordinates, own empirical second-moment whitening and uniform state/group
weights. Different conditions have different banks/coordinate metrics. The
vertical segments are deterministic class-risk bounds, not confidence intervals,
not a fitted action's measured loss. No held-out witness loss (training metric)
is mixed into this figure. Horizontal offsets only separate overlapping symbols.

The original r=8/16 tables, aggregate weights, thresholds, zero lower endpoints,
and failure records are untouched. This figure does not perform family influence
analysis or additional dimensional sensitivity.
