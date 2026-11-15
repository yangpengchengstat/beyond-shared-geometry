# Submission presentation, v3

This directory records layout-only changes after complete-project-v2. Existing
research outputs, metrics, models, splits, tolerances and completed/stopped states
are unchanged. LAYOUT_PROVENANCE.json and layout_changes.diff record caption
placement, pilot naming and evidence-path clarifications. The public project
retains prior bytes under release/history/before_completion_v3; the full owner
package also retains private inputs. Older caption receipts describe those
historical outputs, not the current caption placement.

The following optional rendering commands consume saved scalar endpoints only
and require an already installed ReportLab environment. They perform no model
execution, fitting, whitening, resampling or new risk computation:

```bash
python paper/supplementary/submission_presentation_v3/render_concept.py --output paper/figures/evaluation_targets.pdf
python paper/supplementary/submission_presentation_v3/render_saved_risks.py
```

The latter verifies its two retained inputs through the original INPUT_MAPPING,
and requires plotted_endpoints.csv to be byte-identical to the previous 48-cell
plot. Font and layout changes preserve every risk endpoint, condition and metric.
The original renderers and their provenance remain available as prior stages.
The current source figures have been rendered and viewed; whole-manuscript PDF
review is separately pending until a new full PDF is exported.

Reference-layout adjustment, 2026-11-08: the concept diagram is 468 by 180 pt
and the saved-risk diagram is 540 by 405 pt. This recovers a compact aspect
ratio with readable type. The Introduction ends with a float barrier. Table
captions remain above their bodies. No table, formula or saved endpoint changes.
These source constraints do not certify identical page numbers; a newly exported
whole-manuscript PDF must still be reviewed. See release/LAYOUT_UPDATE_V3.json
in the enclosing public project for the current layout receipt.
