Main-text presentation supplement; original formatting v0.5.8, caption integration v0.6.3

The coverage file, numerical inputs and formatter are unchanged. The formatter
selects saved gamma means, joint-premium medians and unique-leading counts;
it performs no matrix analysis. From this directory:
python3 format_sensitivity.py --coverage spectral_dimension_coverage.json --output /tmp/spectral_sensitivity_main.tex
The output path must be new. Compare its output with
paper/generated/caption_sources/v062/spectral_sensitivity_main.tex, the preserved
upstream table. The current paper/generated/spectral_sensitivity_main.tex has
only a later caption-navigation edit, recorded in its v063.provenance.json.
All numerical table entries are identical.

The witness table follows the same caption-only chain: its original source
receipt is preserved with the prior table under generated/caption_sources/v062/;
the active v063.provenance.json records the final caption change.

Conceptual figure (no empirical input):
python3 generate_concept_figure.py --output /tmp/evaluation_targets.pdf
This requires ReportLab. The existing figure and all scientific outcomes remain unchanged.
