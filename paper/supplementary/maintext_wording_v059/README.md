# Caption-only presentation derivative

`format_caption.py` removes two pieces of presentation-history wording from the inherited core risk table. It reads the preserved TeX, changes only two declared prose strings, verifies unchanged numeric tokens, and writes a new table plus a source/output hash receipt. No results are recomputed. Run from the project root: `python analysis/maintext_wording_v059/format_caption.py`.

For an Overleaf-only checkout: `python supplementary/maintext_wording_v059/format_caption.py .` from the paper root.
