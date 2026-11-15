# Read-only evidence and design indexes

These files index archived inputs or describe finite combinatorial designs. They contain no newly extracted activations, fitted estimators, or new LLM findings.

- `FULL_CACHE_AND_CHECKPOINT_HEADERS.json`: all 1,188 inspected NPZ headers, including the 888 structural checkpoint hash checks and 300 raw-cache identities.
- `CELL_INPUT_INDEX.json`: 444 historical structural cells and their saved input/split identities; 240 are the primary grid.
- `CACHE_COVERAGE.csv`: 300 raw activation caches. Repeated split/dimension checkpoints do not create independent raw data.
- `EMBEDDING_CACHE_HEADERS.json`: four saved AFS-D embedding inputs, with repository-relative paths.
- `CONSISTENT_ACTION_DESIGN.json`: exact consistent-action conjugation tables, with 120 C7 and three D4 action-image families.
- `enumerate_consistent_actions.py`: pure standard-library finite-design enumeration. Running it writes only its sibling design JSON; it does not access models or activations.
- `PRIMARY_SOURCE_MANIFEST.json`: twelve newly read full-HTML primary sources, with versions, URLs and snapshot hashes. Full source bytes remain in the private audit workspace and are not redistributed.
- `PREVIOUS_PRIMARY_PAPER_RECHECK.json`: prior primary-paper review manifest for the requested references and additional directly relevant papers; it states review scope and external reproduction limits.
- `HISTORICAL_CAUSAL_ADDITIVE.csv`: unchanged historical source table copied from `submission/icml2027/paper/figures/causal_additive.source.csv`. Its values are old target-logit metrics, not new accuracy estimates. The current release artifact manifest retains the original run/source provenance.

The canonical FULL archive is identified relative to the research repository in the implementation document. Its whole-file hash was verified previously; this planning task checked member/checkpoint identities and did not recompute the entire archive digest. Copies of indexes are derived audit artifacts, not replacements for immutable source archives. Personal host paths, raw arrays, model weights and private conversation transcripts are excluded.
