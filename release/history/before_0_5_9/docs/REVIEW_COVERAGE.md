# Review coverage and evidence limits

**Current addendum, 21 October 2026:** See [the project re-audit](PROJECT_REAUDIT_2026_10_21_V2.md) for the latest manuscript/theory/code assessment and completed characterization audit. The new metadata inventory covers 4,289 project files after excluding caches/tooling/Git and the current audit working directory; it is not a claim to have reread every byte. Exact selected evidence/source checks, independent coordinate calculations, code-path reviews and visual manuscript checks have separately recorded scope. The following 16 October record is retained as dated history; its conversation and full-archive coverage claims are not claims of a new complete reread in this audit.

Review date: 16 October 2026. This revision corrects the earlier delivery's incomplete literature coverage. It separates file integrity, scientific checks, manuscript revision, and access to historical discussions.

## Research directory

The historical research tree contained 1,604 regular files totaling 3,885,565,234 bytes, excluding Git internals, symbolic links, and this derived submission directory. All were hashed. The review read every member of 35 distinct archives; two additional archive files were byte-identical duplicates. Across the archives, 5,916 regular members were read and hashed. There were no file-read, archive-read, or parsing errors.

Duplicate content was identified by SHA256. Across loose files and archive members, 375 distinct text contents were extracted, 44 distinct PDF contents had all pages extracted, 4,040 distinct CSV contents received a complete row-width/header scan, and 1,216 distinct NPZ contents received directory/CRC validation. SVG/XML and the unique image assets were parsed or validated. Compiled Python caches and other opaque binary metadata were hashed rather than executed. Git objects were not treated as research artifacts. The file-level record is `release/historical_file_inventory.json`; archive-member and extracted-content records remain in the private review workspace.

These are coverage and integrity statements. They do not mean every historical measurement was scientifically rederived, every PDF was visually inspected, or every raw tensor element was independently validated. Duplicate historical manuscripts are versions of the same work, not independent experiments. The current publication's selected evidence has a separate source/transform registry and artifact manifest.

## Scientific checks

The code/configuration review traced group conventions, state and template splits, activation/PCA fitting, independent and joint objectives, all published metrics, AFS null constructions, interventions, cross-model fits, and resampling. Historical source identities are distinguished from the later exported source tree. The release keeps the numerical source and configurations byte-locked.

Frozen-manifest checks covered 47 FULL entries, 30 AFS-A/B/C entries, and 27 rank-aware AFS-D entries. The 49 extracted FULL files matched their archive. Completion ledgers and original retry records were retained. Published means were cross-checked from the archived outputs; all 960 AFS-B summaries were recomputed from 2.4 million permutation rows, all 12 global-additive bootstrap intervals were recomputed, and all 240 primary global-additive NPZ fields were inspected numerically. This did not rerun every historical bootstrap, train a new model, or reproduce GPU activation extraction.

The current build separately verifies all packaged evidence hashes, reconstructs figure/table assets from the curated data, runs CPU checks, compiles the manuscript, checks source/PDF binding, and tests extracted release bundles. New theoretical discussion and literature comparisons do not change frozen results or upgrade historical gates.

## Conversations and uploaded material

The app returned ten earlier conversations associated with the project in its recent-chat inventory, containing 39 accessible turns. Those available message bodies were reviewed across the audit. One assistant message is explicitly truncated at 20,000 characters. The API provides no older-page cursor for these cached histories. A conversation whose title concerns SVD, for example, exposes only recent attachment-download troubleshooting, so its earlier mathematical discussion cannot be reconstructed from the title. The inventory itself is limited to the 50 most recent chats globally; no archived ChatGPT chats were returned. Native-app inventory was also unavailable in this session.

Accordingly, **complete historical coverage of every conversation in the project is not established**. The accessible caches, earlier assistant recommendations, and their reference links are evidence to investigate, not authoritative scientific claims. Missing older turns and content beyond the truncation remain an access limitation; a complete exported project history would be needed to close it.

Nine accessible uploaded PDFs and one uploaded Overleaf ZIP were also inspected through their attachment paths. Every PDF page was extracted and the ZIP members were read. File bytes differ from some local builds, so equivalence was not assumed from filenames. These uploads supplement the local manuscript archaeology; they do not create new experimental results. Private transcripts and uploaded research-history files are excluded from public archives.

## Literature

The seven explicitly requested papers were verified against their original complete papers, with relevant methods, theoretical assumptions, experimental evidence, and appendices reviewed. Further references recovered from the accessible discussions were checked where they affect the manuscript. `LITERATURE_AUDIT.md` explains the comparisons, corrected earlier claims, and novelty implications; `release/literature_sources.json` records primary versions, URLs, and hashes. Background references are not described as a newly exhaustive systematic review.

## Preservation and submission status

The original 1,604 files retained their contents and modification times during the audit. Only the separate publication project is revised. The previous delivery is preserved as a local snapshot before replacement. No published repository is modified, no remote is pushed, and no model execution is implied by packaging.

The manuscript is an English anonymous draft with reproducible assets. Official ICML 2027 author instructions and templates could not be verified on the review date. The unchanged official ICML 2026 style is an explicit fallback, with its eight-page main-text limit used for local checks. This is an editable submission preparation, not a certification of 2027 compliance or scientific acceptance. Missing semantic-identification experiments are stated in `RESEARCH_ROADMAP.md` and are not silently represented as completed.


## Current project integration audit (21 October 2026)

The current local-only inventory and focused runner/manuscript/offline-tool review are documented in [Project integration audit](PROJECT_AUDIT_2026_10_21.md). Its 2,634-file inventory does not imply substantive reading of every file or complete conversation access. Earlier review counts below/above remain records of their own dated scope; they are not silently reinterpreted as current remote execution verification.
