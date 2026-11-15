# Artifacts and provenance

`results/registry.json` inventories the selected evidence exported from frozen runs. Public run-provenance files retain scientific identities and hashes while excluding private host locations. Selected original evidence belongs under `results/frozen/`, and selected portable derivatives under `results/derived/`. Generated publication tables belong under `paper/generated/`; figures and their source CSVs belong under `paper/figures/`.

`release/artifact_manifest.json` uses an `artifacts` list. Every entry contains:

| Field | Meaning |
| --- | --- |
| `path` | Repository-relative path to the delivered file |
| `sha256` | Hash of the exact delivered bytes |
| `source_run` | Frozen run identity or an explicit derived-source identity |
| `source_path` | Path within the original archive or portable upstream artifact tree |
| `role` | Evidence, publication table/figure, or generated manuscript asset |

Derivative records may additionally describe input hashes and transformations. A redacted provenance record or typeset figure is a derivative and must not be described as byte-identical original evidence. The original archive checksum, selected-file checksum, and derivative checksum refer to different objects.

`release/source_manifest.json` locks all exported numerical implementation/configuration files. `release/release_manifest.json` records the target year, fallback template year/status, anonymous paper mode, exported-source identity, and historical FULL commit. An exported tree can include later AFS tooling without implying that those files generated the earlier FULL run.

`scripts/verify_frozen.py` checks the bundled evidence registry. `scripts/check_release.py` checks source/artifact hashes, traceability of selected publication outputs, TeX/BibTeX/graphics/style closure, citation keys, and portable publication contents. It rejects private host paths, likely credentials, symlinks, escaping dependency paths, and untraced figure/table assets. These checks do not recover unavailable raw activations or prove the correctness of model-derived measurements.

When `release/pdf_qa.json` is present, the release check also verifies its compiled-PDF hash and complete manuscript dependency hashes, and rejects recorded QA errors. This catches a PDF left stale after source, bibliography, style, or figure edits. Reading that attestation requires only the standard library; creating it with `scripts/check_pdf.py` uses the optional paper/PDF environment.

`scripts/package_release.py` creates:

- `dist/github-source.zip`: the portable public source project, selected evidence, and preserved original MIT license/author metadata. It excludes Git history, caches, raw activation arrays, checkpoints, weights, credentials, and temporary bundles.
- `dist/overleaf.zip`: the anonymous manuscript dependency closure, bibliography, local official style, required assets, submission metadata, instructions, and compiled `main.pdf` when present. It excludes public repository identity documents.

Both archives have companion SHA-256 files and a `dist/release_report.json`. Archive members are sorted, have fixed timestamps/permissions, and are byte-deterministic within the same compression runtime for unchanged inputs. Extracting the paper archive must leave its source buildable without the public source archive or the original research folder.

The code archive is prepared for a later public GitHub release. Publishing it, pushing a repository, uploading to a conference, or launching models is a separate action. The anonymous manuscript uses the official ICML 2026 fallback and does not claim confirmed 2027 submission compliance.
