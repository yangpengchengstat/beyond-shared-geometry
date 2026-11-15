# Publishing the public project

The github-source.zip expands to beyond-shared-geometry/. That folder is the
public research repository: code, configs, saved results, source identities,
tests, manuscript and historical scientific failure records. Existing author
metadata and MIT license are retained. It is not anonymous submission material.

No remote repository has been created, no Git history imported, and nothing
has been uploaded. The owner chooses account, repository name and visibility.
Do not upload the delivery's private/ directory, raw archives, tokens, model
caches or personal audit files. Use the separately prepared public ZIP.

Before committing, run `python3 -B scripts/verify_delivery.py`, inspect
`git status` and the staged diff, and confirm author and third-party rights.
Do not publish the enclosing full-delivery folder. Do not use historical
scripts/package_release.py to regenerate this editorial release: its assumptions
refer to its historical metadata. Use the supplied verified bundles until a new
release is deliberately prepared.

CITATION.cff carries the verified author/software baseline only; it does not
claim acceptance, a DOI, or an existing public repository. CI performs static
checks, not numerical replication or a manuscript build.
