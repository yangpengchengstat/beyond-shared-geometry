# Publish the public project

The separately supplied `github-source.zip` expands to `beyond-shared-geometry/`.
That folder is the public research repository: code, configurations, retained
measurements, source identities, tests, manuscript and historical failure records.
Existing author attribution and MIT licensing are preserved. It is not anonymous
review material.

## Before the owner uploads

- Run `python3 -B scripts/verify_delivery.py` from the extracted repository.
  This requires Python 3.10+ but no scientific packages, GPU or model access.
- Read `docs/DATA_AVAILABILITY_CURRENT.md`. The release contains supporting scalar
  evidence, not all original activation caches. No public data link is invented.
- Inspect `LICENSE`, `CITATION.cff` and `THIRD_PARTY_NOTICES.md`; confirm the rights
  and author details as the repository owner.
- Upload only the public repository tree. Exclude the enclosing delivery,
  `private/`, owner-only verification logs, raw result archives, model caches,
  access tokens and downloaded reference papers. The supplied `.gitignore` keeps
  common local/generated files out of a normal commit.
- If peer review is ongoing, keep the anonymous submission free of identifying
  repository links. Review the journal's current anonymity rules before linking it.

The owner chooses account, repository name and visibility. No remote repository
has been created and no upload or push has occurred. For a new local repository,
ordinary Git staging may be used after inspection:

```sh
git init
git add .
git diff --cached --stat
git status
```

Review the staged changes before committing or adding the owner's actual remote.
Do not invent a repository URL or DOI in citations. These commands are instructions;
they were not run when preparing this delivery.

## Maintenance

Default `make check` and CI check identities, syntax and manuscript references only.
They do not compile, download dependencies or execute numerical tests. Model/study
launchers remain opt-in and require their own documented inputs and authorization.
A CI PASS is not experimental replication or final PDF acceptance.

The current packaging tool is `scripts/package_submission.py`; the legacy
`package_release.py` entry forwards to that tool, while the former implementation
is retained as provenance. After a successful static check, use a fresh destination:

```sh
python3 -B scripts/package_submission.py --output ../dist
```

This prepares the public, Overleaf and anonymous source archives plus SHA256 sums.
It refuses an existing output directory. Do not use it to imply compilation or
publication. To revise a bundle, update the declared current source/derivative
manifests and rebuild deliberately; do not relabel old hashes. Keep old outcomes
and failed runs unchanged.

## Execution source versus publication tooling

The context generalization study's frozen README records the pre-execution plan.
The completed authored-context result and its comparison protocol are documented
in the active manuscript and saved follow-up records, rather than in that historical README.

`release/prospective_source_manifest.json` remains the unchanged execution-era
source lock. Four engineering files have newer publication versions: CI,
`pyproject.toml`, the release checker, and the packaging entry point.
`release/engineering_source_bindings.json` binds each current derivative to its
exact hash-matching execution source preserved under `release/history/`.

The current release checker uses `scripts/verify_prospective_delivery.py` only
when that explicit binding exists. All other prospective sources must match their
active frozen bytes; scientific files cannot use a historical fallback. The
original strict `verify_prospective.py` is retained unchanged and still reports
the expected mismatch if invoked against these publication derivatives. This is
an explicit distinction between execution and distribution sources, not a claim
that the current CI or packaging wrappers ran the historical experiments.
