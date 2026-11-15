# Publish the public project

Upload the delivery's `github/` contents, or extract `dist/github-source.zip`. Do not upload the enclosing owner folder, private records, build logs or parent result archives.

Run:

```sh
python3 -B scripts/verify_delivery.py
python3 -B scripts/verify_submission_surface.py
```

These checks use standard-library Python 3.10+ and do not run experiments. Study dependencies and commands are separate.

All distributed files are checked against GitHub's 100 MiB limit. Use Git or GitHub Desktop for the whole project: some records exceed the web upload's 25 MiB per-file limit. Explicit `.gitignore` exceptions retain the distributed coordinate NPZs while local activation caches remain ignored. See [GitHub's file-size guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

After reviewing the public tree:

```sh
git init
git add .
git diff --cached --stat
git status
```

Choose the actual account, repository and visibility before adding a remote. This is a local publication staging copy. Repository creation and upload must be verified separately. The published paper DOI is https://doi.org/10.5281/zenodo.24238471; it is not a software DOI.

Keep identified repository links out of anonymous review materials. MIT licensing, author attribution and third-party notices remain in this public tree.

To repackage after updating receipts and passing checks, select a new destination:

```sh
python3 -B scripts/package_submission.py --output ../dist-new
```

The packager checks member integrity, private paths, credential patterns, anonymous content and archive size. Historical reports retain their original scope.
