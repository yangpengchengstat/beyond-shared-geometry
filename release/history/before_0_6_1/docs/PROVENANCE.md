> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# Repository Provenance

This public-facing canonical repository was migrated in September 2026 from a separate private research workspace. The private workspace retains the complete exploratory history, archived experiment pipelines, recovery artifacts, raw result bundles, and historical deployment packages. None of that archive is copied here.

Migration occurred only after the canonical config, experiment entry point, scientific safeguards, CPU tests, runtime checks, and deployment-package validation had passed locally. The source workspace HEAD at migration was `5a647ba9946217551e731efe8c14a0223e946872`; the canonical transition also contained validated working-tree changes that are represented by this repository's initial commit.

From this repository onward:

- Code history is represented by Git commits.
- Experiment history is represented by run IDs.
- Each run captures a config snapshot, source commit, resolved model revisions, `manifest.json`, and `environment.json`.
- Selected small publication artifacts may be promoted into `results/tables/` and `results/figures/` only after review.
- Raw outputs, activation caches, model weights, credentials, and historical archives remain outside Git.

This repository is the sole actively maintained codebase for the project. Parallel versioned repositories or experiment filenames are not part of the maintenance model.
