# Recovery protocol: 2026-10-19

## Immutable evidence and failure

The original BF16 CPU study is preserved as a failed prospective run. Both the
downloaded result ZIP and the complete run TAR matched their SHA-256 sidecars.
The TAR contains 555,949 members and 1,137,227,178 uncompressed bytes. No `.npy`
or `.npz` files occur in this CPU-stage backup; no prospective model activation
extraction stage was reached. The two different compressed archive sizes are
therefore not evidence of missing activations.

The original study completed 14,400 calibration datasets, 60,000 negative-control
datasets, and 240 historical archive cells. It stopped at the registered positive
sensitivity gate. The four required Y4 cells had joint successes of 200/200,
174/200, 198/200, and 165/200. The two noisy cells failed the unchanged 90% rule.
Cell labels C7/D4 denote independent seed banks here: each trial evaluates both
groups and both primary endpoints. They are not isolated single-group powers.

In the first noisy bank, context endpoints passed 186/200 and 187/200, while
prediction endpoints both passed 200/200. Requiring all four claims jointly
produced 174/200. In the other bank, context passed 182/200 and 183/200 and
prediction passed 198/200 and 199/200, producing 165/200 jointly. The family-level
context contrasts are skewed and variable. The observed joint-power failure is
not evidence of a CUDA, BF16, or memory fault. No direct pairing, clustering, or
standard-error implementation error was identified in this audit.

The negative-control summaries record no true-null rejections in their 60,000
datasets. The 120 primary crossed-scenario summaries satisfy their registered
upper-bound tolerances. These are checks under the specified generators, not a
general guarantee of error control at every inferential boundary. The failed
full gate was never relabeled PASS.

## Engineering repairs that preserve numerical decisions

1. Python allocation tracing was previously enabled throughout simulation. Three
   local same-seed comparisons measured 16.45x, 17.46x, and 16.93x slowdowns.
   Numeric result hashes excluding elapsed time were identical. Tracing is now
   disabled by default and available through
   `execution.trace_python_allocations=true`. No replacement estimator was used.
   These microbenchmarks do not establish a corresponding speedup for the whole
   cloud study, particularly archive refits or file I/O.
2. The pipeline now evaluates the four registered positive cells first (800
   datasets, preserving their original full-grid cell indices and seed mapping).
   It stops before the large negative/calibration/archive stages if this check
   fails. A second positive gate follows complete calibration; the final full
   gate remains mandatory for pilot and confirmation. Failures list every
   failed positive cell rather than only the first one.
3. Default CPU calibration no longer repeats historical archive refits. The
   original archive results remain evidence. A separate `retrospective` phase or
   explicit CPU `--include-retrospective` option retains the implementation for
   cases where a revised historical analysis is actually required. An explicit
   combined `full` run still includes retrospective diagnostics.
4. Simulation progress is written every 50 completed datasets per worker.
   Collection reports scan, hash, and compression progress to stderr. Archives
   use a temporary `.partial` name until the ZIP is closed; checksums follow.
5. CPU and GPU environments/launchers are separate. CPU work on detected GPU
   hardware, and a combined `full` run, require explicit opt-ins. HF transfer
   acceleration is off by default. The default runtime profile uses BF16 after
   the user's six-checkpoint A40 smoke succeeded. No model inference or paid
   cloud job was launched while developing this revision.

## Prospective scientific amendment

The old `configs/prospective/full.json`, its failed outputs, and all historical
source/result locks remain unchanged. `bf16_revision.json` is a new study with:

- 64 independent test families in synthetic calibration;
- 64 structural confirmation families, also enforced by the structural power
  candidate list `[64]`, rather than increasing only the simulation sample;
- independent positive-calibration seed namespace 2026102900;
- BF16 runtime, unchanged pinned models, train/validation splits, estimands,
  superiority/equivalence margins, alpha allocations, and four-claim rule;
- unchanged native-specific pilot/power procedure and crossed-orbit requirements.

A small development screen used 12 independent paired seeds starting at
2026101900. Joint detection was 10/12 at 32 families and 12/12 at both 48 and 64.
These records are permanently technical, excluded from paper promotion, and
insufficient to establish 90% power. They motivated a conservative 64-family
candidate, not a retrospective declaration that the old design passed.

The additional independent positive-only check used 800 fresh datasets, 200 per
required cell. All four cells had 200/200 joint detections. It took 128.45 seconds
on the local CPU. These seeds were separate from the small development screen.
This is not a complete revised
calibration or permission to skip negative/crossed checks and measured model
pilot gates. See `release/recovery_validation.json` for the current validation
status. No new pretrained-model scientific effects or publication outcomes are
claimed.

## Reuse boundary

The optional profiler repair does not change the tested numerical outputs, and
the historical estimator is unchanged. Nevertheless this release has a new
source identity and the 64-family design changes the sampling distribution.
Do not copy old identity files into a new run, rewrite their hashes, silently
import old 32-family calibration as 64-family evidence, or delete failures to
force a resume. New full calibration has not been performed by this build.
