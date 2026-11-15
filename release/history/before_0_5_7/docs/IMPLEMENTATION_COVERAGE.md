## Current revision — 24 October 2026

Version 0.4.0 integrates the completed returned replay as reviewed descriptive evidence. Compact/context and secondary extension results are now available. Native competence still failed, structural power remains limited, and independent confirmation is blocked. Original frozen results and source capsules retain their identities.

For current instructions use the root README, `docs/REPRODUCTION_REVIEW_20261024.txt`, and `release/reproduction_review.json`. Rebuild tables offline with `python scripts/build_reproduction_assets.py`; no new calibration or full model replay is needed. Maintenance v2 defaults to checking only and requires a separate explicit choice before any execution. Prior dated instructions below describe earlier project states and should not be used to launch another run automatically.

## Preserved earlier documentation

# Prospective implementation coverage — 17 October 2026

**Current status (21 October 2026):** Calibration and excluded development analyses have completed; structural power is limited and the original native branch stopped at clean competence. Continue with [independent characterization v1](CHARACTERIZATION_WORKFLOW.md). Its frozen source is included, while new holdout results await completion and audit. The earlier protocol below is retained as scope/history, not an instruction to rerun the full pipeline.

This ledger links the accessible discussion, identification protocol and comparison protocol to executable implementations. It does not certify access to every historical ChatGPT turn. The preserved review-coverage record remains authoritative about that limit. No new pretrained-model research result is included in this release. Synthetic datasets and randomly initialized tiny models are technical validation, never fabricated empirical evidence.

The immutable historical implementation remains under `src/bsg`; every file in its 53-file source manifest is verified independently. New experiments are isolated under `src/bsg_next`. The original production protocol is `configs/prospective/full.json`; the frozen characterization contract now lives in `studies/characterization_v1/frozen_run/study_config.json`. Pre-outcome clarifications are recorded in `PROTOCOL_AMENDMENTS.md`. The complete planning documents are preserved in `docs/prospective/icml2027_identification_v1` and `docs/prospective/icml2027_comparison_v1`.

| Requirement | Implementation | Execution / interpretation |
|---|---|---|
| Y1 category, Y2 shared context, Y3 noise, Y4 equivariant, Y5 family-specific, Y6 unsupported ambient directions | `synthetic.py`, `estimators.py` | CPU calibration: 72 cells × 200 seeds; category-code rejection is recorded separately from statistical true-null error |
| Timing and runtime reserve | `synthetic.py` timing | 120 datasets before full calibration; budget from measured runtime, not a promised GPU-hour estimate |
| Independent true-null calibration | `gates.py`, `synthetic.py` negative | 12 cells × 5,000 independent seeds; complete refitting and eight-hypothesis Holm; one-sided 95% binomial upper ≤ .06 |
| High-signal recovery | `gates.py` | Y4 high-signal empirical sensitivity ≥ .90; interval reported |
| Crossed inference calibration and stress | `synthetic.py`, `gates.py` | 120 primary scenarios × 2,000; additional heteroscedastic stress reported separately; approximate .01-family diagnostic upper tolerance .02, not an exact universal theorem |
| Bootstrap sensitivity | `synthetic.py` bootstrap / crossed_bootstrap | Family and crossed diagnostics; bootstrap never nested inside the 60,000 formal negative fits |
| Historical 16-cell pilot and 240-cell diagnostic | `offline.py` | Read-only, SHA-checked private FULL archive and cell index; legacy checkpoint replay and new comparisons; never overwrite frozen evidence |
| Training-only centering and PCA | `structural.py`, `estimators.py` | Construction data only; q=16 primary, q=2/4/8/32/64 sensitivity |
| Strong category/centroid/scalar/source-conditioned controls | `estimators.py`, `structural.py` | Oracle-label and independently decoded-label tracks; no test-label norm matching |
| Independent ridge, affine, Procrustes, residual, generator and joint fits | `estimators.py`, `structural.py` | Scale-normalized registered ridge grid; Moore–Penrose tolerance; joint/algebra-regularized results secondary |
| Context differences and residual energy | `structural.py` | Within-family paraphrase differences, context variance floor .05; ineligible hypotheses retained with p=1 |
| Legal action refits | `groups.py`, `structural.py` | C7 120 generator images; D4 three corner-action images; low-dimensional exact fits are not automatically semantic evidence |
| Maximum test pool sealed before pilot | `data.py`, `pipeline.py` | Maximum 64 family and 24 new-entity pools; split identities and prompts hashed before pilot outcomes |
| Grammar sensitivity | `grammar.py` | Per-root effects and leave-one-root-out family/orbit summaries under the fixed generator; no new authored-grammar or population claim |
| Two paraphrases and family sampling | `data.py` | IID scenario ASTs under fixed grammar roots; grammar-root sensitivity required; do not call generated ASTs independently authored grammars |
| Six pinned base models and four depths | `configs/prospective/full.json`, `hf.py` | Frozen model/tokenizer commits; block outputs; exact source-token span or declared final token |
| S3 role/entity transfer | `data.py`, `structural.py` | Separate context-family and entity-triple axes; final-token carrier and frozen entity potential field |
| Zero-shot potential and equal support | `hf.py`, `structural.py` | Input embeddings with no transformer forward; four dedicated support contexts/state/test entity, separate from scored targets |
| Embedding floor and S3 zero-rank diagnostic | `embedding.py` | C7 q=6 and D4 q=3; fixed-token S3 rank diagnostic; no native-computation claim |
| Main eight hypotheses | `statistics.py` | .04 Holm, prediction/context superiority .01 and equivalence ±.005; family t primary, bootstrap sensitivity |
| Crossed S3 four hypotheses | `statistics.py` | .01 Holm, family × entity units; conservative df=min(F−1,O−1); approximation and sparse-pilot limits reported |
| Power before confirmation | `statistics.py`, `pipeline.py` | Pilot-derived covariance, 32/48/64 families and 16/24 S3 test orbits; 90% marginal / 80% joint lower Monte Carlo bounds under stress |
| Native computation and noncommuting D4 | `data.py`, `native.py` | C7 and asymmetric free eight-pose D4; native prefix and continuation; early edit, natural counterfactual and correctly conjugated late edit |
| Competence before effects | `native.py` | Validation clean full-answer accuracy ≥ .80 including transfer-length words; no test-correct-only selection |
| Native power and freeze | `native.py` | Actual pilot contrast covariance; .10 planning effect / .05 sensitivity; freeze layer/readout/baseline/size before test |
| Native strong controls | `native.py`, `peers.py` | Identity, delta-zero sham, legacy mean overwrite, donor, centroid, source displacement, shrinkage, affine, Procrustes, eight kernel bandwidths, decoded centroid, five fixed random seeds |
| Full-answer outcome | `hf.py`, `native.py` | Sequence candidate scores and unconstrained full decoding; first-token and distance metrics are secondary |
| Native four hypotheses and practical margin | `native.py` | .05 point gain plus positive Bonferroni 98.75% lower CI, family t/Holm; entire decision calibrated in power |
| Native capacity sensitivity | `native.py` | q=2/4/8 at the frozen primary sites, own construction/validation fits, held-out long words; rank-2 action versus rank-2 calendar plane, primary q16 unchanged |
| Cached local-displacement control | `structural.py`, `estimators.py`, `offline.py` | Eight validation-selected kernel bandwidths; separate secondary target/family contrasts, not folded into analytic linear-estimator calibration |
| Dose and transfer words | `native.py`, `data.py` | 0/.25/.5/1/2 sparse-family dose panel; 1–4 construction lengths versus 6–8 transfer lengths |
| Operation and joint-variable edits | `native.py` | Own construction-fitted carriers, separate from state-carrier edits |
| Pathway and matched random/rescue | `hf.py`, `native.py` | Actual MLP/attention/block hooks; development-only fixed candidate selection; zero/resample/mean and rescue controls |
| Nonce rebinding and task shams | `native.py` | S3 crossed new entities/contexts, fixed token, natural continuation; task controls scored explicitly |
| Engels calendar comparator | `peers.py` | Two-dimensional circle inside PCA5; original complement ablation and matched complement preservation separately |
| Wurgaft manifold comparator | `peers.py` | Cyclic spline, 50-point paths, matched linear paths, Hellinger energy; D4 is unsupported, not assigned an artificial failure score |
| Original-task anchors | `peers.py`, `peer_anchors.json` | Original model families pinned; bounded independently implemented calendar intervention anchors, not complete author-code reproductions |
| Conditional mean and factorial ANOVA | `peers.py` | Dedicated complete state × operation × operation grids; construction-fitted tables; full-D and rank-matched edits; test ANOVA descriptive only |
| Held-out cross-model transport | `crossmodel.py`, `wordfields.py` | Whole-family/entity/word split; ridge, Procrustes, identity where dimensionally meaningful, nulls and rank/conditioning diagnostics |
| Hu equation 12 | `crossmodel.py`, `wordfields.py` | Source-model test displacement allowed, target-model test displacement excluded from fitting; absolute weight denominator; Gaussian RMS baseline and refitted weight null |
| Word-field support and limits | `word_field_vocabulary.json` | Controlled 128-word adaptation, six contexts, 70/10/20 whole-word split; not the original stratified 1,000-word dataset |
| Held-out-element intertwiners | `crossmodel.py`, structural transport adapter | Generator-only constraints; composites held out; legal relabelings refit; singular/rank-deficient maps do not establish mechanism equivalence |
| Sun–Toneva Congruity | `crossmodel.py` | Actual externally supplied behavior-validated implementation families required; GetImpl/automatic mechanism discovery not implemented or claimed; an intertwiner is never relabeled Congruity |
| Product C4×C4 action extension | `extensions.py` | Real torus prompts/fit/evaluation; conditional separate phase and run identity |
| Learned-interface extension | `extensions.py` | Low-rank reconstruction, complete-orbit augmentation, consistency and group regularization; equal steps, validation weight selection, held-out targets, checkpoints |
| Third lineage | `third_lineage.json` | Separate pinned Gemma-2-9B structural replication; explicitly selected, not added after observing a favorable effect |
| Competence fallback | `instruct_fallback.json` | Separately registered native study, chat template preserved; only after base-model competence failure, never choose by observed intervention gain |
| External multilingual/authored banks | `data.py` import_dataset | User-supplied validated JSONL with explicit family/grammar/entity identities and complete action support; translations/authored datasets are not fabricated |
| Cache and resume | `extraction.py`, `io.py`, `pipeline.py` | Source/config/input identity, chunk checksums, writer locks, atomic stage state; changed science uses a new run folder |
| Result export | `reporting.py` | Production-only, confirmation freeze required; negative results retained; technical/synthetic fixtures cannot fill manuscript rows |
| Anonymous Overleaf | `paper/main.tex`, `sections/prospective.tex` | Historical evidence retained; independent characterization slots Awaiting audit; old native branch stopped. Official 2026 fallback; 2027 rules unverified |
| Real-checkpoint hardware smoke | `smoke.py`, `model-smoke` | Construction-only features, multi-token scores and zero-delta generation; permanently technical and never paper evidence |
| GPU launch and recovery | `runpod/next_*`, `PROSPECTIVE_RUNPOD.md` | Explicit user-run commands; no automatic cloud provisioning, weight download or paid run during release validation |

## Required evidence before stronger claims

Passing a program's execution check is not a scientific success. Calibration, independent transfer, comparison against the strongest validation-selected comparator, clean native competence, and causal counterfactual accuracy all retain separate decisions. A failed gate stops confirmation and produces a reviewable result. A passing result would support an intervention-relative representation claim within the registered task distribution, not a unique native algorithm or universal LLM mechanism.

The literature positioning remains constrained by Lenc–Vedaldi, Chughtai, Engels, Nanda, Wu, Sun and Sun–Toneva as documented in `LITERATURE_AUDIT.md`. Adding a reference does not by itself resolve overlap; the new identification and native-computation experiments address those gaps empirically. No implementation or experimental plan can guarantee ICML acceptance.
