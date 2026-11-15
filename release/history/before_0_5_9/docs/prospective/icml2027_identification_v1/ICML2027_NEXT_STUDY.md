# Beyond Shared Geometry: the next decisive study

**Planning version 1; evidence cutoff: 16 October 2026.** This document proposes work. It contains no new LLM findings, synthetic outcomes, or acceptance prediction. The historical FULL/AFS runs and released submission remain immutable. Synthetic and sham experiments are disclosed controls with known ground truth, never fabricated observations.

## 1. Decision and research objective

The next investment should resolve an identification problem: **when does a recovered finite-group action convey information beyond a stable category code, and does that additional information predict counterfactual behavior in the model's own computation?** Repeating the existing extraction on more checkpoints cannot resolve this by itself.

The current evidence supports fitted action compatibility on fixed lexical states. It does not yet establish that the recovered algebra is a native computational mechanism or that its interventions improve on strong state-conditioned alternatives. The recommended paper should test that boundary, including cases where the diagnostic fails. A rigorous negative result can motivate a diagnostic contribution, but only if its empirical scope and useful new insight distinguish it from existing nonidentification arguments.

Use a working title such as *Beyond Category Transport: Testing the Identification and Causal Relevance of Recovered Actions in Language Models*. This is a proposed direction, not a replacement title for the released manuscript.

The available [ICML 2026 reviewer instructions](https://icml.cc/Conferences/2026/ReviewerInstructions) assess soundness, presentation, significance, and originality; they also recognize contributions from careful evaluation of existing methods. Official 2027 requirements were not verified at this cutoff. No deadline, template, page allowance, or acceptance probability for 2027 is assumed.

## 2. Why the current evidence needs a stronger test

| Finding from the archived implementation/evidence | Consequence |
|---|---|
| A centered, affinely independent code for n categories has rank n−1 and admits every permutation of those categories as a linear action on its span. | Small prediction/relation errors can arise without a semantic or native algorithmic explanation. This elementary construction is a calibration control, not a new general theory. |
| A state-conditioned displacement replaces the training category mean and preserves the source residual. Its increments telescope if the current state is updated. | Correct composition and noncommuting order effects do not alone beat the category explanation. Give this baseline correct bookkeeping. |
| Archived learned-minus-additive target-logit contrasts are −0.104680 for C7, CI [−0.226831, −0.014395], and −0.006674 for D4, CI [−0.029723, 0.012301]. | A useful effect against identity is not an advantage against the stronger additive control. These are archived metrics, not complete-answer accuracy or new estimates. |
| Historical FULL composition compares matrix products with another fitted operator; AFS-A provides stronger actual-target generator evaluation within a seen state vocabulary. | New claims must score actual held-out target activations and native answers. Do not equate algebraic self-consistency with target fidelity. |
| The old hook overwrites each source-span token with the same vector; identity skips that hook. | Add a matched mean-overwrite sham and a new delta-preserving hook before interpreting multi-token causal differences. |
| Historical causal rows score single-token continuations and reuse templates/splits. | Score complete answers; repeated rows and overlapping partitions are not independent replication. |
| AFS-B/C address their particular registration/correspondence nulls. The old cross-model map can be singular and uses the same elements for fitting/evaluation. | Neither establishes semantic identification or mechanistic equivalence. Keep those claims conditional and secondary. |

The archived causal contrasts are retained in [their historical source table](evidence/HISTORICAL_CAUSAL_ADDITIVE.csv). The raw FULL archive is usable: the read-only feasibility audit inspected 1,188 NPZ headers, verified 888 structural checkpoint hashes against the ledger, and checked 300 activation-cache metadata identities. Consequently, the first discriminating analyses need no new LLM forwards. This is integrity/feasibility evidence, not a rerun of the scientific results. See [implementation specification](FEASIBILITY_AND_IMPLEMENTATION.md).

## 3. Literature changes the contribution

The seven requested prior works remain essential: Lenc–Vedaldi establishes learned equivariance/equivalence evaluation; Chughtai, Nanda, and Wu establish important group-operation mechanisms; Engels establishes cyclic calendar geometry and interventions; Yuan Sun addresses action realisability, transport and composition; Sun–Toneva addresses interpretation equivalence under specified implementation/alignment assumptions. Cite their precise scope rather than claiming first recovery of actions, noncommutativity, or cross-model mechanisms.

The fresh primary-source search adds particularly close work: [Arora et al.](https://arxiv.org/html/2609.35970v1) already separates interaction terms and includes conditional-mean interventions and list controls; [Hindupur et al.](https://arxiv.org/html/2609.37680v1) distinguishes curved geometry from the linear directions used in a comparison task; [Todd et al.](https://proceedings.iclr.cc/paper_files/paper/2026/hash/82aec8518602748540a42b783468c94d-Abstract-Conference.html) studies rebinding and in-context group computation. These observations narrow the gap; none independently validates our data.

The proposed distinction is a **calibrated, prospectively evaluated connection between finite-action diagnostics, specific category-code alternatives, lexical/context transfer, and native counterfactual fidelity**. It must outperform or sharply characterize simpler explanations. One additional list control, one new group, or more model sizes is insufficient. The [deep-search report](TARGETED_DEEP_SEARCH_AND_NOVELTY.md) records sources, versions, overlap, and limits; it is targeted rather than exhaustive.

## 4. Ordered work packages

### E0 — Freeze claims and an honest analysis contract

Before new numerical analysis, create a versioned protocol, data/condition IDs, input hashes, analysis code identity, endpoint definitions and disclosure log. Keep old outputs unchanged. All existing-data analyses are retrospective/exploratory even if their code is frozen now. A new partition of previously inspected prompts does not become fresh confirmation.

Write one short table linking each proposed claim to assumptions and a falsifier. Examples: category-independent transport fails if centroid/shrinkage controls match it; semantic selectivity fails under coherent arbitrary-action refits; native-use evidence fails if only final-answer patches work; a unique algorithm is unidentified when competing algorithms predict the same interventions. Never promote a claim because a historical PASS/GO string exists.

### E1 — Calibrate on known synthetic counterexamples

Implement six data-generating regimes: category code only; category plus shared context; independent state-context noise; genuinely equivariant context interaction; family-specific transport; and an action specified only on the occupied subspace with arbitrary ambient extension. Evaluate the actual estimators, selection and reporting pipeline. The first three must not produce a claim of beyond-category contextual transport. The fourth is a sensitivity positive control; the fifth tests transfer failure; the sixth tests unsupported ambient claims.

Primary grid: two groups × six regimes × three noise ratios {0, .1, .3} × two conditioning levels {1, 30} = **72 cells**, with 200 independent datasets each after a small schema/timing pilot. Follow with the prespecified negative-pipeline calibration and power simulations in the [statistical protocol](STATISTICAL_AND_SYNTHETIC_PROTOCOL.md). These are CPU simulations with measured cost, not assumed cheap or simulated LLM performance. Save every seed, construction and expected behavior. Stop and repair the diagnostic if its false-positive calibration fails; use fresh calibration draws after any scientific change.

### E2 — Reanalyse the frozen activation caches against strong explanations

Start with **16 cells**: Llama-1B/Qwen-1.5B × C7/D4 × depths .4/.8 × splits 101/303. Then cover the **240 primary historical cells** if import/orientation/metric contracts agree. Saved q=16 PCA isolates estimator changes; raw hidden states permit a separate training-only PCA/rank analysis.

Compare identity, training centroids, source-conditioned centroid displacement, validation-selected residual retention, independent linear/affine ridge, orthogonal Procrustes, generator-only maps, and declared Joint/residual variants. Run oracle-source-label and train-only decoded-label tracks separately. Declare parameter count, supplied metadata, effective rank and tuning budget. A direct independently fitted linear map is ridge's hypothesis class, not a distinct competitor that ridge can meaningfully outperform.

Use two main diagnostics: independent linear ridge target prediction over the strongest validation-frozen category/context baseline, and independent ridge transport of **within-family context differences**, compared with zero/scalar identity transport. Their regularization is validation-selected; Joint, affine, Procrustes and alternative residual variants remain secondary. Also report training-derived scales, explained variance, singular values, conditioning and supported-subspace defects. The weak global additive mean cancels on balanced complete orbits; retain it only as a historical control.

Keep features and lexical labels fixed and refit under coherent alternative actions. Exhaust **120 C7 action-image families and three D4 image families**. The four-corner D4 all-element reference has minimum possible tail rank 1/3; 5,000 repeated draws cannot make it a .05-level semantic test. A rank also needs an exchangeability justification before it is called a p-value. Distinguish global consistent reassignments from inconsistent per-context scrambles. See [exact design tables](evidence/CONSISTENT_ACTION_DESIGN.json).

**Decision:** if category and legal-action explanations exhaust the apparent benefit, stop expanding the positive native-algebra claim. Finish the boundary/diagnostic study or revise the question; do not blindly repeat FULL on more checkpoints.

### E3 — Collect a new sealed structural benchmark

Use the six pinned historical checkpoints and four fixed relative depths .2/.4/.6/.8. Primary q=16 is fixed; rank failure is recorded. Secondary q=2/4 probes below-category-rank behavior; q=8/32/64 preserves broader sensitivity. None replaces a failed primary. Publish exact layer indices and model/tokenizer revisions.

Per domain, author/generate **12 additional pilot families**, then a default main bank of **96 families**, two paraphrases each, split **48 training / 16 validation / 32 sealed test families**. Separate underlying scenario/grammar families rather than splitting near-duplicate strings. The two paraphrases form one predeclared difference pair, not two independent families. Define the sampling generator if claiming population uncertainty; otherwise report a finite authored benchmark and resampling sensitivity. Power planning may enlarge the test to 48 or 64 families before confirmatory outcome collection.

| Domain | Full default row axis per model | Main test |
|---|---:|---|
| C7 weekdays, legacy state-span mean | 192 contexts × 7 states = 1,344 | New families, known state vocabulary |
| D4 corners, legacy state-span mean | 192 × 4 = 768 | New families, known state vocabulary |
| S3 entity-to-role binding, one final-position readout | 192 × 48 disjoint entity triples × 6 assignments = 55,296 | Joint new families and new triples |

S3 uses entity triples split **24/8/16**; fit only training families × training triples, select only validation × validation. The joint-new block has **6,144 rows/model**, but only 32 family and 16 orbit clusters. Never concatenate role-aligned hidden vectors as the primary measurement: that would engineer the desired permutation representation. Use one fixed final token after the complete assignment clause.

Zero-shot S3 is primary: no held-out triple centroids or target activation statistics. Add a strong **compositional potential-field** control trained on source entity input embeddings and role metadata, with capacity selected on validation; disclose its richer information. Equal-support adaptation is a separate secondary arm with explicitly charged support examples. It is not unseen-state recovery.

Default structural extraction is **344,448 prompts over six models**, excluding pilot/support and optional readouts. Its 72 float32 primary arrays have **15,049,162,752 bytes** of activation payload before overhead. Stream per model/task; do not retain whole-model hidden states. Runtime, VRAM, logits, retries, causal caches and optimizer workspace require measured pilot estimates.

For generator-only claims, train/tune only r for C7, r/s for D4, and two adjacent transpositions for S3. Evaluate excluded elements against actual targets; never use those targets for validation or a relation penalty. Provide category controls correct state updates under composition.

### E4 — Test native computation, with a separately registered causal study

New causal inference is necessary: old mean activations cannot produce a new downstream answer. Pilot the pinned Llama-8B and Qwen-7B on real state-plus-operation questions. Use C7 and a **new free eight-state D4 asymmetric-pose task**, distinguished explicitly from the old four corners. A public simulator fixes active transformations, axes, viewing convention and application order. Clean competence and tokenization must be adequate before interpreting patch failures.

The default plan is 16 construction / eight validation / 32 sealed scenario families per domain; two paraphrases and 12 balanced source/word cases per family yield **3,072 test recipient cases across two domains and two models**. Fit native carriers using native construction prompts. Freeze sites/alignments on development data. Old operators on changed prompts are a separate transfer condition. A validation competence threshold of 80% is a proposed feasibility rule, not an ICML standard; report all test cases rather than retaining only naturally correct answers.

Predict changed answers when an initial or intermediate state is replaced and the model continues the remaining operations. Distinguish w(g(i)) from g(w(i)) when they differ; a free D4 orbit removes per-corner aliases. Natural donor state/operation recombination and consumer/pathway tests should catch answer transplantation. Operation-carrier fits require their own training; state operators do not automatically implement operation edits.

Use natural donor interchange, matched-q centroid/shrinkage, decoded-label versions, affine/Procrustes/local-field controls, and fixed-seed random controls. Keep the same masks, sites, complement and allowed information. For a span, add desired_mean−live_mean to every live token; identity passes through the same hook with delta zero. Keep the legacy mean-overwrite sham separately. Freeze a dose grid {0, .25, .5, 1, 2} on development, not target-norm matching on test data.

Score **complete-answer exact accuracy** and full-sequence target-versus-alternative likelihood margin; include an other/invalid answer category and unrelated-output checks. A proposed practical planning target is a five-percentage-point advantage over the strongest validation-frozen baseline, with a multiplicity-adjusted lower bound above zero; report smaller, equivalent and inconclusive effects honestly. A late patch alone supports controllability; it does not establish native implementation. See [native causal protocol](NATIVE_CAUSAL_PROTOCOL.md) for sampling, counts and gates.

## 5. Statistical decisions

Primary structural endpoints are training-scale-normalized paired improvements in target prediction and context-difference transport. Practical superiority margin **.01** and equivalence interval **[−.005, .005]** are proposed design choices, not borrowed ICML thresholds. Their interpretation and power must be checked before registration. Low training context energy is an uninformative endpoint, not a favorable exclusion.

Average the fixed six models/four depths within each family; use the family as the main sampling unit. Allocate .04 FWER to the eight C7/D4 superiority/equivalence hypotheses and .01 to the four primary zero-shot S3 hypotheses, with Holm within each family. S3 requires both family and orbit uncertainty and dedicated calibration; a family-only t test is invalid for that population claim. The causal follow-up is a separate preregistered study, not a way to rescue a failed structural confirmatory family.

Use 12 excluded pilot families for variance/cost, not to choose a more favorable effect margin. Simulate power for test-family counts 32/48/64; the proposed structural design target is 90% marginal and 80% joint power for the four superiority contrasts under declared alternatives and stress distributions. Lock size before confirmatory outcomes; do not top up after looking at p-values. Report estimates, ordinary intervals, adjusted tests and conservative decision bounds. Nonsignificance does not demonstrate equivalence. All exact estimands, cutoffs, seeds and simulation criteria are in the statistical supplement.

## 6. Deliverables and execution order

| Order | Concrete output | Gate before the next large investment |
|---|---|---|
| 1 | Read-only importer, row/action/split manifests, unchanged historical metric replay | No unexplained schema/hash/orientation discrepancy |
| 2 | Synthetic generator, baseline library, calibrated endpoint code | Known category/noise regimes do not earn a beyond-category claim |
| 3 | Sixteen-cell diagnostic report, then 240-cell exploratory report | A useful unresolved scientific question remains; report failures |
| 4 | Prompt/simulator bank, pilot variance and throughput, power report, final preregistration | Test data sealed, counts/cost measured, all selections frozen |
| 5 | New structural benchmark with all methods and transfer strata | Claim follows its prespecified positive/negative/inconclusive branch |
| 6 | Native behavior/patch pilot, sealed causal follow-up, independent case replication | Prospective fidelity exceeds or meaningfully characterizes strong alternatives |
| 7 | Claim-limited English manuscript, reproducible assets, anonymous Overleaf package | Every empirical statement maps to immutable new or historical evidence; official 2027 rules rechecked |

Do not launch a full historical rerun merely because the files are old. Reuse archived activations for E2; **new prompt families, lexical orbits, token positions and native interventions require new extraction/inference**. New objectives/configurations get new run IDs. Extract once and reuse cached activations for declared estimators.

Indicative human work blocks are implementation/calibration, retrospective diagnostics, dataset/pilot/registration, new extraction/analysis, causal validation, then writing. Their durations depend on measured failures and throughput; no unmeasured GPU-hour or calendar promise is made. Independent literature/simulator checks can run in parallel, while confirmatory evaluation waits for frozen code and datasets.

The manuscript's main figures should show calibration failures/successes, operator-versus-strong-baseline prediction and context transport, joint family/orbit transfer, and native counterfactual fidelity including failure cases. Norm/rank, alternative-action, dose and model/depth results support those conclusions. Cross-model compatibility stays secondary unless tested with new held-out functional transfer and rank/conditioning controls.

**Recommended immediate action:** implement and verify the read-only importer, synthetic constructions and strong baselines; run the 16-cell retrospective pilot into a fresh output namespace. That first result determines which expensive positive or negative study is justified. This planning delivery itself has not run that pilot, fitted a model, altered frozen evidence, or published anything.
