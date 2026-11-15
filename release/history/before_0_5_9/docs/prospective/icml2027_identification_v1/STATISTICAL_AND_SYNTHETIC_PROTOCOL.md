# Proposed statistical and synthetic protocol

**Status: planning only, 16 October 2026.** No extraction, fitting, model run or synthetic experiment was executed to write this document. A small enumeration of abstract permutations checked the combinatorial counts below; it used no research data. All numbers, margins and sample-size rules below are proposed design choices, not ICML requirements. Register the final version, implementation, data-generation specification, exclusions and analysis before accessing confirmatory outcomes. Keep the historical FULL/AFS/additive records immutable and give all new work separate run IDs.

## 1. Questions and scope of inference

The confirmatory question is whether transformations learned from activations predict **new context-dependent state variation beyond strong category/centroid predictors**, not merely whether a finite vocabulary admits fitted group actions. Separate three claims:

1. Predictive improvement over category lookup and source-conditioned displacement on newly sampled context families.
2. Recovery of nontrivial context-dependent transport, beyond zero or scalar identity transport of context differences.
3. Transfer of a frozen operator to genuinely new lexical orbits, with a declared amount of support information.

Successful relations, legal-action ranks or generator composition alone do not identify semantics or the model's native computation. The six historical models are a **fixed panel**. The principal average concerns these six checkpoints and the specified prompt-family generator; it is not an estimate over the population of LLM architectures. Llama/Qwen sizes sharing pretraining lineage are not independent architecture replications. Report each model, each architecture family and each group alongside the panel average.

Historical overlapping split seeds describe sensitivity of a reused dataset. Do not enter them as independent observations, pool their repeated test rows as new evidence, or use their narrow uncertainty as prospective power information without accounting for the overlap.

## 2. Prospective datasets and partitions

### Main context-generalization cohort

Use the original visible vocabularies for this cohort: seven C7 states and four D4 corner states. This deliberately tests context generalization with a seen vocabulary. Create **48 training and 16 validation context families**, plus a fresh confirmatory cohort of **32, 48 or 64 families**, selected once by the power rule in Section 9. Use **12 additional pilot families** excluded from all these partitions. Each family contains two paraphrases, constituting one context pair specified before outcomes are extracted. A family is one independently generated/commissioned scenario and grammar, not its two paraphrases relabeled as two independent families. The default bank is 96 families (48/16/32) and 192 templates per domain; reserve the maximum candidate test-family pool before the pilot and select its final prefix by the power rule before collecting outcomes. Produce complete state orbits for every template. Match entities and other nuisance attributes across states.

The same family IDs cross all six models, both groups and the four primary relative depths **0.20, 0.40, 0.60 and 0.80**. Average the four depths equally; no best-layer selection. If both tasks cannot be rendered by one family design, treat the group-specific families as separate sampling frames and retain the different IDs. Primary latent dimension is **16** where the training matrix has adequate rank. A rank deficiency is a recorded failure, not permission to choose a more favorable dimension. Dimensions 8, 32 and 64 are secondary sensitivity analyses, not replacements for a failed primary. Add q=2 and q=4 as explicitly secondary category-rank diagnostics below the centered-category bound where applicable; fit them from the same training raw activations, never select the primary dimension on their test results.

Deduplicate families against old templates and each other using both exact text and a documented grammar/source review. A shared root grammar remains one family even when it produces many strings. Publish the generation distribution and its limits. Generalization is to that distribution; hand-picked convenience examples do not support a claim about all natural language.

Fit centering, PCA, state centroids, ridge parameters and every regularizer only on training data. Validation selects hyperparameters. The pilot is excluded from training and final testing. Seal a preordered maximum candidate test pool, including family IDs, prompts and seeds, before pilot analysis; the power rule selects one prefix size before any confirmatory outcome is collected. Reserve candidate extra lexical orbits similarly. Unused candidates cannot be added after inspecting confirmatory results. A test target may appear only in a test loss, never in preprocessing, centroid estimation, norm matching, hyperparameter selection or exclusion decisions.

### Separate S3 lexical/orbit-transfer cohort

Use the proposed S3 role-assignment task: **48 disjoint entity triples**, partitioned **24 training / eight validation / 16 sealed test triples**, with all **six role assignments** for every triple. Cross the default 96 context families × two paraphrases with all 48 triples and six assignments: **55,296 prompt rows per model**, before any technical retry. Read one registered fixed final token after the full assignment clause; do not concatenate three role slots, which would construct a permutation representation in the measured feature itself.

Training uses training-family × training-triple cells; validation uses validation-family × validation-triple cells. Report three distinct tests: new families/known triples; known training families/new triples; and **joint new families/new triples**. The latter contains 32 × two × 16 × six = **6,144 rows per model** in the default matrix. Its independent axes are at most 32 families and 16 triples, not 6,144 observations. Freeze all operators/PCA on training cells. The eight validation triples are development data, not held-out test orbits.

Use two separately reported arms:

- **Zero-shot transfer:** no activation-derived statistics from a held-out triple are available before prediction. Use the PCA and operators frozen from training triples. A held-out-triple centroid oracle is unavailable in this arm and cannot be supplied to only one method.
- **Equal-support transfer:** provide exactly **four additional, disjoint support contexts per assignment state**, containing no evaluated action outcomes. Both operator and lookup methods may use these activations. Estimate the new triple's centroids from this support; do not refit its transformation matrices. Evaluation contexts are disjoint from support. Record that the support contains every state; this is adaptation to a new vocabulary, not an unseen-state test.

The primary transfer estimate is the joint-new cell. Zero-shot and equal-support address different estimands. The main eight-hypothesis C7/D4 family below does not authorize a confirmatory S3 claim: register the S3 outcomes, arm and an overall multiplicity allocation before collection. A concrete conservative allocation is **.04 to the eight C7/D4 hypotheses and .01 to a separately Holm-adjusted S3 family**, containing the prediction/residual superiority and equivalence hypotheses in the chosen primary arm. The alternative arm and the two easier transfer cells are secondary. This global allocation retains .05 familywise control when the component tests are valid.

## 3. Predictors and fair calibration

Write row activations as \(z_{t,x}\), the abstract action as \(a_gx\), and the training centroid for state \(x\) as \(\mu_x\). All methods use the same extracted rows and training-fitted coordinates.

| Predictor | Prediction for target state \(y=a_gx\) | Purpose |
|---|---|---|
| State centroid | \(\mu_y\) | Category lookup control; source/target labels explicitly supplied |
| Source-conditioned displacement | \(z_{t,x}+\mu_y-\mu_x\) | Preserves source context and replaces only state mean |
| Calibrated context retention | \(\mu_y+\alpha(z_{t,x}-\mu_x)\) | Includes centroid and displacement as \(\alpha=0,1\) |
| Independent operator | \(z_{t,x}R_g\) | Unconstrained linear transport; do not relabel generic linear fitting as native mechanism |
| Residual operator | \(\mu_y+(z_{t,x}-\mu_x)Q_g\) | Tests transferable context variation after removing training means |
| Joint operator | Same output form, with registered relation penalty | Tests regularization, separately from independent recovery |

The category controls are intentionally strong, label-informed diagnostics. If source labels are not provided in the target use case, report a separate train-only state-decoding version; do not hide oracle label access. The target label follows from the externally specified action, not model computation.

For zero-shot S3 transfer, a global six-state centroid may wash out entity identity and is a weak comparator. Add a **compositional potential-field baseline**: construct φ(i,o) by concatenating train-PCA-compressed frozen input-embedding vectors for the three source entities in their declared role order. This concatenation is a baseline feature only, never the primary measured activation. Fit f(φ)→z on training-family × training-triple cells; choose its capacity on validation × validation. Use ridge or a registered one-hidden-layer MLP with widths {32, 128}, with regularization/capacity selected without held-out triples. Predict z + f(φ(g i,o)) − f(φ(i,o)). It uses the observed source entities, role metadata and known action but no held-out target hidden state. Potential differences telescope under composition, so this baseline can compose and transfer new entities without fitting a native group circuit. Report its richer metadata/capacity explicitly and compare it as an explanatory control, not a fair same-information algorithm ranking. In zero-shot S3, include it in the validation-frozen strong-baseline set. Never substitute per-held-out-triple centroids estimated from test activations.

Choose \(\alpha\) from **{0, 0.25, 0.5, 0.75, 1, 1.25}** using validation only. Choose ridge \(\lambda\) from **{0, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 1}**, multiplying by \(\operatorname{tr}(X^\top X)/d\). Use float64 pinv(X) with rcond=10^-12 on the singular values of X, rather than its Gram matrix, for the zero-penalty candidate; record the cutoff and resulting rank. The primary full-activation operator O* is **independent linear ridge**, with its lambda selected on validation; affine, Procrustes, Joint and residual-centered variants are secondary procedures. Select the strongest category/context predictor C* separately for each model/depth/group using mean validation prediction loss; ties within **10^-8** select the simpler predictor and then the smaller-norm fit. The primary category control intentionally has oracle source labels to test a strong explanation; the primary linear map uses z and g. This is an explanatory comparison, not an equal-information algorithm ranking. Publish the separate train-decoded-label track. Freeze those choices for the primary analysis. Publish every candidate's validation and test loss, so a winner does not conceal a failed independent estimator. The co-primary context estimator Q* is independent ridge on the registered training difference pairs, with lambda selected only by validation difference loss. Joint and alternative residual variants do not replace this primary after outcomes are inspected. Secondary procedures need their own calibration before any future confirmatory promotion. For generator-only experiments, every fit and hyperparameter choice uses generator-only training and generator-only validation pairs; composite validation outcomes are sealed with composite test outcomes.

Report raw parameter counts, rank and ridge effective degrees of freedom. For a shared design matrix, a multioutput ridge fit has fitted-value degrees of freedom \(d\,\operatorname{tr}[X(X^\top X+\rho I)^{-1}X^\top]\), where the actual penalty is \(\rho=\lambda\operatorname{tr}(X^\top X)/d\); use the matching pseudoinverse convention when the penalty is zero. This is a descriptive capacity measure, not proof that two nonlinear/regularized procedures are identical in capacity. Generator-only and all-element methods have different training information and parameter budgets; compare them in their declared regimes.

For norm/capacity controls, give each comparator the same validation budget. A random map may receive a scalar gain or shrinkage selected on training/validation. Norms may be matched to a training-derived reference. **Never scale a comparator to match held-out target norms or select random seeds by test prediction quality.** Report both uncalibrated and calibrated versions. Do not claim that independent ridge beats a generic per-element linear predictor: those are the same hypothesis class. A useful algebra-specific contrast is joint regularization versus validation-tuned unconstrained ridge, or generator-only prediction versus equally informed alternatives.

Nested separation is needed because repeatedly choosing models or hyperparameters using performance estimates can itself overfit the selection criterion. [Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)

## 4. Co-primary estimands

### A. Improvement over category/context baselines

For each model/depth/group, freeze
\[
S=\mathbb E_{\rm train}\|z_{t,x}-\mu_{\rm all}\|_2^2.
\]
Use the train-only value as the denominator for every method in that cell. Define \(\ell_P=\|z_{t,y}-\widehat z^P_{t,y}\|_2^2/\max(S,10^{-12})\). Average equally over changed source-target pairs (a_g x != x), then templates within family. Identity and fixed-point transitions are separate diagnostics; their trivially unchanged targets can dilute the contrast.

Let \(C^*\) be the validation-frozen category/context baseline and \(O^*\) the validation-frozen operator predictor. The family-level paired contrast is
\[
d^{\rm pred}_{f,G}=\frac1{24}\sum_{m=1}^6\sum_{l\in\{.2,.4,.6,.8\}}
\operatorname{mean}_{t,x,g:a_gx\ne x\mid f,G}(\ell_{C^*}-\ell_{O^*}).
\]
The estimand \(\Delta^{\rm pred}_G=\mathbb E_f d^{\rm pred}_{f,G}\) is conditional on the fixed training data, chosen estimators and six checkpoints. Positive values favor operators. Also publish the explicit centroid-minus-independent and displacement-minus-independent contrasts; these cannot be replaced by a weaker baseline after inspection.

### B. Nontrivial transport of context differences

For the one predeclared paraphrase/context pair \((t,t')\) in a family, form
\[
u_x=z_{t,x}-z_{t',x},\qquad v_{g,x}=z_{t,a_gx}-z_{t',a_gx}.
\]
Training centroids cancel. Define \(V=\mathbb E_{\rm train}\|v_{g,x}\|_2^2\), estimated from the same fixed pair-generation rule. Compare \(u_xQ_g\) against \(\alpha u_x\), with \(\alpha\) selected from the grid above on validation difference loss. Include zero and identity in that grid: zero guards against apparent gains produced by shrinking independent noise; identity guards against merely preserving a state-invariant context component. Define pair losses using the **same train-only** denominator \(\max(V,10^{-12})\), and average the baseline-minus-operator loss as above to obtain \(\Delta^{\rm ctx}_G\).

Fit \(Q_g\) from training residuals/pairs only, with its ridge choice fixed on validation difference loss. Fit no state-specific test centering. As a secondary sensitivity analysis, use eight fixed training-anchor families selected by ID without metrics; average all anchor comparisons within each held-out family. Do not count the eight anchors as independent observations. Training-anchor subtraction can carry reproducible anchor/state offsets learned during fitting, so it is not the primary test of new context interaction. New lexical orbits cannot use same-orbit training anchors in the zero-shot arm. Subtracting a mean across all states of a held-out orbit from its source feature would use the future target and can create prediction leakage. If a neutral-context encoder is added, it must be an independently available input supplied equally to all methods and registered as a different protocol.

Require training \(V/S\ge .05\) for an informative residual endpoint. Below that threshold, report insufficient context variation and retain the rows; do not replace them with a favorable layer. For the fixed-panel mean, any ineligible primary cell blocks an all-panel context-transport claim and its equivalence claim; eligible-cell summaries are explicitly conditional secondary estimates. Variation above this floor alone is not evidence of structured residual information.

Context-difference success is still a predictive claim about observed contexts and vocabularies. It can be generated by an arbitrary stable state-by-context code; lexical/orbit transfer and native-task evidence are needed for broader interpretation.

These prospective fixed-scale losses differ from the historical mean of per-row normalized ratios. Preserve historical numbers and definitions; do not mix the two scales or apply the new practical margins to old results.

## 5. What leave-one-state-out can identify

There is no assumption-free unseen-lexical-state test on one finite closed orbit. If training source vectors span \(S\subsetneq\mathbb R^d\), two operators differing by a matrix that vanishes on \(S\) make identical training predictions but can disagree on an unseen vector outside \(S\). Group constraints may reduce the ambiguity under additional assumptions; they do not generally remove it. A known minimal representation or a known affine/simplex relation can make some special cases identifiable, but that is a declared prior, not discovery of unseen semantic structure.

Leaving out only the name in evaluation while its activation enters PCA, global centering, centroid estimation, other action targets, or relation fitting is not a clean lexical holdout. Removing every occurrence can also remove the equations needed to identify the action. Therefore use the new-orbit design above for the main lexical claim. An optional leave-one-state-out analysis must state exactly which sources/targets are withheld, the identification assumptions, and a rank/uniqueness check. A failure can reflect underidentification; a success can exploit a known finite code.

### The augmented-state centroid composition oracle

A centroid table can be made exactly compositional without any native group computation in the model. Carry a one-hot external state \(q_x\), update \(q_x\mapsto q_xP_g\), and decode \(q_xP_gM=\mu_{a_gx}\), where \(M\) stacks centroids. Repeated permutations satisfy the group law exactly. Equivalently, transport a finite categorical code by change of basis. Thus a centroid baseline should receive the same explicit state bookkeeping when comparing composed predictions. Denying it bookkeeping and declaring it noncompositional manufactures an advantage.

This oracle adds a label state and a known transition table. It is a useful sufficiency/identification control, not evidence that the hidden model has that mechanism. Report augmented dimensions and provided information. Do not compare the oracle's exact label-space group law to a hidden-space operator defect as if they were the same object.

## 6. Exact legal fixed-feature action references

### Construction

Keep every activation vector, template, state label and feature coordinate fixed. Vary the **action on state indices**, rebuild all training and evaluation source-target pairs coherently, and fit each candidate under the same training/validation rules. Never change only a reported element name while leaving incompatible pairs or relations in place. Apply one chosen action globally to every occurrence of a lexical orbit, across templates, models and depths. Independent per-template actions would destroy the shared-action hypothesis instead of providing a legal common-action comparison.

Use the canonical presentation \(C_7=\langle r:r^7=e\rangle\) and \(D_4=\langle r,s:r^4=s^2=e,srs=r^{-1}\rangle\). Fix the actual D4 convention in the implementation. The natural corner action used here has a fixed-point-free reflection generator; a diagonal reflection with two fixed corners is a different marked action class. Do not swap them silently.

The following counts were independently verified by enumerating permutations of seven/four abstract indices and checking orders, relations and image sets:

| Reference | Marked assignments | Distinct image subgroups | Assignments/image | Best possible lower-tail rank for an all-element invariant score |
|---|---:|---:|---:|---:|
| C7 faithful actions on seven states | 720 seven-cycles | 120 | 6 | 1/120 |
| D4 faithful four-state actions, fixed-point-free \(s\) | 12 | 3 | 4 | 1/3 |
| D4 faithful four-state actions, two-fixed-point \(s\) | 12 | 3 | 4 | 1/3 |
| D4 both reflection types combined | 24 | 3 | 8 | 1/3 |

Derivation: C7 has \((7-1)!=720\) seven-cycles. Each cyclic subgroup has six generators. D4 has six four-cycles \(r\); for each, four involutions outside \(\langle r\rangle\) invert \(r\), two of each reflection type. These produce three order-eight image subgroups of \(S_4\). All C7 marked actions are conjugate under state relabeling. D4 has two state-conjugacy classes of marked faithful actions, distinguished by the cycle type of \(s\); allowing the relevant abstract-group automorphisms connects them. This is not a collection of nonisomorphic abstract algebras.

An independently fitted direct loss averaged equally over **all** elements is invariant under an automorphism that just permutes those elements, assuming identical per-element fitting/selection. It consequently has the ties shown above. Averaging nonidentity elements retains these ties. For that score, compute 120 C7 image-subgroup representatives and three D4 representatives, or enumerate all marked assignments with their exact duplicate weights; both give the same reference distribution. If a shared regularizer or selection procedure breaks the invariance, verify and report that difference rather than assuming these score counts.

Generator-marked scores, including a particular generator-only training program, may distinguish marked assignments. They must use the full relevant 720/12-member space and state that generator choice matters; they do not inherit the all-element quotient automatically. Additional accidental ties can further reduce resolution. Use inclusive tails and a fixed numerical comparison tolerance of **10^-10 times max(1, |score|)**. Never deduplicate floating-point scores and then give each distinct number equal mass.

### Meaning and limitations

For lower-is-better score \(T\), an exhaustive conditional rank is
\[
p_{\rm ref}=\frac{\#\{a\in\mathcal A:T(a)\le T(a_0)\}}{|\mathcal A|},
\]
including the observed assignment and tie mass. This is an exact rank under a specified **uniform registration model**. It is a valid randomization p-value only if the observed registration really is uniform/exchangeable under the stated null. Natural weekday/corner semantics were not randomly assigned to frozen features. Their rank is therefore an exhaustive structural reference, not an automatic test of a semantic or mechanistic null. Exchangeability assumptions must be defended, not inferred from the word permutation. [Winkler et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/)

All 120 C7 and all three D4 image references are cheap enough to enumerate. **D4's all-element rank cannot reach .05 on one four-state orbit.** Five thousand sampled assignments cannot overcome the three-image support, and correlated templates do not multiply it. Do not make this reference a D4 pass/fail gate at .05. Group automorphisms are legitimate equivalent registrations; excluding them from a semantic-null tail can remove exactly the tie mass that must be counted.

Multiple genuinely separate orbit registrations can define a product reference, but only under an explicitly justified joint registration law. Apply a sampled registration consistently within each orbit. Do not pretend that repeated contexts of one orbit supply independent registrations. For a very large product space, use **19,999** preseeded Monte Carlo draws plus the observed assignment, with the appropriate observed-inclusive Monte Carlo test and reported Monte Carlo uncertainty. The familiar add-one rule does not repair a wrong exchangeability model. [Phipson and Smyth, 2010](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf)

A prospective randomized rule-card experiment can supply an actual registration mechanism: assign a legal mapping uniformly to an opaque vocabulary before prompts are generated. Its Fisher null must specify that the measured potential activation/features do not change with that assignment. Because changing a displayed rule can itself change activations, this is a sharp relation-insensitivity null, not a free license to hold arbitrary post-treatment features fixed under all alternatives. Record assignments and random seeds, and distinguish this experiment from the fixed-feature descriptive reference.

## 7. Dependence and uncertainty

For the main C7/D4 cohort, first form the paired family contrasts in Section 4. Compute the mean and sample standard error across confirmatory families. The **formal primary decision pipeline is the family-level one-sample t analysis with \(F-1\) degrees of freedom, followed by the registered Holm correction**. The **10,000-replicate whole-family paired bootstrap is a real-study sensitivity analysis only**; bootstrap significance does not replace or override the registered t/Holm decision. Resample one family ID jointly across all models, depths, actions, states and methods. Do not independently resample a template within each model: those crossed repeated measurements share text. Pairing is retained in every replicate. If the estimated SE is zero/nonfinite, retain the point estimate and flag a degenerate endpoint; assign p=1 rather than creating infinite significance or quietly adding a favorable numerical epsilon. Analytical noiseless oracle checks remain separate. The t analysis is an independent-family, finite-variance approximation, not a finite-sample distribution-free guarantee; its calibration is evaluated prospectively in Section 10.

For a human-authored **fixed** family bank, its held-out mean is a finite-benchmark quantity. Bootstrap intervals then describe sensitivity to family reweighting; they do not guarantee coverage for an unspecified population of future prompts. Population p-values/intervals below require a specified family sampling generator and defensible independent sampling/exchangeability assumptions. A random split of a hand-picked bank alone does not supply that future-population guarantee. State whether the main study uses that generator or the finite-bank interpretation.

The primary estimand is conditional on the fixed training corpus and fitted models, so the bootstrap does not refit training. A separately labeled training-robustness study may resample whole training families, refit all methods and evaluate the same sealed target population. Its replicates are sensitivity analyses, not additional independent tests of the original test corpus.

For the S3 cohort, contexts and lexical orbits are **crossed**, not nested. Its primary analysis cannot use the C7/D4 family-only standard error. First aggregate to a balanced cell contrast D_fo, averaging the fixed models/depths, paraphrases, assignments and actions within family f and held-out triple o. Under an explicit additive crossed random-effects model D_fo = Δ + A_f + B_o + E_fo, assume independent zero-mean family, orbit and interaction terms with finite variances. Use

\[
\widehat{\mathrm{SE}}_{\rm cross}
=\sqrt{s^2_{\rm family\ means}/F+s^2_{\rm orbit\ means}/O},
\qquad \nu=\min(F-1,O-1).
\]

For the default joint-new cell, F=32 and O=16, hence ν=15. The variance expression deliberately retains, rather than subtracts, the interaction contribution in both marginal terms. Under the stated balanced independent random-effects model its expectation is conservative for the mean's variance; this does **not** make the resulting t reference exact. Use the same shifted t/TOST logic with this SE and ν, with Holm within the S3 **.01** allocation. With four S3 hypotheses this corresponds to conservative Bonferroni 99.75% one-sided superiority bounds and 99.5% two-sided equivalence intervals. This is a proposed approximate analysis, not finally registered as calibrated until dedicated crossed simulations validate it. It does not inherit calibration from family-only C7/D4 simulations. If the stated crossed model is unsuitable or calibration fails, do not issue the nominal S3 confirmatory claim; revise and preregister the method on fresh calibration seeds before collecting test outcomes.

For S3 sensitivity, use **10,000** crossed bootstrap replicates. Use independent bootstrap weights for whole family IDs and whole orbit IDs, applying their product to each row; share those weights across fixed models/depths/methods. Report per-orbit effects and sensitivity to each axis. A crossed bootstrap is approximate and can be conservative; it is not an exact bootstrap for arbitrary arrays. With only 16 held-out triples, retain that finite-cluster limitation. [Owen and Eckles, 2012](https://arxiv.org/abs/1106.2125)

Do not resample the six models as if they were six independent draws from a population of architectures. A broader model-population claim would require a separately sampled set of independently trained model lineages and a revised design/power analysis. Two architecture families do not support that inference.

## 8. Practical margins, multiplicity and decision rules

Proposed superiority margins are **0.01** for both normalized prediction improvement and normalized context-difference improvement. They mean reductions equal to 1% of the relevant training activation/context-difference energy. Proposed equivalence intervals are **[-0.005, 0.005]** for both outcomes. They are deliberately narrower than the superiority margins, leaving an intermediate region. These choices need scientific agreement before the pilot; they are not borrowed from an ICML standard or retrofitted to frozen effects.

The main confirmatory family has **eight hypotheses**: superiority beyond the relevant positive margin for two outcomes by two groups, and equivalence around zero for the same four contrasts. Use **Holm familywise error control at .05 across all eight** if these are the only confirmatory hypotheses. If the S3 family is also confirmatory, use the .04/.01 allocation above instead. An equivalence p-value is the maximum of its two one-sided p-values, not the minimum. Ineligible context endpoints receive p=1 and an uninformative status; do not remove them to ease the multiplicity correction. Report raw and adjusted p-values, effect estimates, ordinary 95% intervals, and conservative Bonferroni decision intervals. At the .05/eight-test allocation, these are 99.375% one-sided lower bounds for superiority and 98.75% two-sided intervals for equivalence. At the .04/eight-test allocation, use 99.5% one-sided lower bounds and 99% two-sided equivalence intervals. These practical criteria govern the wording of scientific claims, not publication eligibility. Equivalence requires that its interval lie inside the predeclared bounds; nonsignificance is not equivalence. [Holm, 1979](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf), [Lakens, 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/)

Use four possible interpretations per outcome/group: meaningful superiority; practical equivalence; evidence of inferiority if an adjusted interval lies entirely below the negative superiority margin; or inconclusive. Formal inferiority tests are not added to the eight-test family after inspection; interval-based descriptions must be labeled exploratory unless preregistered in the family. A meaningful contextual-transport headline requires both prediction and context-difference superiority in both groups, eligible primary cells, and successful calibration. A failure in one group limits the claim to the other group if its own adjusted hypotheses pass. No pooled average may override a failed group.

Per-model, per-depth, secondary dimensional, relation, generator-word and lexical-arm results receive estimates and declared descriptive intervals. If formal claims are desired for them, preregister a separate multiplicity family before collection. Cross-model concordance is not six extra opportunities to choose a favorable headline. Keep the historical AFS pass/fail records separate from these new decisions.

## 9. Pilot, precision and sample size

Use the 12 new pilot families only to check extraction correctness, denominators, family variability, resource use and prospective power. Publish their results as exploratory; do not recycle them into confirmation. Pilot inspection may improve implementation, but any changed method triggers a new protocol hash. Do not change the practical margins in response to the observed pilot mean.

Choose the smallest \(F\) in **{32, 48, 64}** for which prospective simulation gives at least **90% marginal power for each of the four superiority tests and 80% joint power**, under declared planning alternatives \(\Delta^{\rm pred}=.02\) and \(\Delta^{\rm ctx}=.02\). Thus the alternatives are above the superiority boundaries, not on them. Also report equivalence power under exactly zero. Do not claim that power at the margin can be 90% for a level-.05 superiority test.

Simulate the joint family-level contrast vector using centered pilot variation, with variance inflated by **2**, plus a Student-t(5) heavy-tail sensitivity and a mixture with **10%** high-variance families. Preserve cross-outcome and cross-group correlations where estimable; also examine correlations 0 and 0.8. Use **5,000 simulation replicates per candidate F/scenario**, MCSE and binomial confidence bounds. Require the lower 95% Monte Carlo bound to meet the power targets in the primary planning scenario; report sensitivity failures. Calibration of the testing procedure is a separate requirement. The pilot is too small to establish a population tail law; these are transparent planning assumptions.

If no candidate meets the target, do not announce an adequately powered confirmation. Either register a larger independently funded design before collecting its outcomes, or run the fixed maximum as an explicitly limited precision study. Do not repeatedly add families until significance. For the S3 joint-new endpoint, simulate crossed family and triple random effects, preserving within-family, within-triple and shared-model correlations. Run the actual crossed-SE/df/Holm primary analysis inside each simulation; increasing prompt paraphrases does not substitute for the 16 orbit clusters. If the orbit effect dominates, expand the number of **new held-out triples** prospectively rather than counting more prompts from the same triples. The 32→48→64 family grid alone may not deliver adequate S3 power.

The final F is frozen once, before confirmatory extraction. Stop only for predeclared technical reasons, never for interim effect size. Report the achieved interval width even if the planned hypothesis fails.

## 10. Selection-free synthetic calibration

Follow an explicit aims/data-generating mechanisms/estimands/methods/performance structure. Ground truth is assigned by construction; success is not chosen by selecting attractive seeds or a dimension that makes the intended conclusion appear. [Morris, White and Crowther, 2019](https://doi.org/10.1002/sim.8086)

### Six data-generating regimes (Y1–Y6, distinct from the S3 task name)

Use the same complete C7 seven-state and D4 four-state actions, primary d=16 and the same train/validation/test family structure as the real study. Predetermine all random matrices and seeds.

| Regime | Construction | Required interpretation |
|---|---|---|
| Y1: stable category code | Generic centered state vectors \(\mu_x\), with no context variation | Exact finite-state algebra can be recovered. This must **not** count as beyond-category evidence; the residual endpoint is ineligible. |
| Y2: shared context nuisance | \(z_{t,x}=\mu_x+c_t\) in separate subspaces | Category algebra plus state-invariant context. Scalar identity transport should remove any supposed residual advantage. |
| Y3: independent state-context variation | \(z_{t,x}=\mu_x+\epsilon_{t,x}\), independent across unseen contexts/states | Zero/shrinkage residual baseline is optimal in expectation; fitting category structure must not produce systematic residual superiority. |
| Y4: shared equivariant context interaction | Concatenate a categorical code with \(c_t\otimes e_x\), then apply one fixed invertible coordinate map | A true nontrivial shared action transports both state means and context interactions to new c. Prediction/residual estimands can be positive. |
| Y5: family-specific transport | Same means, but independent family-specific changes of basis in the interaction subspace | Fits within a family can look structured; a single common operator need not generalize to a new family. Detect that limitation. |
| Y6: unsupported ambient behavior | Exact action on an occupied subspace, arbitrary transformations on an unused orthogonal complement | Data-supported recovery and global operator defect can disagree. Report support and rank; do not classify absent directions as identified. |

For Y4 use an explicit block representation: row state \(e_x\), one scalar context coefficient \(c_t\), and blocks \([e_x, c_te_x]\), padded to d=16. The action is \(\operatorname{diag}(P_g,P_g,I)\), transformed by the fixed invertible coordinate map. Use \(c_t\) centered with nonzero variance; disjoint families receive independent coefficients. This works for the natural D4 action on four corners as well as C7; it does not incorrectly treat four D4 states as eight group elements. Add observation perturbations only after generating the clean signal. Provide an oracle using the known operator, plus the real train-fitted estimators. A benchmark may distinguish Y4 from the listed negatives, while still failing to distinguish every imaginable semantic and categorical theory; say so.

For Y6 make the complement explicitly unobserved; the oracle global extension is deliberately nonidentifiable. Do not manufacture a positive theorem by selecting an extension with small global defect. For Y1, arbitrary full-rank or simplex codes are essential negatives **for the semantic interpretation**, although they are positive examples of algebraic recoverability. A calibration that insists they fail the structural score is miscalibrated conceptually.

### Staged computational protocol

The synthetic audits run the **actual primary fitting, validation selection, fixed-scale loss aggregation, eligibility checks, family-t/Holm decision pipeline**. They do **not** nest 10,000 sensitivity bootstrap replicates inside every synthetic dataset. Sixty thousand negative-audit datasets times 10,000 resamples would be an unnecessary 600 million bootstrap analyses. The real-study sensitivity bootstrap is computed once for the real final dataset. S3 simulations use its own crossed-SE/df/Holm primary procedure, not the family-only one.

Execute these stages in order, with a separately frozen seed namespace for each:

1. **Schema and timing pilot:** ten independent datasets for each of the six regimes and two groups at d=16, condition number 1 and noise ratio .10: **120 datasets**. Check schema, exact analytical cases, row counts, no test-derived calibration, and deterministic reruns. This is technical calibration, excluded from formal rate estimates. Measure feature generation, fitting, validation, aggregation, primary inference and optional bootstrap time separately. Record median and 95th-percentile CPU seconds, peak memory, linear-algebra thread count and hardware. Do not extrapolate from one favorable dataset.
2. **Broad calibration:** the prespecified **72 × 200 = 14,400** dataset grid below. Every dataset executes the primary pipeline; no inner sensitivity bootstrap. Store all selected parameters, estimates, test statistics, raw/adjusted p-values, eligibility and failures.
3. **Formal negative audit:** the prespecified **12 × 5,000 = 60,000** end-to-end primary-pipeline datasets below. It needs no inner 10,000 bootstrap. Include both groups in each joint decision vector so the registered Holm family is real; designate one group's regime/conditioning by the cell index and use a prespecified companion-group regime. Use matched-negative companions for half the replicate IDs and high-signal Y4 companions for the other half, reporting them separately to examine complete and partial nulls. Do not manufacture independent observations from a shared generated dataset.
4. **S3 crossed calibration:** before its .01 confirmatory allocation is used, run **2,000 primary-inference replicates per cell** over F in {32,48,64}, O in {16,24}, and family:orbit:interaction variance ratios **{1:0:1, 0:1:1, 1:1:1, 1:4:1, 4:1:1}**; include normal and Student-t(5) standardized random effects and prespecified missing-cell/heteroscedastic sensitivity. Use four correlated endpoint vectors with correlations 0 and .8 as separate cases. Test null boundaries and stated alternatives. This is a separately timed statistical calibration using full crossed tables and the actual S3 inference module; add representative full-feature S3 fits to verify that table construction matches the end-to-end system. Do not call the family-only negative audits validation of S3. Report type-I error, coverage and power with Monte Carlo uncertainty; 16 orbit clusters remain a limitation even if these scenarios calibrate.
5. **Bootstrap coverage sensitivity subset:** use a separate fixed **50 dataset seeds in each of 12 negative cells**, with **999 resamples per dataset**, only to assess whether the chosen bootstrap sensitivity behaves broadly as expected. For S3, use 50 fixed seeds in each of the five variance-ratio settings at F=32/O=16 and normal effects, with 999 crossed resamples. These are deliberately smaller, prespecified diagnostics with reported Monte Carlo uncertainty; they do not replace primary calibration and cannot justify exact rare-tail coverage claims for a 10,000-resample real-study analysis. Increasing this diagnostic subset is allowed only by a predeclared budget revision, not by selecting failed/successful seeds.

**Measured CPU projection before scaling:** after Stage 1, project Stage 2/3 cost from per-cell timing (not merely resample count). Publish total projected CPU-hours, expected wall time at the measured worker/thread count, peak aggregate RAM and disk footprint, plus a **2× runtime reserve**. Estimate Stage 4 separately because S3's crossed tables and feature fits differ. Time the 999-resample subset independently. If projected resources are insufficient, freeze a smaller explicitly exploratory study or obtain resources; do not silently cut formal replicates, drop costly null cases or relabel a partial audit as passed. Parallelize across independent seeds with fixed numerical-thread settings, checkpoint by immutable seed ID and count failed/retried seeds in denominators.

### Fixed calibration grid and reporting

Use **two groups × six regimes × observation-noise ratios {0, .10, .30} × coordinate condition numbers {1, 30} = 72 primary cells**, with **200 independent dataset seeds per cell**. Define noise ratio as expected squared perturbation divided by clean signal energy, not an ambiguous standard deviation. Use fixed singular-value schedules for the coordinate maps; report them. Fix the relative context-signal energy at **0.5** in eligible regimes. At noise zero, Y1/Y2 should satisfy their analytical limits up to numerical precision. Prespecified secondary dimension/signal sweeps use d in {8, 32, 64} only when the constructed signal rank fits; do not truncate a true representation silently to make d=8 work.

For a formal false-positive audit, additionally run **5,000 full-pipeline replicates** in the **12 preregistered negative cells Y1/Y2/Y3 × two groups × two condition numbers at noise ratio .10**. Count any spurious superiority and contextual/beyond-category headline in the designated negative group using the complete multiplicity rule, not merely a low structural error. Distinguish **statistical Type-I error** from a failure of semantic identification: after fitting/selection, compute the true conditional expected squared losses from the known synthetic first/second moments (or a prespecified independent oracle Monte Carlo set with its error reported), and identify which margin nulls are actually true. A pure category generator can still give one finite-sample estimator better prediction than another; that is an identification counterexample, not automatically a statistical false positive. Never label every rejection under a category DGP a Type-I error without checking the estimand. Report both quantities. Require each cell's one-sided 95% binomial upper bound to be at most **.06**. Y1 ineligibility must remain in the decision logic. This checks approximate calibration in the specified regimes, not universal Type I control. If it fails, correct the method and preregister a new untouched calibration seed batch before real confirmation.

When the real-family bank is fixed rather than sampled, this simulation calibrates the stipulated synthetic/sampling model, not unspecified prompt-population inference.

The high-signal Y4 cells at condition number 1 and noise ratios 0/.10 are the prespecified positive sensitivity check: require at least **90%** end-to-end detection with its Monte Carlo interval reported. Other Y4 cells test degradation rather than being discarded for failing. Report all 72 cells, estimation bias, coverage, false-positive rates, detection, rank, conditioning and runtime, including failed fits. Use Wilson/exact binomial intervals for rates; report Monte Carlo standard errors. Mathematical oracle checks use a relative tolerance **10^-8** for well-conditioned noiseless cases; condition-30 cases receive a predeclared forward-error bound based on the exact generated matrices rather than a relaxed, retrospectively chosen tolerance.

Use **seed_base=2027101600**, with seed = base + 100,000 × cell_index + replicate_index, lexicographically ordered cell IDs and replicate_index < 5,000. Publish the mapping and assert uniqueness. Use distinct bases **2127101600** for power simulations and **2227101600** for the formal negative audit. Show a fixed first seed and aggregate plots, not a selected illustrative seed. Hyperparameters are selected by the same training/validation rule in every synthetic cell, including null candidates. No test-set norm matching or selection of the most successful synthetic generator is allowed.

## 11. Generator, relation and behavioral follow-ups

Generator-only assessment uses only generator pairs to fit operators; composites are evaluated on held-out families. Declare row multiplication order, word list and equivalence of each word to a group element. Keep cycles and seen generators out of the unseen-composite mean. Longer words representing the same element are repeated measurements, not new independent algebraic targets. For D4 compare correct and wrong words matched in length and intervention budget, and list collisions explicitly. Whole-family uncertainty remains the unit of inference.

A native behavioral follow-up must actually implement the registered sequence of interventions/forward continuations if that is the claim. Externally multiplying matrices and patching once is a different procedure. Use complete answer-sequence log probability and decoded accuracy with exact tokenizer metadata, rather than assuming that first-token logits represent a multi-token label. Predeclare strong source-conditioned additive, centroid/state-bookkeeping, identity, random and wrong-order comparators and choose any patch amplitude on validation only. This behavioral program needs its own outcomes, margins and multiplicity family; do not add unregistered behavior endpoints to rescue a failed representation result.

## 12. Required implementation outputs before execution

The final preregistration must contain prompt/family/orbit and support manifests; train/validation/pilot/test IDs; exact model and tokenizer revisions; extraction/span convention; shared PCA hashes and ranks; metric equations; all candidate and null constructors; finite-action count assertions; hyperparameter/tie rules; estimand-weight checks; family/orbit dependence rules; pilot/power code and selected F; synthetic seeds and expected analytical cases; technical failure/retry policy; immutable source commit and package hash. Store every candidate's fit status, selected hyperparameters, training-only scales and row keys. Missingness is never silently converted to a success or zero error.

Before GPU collection, an independent implementation review should confirm the permutation counts, all-element automorphism ties, row/word conventions, no held-out preprocessing, the augmented-centroid oracle, analytical Y1/Y2/Y4 cases, and whole-family resampling on a small artificial table with known dependencies. These checks validate a concrete protocol; they do not execute or alter the historical research record.
