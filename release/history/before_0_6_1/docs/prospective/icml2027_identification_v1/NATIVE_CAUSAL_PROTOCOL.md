# Native causal validation: a separate preregistered study

Proposed protocol, 16 October 2026. No experiment has been run under this protocol. This study is separate from the structural C7/D4-corner/S3 experiment and its multiplicity budget. Original results and released sources remain immutable. “Sham” and synthetic controls are disclosed falsification experiments, never invented findings.

## 1. Question and claim boundary

Test whether group-action edits of an identified state carrier predict the counterfactual answer when the model continues its own computation, and whether they outperform equally informed alternatives. The primary comparison concerns state/intermediate substitution. Operation-only and joint-variable interchange require separately identified operation carriers and belong to a secondary mechanism-specific stage.

Recovering matrices, steering weekdays, adding noncommuting operations, and using natural donor patches are not individually novel. Successful edits support an intervention-relative carrier or abstraction; they do not prove that the transformer multiplies the fitted matrices, exclude all finite lookup implementations, establish necessity, or identify a unique algorithm.

## 2. Tasks, simulator, and model gate

Start with pinned base Llama-3.1-8B and Qwen2.5-7B checkpoints. Record exact revisions, tokenizers, precision, evaluation wrappers, and code hashes. Use familiar C7 weekday arithmetic and a new D4 **free eight-state asymmetric-pose task**, distinct from the archived four-corner action. Four-corner D4 is globally faithful, but individual corners have stabilizers; a free pose orbit makes every nonidentity edit change the complete pose.

The public simulator represents a square with four distinct corner marks. Fix the reference ordered tuple `(NW, NE, SE, SW)`, an active clockwise rotation, and reflection across the vertical image axis. A word executes left to right. Generate its full permutation table, verify identity/inverse/composition and `r^4=e`, `s^2=e`, `srs=r^{-1}`, and assert eight distinct poses. A pose answer specifies the complete ordered tuple; token-level coincidences never count as matching poses. C7 fixes the weekday origin, forward/backward primitives, and modular convention similarly.

Each domain gets **12 additional exploratory pilot families**, excluded from confirmation. Select an earlier consumer site as primary and a late answer-carrier site as positive control, fixing exact block, hook, and token/span rules. On eight validation families, require **80% unedited complete-answer accuracy in each model/domain cell**, including transfer. This proposed feasibility criterion may trigger exploratory use of instruction-tuned counterparts, with new checkpoint pins and fresh validation before sealing; never pool them with the original base pair. Otherwise revise the task or stop. Never retain only correct sealed-test items.

## 3. Families and word catalog

Per domain, reserve **16 construction, 8 validation, and 32 sealed test families**, with two paraphrases per family. Families independently vary scenario, vocabulary realization, object description, and instruction framing; related states, donors, and paraphrases stay together. Freeze family-generation rules after the pilot.

Lock a public 16-word catalog indexing the **native continuation/program `v`**: eight words of length 1–4 for construction/development, and eight length-6–8 words for transfer. Record primitive strings, reduced elements, inverses, and target predictions. Words may reduce to the same element; call the holdout **word transfer**, not unseen-element or unseen-state transfer. For C7, include forward and backward primitives. For D4, include rotation/reflection mixtures, cancellation, and equal-length order-sensitive pairs. Compare order only when simulator outcomes differ.

Each family supplies **12 source/word scenarios × two paraphrases = 24 recipient cases**: eight transfer-catalog cases and four rotating familiar-word cases at sealed evaluation. Carrier fitting and intervention-based site/hyperparameter selection use development words only. Unedited long-word validation checks feasibility; long words are therefore held out from fitting, not entirely unseen during design. Deterministic assignments rotate source states and familiar words across families, with marginal counts differing by at most one wherever feasible. Twelve cases cannot be exactly balanced over seven or eight states within each family; publish the actual allocation. Fix the algorithm and seeds before inspecting outcomes.

For the default test, `2 models × 2 domains × 32 families × 24 cases = 3,072 recipients`. Construction and validation provide 1,536 and 768 recipient-equivalent cases across both models/domains; additional paired fitting prompts are separately enumerated.

## 4. Counterfactual contract and native caches

A scenario records initial state `S0`, optional native prefix `p`, injected group element `g`, native continuation `v`, and simulator intermediate state `Sj=p(S0)`. Its counterfactual target is **`Y*=v(g(Sj))`**. The model itself executes catalog word `v`; it does not receive an externally computed product as its answer. Use one generator edit as primary: forward rotation for C7, and balanced rotation/reflection edits for D4. Composite external edits are a separately budgeted secondary condition. For `p=e`, this is initial-state substitution.

A natural counterpart keeps native instructions fixed and changes the initial state to `S0'=p^{-1} g p(S0)`. It therefore reaches `g(Sj)` before the same consumer. Cache original/counterpart prompts; match prefix and continuation token lengths within pairs. Distinguish this target from answer transport **`g(v(Sj))`**: D4 scenarios where they disagree test whether the edit precedes native continuation. Retain coincident-target cases in overall accuracy, but identify the disagreement stratum in advance. Donors preserve nuisance context where possible; include same-state/different-context and wrong-state/matched-context controls. Donors whose own answer differs from a later joint-variable recombination target help detect answer transplantation.

Fit PCA, state decoders, centroids, generator maps, and baselines using **native construction prompts at the exact selected hook**. An intermediate site must encode `Sj` without inserting its ground-truth value into the prompt, and its downstream consumer must perform the actual remaining program `v`. Patching an old source-word representation cannot establish this. Freeze sites and hyperparameters on validation. Archived span means are not substitutes for new prompts/sites; old operators are an optional named transfer condition.

## 5. Interventions and matched controls

For span-token activations `h_j`, compute the live mean `hbar` and proposed mean `hbar*`. The primary hook applies

`h_j' = h_j + alpha (hbar* − hbar)`.

This preserves within-span deviations and the recipient's orthogonal complement. Primary competitors share carrier, rank, site, dose, and token mask. Random subspaces preserve their own complements. The full-dimensional mean-overwrite sham is an explicit exception to the matched-rank transformation mask, using the same hook/span to isolate flattening.

At the earlier site and dose one, evaluate these **15 conditions** on every test recipient:

| Condition | Definition and information |
|---|---|
| Group action | Generator map fitted to native construction pairs; no sealed target activation. |
| Source-conditioned centroid | `z − mu_i + mu_gi`, with train-only means and oracle source identity, in the matched carrier. |
| Context shrinkage | `mu_gi + lambda(z − mu_i)`; validation-selected lambda, same oracle information. |
| Affine ridge and orthogonal Procrustes | Two independently fitted paired-map comparators, with validation-only regularization. |
| Local displacement regression | Oracle-source-conditioned kernel regression on construction paired displacements, as specified below; native `v` is executed by the model. |
| Label-blind centroid | Same centroid rule using a construction-trained source decoder; report decoder accuracy and error propagation. |
| Delta-preserving sham | Set the replacement mean to the live original mean, producing exactly zero displacement. |
| Legacy mean-overwrite sham | Replace each full-D span token by its live full-D mean; measures historical flattening. |
| Natural donor | Projected counterpart activation; an oracle validity/reference condition, excluded from trainable-baseline ranking. |
| Five random subspaces | Rank-matched donor patches with seeds fixed before testing. |

Use native rank 16, subject to construction rank; freeze any alternative-rank rule on validation. Record retained state/context variance, capacity, and norms. Oracle-source baselines and the label-blind track answer different information-access questions.

For local regression, construction pair `j` has source `i`, residual `r_j=z_j−mu_i`, and displacement `Delta_gj=z_gj−z_j`. Predict `delta_g(z,i)=sum_j a_j Delta_gj`, with normalized weights `a_j proportional to exp(−||(z−mu_i)−r_j||²/(2 sigma²))` over same-source construction pairs. Select sigma from **{.125,.25,.5,1,2,4,8,infinity} times the construction median positive within-source pair distance**, using validation only. Infinity gives uniform weights and the state-conditional mean displacement; if no positive distance exists, use infinity. No sealed target activations enter. This is distinct from the structural S3 potential-field experiment.

Select `B*` from centroid, shrinkage, affine, Procrustes, and local regression on validation counterfactual accuracy. Freeze tie-breaking; never select it on test. Report every comparator's sealed result even if the winner changes rank.

Run group, `B*`, and natural-donor conditions at the late site, dose one. On a deterministic balanced quarter of test families, run sensitivity doses **0, .25, .5, 1, 2** for those three conditions at both sites; zero and one reuse their existing evaluations. Dose scales and the subset are fixed before sealing. Inspect collateral outputs and norm curves rather than using post hoc norm normalization as proof of fairness.

## 6. Endpoints and inference

The primary effect is the paired difference in **complete-answer counterfactual accuracy**, group minus validation-selected `B*`, at the earlier site and dose one. Score all recipients, including unedited failures. The planning target is an advantage of **at least five percentage points**, with a multiplicity-adjusted interval excluding zero. This means a five-point point estimate plus evidence of positive difference; claiming the true advantage exceeds five points would require the lower bound itself to exceed .05.

Average scenario accuracy within family; analyze paired family differences separately in each model/domain cell. Primary inference uses two-sided family-level t-tests with Holm adjustment across four cells at familywise .05, plus conservative **98.75% t-intervals** (Bonferroni .05/4, F−1 degrees of freedom). Require point advantage >=.05 and lower bound >0; disclose the finite-family approximation. A 10,000-draw paired family bootstrap is sensitivity analysis only, preserving common family draws across models. Its tail interval is not a mandatory claim gate. This separate preregistered study does not reuse the structural .04/.01 allocation; umbrella claims require an explicit combined policy.

Secondary endpoints include full-answer sequence log-probability margins against simulator-defined alternatives, constrained-choice accuracy, unedited competence, donor-reference fidelity, wrong-order specificity, and collateral task effects. Preserve “other” outputs. Avoid first-token-only scores, desirable-state-only normalization, and recovery ratios with unstable denominators.

Before sealing, simulate the **entire decision**, including point advantage >=.05, positive adjusted interval, and Holm tests, using pilot/development family variation and cross-model dependence. A true .05 advantage cannot deliver 90% power for a point-estimate >=.05 boundary gate. Therefore use **true advantage .10 as the planning alternative**, show .05 sensitivity, and target >=90% marginal power per cell and >=80% joint power across four cells. Choose the smallest **32, 48, or 64 families/domain** meeting both targets under declared assumptions: 3,072, 4,608, or 6,144 recipients. If none qualifies, declare infeasibility or revise before fresh confirmation. No test peeking or favorable-result stopping.

## 7. Forward counts and secondary mechanism stage

Let `N=96F` recipients across models/domains, `Ns=N/4` sensitivity recipients, and `U` unique unedited original/counterpart prompts after exact deduplication. The core requires

`Epatch = 15N + 3N + (3 methods × 2 sites × 3 extra doses)Ns = 22.5N`.

At `F=32`, this is **69,120 patched condition evaluations**, plus `U` unedited evaluations (at most `2N` if each case has one unique counterpart). This is not a full method-by-dose-by-site factorial. Additional fitting pairs, localization, and exploratory interventions require separate ledger entries.

A condition evaluation is not necessarily one model forward. For candidate set `A_c`, naive teacher forcing costs `sum_c |A_c|` full-sequence scoring forwards over all conditions. Free generation additionally costs the sum of actual decoding steps per condition; cached-prefix reuse changes the executed count and must be logged. Report these measured counts, sequence lengths, batching, and memory from a later authorized pilot; do not promise GPU hours from another paper's hardware.

Only after identifying a usable operation carrier may a separate secondary stage interchange operation variables alone and jointly with states. Fit its alignment on construction data, prevent final-answer leakage, and test novel recombinations against competing causal models. State-action operators are not trained operation editors. Component freeze/resample and restoration tests can strengthen pathway evidence, but require their own frozen candidate sets, counts, and inference family.

## 8. Decisions and methodological basis

If only late patches succeed, conclude that an interface is controllable. If centroid/local alternatives match the group method, there is no demonstrated added value from recovered algebra. Structural success with earlier-site failure leaves native use unresolved. Precise exclusion of five-point gains is negative evidence for the practical target; wide intervals are inconclusive. Poor clean competence fails feasibility. Success across the primary contrasts supports a bounded carrier/abstraction claim and warrants an independent replication, not uniqueness or guaranteed publication.

[Makelov et al., §§3–5 and 8](https://arxiv.org/html/2311.17030v2) motivates checking whether subspace patches exploit alternate pathways, while also describing successful patching cases. It does not make all patching evidence invalid. [Zhang and Nanda, §§3–6](https://arxiv.org/html/2309.16042v2) demonstrates sensitivity to corruption and metrics, motivating natural counterfactuals and complete-answer/raw-margin reporting. Together with the newer primary-source audit, these are design constraints; applying them alone is not the new contribution.
