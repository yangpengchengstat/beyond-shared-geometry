# Context-generalization results audit

The completed study supports a conditional, heterogeneous transfer result. Direct difference fitting lowers average recorded C7 transport loss relative to coordinate fitting, but raises average D4 loss. Their equal-domain average is close to zero because the two domain effects oppose one another. It does **not** establish equivalence, general superiority, a population confidence certificate, or independent confirmation of a theorem.

This audit reads the existing archive and summarizes recorded losses. It does not run a model, refit predictors, rescore coordinates, regenerate prompts, rerun a bootstrap, or alter the original archive or project source. All new files are confined to this audit directory.

## Recorded result and interpretation

The primary contrast is **difference-fitted minus coordinate-fitted normalized context-transport loss**. Negative values favor difference fitting. Both primary arms selected their penalty using the same old-validation context-difference loss.

| Fixed panel | Difference − coordinate | Stored 2.5th–97.5th family-bootstrap percentiles | Difference lower than coordinate, cells |
|---|---:|---:|---:|
| C7, 24 cells | −0.005832764 | [−0.006680294, −0.005007947] | 20/24 |
| D4, 24 cells | +0.006189720 | [+0.003969767, +0.008406254] | 8/24 |
| Primary equal-domain panel, 48 cells | +0.000178478 | [−0.001034603, +0.001361832] | 28/48 |

The intervals above are copied from the archived `summary.json`, not recomputed. They are the protocol's conditional sensitivity summaries across the fixed authored bank. Call them **family-bootstrap sensitivity percentiles**, not population confidence intervals. The primary interval straddling zero is not an equivalence result. No equivalence margin or significance decision was specified or recorded.

The secondary identity control matters:

| Fixed panel | Identity | Coordinate fitted | Difference fitted | Coordinate, native-validation selected |
|---|---:|---:|---:|---:|
| C7 | 0.043570838 | 0.050311702 | 0.044478939 | 0.050356523 |
| D4 | 0.166152706 | 0.122730596 | 0.128920316 | 0.122723560 |
| Equal domains | 0.104861772 | 0.086521149 | 0.086699627 | 0.086540042 |

On C7, identity has the lowest average loss; the difference arm's improvement over the coordinate arm therefore does not show that learned difference maps improve on leaving context differences unchanged. On D4, both fitted arms improve on identity on average, with coordinate fitting better. Changing to the secondary coordinate-native-validation comparator barely changes the aggregate contrast (+0.000159586), and the C7/D4 signs remain the same. These secondary comparisons are descriptive; this audit computes no additional uncertainty intervals.

All 48 planned cells completed. Arithmetic averages of the 64 per-family records agree with each cell's stored means and the CSV; averaging paired family contrasts agrees with all three archived summary point estimates to absolute tolerance 1e−14.

## Statistical units, weighting, and estimand

There are six fixed, revision-pinned models, four original layers per model, and two domains. All use the old rank-16 PCA center and basis and the `span_mean` readout. The bank contains 64 explicitly authored **paired families shared across the two domains**. Each family has two renderings and all seven weekdays or four corner states: 896 C7 and 512 D4 prompts, totaling 1,408 prompts per model. The same bank is encoded by all models.

For a family and state, the analysis forms the representation difference between its two renderings. For each nonidentity group action, it predicts the destination state's context difference from the source state's context difference. The squared prediction error is divided by the **old-training uncentered mean squared context-difference energy**, fixed separately for each model/layer/domain cell. This is a normalized representation-transport loss, not task accuracy, language-generation quality, or a fairness metric.

Evaluation first equally averages changed states within an action, then nonidentity actions within a family, then 24 fixed cells within each domain, and finally the two domains equally. C7 has six nonidentity actions, each changing all seven states. D4 has seven nonidentity actions: five change four states and two change two states. Equal action weighting therefore differs from pooling all changed state/action pairs. Fixed states remain in fitting but are excluded from validation selection and evaluation.

The original 2,000-draw bootstrap (seed 2026102901) resamples the 64 family indices, sharing each draw across models, layers, and domains. The resampling unit is **not** a prompt, state, action, cell, or family-domain pair. The 3,072 family-by-cell losses are not 3,072 independent observations. Models and layers are never resampled; the interval does not quantify variation across model populations, training seeds, or possible languages. The frames are a fixed assistant-authored bank with related designs, not a random sample of natural language; even family resampling should retain the protocol's limited sensitivity interpretation.

## Heterogeneity

These model means average the four fixed layers and are descriptive subdivisions of the completed panel:

| Model | C7 difference − coordinate | D4 difference − coordinate | Equal-domain mean |
|---|---:|---:|---:|
| Llama-3.2-1B | −0.003673166 | +0.015708509 | +0.006017671 |
| Llama-3.2-3B | −0.017054986 | −0.005275277 | −0.011165132 |
| Llama-3.1-8B | −0.003262915 | −0.004437933 | −0.003850424 |
| Qwen2.5-1.5B | −0.001935548 | +0.013639464 | +0.005851958 |
| Qwen2.5-3B | −0.002365711 | +0.009363670 | +0.003498979 |
| Qwen2.5-7B | −0.006704257 | +0.008139890 | +0.000717817 |

All six model-level C7 averages favor difference fitting, although four individual C7 cells have the opposite sign. D4 is more heterogeneous: Llama-3.2-3B and Llama-3.1-8B favor difference fitting on average, while the other four model averages favor coordinate fitting. This does not support a universal model-size trend or architecture-level claim.

After averaging cells, 63 of 64 C7 family contrasts are negative, compared with 13 of 64 D4 family contrasts and 26 of 64 equal-domain family contrasts. These are finite-bank counts, not independent success trials. Full recorded cell means and derived family contrasts are in `context_cell_metrics.csv` and `context_family_contrasts.csv`; none were removed or selected for presentation.

## Leakage and provenance checks

The archive contains the entire planned context panel. The extraction script independently verifies exported member hashes, the study source manifest, all prepared input hashes, frozen predictor receipts, projected-coordinate receipts, result receipts, and per-model old/new runtime identities. It verifies 546 context/study exported members, 39 study source/dependency files, 148 prepared input files, 48 cell receipt chains, and 936 recorded action/arm penalty-selection entries. All checks pass. The complete evidence ledger records 593 archive members actually read, their byte lengths, and SHA-256 hashes.

The fitting code has no test argument. Prepared development arrays contain only `train` and `validation`, with shapes 48 × 2 × states × 16 and 16 × 2 × states × 16. This audit inspects array headers, not their numerical values. Coordinate fits use both renderings' state coordinates; difference fits use the paired rendering differences. Both fit all training states per action, with no intercept, using the same grid `[0, 1e−5, 0.001, 0.1, 1]`; ridge scaling is lambda times training squared Frobenius norm divided by rank. Both primary arms select on old-validation changed-state context-difference loss. The secondary coordinate-native-validation arm selects on coordinate loss. Every stored selected index is the first minimum of its recorded candidate losses. No lambda of 1 was selected, but all grid candidates remain recorded.

Frozen fit receipts precede new extraction. The fit manifest records `new_test_outputs_seen: false`; extraction binds the full frozen fit manifest and the prepared input contract. All new coordinate receipts bind the same extraction contract, original PCA hashes, bank rows, and extraction manifests. The old and new runtime identities match for all cells. No code path fits or centers PCA on the new bank, or selects layers/ranks/families using these new test results. These checks support the separation claimed by the implemented pipeline; they cannot prove the absence of unrecorded external activity.

The input manifest records zero exact prompt overlap against the entire archived old coordinate banks and hash-binds the restored old dataset rows. The restoration code aligns full prompts by exact row ID, verifies canonical dataset and serialized extraction-row hashes, and does not regenerate old prompts. The original parent archive itself is not embedded in this results archive; this audit therefore verifies those recorded checks and their code, rather than independently comparing every old prompt string. Exact string disjointness does not establish semantic independence.

The new bank has 1,408 distinct full texts, complete paired state blocks, and one unambiguous target occurrence per row. There is no competing weekday/corner token in the checked text. The target token is deliberately present because this is representation extraction, not answer prediction; its presence is not evidence of answer leakage. The causal models' readout is at the target token span, so narrative material after those tokens does not provide additional context to those hidden states. Claims about transfer should concern this target-span representation protocol.

The review receipt binds the correct bank and source manifest; the fit manifest binds its exact hash. Its wording explicitly says the bank was assistant-authored and user-reviewed through an operator attestation, **not an authenticated digital signature**. It does not identify an independent author or reviewer and contains no frame-level annotations. The record supports “operator-attested semantic review before execution,” not independently verified human authorship or a blinded human study. The preparation-time `pending` review fields and `prepared_not_executed` protocol status are immutable historical metadata; the later review and completion receipts establish subsequent execution. They should not be edited to imply they were originally execution records.

The first extraction attempt stopped on gated-model authentication before completing the panel; its failure/logs were retained. The successful resumed extraction ended at 2026-10-30 20:31:46 UTC and analysis at 20:31:47 UTC. The four context stage receipts match every referenced final available file; there is no missing or changed file among those receipts. Completion is a provenance/workflow status, not a scientific pass.

## Material wording correction: new contexts also change fixed factual wording

The frozen protocol says “only outer narrative frames are new.” The archived implementations support a narrower description:

| Domain | Old source generator's fixed factual sentence | New bank's fixed factual sentence |
|---|---|---|
| C7 | `the designated weekday is {state}.` | `The scheduled weekday is {token}.` |
| D4 | `the marked corner is {state}.` | `The marked square corner is {token}.` |

The old structural generator also appends ` Summary:` after its rendered prompt; the new bank does not. This is not a clean intervention changing only an outer wrapper. The known weekday/corner vocabulary, state structure, and factual meaning are retained, but the new bank changes the surrounding narratives **and modest fixed factual wording/capitalization**. Consequently, one cannot attribute any observed difference uniquely to narrative wrapper changes, or claim unseen factual vocabulary or independent core grammar. The manuscript should disclose this without altering the frozen protocol or historic measurements.

Evidence: archived `source_snapshot/studies/characterization_v1/frozen_run/src/bsg_next/data.py`, lines 124–145; archived `source_snapshot/studies/context_generalization_v1/bank.py`, `build_bank`. The old sentences above are read from the source snapshot, not regenerated or substituted into the result bank.

## Suggested manuscript text

**Methods.** “We evaluated frozen predictors on 64 assistant-authored paired narrative families, with operator-attested semantic review before execution. Each family instantiated all seven weekday states and all four square-corner states, yielding 1,408 prompts per model. The new bank retained the known state vocabulary while changing narrative frames and modest fixed factual wording. We reused the original rank-16 PCA and six-model, four-layer panel. Predictors were fitted on 48 old training families; both primary arms selected regularization using context-difference loss on 16 old validation families before new prompt extraction. We averaged normalized transport losses over changed states, nonidentity actions, fixed cells within each domain, and the two domains equally. The archived 2,000-draw paired-family bootstrap reports conditional sensitivity within this authored bank, not population coverage.”

**Results.** “All 48 planned cells completed. The equal-domain difference-minus-coordinate contrast was +0.000178, with archived family-bootstrap sensitivity percentiles [−0.001035, +0.001362]. This aggregate masks opposing domain effects: difference fitting reduced C7 loss by 0.005833 relative to coordinate fitting, whereas it increased D4 loss by 0.006190. The corresponding sensitivity percentiles were [−0.006680, −0.005008] and [+0.003970, +0.008406]. Difference fitting had lower loss in 20/24 C7 cells and 8/24 D4 cells. Identity achieved the lowest mean C7 loss, while coordinate fitting achieved the lowest mean D4 loss. These results show heterogeneous transfer within the fixed bank and panel; they establish neither equivalence nor uniform superiority of either fitting objective.”

Do not replace this with “difference fitting generalizes better,” “the approaches are equivalent,” “significant across unseen contexts,” “independently human-authored contexts,” “no distribution shift except outer context,” or “the theorem is validated by 48 independent experiments.”

## Evidence identity and reproduction

Original results archive: `tmlr-all-v2.20261030T203725Z-5489fd.results.zip`.

SHA-256: `ef87e9aaeb8f093ba0c2438590d6a30b5dffcd256993e2366bd600cca3cb5bbb`.

Context archive prefix: `context-b73c5b86cc8f4257961c9d2b1d91b121/`.

Parent-input archive SHA-256 recorded by all relevant receipts: `6a4c38237bf1368d4c77c97504f666d0d933bfebed9417d173fbf7441c0fecef`.

The following members are relative to the context prefix:

| Member | SHA-256 |
|---|---|
| `summary.json` | `e67043e9d57f0ef91b08115fe3799363f650c40f0cc4a84144c2e2c4bc2798b0` |
| `cell_summary.csv` | `6323289f9342d341eca29d1b3cfe16b434b4c516878d91f50d478d0a539a00ce` |
| `bank.json` | `22faa22683cba96f6476ca392c3e71ead8b13574e8475c7cf3cc5d839df4c274` |
| `human_review.json` | `edc0ba104c746697a8caf565d9c1b92ee934472b04b502c4bd739d5e51d2f6e8` |
| `protocol.json` | `bf2e23cbbab760fd1e563e7b5ab17aa576eefe9f97444965f94fa3b54fdf773f` |
| `input_manifest.json` | `8c10393801f3cf05f53e7beb25a2222d45b4b65c5ff8818013fa6eddf01985e6` |
| `fits_manifest.json` | `c1ff091468095dfb1a82e3326ba5173b53c3e9c2861bd335801b21d56086c1b5` |
| `analysis_complete.json` | `8e3431dcf06bedf7aad2430fff3a6b1f51c0c08cb6d61678821bb788ce10ca51` |

The bank's **canonical rows hash**, `453a02ad7ee5fadb14821a8715e1b24648c58103b1ad742cc2e9890692763299`, differs appropriately from the `bank.json` file-byte hash. Similarly, the source protocol file and copied run protocol have equal parsed content but different serialization hashes; this is not a protocol change.

The source manifest at `source_snapshot/studies/context_generalization_v1/source_manifest.json` has SHA-256 `a2ea25071b12b7ae628e2250142e1cb700294554059f4b7235e84c5f501eee73`. `context_evidence.json` records exact full archive paths and hashes for every audited result, per-family evaluation, receipt, input and source dependency, plus the audit script hash. `context_metrics.json` contains the compact numerical summaries, model subdivisions and check counts.

To reproduce this audit, run `python3 scripts/extract_context_summary.py RESULTS_ZIP OUTPUT_DIRECTORY`. It uses only the Python standard library, reads the original ZIP without extraction, and writes four derived audit files. It copies the archived percentile endpoints and performs deterministic arithmetic summaries only; it never imports the experiment modules or recalculates their bootstrap or losses.

Public compact outputs are in `results/tmlr_followup_v2/`.
