# Writing examples: bounded structural lessons for v0.5.8

This note records editorial observations, not a scientific review of the example papers or an independent check of their publication status. No sentences, diagrams, experimental results or methodological claims are copied into the manuscript. The requested directory spelling, `TMLR_accepted_eamples`, is retained.

## Materials actually inspected

| Supplied example | Inspection | Applicable lesson |
|---|---|---|
| *Semantic Anchor Transport: Robust Test-Time Adaptation for Vision-Language Models* (`7298_Semantic_Anchor_Transport.pdf`, 23 pages) | Text of pages 1–6; rendered page 2 | The introduction presents a concrete failure mode before technical machinery. A nearby figure makes the motivating question visible. Related Work has its own main-text section before the method. For BSG, introduce a concrete state transformation, then explain the three distinct questions before the spectral construction. |
| *LRanker: Scaling LLM Ranking to Millions of Candidates with Test-Time Scaling* (`10078_LRanker_Scaling_LLM_Rank.pdf`, 27 pages) | Text of pages 1–6; rendered page 4 | The introduction states a specific question, explains separate component roles and gives observed outcomes with scope. Problem Formulation precedes the framework; the diagram distinguishes training from testing. For BSG, separate fitted-predictor transfer from same-bank class risk, and let the conceptual diagram show parallel evaluation targets. |
| MoEFL | No matching filename or first-page title found among the 18 supplied PDFs | No MoEFL-specific writing lesson is claimed. Its absence does not block the requested reconstruction: the other two examples and the explicit revision plan supply sufficient editorial guidance. |

The second LRanker filename ending in `(1)` is a duplicate example, not a third independent writing sample. Other directory entries were inspected only at the first-page/title level to locate the requested material; they are not additional substantive stylistic authorities.

## Decisions for this manuscript

1. **Problem before machinery.** The weekday example introduces source/target representations and sharing across contexts. The new conceptual figure then separates prediction, exact-action risk and registered alternatives; it must not imply a progression from successful prediction to semantic uniqueness or native computation.
2. **Literature within the argument.** A compact Related Work section states what the closest work studies and how the present fixed-coordinate, fixed-real-dimension risk target differs. Longer comparisons and proofs remain in the appendix. Citation counts are not success criteria.
3. **Definitions before interpretation.** State the finite observation object, the declared action class and metric, and the invariant-subspace budget directly in the main text. Do not import the full Hilbert-space proof merely to avoid an appendix reference.
4. **Results organized by question.** Separate prediction controls, finite-bank costs, constructed-action transfer, registration/dimension sensitivity and new-context transport. Each subsection supplies a comparison, a result and its interpretation before referring to complete records.
5. **One home for repeated evidence.** Reuse the existing same-metric core table; relocate duplicate summary tables intact. Keep the training-metric transfer table separate. A short 8/16 display may select existing saved fields without recomputing data.
6. **Limits where they matter.** Keep metric differences, fixed-panel scope, lower-bound concentration and numerical-tolerance qualifications beside the relevant claims. Consolidate repeated historical execution details and broad limitations elsewhere.

We do not borrow the examples' superiority claims, task-specific rankings, visual assets, dense captions or page lengths. These papers do not establish a required TMLR outline or an acceptance guarantee. For BSG, clear communication of a bounded result is the objective; stronger empirical claims are not introduced to imitate an application-method paper.

The baseline is the verified 71-page v0.5.7 manuscript, not the earlier 67-page state described by the supplied plan. Its completed independent-orthogonal comparison and C7 heterogeneity are inherited, while LOFO remains unexecuted.
