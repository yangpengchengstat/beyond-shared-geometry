> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# Metric Definitions

All metrics use the canonical row-vector convention. Lower is better unless noted. For D4, `compose(g1,g2)` means apply `g2` first and then `g1`, so row-vector products use `R_g2 @ R_g1`.

## E_eq
Formula: mean over held-out template rows and non-identity group elements of `||z_x R_g - z_{g.x}||^2 / (||z_{g.x}||^2 + eps)`.
Domain: activation-supported latent states. Split: evaluation on held-out templates. Relationship to training: data fitting uses the same prediction form on training rows; held-out E_eq is evaluation. Primary metric.

## E_hom_data
Formula: mean over held-out activations and all `g1,g2` of `||z R_g2 R_g1 - z R_compose(g1,g2)||^2 / (||z R_compose(g1,g2)||^2 + eps)`.
Domain: activation-supported latent region. Split: held-out templates. Relationship to training: homomorphism is a joint regularizer, but E_hom_data is evaluated separately on held-out activations. Primary metric.

## E_hom_op
Formula: mean relative Frobenius `||R_g2 R_g1 - R_compose(g1,g2)||_F / ||R_compose(g1,g2)||_F`.
Domain: whole latent operator space. Split: no data split. Relationship to training: auxiliary diagnostic for the ambient operator, not the primary held-out metric.

## E_rel_data
C7 formula averages activation-conditioned `r^7=e`, generator-power consistency, and inverse consistency. D4 formula averages activation-conditioned `r^4=e`, `s^2=e`, and `srs=r^{-1}`.
Domain: activation-supported latent states. Split: held-out templates. Relationship to training: defining relations are joint regularizers; E_rel_data is the primary data-conditioned evaluation.

## E_rel_op
Formula: relative Frobenius defining-relation defect in ambient operator space. Auxiliary diagnostic.

## E_comp_data
Conservative held-out composition consistency: activation-conditioned comparison of sequential operator products with the direct fitted operator. It must not be described as fully independent group-word generalization when equivalent constraints appear in the joint objective. Primary consistency diagnostic, not the strongest unseen-word claim.

## E_comp_op
Operator-level Frobenius analogue of E_comp_data. Auxiliary diagnostic.

## E_word
Generator-only group-word generalization. C7 derives `r^k` from the generator. D4 derives ordered words from `r` and `s`, explicitly preserving `rs != sr`. This is the stronger algebraic-generalization metric.

## E_inverse_data / E_inverse_op

Inverse consistency compares `z R_g R_g^-1` with `z` on held-out templates and compares `R_g R_g^-1` with identity in operator space. Data-conditioned error is primary; operator-space error is diagnostic.

## Identity Baseline
Evaluation-only normalized error from using the identity map for non-identity transformations.

## Cross-model metrics

Linear CKA measures geometry on full held-out hidden representations. Orthogonal Procrustes measures geometry after matched-dimensional train-only PCA. Intertwining error measures whether a learned latent map satisfies the row-vector relation `R_A(g)T ~= T R_B(g)`. `geometry_algebra_gap = intertwiner_error - procrustes_error`; it is a contrast, not a universal calibrated effect size.

## Activation-intervention metrics

The canonical behavioral outputs are target-token logit change, target-token probability change, top-1 success, and target-vs-original logit margin. C7 `k=7` reports cycle recovery separately from `k=1,...,6` transformation success. D4 relation rows report identity recovery for `r^4`/`s^2` and operator consistency for `srs` versus directly learned `r^-1`. Learned-minus-baseline 95% intervals are paired and hierarchical.
