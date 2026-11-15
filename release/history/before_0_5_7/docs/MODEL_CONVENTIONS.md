> Frozen source record. This document preserves the protocol or implementation history of the archived experiments. Authorization and project-status statements refer to that historical stage. For the current release scope and commands, see [SCIENTIFIC_SCOPE.md](SCIENTIFIC_SCOPE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# Model and Layer Conventions

- Canonical models are Llama 3.2 1B/3B, Llama 3.1 8B, and Qwen2.5 1.5B/3B/7B, using the exact IDs in `configs/main_experiment.yaml`.
- Hidden-state index 0 is the embedding output and is not a primary transformer-computed layer.
- Transformer block outputs are indexed `1..N`; normalized depth is `layer/N`.
- Configured block counts are validated against `model.config.num_hidden_layers` after loading. Resolved Hugging Face commit hashes are written to the run manifest.
- Structural analysis may use block `N`. Source-token causal intervention requires `layer<N` so a later attention block can propagate the edit to the scored query position.
- Target-span extraction uses fast-tokenizer offset mappings and mean-pools every token overlapping the unique target string.
- Operators use row vectors: `z_transformed = z @ R_g`.
- D4 `compose(g1,g2)` means apply `g2` first, then `g1`; row-vector products use `R_g2 @ R_g1` for `R_compose(g1,g2)`.
- Splits are by `template_id`, never rows. PCA is fit only on training-template activations.
- The intervention convention is `h - zC + (zR)C`, preserving the residual outside the PCA subspace.
- fp16 and automatic device placement are defaults on RunPod. CPU checks do not load Hugging Face models.
