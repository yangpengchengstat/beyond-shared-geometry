"""Canonical activation-intervention API reconciled from V6/V7 pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bsg.representations.extract import unique_target_char_span


@dataclass(frozen=True)
class InterventionSpec:
    group: str
    source_text: str
    source_target: str
    group_element: object
    intervention_layer: int
    natural_reference_text: str
    natural_reference_answer: str


class SteeringBaseline(Protocol):
    """Extension point for baselines, including future Manifold Steering."""

    name: str

    def intervene(self, hidden_vector, *, latent_vector, pca, operator, context): ...


def transform_latent_activation(latent_vector, operator):
    return latent_vector @ operator


def repeated_operator(operator, steps: int):
    """Return a row-vector operator applied repeatedly for the given steps."""
    import numpy as np

    op = np.asarray(operator, dtype=float)
    if op.ndim != 2 or op.shape[0] != op.shape[1]:
        raise ValueError("operator must be a square matrix")
    if int(steps) < 0:
        raise ValueError("steps must be non-negative")
    result = np.eye(op.shape[0], dtype=float)
    for _ in range(int(steps)):
        result = result @ op
    return result


def residual_preserving_intervention(hidden_vector, pca, operator):
    """Transform the PCA component while preserving its orthogonal residual.

    The repository uses row vectors. If z = E(h) and C contains the PCA
    decoder rows, this computes h - z C + (z R) C.
    """
    import numpy as np

    hidden = np.asarray(hidden_vector, dtype=float)
    if hidden.ndim != 1:
        raise ValueError("hidden_vector must be one-dimensional")
    if pca.components_ is None:
        raise RuntimeError("PCA must be fit before intervention")
    latent = pca.transform(hidden[None, :])[0]
    transformed = transform_latent_activation(latent, operator)
    return hidden - latent @ pca.components_ + transformed @ pca.components_


def mean_difference_steering(hidden_vector, pca, latent_delta):
    """Apply a latent mean-difference vector while preserving the residual."""
    import numpy as np

    hidden = np.asarray(hidden_vector, dtype=float)
    delta = np.asarray(latent_delta, dtype=float)
    if delta.shape != (pca.dim,):
        raise ValueError(f"latent_delta must have shape {(pca.dim,)}, got {delta.shape}")
    return hidden + delta @ pca.components_


def fit_source_conditioned_additive(
    frame,
    hidden_representations,
    train_mask,
    elements,
    transform_state,
):
    """Fit train-only additive vectors indexed by group element and source state.

    For row-vector activations this estimates
    v[g, s] = mean_t(h[g s, t] - h[s, t]) over training templates.
    Conditioning on the source state avoids the exact cancellation of a
    complete, balanced group orbit while remaining a purely additive control.
    """
    import numpy as np

    hidden = np.asarray(hidden_representations, dtype=float)
    mask = np.asarray(train_mask, dtype=bool)
    if hidden.ndim != 2 or len(hidden) != len(frame) or len(mask) != len(frame):
        raise ValueError("frame, hidden_representations, and train_mask must align")
    lookup = {
        (int(row.template_id), int(row.state_id)): position
        for position, row in enumerate(frame.itertuples())
    }
    train_templates = sorted(map(int, frame.loc[mask, "template_id"].unique()))
    source_states = sorted(map(int, frame.state_id.unique()))
    vectors = {}
    for element in elements:
        for source_state in source_states:
            target_state = int(transform_state(source_state, element))
            deltas = []
            for template_id in train_templates:
                source_index = lookup[(template_id, source_state)]
                target_index = lookup[(template_id, target_state)]
                if not mask[source_index] or not mask[target_index]:
                    raise ValueError("group-expanded training template is incomplete")
                deltas.append(hidden[target_index] - hidden[source_index])
            if not deltas:
                raise ValueError("source-conditioned additive fit has no training templates")
            vectors[(element, source_state)] = np.mean(deltas, axis=0)
    return vectors


def source_conditioned_additive_intervention(hidden_vector, vectors, element, source_state):
    """Apply the train-fitted additive vector for one element/source pair."""
    import numpy as np

    hidden = np.asarray(hidden_vector, dtype=float)
    key = (element, int(source_state))
    if key not in vectors:
        raise KeyError(f"missing source-conditioned additive vector for {key!r}")
    vector = np.asarray(vectors[key], dtype=float)
    if vector.shape != hidden.shape:
        raise ValueError(f"additive vector shape {vector.shape} does not match hidden {hidden.shape}")
    return hidden + vector


def intervention_recovery_errors(hidden_vector, intervened_vector, pca):
    """Return relative hidden and latent recovery errors."""
    import numpy as np

    hidden = np.asarray(hidden_vector, dtype=float)
    intervened = np.asarray(intervened_vector, dtype=float)
    z = pca.transform(hidden[None, :])[0]
    z_intervened = pca.transform(intervened[None, :])[0]
    eps = np.finfo(float).eps
    return {
        "hidden_relative_error": float(np.linalg.norm(intervened - hidden) / max(np.linalg.norm(hidden), eps)),
        "latent_relative_error": float(np.linalg.norm(z_intervened - z) / max(np.linalg.norm(z), eps)),
    }


def continuation_token_span(prompt: str, answer: str, tokenizer):
    full_text = prompt + answer
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]
    if full_ids[:len(prompt_ids)] == prompt_ids:
        return full_text, full_ids, len(prompt_ids)
    enc = tokenizer(full_text, add_special_tokens=True, return_offsets_mapping=True)
    boundary = len(prompt)
    positions = [i for i, (_s, e) in enumerate(enc["offset_mapping"]) if e > boundary]
    if not positions:
        raise ValueError("Could not locate answer continuation tokens")
    return full_text, enc["input_ids"], min(positions)


def continuation_logprob_from_logits(logits, input_ids, answer_start_index: int):
    import numpy as np
    ids = np.asarray(input_ids, dtype=int)
    try:
        import torch
        if isinstance(logits, torch.Tensor):
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)
            vals = []
            for j in range(answer_start_index, len(input_ids)):
                if j == 0:
                    continue
                vals.append(log_probs[j - 1, int(ids[j])].item())
            return float(np.sum(vals)), len(vals)
    except ModuleNotFoundError:
        pass
    arr = np.asarray(logits, dtype=float)[0]
    shifted = arr - arr.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    vals = []
    for j in range(answer_start_index, len(input_ids)):
        if j == 0:
            continue
        vals.append(float(log_probs[j - 1, ids[j]]))
    return float(np.sum(vals)), len(vals)


def mean_continuation_logprob(logits, input_ids, answer_start_index: int):
    total, n = continuation_logprob_from_logits(logits, input_ids, answer_start_index)
    return total / max(n, 1), n


def single_token_continuation_id(prompt: str, answer: str, tokenizer) -> int:
    """Resolve a context-sensitive continuation and require one target token."""
    _, input_ids, answer_start = continuation_token_span(prompt, answer, tokenizer)
    answer_ids = input_ids[answer_start:]
    if len(answer_ids) != 1:
        raise ValueError(
            f"Causal pilot target {answer!r} must tokenize as one continuation token; "
            f"got {len(answer_ids)} tokens"
        )
    return int(answer_ids[0])


def next_token_metrics(logits, target_token_id: int, original_token_id: int, position: int = -1):
    """Score a target token at a selected causal prediction position."""
    import numpy as np

    try:
        import torch
        if isinstance(logits, torch.Tensor):
            scores = logits[0, position].detach().float().cpu().numpy()
        else:
            scores = np.asarray(logits, dtype=float)[0, position]
    except ModuleNotFoundError:
        scores = np.asarray(logits, dtype=float)[0, position]
    shifted = scores - np.max(scores)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    target = int(target_token_id)
    original = int(original_token_id)
    top1 = int(np.argmax(scores))
    return {
        "target_logit": float(scores[target]),
        "target_probability": float(probabilities[target]),
        "logit_margin_vs_original": float(scores[target] - scores[original]),
        "top1_token_id": top1,
        "top1_success": bool(top1 == target),
    }


def sequence_outputs(model, tokenizer, text: str, device=None):
    """Return logits and the final residual-stream state for an unpatched prompt."""
    import torch

    if device is None:
        device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt")
    model_inputs = {key: value.to(device) for key, value in enc.items()}
    with torch.inference_mode():
        out = model(**model_inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    return {
        "logits": out.logits.detach().float().cpu(),
        "final_hidden": out.hidden_states[-1][0, -1].detach().float().cpu().numpy(),
    }


def sequence_logits(model, tokenizer, text: str, device=None):
    return sequence_outputs(model, tokenizer, text, device=device)["logits"]


def patched_sequence_outputs(model, tokenizer, full_text: str, state_target: str, hidden_state_index: int, replacement_vector, device=None):
    import torch
    if hidden_state_index == 0:
        raise ValueError("Behavioral patching uses transformer hidden-state index >= 1")
    if device is None:
        device = next(model.parameters()).device
    block_index = hidden_state_index - 1
    enc = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    start_char, end_char = unique_target_char_span(full_text, state_target)
    positions = [j for j, (s, e) in enumerate(offsets) if e > start_char and s < end_char]
    if not positions:
        raise ValueError(f"Could not locate state target {state_target!r}")
    model_inputs = {k: v.to(device) for k, v in enc.items()}
    rep = torch.tensor(replacement_vector, dtype=torch.float32, device=device).to(next(model.parameters()).dtype)
    def hook_fn(_module, _module_input, module_output):
        if isinstance(module_output, tuple):
            hidden = module_output[0].clone()
            for pos in positions: hidden[0, pos, :] = rep
            return (hidden,) + module_output[1:]
        hidden = module_output.clone()
        for pos in positions: hidden[0, pos, :] = rep
        return hidden
    handle = model.model.layers[block_index].register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            out = model(**model_inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    finally:
        handle.remove()
    return {
        "logits": out.logits.detach().float().cpu(),
        "final_hidden": out.hidden_states[-1][0, -1].detach().float().cpu().numpy(),
    }


def patched_sequence_logits(model, tokenizer, full_text: str, state_target: str, hidden_state_index: int, replacement_vector, device=None):
    return patched_sequence_outputs(
        model, tokenizer, full_text, state_target, hidden_state_index, replacement_vector, device=device
    )["logits"]
