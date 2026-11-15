"""Hidden-state extraction for Hugging Face causal language models."""

from __future__ import annotations

import os


def get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    return token if token else None


def unique_target_char_span(text: str, target: str) -> tuple[int, int]:
    starts = []
    start = text.find(target)
    while start != -1:
        starts.append(start)
        start = text.find(target, start + 1)
    if not starts:
        raise ValueError(f"Could not locate target={target!r} in text={text!r}")
    if len(starts) > 1:
        raise ValueError(f"Ambiguous target={target!r}: found {len(starts)} occurrences in text={text!r}")
    return starts[0], starts[0] + len(target)


def target_token_indices(text: str, target: str, tokenizer) -> list[int]:
    start_char, end_char = unique_target_char_span(text, target)
    enc = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    idx = [j for j, (s, e) in enumerate(enc["offset_mapping"]) if e > start_char and s < end_char]
    if not idx:
        raise ValueError(f"Could not locate target token span for target={target!r} in text={text!r}")
    return idx


def load_tokenizer_and_model(
    model_name: str,
    dtype: str = "float16",
    device_map: str = "auto",
    revision: str | None = None,
    tokenizer_settings: dict | None = None,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = get_hf_token()
    settings = tokenizer_settings or {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=token, use_fast=bool(settings.get("use_fast", True)), revision=revision
    )
    tokenizer.padding_side = str(settings.get("padding_side", "right"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token, dtype=model_dtype, device_map=device_map, revision=revision)
    model.eval()
    return tokenizer, model


def extract_hidden_representations(frame, tokenizer, model, batch_size: int = 8):
    import torch

    device = next(model.parameters()).device
    reps = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start:start + batch_size]
            texts = batch.text.tolist()
            targets = batch.target.tolist()
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            model_inputs = {k: v.to(device) for k, v in enc.items()}
            out = model(**model_inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            for b, (text, target) in enumerate(zip(texts, targets)):
                a, z = unique_target_char_span(text, target)
                pos = [j for j, (s, e) in enumerate(offsets[b].tolist()) if e > a and s < z]
                if not pos:
                    raise ValueError(f"Could not locate target token span for target={target!r} in text={text!r}")
                reps.append(torch.stack([h[b, pos, :].mean(0).float().cpu() for h in out.hidden_states]))
            del out, model_inputs, enc, offsets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return torch.stack(reps)


def extract_embedding_representations(frame, tokenizer, model, batch_size: int = 8):
    """Extract only HF hidden_states[0] with the canonical target-span mean."""
    import torch

    device = next(model.parameters()).device
    reps = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start:start + batch_size]
            texts = batch.text.tolist()
            targets = batch.target.tolist()
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            model_inputs = {key: value.to(device) for key, value in enc.items()}
            out = model(**model_inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            if not out.hidden_states:
                raise ValueError("model did not return Hugging Face hidden_states[0]")
            embedding_hidden = out.hidden_states[0]
            for batch_index, (text, target) in enumerate(zip(texts, targets)):
                start_char, end_char = unique_target_char_span(text, target)
                positions = [
                    index for index, (left, right) in enumerate(offsets[batch_index].tolist())
                    if right > start_char and left < end_char
                ]
                if not positions:
                    raise ValueError(f"Could not locate target token span for target={target!r} in text={text!r}")
                reps.append(embedding_hidden[batch_index, positions, :].mean(0).float().cpu())
            del out, embedding_hidden, model_inputs, enc, offsets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return torch.stack(reps)
