"""Lazy Hugging Face execution, pooled features, and prefix-only interventions.

Importing this module never imports transformers, downloads weights, or executes
a model. Callers explicitly request inference. Injected tiny models/tokenizers
support offline tests. Complete candidate answers are teacher-forced and scored.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import time
import sys
from importlib import metadata as package_metadata

import numpy as np

MODEL_SPECS = {
    "meta-llama/Llama-3.2-1B": {"revision": "4e20de362430cd3b72f300e6b0f18e50e7166e08", "hidden_size": 2048, "n_layers": 16, "layers": [3, 6, 10, 13]},
    "meta-llama/Llama-3.2-3B": {"revision": "13afe5124825b4f3751f836b40dafda64c1ed062", "hidden_size": 3072, "n_layers": 28, "layers": [6, 11, 17, 22]},
    "meta-llama/Llama-3.1-8B": {"revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b", "hidden_size": 4096, "n_layers": 32, "layers": [6, 13, 19, 26]},
    "Qwen/Qwen2.5-1.5B": {"revision": "8faed761d45a263340a0528343f099c05c9a4323", "hidden_size": 1536, "n_layers": 28, "layers": [6, 11, 17, 22]},
    "Qwen/Qwen2.5-3B": {"revision": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b", "hidden_size": 2048, "n_layers": 36, "layers": [7, 14, 22, 29]},
    "Qwen/Qwen2.5-7B": {"revision": "d149729398750b98c0af14eb82c78cfe92750796", "hidden_size": 3584, "n_layers": 28, "layers": [6, 11, 17, 22]},
}
PINNED_MODELS = MODEL_SPECS
HF_RUNTIME_VERSION = "pooled_prefix_delta_v1"


def dependency_versions(base=True):
    """Read distribution metadata only; never imports torch or initializes GPU."""
    values = {}
    for name in ("numpy", "torch", "transformers", "tokenizers", "huggingface-hub"):
        try:
            value = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            value = None
        values[name] = value.split("+", 1)[0] if base and value is not None else value
    return values


def _memory_metrics():
    import resource
    import torch
    metrics = {"peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
               "memory_peak_scope": "process_since_external_reset", "gpu_peaks_by_device": {}}
    if torch.cuda.is_available():
        metrics["gpu_peaks_by_device"] = {str(i): {"allocated_bytes": torch.cuda.max_memory_allocated(i),
                                                  "reserved_bytes": torch.cuda.max_memory_reserved(i)}
                                           for i in range(torch.cuda.device_count())}
    return metrics


@dataclass
class PatchSpec:
    layer: int
    positions: list[int] | None = None
    span: tuple[int, int] | list[int] | None = None
    replacement_mean: object | None = None
    delta: object | None = None
    dose: float = 1.0
    mode: str = "delta"
    replacement: object | None = None

    def __post_init__(self):
        self.layer = int(self.layer)
        if self.replacement_mean is None and self.replacement is not None:
            self.replacement_mean = self.replacement
        if self.mode not in ("delta", "mean_overwrite"):
            raise ValueError("Patch mode must be delta or mean_overwrite")
        if self.replacement_mean is not None and self.delta is not None:
            raise ValueError("Supply replacement_mean or delta, not both")
        if not np.isfinite(float(self.dose)):
            raise ValueError("Patch dose must be finite")


def _span_positions(offsets, span):
    a, b = map(int, span)
    positions = [i for i, (start, end) in enumerate(offsets) if end > a and start < b]
    if not positions or b <= a:
        raise ValueError("Readout span has no tokens")
    return positions


def _record_span(record, readout):
    if readout == "carrier_span":
        if "carrier_span" not in record:
            raise ValueError("carrier_span readout requires explicit record carrier_span")
        return record["carrier_span"]
    if "source_span" in record:
        return record["source_span"]
    target, text = record.get("target"), record["text"]
    if not target or text.count(target) != 1:
        raise ValueError("span_mean requires a unique target or explicit source_span")
    start = text.index(target)
    return start, start + len(target)


class ModelRuntime:
    def __init__(self, model_id, revision=None, dtype="float16", device_map="auto", batch_size=8,
                 local_files_only=False, model=None, tokenizer=None, device=None, chat_template=False,
                 allow_cpu_offload=False, deterministic=True, attention_implementation="eager", **kwargs):
        if kwargs:
            raise TypeError(f"Unknown runtime arguments: {sorted(kwargs)}")
        self.model_id = str(model_id)
        spec = MODEL_SPECS.get(self.model_id, {})
        self.revision = revision or spec.get("revision")
        if self.revision is None and model is None:
            raise ValueError("Custom models require an exact revision")
        self.revision = self.revision or "injected-local-model"
        self.dtype, self.device_map = dtype, device_map if device is None else device
        self.batch_size, self.local_files_only = int(batch_size), bool(local_files_only)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model, self.tokenizer = model, tokenizer
        self._configured = False
        self.chat_template = bool(chat_template)
        self.allow_cpu_offload = bool(allow_cpu_offload)
        self.deterministic = bool(deterministic)
        self.attention_implementation = attention_implementation
        self._dependency_versions = dependency_versions()
        self.telemetry = [{"stage":"runtime_environment", "dependency_build_versions":dependency_versions(base=False)}]
        self._entity_embedding_cache = {}

    @property
    def n_layers(self):
        if self.model is not None:
            return int(self.model.config.num_hidden_layers)
        if self.model_id not in MODEL_SPECS:
            self.load()
            return int(self.model.config.num_hidden_layers)
        return int(MODEL_SPECS[self.model_id]["n_layers"])

    @property
    def hidden_size(self):
        if self.model is not None:
            return int(self.model.config.hidden_size)
        if self.model_id not in MODEL_SPECS:
            self.load()
            return int(self.model.config.hidden_size)
        return int(MODEL_SPECS[self.model_id]["hidden_size"])

    def identity(self):
        return {"model_id": self.model_id, "model_revision": self.revision, "tokenizer_revision": self.revision,
                "dtype": str(self.dtype), "device_map": str(self.device_map),
                "padding_side": "right", "truncation": False, "runtime_version": HF_RUNTIME_VERSION,
                "batch_size": self.batch_size, "add_special_tokens": not self.chat_template,
                "dependency_versions": self._dependency_versions,
                "chat_template": self.chat_template, "deterministic": self.deterministic,
                "allow_cpu_offload": self.allow_cpu_offload, "attention_implementation": self.attention_implementation,
                "layer_convention": "0=input_embedding; l>=1=raw_block_l_minus_1_output_before_final_norm"}

    def close(self):
        import gc
        self.model, self.tokenizer = None, None
        self._configured = False
        self._entity_embedding_cache.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    clear_gpu = close

    def _blocks(self):
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            layers = getattr(getattr(self.model, "transformer", None), "h", None)
        if layers is None:
            raise ValueError("Model architecture has no registered block-hook adapter")
        return layers

    def load(self):
        if self._configured:
            return self
        if self.model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, use_fast=True,
                                                          local_files_only=self.local_files_only)
            model_dtype = getattr(torch, self.dtype) if isinstance(self.dtype, str) else self.dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, torch_dtype=model_dtype,
                device_map=self.device_map, local_files_only=self.local_files_only,
                attn_implementation=self.attention_implementation)
            if not self.allow_cpu_offload and str(self.device_map) == "auto" and torch.cuda.is_available():
                mapping = getattr(self.model, "hf_device_map", {})
                if any(str(device) in ("cpu", "disk") for device in mapping.values()):
                    self.close()
                    raise RuntimeError("Automatic CPU/disk offload is disabled; use sufficient GPU memory or explicitly register allow_cpu_offload")
        if self.tokenizer is None:
            raise ValueError("An injected model needs an injected tokenizer")
        if hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation(self.attention_implementation)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        if self.deterministic:
            import torch
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cuda"):
                torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        self._configured = True
        return self

    def _wrap_prompt(self, text):
        self.load()
        if not self.chat_template:
            return text, 0
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("chat_template=True requires a pinned tokenizer chat template")
        wrapped = self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                    tokenize=False, add_generation_prompt=True)
        if wrapped.count(text) != 1:
            raise ValueError("Chat template must preserve the raw prompt verbatim exactly once for span alignment")
        return wrapped, wrapped.index(text)

    def _encode_prompt(self, text, offsets=True):
        wrapped, shift = self._wrap_prompt(text)
        return self._encode([wrapped], offsets=offsets), wrapped, shift

    @property
    def device(self):
        self.load()
        return self.model.get_input_embeddings().weight.device

    def _encode(self, texts, offsets=True, padding=True):
        self.load()
        encoded = self.tokenizer(texts, return_tensors="pt", padding=padding, truncation=False,
                                 return_offsets_mapping=offsets, add_special_tokens=not self.chat_template)
        encoded.pop("token_type_ids", None)
        if "attention_mask" not in encoded:
            raise ValueError("Tokenizer must supply an attention mask")
        lengths = encoded["attention_mask"].sum(-1)
        maximum = int(getattr(self.model.config, "max_position_embeddings", 2**31))
        if int(lengths.max()) > maximum:
            raise ValueError(f"Prompt length exceeds registered model context window {maximum}; no truncation allowed")
        return encoded

    def features(self, records, layers, readout="span_mean", batch_size=None, return_positions=False):
        import torch
        self.load()
        readout = "span_mean" if readout == "source_span" else readout
        layers = list(map(int, layers))
        if len(set(layers)) != len(layers) or not layers or min(layers) < 0 or max(layers) > self.n_layers:
            raise ValueError("Invalid or duplicate hidden-state indices")
        if readout not in ("span_mean", "final_token", "carrier_span"):
            raise ValueError(f"Unknown pooled readout {readout}")
        if layers == [0]:
            yield from self.embedding_only(records, readout=readout, batch_size=batch_size, return_positions=return_positions)
            return
        size = int(batch_size or self.batch_size)
        for begin in range(0, len(records), size):
            batch = records[begin:begin + size]
            wrapped = [self._wrap_prompt(r["text"]) for r in batch]
            encoded = self._encode([value[0] for value in wrapped])
            offsets = encoded.pop("offset_mapping").tolist()
            masks = encoded["attention_mask"].tolist()
            positions = []
            for i, row in enumerate(batch):
                if readout == "final_token":
                    pos = [sum(masks[i]) - 1]
                else:
                    span = _record_span(row, readout)
                    pos = _span_positions(offsets[i], (span[0] + wrapped[i][1], span[1] + wrapped[i][1]))
                if any(not masks[i][p] for p in pos):
                    raise ValueError("Readout points into padding")
                positions.append(pos)
            inputs = {k: v.to(self.device) for k, v in encoded.items()}
            started = time.perf_counter()
            output, handles = {}, []
            def capture(layer):
                def hook(_module, _inputs, value):
                    hidden = value[0] if isinstance(value, tuple) else value
                    output[layer] = np.stack([hidden[i, pos].float().mean(0).detach().cpu().numpy()
                                              for i, pos in enumerate(positions)]).astype(np.float32)
                return hook
            for layer in layers:
                module = self.model.get_input_embeddings() if layer == 0 else self._blocks()[layer - 1]
                handles.append(module.register_forward_hook(capture(layer)))
            try:
                with torch.inference_mode():
                    base = getattr(self.model, "model", None)
                    feature_model = base if callable(base) else self.model
                    result = feature_model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
            finally:
                for handle in handles:
                    handle.remove()
            if set(output) != set(layers):
                raise ValueError("A registered extraction hook did not execute")
            self.telemetry.append({"stage": "features", "rows": len(batch), "tokens": int(sum(map(sum, masks))),
                                   "seconds": time.perf_counter() - started, **_memory_metrics()})
            del result, inputs, encoded
            value = (list(range(begin, begin + len(batch))), output)
            yield (*value, positions) if return_positions else value

    def embedding_only(self, records, readout="span_mean", batch_size=None, return_positions=False):
        """Raw input-embedding floor, without executing transformer blocks."""
        import torch
        self.load()
        readout = "span_mean" if readout == "source_span" else readout
        if readout not in ("span_mean", "carrier_span", "final_token"):
            raise ValueError("Unsupported embedding readout")
        size = int(batch_size or self.batch_size)
        for begin in range(0, len(records), size):
            batch = records[begin:begin + size]
            wrapped = [self._wrap_prompt(row["text"]) for row in batch]
            encoded = self._encode([value[0] for value in wrapped])
            offsets = encoded.pop("offset_mapping").tolist()
            masks = encoded["attention_mask"].tolist()
            positions = []
            for i, row in enumerate(batch):
                if readout == "final_token":
                    positions.append([sum(masks[i]) - 1])
                else:
                    span = _record_span(row, readout)
                    positions.append(_span_positions(offsets[i], (span[0] + wrapped[i][1], span[1] + wrapped[i][1])))
            started = time.perf_counter()
            with torch.inference_mode():
                embeddings = self.model.get_input_embeddings()(encoded["input_ids"].to(self.device))
                values = np.stack([embeddings[i, pos].float().mean(0).cpu().numpy() for i, pos in enumerate(positions)]).astype(np.float32)
            self.telemetry.append({"stage": "embedding_only", "rows": len(batch), "transformer_forwards": 0,
                                   "seconds": time.perf_counter() - started, **_memory_metrics()})
            output = (list(range(begin, begin + len(batch))), {0: values})
            yield (*output, positions) if return_positions else output

    def input_entity_features(self, records):
        """Mean input embeddings of the three source entities, in role order."""
        import torch
        self.load()
        result = []
        with torch.inference_mode():
            embedding = self.model.get_input_embeddings()
            for row in records:
                if len(row.get("entities", [])) != 3 or "assignment" not in row:
                    raise ValueError("S3 potential features need entities and assignment")
                vectors = []
                for entity_index in row["assignment"]:
                    text = row["entities"][entity_index]
                    if text not in self._entity_embedding_cache:
                        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                        if not ids:
                            raise ValueError("Empty entity tokenization")
                        tensor = torch.tensor(ids, device=embedding.weight.device)
                        self._entity_embedding_cache[text] = embedding(tensor).float().mean(0).cpu().numpy()
                    vectors.append(self._entity_embedding_cache[text])
                result.append(np.concatenate(vectors))
        return np.asarray(result, dtype=np.float32)

    def token_positions(self, prompt, span):
        encoded, _, shift = self._encode_prompt(prompt)
        return _span_positions(encoded["offset_mapping"][0].tolist(), (span[0] + shift, span[1] + shift))

    def _component_module(self, component):
        layer = int(component["layer"])
        if not 1 <= layer <= self.n_layers:
            raise ValueError("Component layer must be1..n_layers")
        block = self._blocks()[layer - 1]
        kind = component.get("module", "block")
        if kind == "block":
            return block
        if kind == "mlp":
            module = getattr(block, "mlp", None)
        elif kind in ("attention", "attn"):
            module = getattr(block, "self_attn", getattr(block, "attn", None))
        else:
            raise ValueError(f"Unknown component module {kind}")
        if module is None:
            raise ValueError(f"Model has no registered {kind} component adapter")
        return module

    def _component_positions(self, prompt, component, donor=False):
        positions = component.get("donor_positions" if donor else "positions")
        span = component.get("donor_span" if donor else "span")
        if positions is None and span is None and donor:
            positions, span = component.get("positions"), component.get("span")
        if positions is None and span is None:
            raise ValueError("Component interventions require explicit positions or character span")
        positions = list(map(int, positions)) if positions is not None else self.token_positions(prompt, span)
        length = int(self._encode_prompt(prompt)[0]["attention_mask"].sum())
        positions = [position + length if position < 0 else position for position in positions]
        if not positions or len(set(positions)) != len(positions) or min(positions) < 0 or max(positions) >= length:
            raise ValueError("Component positions must be unique and inside the original prefix")
        return positions

    def _donor_component_values(self, donor_prompt, components):
        import torch
        required = [i for i, c in enumerate(components) if c.get("mode", "zero") in ("resample", "mean") and c.get("value") is None]
        if not required:
            return {}
        if donor_prompt is None:
            raise ValueError("Resample/mean components require a donor_prompt or explicit construction-derived value")
        prepared = [(i, self._component_positions(donor_prompt, components[i], donor=True),
                     self._component_module(components[i])) for i in required]
        values, handles = {}, []
        for i, positions, module in prepared:
            def capture(_module, _inputs, output, index=i, pos=positions):
                hidden = output[0] if isinstance(output, tuple) else output
                values[index] = hidden[0, pos].detach().float().cpu().clone()
            handles.append(module.register_forward_hook(capture))
        encoded = self._encode_prompt(donor_prompt, offsets=False)[0]
        try:
            with torch.inference_mode():
                base = getattr(self.model, "model", None)
                (base if callable(base) else self.model)(**{k: v.to(self.device) for k, v in encoded.items()},
                                                         use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if set(values) != set(required):
            raise ValueError("A donor component hook did not execute")
        return values

    @contextmanager
    def _component_patch(self, prompt, components, donor_values):
        import torch
        prepared = []
        for component in components or []:
            mode = component.get("mode", "zero")
            if mode not in ("zero", "resample", "mean") or not np.isfinite(float(component.get("dose", 1.0))):
                raise ValueError("Component mode must be zero/resample/mean with finite dose")
            prepared.append((component, self._component_positions(prompt, component), self._component_module(component), mode))
        handles = []
        for index, (component, positions, module, mode) in enumerate(prepared):
            called = [False]
            def hook(_module, _inputs, output, c=component, i=index, pos=positions, state=called, selected_mode=mode):
                if state[0]:
                    return output
                state[0] = True
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.shape[0] != 1 or hidden.shape[1] <= max(pos):
                    raise ValueError("Component hook did not receive the original prompt prefill")
                changed = hidden.clone()
                original = changed[0, pos, :]
                if selected_mode == "zero":
                    replacement = torch.zeros_like(original)
                else:
                    value = c.get("value")
                    value = donor_values.get(i) if value is None else value
                    replacement = torch.as_tensor(value, device=hidden.device, dtype=hidden.dtype)
                    if selected_mode == "mean":
                        if replacement.ndim == 2:
                            replacement = replacement.mean(0)
                        replacement = replacement.expand_as(original)
                    elif replacement.ndim == 1:
                        replacement = replacement.expand_as(original)
                    if replacement.shape != original.shape:
                        raise ValueError("Donor/recipient component positions must match; use mode mean for explicit mean broadcasting")
                if not torch.isfinite(replacement).all():
                    raise ValueError("Nonfinite component intervention")
                delta = replacement - original
                if c.get("basis") is not None:
                    basis = torch.as_tensor(c["basis"], device=hidden.device, dtype=hidden.dtype)
                    if basis.ndim != 2 or basis.shape[0] != hidden.shape[-1]:
                        raise ValueError("Component basis must be a full-D by rank matrix")
                    gram = basis.float().T @ basis.float()
                    if not torch.allclose(gram, torch.eye(basis.shape[1], device=basis.device), rtol=2e-3, atol=2e-3):
                        raise ValueError("Component basis must have orthonormal columns")
                    delta = delta @ basis @ basis.T
                changed[0, pos, :] = original + float(c.get("dose", 1.0)) * delta
                return (changed,) + output[1:] if isinstance(output, tuple) else changed
            handles.append(module.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def _patch(self, prompt, patch, once=False):
        if patch is None:
            yield
            return
        import torch
        if isinstance(patch, dict):
            patch = PatchSpec(**patch)
        if patch.layer < 1 or patch.layer > self.n_layers:
            raise ValueError("Patching requires a raw transformer block output index1..n_layers")
        if patch.positions is None and patch.span is None:
            raise ValueError("Patch requires explicit positions or character span")
        positions = patch.positions if patch.positions is not None else self.token_positions(prompt, patch.span)
        encoded = self._encode_prompt(prompt)[0]
        prompt_length = int(encoded["attention_mask"].sum())
        positions = [int(position) + prompt_length if int(position) < 0 else int(position) for position in positions]
        if not positions or len(set(positions)) != len(positions) or min(positions) < 0 or max(positions) >= prompt_length:
            raise ValueError("Patch positions must be within the original unpadded prefix")
        layers = self._blocks()
        calls = [0]
        def hook(_module, _inputs, output):
            if once and calls[0]:
                return output
            calls[0] += 1
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[0] != 1 or hidden.shape[1] <= max(positions):
                raise ValueError("Patch hook received an incompatible prefill tensor")
            h = hidden.clone()
            original = h[0, positions, :]
            mean = original.mean(0)
            if patch.delta is not None:
                delta = torch.as_tensor(patch.delta, device=h.device, dtype=h.dtype)
                replacement = mean + delta
            else:
                replacement = mean if patch.replacement_mean is None else torch.as_tensor(patch.replacement_mean, device=h.device, dtype=h.dtype)
                delta = replacement - mean
            if replacement.shape != mean.shape or not torch.isfinite(replacement).all():
                raise ValueError("Patch replacement/delta must be a finite full-D vector")
            if patch.mode == "delta":
                h[0, positions, :] = original + float(patch.dose) * delta
            else:
                h[0, positions, :] = original + float(patch.dose) * (replacement - original)
            return (h,) + output[1:] if isinstance(output, tuple) else h
        handle = layers[patch.layer - 1].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def score(self, prompt, candidates, patch=None, generation=False, max_new_tokens=32,
              components=None, donor_prompt=None):
        import torch
        self.load()
        started = time.perf_counter()
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("Candidate answers must be nonempty and unique")
        prefix, wrapped_prompt, _ = self._encode_prompt(prompt)
        prefix_ids = prefix["input_ids"][0].tolist()
        donor_values = self._donor_component_values(donor_prompt, components or [])
        prefill_inputs = {k: v.to(self.device) for k, v in prefix.items() if k != "offset_mapping"}
        with self._patch(prompt, patch), self._component_patch(prompt, components, donor_values), torch.inference_mode():
            prefill = self.model(**prefill_inputs, output_hidden_states=False, use_cache=False, return_dict=True)
            prefill_logits = prefill.logits[0, len(prefix_ids) - 1].float().detach().clone()
            prefill_logp = prefill_logits.log_softmax(-1)
        del prefill, prefill_inputs
        scores, means, counts, first_logits, first_probs, first_ids = [], [], [], [], [], []
        for candidate in candidates:
            suffix = candidate if candidate[:1].isspace() else " " + candidate
            encoded = self._encode([wrapped_prompt + suffix])
            offsets = encoded.pop("offset_mapping")[0].tolist()
            candidate_positions = [i for i, (a, b) in enumerate(offsets) if b > len(wrapped_prompt)]
            if not candidate_positions or min(candidate_positions) < 1:
                raise ValueError("Candidate answer has no scoreable continuation tokens")
            first = min(candidate_positions)
            if encoded["input_ids"][0, :first].tolist() != prefix_ids:
                raise ValueError("Tokenizer changes the prompt boundary; use a stable explicit answer separator")
            inputs = {k: v.to(self.device) for k, v in encoded.items()}
            with self._patch(prompt, patch), self._component_patch(prompt, components, donor_values), torch.inference_mode():
                out = self.model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
                logp = out.logits[0].float().log_softmax(-1)
                values = [logp[position - 1, inputs["input_ids"][0, position]] for position in candidate_positions]
                score = float(torch.stack(values).sum().cpu())
                first_logits.append(float(prefill_logits[inputs["input_ids"][0, first]].cpu()))
                first_probs.append(float(prefill_logp[inputs["input_ids"][0, first]].exp().cpu()))
                first_ids.append(int(inputs["input_ids"][0, first]))
            scores.append(score)
            counts.append(len(candidate_positions))
            means.append(score / len(candidate_positions))
            del out, inputs, encoded
        prediction = candidates[int(np.argmax(scores))]
        generated_text = None
        if generation:
            if len(prefix_ids) + int(max_new_tokens) > int(getattr(self.model.config, "max_position_embeddings", 2**31)):
                raise ValueError("Requested generation would exceed the model context window")
            encoded = self._encode_prompt(prompt, offsets=False)[0]
            inputs = {k: v.to(self.device) for k, v in encoded.items()}
            with self._patch(prompt, patch, once=True), self._component_patch(prompt, components, donor_values), torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=int(max_new_tokens),
                                                use_cache=True, pad_token_id=self.tokenizer.pad_token_id)
            generated_text = self.tokenizer.decode(generated[0, len(prefix_ids):], skip_special_tokens=True).strip()
        self.telemetry.append({"stage": "score", "teacher_forcing_forwards": len(candidates),
                               "diagnostic_prefill_forwards": 1,
                               "donor_component_forwards": int(bool(donor_values)),
                               "generation_forwards": int(generated.shape[1] - len(prefix_ids)) if generation else 0,
                               "candidate_tokens": int(sum(counts)), "prefix_tokens": len(prefix_ids),
                               "seconds": time.perf_counter() - started, **_memory_metrics()})
        return {"candidate_logprobs": scores, "candidate_mean_logprobs": means,
                "candidate_first_token_logits": first_logits,
                "candidate_first_token_probabilities": first_probs, "candidate_first_token_ids": first_ids,
                "other_first_token_probability": max(0.0, 1.0 - sum(dict(zip(first_ids, first_probs)).values())),
                "candidate_token_counts": counts, "candidates": list(candidates),
                "candidate_scores": dict(zip(candidates, scores)), "prediction": prediction,
                "constrained_prediction": prediction, "generated_text": generated_text}

    def score_components(self, prompt, candidates, components, donor_prompt=None,
                         generation=False, max_new_tokens=32, patch=None):
        """Real component ablation/resampling; each score/generation prefill resets hooks."""
        return self.score(prompt, candidates, patch=patch, generation=generation,
                          max_new_tokens=max_new_tokens, components=components, donor_prompt=donor_prompt)


def runtime_from_config(model_config, runtime_config=None):
    runtime_config = dict(runtime_config or {})
    return ModelRuntime(model_config["id"], revision=model_config.get("revision"),
                        dtype=runtime_config.get("dtype", "float16"),
                        device_map=runtime_config.get("device_map", runtime_config.get("device", "auto")),
                        batch_size=runtime_config.get("batch_size", 8),
                        local_files_only=runtime_config.get("local_files_only", False),
                        chat_template=model_config.get("chat_template", runtime_config.get("chat_template", False)),
                        allow_cpu_offload=runtime_config.get("allow_cpu_offload", False),
                        deterministic=runtime_config.get("deterministic", True),
                        attention_implementation=runtime_config.get("attention_implementation", runtime_config.get("attn_implementation", "eager")))
