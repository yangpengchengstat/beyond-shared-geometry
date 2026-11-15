"""Scientific row/action/cache contracts and actual tiny offline model checks."""
import json
from collections import Counter

import numpy as np
import pytest

from bsg_next.groups import get_group, pose_tuple
from bsg_next.data import (build_structural_dataset, build_native_dataset, native_word_catalog,
                           action_target_indices, seal_dataset, load_sealed_dataset,
                           build_study_records, select_phase_records, render_native_record)
from bsg_next.extraction import extract_features, read_cache, load_cache, merge_feature_caches


@pytest.mark.parametrize("name", ["C7", "D4", "D4_free8", "S3"])
def test_actions_are_actual_left_actions_and_row_products(name):
    group = get_group(name)
    matrices = {}
    for g in group.elements:
        assert group.compose(g, group.inverse(g)) == group.identity
        matrices[g] = np.eye(group.state_count)[[group.apply(g, i) for i in range(group.state_count)]]
    for g in group.elements:
        for h in group.elements:
            for i in range(group.state_count):
                assert group.apply(group.compose(g, h), i) == group.apply(g, group.apply(h, i))
            assert np.array_equal(matrices[h] @ matrices[g], matrices[group.compose(g, h)])
    if group.name == "D4_free8":
        assert len(set(pose_tuple(i) for i in range(8))) == 8
        assert all(group.apply(g, i) != i for g in group.elements if g != group.identity for i in range(8))


def test_structural_full_s3_cross_has_disjoint_entities_and_no_pair_leakage():
    rows = build_structural_dataset("S3")
    assert len(rows) == 55296
    assert Counter(r["split"] for r in rows) == {"train": 13824, "validation": 1536, "test_joint": 6144,
                                               "test_context": 9216, "test_entity": 9216, "reserved": 15360}
    entities = {split: {name for r in rows if r["entity_split"] == split for name in r["entities"]}
                for split in ("train", "validation", "test")}
    assert [len(entities[s]) for s in entities] == [72, 24, 48]
    assert not (entities["train"] & entities["test"] | entities["validation"] & entities["test"])
    development = select_phase_records(rows, "development")
    assert not any(row["entity_split"] == "test" for row in development)
    confirmation = select_phase_records(rows, "confirmation")
    assert {r["row_id"] for r in development}.isdisjoint(r["row_id"] for r in confirmation)
    assert len(development) + len(confirmation) == len(rows)
    targets = action_target_indices(rows)
    for g, indices in targets.items():
        for i in range(0, len(rows), 317):
            source, target = rows[i], rows[indices[i]]
            assert (source["family_id"], source["paraphrase_id"], source["orbit_id"], source["split"]) == (
                target["family_id"], target["paraphrase_id"], target["orbit_id"], target["split"])
            assert target["state_id"] == get_group("S3").apply(g, source["state_id"])


def test_sealed_maximum_pool_prefix_is_not_regenerated_from_pilot_results(tmp_path):
    maximum = build_study_records("C7")
    seal_dataset(maximum, tmp_path / "sealed")
    loaded = load_sealed_dataset(tmp_path / "sealed")
    selected = select_phase_records(loaded, "selected", 32)
    assert selected == build_structural_dataset("C7", test_families=32)
    assert len(selected) == 1344
    development = select_phase_records(loaded, "development", 32)
    assert not any(r["context_split"] == "test" for r in development)
    assert len({r["family_id"] for r in loaded if r["split"] == "pilot"}) == 12
    assert all("conditional_on_fixed_12_root_generator" in r["sampling_scope"] for r in loaded)
    path = tmp_path / "sealed" / "rows.jsonl"
    path.write_text(path.read_text().replace("Monday", "Fakeday", 1))
    with pytest.raises(ValueError, match="seal mismatch"):
        load_sealed_dataset(tmp_path / "sealed")


@pytest.mark.parametrize("domain,nstates", [("C7", 7), ("D4_free8", 8)])
def test_native_catalog_counterfactuals_and_natural_variants(domain, nstates):
    group = get_group(domain)
    words = native_word_catalog(domain)
    assert len(words) == 16 and all(1 <= w["length"] <= 4 for w in words[:8])
    assert all(6 <= w["length"] <= 8 for w in words[8:])
    rows = build_native_dataset(domain)
    assert len(rows) == 56 * 24
    assert sum(r["split"] == "test" for r in rows) == 32 * 24
    assert not any(r["fit_eligible"] for r in rows if r["word_split"] == "transfer")
    assert {r["intermediate_state"] for r in rows if r["split"] == "train"} == set(range(nstates))
    counts = Counter(r["intermediate_state"] for r in rows if r["split"] == "train" and r["paraphrase_id"] == 0)
    assert max(counts.values()) - min(counts.values()) <= 1
    for family in {r["family_id"] for r in rows if r["split"] == "train"}:
        assert {r["intermediate_state"] for r in rows if r["family_id"] == family} == set(range(nstates))
    for row in rows[::29]:
        assert row["intermediate_state"] == group.apply_word(row["initial_state"], row["prefix_elements"])
        assert group.apply_word(row["counterfactual_initial_state"], row["prefix_elements"]) == group.apply(row["edit_element"], row["intermediate_state"])
        for g in group.generators:
            variant = render_native_record(row, edit_element=g)
            assert variant["counterfactual_answer"] == group.state_names[group.apply_word(group.apply(g, row["intermediate_state"]), row["continuation_elements"])]
            a, b = variant["carrier_span"]
            assert variant["text"][a:b] == "Continue"
    factorial = build_native_dataset(domain, factorial=True, splits=("validation",))
    assert len(factorial) == 8 * 2 * nstates * 8
    pilot = build_native_dataset(domain, pilot=True)
    assert sum(row["word_split"] == "transfer" for row in pilot) == 12*2*8
    assert not any(row["fit_eligible"] or row["selection_eligible"] for row in pilot)


def test_equal_support_is_separate_four_contexts_and_not_training():
    from bsg_next.data import build_support_dataset
    rows = build_support_dataset()
    assert len(rows) == 384
    counts = Counter((r["orbit_id"], r["state_id"]) for r in rows)
    assert set(counts.values()) == {4}
    assert all(r["split"] == r["context_split"] == "support" and r["entity_split"] == "test" for r in rows)
    assert not select_phase_records(rows, "development")
    assert select_phase_records(rows, "confirmation") == rows
    maximum = build_support_dataset(test_orbits=24)
    assert len(maximum) == 576
    assert select_phase_records(maximum, "confirmation", test_orbits=16) == rows


class FakeRuntime:
    def __init__(self, stop=False):
        self.stop, self.calls = stop, 0
        self.telemetry = []

    def identity(self):
        return {"model_id": "offline-mock", "model_revision": "1", "dtype": "float32"}

    def features(self, records, layers, readout):
        for begin in range(0, len(records), 3):
            if self.stop and begin >= 3:
                raise RuntimeError("simulated process interruption")
            batch = records[begin:begin + 3]
            self.calls += len(batch)
            x = np.asarray([[r["state_id"], len(r["text"]), int(r["paraphrase_id"]), 1] for r in batch], dtype=np.float32)
            yield list(range(begin, begin + len(batch))), {layer: x + layer for layer in layers}


def test_partial_cache_resume_validates_chunks_and_hit_needs_no_model(tmp_path):
    rows = build_structural_dataset("C7")[:14]
    with pytest.raises(RuntimeError):
        extract_features(FakeRuntime(stop=True), rows, [1, 2], "span_mean", tmp_path / "cache")
    with pytest.raises(ValueError, match="incomplete"):
        load_cache(tmp_path / "cache")
    runtime = FakeRuntime()
    extract_features(runtime, rows, [1, 2], "span_mean", tmp_path / "cache")
    assert runtime.calls == len(rows) - 3
    runtime_hit = FakeRuntime(stop=True)
    extract_features(runtime_hit, rows, [1, 2], "span_mean", tmp_path / "cache")
    assert runtime_hit.calls == 0
    saved_rows, arrays, _ = read_cache(tmp_path / "cache")
    assert saved_rows == rows and arrays[1].dtype == np.float32
    damaged = np.load(tmp_path / "cache" / "layer_1.npy", mmap_mode="r+")
    damaged[0, 0] += 1
    damaged.flush()
    with pytest.raises(ValueError, match="checksum"):
        load_cache(tmp_path / "cache")


def test_cache_merge_reorders_only_verified_disjoint_rows_without_model(tmp_path):
    rows = build_structural_dataset("D4")[:8]
    for label, part in (("a", rows[:4]), ("b", rows[4:])):
        extract_features(FakeRuntime(), part, [1], "span_mean", tmp_path / label)
    merge_feature_caches([tmp_path / "a", tmp_path / "b"], tmp_path / "merged", ordered_records=rows[::-1])
    saved, arrays, _ = read_cache(tmp_path / "merged")
    assert saved == rows[::-1]
    assert arrays[1][0, 0] == rows[-1]["state_id"] + 1
    with pytest.raises(ValueError, match="Duplicate phase"):
        merge_feature_caches([tmp_path / "a", tmp_path / "a"], tmp_path / "bad")


def tiny_runtime(architecture="llama"):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from bsg_next.hf import ModelRuntime
    vocab = {word: i for i, word in enumerate(["[UNK]", "[PAD]", "Monday", "Tuesday", "Wednesday", "is", "day", ".", "Answer", ":", "Today", "The", "long", "record"])}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    torch.manual_seed(741)
    arguments = dict(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                     max_position_embeddings=512, pad_token_id=1, bos_token_id=None, eos_token_id=None)
    if architecture == "qwen2":
        config, cls = transformers.Qwen2Config(**arguments), transformers.Qwen2ForCausalLM
    elif architecture == "gemma2":
        config, cls = transformers.Gemma2Config(**arguments, head_dim=8, sliding_window=256), transformers.Gemma2ForCausalLM
    else:
        config, cls = transformers.LlamaConfig(**arguments), transformers.LlamaForCausalLM
    config._attn_implementation = "eager"
    model = cls(config)
    return ModelRuntime("injected-tiny-" + architecture, revision="test-config", dtype="float32", model=model, tokenizer=tokenizer, batch_size=2)


def test_checkpoint_hardware_smoke_runs_real_tiny_hooks_and_complete_answers(tmp_path):
    from bsg_next.smoke import run_model_smoke
    runtime = tiny_runtime()
    config = {"seed":101, "models":[{"id":"injected-tiny-llama", "layers":[1,2], "hidden_dimension":16}],
              "runtime":{"batch_size":2}}
    result = run_model_smoke(config,tmp_path/"smoke",lambda model,policy:runtime)
    assert result["technical_test"] and not result["paper_eligible"]
    cell = result["models"][0]
    assert cell["zero_delta_candidate_scores_exactly_equal"]
    assert cell["zero_delta_generation_exactly_equal"]
    assert cell["construction_rows"] == 8 and cell["native_construction_rows"] == 1
    assert any(row.get("generation_forwards") == 2 for row in cell["telemetry"])
    assert runtime.model is None


def test_actual_tiny_llama_pooled_readouts_respect_padding_and_raw_terminal():
    runtime = tiny_runtime()
    records = [{"text": "Today is Monday. Answer:", "target": "Monday"},
               {"text": "The long record is Tuesday. Answer:", "target": "Tuesday"}]
    batched = next(runtime.features(records, [0, 1, 2], "span_mean"))[1]
    for i, record in enumerate(records):
        single = next(runtime.features([record], [0, 1, 2], "span_mean"))[1]
        for layer in batched:
            np.testing.assert_allclose(batched[layer][i], single[layer][0], rtol=2e-5, atol=2e-6)
    # The last readout is raw block output, not final LayerNorm output.
    import torch
    encoded = runtime._encode([records[0]["text"]]); offsets = encoded.pop("offset_mapping")
    with torch.inference_mode():
        normalized = runtime.model(**encoded, output_hidden_states=True).hidden_states[-1]
    pos = runtime.token_positions(records[0]["text"], (9, 15))
    assert not np.allclose(batched[2][0], normalized[0, pos].mean(0).detach().numpy())


def test_actual_tiny_llama_complete_answers_sham_and_prefill_only_generation():
    from bsg_next.hf import PatchSpec
    runtime = tiny_runtime()
    prompt = "Today is Monday. Answer:"
    candidates = ["Monday Tuesday", "Wednesday"]
    plain = runtime.score(prompt, candidates)
    assert plain["candidate_token_counts"] == [2, 1]
    sham = runtime.score(prompt, candidates, PatchSpec(layer=1, span=(9, 15)))
    np.testing.assert_allclose(plain["candidate_logprobs"], sham["candidate_logprobs"], rtol=0, atol=0)
    assert 0 <= sham["other_first_token_probability"] <= 1
    generated = runtime.score(prompt, candidates, PatchSpec(layer=1, span=(9, 15)), generation=True, max_new_tokens=2)
    assert isinstance(generated["generated_text"], str)


def test_actual_components_zero_resample_mean_and_hook_cleanup():
    runtime = tiny_runtime()
    prompt, donor = "Today is Monday. Answer:", "Today is Tuesday. Answer:"
    candidates = ["Monday", "Tuesday"]
    position = runtime.token_positions(prompt, (len(prompt) - 1, len(prompt)))[0]
    zero = runtime.score_components(prompt, candidates, [{"layer": 2, "module": "block", "positions": [position], "mode": "zero"}], generation=True, max_new_tokens=2)
    np.testing.assert_allclose(zero["candidate_first_token_logits"], 0, atol=1e-9)
    natural = runtime.score(donor, candidates)
    swapped = runtime.score_components(prompt, candidates, [{"layer": 2, "module": "block", "positions": [position], "mode": "resample"}], donor_prompt=donor)
    np.testing.assert_allclose(swapped["candidate_first_token_logits"], natural["candidate_first_token_logits"], atol=1e-7)
    for module in ("mlp", "attention"):
        baseline = runtime.score(prompt, candidates)
        sham = runtime.score_components(prompt, candidates, [{"layer": 1, "module": module, "positions": [position], "mode": "zero", "dose": 0}])
        assert sham["candidate_logprobs"] == baseline["candidate_logprobs"]
        changed = runtime.score_components(prompt, candidates, [{"layer": 1, "module": module, "positions": [position], "mode": "mean"}], donor_prompt=donor)
        assert all(np.isfinite(changed["candidate_logprobs"]))
    with pytest.raises(ValueError, match="positions must match"):
        runtime.score_components(prompt, candidates, [{"layer": 1, "module": "block", "positions": [position], "mode": "resample", "value": np.ones((2, 16))}])
    assert not runtime.model.model.layers[0]._forward_hooks


def test_chat_wrapper_preserves_raw_span_alignment_for_features_and_patches():
    from bsg_next.hf import PatchSpec
    runtime = tiny_runtime()
    runtime.chat_template = True
    runtime.tokenizer.chat_template = "User: {{ messages[0]['content'] }} Assistant:"
    prompt = "Today is Monday. Answer:"
    positions = runtime.token_positions(prompt, (9, 15))
    wrapped, shift = runtime._wrap_prompt(prompt)
    encoded = runtime._encode([wrapped])
    actual = [i for i, (a, b) in enumerate(encoded["offset_mapping"][0].tolist()) if b > shift + 9 and a < shift + 15]
    assert positions == actual and positions != tiny_runtime().token_positions(prompt, (9, 15))
    record = {"text": prompt, "target": "Monday"}
    output = next(runtime.features([record], [1], "span_mean", return_positions=True))
    assert output[2] == [positions]
    plain = runtime.score(prompt, ["Monday", "Tuesday"])
    sham = runtime.score(prompt, ["Monday", "Tuesday"], PatchSpec(layer=1, span=(9, 15)))
    assert plain["candidate_logprobs"] == sham["candidate_logprobs"]
    assert runtime.identity()["add_special_tokens"] is False


@pytest.mark.parametrize("architecture", ["qwen2", "gemma2"])
def test_qwen_and_gemma_actual_tiny_forward_patch_and_multitoken_score(architecture):
    from bsg_next.hf import PatchSpec
    runtime = tiny_runtime(architecture)
    prompt = "Today is Monday. Answer:"
    record = {"text": prompt, "target": "Monday"}
    values = next(runtime.features([record], [1, 2], "span_mean"))[1]
    assert all(value.shape == (1, 16) for value in values.values())
    plain = runtime.score(prompt, ["Monday Tuesday", "Wednesday"])
    sham = runtime.score(prompt, ["Monday Tuesday", "Wednesday"], PatchSpec(layer=1, span=(9, 15)), generation=True, max_new_tokens=2)
    assert plain["candidate_token_counts"] == [2, 1]
    np.testing.assert_allclose(plain["candidate_logprobs"], sham["candidate_logprobs"], atol=0, rtol=0)
    assert isinstance(sham["generated_text"], str)


def test_embedding_floor_does_not_execute_transformer(monkeypatch):
    runtime = tiny_runtime()
    runtime.load()
    def forbidden(*args, **kwargs):
        raise AssertionError("Embedding-only floor executed a transformer")
    monkeypatch.setattr(runtime.model.model, "forward", forbidden)
    records = [{"text": "Today is Monday. Answer:", "target": "Monday"}]
    output = next(runtime.features(records, [0], "span_mean"))[1][0]
    assert output.shape == (1, 16)
    assert runtime.telemetry[-1]["transformer_forwards"] == 0


def test_multitoken_delta_hook_preserves_live_token_deviations():
    torch = pytest.importorskip("torch")
    from bsg_next.hf import PatchSpec
    runtime = tiny_runtime()
    prompt = "Today is Monday Tuesday. Answer:"
    runtime.load()
    positions = runtime.token_positions(prompt, (9, 23))
    assert len(positions) == 2
    values = {}
    def save(key):
        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            values[key] = hidden[0, positions].detach().clone()
        return capture
    block = runtime.model.model.layers[0]
    before = block.register_forward_hook(save("before"))
    encoded = runtime._encode([prompt], offsets=False)
    try:
        with runtime._patch(prompt, PatchSpec(layer=1, positions=positions, delta=np.full(16, .2))):
            after = block.register_forward_hook(save("after"))
            try:
                with torch.inference_mode():
                    runtime.model(**encoded)
            finally:
                after.remove()
    finally:
        before.remove()
    np.testing.assert_allclose((values["after"] - values["before"]).numpy(), .2, atol=1e-7)
    np.testing.assert_allclose((values["after"][0] - values["after"][1]).numpy(),
                               (values["before"][0] - values["before"][1]).numpy(), atol=1e-7)
    base = runtime.score(prompt, ["Monday", "Tuesday"])
    negative_sham = runtime.score(prompt, ["Monday", "Tuesday"], PatchSpec(layer=2, positions=[-1]))
    assert base["candidate_logprobs"] == negative_sham["candidate_logprobs"]


def test_rebinding_and_task_shams_have_actual_targets_without_role_concatenation():
    from bsg_next.data import build_rebinding_records, render_rebinding_record, render_native_sham
    rows = build_rebinding_records(pilot=True, splits=("pilot",))
    row = rows[0]
    moved = render_rebinding_record(row, state_id=get_group("S3").apply("swap_12", row["state_id"]))
    assert row["answer"] != moved["answer"]
    assert row["text"][slice(*row["carrier_span"])] == "Answer"
    assert row["readout"] == "final_token"
    native = build_native_dataset("C7", splits=("validation",))[0]
    for kind in ("list", "initial_state", "unrelated_attribute"):
        sham = render_native_sham(native, kind)
        assert sham["answer"] in sham["candidates"]
        assert sham["counterfactual_answer"] in sham["candidates"]
