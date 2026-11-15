from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bsg.data.c7 import build_dataset as build_c7, transform_state as c7_transform
from bsg.data.d4 import build_dataset as build_d4, transform_state as d4_transform
from bsg.data.splits import split_templates_three_way
from bsg.experiments.algebra_falsification_config import afs_matrix, resolve_afs_config, validate_primary_coverage
from bsg.experiments.algebra_falsification import _build_manifest, _execution_modules
from bsg.experiments.algebra_falsification_units import (
    HistoricalCheckpointSource, d4_canonical_words, d4_matched_wrong_words, fit_generators,
    fit_scrambled_operators, generator_composition_rows, identity_preserving_permutations,
    is_automorphism, permutation_null_rows, scrambled_training_pairs, summarize_permutation_null,
    verify_embedding_architecture,
)
from bsg.experiments.algebra_falsification_units import _changes_registered_action
from bsg.experiments.fitting import GroupSpec
from bsg.groups import c7, d4
from bsg.metrics.relations import c7_relations, d4_relations
from bsg.operators.independent import fit_independent_ridge


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "algebra_falsification.yaml"


def _spec(name):
    if name == "C7":
        return GroupSpec("C7", build_c7(), c7.ELEMENTS, c7.IDENTITY, c7.compose, c7.inverse, c7_transform, c7_relations())
    return GroupSpec("D4", build_d4(), d4.ELEMENTS, d4.IDENTITY, d4.compose, d4.inverse, d4_transform, d4_relations())


def _artifact(spec, dim=4, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(len(spec.frame), dim))
    split = split_templates_three_way(spec.frame, seed=101, train_templates=28, validation_templates=6, test_templates=6)
    operators = fit_independent_ridge(spec.frame, z, split.train_mask, spec.elements, spec.transform_state, alpha=0.01)
    pca = SimpleNamespace(dim=dim, explained_variance=0.75)
    return {"z": z, "operators": operators, "pca": pca}, split


def test_afs_tier_matrices_and_rng_namespaces_are_exact():
    expected = {
        "smoke": (2, 2, 12, 200, 4, 200),
        "pre_full": (16, 8, 63, 16000, 160, 2000),
        "full": (240, 60, 799, 1200000, 2400, 5000),
    }
    for tier, values in expected.items():
        config = resolve_afs_config(CONFIG, tier)
        matrix = afs_matrix(config)
        observed = tuple(matrix[key] for key in (
            "primary_cells", "embedding_cells", "scientific_units", "permutation_draws",
            "scramble_fits", "bootstrap_replicates_per_contrast",
        ))
        assert observed == values
        namespaces = [set(config["split_seeds"]), set(config["scramble_seeds"]), {config["algebra_permutation_seed"]}, {config["bootstrap_seed"]}, {config["random_generator_seed"]}]
        assert all(not namespaces[i] & namespaces[j] for i in range(len(namespaces)) for j in range(i + 1, len(namespaces)))


def test_abc_only_execution_manifest_preserves_frozen_matrix_and_omits_d():
    config = resolve_afs_config(CONFIG, "full")
    manifest = _build_manifest(config, "afs_full_abc_test", ROOT, include_embedding=False)
    assert _execution_modules(False) == ["A", "B", "C"]
    assert manifest["execution_modules"] == ["A", "B", "C"]
    assert manifest["matrix"]["primary_cells"] == 240
    assert manifest["matrix"]["executed_primary_units"] == 720
    assert manifest["matrix"]["executed_embedding_units"] == 0


def test_c7_generator_indexing_cycle_and_train_only_fit():
    spec = _spec("C7")
    artifact, split = _artifact(spec)
    fitted = fit_generators(spec, artifact["z"], split.train_mask, 0.01)
    changed = artifact["z"].copy()
    changed[~split.train_mask] += 1000.0
    refitted = fit_generators(spec, changed, split.train_mask, 0.01)
    assert set(fitted) == {c7.GENERATOR}
    assert np.array_equal(fitted[c7.GENERATOR], refitted[c7.GENERATOR])
    rows = pd.DataFrame(generator_composition_rows(spec, artifact, split, alpha=0.01, random_seed=99))
    primary = rows[rows.evaluation_role == "unseen_composite"]
    assert set(primary.step.dropna().astype(int)) == {2, 3, 4, 5, 6}
    assert set(rows[rows.evaluation_role == "cycle_diagnostic"].step.astype(int)) == {7}
    assert set(primary[primary.method == "generator_composed"].word) == {"r^2", "r^3", "r^4", "r^5", "r^6"}
    assert c7.power(c7.GENERATOR, 7) == c7.IDENTITY


def test_d4_canonical_words_row_order_and_wrong_words_are_exact():
    words = d4_canonical_words()
    assert set(words) == set(d4.ELEMENTS)
    assert all(d4.row_word_product(word) == element for element, word in words.items())
    assert d4.row_word_product((d4.S, d4.R, d4.S)) == d4.inverse(d4.R)
    wrong = d4_matched_wrong_words()
    assert set(wrong) == set(d4.ELEMENTS) - {d4.IDENTITY, d4.R, d4.S}
    for target, word in wrong.items():
        assert len(word) == len(words[target])
        assert d4.row_word_product(word) != target


def test_permutation_null_excludes_exact_automorphisms_and_preserves_operators():
    expected_automorphisms = {"C7": 6, "D4": 8}
    for name in ("C7", "D4"):
        spec = _spec(name)
        autos = identity_preserving_permutations(spec, automorphisms=True)
        nulls = identity_preserving_permutations(spec, automorphisms=False)
        assert len(autos) == expected_automorphisms[name]
        assert all(permutation[spec.elements.index(spec.identity)] == spec.elements.index(spec.identity) for permutation in nulls)
        assert all(is_automorphism(spec, permutation) for permutation in autos)
        assert all(not is_automorphism(spec, permutation) for permutation in nulls)
        artifact, split = _artifact(spec)
        before = b"".join(artifact["operators"][element].tobytes() for element in spec.elements)
        rows = permutation_null_rows(spec, artifact, split, samples=11, seed=8675309)
        after = b"".join(artifact["operators"][element].tobytes() for element in spec.elements)
        assert before == after
        assert len(rows) == 22
        assert all(row["sampling"] == "with_replacement" for row in rows)


def test_permutation_lower_tail_formula_is_exact():
    frame = pd.DataFrame({
        "group": ["C7"] * 4, "data_split": ["test"] * 4,
        "true_E_hom_op": [1.0] * 4, "null_E_hom_op": [0.5, 1.0, 2.0, 3.0],
        "true_E_hom_data": [1.0] * 4, "null_E_hom_data": [0.5, 1.0, 2.0, 3.0],
    })
    summary = summarize_permutation_null(frame)
    assert all(row["empirical_p"] == (1 + 2) / (4 + 1) for row in summary)
    assert all(row["true_percentile"] == 50.0 for row in summary)


def test_scramble_preserves_multisets_splits_pca_and_never_reads_heldout_rows():
    spec = _spec("C7")
    artifact, split = _artifact(spec)
    pca_id = id(artifact["pca"])
    x, y, permutations = scrambled_training_pairs(
        spec, artifact["z"], split.train_ids, c7.GENERATOR,
        variant="train_only_state_relabel", seed=70001,
    )
    original_train = artifact["z"][split.train_mask]
    assert sorted(map(tuple, np.round(x, 12))) == sorted(map(tuple, np.round(original_train, 12)))
    assert sorted(map(tuple, np.round(y, 12))) == sorted(map(tuple, np.round(original_train, 12)))
    assert len({tuple(p.tolist()) for p in permutations.values()}) > 1
    assert all(_changes_registered_action(spec, permutation) for permutation in permutations.values())
    changed = artifact["z"].copy(); changed[~split.train_mask] += 999.0
    first, _ = fit_scrambled_operators(spec, artifact, split, variant="template_independent_target_scramble", seed=70003, alpha=0.01)
    second, _ = fit_scrambled_operators(spec, {**artifact, "z": changed}, split, variant="template_independent_target_scramble", seed=70003, alpha=0.01)
    assert all(np.array_equal(first[element], second[element]) for element in spec.elements)
    assert id(artifact["pca"]) == pca_id
    assert set(split.train_ids).isdisjoint(split.validation_ids) and set(split.train_ids).isdisjoint(split.test_ids)
    _, _, second_element_permutations = scrambled_training_pairs(
        spec, artifact["z"], split.train_ids, 2,
        variant="train_only_state_relabel", seed=70001,
    )
    assert all(np.array_equal(permutations[template], second_element_permutations[template]) for template in split.train_ids)


def test_d4_scramble_uses_exact_action_for_every_source_and_element():
    spec = _spec("D4")
    z = np.arange(len(spec.frame), dtype=float)[:, None]
    train_ids = (0,)
    for element in d4.ELEMENTS:
        x, y, permutations = scrambled_training_pairs(spec, z, train_ids, element, variant="train_only_state_relabel", seed=70009)
        permutation = permutations[0]
        lookup = {(int(row.template_id), int(row.state_id)): i for i, row in spec.frame.iterrows()}
        for source in range(4):
            assert x[source, 0] == lookup[(0, int(permutation[source]))]
            assert y[source, 0] == lookup[(0, int(permutation[d4.apply(element, source)]))]


def test_embedding_layer_zero_architecture_and_span_contract():
    class Config:
        model_type = "llama"
        num_hidden_layers = 16
    class Model:
        config = Config()
        def get_input_embeddings(self): return object()
    verify_embedding_architecture(Model(), "llama", 16)
    bad = Model(); bad.config = SimpleNamespace(model_type="qwen2", num_hidden_layers=16)
    try:
        verify_embedding_architecture(bad, "llama", 16)
    except ValueError:
        pass
    else:
        raise AssertionError("architecture mismatch was accepted")
    source = (ROOT / "src" / "bsg" / "representations" / "extract.py").read_text(encoding="utf-8")
    body = source.split("def extract_embedding_representations", 1)[1]
    assert "out.hidden_states[0]" in body
    assert "unique_target_char_span" in body
    assert ".mean(0)" in body


def test_primary_cell_coverage_rejects_missing_cells():
    config = resolve_afs_config(CONFIG, "smoke")
    model = config["selected_models"][0]["id"]
    frame = pd.DataFrame([
        {"model": model, "group": group, "depth_slot": 0.4, "split_seed": 101}
        for group in config["groups"]
    ])
    validate_primary_coverage(frame, config)
    try:
        validate_primary_coverage(frame.iloc[:1], config)
    except ValueError:
        pass
    else:
        raise AssertionError("missing AFS primary cell was accepted")


def test_protocol_freezes_gates_nulls_and_forbidden_interpretations():
    protocol = (ROOT / "docs" / "ALGEBRA_FALSIFICATION_PROTOCOL.md").read_text(encoding="utf-8")
    for required in (
        "FROZEN BEFORE GPU EXECUTION", "sampling is with replacement", "5,000",
        "train_only_state_relabel", "template_independent_target_scramble",
        "A1 C7", "A2 D4", "Models are a fixed robustness panel",
        "proves exact internal groups", "hidden_states[0]",
    ):
        assert required in protocol
