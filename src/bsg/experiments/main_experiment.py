"""Single resumable entry point for the canonical final experiment."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bsg.controls.random_orthogonal import random_orthogonal
from bsg.data.c7 import DAYS, TEMPLATES as C7_TEMPLATES, build_dataset as build_c7, transform_state as c7_transform
from bsg.data.d4 import STATES, TEMPLATES as D4_TEMPLATES, build_dataset as build_d4, transform_state as d4_transform
from bsg.data.splits import split_templates_three_way
from bsg.experiments.canonical_units import (
    evaluate_operators,
    fit_base_artifact,
    fit_joint_artifact,
    identity_operators,
    load_artifact,
    matched_random_operators,
    representation_row,
    save_artifact,
)
from bsg.experiments.fitting import GroupSpec
from bsg.experiments.main_config import config_hash, resolve_main_config
from bsg.experiments.main_outputs import (
    STABLE_FIGURES,
    STABLE_TABLES,
    artifact_checksums,
    create_output_tree,
    validate_stable_artifacts,
    write_figure_pair,
    write_frame,
)
from bsg.experiments.manifest import (
    build_manifest,
    git_state,
    package_versions,
    validate_git_policy,
    write_manifest,
)
from bsg.experiments.runtime_state import (
    RunLedger,
    RuntimeTelemetry,
    atomic_write_frame,
    atomic_write_json,
    atomic_write_text,
    experiment_key,
)
from bsg.groups import c7, d4
from bsg.interventions.causal import (
    continuation_token_span,
    fit_source_conditioned_additive,
    intervention_recovery_errors,
    mean_difference_steering,
    next_token_metrics,
    patched_sequence_outputs,
    repeated_operator,
    residual_preserving_intervention,
    sequence_outputs,
    source_conditioned_additive_intervention,
)
from bsg.metrics.cross_model import linear_cka, orthogonal_procrustes
from bsg.metrics.intertwiner import (
    fit_intertwiner,
    intertwining_error,
    row_map,
    row_map_commutation_error,
    shuffled_operators,
)
from bsg.representations.cache import ActivationCache, CacheMetadataMismatch, activation_cache_key
from bsg.representations.extract import extract_hidden_representations, load_tokenizer_and_model
from bsg.statistics.hierarchical import (
    hierarchical_mean_ci,
    hierarchical_spearman_ci,
    paired_baseline_ci,
    paired_difference_frame,
)


def _group_specs() -> dict[str, GroupSpec]:
    return {
        "C7": GroupSpec("C7", build_c7(), c7.ELEMENTS, c7.IDENTITY, c7.compose, c7.inverse, c7_transform, c7_relations()),
        "D4": GroupSpec("D4", build_d4(), d4.ELEMENTS, d4.IDENTITY, d4.compose, d4.inverse, d4_transform, d4_relations()),
    }


def c7_relations():
    from bsg.metrics.relations import c7_relations as relations
    return relations()


def d4_relations():
    from bsg.metrics.relations import d4_relations as relations
    return relations()


def _limit_templates(frame: pd.DataFrame, limit) -> pd.DataFrame:
    if limit is None:
        return frame.reset_index(drop=True)
    ids = sorted(frame.template_id.unique())[: int(limit)]
    return frame[frame.template_id.isin(ids)].reset_index(drop=True)


def _split(spec, config, split_seed):
    counts = config["template_split"]
    return split_templates_three_way(
        spec.frame,
        seed=int(split_seed),
        train_templates=int(counts["train_templates"]),
        validation_templates=int(counts["validation_templates"]),
        test_templates=int(counts["test_templates"]),
    )


def _depth_slot(model_meta, layer):
    return model_meta["depth_slot_by_layer"][str(int(layer))]


def _cache_metadata(config, model_meta, resolved_revision, task, layer, frame, split_identity) -> dict:
    return {
        "model_id": model_meta["id"],
        "model_revision": resolved_revision,
        "tokenizer_revision": resolved_revision,
        "task": task,
        "group_task": task,
        "layer": int(layer),
        "prompt_template_split": split_identity,
        "template_ids": sorted(map(int, frame.template_id.unique())),
        "tokenizer_settings": config["tokenizer"],
        "prompt_construction_version": config["prompt_construction_version"],
        "activation_extraction_convention": config["activation_extraction_convention"],
    }


def _cached_layers(frame, tokenizer, model, model_meta, resolved_revision, task, layers, config, cache, split_identity):
    found, cache_info, keys = {}, {"bytes_read": 0, "bytes_written": 0}, {}
    for layer in layers:
        metadata = _cache_metadata(config, model_meta, resolved_revision, task, layer, frame, split_identity)
        key = activation_cache_key(**metadata)
        keys[layer] = (key, metadata)
        try:
            cached = cache.load(key, expected_metadata=metadata)
        except CacheMetadataMismatch:
            cache.discard(key)
            cached = None
        cache_info["bytes_read"] += int(cache.last_event["bytes_read"])
        if cached is not None:
            found[layer] = cached
    missing = [layer for layer in layers if layer not in found]
    extraction_seconds = 0.0
    if missing:
        started = time.perf_counter()
        activations = extract_hidden_representations(frame, tokenizer, model, batch_size=int(config["batch_size"]))
        extraction_seconds = time.perf_counter() - started
        if hasattr(activations, "numpy"):
            activations = activations.numpy()
        for layer in missing:
            found[layer] = np.asarray(activations[:, layer, :], dtype=np.float32)
            cache.store(keys[layer][0], found[layer], keys[layer][1])
            cache_info["bytes_written"] += int(cache.last_event["bytes_written"])
        del activations
    return found, keys, cache_info, extraction_seconds


def _checkpoint_paths(output_dir: Path, module: str, key: str, suffixes=("csv",)) -> list[Path]:
    digest = key.split(":", 1)[1]
    root = output_dir / "checkpoints" / module
    root.mkdir(parents=True, exist_ok=True)
    return [root / f"{digest}.{suffix}" for suffix in suffixes]


def _common_fields(model_meta, group, layer, latent_dim, split_seed, optimization_seed=None, random_seed=None):
    slot = _depth_slot(model_meta, layer)
    return {
        "model_key": model_meta["key"],
        "model": model_meta["id"],
        "group": group,
        "layer": int(layer),
        "depth_slot": slot,
        "realized_depth": int(layer) / int(model_meta["blocks"]),
        "latent_dim": int(latent_dim),
        "split_seed": int(split_seed),
        "optimization_seed": None if optimization_seed is None else int(optimization_seed),
        "random_baseline_seed": None if random_seed is None else int(random_seed),
    }


def _with_common(rows, common):
    return [{**common, **row} for row in rows]


def _historical_seed_replication(config) -> bool:
    return config.get("learned_fit_semantics") == "historical_seed_replicated"


def _learned_fit_seeds(config) -> list[int]:
    return list(map(int, config["learned_fit_seeds"]))


def _random_seeds_for_fit(config, optimization_seed: int) -> list[int]:
    if _historical_seed_replication(config):
        mapping = config["random_seed_by_optimization_seed"]
        return [int(mapping[str(int(optimization_seed))])]
    return list(map(int, config["random_baseline_seeds"]))


def _full_checkpoint_fields(fields: dict, config: dict, random_seeds, **contracts) -> dict:
    """Bind the complete FULL control contract without changing historical keys."""
    if config.get("tier") != "full":
        return fields
    return {
        **fields,
        "random_baseline_seeds": list(map(int, random_seeds)),
        **contracts,
    }


def _include_structural_cell(config, model_meta, layer, latent_dim, split_seed) -> bool:
    if config["tier"] not in {"pre_full", "full"}:
        return True
    dim = int(latent_dim)
    if dim in set(map(int, config["primary_structural_latent_dimensions"])):
        return True
    slot = _depth_slot(model_meta, layer)
    return (
        slot != "terminal"
        and dim in set(map(int, config["sensitivity_latent_dimensions"]))
        and float(slot) in set(map(float, config["sensitivity_depth_slots"]))
        and int(split_seed) in set(map(int, config["sensitivity_split_seeds"]))
    )


def _cross_latent_dimensions(config) -> list[int]:
    return list(map(int, config.get("cross_model_latent_dimensions", config["structural_latent_dimensions"])))


def _load_or_run_base(spec, hidden, split, model_meta, layer, dim, config, output_dir, ledger, telemetry, required_keys):
    fields = _common_fields(model_meta, spec.name, layer, dim, split.seed)
    key = experiment_key("structural_base", fields)
    required_keys.add(key)
    metrics_path, artifact_path, irrep_path = _checkpoint_paths(output_dir, "structural_base", key, ("csv", "npz", "irrep.csv"))
    if ledger.should_skip(key):
        return load_artifact(artifact_path, spec.elements, seed=split.seed), metrics_path, irrep_path, artifact_path
    started = time.perf_counter()
    telemetry.reset_gpu_peak()
    ledger.begin(key, fields)
    try:
        artifact = fit_base_artifact(spec, hidden, split, dim, config)
        rows = evaluate_operators(
            spec, artifact, split, "independent_ridge",
            optimization_seed=None, random_baseline_seed=None,
        )
        atomic_write_frame(pd.DataFrame(_with_common(rows, fields)), metrics_path)
        irrep = {**fields, "method": "independent_ridge", **representation_row(spec, artifact)}
        atomic_write_frame(pd.DataFrame([irrep]), irrep_path)
        save_artifact(artifact_path, artifact, spec.elements)
        ledger.complete(key, fields, [metrics_path, artifact_path, irrep_path])
        telemetry.record(key, "structural_base", fields, started)
        return artifact, metrics_path, irrep_path, artifact_path
    except Exception as error:
        ledger.fail(key, fields, error)
        telemetry.record(key, "structural_base", fields, started)
        return None, None, None, None


def _load_or_run_joint(spec, base, split, model_meta, layer, dim, opt_seed, random_seeds, config, output_dir, ledger, telemetry, required_keys):
    historical = _historical_seed_replication(config)
    checkpoint_random_seed = int(random_seeds[0]) if historical else None
    fields = _common_fields(model_meta, spec.name, layer, dim, split.seed, opt_seed, checkpoint_random_seed)
    checkpoint_fields = _full_checkpoint_fields(fields, config, random_seeds)
    key = experiment_key("structural_joint", checkpoint_fields)
    required_keys.add(key)
    metrics_path, artifact_path, irrep_path = _checkpoint_paths(output_dir, "structural_joint", key, ("csv", "npz", "irrep.csv"))
    if ledger.should_skip(key):
        return load_artifact(artifact_path, spec.elements, seed=split.seed), metrics_path, irrep_path, artifact_path
    if base is None:
        error = RuntimeError("structural base dependency is incomplete")
        ledger.begin(key, checkpoint_fields); ledger.fail(key, checkpoint_fields, error)
        return None, None, None, None
    started = time.perf_counter()
    telemetry.reset_gpu_peak()
    ledger.begin(key, checkpoint_fields)
    try:
        learned, terms = fit_joint_artifact(spec, base, split, opt_seed, config)
        identity_artifact = {
            "pca": base["pca"], "z": base["z"],
            "operators": identity_operators(spec, dim),
        }
        rows = []
        rows.extend(evaluate_operators(
            spec, learned, split, "group_consistent_joint",
            optimization_seed=opt_seed, random_baseline_seed=checkpoint_random_seed, terms=terms,
        ))
        for random_seed in random_seeds:
            random_artifact = {
                "pca": base["pca"], "z": base["z"],
                "operators": matched_random_operators(spec, dim, random_seed),
            }
            rows.extend(evaluate_operators(
                spec, random_artifact, split, "random_orthogonal",
                optimization_seed=opt_seed, random_baseline_seed=int(random_seed),
            ))
        rows.extend(evaluate_operators(
            spec, identity_artifact, split, "identity",
            optimization_seed=opt_seed, random_baseline_seed=checkpoint_random_seed,
            identity_algebra_not_applicable=True,
        ))
        atomic_write_frame(pd.DataFrame(_with_common(rows, fields)), metrics_path)
        irrep = {**fields, "method": "group_consistent_joint", **representation_row(spec, learned)}
        atomic_write_frame(pd.DataFrame([irrep]), irrep_path)
        save_artifact(artifact_path, learned, spec.elements)
        ledger.complete(key, checkpoint_fields, [metrics_path, artifact_path, irrep_path])
        telemetry.record(key, "structural_joint", fields, started)
        return learned, metrics_path, irrep_path, artifact_path
    except Exception as error:
        ledger.fail(key, checkpoint_fields, error)
        telemetry.record(key, "structural_joint", fields, started)
        return None, None, None, None


def _mean_latent_delta(spec, z, train_mask, element) -> np.ndarray:
    lookup = {(int(row.template_id), int(row.state_id)): index for index, row in spec.frame.iterrows()}
    deltas = []
    for index, row in spec.frame.iterrows():
        if train_mask[index]:
            target = spec.transform_state(int(row.state_id), element)
            deltas.append(z[lookup[(int(row.template_id), target)]] - z[index])
    return np.mean(deltas, axis=0)


def _continuation_info(prompt: str, answer: str, tokenizer):
    _, ids, start = continuation_token_span(prompt, answer, tokenizer)
    answer_ids = ids[start:]
    return int(answer_ids[0]), len(answer_ids)


def _causal_eval_frame(spec, split):
    labels = DAYS if spec.name == "C7" else STATES
    templates = C7_TEMPLATES if spec.name == "C7" else D4_TEMPLATES
    return pd.DataFrame([
        {
            "template_id": int(template_id), "state_id": state_id, "state": label,
            "target": label, "text": templates[template_id].format(label) + " Report the transformed state:",
        }
        for template_id in split.test_ids
        for state_id, label in enumerate(labels)
    ])


def _d4_interventions(operators, wrong_order_controls=()):
    r, s = operators[d4.R], operators[d4.S]
    interventions = [
        ("r", d4.R, d4.R, r, "transformation"),
        ("s", d4.S, d4.S, s, "transformation"),
        ("r2", d4.power(d4.R, 2), d4.power(d4.R, 2), np.linalg.matrix_power(r, 2), "transformation"),
        ("r3", d4.power(d4.R, 3), d4.power(d4.R, 3), np.linalg.matrix_power(r, 3), "transformation"),
        ("r4_identity", d4.IDENTITY, d4.IDENTITY, np.linalg.matrix_power(r, 4), "relation_recovery"),
        ("s2_identity", d4.IDENTITY, d4.IDENTITY, np.linalg.matrix_power(s, 2), "relation_recovery"),
        ("srs_r_inverse", d4.inverse(d4.R), d4.inverse(d4.R), s @ r @ s, "relation_consistency"),
        ("r_inverse_direct", d4.inverse(d4.R), d4.inverse(d4.R), operators[d4.inverse(d4.R)], "relation_reference"),
    ]
    if "rsr_wrong_order" in wrong_order_controls:
        interventions.insert(-1, (
            "rsr_wrong_order", d4.inverse(d4.R),
            d4.row_word_product((d4.R, d4.S, d4.R)), r @ s @ r,
            "order_control",
        ))
    return interventions


def _causal_protocol_identity(spec, config) -> list[dict]:
    if spec.name == "C7":
        return [
            {
                "intervention": f"R^{step}",
                "target_element": repr(c7.power(c7.GENERATOR, step)),
                "word_element": repr(c7.power(c7.GENERATOR, step)),
                "role": "cycle_recovery" if step == 7 else "transformation",
            }
            for step in config["causal"]["c7_steps"]
        ]
    identity = np.eye(1)
    operators = {element: identity for element in d4.ELEMENTS}
    return [
        {
            "intervention": name,
            "target_element": repr(target_element),
            "word_element": repr(word_element),
            "role": role,
        }
        for name, target_element, word_element, _, role in _d4_interventions(
            operators, config["causal"].get("d4_wrong_order_controls", ()),
        )
    ]


def _full_causal_checkpoint_fields(fields, spec, config, random_seeds) -> dict:
    methods = ["learned_operator", "identity"]
    methods.extend(
        method for method in ("source_conditioned_additive", "mean_difference")
        if method in config["causal"]["baselines"]
    )
    methods.append("random_orthogonal")
    return _full_checkpoint_fields(
        fields,
        config,
        random_seeds,
        methods=methods,
        interventions=_causal_protocol_identity(spec, config),
    )


def _random_d4_operator(name, random_r, random_s):
    if name == "r": return random_r
    if name == "s": return random_s
    if name == "r2": return random_r @ random_r
    if name == "r3": return np.linalg.matrix_power(random_r, 3)
    if name == "r4_identity": return np.linalg.matrix_power(random_r, 4)
    if name == "s2_identity": return random_s @ random_s
    if name == "srs_r_inverse": return random_s @ random_r @ random_s
    if name == "rsr_wrong_order": return random_r @ random_s @ random_r
    if name == "r_inverse_direct": return random_r.T
    raise ValueError(f"unknown D4 intervention: {name}")


def _run_causal(spec, fit_hidden, eval_frame, eval_hidden, base_outputs, model, tokenizer, layer, artifact, split, opt_seed, random_seeds, model_meta, config):
    pca, z, operators = artifact["pca"], artifact["z"], artifact["operators"]
    labels = DAYS if spec.name == "C7" else STATES
    if spec.name == "C7":
        generator = operators[c7.GENERATOR]
        interventions = [
            (f"R^{step}", c7.power(c7.GENERATOR, step), c7.power(c7.GENERATOR, step), repeated_operator(generator, step), "cycle_recovery" if step == 7 else "transformation")
            for step in config["causal"]["c7_steps"]
        ]
    else:
        interventions = _d4_interventions(
            operators, config["causal"].get("d4_wrong_order_controls", ()),
        )
    elements = sorted({intervention[1] for intervention in interventions}, key=repr)
    additive_vectors = fit_source_conditioned_additive(
        spec.frame, fit_hidden, split.train_mask, elements, spec.transform_state,
    ) if "source_conditioned_additive" in config["causal"]["baselines"] else None
    random_generators = {}
    for random_seed in random_seeds:
        rng = np.random.default_rng(int(random_seed))
        random_generators[int(random_seed)] = (
            random_orthogonal(pca.dim, rng), random_orthogonal(pca.dim, rng)
        )
    historical = _historical_seed_replication(config)
    historical_random_seed = int(random_seeds[0]) if historical else None
    rows = []
    eps = np.finfo(float).eps
    for index, example in enumerate(eval_frame.itertuples()):
        hidden = np.asarray(eval_hidden[index], dtype=float)
        source_id, source = int(example.state_id), str(example.state)
        source_token, _ = _continuation_info(example.text, f" {source}", tokenizer)
        behavior = {}
        example_rows = []
        for name, target_element, word_element, learned_op, role in interventions:
            target_id = spec.transform_state(source_id, target_element)
            target = labels[target_id]
            target_token, target_token_count = _continuation_info(example.text, f" {target}", tokenizer)
            base = base_outputs[index]
            base_score = next_token_metrics(base["logits"], target_token, source_token)
            methods = [
                ("learned_operator", historical_random_seed, residual_preserving_intervention(hidden, pca, learned_op)),
                ("identity", historical_random_seed, hidden.copy()),
            ]
            if additive_vectors is not None:
                methods.append((
                    "source_conditioned_additive", historical_random_seed,
                    source_conditioned_additive_intervention(
                        hidden, additive_vectors, target_element, source_id,
                    ),
                ))
            if "mean_difference" in config["causal"]["baselines"]:
                delta = _mean_latent_delta(spec, z, split.train_mask, target_element)
                methods.append((
                    "mean_difference", historical_random_seed,
                    mean_difference_steering(hidden, pca, delta),
                ))
            for random_seed, (random_r, random_s) in random_generators.items():
                random_op = (
                    repeated_operator(random_r, int(name.split("^")[1]))
                    if spec.name == "C7" else _random_d4_operator(name, random_r, random_s)
                )
                methods.append((
                    "random_orthogonal", int(random_seed),
                    residual_preserving_intervention(hidden, pca, random_op),
                ))
            for method, method_random_seed, replacement in methods:
                outputs = base if method == "identity" else patched_sequence_outputs(model, tokenizer, example.text, source, layer, replacement)
                score = next_token_metrics(outputs["logits"], target_token, source_token)
                recovery = intervention_recovery_errors(hidden, replacement, pca)
                final_logits = outputs["logits"][0, -1].detach().float().cpu().numpy()
                base_logits = base["logits"][0, -1].detach().float().cpu().numpy()
                downstream = np.asarray(outputs["final_hidden"], dtype=float)
                base_downstream = np.asarray(base["final_hidden"], dtype=float)
                behavior[(name, method, method_random_seed)] = (final_logits, downstream, score)
                row = {
                    "model": model_meta["id"], "model_key": model_meta["key"], "group": spec.name,
                    "layer": int(layer), "depth_slot": _depth_slot(model_meta, layer),
                    "realized_depth": int(layer) / int(model_meta["blocks"]), "latent_dim": int(pca.dim),
                    "split_seed": int(split.seed), "optimization_seed": int(opt_seed),
                    "random_baseline_seed": method_random_seed, "template_id": int(example.template_id),
                    "source_state": source, "target_state": target, "intervention": name,
                    "evaluation_role": role, "method": method,
                    "word_group_element": repr(word_element),
                    "target_logit_change": score["target_logit"] - base_score["target_logit"],
                    "target_probability_change": score["target_probability"] - base_score["target_probability"],
                    "top1_success": int(score["top1_success"]),
                    "logit_margin_vs_original": score["logit_margin_vs_original"],
                    "patched_hidden_displacement": recovery["hidden_relative_error"],
                    "downstream_hidden_displacement": float(np.linalg.norm(downstream - base_downstream) / max(np.linalg.norm(base_downstream), eps)),
                    "final_logit_displacement": float(np.linalg.norm(final_logits - base_logits) / max(np.linalg.norm(base_logits), eps)),
                    "target_sequence_log_probability": math.log(max(score["target_probability"], eps)) if target_token_count == 1 else np.nan,
                    "target_sequence_token_count": int(target_token_count),
                    "cycle_hidden_relative_error": recovery["hidden_relative_error"] if role in {"cycle_recovery", "relation_recovery"} else np.nan,
                    "cycle_latent_relative_error": recovery["latent_relative_error"] if role in {"cycle_recovery", "relation_recovery"} else np.nan,
                    "cycle_probability_recovery_error": abs(score["target_probability"] - base_score["target_probability"]) if role in {"cycle_recovery", "relation_recovery"} else np.nan,
                    "relation_final_logit_distance": np.nan,
                    "relation_downstream_hidden_distance": np.nan,
                    "relation_target_probability_difference": np.nan,
                    "relation_logit_margin_difference": np.nan,
                    "relation_sequence_log_probability_difference": np.nan,
                }
                example_rows.append(row)
        if spec.name == "D4":
            method_keys = sorted({(method, seed) for _, method, seed in behavior}, key=repr)
            for method, method_random_seed in method_keys:
                srs_logits, srs_hidden, srs_score = behavior[("srs_r_inverse", method, method_random_seed)]
                direct_logits, direct_hidden, direct_score = behavior[("r_inverse_direct", method, method_random_seed)]
                logit_distance = float(np.linalg.norm(srs_logits - direct_logits) / max(np.linalg.norm(direct_logits), eps))
                hidden_distance = float(np.linalg.norm(srs_hidden - direct_hidden) / max(np.linalg.norm(direct_hidden), eps))
                probability_difference = abs(srs_score["target_probability"] - direct_score["target_probability"])
                margin_difference = abs(
                    srs_score["logit_margin_vs_original"] - direct_score["logit_margin_vs_original"]
                )
                for row in example_rows:
                    same_method = row["method"] == method and row["random_baseline_seed"] == method_random_seed
                    if row["intervention"] == "srs_r_inverse" and same_method:
                        row["relation_final_logit_distance"] = logit_distance
                        row["relation_downstream_hidden_distance"] = hidden_distance
                        row["relation_target_probability_difference"] = probability_difference
                        row["relation_logit_margin_difference"] = margin_difference
                        row["relation_sequence_log_probability_difference"] = abs(
                            math.log(max(srs_score["target_probability"], eps))
                            - math.log(max(direct_score["target_probability"], eps))
                        )
                if "rsr_wrong_order" not in {intervention[0] for intervention in interventions}:
                    continue
                wrong_logits, wrong_hidden, wrong_score = behavior[("rsr_wrong_order", method, method_random_seed)]
                wrong_logit_distance = float(np.linalg.norm(wrong_logits - direct_logits) / max(np.linalg.norm(direct_logits), eps))
                wrong_hidden_distance = float(np.linalg.norm(wrong_hidden - direct_hidden) / max(np.linalg.norm(direct_hidden), eps))
                for row in example_rows:
                    same_method = row["method"] == method and row["random_baseline_seed"] == method_random_seed
                    if row["intervention"] == "rsr_wrong_order" and same_method:
                        row["relation_final_logit_distance"] = wrong_logit_distance
                        row["relation_downstream_hidden_distance"] = wrong_hidden_distance
                        row["relation_target_probability_difference"] = abs(
                            wrong_score["target_probability"] - direct_score["target_probability"]
                        )
                        row["relation_logit_margin_difference"] = abs(
                            wrong_score["logit_margin_vs_original"] - direct_score["logit_margin_vs_original"]
                        )
                        row["relation_sequence_log_probability_difference"] = abs(
                            math.log(max(wrong_score["target_probability"], eps))
                            - math.log(max(direct_score["target_probability"], eps))
                        )
        rows.extend(example_rows)
    return rows


def _load_or_run_causal(spec, fit_hidden, eval_frame, eval_hidden, base_output_cache, model, tokenizer, layer, artifact, split, opt_seed, random_seeds, model_meta, config, output_dir, ledger, telemetry, required_keys):
    checkpoint_random_seed = int(random_seeds[0]) if _historical_seed_replication(config) else None
    fields = _common_fields(model_meta, spec.name, layer, config["primary_latent_dimension"], split.seed, opt_seed, checkpoint_random_seed)
    checkpoint_fields = _full_causal_checkpoint_fields(fields, spec, config, random_seeds)
    key = experiment_key("causal", checkpoint_fields)
    required_keys.add(key)
    [path] = _checkpoint_paths(output_dir, "causal", key)
    if ledger.should_skip(key):
        return path
    if artifact is None:
        error = RuntimeError("joint artifact dependency is incomplete")
        ledger.begin(key, checkpoint_fields); ledger.fail(key, checkpoint_fields, error)
        return None
    started = time.perf_counter()
    telemetry.reset_gpu_peak()
    ledger.begin(key, checkpoint_fields)
    try:
        base_key = (model_meta["key"], spec.name, int(split.seed))
        if base_key not in base_output_cache:
            base_output_cache[base_key] = [sequence_outputs(model, tokenizer, row.text) for row in eval_frame.itertuples()]
        rows = _run_causal(
            spec, fit_hidden, eval_frame, eval_hidden, base_output_cache[base_key], model, tokenizer, layer,
            artifact, split, opt_seed, random_seeds, model_meta, config,
        )
        atomic_write_frame(pd.DataFrame(rows), path)
        ledger.complete(key, checkpoint_fields, [path])
        telemetry.record(key, "causal", fields, started)
        return path
    except Exception as error:
        ledger.fail(key, checkpoint_fields, error)
        telemetry.record(key, "causal", fields, started)
        return None


def _read_artifact_index_entry(index, model_key, group, slot, split_seed, dim, opt_seed):
    return index[(model_key, group, float(slot), int(split_seed), int(dim), int(opt_seed))]


def _structural_index_key(model_key, group, slot, split_seed, dim, opt_seed):
    return (model_key, group, float(slot), int(split_seed), int(dim), int(opt_seed))


def _cross_work_pairs(config):
    pairs = [(key_a, key_b, "fixed_pair") for key_a, key_b in config["selected_cross_model_pairs"]]
    if not pairs and config["tier"] == "smoke":
        model_key = config["selected_models"][0]["key"]
        pairs.append((model_key, model_key, "self_smoke_diagnostic"))
    return pairs


def _run_cross_units(config, specs, artifact_index, hidden_index, cache, output_dir, ledger, telemetry, required_keys):
    paths = []
    selected = {model["key"]: model for model in config["selected_models"]}
    geometry_cache = {}
    cka_cache = {}
    for key_a, key_b, comparison_type in _cross_work_pairs(config):
        model_a, model_b = selected[key_a], selected[key_b]
        for group_name in config["groups"]:
            spec = specs[group_name]
            for slot in config["depth_slots"]:
                layer_a = model_a["depth_layers"][float(slot)]
                layer_b = model_b["depth_layers"][float(slot)]
                for split_seed in config["split_seeds"]:
                    split = _split(spec, config, split_seed)
                    for dim in _cross_latent_dimensions(config):
                        for opt_seed in _learned_fit_seeds(config):
                            random_seeds = _random_seeds_for_fit(config, opt_seed)
                            checkpoint_random_seed = int(random_seeds[0]) if _historical_seed_replication(config) else None
                            fields = {
                                "model_A": model_a["id"], "model_B": model_b["id"],
                                "model_key_A": key_a, "model_key_B": key_b,
                                "comparison_type": comparison_type,
                                "group": group_name, "depth_slot": float(slot),
                                "layer_A": int(layer_a), "layer_B": int(layer_b),
                                "split_seed": int(split_seed), "optimization_seed": int(opt_seed),
                                "random_baseline_seed": checkpoint_random_seed, "latent_dim": int(dim),
                            }
                            checkpoint_fields = _full_checkpoint_fields(fields, config, random_seeds)
                            unit_key = experiment_key("cross_model", checkpoint_fields)
                            required_keys.add(unit_key)
                            [path] = _checkpoint_paths(output_dir, "cross_model", unit_key)
                            if ledger.should_skip(unit_key):
                                paths.append(path); continue
                            started = time.perf_counter(); telemetry.reset_gpu_peak(); ledger.begin(unit_key, checkpoint_fields)
                            try:
                                path_a = _read_artifact_index_entry(artifact_index, key_a, group_name, slot, split_seed, dim, opt_seed)
                                path_b = _read_artifact_index_entry(artifact_index, key_b, group_name, slot, split_seed, dim, opt_seed)
                                a = load_artifact(path_a, spec.elements, seed=split_seed)
                                b = load_artifact(path_b, spec.elements, seed=split_seed)
                                hidden_a = cache.load(*hidden_index[(key_a, group_name, layer_a)])
                                hidden_b = cache.load(*hidden_index[(key_b, group_name, layer_b)])
                                if hidden_a is None or hidden_b is None:
                                    raise RuntimeError(
                                        "cross-model activation cache is missing despite a COMPLETE extraction checkpoint"
                                    )
                                geo_key = (key_a, key_b, group_name, float(slot), int(split_seed), int(dim))
                                if geo_key not in geometry_cache:
                                    x, y = a["z"][split.test_mask], b["z"][split.test_mask]
                                    q, proc_error = orthogonal_procrustes(x, y)
                                    geometry_cache[geo_key] = (q, proc_error)
                                q, proc_error = geometry_cache[geo_key]
                                cka_key = (key_a, key_b, group_name, float(slot), int(split_seed))
                                if cka_key not in cka_cache:
                                    cka_cache[cka_key] = linear_cka(
                                        hidden_a[split.test_mask], hidden_b[split.test_mask]
                                    )
                                cka = cka_cache[cka_key]
                                p = fit_intertwiner(a["operators"], b["operators"])
                                fitted_error = intertwining_error(p, a["operators"], b["operators"])
                                proc_comm = row_map_commutation_error(q, a["operators"], b["operators"])
                                deterministic_metrics = {
                                    "linear_cka": cka,
                                    "procrustes_reconstruction_error": proc_error,
                                    "fitted_intertwiner_commutation_error": fitted_error,
                                    "procrustes_map_commutation_error": proc_comm,
                                    "fitted_minus_procrustes_commutation": fitted_error - proc_comm,
                                }
                                rows = [
                                    {**fields, "metric": metric, "value": value}
                                    for metric, value in deterministic_metrics.items()
                                ]
                                for random_seed in random_seeds:
                                    random_map = random_orthogonal(int(dim), np.random.default_rng(int(random_seed)))
                                    random_error = row_map_commutation_error(random_map, a["operators"], b["operators"])
                                    shuffled_b = shuffled_operators(b["operators"], int(random_seed), identity=spec.identity)
                                    shuffled_error = row_map_commutation_error(row_map(p), a["operators"], shuffled_b)
                                    rows.extend([
                                        {**fields, "random_baseline_seed": int(random_seed), "metric": "random_map_commutation_error", "value": random_error},
                                        {**fields, "random_baseline_seed": int(random_seed), "metric": "shuffled_label_commutation_error", "value": shuffled_error},
                                        {**fields, "random_baseline_seed": int(random_seed), "metric": "fitted_minus_shuffled_commutation", "value": fitted_error - shuffled_error},
                                    ])
                                atomic_write_frame(pd.DataFrame(rows), path)
                                ledger.complete(unit_key, checkpoint_fields, [path])
                                telemetry.record(unit_key, "cross_model", fields, started)
                                paths.append(path)
                            except Exception as error:
                                ledger.fail(unit_key, checkpoint_fields, error)
                                telemetry.record(unit_key, "cross_model", fields, started)
    return paths


def _concat_csv(paths, destination: Path) -> pd.DataFrame:
    valid = [Path(path) for path in paths if path is not None and Path(path).is_file()]
    frame = pd.concat([pd.read_csv(path) for path in valid], ignore_index=True) if valid else pd.DataFrame()
    write_frame(frame, destination)
    return frame


def _bootstrap_rows(structural, causal, cross, config, progress_callback=None):
    rows = []
    test = structural[structural.data_split == "test"] if not structural.empty else structural
    if not test.empty:
        primary_depth = pd.to_numeric(test.depth_slot, errors="coerce").notna()
        test = test[(test.latent_dim == int(config["primary_latent_dimension"])) & primary_depth]
    metrics = ["E_eq", "E_rel_data", "E_hom_data", "E_comp_data", "E_inverse_data", "E_word"]
    learned_name = "group_consistent_joint"
    if not test.empty:
        grouped_parts = list(test.groupby(["model", "group"]))
        grouped_parts.extend([(("ALL", group), part) for group, part in test.groupby("group")])
        for (model, group), part in grouped_parts:
            for offset, metric in enumerate(metrics):
                learned = part[part.method == learned_name]
                if not learned.empty:
                    summary = hierarchical_mean_ci(
                        learned, metric, n_boot=config["bootstrap_samples"],
                        seed=config["bootstrap_seed"] + offset, progress_callback=progress_callback,
                    )
                    rows.append({"analysis": "estimate", "model": model, "group": group, "metric": metric, **summary})
                for baseline_index, baseline in enumerate(("random_orthogonal", "identity")):
                    if baseline == "identity" and metric != "E_eq":
                        continue
                    try:
                        summary = paired_baseline_ci(
                            part, metric, learned_name, baseline,
                            n_boot=config["bootstrap_samples"], seed=config["bootstrap_seed"] + 100 + 10 * baseline_index + offset,
                            progress_callback=progress_callback,
                        )
                        rows.append({"analysis": f"learned_minus_{baseline}", "model": model, "group": group, "metric": metric, **summary})
                    except ValueError:
                        pass
                if metric in {"E_eq", "E_rel_data", "E_hom_data", "E_comp_data", "E_inverse_data"}:
                    try:
                        summary = paired_baseline_ci(
                            part, metric, learned_name, "independent_ridge",
                            n_boot=config["bootstrap_samples"],
                            seed=config["bootstrap_seed"] + 300 + offset,
                            progress_callback=progress_callback,
                        )
                        rows.append({
                            "analysis": "joint_minus_independent_ridge",
                            "model": model,
                            "group": group,
                            "metric": metric,
                            "interpretation": "paired difference; no equivalence claim",
                            **summary,
                        })
                    except ValueError:
                        pass
    if not causal.empty:
        causal_parts = list(causal.groupby(["model", "group", "evaluation_role"]))
        causal_parts.extend([(("ALL", group, role), part) for (group, role), part in causal.groupby(["group", "evaluation_role"])])
        for (model, group, role), part in causal_parts:
            for offset, metric in enumerate(("target_logit_change", "target_probability_change", "top1_success", "logit_margin_vs_original", "relation_final_logit_distance")):
                for baseline_index, baseline in enumerate(("random_orthogonal", "identity", "source_conditioned_additive")):
                    try:
                        summary = paired_baseline_ci(
                            part, metric, "learned_operator", baseline,
                            n_boot=config["bootstrap_samples"], seed=config["bootstrap_seed"] + 500 + 10 * baseline_index + offset,
                            progress_callback=progress_callback,
                        )
                        rows.append({"analysis": f"causal_learned_minus_{baseline}", "model": model, "group": group, "method": role, "metric": metric, **summary})
                    except ValueError:
                        pass
        d4_learned = causal[
            (causal.group == "D4")
            & (causal.method == "learned_operator")
            & causal.intervention.isin(["srs_r_inverse", "rsr_wrong_order"])
        ]
        for offset, metric in enumerate((
            "relation_final_logit_distance",
            "relation_downstream_hidden_distance",
            "relation_target_probability_difference",
            "relation_logit_margin_difference",
            "relation_sequence_log_probability_difference",
        )):
            if metric not in d4_learned:
                continue
            d4_parts = list(d4_learned.groupby("model"))
            d4_parts.append(("ALL", d4_learned))
            for model, part in d4_parts:
                try:
                    summary = paired_baseline_ci(
                        part, metric, "srs_r_inverse", "rsr_wrong_order",
                        method_col="intervention",
                        n_boot=config["bootstrap_samples"],
                        seed=config["bootstrap_seed"] + 700 + offset,
                        progress_callback=progress_callback,
                    )
                    rows.append({
                        "analysis": "d4_true_minus_wrong_order",
                        "model": model,
                        "group": "D4",
                        "method": "learned_operator",
                        "metric": metric,
                        "interpretation": "negative favors the true srs relation",
                        **summary,
                    })
                except ValueError:
                    pass
    if not cross.empty:
        cross = cross[cross.latent_dim == int(config["primary_latent_dimension"])]
        for metric in ("fitted_minus_procrustes_commutation", "fitted_minus_shuffled_commutation"):
            part = cross[cross.metric == metric].copy()
            if part.empty:
                continue
            part["model_pair"] = part.model_A.astype(str) + "::" + part.model_B.astype(str)
            part["pair_stratum"] = np.where(
                part.model_A.str.startswith("meta-llama") == part.model_B.str.startswith("meta-llama"),
                "within_family", "cross_family",
            )
            for (group, stratum), selected in part.groupby(["group", "pair_stratum"]):
                try:
                    summary = hierarchical_mean_ci(
                        selected, "value", n_boot=config["bootstrap_samples"],
                        seed=config["bootstrap_seed"] + (800 if metric.endswith("procrustes_commutation") else 900),
                        cluster_columns=("model_pair", "depth_slot", "split_seed", "random_baseline_seed"),
                        preserve_random_seed=metric == "fitted_minus_shuffled_commutation",
                        progress_callback=progress_callback,
                    )
                    rows.append({
                        "analysis": "cross_model_contrast", "model": "ALL", "group": group,
                        "method": stratum, "metric": metric, **summary,
                    })
                except ValueError:
                    pass
        cross_wide = cross.pivot_table(
            index=[
                "model_A", "model_B", "group", "depth_slot", "layer_A", "layer_B",
                "split_seed", "optimization_seed", "latent_dim",
            ],
            columns="metric", values="value", aggfunc="mean",
        ).reset_index()
        required = {"linear_cka", "fitted_intertwiner_commutation_error"}
        if required <= set(cross_wide.columns):
            cross_wide["negative_fitted_commutation"] = -cross_wide["fitted_intertwiner_commutation_error"]
            cross_wide["model_pair"] = cross_wide.model_A.astype(str) + "::" + cross_wide.model_B.astype(str)
            try:
                summary = hierarchical_spearman_ci(
                    cross_wide, "linear_cka", "negative_fitted_commutation",
                    n_boot=config["bootstrap_samples"], seed=config["bootstrap_seed"] + 1000,
                    cluster_columns=("model_pair", "depth_slot", "split_seed"),
                    progress_callback=progress_callback,
                )
                rows.append({
                    "analysis": "cross_model_association", "model": "ALL", "group": "ALL",
                    "method": "spearman", "metric": "cka_vs_negative_fitted_commutation", **summary,
                })
            except ValueError:
                pass
    return pd.DataFrame(rows)


def _generate_stable_artifacts(output_dir, structural, irrep, cross, causal, bootstrap, stage_callback=None):
    table_started = time.perf_counter()
    if stage_callback is not None:
        stage_callback("table_aggregation", "STARTED", table_started)
    tables = output_dir / "results" / "tables"
    figures = output_dir / "results" / "figures"
    test = structural[structural.data_split == "test"] if not structural.empty else structural
    primary = test[(test.latent_dim == 16) & (test.depth_slot != "terminal")] if not test.empty else test
    structural_main = primary.groupby(["model", "group", "method", "depth_slot", "layer"], as_index=False).mean(numeric_only=True) if not primary.empty else pd.DataFrame()
    c7 = causal[(causal.group == "C7")] if not causal.empty else causal
    d4_frame = causal[(causal.group == "D4")] if not causal.empty else causal
    c7_main = c7.groupby(["model", "method", "evaluation_role", "intervention", "depth_slot", "layer"], as_index=False).mean(numeric_only=True) if not c7.empty else pd.DataFrame()
    d4_main = d4_frame.groupby(["model", "method", "evaluation_role", "intervention", "depth_slot", "layer"], as_index=False).mean(numeric_only=True) if not d4_frame.empty else pd.DataFrame()
    cross_index = ["model_A", "model_B", "group", "depth_slot", "layer_A", "layer_B", "split_seed", "optimization_seed", "latent_dim"]
    if "comparison_type" in cross.columns:
        cross_index.append("comparison_type")
    cross_main = cross.pivot_table(
        index=cross_index,
        columns="metric", values="value", aggfunc="mean",
    ).reset_index() if not cross.empty else pd.DataFrame()
    depth_test = test[pd.to_numeric(test.depth_slot, errors="coerce").notna()] if not test.empty else test
    sensitivity = depth_test.groupby(["model", "group", "method", "latent_dim"], as_index=False).mean(numeric_only=True) if not depth_test.empty else pd.DataFrame()
    scale = primary.groupby(["model", "group", "method"], as_index=False).mean(numeric_only=True) if not primary.empty else pd.DataFrame()
    inference_columns = [
        "analysis", "model", "group", "method", "metric", "learned_estimate",
        "baseline_estimate", "mean", "ci_low", "ci_high", "n_pairs",
        "n_observations", "n_models", "n_depths", "n_splits", "n_templates",
        "n_random_seeds", "cluster_columns", "interpretation",
        "joint_estimate", "independent_ridge_estimate",
        "learned_operator_estimate", "source_conditioned_additive_estimate",
        "true_relation_estimate", "wrong_order_estimate", "paired_difference",
    ]

    def inference_table(analysis, estimate_names):
        if bootstrap.empty or "analysis" not in bootstrap:
            return pd.DataFrame(columns=inference_columns)
        selected = bootstrap[bootstrap.analysis == analysis].copy()
        selected[estimate_names[0]] = selected.get("learned_estimate")
        selected[estimate_names[1]] = selected.get("baseline_estimate")
        selected["paired_difference"] = selected.get("mean")
        return selected.reindex(columns=inference_columns)

    frames = {
        "structural_main.csv": structural_main,
        "c7_causal_main.csv": c7_main,
        "d4_causal_main.csv": d4_main,
        "cross_model_main.csv": cross_main,
        "latent_dim_sensitivity.csv": sensitivity,
        "model_scale_summary.csv": scale,
        "joint_vs_ridge_inference.csv": inference_table(
            "joint_minus_independent_ridge", ("joint_estimate", "independent_ridge_estimate"),
        ),
        "causal_additive_baseline_summary.csv": inference_table(
            "causal_learned_minus_source_conditioned_additive",
            ("learned_operator_estimate", "source_conditioned_additive_estimate"),
        ),
        "d4_order_control_summary.csv": inference_table(
            "d4_true_minus_wrong_order", ("true_relation_estimate", "wrong_order_estimate"),
        ),
    }
    table_sources = {
        "structural_main.csv": "structural/template_metrics.csv",
        "c7_causal_main.csv": "causal/intervention_metrics.csv",
        "d4_causal_main.csv": "causal/intervention_metrics.csv",
        "cross_model_main.csv": "intertwiner/cross_model_metrics.csv",
        "latent_dim_sensitivity.csv": "structural/template_metrics.csv",
        "model_scale_summary.csv": "structural/template_metrics.csv",
        "joint_vs_ridge_inference.csv": "bootstrap/confidence_intervals.csv",
        "causal_additive_baseline_summary.csv": "bootstrap/confidence_intervals.csv",
        "d4_order_control_summary.csv": "bootstrap/confidence_intervals.csv",
    }
    for name in STABLE_TABLES:
        frame = frames[name].copy()
        frame["source_csv"] = table_sources[name]
        frames[name] = frame
        write_frame(frame, tables / name)
    if stage_callback is not None:
        stage_callback("table_aggregation", "COMPLETE", table_started)

    figure_started = time.perf_counter()
    if stage_callback is not None:
        stage_callback("figure_generation", "STARTED", figure_started)
    empty = pd.DataFrame({"x": [], "y": []})
    c7_transform = c7_main[
        (c7_main.evaluation_role == "transformation")
        & (c7_main.method == "learned_operator")
    ] if not c7_main.empty else c7_main
    c7_cycle = c7_main[
        (c7_main.evaluation_role == "cycle_recovery")
        & (c7_main.method == "learned_operator")
    ] if not c7_main.empty else c7_main
    figure_specs = {
        "structural_by_layer": (structural_main.groupby("layer", as_index=False).E_eq.mean() if not structural_main.empty else empty, "layer", "E_eq"),
        "structural_by_model": (scale.assign(model_index=np.arange(len(scale))) if not scale.empty else empty, "model_index" if not scale.empty else "x", "E_eq" if not scale.empty else "y"),
        "c7_causal_by_layer": (c7_transform.groupby("layer", as_index=False).target_logit_change.mean() if not c7_transform.empty else empty, "layer" if not c7_transform.empty else "x", "target_logit_change" if not c7_transform.empty else "y"),
        "c7_causal_by_step": (c7_transform.assign(step=c7_transform.intervention.str.extract(r"(\d+)")[0].astype(float)).groupby("step", as_index=False).target_logit_change.mean() if not c7_transform.empty else empty, "step" if not c7_transform.empty else "x", "target_logit_change" if not c7_transform.empty else "y"),
        "c7_cycle_recovery": (c7_cycle.groupby("layer", as_index=False).cycle_hidden_relative_error.mean() if not c7_cycle.empty else empty, "layer" if not c7_cycle.empty else "x", "cycle_hidden_relative_error" if not c7_cycle.empty else "y"),
        "d4_relation_intervention": (d4_main.assign(intervention_index=np.arange(len(d4_main))) if not d4_main.empty else empty, "intervention_index" if not d4_main.empty else "x", "relation_final_logit_distance" if not d4_main.empty else "y"),
        "irrep_spectra": (irrep.assign(row_index=np.arange(len(irrep))) if not irrep.empty else empty, "row_index" if not irrep.empty else "x", "generator_root_angle_rmse" if "generator_root_angle_rmse" in irrep else "row_index" if not irrep.empty else "y"),
        "intertwiner_vs_cka": (cross_main if not cross_main.empty else empty, "linear_cka" if not cross_main.empty else "x", "fitted_intertwiner_commutation_error" if not cross_main.empty else "y"),
        "scale_analysis": (scale.assign(model_index=np.arange(len(scale))) if not scale.empty else empty, "model_index" if not scale.empty else "x", "E_rel_data" if not scale.empty else "y"),
    }
    for stem in STABLE_FIGURES:
        frame, x, y = figure_specs[stem]
        write_figure_pair(frame, x, y, figures, stem, stem.replace("_", " ").title())
    if stage_callback is not None:
        stage_callback("figure_generation", "COMPLETE", figure_started)
    return frames


def _depth_point_rule(frame, group, metric, learned, baseline, *, direction):
    selected = frame[
        (frame.group == group)
        & frame.method.isin([learned, baseline])
        & pd.to_numeric(frame.depth_slot, errors="coerce").notna()
    ].copy()
    try:
        paired = paired_difference_frame(selected, metric, learned, baseline)
    except ValueError:
        return {"status": "NOT_EVALUABLE", "successful_depths": 0, "evaluated_depths": 0, "required_depths": 3}
    depth_means = paired.groupby("depth_slot").learned_minus_baseline.mean()
    successes = int((depth_means < 0).sum() if direction == "negative" else (depth_means > 0).sum())
    status = "NOT_EVALUABLE" if len(depth_means) < 4 else ("PASS" if successes >= 3 else "FAIL")
    return {
        "status": status,
        "successful_depths": successes,
        "evaluated_depths": int(len(depth_means)),
        "required_depths": 3,
        "point_estimates": {str(depth): float(value) for depth, value in depth_means.items()},
    }


def _combine_gate_statuses(statuses):
    if "FAIL" in statuses:
        return "FAIL"
    if "NOT_EVALUABLE" in statuses:
        return "NOT_EVALUABLE"
    return "PASS"


def _cka_association_summary(association):
    if association.empty or not np.isfinite(float(association.iloc[0]["mean"])):
        return {"status": "NOT_EVALUABLE"}
    association_row = association.iloc[0]
    ci_low, ci_high = float(association_row["ci_low"]), float(association_row["ci_high"])
    if ci_low <= 0 <= ci_high:
        status = "NO_RELIABLE_ASSOCIATION"
        interpretation = "No reliable association between CKA and algebraic similarity was detected."
    else:
        status = "POSITIVE_ASSOCIATION" if float(association_row["mean"]) > 0 else "NEGATIVE_ASSOCIATION"
        interpretation = "The interval excludes zero; this is an association estimate, not an independence or causal claim."
    return {
        "status": status,
        "estimate": float(association_row["mean"]),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "interpretation": interpretation,
    }


def _gate_summary(structural, causal, cross, bootstrap, ledger, required_keys, tier):
    execution_complete = all(ledger.should_skip(key) for key in required_keys)

    def ci_rule(analysis, metric, groups, *, method=None, direction="negative"):
        selected = bootstrap[(bootstrap.analysis == analysis) & (bootstrap.metric == metric) & (bootstrap.model == "ALL")]
        if method is not None and "method" in selected:
            selected = selected[selected.method == method]
        selected = selected[selected.group.isin(groups)]
        if set(selected.group) != set(groups):
            return "NOT_EVALUABLE"
        bound = selected.ci_high if direction == "negative" else selected.ci_low
        passed = bool((bound < 0).all()) if direction == "negative" else bool((bound > 0).all())
        return "PASS" if passed else "FAIL"

    structural_rules = [
        ci_rule("learned_minus_random_orthogonal", "E_eq", ["C7", "D4"]),
        ci_rule("learned_minus_random_orthogonal", "E_rel_data", ["C7", "D4"]),
        ci_rule("learned_minus_identity", "E_eq", ["C7", "D4"]),
    ]
    structural_test = structural[
        (structural.data_split == "test")
        & (structural.latent_dim == 16)
    ] if not structural.empty else structural
    structural_depth = {
        f"{group}_{metric}": _depth_point_rule(
            structural_test, group, metric, "group_consistent_joint", "random_orthogonal", direction="negative"
        )
        for group in ("C7", "D4") for metric in ("E_eq", "E_rel_data")
    }
    structural_statuses = structural_rules + [rule["status"] for rule in structural_depth.values()]
    structural_gate = _combine_gate_statuses(structural_statuses)
    causal_rules = [
        ci_rule("causal_learned_minus_random_orthogonal", "target_logit_change", ["C7"], method="transformation", direction="positive"),
        ci_rule("causal_learned_minus_identity", "target_logit_change", ["C7"], method="transformation", direction="positive"),
    ]
    causal_transformations = causal[(causal.group == "C7") & (causal.evaluation_role == "transformation")] if not causal.empty else causal
    causal_depth = {
        baseline: _depth_point_rule(
            causal_transformations, "C7", "target_logit_change", "learned_operator", baseline, direction="positive"
        )
        for baseline in ("random_orthogonal", "identity")
    }
    causal_statuses = causal_rules + [rule["status"] for rule in causal_depth.values()]
    causal_gate = _combine_gate_statuses(causal_statuses)
    d4_random_gate = ci_rule(
        "causal_learned_minus_random_orthogonal", "relation_final_logit_distance", ["D4"],
        method="relation_consistency", direction="negative",
    )
    d4_order_gate = ci_rule(
        "d4_true_minus_wrong_order", "relation_final_logit_distance", ["D4"],
        method="learned_operator", direction="negative",
    )
    d4_gate = _combine_gate_statuses([d4_random_gate, d4_order_gate])
    cross_rules = [
        ci_rule("cross_model_contrast", metric, ["C7", "D4"], method=stratum, direction="negative")
        for metric in ("fitted_minus_procrustes_commutation", "fitted_minus_shuffled_commutation")
        for stratum in ("within_family", "cross_family")
    ]
    cross_gate = _combine_gate_statuses(cross_rules)
    association = bootstrap[
        (bootstrap.analysis == "cross_model_association")
        & (bootstrap.metric == "cka_vs_negative_fitted_commutation")
    ]
    association_summary = _cka_association_summary(association)
    if not execution_complete:
        overall = "NO_GO"
    elif tier == "core" and all(value == "PASS" for value in (structural_gate, causal_gate, d4_gate, cross_gate)):
        overall = "GO"
    elif tier == "core":
        overall = "CONDITIONAL_GO"
    elif tier == "smoke":
        overall = "CONDITIONAL_GO"
    elif tier == "pre_full":
        statuses = (structural_gate, causal_gate, d4_gate, cross_gate)
        overall = "GO" if all(value == "PASS" for value in statuses) else ("NO_GO" if "FAIL" in statuses else "CONDITIONAL_GO")
    else:
        overall = "GO"
    return {
        "tier": tier,
        "overall": overall,
        "structural_gate": structural_gate,
        "causal_gate": causal_gate,
        "d4_relation_gate": d4_gate,
        "cross_model_gate": cross_gate,
        "execution_completeness_gate": "PASS" if execution_complete else "FAIL",
        "required_unit_count": len(required_keys),
        "failed_unit_count": len(ledger.failed_units()),
        "rules": {
            "structural": structural_rules,
            "structural_depth_points": structural_depth,
            "causal": causal_rules,
            "c7_causal_depth_points": causal_depth,
            "d4_relation_vs_random": d4_random_gate,
            "d4_true_vs_wrong_order": d4_order_gate,
            "cross_model": cross_rules,
            "cross_model_cka_algebra_association": association_summary,
        },
        "note": "Scientific gates use the frozen paired-CI directions; execution completeness is evaluated independently.",
    }


def _full_workload_unit_targets(config) -> dict[str, int]:
    full = config["tiers"]["full"]
    model_keys = list(full["model_keys"])
    model_count = len(model_keys)
    group_count = len(config["groups"])
    depth_count = len(full["depth_slots"])
    split_count = len(full["split_seeds"])
    primary_cells_per_model_group = (
        (depth_count + int(bool(full.get("include_terminal_structural"))))
        * split_count
        * len(full["primary_structural_latent_dimensions"])
    )
    sensitivity_cells_per_model_group = (
        len(full["sensitivity_depth_slots"])
        * len(full["sensitivity_split_seeds"])
        * len(full["sensitivity_latent_dimensions"])
    )
    structural_cells = model_count * group_count * (
        primary_cells_per_model_group + sensitivity_cells_per_model_group
    )
    learned_fit_count = 1
    selected = set(model_keys)
    pair_count = sum(
        key_a in selected and key_b in selected for key_a, key_b in config["cross_model_pairs"]
    )
    return {
        "model_load": model_count,
        "activation_extraction": model_count * group_count * (1 + split_count),
        "structural_base": structural_cells,
        "structural_joint": structural_cells * learned_fit_count,
        "causal": model_count * group_count * depth_count * split_count * learned_fit_count,
        "cross_model": pair_count * group_count * depth_count * split_count * len(full["cross_model_latent_dimensions"]) * learned_fit_count,
    }


def _estimate_full_cost(profile: pd.DataFrame, targets: dict[str, int], workload_multipliers=None) -> dict:
    required = {"experiment_module", "wall_clock_seconds"}
    missing_columns = required - set(profile.columns)
    if missing_columns:
        raise ValueError(f"runtime profile missing columns: {sorted(missing_columns)}")
    multipliers = workload_multipliers or {}
    rows, projected_total = [], 0.0
    for module, full_units in targets.items():
        calibration_multiplier = float(multipliers.get(module, 1.0))
        values = pd.to_numeric(
            profile.loc[profile.experiment_module == module, "wall_clock_seconds"], errors="coerce"
        ).to_numpy(dtype=float)
        if not len(values):
            rows.append({"module": module, "status": "NOT_MEASURED", "full_units": int(full_units)})
            continue
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"runtime profile contains invalid timing for {module}")
        observed_units = int(len(values))
        observed_total = float(values.sum())
        observed_mean = float(values.mean())
        ratio = float(full_units) / observed_units
        projected = observed_total * ratio * calibration_multiplier
        if not np.isfinite(projected):
            raise ValueError(f"non-finite FULL projection for {module}")
        projected_total += projected
        rows.append({
            "module": module,
            "status": "MEASURED",
            "observed_units": observed_units,
            "observed_total_seconds": observed_total,
            "observed_mean_seconds": observed_mean,
            "full_units": int(full_units),
            "workload_ratio": ratio,
            "calibration_multiplier": calibration_multiplier,
            "projected_seconds": projected,
        })
    measured = [row for row in rows if row["status"] == "MEASURED"]
    if not measured or not np.isfinite(projected_total):
        raise ValueError("runtime profile contains no finite timings usable for FULL projection")
    return {
        "modules": rows,
        "projected_seconds": float(projected_total),
        "projected_hours": float(projected_total / 3600.0),
        "coverage": "COMPLETE" if len(measured) == len(rows) else "PARTIAL",
        "missing_modules": [row["module"] for row in rows if row["status"] != "MEASURED"],
    }


def _render_cost_projection(projection: dict, source_tier: str) -> str:
    lines = [
        "# FULL Cost Projection",
        "",
        f"Source tier: {source_tier}",
        f"Telemetry coverage: {projection['coverage']}",
        "",
        "Estimator: for each module, observed_total_seconds * (FULL_units / observed_units) * calibration_multiplier; module projections are then added.",
        "",
        "| Module | Observed units | Mean seconds/unit | FULL units | Workload ratio | Calibration multiplier | Projected seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in projection["modules"]:
        if row["status"] == "MEASURED":
            lines.append(
                f"| {row['module']} | {row['observed_units']} | {row['observed_mean_seconds']:.6f} "
                f"| {row['full_units']} | {row['workload_ratio']:.3f} | {row['calibration_multiplier']:.3f} | {row['projected_seconds']:.2f} |"
            )
        else:
            lines.append(f"| {row['module']} | NOT MEASURED | NOT MEASURED | {row['full_units']} | NOT MEASURED | NOT MEASURED | NOT MEASURED |")
    lines.extend([
        "",
        f"Projected FULL serial-equivalent wall time from measured modules: {projection['projected_seconds']:.2f} seconds ({projection['projected_hours']:.2f} hours).",
    ])
    if projection["missing_modules"]:
        lines.append(f"Missing module telemetry: {', '.join(projection['missing_modules'])}. The total is a partial estimate.")
    if source_tier == "smoke":
        lines.append("SMOKE-derived FULL estimates are preliminary. PRE-FULL telemetry is the authoritative calibration stage for the revised design.")
    elif source_tier == "core":
        lines.append("CORE telemetry is historical calibration data. PRE-FULL is authoritative because it measures deterministic fitting and the revised control workload.")
    elif source_tier == "pre_full":
        lines.append("This PRE-FULL-derived estimate is the authoritative calibration for the revised FULL implementation.")
    lines.extend([
        "This additive estimate assumes serial execution and does not model hardware changes, cache warmness, or nonlinear model-size effects.",
        "This projection is calibration metadata, not authorization to run FULL.",
        "",
    ])
    return "\n".join(lines)


def _full_calibration_multipliers(config) -> dict[str, float]:
    if config["tier"] == "full":
        return {}
    source_random_count = 1 if _historical_seed_replication(config) else len(config["random_baseline_seeds"])
    source_wrong_order = int("rsr_wrong_order" in config["causal"].get("d4_wrong_order_controls", ()))
    source_vector_baselines = sum(
        baseline not in {"identity", "random_orthogonal"}
        for baseline in config["causal"]["baselines"]
    )
    source_c7_forwards = 7 * (1 + source_vector_baselines + source_random_count) * 7
    source_d4_forwards = (8 + source_wrong_order) * (1 + source_vector_baselines + source_random_count) * 4
    full_causal_forwards = 7 * 7 * 7 + 9 * 7 * 4
    source_cross_metrics = 5 + 3 * source_random_count
    return {
        "causal": full_causal_forwards / (source_c7_forwards + source_d4_forwards),
        "cross_model": 20 / source_cross_metrics,
    }


def _write_cost_projection(output_dir, telemetry, config):
    targets = _full_workload_unit_targets(config)
    multipliers = _full_calibration_multipliers(config)
    projection = _estimate_full_cost(telemetry.frame, targets, multipliers)
    text = _render_cost_projection(projection, config["tier"])
    atomic_write_text(output_dir / "FULL_COST_PROJECTION.md", text)
    return projection


def _scientific_unit_targets(config) -> dict[str, int]:
    model_count = len(config["selected_models"])
    group_count = len(config["groups"])
    split_count = len(config["split_seeds"])
    learned_fit_count = len(_learned_fit_seeds(config))
    depth_count = len(config["depth_slots"])
    structural_base = sum(
        1
        for model in config["selected_models"]
        for _group in config["groups"]
        for split_seed in config["split_seeds"]
        for latent_dim in config["structural_latent_dimensions"]
        for layer in model["layers"]
        if _include_structural_cell(config, model, layer, latent_dim, split_seed)
    )
    return {
        "model_load": model_count,
        "activation_extraction": model_count * group_count,
        "causal_activation_extraction": model_count * group_count * split_count,
        "structural_base": structural_base,
        "structural_joint": structural_base * learned_fit_count,
        "causal": model_count * group_count * depth_count * split_count * learned_fit_count,
        "cross_model": len(_cross_work_pairs(config)) * group_count * depth_count * split_count * len(_cross_latent_dimensions(config)) * learned_fit_count,
    }


def _bootstrap_analysis_targets(config) -> int:
    model_summaries = len(config["selected_models"]) + 1
    group_count = len(config["groups"])
    structural = model_summaries * group_count * 18

    baseline_count = 2 + int("source_conditioned_additive" in config["causal"]["baselines"])
    has_wrong_order = "rsr_wrong_order" in config["causal"].get("d4_wrong_order_controls", ())
    causal_roles = 2 + 4 + int(has_wrong_order)
    causal_continuous = model_summaries * causal_roles * 4 * baseline_count
    causal_relation_distance = model_summaries * (1 + int(has_wrong_order)) * baseline_count
    d4_order = model_summaries * 5 * int(has_wrong_order)

    pairs = _cross_work_pairs(config)
    strata = {
        "within_family" if key_a.split("_", 1)[0] == key_b.split("_", 1)[0] else "cross_family"
        for key_a, key_b, _comparison_type in pairs
    }
    cross_contrasts = group_count * len(strata) * 2
    cross_association = int(bool(pairs))
    return int(
        structural + causal_continuous + causal_relation_distance
        + d4_order + cross_contrasts + cross_association
    )


def _aggregate_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "structural": output_dir / "structural" / "template_metrics.csv",
        "irrep": output_dir / "irrep" / "representation_diagnostics.csv",
        "cross": output_dir / "intertwiner" / "cross_model_metrics.csv",
        "causal": output_dir / "causal" / "intervention_metrics.csv",
    }


def _module_keys(ledger: RunLedger, module: str) -> list[str]:
    prefix = f"{module}:"
    return sorted(key for key in ledger.data.get("units", {}) if key.startswith(prefix))


def _verified_module_keys(ledger: RunLedger, module: str, expected_count: int) -> list[str]:
    keys = _module_keys(ledger, module)
    if len(keys) != int(expected_count) or not all(ledger.should_skip(key) for key in keys):
        return []
    return keys


def _can_resume_cross_model(ledger: RunLedger, config: dict) -> bool:
    """Return true when every expensive prerequisite is checkpoint-complete."""
    targets = _scientific_unit_targets(config)
    prerequisites = (
        "model_load", "activation_extraction", "causal_activation_extraction",
        "structural_base", "structural_joint", "causal",
    )
    return all(
        bool(_verified_module_keys(ledger, module, targets[module]))
        for module in prerequisites
    )


def _configured_specs(config: dict) -> dict[str, GroupSpec]:
    specs = _group_specs()
    for group_name, base_spec in list(specs.items()):
        frame = _limit_templates(base_spec.frame, config.get("template_limit"))
        specs[group_name] = GroupSpec(
            base_spec.name, frame, base_spec.elements, base_spec.identity,
            base_spec.compose, base_spec.inverse, base_spec.transform_state,
            base_spec.relations,
        )
    return specs


def _hydrate_structural_checkpoints(config, specs, output_dir, ledger):
    """Restore exact structural artifact indexes and aggregate inputs from checkpoints."""
    artifact_index = {}
    structural_paths, irrep_paths, required_keys = [], [], set()
    for model_meta in config["selected_models"]:
        for group_name in config["groups"]:
            spec = specs[group_name]
            for split_seed in config["split_seeds"]:
                split = _split(spec, config, split_seed)
                for dim in config["structural_latent_dimensions"]:
                    for layer in model_meta["layers"]:
                        if not _include_structural_cell(config, model_meta, layer, dim, split_seed):
                            continue
                        base_fields = _common_fields(
                            model_meta, group_name, layer, dim, split.seed,
                        )
                        base_key = experiment_key("structural_base", base_fields)
                        base_metrics, base_artifact, base_irrep = _checkpoint_paths(
                            output_dir, "structural_base", base_key,
                            ("csv", "npz", "irrep.csv"),
                        )
                        if not ledger.should_skip(base_key):
                            raise RuntimeError(f"cannot hydrate incomplete structural base checkpoint: {base_key}")
                        required_keys.add(base_key)
                        structural_paths.append(base_metrics)
                        irrep_paths.append(base_irrep)
                        for opt_seed in _learned_fit_seeds(config):
                            random_seeds = _random_seeds_for_fit(config, opt_seed)
                            checkpoint_random_seed = (
                                int(random_seeds[0]) if _historical_seed_replication(config) else None
                            )
                            fields = _common_fields(
                                model_meta, group_name, layer, dim, split.seed,
                                opt_seed, checkpoint_random_seed,
                            )
                            checkpoint_fields = _full_checkpoint_fields(
                                fields, config, random_seeds,
                            )
                            joint_key = experiment_key("structural_joint", checkpoint_fields)
                            joint_metrics, joint_artifact, joint_irrep = _checkpoint_paths(
                                output_dir, "structural_joint", joint_key,
                                ("csv", "npz", "irrep.csv"),
                            )
                            if not ledger.should_skip(joint_key):
                                raise RuntimeError(
                                    f"cannot hydrate incomplete structural joint checkpoint: {joint_key}"
                                )
                            # Loading validates every serialized PCA/operator array without refitting.
                            load_artifact(joint_artifact, spec.elements, seed=split.seed)
                            required_keys.add(joint_key)
                            structural_paths.append(joint_metrics)
                            irrep_paths.append(joint_irrep)
                            slot = _depth_slot(model_meta, layer)
                            if slot != "terminal":
                                index_key = _structural_index_key(
                                    model_meta["key"], group_name, slot,
                                    split.seed, dim, opt_seed,
                                )
                                artifact_index[index_key] = joint_artifact
    return artifact_index, structural_paths, irrep_paths, required_keys


def _hydrate_hidden_index(config, specs, cache: ActivationCache) -> dict:
    """Reconstruct structural activation-cache identities without model inference."""
    hidden_index = {}
    for model_meta in config["selected_models"]:
        resolved = model_meta["revision"]
        for group_name in config["groups"]:
            frame = specs[group_name].frame
            split_identity = {
                "kind": "template_pool",
                "template_ids": sorted(map(int, frame.template_id.unique())),
                "split_seeds": config["split_seeds"],
                "split_counts": config["template_split"],
            }
            for layer in model_meta["primary_layers"]:
                metadata = _cache_metadata(
                    config, model_meta, resolved, f"structural_{group_name}",
                    layer, frame, split_identity,
                )
                cache_key = activation_cache_key(**metadata)
                if not cache.path(cache_key).is_file():
                    raise RuntimeError(
                        "cannot hydrate cross-model state: COMPLETE activation extraction "
                        f"has no cache for {(model_meta['key'], group_name, layer)}"
                    )
                hidden_index[(model_meta["key"], group_name, layer)] = (cache_key, metadata)
    return hidden_index


def _checkpoint_csv_paths(output_dir, ledger, module, suffix="csv") -> list[Path]:
    paths = []
    for key in _module_keys(ledger, module):
        if ledger.should_skip(key):
            [path] = _checkpoint_paths(output_dir, module, key, (suffix,))
            paths.append(path)
    return paths


def _aggregate_checkpoint_frames(
    output_dir, ledger, structural_paths, irrep_paths, cross_paths=None,
):
    cross_paths = cross_paths or _checkpoint_csv_paths(output_dir, ledger, "cross_model")
    causal_paths = _checkpoint_csv_paths(output_dir, ledger, "causal")
    groups = {
        "structural": (structural_paths, output_dir / "structural" / "template_metrics.csv", {"layer", "E_eq"}),
        "irrep": (irrep_paths, output_dir / "irrep" / "representation_diagnostics.csv", {"layer", "method"}),
        "cross": (cross_paths, output_dir / "intertwiner" / "cross_model_metrics.csv", {"metric", "value"}),
        "causal": (causal_paths, output_dir / "causal" / "intervention_metrics.csv", {"layer", "method"}),
    }
    frames = {}
    for name, (paths, destination, required_columns) in groups.items():
        frame = _concat_csv(paths, destination)
        missing = required_columns - set(frame.columns)
        if frame.empty or missing:
            raise RuntimeError(
                f"checkpoint aggregation produced invalid {name} output; "
                f"rows={len(frame)} missing_columns={sorted(missing)}"
            )
        frames[name] = frame
    return frames


def _tokenizer_identity(tokenizer, model_meta, resolved):
    return {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", model_meta["id"])),
        "revision": resolved,
        "padding_side": str(tokenizer.padding_side),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def _hydrate_manifest_provenance(config, output_dir, ledger, manifest):
    """Restore model provenance from exact model-load markers on process resume."""
    for model_index, model_meta in enumerate(config["selected_models"]):
        fields = {"model": model_meta["id"], "model_revision": model_meta["revision"]}
        key = experiment_key("model_load", fields)
        [marker] = _checkpoint_paths(output_dir, "model_load", key, ("json",))
        if not ledger.should_skip(key):
            raise RuntimeError(f"cannot hydrate incomplete model provenance checkpoint: {key}")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        resolved = payload.get("resolved_revision")
        if resolved != model_meta["revision"]:
            raise RuntimeError(f"model provenance checkpoint revision mismatch: {model_meta['id']}")
        identity = payload.get("tokenizer_identity")
        if identity is None:
            identity = {
                "class": "NOT_RECORDED_IN_LEGACY_CHECKPOINT",
                "name_or_path": model_meta["id"],
                "revision": resolved,
                "padding_side": config["tokenizer"]["padding_side"],
                "pad_token_id": None,
                "eos_token_id": None,
                "recording_status": "PARTIAL_LEGACY_CHECKPOINT",
            }
        manifest["models"][model_index]["resolved_revision"] = resolved
        manifest["models"][model_index]["tokenizer_identity"] = identity


def _can_resume_postprocessing(output_dir: Path, ledger: RunLedger, config: dict) -> bool:
    required_columns = {
        "structural": {"layer", "E_eq"},
        "irrep": {"layer", "method"},
        "cross": {"metric", "value"},
        "causal": {"layer", "method"},
    }
    for name, path in _aggregate_output_paths(output_dir).items():
        if not path.is_file():
            return False
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            return False
        if frame.empty or not required_columns[name] <= set(frame.columns):
            return False
    units = ledger.data.get("units", {})
    for module, expected_count in _scientific_unit_targets(config).items():
        keys = [key for key in units if key.split(":", 1)[0] == module]
        if len(keys) != expected_count or not all(ledger.should_skip(key) for key in keys):
            return False
    return True


def _stage_event(telemetry, ledger, module, event, started, *, completed=None, total=None):
    fields = {"run_id": ledger.data["run_id"]}
    key = experiment_key("postprocess_stage", {**fields, "stage": module})
    telemetry.record(
        key, module, fields, started, event=event,
        progress_completed=completed, progress_total=total,
    )
    elapsed = max(0.0, time.perf_counter() - started)
    progress = "" if completed is None else f" progress={completed}/{total if total is not None else '?'}"
    print(f"POSTPROCESS {module} {event}: {elapsed:.2f}s{progress}", flush=True)


def _run_postprocessing(output_dir, structural, irrep, cross, causal, config, ledger, telemetry, required_keys):
    bootstrap_started = time.perf_counter()
    _stage_event(telemetry, ledger, "bootstrap_statistics", "STARTED", bootstrap_started, completed=0)
    completed_analyses = 0
    last_progress_write = bootstrap_started

    def bootstrap_progress(completed, total):
        nonlocal completed_analyses, last_progress_write
        if completed == total:
            completed_analyses += 1
        now = time.perf_counter()
        should_report = now - last_progress_write >= 30.0 or (
            completed == total and completed_analyses % 10 == 0
        )
        if should_report:
            _stage_event(
                telemetry, ledger, "bootstrap_statistics", "PROGRESS", bootstrap_started,
                completed=completed_analyses,
            )
            print(f"Bootstrap replicate progress in current analysis: {completed}/{total}", flush=True)
            last_progress_write = now

    bootstrap = _bootstrap_rows(structural, causal, cross, config, progress_callback=bootstrap_progress)
    write_frame(bootstrap, output_dir / "bootstrap" / "confidence_intervals.csv")
    _stage_event(
        telemetry, ledger, "bootstrap_statistics", "COMPLETE", bootstrap_started,
        completed=completed_analyses, total=completed_analyses,
    )

    stage_callback = lambda module, event, started: _stage_event(
        telemetry, ledger, module, event, started
    )
    _generate_stable_artifacts(
        output_dir, structural, irrep, cross, causal, bootstrap,
        stage_callback=stage_callback,
    )
    validate_stable_artifacts(output_dir / "results")

    gate_started = time.perf_counter()
    _stage_event(telemetry, ledger, "go_no_go_evaluation", "STARTED", gate_started)
    gate = _gate_summary(structural, causal, cross, bootstrap, ledger, required_keys, config["tier"])
    atomic_write_json(output_dir / "go_no_go.json", gate)
    _stage_event(telemetry, ledger, "go_no_go_evaluation", "COMPLETE", gate_started)

    projection_started = time.perf_counter()
    _stage_event(telemetry, ledger, "cost_projection", "STARTED", projection_started)
    _write_cost_projection(output_dir, telemetry, config)
    _stage_event(telemetry, ledger, "cost_projection", "COMPLETE", projection_started)
    ledger.assert_complete(required_keys)


def run(config, output_dir, manifest, cache_dir, ledger, telemetry):
    if _can_resume_postprocessing(output_dir, ledger, config):
        print("RESUME: all scientific units verified; entering post-processing-only mode", flush=True)
        aggregation_started = time.perf_counter()
        _stage_event(telemetry, ledger, "checkpoint_aggregation", "STARTED", aggregation_started)
        paths = _aggregate_output_paths(output_dir)
        structural = pd.read_csv(paths["structural"])
        irrep = pd.read_csv(paths["irrep"])
        cross = pd.read_csv(paths["cross"])
        causal = pd.read_csv(paths["causal"])
        _stage_event(telemetry, ledger, "checkpoint_aggregation", "COMPLETE", aggregation_started)
        return _run_postprocessing(
            output_dir, structural, irrep, cross, causal, config, ledger, telemetry,
            set(ledger.data["units"]),
        )

    specs = _configured_specs(config)
    cache = ActivationCache(cache_dir)
    if _can_resume_cross_model(ledger, config):
        print(
            "RESUME: expensive scientific prerequisites verified; hydrating checkpoints for cross-model recovery",
            flush=True,
        )
        _hydrate_manifest_provenance(config, output_dir, ledger, manifest)
        write_manifest(output_dir / "manifest.json", manifest)
        artifact_index, structural_paths, irrep_paths, required_keys = (
            _hydrate_structural_checkpoints(config, specs, output_dir, ledger)
        )
        hidden_index = _hydrate_hidden_index(config, specs, cache)
        for module in (
            "model_load", "activation_extraction", "causal_activation_extraction", "causal",
        ):
            required_keys.update(_module_keys(ledger, module))
        cross_paths = _run_cross_units(
            config, specs, artifact_index, hidden_index, cache,
            output_dir, ledger, telemetry, required_keys,
        )
        ledger.assert_complete(required_keys)
        aggregation_started = time.perf_counter()
        _stage_event(telemetry, ledger, "checkpoint_aggregation", "STARTED", aggregation_started)
        frames = _aggregate_checkpoint_frames(
            output_dir, ledger, structural_paths, irrep_paths, cross_paths,
        )
        _stage_event(telemetry, ledger, "checkpoint_aggregation", "COMPLETE", aggregation_started)
        return _run_postprocessing(
            output_dir, frames["structural"], frames["irrep"], frames["cross"],
            frames["causal"], config, ledger, telemetry, required_keys,
        )

    specs = _group_specs()
    structural_paths, irrep_paths, causal_paths, cross_paths = [], [], [], []
    artifact_index, hidden_index, required_keys = {}, {}, set()
    for model_index, model_meta in enumerate(config["selected_models"]):
        load_started = time.perf_counter(); RuntimeTelemetry.reset_gpu_peak()
        load_fields = {"model": model_meta["id"], "model_revision": model_meta["revision"]}
        load_key = experiment_key("model_load", load_fields)
        required_keys.add(load_key)
        [load_marker] = _checkpoint_paths(output_dir, "model_load", load_key, ("json",))
        ledger.begin(load_key, load_fields)
        try:
            tokenizer, model = load_tokenizer_and_model(
                model_meta["id"], dtype=config["dtype"], device_map=config["device_map"],
                revision=model_meta["revision"], tokenizer_settings=config["tokenizer"],
            )
            if int(model.config.num_hidden_layers) != int(model_meta["blocks"]):
                raise ValueError(f"block-count mismatch for {model_meta['id']}")
            resolved = str(getattr(model.config, "_commit_hash", None) or model_meta["revision"])
            if resolved != model_meta["revision"]:
                raise ValueError(f"resolved model revision mismatch for {model_meta['id']}: {resolved}")
            tokenizer_identity = _tokenizer_identity(tokenizer, model_meta, resolved)
            atomic_write_json(load_marker, {
                **load_fields,
                "resolved_revision": resolved,
                "tokenizer_identity": tokenizer_identity,
            })
            ledger.complete(load_key, load_fields, [load_marker])
        except Exception as error:
            ledger.fail(load_key, load_fields, error)
            telemetry.record(load_key, "model_load", load_fields, load_started)
            continue
        manifest["models"][model_index]["resolved_revision"] = resolved
        manifest["models"][model_index]["tokenizer_identity"] = tokenizer_identity
        write_manifest(output_dir / "manifest.json", manifest)
        telemetry.record(load_key, "model_load", load_fields, load_started)
        causal_base_outputs = {}
        for group_name in config["groups"]:
            base_spec = specs[group_name]
            frame = _limit_templates(base_spec.frame, config.get("template_limit"))
            spec = GroupSpec(base_spec.name, frame, base_spec.elements, base_spec.identity, base_spec.compose, base_spec.inverse, base_spec.transform_state, base_spec.relations)
            specs[group_name] = spec
            structural_identity = {
                "kind": "template_pool", "template_ids": sorted(map(int, frame.template_id.unique())),
                "split_seeds": config["split_seeds"], "split_counts": config["template_split"],
            }
            extraction_fields = {"model": model_meta["id"], "group": group_name}
            extraction_key = experiment_key("activation_extraction", extraction_fields)
            required_keys.add(extraction_key)
            [extraction_marker] = _checkpoint_paths(output_dir, "activation_extraction", extraction_key, ("json",))
            extraction_started = time.perf_counter()
            ledger.begin(extraction_key, extraction_fields)
            try:
                layer_hidden, layer_keys, cache_info, extraction_seconds = _cached_layers(
                    frame, tokenizer, model, model_meta, resolved, f"structural_{group_name}", model_meta["layers"], config, cache, structural_identity,
                )
                atomic_write_json(extraction_marker, {**extraction_fields, "layers": model_meta["layers"]})
                ledger.complete(extraction_key, extraction_fields, [extraction_marker])
            except Exception as error:
                ledger.fail(extraction_key, extraction_fields, error)
                telemetry.record(extraction_key, "activation_extraction", extraction_fields, extraction_started)
                continue
            telemetry.record(
                extraction_key, "activation_extraction", extraction_fields,
                time.perf_counter() - extraction_seconds,
                cache_read=cache_info["bytes_read"], cache_written=cache_info["bytes_written"],
            )
            for layer in model_meta["primary_layers"]:
                hidden_index[(model_meta["key"], group_name, layer)] = layer_keys[layer]
            for split_seed in config["split_seeds"]:
                split = _split(spec, config, split_seed)
                eval_frame = _causal_eval_frame(spec, split)
                eval_identity = {"kind": "test_split", **split.identity}
                causal_extract_fields = {
                    "model": model_meta["id"], "group": group_name, "split_seed": split_seed,
                }
                causal_extract_key = experiment_key("causal_activation_extraction", causal_extract_fields)
                required_keys.add(causal_extract_key)
                [causal_extract_marker] = _checkpoint_paths(
                    output_dir, "causal_activation_extraction", causal_extract_key, ("json",)
                )
                causal_extract_started = time.perf_counter()
                ledger.begin(causal_extract_key, causal_extract_fields)
                try:
                    causal_hidden, _, causal_cache, causal_extract_seconds = _cached_layers(
                        eval_frame, tokenizer, model, model_meta, resolved,
                        f"causal_{group_name}", model_meta["causal_layers"], config, cache, eval_identity,
                    )
                    atomic_write_json(
                        causal_extract_marker,
                        {**causal_extract_fields, "layers": model_meta["causal_layers"]},
                    )
                    ledger.complete(causal_extract_key, causal_extract_fields, [causal_extract_marker])
                except Exception as error:
                    causal_hidden = None
                    ledger.fail(causal_extract_key, causal_extract_fields, error)
                    telemetry.record(
                        causal_extract_key, "activation_extraction", causal_extract_fields, causal_extract_started
                    )
                else:
                    telemetry.record(
                        causal_extract_key,
                        "activation_extraction", causal_extract_fields,
                        time.perf_counter() - causal_extract_seconds,
                        cache_read=causal_cache["bytes_read"], cache_written=causal_cache["bytes_written"],
                    )
                for dim in config["structural_latent_dimensions"]:
                    for layer in model_meta["layers"]:
                        if not _include_structural_cell(config, model_meta, layer, dim, split_seed):
                            continue
                        base, metrics_path, irrep_path, _ = _load_or_run_base(
                            spec, layer_hidden[layer], split, model_meta, layer, dim, config, output_dir, ledger, telemetry, required_keys,
                        )
                        if metrics_path: structural_paths.append(metrics_path)
                        if irrep_path: irrep_paths.append(irrep_path)
                        for opt_seed in _learned_fit_seeds(config):
                            random_seeds = _random_seeds_for_fit(config, opt_seed)
                            learned, joint_metrics, joint_irrep, artifact_path = _load_or_run_joint(
                                spec, base, split, model_meta, layer, dim, opt_seed, random_seeds,
                                config, output_dir, ledger, telemetry, required_keys,
                            )
                            if joint_metrics: structural_paths.append(joint_metrics)
                            if joint_irrep: irrep_paths.append(joint_irrep)
                            slot = _depth_slot(model_meta, layer)
                            if slot != "terminal" and artifact_path:
                                artifact_index[(model_meta["key"], group_name, float(slot), int(split_seed), int(dim), int(opt_seed))] = artifact_path
                            if causal_hidden is not None and layer in model_meta["causal_layers"] and int(dim) == int(config["primary_latent_dimension"]):
                                causal_path = _load_or_run_causal(
                                    spec, layer_hidden[layer], eval_frame, causal_hidden[layer], causal_base_outputs,
                                    model, tokenizer, layer, learned,
                                    split, opt_seed, random_seeds, model_meta, config, output_dir, ledger, telemetry, required_keys,
                                )
                                if causal_path: causal_paths.append(causal_path)
        del model, tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass
    cross_paths = _run_cross_units(config, specs, artifact_index, hidden_index, cache, output_dir, ledger, telemetry, required_keys)
    aggregation_started = time.perf_counter()
    _stage_event(telemetry, ledger, "checkpoint_aggregation", "STARTED", aggregation_started)
    frames = _aggregate_checkpoint_frames(
        output_dir, ledger, structural_paths, irrep_paths, cross_paths,
    )
    _stage_event(telemetry, ledger, "checkpoint_aggregation", "COMPLETE", aggregation_started)
    return _run_postprocessing(
        output_dir, frames["structural"], frames["irrep"], frames["cross"],
        frames["causal"], config, ledger, telemetry, required_keys,
    )


def _validate_calibration(path, expected_tier="pre_full"):
    if not path:
        raise RuntimeError(f"FULL is gated: provide a completed {expected_tier.upper()} calibration manifest")
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("tier") != expected_tier or manifest.get("status") != "SUCCESS":
        raise RuntimeError(f"FULL is gated: calibration manifest is not a successful {expected_tier.upper()} run")
    gate_path = Path(path).parent / "go_no_go.json"
    projection = Path(path).parent / "FULL_COST_PROJECTION.md"
    if not gate_path.is_file() or not projection.is_file():
        raise RuntimeError(f"FULL is gated: {expected_tier.upper()} gate summary or cost projection is missing")
    if json.loads(gate_path.read_text(encoding="utf-8")).get("overall") not in {"GO", "CONDITIONAL_GO"}:
        raise RuntimeError(f"FULL is gated by {expected_tier.upper()} go/no-go status")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Canonical Beyond Shared Geometry experiment")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--core", action="store_true")
    modes.add_argument("--pre-full", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument("--config", default="configs/main_experiment.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", default="run_outputs/main_experiment")
    parser.add_argument("--cache-dir", default="run_outputs/activation_cache")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-dirty-full", action="store_true")
    parser.add_argument("--core-calibration-manifest")
    parser.add_argument("--pre-full-calibration-manifest")
    args = parser.parse_args(argv)
    tier = "smoke" if args.smoke else "core" if args.core else "pre_full" if args.pre_full else "full"
    config = resolve_main_config(args.config, tier)
    if args.validate_only:
        print(json.dumps({"status": "VALID", "tier": tier, "models": [model["id"] for model in config["selected_models"]]}, sort_keys=True))
        return 0
    repo_root = Path(__file__).resolve().parents[3]
    _, dirty = git_state(repo_root)
    if dirty and tier in {"core", "full"}:
        print(f"WARNING: {tier.upper()} requested from a dirty Git working tree", flush=True)
    validate_git_policy(tier, dirty, args.allow_dirty_full)
    if tier == "full":
        _validate_calibration(args.pre_full_calibration_manifest, "pre_full")
    run_id = args.run_id or datetime.now(timezone.utc).strftime(f"{tier}_%Y%m%dT%H%M%SZ")
    output_dir = create_output_tree(args.output_root, run_id)
    ledger = RunLedger(output_dir / "run_ledger.json", run_id, config_hash(config))
    telemetry = RuntimeTelemetry(output_dir / "runtime_profile.csv")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = build_manifest(config, run_id, repo_root)
    write_manifest(manifest_path, manifest)
    try:
        import yaml
        atomic_write_text(output_dir / "config_snapshot.yaml", yaml.safe_dump(config, sort_keys=True))
        atomic_write_json(output_dir / "environment.json", {"platform": platform.platform(), "packages": package_versions()})
        run(config, output_dir, manifest, Path(args.cache_dir), ledger, telemetry)
        manifest["status"] = "SUCCESS"
        manifest["artifacts"] = artifact_checksums(output_dir)
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        raise
    finally:
        write_manifest(manifest_path, manifest)
    print(f"canonical results: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
