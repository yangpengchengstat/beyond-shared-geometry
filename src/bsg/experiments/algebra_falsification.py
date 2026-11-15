"""Run the isolated, preregistered Algebra Falsification Suite."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bsg.data.c7 import build_dataset as build_c7, transform_state as c7_transform
from bsg.data.d4 import build_dataset as build_d4, transform_state as d4_transform
from bsg.data.splits import split_templates_three_way
from bsg.experiments.algebra_falsification_config import afs_config_hash, afs_matrix, resolve_afs_config, validate_primary_coverage
from bsg.experiments.algebra_falsification_outputs import (
    evaluate_gates, generator_inference, scramble_inference, write_stable_outputs,
)
from bsg.experiments.algebra_falsification_units import (
    HistoricalCheckpointSource, correspondence_scramble_rows, generator_composition_rows,
    permutation_null_rows, stable_cell_seed, summarize_permutation_null, verify_embedding_architecture,
)
from bsg.experiments.canonical_units import evaluate_operators, fit_base_artifact
from bsg.experiments.fitting import GroupSpec
from bsg.experiments.main_config import CANONICAL_REVISIONS
from bsg.experiments.main_outputs import artifact_checksums
from bsg.experiments.manifest import git_state, package_versions
from bsg.experiments.runtime_state import (
    RunLedger, RuntimeTelemetry, atomic_write_frame, atomic_write_json, atomic_write_text, experiment_key,
)
from bsg.groups import c7, d4
from bsg.metrics.relations import c7_relations, d4_relations
from bsg.representations.extract import (
    extract_embedding_representations, load_tokenizer_and_model,
)


def group_specs() -> dict[str, GroupSpec]:
    return {
        "C7": GroupSpec("C7", build_c7(), c7.ELEMENTS, c7.IDENTITY, c7.compose, c7.inverse, c7_transform, c7_relations()),
        "D4": GroupSpec("D4", build_d4(), d4.ELEMENTS, d4.IDENTITY, d4.compose, d4.inverse, d4_transform, d4_relations()),
    }


def template_split(spec, config, seed):
    counts = config["template_split"]
    return split_templates_three_way(
        spec.frame, seed=int(seed), train_templates=int(counts["train_templates"]),
        validation_templates=int(counts["validation_templates"]), test_templates=int(counts["test_templates"]),
    )


def create_output_tree(root, run_id: str) -> Path:
    output = Path(root) / run_id
    if output.exists() and not (output / "run_ledger.json").is_file():
        raise FileExistsError(f"existing AFS directory is not resumable: {output}")
    for relative in ("afs", "checkpoints", "logs", "results/tables", "results/figures"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    return output


def _checkpoint(output_dir: Path, module: str, key: str, suffix: str = "csv") -> Path:
    digest = key.split(":", 1)[1]
    path = output_dir / "checkpoints" / module / f"{digest}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _metadata(model, group, depth, layer, split_seed) -> dict:
    return {
        "model_key": model["key"], "model": model["id"],
        "model_family": "Llama" if model["id"].startswith("meta-llama/") else "Qwen",
        "group": group, "depth_slot": float(depth), "layer": int(layer),
        "realized_depth": int(layer) / int(model["blocks"]), "latent_dim": 16,
        "split_seed": int(split_seed),
    }


def _run_dataframe_unit(module, fields, path, ledger, telemetry, function):
    key = experiment_key(module, fields)
    if ledger.should_skip(key):
        return key, path
    started = time.perf_counter()
    ledger.begin(key, fields)
    try:
        frame = pd.DataFrame(function())
        if frame.empty:
            raise ValueError(f"AFS {module} produced no rows")
        atomic_write_frame(frame, path)
        ledger.complete(key, fields, [path], {"rows": len(frame)})
    except Exception as error:
        ledger.fail(key, fields, error)
        raise
    finally:
        telemetry.record(key, module, fields, started)
    return key, path


def _attach(rows, metadata):
    return [{**metadata, **row} for row in rows]


def _collect(ledger, module: str) -> pd.DataFrame:
    frames = []
    for key, unit in ledger.data["units"].items():
        if not key.startswith(module + ":") or unit.get("status") != "COMPLETE":
            continue
        for output in unit.get("outputs", []):
            path = Path(output["path"])
            if path.suffix == ".csv" and path.is_file():
                frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError(f"no completed AFS checkpoints for {module}")
    return pd.concat(frames, ignore_index=True)


def _run_reused_modules(config, source, specs, output_dir, ledger, telemetry):
    required = set()
    for model in config["selected_models"]:
        for group in config["groups"]:
            spec = specs[group]
            for depth in config["depth_slots"]:
                layer = model["depth_layers"][round(float(depth), 2)]
                for split_seed in config["split_seeds"]:
                    metadata = _metadata(model, group, depth, layer, split_seed)
                    split = template_split(spec, config, split_seed)
                    artifact = source.load_cell(
                        spec, model_key=model["key"], model=model["id"], depth_slot=depth,
                        layer=layer, split_seed=split_seed, latent_dim=16,
                    )
                    a_fields = {**metadata, "experiment_module": "generator_composition"}
                    a_key = experiment_key("generator_composition", a_fields)
                    a_path = _checkpoint(output_dir, "generator_composition", a_key)
                    key, _ = _run_dataframe_unit(
                        "generator_composition", a_fields, a_path, ledger, telemetry,
                        lambda spec=spec, artifact=artifact, split=split, metadata=metadata: _attach(
                            generator_composition_rows(
                                spec, artifact, split, alpha=float(config["ridge_alpha"]),
                                random_seed=stable_cell_seed(config["random_generator_seed"], *metadata.values()),
                            ), metadata
                        ),
                    )
                    required.add(key)
                    b_fields = {**metadata, "experiment_module": "algebra_permutation_null", "permutation_samples": int(config["permutation_samples"])}
                    b_key = experiment_key("algebra_permutation_null", b_fields)
                    b_path = _checkpoint(output_dir, "algebra_permutation_null", b_key)
                    key, _ = _run_dataframe_unit(
                        "algebra_permutation_null", b_fields, b_path, ledger, telemetry,
                        lambda spec=spec, artifact=artifact, split=split, metadata=metadata: _attach(
                            permutation_null_rows(
                                spec, artifact, split, samples=int(config["permutation_samples"]),
                                seed=stable_cell_seed(config["algebra_permutation_seed"], *metadata.values()),
                            ), metadata
                        ),
                    )
                    required.add(key)
                    c_fields = {**metadata, "experiment_module": "correspondence_scramble", "scramble_seeds": list(map(int, config["scramble_seeds"]))}
                    c_key = experiment_key("correspondence_scramble", c_fields)
                    c_path = _checkpoint(output_dir, "correspondence_scramble", c_key)
                    key, _ = _run_dataframe_unit(
                        "correspondence_scramble", c_fields, c_path, ledger, telemetry,
                        lambda spec=spec, artifact=artifact, split=split, metadata=metadata: _attach(
                            correspondence_scramble_rows(
                                spec, artifact, split, scramble_seeds=config["scramble_seeds"],
                                alpha=float(config["ridge_alpha"]),
                            ), metadata
                        ),
                    )
                    required.add(key)
    source.assert_unchanged()
    ledger.assert_complete(required)


def _embedding_cache_path(output_dir, model, group):
    token = hashlib_token(model["key"], group, model["revision"])
    return output_dir / "checkpoints" / "embedding_extraction" / f"{token}.npz"


def hashlib_token(*parts):
    import hashlib
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def _run_embedding_modules(config, specs, output_dir, ledger, telemetry, manifest):
    required = set()
    for model_index, model_meta in enumerate(config["selected_models"]):
        model_fields = {"model": model_meta["id"], "revision": model_meta["revision"], "experiment_module": "embedding_model_load"}
        load_key = experiment_key("embedding_model_load", model_fields)
        load_marker = _checkpoint(output_dir, "embedding_model_load", load_key, "json")
        extraction_keys = [
            experiment_key("embedding_extraction", {
                "model": model_meta["id"], "group": group, "revision": model_meta["revision"],
                "experiment_module": "embedding_extraction",
            }) for group in config["groups"]
        ]
        needs_model = any(not ledger.should_skip(key) for key in extraction_keys)
        if needs_model and not ledger.should_skip(load_key):
            started = time.perf_counter(); ledger.begin(load_key, model_fields)
            try:
                tokenizer, model = load_tokenizer_and_model(
                    model_meta["id"], dtype=config["dtype"], device_map=config["device_map"],
                    revision=model_meta["revision"], tokenizer_settings=config["tokenizer"],
                )
                verify_embedding_architecture(model, model_meta["architecture"], model_meta["blocks"])
                resolved = str(getattr(model.config, "_commit_hash", None) or model_meta["revision"])
                if resolved != model_meta["revision"]:
                    raise ValueError("resolved AFS embedding revision mismatch")
                atomic_write_json(load_marker, {**model_fields, "resolved_revision": resolved})
                ledger.complete(load_key, model_fields, [load_marker])
            except Exception as error:
                ledger.fail(load_key, model_fields, error); raise
            finally:
                telemetry.record(load_key, "embedding_model_load", model_fields, started)
        elif needs_model:
            tokenizer, model = load_tokenizer_and_model(
                model_meta["id"], dtype=config["dtype"], device_map=config["device_map"],
                revision=model_meta["revision"], tokenizer_settings=config["tokenizer"],
            )
            verify_embedding_architecture(model, model_meta["architecture"], model_meta["blocks"])
        else:
            tokenizer, model = None, None
        required.add(load_key)
        manifest["models"][model_index]["resolved_revision"] = model_meta["revision"]
        for group in config["groups"]:
            spec = specs[group]
            fields = {"model": model_meta["id"], "group": group, "revision": model_meta["revision"], "experiment_module": "embedding_extraction"}
            key = experiment_key("embedding_extraction", fields)
            cache_path = _embedding_cache_path(output_dir, model_meta, group)
            if ledger.should_skip(key):
                with np.load(cache_path, allow_pickle=False) as data:
                    hidden = np.asarray(data["hidden"], dtype=np.float32)
            else:
                started = time.perf_counter(); ledger.begin(key, fields)
                try:
                    if tokenizer is None or model is None:
                        raise RuntimeError("embedding extraction checkpoint missing after model-free resume decision")
                    hidden = extract_embedding_representations(spec.frame, tokenizer, model, batch_size=int(config["batch_size"]))
                    hidden = hidden.numpy() if hasattr(hidden, "numpy") else np.asarray(hidden)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cache_path, hidden=np.asarray(hidden, dtype=np.float32))
                    ledger.complete(key, fields, [cache_path], {"rows": len(hidden), "hidden_dim": hidden.shape[1]})
                except Exception as error:
                    ledger.fail(key, fields, error); raise
                finally:
                    telemetry.record(key, "embedding_extraction", fields, started)
            if hidden.shape[0] != len(spec.frame):
                raise ValueError("AFS embedding extraction row contract mismatch")
            required.add(key)
            for split_seed in config["split_seeds"]:
                split = template_split(spec, config, split_seed)
                metadata = {
                    "model_key": model_meta["key"], "model": model_meta["id"],
                    "model_family": "Llama" if model_meta["id"].startswith("meta-llama/") else "Qwen",
                    "group": group, "depth_slot": "embedding", "layer": 0,
                    "realized_depth": 0.0, "latent_dim": 16, "split_seed": int(split_seed),
                }
                fields = {**metadata, "experiment_module": "embedding_analysis"}
                unit_key = experiment_key("embedding_analysis", fields)
                path = _checkpoint(output_dir, "embedding_analysis", unit_key)
                def analyze(spec=spec, hidden=hidden, split=split, metadata=metadata):
                    artifact = fit_base_artifact(spec, hidden, split, 16, config)
                    direct = evaluate_operators(spec, artifact, split, "direct_ridge", optimization_seed=None, random_baseline_seed=None)
                    generator = generator_composition_rows(
                        spec, artifact, split, alpha=float(config["ridge_alpha"]),
                        random_seed=stable_cell_seed(config["random_generator_seed"], metadata["model_key"], group, split.seed, "embedding"),
                    )
                    normalized = _attach(direct, metadata)
                    normalized.extend(_attach([
                        {**row, "E_eq": row["E_gen_eq"], "E_rel_data": np.nan, "E_rel_op": np.nan}
                        for row in generator
                    ], metadata))
                    return normalized
                completed, _ = _run_dataframe_unit("embedding_analysis", fields, path, ledger, telemetry, analyze)
                required.add(completed)
        del model, tokenizer
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass
    ledger.assert_complete(required)


def _permutation_summaries(raw: pd.DataFrame) -> pd.DataFrame:
    cell_columns = ["model", "model_key", "model_family", "group", "depth_slot", "layer", "split_seed", "latent_dim"]
    rows = []
    for keys, cell in raw.groupby(cell_columns, dropna=False):
        metadata = dict(zip(cell_columns, keys))
        rows.extend(_attach(summarize_permutation_null(cell), metadata))
    return pd.DataFrame(rows)


def _execution_modules(include_embedding: bool) -> list[str]:
    return ["A", "B", "C", "D"] if include_embedding else ["A", "B", "C"]


def _build_manifest(config, run_id, repo_root, *, include_embedding: bool = True):
    commit, dirty = git_state(repo_root)
    modules = _execution_modules(include_embedding)
    matrix = afs_matrix(config)
    matrix["executed_primary_units"] = 3 * int(matrix["primary_cells"])
    matrix["executed_embedding_units"] = (
        len(config["selected_models"])
        + len(config["selected_models"]) * len(config["groups"])
        + int(matrix["embedding_cells"])
        if include_embedding else 0
    )
    return {
        "experiment": config["experiment"], "run_id": run_id, "tier": config["tier"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "git_dirty": dirty,
        "config_hash": afs_config_hash(config), "historical_source": config["historical_source"],
        "models": [{"key": m["key"], "id": m["id"], "requested_revision": m["revision"], "resolved_revision": None, "blocks": m["blocks"]} for m in config["selected_models"]],
        "groups": config["groups"], "depth_slots": config["depth_slots"], "split_seeds": config["split_seeds"],
        "primary_latent_dimension": 16, "embedding_convention": config["embedding_convention"],
        "seed_namespaces": {"permutation": config["algebra_permutation_seed"], "scramble": config["scramble_seeds"], "bootstrap": config["bootstrap_seed"], "random_generator": config["random_generator_seed"]},
        "execution_modules": modules, "matrix": matrix,
        "package_versions": package_versions(), "hardware": {"platform": platform.platform()},
        "run_ledger": "run_ledger.json", "status": "RUNNING", "artifacts": {},
    }


def _validate_full_authorization(path):
    if not path:
        raise RuntimeError("AFS_FULL requires an audited PRE_FULL authorization manifest")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("authorized_for_afs_full") is not True or payload.get("pre_full_run_id") is None:
        raise RuntimeError("AFS_FULL authorization manifest is not explicit")


def run(config, source_root, output_dir, ledger, telemetry, manifest, *, include_embedding: bool = True):
    specs = group_specs()
    source = HistoricalCheckpointSource(source_root, config["historical_source"])
    _run_reused_modules(config, source, specs, output_dir, ledger, telemetry)
    if include_embedding:
        _run_embedding_modules(config, specs, output_dir, ledger, telemetry, manifest)
    started = time.perf_counter()
    generator = _collect(ledger, "generator_composition")
    permutation_raw = _collect(ledger, "algebra_permutation_null")
    scramble = _collect(ledger, "correspondence_scramble")
    embedding = _collect(ledger, "embedding_analysis") if include_embedding else None
    validate_primary_coverage(generator, config)
    validate_primary_coverage(permutation_raw, config)
    validate_primary_coverage(scramble, config)
    if include_embedding:
        validate_primary_coverage(embedding, config, embedding=True)
    permutation_summary = _permutation_summaries(permutation_raw)
    generator_bootstrap = generator_inference(generator, n_boot=int(config["bootstrap_samples"]), seed=int(config["bootstrap_seed"]))
    scramble_bootstrap = scramble_inference(scramble, n_boot=int(config["bootstrap_samples"]), seed=int(config["bootstrap_seed"]))
    gate = evaluate_gates(generator, generator_bootstrap, permutation_summary, scramble_bootstrap, config)
    gate["execution_modules"] = _execution_modules(include_embedding)
    if not include_embedding:
        gate["D"] = "NOT_RUN"
    write_stable_outputs(output_dir, generator, permutation_raw, permutation_summary, scramble, embedding, generator_bootstrap, scramble_bootstrap)
    atomic_write_json(output_dir / "go_no_go.json", gate)
    source.assert_unchanged()
    telemetry.record("afs_postprocessing", "afs_postprocessing", {}, started)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Beyond Shared Geometry Algebra Falsification Suite")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--pre-full", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument("--config", default="configs/algebra_falsification.yaml")
    parser.add_argument("--historical-root", required=False)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", default="run_outputs/algebra_falsification")
    parser.add_argument("--full-authorization-manifest")
    parser.add_argument(
        "--modules-abc-only", action="store_true",
        help="run frozen checkpoint-backed modules A-C without embedding diagnostic D",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    tier = "smoke" if args.smoke else "pre_full" if args.pre_full else "full"
    config = resolve_afs_config(args.config, tier)
    if args.validate_only:
        print(json.dumps({"status": "VALID", "tier": tier, "matrix": afs_matrix(config)}, sort_keys=True))
        return 0
    if not args.historical_root:
        raise RuntimeError("AFS requires --historical-root pointing to immutable main_full_003")
    if tier == "full":
        _validate_full_authorization(args.full_authorization_manifest)
    repo_root = Path(__file__).resolve().parents[3]
    run_id = args.run_id or datetime.now(timezone.utc).strftime(f"afs_{tier}_%Y%m%dT%H%M%SZ")
    output_dir = create_output_tree(args.output_root, run_id)
    ledger = RunLedger(output_dir / "run_ledger.json", run_id, afs_config_hash(config))
    telemetry = RuntimeTelemetry(output_dir / "runtime_profile.csv")
    manifest_path = output_dir / "manifest.json"
    include_embedding = not args.modules_abc_only
    execution_modules = _execution_modules(include_embedding)
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else _build_manifest(config, run_id, repo_root, include_embedding=include_embedding)
    )
    if manifest.get("execution_modules") != execution_modules:
        raise ValueError("AFS resume execution-module selection mismatch")
    atomic_write_json(manifest_path, manifest)
    atomic_write_text(output_dir / "config_snapshot.yaml", __import__("yaml").safe_dump(config, sort_keys=True))
    try:
        run(
            config, Path(args.historical_root), output_dir, ledger, telemetry, manifest,
            include_embedding=include_embedding,
        )
        manifest["status"] = "SUCCESS"
        manifest["artifacts"] = artifact_checksums(output_dir)
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        atomic_write_json(manifest_path, manifest)
    print(f"AFS results: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
