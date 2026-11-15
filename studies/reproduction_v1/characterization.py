#!/usr/bin/env python3
"""Regenerate the complete characterization panel from pinned models and prompts.

The numerical dependency is immutable. This is a descriptive same-design replay
of previously inspected rows, not new independent confirmation. No calibration,
parent run, downloaded result archive, or old activation cache is required.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
FROZEN = HERE.parent / "characterization_v1" / "frozen_run"
SCHEMA = 1
STUDY_ID = "icml2027-characterization-from-zero-v1"
EXPECTED_FROZEN_MANIFEST = "5832afe9f3d31b4962ab30ba6c717f151bcb99a14b23ae25f146108a3396f739"
DEPENDENCIES = {"numpy": "2.2.6", "scipy": "1.15.3", "torch": "2.8.0+cu128",
                "transformers": "4.57.1", "tokenizers": "0.22.2", "huggingface-hub": "0.36.2"}
DOMAIN_COUNTS = {"C7": (1960, 1064, 896), "D4": (1120, 608, 512), "S3": (87744, 25728, 40320)}
ROW_KEYS = ("row_id", "family_id", "split", "state_id", "paraphrase_id", "orbit_id", "domain")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def safe_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe artifact path")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes its directory")
    return path


def configure_environment():
    # Must run before importing NumPy/Torch in the command-line process.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def verify_frozen():
    manifest_path = FROZEN / "source_manifest.json"
    if digest(manifest_path) != EXPECTED_FROZEN_MANIFEST:
        raise ValueError("Frozen dependency manifest identity differs")
    manifest = read(manifest_path)
    for name, sha in manifest.items():
        if digest(safe_file(FROZEN, name)) != sha:
            raise ValueError("Frozen numerical dependency changed: " + name)
    return EXPECTED_FROZEN_MANIFEST


def numerical_imports():
    verify_frozen()
    for name, module in list(sys.modules.items()):
        if name == "bsg_next" or name.startswith("bsg_next."):
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to((FROZEN / "src").resolve()):
                raise ValueError("Run characterization in a fresh subprocess; another bsg_next is imported")
    sys.path.insert(0, str(FROZEN / "src"))
    sys.path.insert(0, str(FROZEN))


def config():
    return read(FROZEN / "study_config.json")


def planned_cells(cfg, limit=None):
    cells = [(spec, domain, layer) for spec in cfg["models"] for domain in DOMAIN_COUNTS for layer in spec["layers"]]
    if limit is not None:
        if not 1 <= limit < len(cells):
            raise ValueError("--limit-cells must be 1..71 and always marks the run technical-only")
        cells = cells[:limit]
    return cells


def make_contract(cfg, limit=None):
    for spec in cfg["models"]:
        if spec["revision"] != spec["tokenizer_revision"]:
            raise ValueError("Frozen runtime requires equal model/tokenizer revisions")
    return {"schema_version": SCHEMA, "study_id": STUDY_ID, "frozen_source_manifest_sha256": verify_frozen(),
            "driver_sha256": digest(__file__), "reference_configuration_sha256": digest(FROZEN / "study_config.json"),
            "models": cfg["models"], "runtime": {**cfg["runtime"], "local_files_only": True},
            "seed": 2027101700, "planned_cells": len(planned_cells(cfg, limit)), "limit_cells": limit,
            "technical_test": limit is not None, "paper_eligible": False, "formal_inference_enabled": False,
            "independent_holdout": False, "prior_outcomes_inspected": True,
            "synthetic_calibration_run": False, "native_run": False,
            "scope": "Complete same-design descriptive characterization replay; not old-run continuation or new confirmation.",
            "cache_layout": "one_direct_development_then_heldout_cache; original_batch8_boundaries_preserved",
            "dependency_build_versions_required": DEPENDENCIES,
            "bitwise_reproduction_guaranteed": False}


@contextmanager
def output_lock(output):
    output = Path(output)
    if output.is_symlink() or output.resolve().is_relative_to(RELEASE):
        raise ValueError("Use a fresh output outside the source release")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "worker.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another characterization worker owns this output") from exc
        try:
            yield output.resolve()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def bind_contract(output, contract):
    path = output / "study_contract.json"
    if path.exists():
        if read(path) != contract:
            raise ValueError("Output contract changed; use a fresh namespace")
    else:
        if any(p.name != "worker.lock" for p in output.iterdir()):
            raise ValueError("Nonempty output without the expected study contract")
        atomic(path, contract)
    return identity(contract)


def generate_banks(cfg):
    from bsg_next.data import build_study_records, build_support_records, dataset_identity, select_phase_records
    banks = {}
    for domain, expected in DOMAIN_COUNTS.items():
        rows = build_study_records(domain, seed=2027101700, max_test_families=64,
                                   max_test_orbits=24, include_pilot=True)
        if domain == "S3":
            rows += build_support_records(domain, seed=2027101700, test_orbits=24)
        if dataset_identity(rows) != cfg["dataset_seals"][domain]["dataset_sha256"]:
            raise ValueError("Regenerated prompt population differs: " + domain)
        development = select_phase_records(rows, "development", 64, 16)
        heldout = select_phase_records(rows, "confirmation", 64, 16)
        if domain == "S3":
            heldout = [r for r in heldout if r["split"] in cfg["included_S3_splits"]]
            if dict(Counter(r["split"] for r in heldout)) != cfg["S3_new_row_breakdown"]:
                raise ValueError("S3 characterization allocation changed")
        if (len(rows), len(development), len(heldout)) != expected or len(development) % 8:
            raise ValueError("Population or original batch boundary mismatch")
        merged = development + heldout
        if len({r["row_id"] for r in merged}) != len(merged):
            raise ValueError("Development and held-out rows overlap")
        banks[domain] = {"all": rows, "development": development, "heldout": heldout, "merged": merged}
    return banks


def prepare(output, cfg, contract):
    from bsg_next.data import seal_dataset
    banks = generate_banks(cfg)
    counts = {}
    for domain, bank in banks.items():
        manifest = seal_dataset(bank["all"], output / "datasets" / domain,
                                metadata={"role": "replication_of_previously_inspected_population",
                                          "new_independent_holdout": False, "seed": 2027101700})
        counts[domain] = {"dataset_sha256": manifest["dataset_sha256"],
                          "merged_rows": len(bank["merged"]),
                          "development_split_counts": dict(Counter(r["split"] for r in bank["development"])),
                          "heldout_split_counts": dict(Counter(r["split"] for r in bank["heldout"]))}
    atomic(output / "provenance" / "population_counts.json", counts)
    atomic(output / "provenance" / "reference_study_config.json", cfg)
    atomic(output / "provenance" / "frozen_source_manifest.json", read(FROZEN / "source_manifest.json"))
    return banks


def runtime_environment(require_gpu=False):
    from importlib.metadata import version
    versions = {name: version(name) for name in DEPENDENCIES}
    for name, expected in DEPENDENCIES.items():
        if versions[name] != expected:
            raise ValueError(f"Dependency mismatch {name}: expected {expected}, found {versions[name]}")
    import torch
    environment = {"versions": versions, "python": sys.version, "platform": sys.platform,
                   "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if require_gpu and not torch.cuda.is_available():
        raise ValueError("Model extraction requires a CUDA GPU; no silent CPU fallback")
    if torch.cuda.is_available():
        environment["gpu"] = {"name": torch.cuda.get_device_name(0),
                              "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    return environment


def expected_raw_bytes(cfg, cells, banks):
    pairs = {(spec["id"], domain) for spec, domain, _ in cells}
    return sum(len(banks[domain]["merged"]) * spec["hidden_dimension"] *
               (len(spec["layers"]) + (3 if domain == "S3" else 0)) * 4
               for spec in cfg["models"] for domain in DOMAIN_COUNTS if (spec["id"], domain) in pairs)


def verify_artifacts(directory, marker, binding):
    path = directory / marker
    if not path.exists():
        return False
    receipt = read(path)
    if receipt.get("binding") != binding or not receipt.get("artifacts"):
        raise ValueError("Artifact receipt belongs to another input/source contract")
    for name, sha in receipt["artifacts"].items():
        if digest(safe_file(directory, name)) != sha:
            raise ValueError("Completed artifact changed: " + str(directory / name))
    return True


def transactional_directory(directory, marker, binding, builder):
    """Never trust an individual result file without its complete bound receipt."""
    if directory.is_symlink():
        raise ValueError("Artifact directory cannot be a symlink")
    if verify_artifacts(directory, marker, binding):
        return False
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = directory.parent / ("." + directory.name + ".building-" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        builder(stage)
        artifacts = {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        if not artifacts:
            raise ValueError("Builder produced no artifacts")
        atomic(stage / marker, {"schema_version": SCHEMA, "binding": binding, "artifacts": artifacts})
        verify_artifacts(stage, marker, binding)
        if directory.exists():
            # Preserve an interrupted directory, but never reuse its contents.
            directory.rename(directory.parent / ("." + directory.name + ".incomplete-" + uuid.uuid4().hex))
        stage.rename(directory)
        return True
    except BaseException:
        # Failed partial artifacts remain visibly incomplete; no receipt is reused.
        raise


def export_shared(output, key, rows, potential, cache_sha, contract_id):
    import numpy as np
    shared = output / "analysis" / "shared" / (key + "__S3")
    binding = {"contract_id": contract_id, "cache_manifest_sha256": cache_sha, "role": "S3_unique_input_metadata"}
    def build(stage):
        keys = sorted({(r["orbit_id"], r["state_id"]) for r in rows})
        first = {}
        for i, row in enumerate(rows):
            first.setdefault((row["orbit_id"], row["state_id"]), i)
        positions = {value: i for i, value in enumerate(keys)}
        inverse = np.array([positions[(r["orbit_id"], r["state_id"])] for r in rows], dtype=np.int32)
        first_ids = np.array([first[value] for value in keys], dtype=np.int64)
        np.savez_compressed(stage / "potential_unique.npz", values=np.asarray(potential[first_ids]),
                            inverse=inverse, first_row_indices=first_ids)
        atomic(stage / "rows.json", {"rows": [{k: r[k] for k in ROW_KEYS} for r in rows], "potential_keys": keys,
                                     "scope": "Unique orbit/state input embeddings and exact reconstruction indices"})
        atomic(stage / "compact_export.json", {name: digest(stage / name) for name in ("potential_unique.npz", "rows.json")})
    transactional_directory(shared, "shared_complete.json", binding, build)
    return shared


def analyze_one(output, spec, domain, layer, rows, arrays, manifest, cache_sha, cfg, contract_id, potential=None):
    import numpy as np
    from bsg_next.config import model_key
    from bsg_next.structural import fit_train_pca, analyze_cell
    from bsg_next.groups import get_group
    from fields import compare_cell, input_comparisons
    key = model_key(spec)
    name = f"{key}__{domain}__{layer}"
    target = output / "analysis" / name
    binding = {"contract_id": contract_id, "cache_manifest_sha256": cache_sha, "layer": layer,
               "domain": domain, "numerical_source_manifest_sha256": EXPECTED_FROZEN_MANIFEST}
    shared = export_shared(output, key, rows, potential, cache_sha, contract_id) if domain == "S3" else None
    def build(stage):
        train = np.array([r["split"] == "train" for r in rows])
        center, basis, pca = fit_train_pca(arrays[layer], train, 16)
        z = np.empty((len(rows), 16), dtype=np.float64)
        for start in range(0, len(rows), 512):
            z[start:start + 512] = (np.asarray(arrays[layer][start:start + 512], float) - center) @ basis
        np.savez_compressed(stage / "pca.npz", center=center, basis=basis)
        group = get_group(domain)
        splits = ("test_joint", "test_context", "test_entity") if domain == "S3" else ("test",)
        legacy = []
        for split in splits:
            print("LEGACY_ENDPOINTS", name, split, flush=True)
            value = analyze_cell(z, rows, group, cfg["structural"], potential, split)
            value.update(model=manifest["model"], layer=layer, pca=pca, cache_manifest_sha256=cache_sha)
            legacy.append(value)
        # The vendored serializer handles NumPy scalars without nonfinite JSON.
        from bsg_next.io import jsonable
        atomic(stage / "legacy.json", jsonable(legacy))
        if domain != "S3":
            generator = compare_cell(z, rows, group, cfg["structural"], evaluation_split="test")
            comparisons = input_comparisons(z, rows, group, cfg["structural"], evaluation_split="test")
            import math
            old = legacy[0]["mean_losses"]
            checks = [(generator["mean_losses"]["unseen_composite"]["generator_ridge"], old["generator_unseen_actual_target"]),
                      (comparisons["controls"]["primary_operator"]["mean_loss"], old["primary_operator"]),
                      (comparisons["controls"]["oracle_local"]["mean_loss"], old["local_displacement"])]
            full = next(c for c in comparisons["storage_curves"] if c["track"] == "generator_only" and c["full_training_support"])
            checks += [(full["results"][kind]["mean_loss"], generator["mean_losses"][kind]["generator_input_only_field"])
                       for kind in ("primitive", "unseen_composite")]
            if any(not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10) for a, b in checks):
                raise ValueError("Unchanged legacy/field estimators disagree")
            atomic(stage / "fields.json", jsonable({"generator": generator, "input_comparisons": comparisons,
                   "model_id": spec["id"], "group": group.name, "layer": layer, "technical_test": False,
                   "formal_inference": False, "paper_eligible": False, "same_design_replay": True}))
            atomic(stage / "rows.json", {"rows": [{k: r[k] for k in ROW_KEYS} for r in rows]})
        else:
            atomic(stage / "coordinate_export.json", {"rows_and_potential_archive_relative_directory": str(shared.relative_to(output)),
                   "shared_input_hashes": read(shared / "compact_export.json"),
                   "reconstruct_potential_features": "potential_unique.values[potential_unique.inverse[row_indices]]",
                   "full_hidden_state_or_PCA_refit_from_raw_exported": False})
        np.savez_compressed(stage / "coordinates.npz", z=z)
    print("ANALYZE_OR_VERIFY", name, flush=True)
    fresh = transactional_directory(target, "cell_complete.json", binding, build)
    print("DONE_CELL" if fresh else "RESUME_CELL", name, flush=True)


def summarize(output, cfg, contract):
    legacy, fields, completed = [], [], 0
    for path in sorted((output / "analysis").glob("*/cell_complete.json")):
        receipt = read(path)
        if receipt["binding"]["contract_id"] != identity(contract):
            raise ValueError("Mixed study cell contract")
        verify_artifacts(path.parent, path.name, receipt["binding"])
        legacy.extend(read(path.parent / "legacy.json"))
        if (path.parent / "fields.json").exists():
            fields.append(read(path.parent / "fields.json"))
        completed += 1
    expected = contract["planned_cells"]
    if completed != expected:
        raise ValueError(f"Incomplete panel: {completed}/{expected} cells")
    if not contract["technical_test"] and (len(legacy) != 120 or len(fields) != 48):
        raise ValueError("Expected 120 split records and 48 field cells")
    groups = defaultdict(list)
    for record in legacy:
        groups[record["group"] + "|" + record["test_split"]].append(record)
    import numpy as np
    means = {key: {name: float(np.mean([r["mean_losses"][name] for r in values]))
                   for name in values[0]["mean_losses"]} for key, values in groups.items()}
    summary = {"status": "complete_same_design_descriptive_replay", "study_id": STUDY_ID,
               "completed_cells": completed, "legacy_split_records": len(legacy), "field_cells": len(fields),
               "technical_test": contract["technical_test"], "paper_eligible": False, "formal_inference_enabled": False,
               "old_power_gate_passed": False, "native_run": False, "synthetic_calibration_run": False,
               "independent_holdout": False, "descriptive_legacy_means": means}
    if not contract["technical_test"]:
        from uncertainty import summarize as sensitivity
        from bsg_next.io import jsonable
        result = sensitivity(legacy, fields, expected_conditions=24,
                             replicates=cfg["descriptive_uncertainty"]["replicates"], seed=cfg["descriptive_uncertainty"]["seed"])
        atomic(output / "analysis" / "descriptive_uncertainty.json", jsonable(result))
        summary["descriptive_uncertainty"] = "analysis/descriptive_uncertainty.json; sensitivity only"
    atomic(output / "summary.json", summary)
    return summary


def collect(output):
    archive = output.with_name(output.name + ".results.zip")
    temp = archive.with_name(archive.name + ".partial")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for folder in ("analysis", "provenance"):
            for p in sorted((output / folder).rglob("*")):
                if p.is_file() and not any(part.startswith(".") for part in p.relative_to(output / folder).parts):
                    z.write(p, output.name + "/" + str(p.relative_to(output)))
        for name in ("study_contract.json", "summary.json", "execution_state.json"):
            if (output / name).exists():
                z.write(output / name, output.name + "/" + name)
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:
            raise ValueError("Export CRC failure")
    temp.replace(archive)
    archive.with_name(archive.name + ".sha256").write_text(digest(archive) + "  " + archive.name + "\n")
    print("RESULT_ARCHIVE", archive, flush=True)
    return archive


def run(args, runtime_factory=None):
    numerical_imports()
    cfg = config()
    contract = make_contract(cfg, args.limit_cells)
    with output_lock(args.output) as output:
        contract_id = bind_contract(output, contract)
        if args.collect_existing:
            if not (output / "summary.json").exists():
                raise ValueError("No completed summary to collect")
            summarize(output, cfg, contract)
            collect(output)
            return 0
        banks = prepare(output, cfg, contract)
        if args.prepare_only:
            print(json.dumps({"status": "prepared_no_model_loaded", "planned_cells": contract["planned_cells"],
                              "counts": read(output / "provenance" / "population_counts.json")}, indent=2), flush=True)
            return 0
        if runtime_factory is None:
            atomic(output / "provenance" / "runtime_environment.json", runtime_environment(args.execute_models))
        from bsg_next.config import model_key
        from bsg_next.hf import runtime_from_config
        from bsg_next.extraction import extract_features, read_cache
        import numpy as np
        factory = runtime_factory or runtime_from_config
        cells = planned_cells(cfg, args.limit_cells)
        needed = expected_raw_bytes(cfg, cells, banks)
        cache_index = []
        existing = sum(p.stat().st_size for p in (output / "cache").rglob("*.npy"))
        if args.execute_models and shutil.disk_usage(output).free < max(0, needed - existing) + 15 * 2**30:
            raise ValueError("Insufficient space for fresh raw cache plus 15 GiB reserve; no extraction started")
        atomic(output / "provenance" / "disk_plan.json", {"raw_numeric_bytes": needed,
               "reserve_bytes": 15 * 2**30, "model_weight_cache_additional": True})
        done = threading.Event()
        started = time.monotonic()
        def heartbeat():
            while not done.wait(60):
                complete = len(list((output / "analysis").glob("*/cell_complete.json")))
                print(f"HEARTBEAT elapsed_this_session_seconds={time.monotonic()-started:.0f} completed_cells={complete}/{len(cells)}", flush=True)
        monitor = threading.Thread(target=heartbeat, daemon=True)
        monitor.start()
        atomic(output / "execution_state.json", {"status": "running", "contract_id": contract_id})
        try:
            for spec in cfg["models"]:
                selected = [(domain, layer) for selected_spec, domain, layer in cells if selected_spec["id"] == spec["id"]]
                if not selected:
                    continue
                key = model_key(spec)
                domains = [domain for domain in DOMAIN_COUNTS if any(d == domain for d, _ in selected)]
                if args.execute_models:
                    runtime = factory(spec, contract["runtime"])
                    try:
                        smoke_path = output / "provenance" / (key + "__repeatability.json")
                        if not smoke_path.exists():
                            smoke = {}
                            for domain in ("C7", "S3"):
                                records = banks[domain]["merged"][:8]
                                readout = "final_token" if domain == "S3" else "span_mean"
                                def forward():
                                    chunks = list(runtime.features(records, spec["layers"], readout))
                                    return {layer: np.concatenate([v[layer] for _, v in chunks]) for layer in spec["layers"]}
                                first, second = forward(), forward()
                                if not all(np.isfinite(first[layer]).all() and np.array_equal(first[layer], second[layer]) for layer in spec["layers"]):
                                    raise ValueError("Within-process model repeatability failed: " + key + "/" + domain)
                                smoke[domain] = {"rows": 8, "layers": spec["layers"], "identical_repeated_forwards": True}
                            atomic(smoke_path, {"contract_id": contract_id, "checks": smoke, "scientific_gate": False})
                        elif read(smoke_path).get("contract_id") != contract_id:
                            raise ValueError("Model repeatability receipt contract mismatch")
                        for domain in domains:
                            print("EXTRACT_OR_VERIFY", key, domain, len(banks[domain]["merged"]), flush=True)
                            extract_features(runtime, banks[domain]["merged"], spec["layers"],
                                             "final_token" if domain == "S3" else "span_mean",
                                             output / "cache" / key / domain, potential_features=domain == "S3")
                    finally:
                        runtime.close()
                for domain in domains:
                    cache = output / "cache" / key / domain
                    rows, arrays, manifest = read_cache(cache)
                    if rows != banks[domain]["merged"]:
                        raise ValueError("Cache rows differ from regenerated ordered population")
                    if manifest["model"] != factory(spec, contract["runtime"]).identity():
                        raise ValueError("Cache model/runtime identity differs")
                    cache_sha = digest(cache / "manifest.json")
                    cache_index.append({"path": str(cache), "manifest_sha256": cache_sha})
                    atomic(output / "provenance" / f"{key}__{domain}__merged.json", manifest)
                    potential = np.load(cache / manifest["potential_features"]["path"], mmap_mode="r", allow_pickle=False) if domain == "S3" else None
                    for selected_domain, layer in selected:
                        if selected_domain == domain:
                            analyze_one(output, spec, domain, layer, rows, arrays, manifest, cache_sha, cfg, contract_id, potential)
            atomic(output / "cache" / "caches.json", {"cache_dirs": [v["path"] for v in cache_index],
                   "cache_manifests": cache_index, "contract_id": contract_id, "independent_confirmation": False})
            if not contract["technical_test"]:
                from bsg_next.embedding import run_embedding_floor
                floor_cfg = {**cfg, "runtime": contract["runtime"]}
                if args.execute_models:
                    run_embedding_floor(floor_cfg, output / "embedding_floor", records_by_domain={d: bank["all"] for d, bank in banks.items()})
                elif not (output / "embedding_floor" / "embedding_floor.json").exists():
                    raise ValueError("Full replay embedding floor missing; use --execute-models to finish")
            summary = summarize(output, cfg, contract)
            atomic(output / "execution_state.json", {"status": "complete", "contract_id": contract_id,
                   "elapsed_this_session_seconds": time.monotonic() - started})
            collect(output)
            print(json.dumps(summary, indent=2), flush=True)
            return 0
        except BaseException as exc:
            atomic(output / "execution_state.json", {"status": "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                   "contract_id": contract_id, "error_type": type(exc).__name__, "error": str(exc),
                   "elapsed_this_session_seconds": time.monotonic() - started})
            raise
        finally:
            done.set()
            monitor.join(timeout=1)


def main(argv=None):
    configure_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute-models", action="store_true")
    mode.add_argument("--analyze-existing", action="store_true")
    mode.add_argument("--collect-existing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-cells", type=int, default=None)
    args = parser.parse_args(argv)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt(f"signal {signum}")))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
