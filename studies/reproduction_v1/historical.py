#!/usr/bin/env python3
"""Fresh historical-design replication; never relabel new data as frozen evidence.

The original numerical functions/configurations are reused read-only. This adapter
owns a new SHA-bound portable source policy and fresh source bindings. In
particular, a new FULL run is not made to impersonate main_full_003.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
STAGES = ("legacy_pre_full", "legacy_full", "afs_pre_full", "afs_full", "afs_d", "additive", "retrospective_pilot", "retrospective_full")
DEPENDENCIES = {"legacy_pre_full": (), "legacy_full": ("legacy_pre_full",),
                "afs_pre_full": ("legacy_full",), "afs_full": ("afs_pre_full", "legacy_full"),
                "afs_d": ("legacy_full", "afs_full"), "additive": ("legacy_full",),
                "retrospective_pilot": ("legacy_full",), "retrospective_full": ("legacy_full",)}
ADAPTER_SCHEMA = 1


class ScientificStop(RuntimeError):
    """A documented scientific/authorization gate; independent studies may continue."""


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def safe_member(root, relative):
    root = Path(root).resolve()
    candidate = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or candidate.is_symlink():
        raise ValueError("Unsafe relative artifact: " + str(relative))
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("Artifact escapes run root")
    return candidate


def file_records(root, *, exclude=()):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink not permitted in immutable evidence: " + str(path))
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        if name in exclude or name.endswith(".partial"):
            continue
        result[name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    return result


def verify_records(root, records):
    for relative, expected in records.items():
        path = safe_member(root, relative)
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError("Immutable artifact changed: " + str(path))


def source_snapshot(project):
    project = Path(project).resolve()
    files = sorted((project / "src/bsg").rglob("*.py")) + sorted((project / "src/bsg_next").rglob("*.py"))
    files += [project / "configs/prospective/bf16_revision.json"]
    files += [project / "configs" / name for name in
              ("main_experiment.yaml", "algebra_falsification.yaml", "afs_d.yaml")]
    files += [project / "scripts/run_additive_field_falsification.py"]
    if not files or any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("Historical numerical source package incomplete")
    return {"schema": ADAPTER_SCHEMA, "numerical_source": {
        path.relative_to(project).as_posix(): sha(path) for path in files},
        "adapter_sha256": sha(Path(__file__)),
        "policy": "immutable_portable_SHA256_source_with_actual_git_metadata_retained",
        "historical_results_overwritten": False,
        "numerical_config_changes": False,
        "source_binding_changes": "fresh actual run IDs, manifests and checkpoint hashes"}


def environment():
    from importlib.metadata import PackageNotFoundError, version
    names = ("numpy", "scipy", "pandas", "scikit-learn", "torch", "transformers", "accelerate", "PyYAML")
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def validate_successful_manifest(root, *, tier=None):
    root = Path(root)
    manifest = read(root / "manifest.json")
    if manifest.get("status") != "SUCCESS" or (tier and manifest.get("tier") != tier):
        raise ValueError("Source is not the required successful run: " + str(root))
    if not manifest.get("replication", {}).get("source_receipt_sha256"):
        raise ValueError("Source lacks fresh portable-source replication receipt")
    return manifest


def verify_ledger(root):
    """Reject incomplete units and verify every recorded output, including NPZs."""
    root = Path(root).resolve()
    ledger = read(root / "run_ledger.json")
    units = ledger.get("units", {})
    if not units:
        raise ValueError("Empty execution ledger")
    for key, unit in units.items():
        if unit.get("status") != "COMPLETE":
            raise ValueError("Incomplete ledger unit: " + key)
        for output in unit.get("outputs", []):
            p = Path(output["path"])
            if not p.is_absolute():
                p = root / p
            if p.is_symlink() or not p.resolve().is_relative_to(root):
                raise ValueError("Ledger output outside source run: " + str(p))
            if not p.is_file() or p.stat().st_size != int(output["bytes"]) or sha(p) != output["sha256"]:
                raise ValueError("Ledger output changed: " + str(p))
    return ledger


def source_binding(root):
    root = Path(root).resolve()
    manifest = validate_successful_manifest(root)
    verify_ledger(root)
    return {"run_id": manifest["run_id"], "experiment": manifest["experiment"],
            "tier": manifest["tier"], "status": manifest["status"],
            "git_commit": manifest.get("git_commit"), "git_dirty": manifest.get("git_dirty"),
            "config_hash": manifest["config_hash"],
            "source_receipt_sha256": manifest["replication"]["source_receipt_sha256"],
            "manifest_sha256": sha(root / "manifest.json"),
            "ledger_sha256": sha(root / "run_ledger.json")}


def validate_pre_full_for_full(root):
    """Use the unchanged historical PRE-FULL scientific gate and cost requirement."""
    from bsg.experiments.main_experiment import _validate_calibration
    validate_successful_manifest(root, tier="pre_full")
    verify_ledger(root)
    try:
        _validate_calibration(Path(root) / "manifest.json", "pre_full")
    except RuntimeError as exc:
        raise ScientificStop(str(exc)) from exc
    return {"basis": "unchanged_PRE_FULL_gate", "manifest_sha256": sha(Path(root) / "manifest.json"),
            "go_no_go_sha256": sha(Path(root) / "go_no_go.json"),
            "gate": read(Path(root) / "go_no_go.json")["overall"]}


def validate_afs_pre_full(root, original_config, *, source_receipt_sha256):
    """Audit a completed small AFS matrix; NOT_EVALUABLE is never a scientific pass.

    User requested the complete replay. This receipt records technical readiness
    for its larger frozen AFS matrix, rather than inventing a favorable pilot gate.
    """
    import pandas as pd
    from bsg.experiments.algebra_falsification_config import validate_primary_coverage
    root = Path(root)
    manifest = validate_successful_manifest(root, tier="pre_full")
    if manifest["replication"]["source_receipt_sha256"] != source_receipt_sha256:
        raise ValueError("AFS pilot used a different source receipt")
    if manifest.get("execution_modules") != ["A", "B", "C", "D"]:
        raise ValueError("AFS pilot must include all four original modules")
    verify_ledger(root)
    for filename in ("generator_composition_metrics.csv", "algebra_permutation_null.csv", "correspondence_scramble_metrics.csv"):
        validate_primary_coverage(pd.read_csv(root / "afs" / filename), original_config)
    # Each embedding analysis checkpoint is an actual completed fit, not a marker.
    ledger = read(root / "run_ledger.json")
    embedding = [unit for key, unit in ledger["units"].items() if key.startswith("embedding_analysis:")]
    required = len(original_config["selected_models"]) * len(original_config["groups"]) * len(original_config["split_seeds"])
    if len(embedding) != required:
        raise ValueError("AFS pilot embedding-cell coverage incomplete")
    gate = read(root / "go_no_go.json")
    if gate.get("overall") != "NOT_EVALUABLE":
        raise ValueError("Unexpected AFS PRE-FULL gate semantics")
    return {"authorized_for_afs_full": True, "pre_full_run_id": manifest["run_id"],
            "authorization_basis": "user_requested_complete_replay_and_verified_pre_full_coverage",
            "scientific_pass": False, "scientific_gate": gate,
            "manifest_sha256": sha(root / "manifest.json"),
            "ledger_sha256": sha(root / "run_ledger.json"),
            "source_receipt_sha256": source_receipt_sha256}


class FreshCheckpointSource:
    """Fresh counterpart of the frozen reader, with actual independently bound IDs."""

    def __init__(self, root, expected):
        from bsg.experiments.main_config import CANONICAL_REVISIONS
        self.root = Path(root).resolve()
        self.expected = deepcopy(expected)
        self.manifest = validate_successful_manifest(self.root, tier="full")
        self.ledger = verify_ledger(self.root)
        actual = source_binding(self.root)
        if actual != expected:
            raise ValueError("Fresh FULL binding changed")
        revisions = {m["id"]: m["requested_revision"] for m in self.manifest["models"]}
        if revisions != CANONICAL_REVISIONS:
            raise ValueError("Fresh FULL model panel/revisions changed")
        if self.manifest.get("primary_latent_dimension") != 16 or self.manifest.get("template_split") != {"train_templates": 28, "validation_templates": 6, "test_templates": 6}:
            raise ValueError("Fresh FULL coordinate or template contract changed")
        self._snapshot = self._file_snapshot()

    def _file_snapshot(self):
        return {str(p.relative_to(self.root)): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in self.root.rglob("*") if p.is_file()}

    def assert_unchanged(self):
        if self._file_snapshot() != self._snapshot:
            raise ValueError("Fresh FULL input changed during downstream analysis")

    def load_cell(self, spec, **kwargs):
        from bsg.experiments.algebra_falsification_units import HistoricalCheckpointSource
        # The inherited loader checks unique cell identity, bytes, SHA, shape and PCA fit rows.
        return HistoricalCheckpointSource.load_cell(self, spec, **kwargs)

    def load_artifact(self, prefix, spec, model, depth, layer, split_seed):
        from bsg.experiments.canonical_units import load_artifact
        matches = []
        for key, unit in self.ledger["units"].items():
            f = unit.get("fields", {})
            try:
                cell_depth = round(float(f.get("depth_slot", -1)), 2)
            except (ValueError, TypeError):
                continue
            if (key.startswith(prefix + ":") and unit.get("status") == "COMPLETE" and
                f.get("model_key") == model["key"] and f.get("model") == model["id"] and
                f.get("group") == spec.name and cell_depth == round(depth, 2) and
                int(f.get("layer", -1)) == layer and int(f.get("split_seed", -1)) == split_seed and
                int(f.get("latent_dim", -1)) == 16):
                matches.append(unit)
        if len(matches) != 1:
            raise ValueError("Expected one fresh FULL checkpoint: " + prefix)
        outputs = [v for v in matches[0]["outputs"] if Path(v["path"]).suffix == ".npz"]
        if len(outputs) != 1:
            raise ValueError("Expected one NPZ per FULL structural unit")
        expected = outputs[0]
        p = self.root / "checkpoints" / prefix / Path(expected["path"]).name
        if not p.is_file() or p.stat().st_size != expected["bytes"] or sha(p) != expected["sha256"]:
            raise ValueError("Fresh FULL artifact mismatch")
        artifact = load_artifact(p, spec.elements, seed=split_seed)
        if artifact["z"].shape != (len(spec.frame), 16) or artifact["pca"]._fit_rows != 28 * len(spec.frame.state_id.unique()):
            raise ValueError("Fresh FULL shape/training-row mismatch")
        self.assert_unchanged()
        return artifact, {"path": p.relative_to(self.root).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}


class FreshMetricSources:
    """AFS-D numerical postprocessing uses immutable provenance, never old IDs."""

    def __init__(self, full_root, afs_root, bindings):
        self.full_root, self.afs_root = Path(full_root).resolve(), Path(afs_root).resolve()
        self.bindings = bindings
        if self.full_root == self.afs_root:
            raise ValueError("FULL and AFS inputs must differ")
        for name, root in (("full", self.full_root), ("afs", self.afs_root)):
            if source_binding(root) != bindings[name]:
                raise ValueError("Fresh AFS-D binding changed")
        self.snapshots = {name: self.snapshot(root) for name, root in (("full", self.full_root), ("afs", self.afs_root))}

    @staticmethod
    def snapshot(root):
        return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}

    def assert_unchanged(self):
        for name, root in (("full", self.full_root), ("afs", self.afs_root)):
            if self.snapshot(root) != self.snapshots[name]:
                raise ValueError("Fresh AFS-D input changed")

    def source_provenance(self):
        return deepcopy(self.bindings)


def fresh_activation_source(module, cache_dir, config):
    """Bind exact required cache arrays before rank-matched fits; retain metadata checks."""
    required = module._internal_activation_requirements(config, module.group_specs())
    records = {}
    for row in required:
        p = Path(cache_dir) / (row["cache_key"] + ".npz")
        if not p.is_file():
            raise FileNotFoundError("Fresh FULL raw cache required for AFS-D: " + str(p))
        records[p.name] = {"bytes": p.stat().st_size, "sha256": sha(p)}

    class BoundActivationSource(module.InternalActivationSource):
        def load(self, key, expected_metadata):
            verify_records(self.source_path, {key + ".npz": records[key + ".npz"]})
            return super().load(key, expected_metadata)

        def provenance(self):
            return {**super().provenance(), "required_cache_files": records,
                    "directory_manifest_sha256": identity(records), "fresh_replay": True}

    source = BoundActivationSource(Path(cache_dir), config)
    return source, records


def load_additive(project):
    p = Path(project) / "scripts/run_additive_field_falsification.py"
    spec = importlib.util.spec_from_file_location("bsg_fresh_additive_numerics", p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HistoricalReplay:
    def __init__(self, project, output):
        self.project, self.output = Path(project).resolve(), Path(output).resolve()
        if self.output == self.project or self.output.is_relative_to(self.project):
            raise ValueError("Experiment output must be outside the immutable project source")
        sys.path.insert(0, str(self.project / "src"))
        self.source = source_snapshot(self.project)
        self.contract = {"study_id": "bsg-historical-design-reproduction-v1", "source": self.source,
                         "environment": environment(), "reproduction_not_new_confirmation": True}
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "source_receipt.json"
        if path.exists():
            if read(path) != self.contract:
                raise ValueError("Source/environment changed; use a fresh historical output directory")
        else:
            if any(self.output.iterdir()):
                raise ValueError("Nonempty historical output has no source contract")
            atomic(path, self.contract)
        self.receipt_sha = sha(path)
        self.cache = self.output / "activation_cache"

    def run_dir(self, stage):
        return self.output / "runs" / stage

    def assert_source(self):
        if source_snapshot(self.project) != self.source:
            raise ValueError("Frozen numerical or adapter source changed during replay")

    def replication(self):
        return {"fresh_replication": True, "source_receipt_sha256": self.receipt_sha,
                "portable_source_policy": self.source["policy"],
                "historical_run_impersonation": False, "new_confirmation": False}

    def finish(self, directory, manifest, *, error=None):
        from bsg.experiments.main_outputs import artifact_checksums
        manifest["status"] = "SUCCESS" if error is None else "FAILED"
        if error is not None:
            manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        manifest["artifacts"] = artifact_checksums(directory)
        atomic(directory / "manifest.json", manifest)

    def base_manifest(self, directory, proposed):
        path = directory / "manifest.json"
        if path.exists():
            value = read(path)
            if value.get("config_hash") != proposed["config_hash"] or value.get("replication") != self.replication():
                raise ValueError("Resume manifest identity mismatch")
            return value
        atomic(path, proposed)
        return proposed

    def legacy(self, tier):
        from bsg.experiments import main_experiment as m
        from bsg.experiments.main_config import config_hash, resolve_main_config
        from bsg.experiments.manifest import build_manifest, package_versions
        from bsg.experiments.runtime_state import RunLedger, RuntimeTelemetry
        stage = "legacy_" + tier
        cfg = resolve_main_config(self.project / "configs/main_experiment.yaml", tier)
        prerequisite = validate_pre_full_for_full(self.run_dir("legacy_pre_full")) if tier == "full" else None
        directory = m.create_output_tree(self.output / "runs", stage)
        ledger = RunLedger(directory / "run_ledger.json", stage, config_hash(cfg))
        telemetry = RuntimeTelemetry(directory / "runtime_profile.csv")
        proposed = build_manifest(cfg, stage, self.project)
        proposed["replication"] = self.replication()
        manifest = self.base_manifest(directory, proposed)
        atomic(directory / "config_snapshot.json", cfg)
        atomic(directory / "environment.json", {"platform": platform.platform(), "packages": package_versions()})
        if prerequisite:
            atomic(directory / "pre_full_authorization.json", prerequisite)
        try:
            m.run(cfg, directory, manifest, self.cache, ledger, telemetry)
            self.finish(directory, manifest)
        except BaseException as exc:
            self.finish(directory, manifest, error=exc)
            raise
        return {"run": str(directory), "gate": read(directory / "go_no_go.json"), "historical_thresholds_unchanged": True}

    def afs(self, tier):
        import pandas as pd
        from bsg.experiments import algebra_falsification as m
        from bsg.experiments.algebra_falsification_config import afs_config_hash, resolve_afs_config, validate_primary_coverage
        from bsg.experiments.algebra_falsification_outputs import generator_inference, scramble_inference, evaluate_gates, write_stable_outputs
        from bsg.experiments.runtime_state import RunLedger, RuntimeTelemetry
        # Resolve/validate the original scientific design before deriving new input bindings.
        original = resolve_afs_config(self.project / "configs/algebra_falsification.yaml", tier)
        cfg = deepcopy(original)
        binding = source_binding(self.run_dir("legacy_full"))
        cfg["historical_source"] = binding
        cfg["replication_adapter"] = self.replication()
        authorization = None
        if tier == "full":
            pre = resolve_afs_config(self.project / "configs/algebra_falsification.yaml", "pre_full")
            authorization = validate_afs_pre_full(self.run_dir("afs_pre_full"), pre, source_receipt_sha256=self.receipt_sha)
        stage = "afs_" + tier
        directory = m.create_output_tree(self.output / "runs", stage)
        ledger = RunLedger(directory / "run_ledger.json", stage, afs_config_hash(cfg))
        telemetry = RuntimeTelemetry(directory / "runtime_profile.csv")
        proposed = m._build_manifest(cfg, stage, self.project, include_embedding=True)
        proposed["replication"] = self.replication()
        manifest = self.base_manifest(directory, proposed)
        atomic(directory / "original_numerical_config.json", original)
        atomic(directory / "derived_fresh_binding_config.json", cfg)
        if authorization:
            atomic(directory / "pre_full_authorization.json", authorization)
        source = FreshCheckpointSource(self.run_dir("legacy_full"), binding)
        try:
            specs = m.group_specs()
            m._run_reused_modules(cfg, source, specs, directory, ledger, telemetry)
            m._run_embedding_modules(cfg, specs, directory, ledger, telemetry, manifest)
            generator = m._collect(ledger, "generator_composition")
            permutation_raw = m._collect(ledger, "algebra_permutation_null")
            scramble = m._collect(ledger, "correspondence_scramble")
            embedding = m._collect(ledger, "embedding_analysis")
            for table in (generator, permutation_raw, scramble):
                validate_primary_coverage(table, cfg)
            validate_primary_coverage(embedding, cfg, embedding=True)
            permutation = m._permutation_summaries(permutation_raw)
            gci = generator_inference(generator, n_boot=int(cfg["bootstrap_samples"]), seed=int(cfg["bootstrap_seed"]))
            sci = scramble_inference(scramble, n_boot=int(cfg["bootstrap_samples"]), seed=int(cfg["bootstrap_seed"]))
            gate = evaluate_gates(generator, gci, permutation, sci, cfg)
            gate["execution_modules"] = ["A", "B", "C", "D"]
            write_stable_outputs(directory, generator, permutation_raw, permutation, scramble, embedding, gci, sci)
            atomic(directory / "go_no_go.json", gate)
            source.assert_unchanged()
            self.finish(directory, manifest)
        except BaseException as exc:
            self.finish(directory, manifest, error=exc)
            raise
        return {"run": str(directory), "scientific_gate": gate, "falsification_failure_is_retained": True}

    def afs_d(self):
        from bsg.experiments import afs_d as m
        from bsg.experiments.runtime_state import RunLedger, RuntimeTelemetry
        original = m.resolve_config(self.project / "configs/afs_d.yaml")
        cfg = deepcopy(original)
        bindings = {"full": source_binding(self.run_dir("legacy_full")), "afs": source_binding(self.run_dir("afs_full"))}
        cfg["immutable_sources"] = bindings
        cfg["rank_matched_internal_inputs"] = {k: v for k, v in cfg["rank_matched_internal_inputs"].items()
                                              if k not in {"archive", "archive_sha256", "cache_subdir"}}
        cfg["rank_matched_internal_inputs"]["source"] = bindings["full"]["run_id"] + ":activation_cache"
        cfg["replication_adapter"] = self.replication()
        source, caches = fresh_activation_source(m, self.cache, cfg)
        cfg["rank_matched_internal_inputs"]["directory_manifest_sha256"] = identity(caches)
        metric_sources = FreshMetricSources(self.run_dir("legacy_full"), self.run_dir("afs_full"), bindings)
        directory = m.create_output_tree(self.output / "runs", "afs_d")
        ledger = RunLedger(directory / "run_ledger.json", "afs_d", m.config_hash(cfg))
        telemetry = RuntimeTelemetry(directory / "runtime_profile.csv")
        proposed = m._build_manifest(cfg, "afs_d", self.project, metric_sources)
        proposed.update(replication=self.replication(), final_scientific_run=False,
                        scientific_design_changed=False, design_repair="replication_of_frozen_rank_aware_design_with_fresh_inputs")
        manifest = self.base_manifest(directory, proposed)
        atomic(directory / "original_numerical_config.json", original)
        atomic(directory / "derived_fresh_binding_config.json", cfg)
        atomic(directory / "fresh_cache_manifest.json", caches)
        try:
            m.run(cfg, metric_sources, source, directory, ledger, telemetry, manifest)
            self.finish(directory, manifest)
        except BaseException as exc:
            self.finish(directory, manifest, error=exc)
            raise
        finally:
            source.close()
        return {"run": str(directory), "rank_matched_control_complete": True,
                "favorable_result_gate": False, "actual_input_run_ids": {k: v["run_id"] for k, v in bindings.items()}}

    def additive(self):
        import pandas as pd
        from bsg.experiments.runtime_state import RunLedger, RuntimeTelemetry, experiment_key, atomic_write_frame
        m = load_additive(self.project)
        binding = source_binding(self.run_dir("legacy_full"))
        source = FreshCheckpointSource(self.run_dir("legacy_full"), binding)
        directory = self.run_dir("additive")
        for name in ("raw", "tables", "checkpoints", "provenance"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        cfg = {"models": m.MODELS, "splits": m.SPLIT_SEEDS, "depths": m.DEPTH_SLOTS,
               "bootstrap_reps": m.BOOTSTRAP_REPS, "bootstrap_seed": m.BOOTSTRAP_SEED,
               "fresh_source": binding, "replication": self.replication()}
        ledger = RunLedger(directory / "run_ledger.json", "additive", identity(cfg))
        telemetry = RuntimeTelemetry(directory / "runtime_profile.csv")
        from bsg.experiments.manifest import git_state
        commit, dirty = git_state(self.project)
        manifest = self.base_manifest(directory, {"run_id": "additive", "experiment": "additive_field_fresh_replication",
            "tier": "full", "config_hash": identity(cfg), "git_commit": commit, "git_dirty": dirty,
            "replication": self.replication(), "status": "RUNNING"})
        atomic(directory / "derived_fresh_binding_config.json", cfg)
        try:
            preflight = m.assert_row_action_preflight()
            specs = m.group_specs()
            paths, required = [], set()
            for model in m.MODELS:
                for group_name, spec in specs.items():
                    for depth in m.DEPTH_SLOTS:
                        layer = model["depth_layers"][depth]
                        for split_seed in m.SPLIT_SEEDS:
                            meta = {"run_id": "additive", "model_key": model["key"], "model": model["id"],
                                "model_family": model["family"], "group": group_name, "depth_slot": depth,
                                "layer": layer, "realized_depth": layer / model["blocks"],
                                "split_seed": split_seed, "latent_dim": 16}
                            key = experiment_key("fresh_additive_cell", meta)
                            required.add(key)
                            checkpoint = directory / "checkpoints" / (key.split(":")[1] + ".csv")
                            paths.append(checkpoint)
                            if ledger.should_skip(key):
                                continue
                            started = time.monotonic()
                            ledger.begin(key, meta)
                            try:
                                base, base_ref = source.load_artifact("structural_base", spec, model, depth, layer, split_seed)
                                joint, joint_ref = source.load_artifact("structural_joint", spec, model, depth, layer, split_seed)
                                rows = m.evaluate_cell(spec, m.template_split(spec, split_seed), meta, base, joint)
                                atomic_write_frame(pd.DataFrame(rows), checkpoint)
                                ref = checkpoint.with_suffix(".inputs.json")
                                atomic(ref, {"full_run_id": binding["run_id"], "base": base_ref, "joint": joint_ref})
                                ledger.complete(key, meta, [checkpoint, ref], {"raw_metric_rows": len(rows)})
                            except BaseException as exc:
                                ledger.fail(key, meta, exc)
                                raise
                            finally:
                                telemetry.record(key, "fresh_additive_cell", meta, started)
                            print("HISTORICAL_ADDITIVE_CELL " + str(len(ledger.data["units"])) + "/240", flush=True)
            ledger.assert_complete(required)
            raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
            action = m.aggregate_action_metrics(raw)
            seeds = m.primary_seed_aggregates(action)
            contrasts, ci = m.bootstrap_contrasts(seeds)
            outcome, details = m.classify_outcome(ci)
            for filename, frame in (("raw/raw_additive_metrics.csv", raw),
                    ("tables/matched_operator_comparator_metrics.csv", action),
                    ("tables/per_seed_aggregates.csv", seeds), ("tables/per_seed_contrasts.csv", contrasts),
                    ("tables/bootstrap_contrasts_confidence_intervals.csv", ci),
                    ("tables/model_depth_descriptive_stratification.csv", m.stratified_descriptives(action)),
                    ("tables/d4_order_separation_diagnostics.csv", action[action["diagnostic"] == True])):
                atomic_write_frame(frame, directory / filename)
            outcome_record = {"classification_under_frozen_rules": outcome, "outcome_details": details,
                "primary_cells_completed": len(paths), "primary_cells_required": 240,
                "row_action_preflight": preflight, "new_confirmation": False,
                "historical_input_replaced_by_actual_fresh_binding": binding}
            atomic(directory / "replication_outcome.json", outcome_record)
            source.assert_unchanged()
            self.finish(directory, manifest)
        except BaseException as exc:
            self.finish(directory, manifest, error=exc)
            raise
        return {"run": str(directory), **outcome_record}

    def retrospective(self, mode):
        """Apply unchanged archived-replay numerics to actual new FULL checkpoints."""
        import numpy as np
        import pandas as pd
        from bsg_next.offline import historical_replay, verify_historical_replay, retrospective_rows
        from bsg_next.structural import analyze_cell
        from bsg_next.groups import get_group
        from bsg_next.io import jsonable
        from bsg.experiments.runtime_state import RunLedger, RuntimeTelemetry, experiment_key
        m = load_additive(self.project)
        source = FreshCheckpointSource(self.run_dir("legacy_full"), source_binding(self.run_dir("legacy_full")))
        controls = read(self.project / "configs/prospective/bf16_revision.json")["structural"]
        stage = "retrospective_" + mode
        directory = self.run_dir(stage);directory.mkdir(parents=True, exist_ok=True)
        cfg = {"source": source.expected, "controls": controls, "mode": mode, "replication": self.replication()}
        ledger = RunLedger(directory / "run_ledger.json", stage, identity(cfg))
        telemetry = RuntimeTelemetry(directory / "runtime_profile.csv")
        manifest = self.base_manifest(directory, {"run_id":stage, "experiment":"retrospective_fresh_FULL_replay", "tier":mode,
            "config_hash":identity(cfg), "replication":self.replication(), "status":"RUNNING"})
        atomic(directory / "config_snapshot.json", cfg)
        reference = pd.read_csv(source.root / "structural/template_metrics.csv")
        keys = set();results = []
        for model in m.MODELS:
            for group, spec in m.group_specs().items():
                for depth in m.DEPTH_SLOTS:
                    for seed in m.SPLIT_SEEDS:
                        if mode == "pilot" and not (model["id"] in {"meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-1.5B"} and depth in (.4,.8) and seed in (101,303)):
                            continue
                        fields = {"model":model["id"], "model_key":model["key"], "group":group, "layer":model["depth_layers"][depth], "depth_slot":depth, "latent_dim":16, "split_seed":seed}
                        key = experiment_key("retrospective_cell", fields);keys.add(key)
                        target = directory / "cells" / (key.split(":")[1]+".json")
                        if ledger.should_skip(key):results.append(str(target.relative_to(directory)));continue
                        started=time.perf_counter();ledger.begin(key,fields)
                        try:
                            partition=m.template_split(spec,seed)
                            split={"seed":seed,"train_ids":list(partition.train_ids),"validation_ids":list(partition.validation_ids),"test_ids":list(partition.test_ids)}
                            cell={"fields":fields,"split":split,"members":{}}
                            for name,prefix in (("base","structural_base"),("joint","structural_joint")):
                                _,ref=source.load_artifact(prefix,spec,model,depth,fields["layer"],seed)
                                with np.load(source.root/ref["path"],allow_pickle=False) as archive:cell[name]={k:archive[k] for k in archive.files}
                                cell["members"][name]=ref
                            replay=historical_replay(cell);check=verify_historical_replay(replay,reference)
                            rows=retrospective_rows(fields,split)
                            derived=analyze_cell(cell["base"]["z"],rows,get_group(group),controls)
                            atomic(target,jsonable({"fields":fields,"source_members":cell["members"],"historical_replay":replay,
                                "historical_replay_check":check,"retrospective_controls":derived,
                                "inference_status":"descriptive_replay_of_template_pairs_not_new_independent_families","paper_eligible":False}))
                            ledger.complete(key,fields,[target]);results.append(str(target.relative_to(directory)))
                        except BaseException as exc:ledger.fail(key,fields,exc);raise
                        finally:telemetry.record(key,"retrospective_cell",fields,started)
                        print(f"RETROSPECTIVE {mode} {len(results)}",flush=True)
        expected=16 if mode=="pilot" else 240
        ledger.assert_complete(keys)
        if len(results)!=expected:raise ValueError("Incomplete retrospective panel")
        source.assert_unchanged()
        summary={"mode":mode,"cells":len(results),"expected_cells":expected,"cell_outputs":results,"formal_confirmatory_decisions":False}
        atomic(directory/"summary.json",summary);self.finish(directory,manifest)
        return summary

    def run_stage(self, stage):
        self.assert_source()
        state_path = self.output / "stage_state" / (stage + ".json")
        if state_path.exists():
            previous = read(state_path)
            if previous.get("source_receipt_sha256") != self.receipt_sha:
                raise ValueError("Stage source identity mismatch")
            if previous.get("status") == "complete":
                verify_records(self.run_dir(stage), previous["outputs"])
                print("RESUME_HISTORICAL " + stage, flush=True)
                return previous
        for prerequisite in DEPENDENCIES[stage]:
            p = self.output / "stage_state" / (prerequisite + ".json")
            if not p.exists() or read(p).get("status") != "complete":
                raise ScientificStop("Prerequisite not complete: " + prerequisite)
            verify_records(self.run_dir(prerequisite), read(p)["outputs"])
        started = time.monotonic()
        state = {"stage": stage, "status": "running", "source_receipt_sha256": self.receipt_sha}
        atomic(state_path, state)
        print("START_HISTORICAL " + stage, flush=True)
        try:
            if stage.startswith("legacy_"):
                result = self.legacy(stage.removeprefix("legacy_"))
            elif stage in {"afs_pre_full", "afs_full"}:
                result = self.afs(stage.removeprefix("afs_"))
            elif stage == "afs_d":
                result = self.afs_d()
            elif stage.startswith("retrospective_"):
                result = self.retrospective(stage.removeprefix("retrospective_"))
            else:
                result = self.additive()
            self.assert_source()
            state.update(status="complete", result=result, outputs=file_records(self.run_dir(stage)))
        except BaseException as exc:
            state.update(status="gate_stopped" if isinstance(exc, ScientificStop) else "failed",
                         error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            state["seconds"] = time.monotonic() - started
            atomic(state_path, state)
        print("DONE_HISTORICAL " + stage, flush=True)
        return state


def plan(project):
    return {"study": "bsg-historical-design-reproduction-v1", "source": source_snapshot(project),
            "stages": [{"stage": name, "dependencies": list(DEPENDENCIES[name])} for name in STAGES],
            "models_loaded": False, "frozen_inputs_required": False,
            "fresh_FULL_inputs_required_for_downstream": True,
            "synthetic_F64_calibration": "separate reproduction orchestrator branch",
            "scientific_thresholds": "original unchanged; failed outcomes retained",
            "legacy_source_policy": "portable_SHA256_source_receipt; observed_git_state_not_falsified"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--execute-models", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(args.project_root), indent=2))
        return 0
    if not args.execute_models:
        parser.error("Explicit --execute-models is required; no execution started")
    if args.output is None:
        parser.error("A fresh --output is required")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(42)
        run = HistoricalReplay(args.project_root, args.output)
        with (run.output / "historical.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another historical replay owns this output") from exc
            blocked = []
            for stage in STAGES if args.stage == "all" else (args.stage,):
                try:
                    run.run_stage(stage)
                except ScientificStop as exc:
                    blocked.append(stage)
                    atomic(run.output / "stage_state" / (stage + ".json"), {
                        "stage": stage, "status": "gate_stopped", "source_receipt_sha256": run.receipt_sha,
                        "error": {"type": type(exc).__name__, "message": str(exc)}})
                    print("BLOCKED_HISTORICAL " + stage + ": " + str(exc), flush=True)
            atomic(run.output / "historical_status.json", {
                "status": "completed_with_scientific_blocks" if blocked else "complete",
                "blocked_stages": blocked, "historical_runs_overwritten": False,
                "scientific_pass_claimed": False})
            return 3 if blocked else 0
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
