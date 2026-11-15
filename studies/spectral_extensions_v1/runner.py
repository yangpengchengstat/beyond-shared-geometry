#!/usr/bin/env python3
"""CPU-only, explicit spectral extensions with immutable per-unit receipts."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import uuid
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "spectral_finite_bank_v1"
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location("_bsg_spectral_input_adapter", BASE / "runner.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sys.path.insert(0, str(HERE))
import numerics

InputError = base.InputError
SCHEMA = 1
STAMP = (2026, 10, 29, 0, 0, 0)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return base.sha(base.canonical(value))


def json_file(path):
    path = Path(path)
    if path.is_symlink():
        raise InputError("Symlink JSON input refused")
    return base._json(path.read_bytes())


def protocol():
    return json_file(HERE / "protocol.json")


def write_json(path, value):
    # Publish immutable receipts atomically. An interruption during serialization
    # must not leave a truncated receipt that prevents a verified resume.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temp.open("x", encoding="utf-8") as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.link(temp, path)  # no replacement of an existing immutable record
    finally:
        if temp.exists():
            temp.unlink()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def _tree_files(root):
    result = {}
    for p in sorted(Path(root).rglob("*")):
        rel = p.relative_to(root)
        if any(x in rel.parts for x in ("__pycache__", ".pytest_cache")) or p.suffix in (".pyc", ".pyo"):
            continue
        if p.is_symlink():
            raise InputError("Symlink in source or prepared input")
        if p.is_file():
            result[rel.as_posix()] = base.file_sha(p)
    return result


def source_lock(_fixture=False):
    if _fixture:
        return {"software_fixture_only": True}
    parent = base.source_lock()
    path = HERE / "source_manifest.json"
    if not path.is_file():
        raise InputError("Missing extension source manifest")
    lock = json_file(path)
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise InputError("Malformed extension source manifest")
    actual = _tree_files(HERE)
    actual.pop("source_manifest.json", None)
    if files != actual:
        raise InputError("Extension source coverage or checksum differs")
    if lock.get("base_source_manifest_sha256") != parent["manifest_sha256"]:
        raise InputError("Frozen base spectral source identity differs")
    dependencies = {"studies/spectral_finite_bank_v1/" + k: v for k, v in parent["files"].items()}
    dependencies["studies/spectral_finite_bank_v1/source_manifest.json"] = parent["manifest_sha256"]
    if lock.get("dependencies") != dependencies or lock.get("protocol_id") != protocol()["protocol_id"]:
        raise InputError("Frozen dependency coverage or protocol identity differs")
    return {"extension_manifest_sha256": base.file_sha(path), "files": files,
            "base_manifest_sha256": parent["manifest_sha256"], "base_files": parent["files"]}


def verify_manifest(root, name, *, exact=True):
    root = Path(root)
    path = root / name
    document = json_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    token = sidecar.read_text().split()[0]
    if not base.HASH_RE.fullmatch(token) or base.file_sha(path) != token:
        raise InputError("Manifest checksum mismatch: " + name)
    files = document.get("files")
    if not isinstance(files, dict):
        raise InputError("Manifest has no file inventory")
    for rel, expected in files.items():
        base.safe_member(rel)
        p = root / rel
        if p.is_symlink() or not p.is_file() or base.file_sha(p) != expected:
            raise InputError("Prepared/receipt artifact checksum mismatch: " + rel)
    if exact:
        actual = _tree_files(root)
        actual.pop(name, None); actual.pop(name + ".sha256", None)
        if actual != files:
            raise InputError("Prepared input inventory differs")
    return document


def seal(root, name, extra):
    inventory = _tree_files(root)
    if name in inventory or name + ".sha256" in inventory:
        raise InputError("Refusing to reseal an existing manifest")
    document = dict(extra, files=inventory)
    write_json(Path(root) / name, document)
    with (Path(root) / (name + ".sha256")).open("x", encoding="utf-8") as f:
        f.write(base.file_sha(Path(root) / name) + "  " + name + "\n")
    return document


def _spectral_input(prepared, archive, _fixture):
    prepared = Path(prepared)
    path = prepared / "input_manifest.json"
    token = (prepared / "input_manifest.json.sha256").read_text().split()[0]
    if not base.HASH_RE.fullmatch(token) or base.file_sha(path) != token:
        raise InputError("Parent spectral preparation checksum differs")
    saved = json_file(path)
    if not _fixture and saved["protocol"] != base.protocol():
        raise InputError("Parent spectral protocol differs from frozen source")
    current = base.inspect_archive(Path(archive), saved["archive_sha256"], saved["protocol"], _skip_source_lock=_fixture)
    if current != saved:
        raise InputError("Parent spectral preparation identity differs")
    return saved, token


def _train_selection(rows, domain, parent_spec, extension):
    train_spec = dict(parent_spec)
    train_spec.update(partition="train", context_blocks_per_cell=extension["training_context_blocks"],
                      test_families=[f"structural_train_{i:03d}" for i in range(extension["training_families"])],
                      paraphrase_ids=extension["paraphrase_ids"])
    return base._selection(rows, domain, train_spec)


def prepare(spectral_prepared, archive, output, *, _fixture=False, _protocol=None):
    extension = protocol() if _protocol is None else _protocol
    lock = source_lock(_fixture)
    if Path(output).exists() or Path(output).is_symlink():
        raise InputError("Preparation output exists; choose a new directory")
    saved, saved_sha = _spectral_input(spectral_prepared, archive, _fixture)
    if len(saved["cells"]) != extension["planned_cells"] or saved["protocol"]["rank"] != extension["rank"]:
        raise InputError("Extension panel differs from parent preparation")
    output = base._fresh(output)
    write_json(output / "parent_input_manifest.json", saved)
    write_json(output / "protocol.json", extension)
    write_json(output / "source_identity.json", lock)
    write_json(output / "preparation.started.json", {"status": "preparing", "parent_manifest_sha256": saved_sha,
               "archive_sha256": saved["archive_sha256"], "coordinate_spectra_computed": False})
    cells = []
    with zipfile.ZipFile(archive) as z:
        index = base.zip_index(z)
        for old in saved["cells"]:
            entry = {k: old[k] for k in ("cell", "domain", "model_id", "layer")}
            if old["status"] != "prepared_metadata_only":
                entry.update(status="not_evaluable_input", reason=old.get("reason", "Parent input not ready"))
            else:
                try:
                    prefix = saved["protocol"]["archive_prefix"] + "/analysis/" + old["cell"]
                    rows_raw = base.read_member(z, index, prefix + "/rows.json")
                    coordinates_raw = base.read_member(z, index, prefix + "/coordinates.npz", base.MAX_NPZ_BYTES)
                    pca_raw = base.read_member(z, index, prefix + "/pca.npz", base.MAX_NPZ_BYTES)
                    for filename, raw in (("rows.json", rows_raw), ("coordinates.npz", coordinates_raw), ("pca.npz", pca_raw)):
                        if base.sha(raw) != old["members"][prefix + "/" + filename]:
                            raise InputError("Selected member changed after parent inspection")
                    base._npz_headers(coordinates_raw)
                    base._npz_headers(pca_raw)
                    with np.load(io.BytesIO(pca_raw), allow_pickle=False) as pca:
                        if not all(np.isfinite(pca[k]).all() for k in ("center", "basis")):
                            raise InputError("Nonfinite saved PCA parameters")
                    rows = base._json(rows_raw)["rows"]
                    train_idx, train_contexts = _train_selection(rows, old["domain"], saved["protocol"], extension)
                    with np.load(io.BytesIO(coordinates_raw), allow_pickle=False) as data:
                        train = data["z"][np.asarray(train_idx, dtype=np.int64)]
                        test = data["z"][np.asarray(old["row_indices"], dtype=np.int64)]
                    if not np.isfinite(train).all() or not np.isfinite(test).all():
                        raise InputError("Nonfinite selected training or test coordinates")
                    folder = output / "cells" / old["cell"]
                    folder.mkdir(parents=True)
                    with (folder / "banks.npz").open("xb") as f:
                        np.savez_compressed(f, train=train, test=test)
                    write_json(folder / "rows.json", {"train_indices": train_idx, "train_contexts": train_contexts,
                               "test_indices": old["row_indices"], "test_contexts": old["contexts"]})
                    entry.update(status="prepared_inputs_only", bank_path="cells/" + old["cell"] + "/banks.npz",
                                 train_contexts=len(train_idx), test_contexts=len(old["row_indices"]),
                                 pca_schema=old["pca_schema"], source_members=old["members"], binding=old["binding"])
                except (InputError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
                    entry.update(status="not_evaluable_input", reason=str(exc))
            cells.append(entry)
    if base.file_sha(archive) != saved["archive_sha256"] or source_lock(_fixture) != lock:
        write_json(output / "preparation.failed.json", {"status": "identity_changed", "eligible": False})
        raise InputError("Archive/source changed during preparation")
    write_json(output / "inputs.json", {"cells": cells, "planned_cells": len(cells), "archive_sha256": saved["archive_sha256"],
               "parent_prepared_manifest_sha256": saved_sha, "protocol": extension, "source_identity": lock,
               "coordinate_spectra_computed": False, "rank_or_conditioning_passed": False})
    return seal(output, "prepared_manifest.json", {"schema_version": SCHEMA, "status": "prepared_inputs_only",
                "protocol_sha256": digest(extension), "source_identity": lock, "planned_cells": len(cells)})


def _prepared(path, extension, lock):
    manifest = verify_manifest(path, "prepared_manifest.json")
    inputs = json_file(Path(path) / "inputs.json")
    if inputs.get("protocol") != extension or inputs.get("source_identity") != lock or manifest.get("protocol_sha256") != digest(extension):
        raise InputError("Prepared source/protocol identity mismatch")
    if len(inputs.get("cells", [])) != extension["planned_cells"]:
        raise InputError("Prepared cell coverage differs")
    return inputs, base.file_sha(Path(path) / "prepared_manifest.json")


def _runtime():
    return {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}


@contextmanager
def controller(root, shared=False):
    path = Path(root) / ".controller.lock"
    with path.open("a+", encoding="utf-8") as f:
        try:
            fcntl.flock(f, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InputError("Another controller is active on this run") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _plan(stage, inputs, extension):
    if stage == "simulate":
        cfg = extension["simulation"]
        plan = []
        for domain in cfg["domains"]:
            for mechanism in cfg["mechanisms"]:
                for noise in cfg["noise_levels"]:
                    for seed in cfg["seeds"]:
                        plan.append({"unit": f"{domain}_{mechanism}_noise{noise:g}_seed{seed}", "domain": domain,
                                     "mechanism": mechanism, "noise": noise, "seed": seed})
        if len(plan) != cfg["planned_units"]:
            raise InputError("Simulation plan count differs")
        return plan, {}
    alternatives = {d: numerics.subgroup_specs(d) for d in extension["domains"]} if stage == "alternatives" else {}
    plan = []
    for cell in inputs["cells"]:
        if stage == "witness":
            plan.append({"unit": cell["cell"], "cell": cell["cell"], "domain": cell["domain"]})
        else:
            for group in alternatives[cell["domain"]]:
                plan.append({"unit": cell["cell"] + "--" + group["id"], "cell": cell["cell"], "domain": cell["domain"],
                             "subgroup_id": group["id"], "semantic_registration": group["semantic"]})
    expected = extension["planned_cells"] if stage == "witness" else extension["alternatives"]["planned_units"]
    if len(plan) != expected:
        raise InputError("Declared complete panel count differs")
    return plan, alternatives


def _source_snapshot(output, lock, _fixture):
    if _fixture:
        return
    for folder, root, files in (("extension", HERE, lock["files"]), ("base", BASE, lock["base_files"])):
        for rel, expected in files.items():
            base.safe_member(rel)
            source = root / rel
            if base.file_sha(source) != expected:
                raise InputError("Source changed before snapshot")
            destination = output / "source_snapshot" / folder / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as f:
                f.write(source.read_bytes())
        shutil.copyfile(root / "source_manifest.json", output / "source_snapshot" / folder / "source_manifest.json")


def _verify_source_snapshot(root, lock):
    if lock.get("software_fixture_only"):
        return
    for folder, files, manifest_hash in (("extension", lock["files"], lock["extension_manifest_sha256"]),
                                        ("base", lock["base_files"], lock["base_manifest_sha256"])):
        directory = Path(root) / "source_snapshot" / folder
        expected = dict(files, **{"source_manifest.json": manifest_hash})
        if _tree_files(directory) != expected:
            raise InputError("Saved source snapshot checksum/coverage differs")


def _event(root, value):
    with (Path(root) / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(utc=now(), **value), sort_keys=True, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def _receipt(root, unit, binding):
    folder = Path(root) / "units" / unit
    path = folder / "receipt.json"
    if not path.is_file():
        return None
    receipt = json_file(path)
    if receipt.get("binding") != binding or receipt.get("unit") != unit:
        raise InputError("Checkpoint binding differs: " + unit)
    attempt = receipt.get("attempt", "")
    base.safe_member(attempt)
    if not attempt.startswith("attempts/" + unit + "/"):
        raise InputError("Checkpoint attempt path differs")
    if (Path(root) / attempt).is_symlink() or not (Path(root) / attempt).resolve().is_relative_to(Path(root).resolve()):
        raise InputError("Checkpoint attempt escapes run")
    for rel, expected in receipt.get("files", {}).items():
        base.safe_member(rel)
        p = Path(root) / attempt / rel
        if p.is_symlink() or not p.is_file() or base.file_sha(p) != expected:
            raise InputError("Checkpoint artifact checksum mismatch: " + unit)
    if "summary.json" not in receipt.get("files", {}):
        raise InputError("Checkpoint summary missing")
    return json_file(Path(root) / attempt / "summary.json")


def _banks(root, cell):
    if cell["status"] != "prepared_inputs_only":
        return None
    rel = cell["bank_path"];base.safe_member(rel)
    path = Path(root) / rel
    if path.stat().st_size > base.MAX_NPZ_BYTES:
        raise InputError("Prepared bank exceeds allocation limit")
    raw = path.read_bytes();headers = base._npz_headers(raw)
    if set(headers) != {"train", "test"}:
        raise InputError("Prepared bank array names differ")
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return data["train"].copy(), data["test"].copy()


def _execute(stage, unit, cells, alternatives, bank_cache, prepared_root, extension):
    if stage == "simulate":
        cfg = extension["simulation"]
        train, test, construction = numerics.simulate_case(unit["domain"], unit["mechanism"], n_train=cfg["n_train"], n_test=cfg["n_test"], noise=unit["noise"], seed=unit["seed"])
        witness, arrays = numerics.witness_comparison(train, test, unit["domain"], ridge_alpha=extension["witness"]["ridge_alpha"])
        test_spectral, test_arrays = numerics.spectral.analyze_bank(test, unit["domain"])
        arrays.update({"test_bank_" + k: v for k, v in test_arrays.items()})
        arrays.update(simulated_train=train, simulated_test=test)
        summary = {"status": "complete" if witness["status"] == test_spectral["status"] == "complete" else "completed_with_not_evaluable",
                   "construction": construction, "witness": witness, "test_bank_spectral": test_spectral, "metrics": witness.get("metrics", {}),
                   "not_a_calibration_study": True, "mechanism_is_valid_negative_null": False}
        return summary, arrays
    cell = cells[unit["cell"]]
    if cell["status"] != "prepared_inputs_only":
        return {"status": "not_evaluable_input", "reason": cell.get("reason", "Prepared input unavailable")}, {}
    if bank_cache.get("cell") != unit["cell"]:
        bank_cache.clear();bank_cache.update(cell=unit["cell"], banks=_banks(prepared_root, cell))
    train, test = bank_cache["banks"]
    if stage == "witness":
        return numerics.witness_comparison(train, test, unit["domain"], ridge_alpha=extension["witness"]["ridge_alpha"])
    group = next(g for g in alternatives[unit["domain"]] if g["id"] == unit["subgroup_id"])
    return numerics.alternative_comparison(train, test, unit["domain"], group)


def _reports(root, stage, plan, binding):
    summaries = []
    for unit in plan:
        summary = _receipt(root, unit["unit"], binding)
        if summary is not None:
            summaries.append(summary)
    directory = Path(root) / "reports" / ("snapshot-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    fields = ["unit", "cell", "domain", "subgroup_id", "semantic_registration", "status", "reason", "train_gamma", "train_witness_risk", "test_witness_risk", "test_ols_risk", "test_ridge_risk", "test_bank_gamma"]
    with (directory / "unit_summary.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader()
        by_unit = {s["unit"]: s for s in summaries}
        for unit in plan:
            s = by_unit.get(unit["unit"], dict(unit, status="not_completed"))
            row = {k: s.get(k, "") for k in fields}
            row.update({k: v for k,v in s.get("metrics", {}).items() if k in fields})
            q = s.get("test_bank_spectral", {})
            row["test_bank_gamma"] = q.get("metrics", {}).get("gamma", "")
            writer.writerow(row)
    if stage == "alternatives":
        ranks = {}
        for cell in sorted({u["cell"] for u in plan}):
            subset = []
            by_unit = {s["unit"]: s for s in summaries}
            for unit in plan:
                if unit["cell"] == cell:
                    subset.append(by_unit.get(unit["unit"], dict(unit, status="not_completed")))
            ranks[cell] = numerics.registration_ranks(subset)
        write_json(directory / "descriptive_ranks.json", {"cells": ranks, "p_values": False})
    successful = sum(s["status"] == "complete" for s in summaries)
    completion = {"stage": stage, "planned": len(plan), "reported_terminal": len(summaries), "not_completed": len(plan) - len(summaries),
                  "successful": successful, "not_evaluable_units": len(summaries) - successful,
                  "all_units_evaluable": successful == len(plan), "scientific_pass": False,
                  "scientific_pass_assessed": False, "scientific_failures_preserved": True,
                  "independent_confirmation": False, "population_certificate": False}
    write_json(directory / "coverage.json", completion)
    return directory.relative_to(root).as_posix(), completion


def run(stage, output, *, prepared=None, execute_analysis=False, resume=False, _fixture=False, _protocol=None, _executor=None):
    if not execute_analysis:
        raise InputError("Research computation requires --execute-analysis")
    if stage not in ("witness", "alternatives", "simulate"):
        raise InputError("Unknown analysis stage")
    extension = protocol() if _protocol is None else _protocol
    lock = source_lock(_fixture)
    root = Path(output)
    exists = root.exists() or root.is_symlink()
    if exists and not resume:
        raise InputError("Run exists; explicit --resume is required")
    if not exists and resume:
        raise InputError("Resume requested for a missing run")
    if root.is_symlink():
        raise InputError("Symlink run directory refused")
    if exists:
        manifest = json_file(root / "run_manifest.json")
        expected = (root / "run_manifest.json.sha256").read_text().split()[0]
        if base.file_sha(root / "run_manifest.json") != expected:
            raise InputError("Run manifest checksum differs")
        if manifest.get("stage") != stage or manifest.get("protocol") != extension or manifest.get("source_identity") != lock:
            raise InputError("Resume source/protocol/stage differs; create a new run")
        if manifest.get("runtime", {}).get("numpy") != np.__version__ or manifest.get("runtime", {}).get("python") != platform.python_version():
            raise InputError("Resume numerical runtime version differs")
        if stage != "simulate":
            inputs, input_sha = _prepared(root / "input_snapshot", extension, lock)
            if input_sha != manifest.get("prepared_manifest_sha256"):
                raise InputError("Run input snapshot identity differs")
            if prepared is not None:
                _, other_sha = _prepared(prepared, extension, lock)
                if other_sha != input_sha:
                    raise InputError("Supplied resume preparation differs")
        else:
            inputs, input_sha = None, None
    else:
        if stage != "simulate":
            if prepared is None:
                raise InputError("Fresh real-coordinate run requires --prepared")
            inputs, input_sha = _prepared(prepared, extension, lock)
        else:
            inputs, input_sha = None, None
        root = base._fresh(root)
        if prepared is not None and stage != "simulate":
            shutil.copytree(prepared, root / "input_snapshot")
            _prepared(root / "input_snapshot", extension, lock)
        _source_snapshot(root, lock, _fixture)
        manifest = {"schema_version": SCHEMA, "stage": stage, "protocol": extension, "source_identity": lock, "runtime": _runtime(),
                    "prepared_manifest_sha256": input_sha, "created_at": now(), "independent_confirmation": False,
                    "population_certificate": False, "automatic_manuscript_promotion": False}
        write_json(root / "run_manifest.json", manifest)
        with (root / "run_manifest.json.sha256").open("x", encoding="utf-8") as f:
            f.write(base.file_sha(root / "run_manifest.json") + "  run_manifest.json\n")
    with controller(root):
        _verify_source_snapshot(root, lock)
        plan, alternatives = _plan(stage, inputs, extension)
        plan_path = root / "plan.json"
        if plan_path.exists():
            if json_file(plan_path) != {"units": plan, "subgroups": alternatives}:
                raise InputError("Resume unit plan differs")
        else:
            write_json(plan_path, {"units": plan, "subgroups": alternatives})
        binding = {"run_manifest_sha256": base.file_sha(root / "run_manifest.json"), "plan_sha256": base.file_sha(plan_path)}
        cells = {} if inputs is None else {c["cell"]: c for c in inputs["cells"]}
        cache = {}
        _event(root, {"event": "session_start", "resume": resume})
        completed = 0
        for unit in plan:
            uid = unit["unit"];base.safe_member(uid)
            if _receipt(root, uid, binding) is not None:
                completed += 1
                print("RESUME_UNIT", uid, flush=True)
                continue
            attempt_rel = "attempts/" + uid + "/" + uuid.uuid4().hex
            attempt = root / attempt_rel;attempt.mkdir(parents=True)
            write_json(attempt / "started.json", {"unit": uid, "binding": binding, "utc": now()})
            atomic_json(root / "state.json", {"status": "running", "stage": stage, "active_unit": uid, "completed": completed, "planned": len(plan)})
            print("START_UNIT", uid, flush=True)
            try:
                if _executor is None:
                    summary, arrays = _execute(stage, unit, cells, alternatives, cache, root / "input_snapshot", extension)
                else:
                    summary, arrays = _executor(stage, unit)
                summary = dict(summary, **unit, independent_confirmation=False, population_certificate=False)
                if not isinstance(summary.get("status"), str):
                    raise InputError("Numerical result has no status")
                write_json(attempt / "summary.json", summary)
                if arrays:
                    with (attempt / "diagnostics.npz").open("xb") as f:
                        np.savez_compressed(f, **arrays)
                files = _tree_files(attempt)
                folder = root / "units" / uid;folder.mkdir(parents=True, exist_ok=True)
                # No receipt exists until all files have been written and hashed.
                write_json(folder / "receipt.json", {"unit": uid, "binding": binding, "attempt": attempt_rel,
                           "files": files, "status": summary["status"], "terminal": True})
                completed += 1
                _event(root, {"event": "unit_complete", "unit": uid, "status": summary["status"]})
                print("DONE_UNIT", uid, summary["status"], f"{completed}/{len(plan)}", flush=True)
            except Exception as exc:
                write_json(attempt / "failure.json", {"status": "technical_failure", "error_type": type(exc).__name__, "error": str(exc), "utc": now()})
                _event(root, {"event": "unit_technical_failure", "unit": uid, "error_type": type(exc).__name__})
                report, coverage = _reports(root, stage, plan, binding)
                atomic_json(root / "state.json", {"status": "technical_failure", "stage": stage, "active_unit": uid, "completed": completed,
                            "planned": len(plan), "report": report, "error": str(exc)})
                raise
        if source_lock(_fixture) != lock:
            atomic_json(root / "state.json", {"status": "source_changed", "results_eligible": False})
            raise InputError("Source changed during computation")
        if inputs is not None:
            _prepared(root / "input_snapshot", extension, lock)
        report, coverage = _reports(root, stage, plan, binding)
        final_status = "complete" if coverage["all_units_evaluable"] else "completed_with_not_evaluable"
        atomic_json(root / "state.json", {"status": final_status, "execution_status": "complete", "stage": stage, "report": report, **coverage})
        _event(root, {"event": "session_complete", "completed": completed})
        return coverage


def collect(run_path, output):
    root, output = Path(run_path), Path(output)
    if root.is_symlink():
        raise InputError("Symlink run directory refused")
    if output.exists() or output.is_symlink() or output.with_name(output.name + ".sha256").exists():
        raise InputError("Collection destination exists; choose a new filename")
    if output.resolve().is_relative_to(root.resolve()):
        raise InputError("Collection ZIP must be outside the run directory")
    if not (root / "run_manifest.json").is_file():
        raise InputError("Not an extension run directory")
    with controller(root, shared=True):
        manifest = json_file(root / "run_manifest.json")
        expected = (root / "run_manifest.json.sha256").read_text().split()[0]
        if base.file_sha(root / "run_manifest.json") != expected:
            raise InputError("Run manifest checksum differs")
        _verify_source_snapshot(root, manifest["source_identity"])
        plan_data = json_file(root / "plan.json")
        binding = {"run_manifest_sha256": expected, "plan_sha256": base.file_sha(root / "plan.json")}
        completed = 0
        successful = 0
        for unit in plan_data["units"]:
            summary = _receipt(root, unit["unit"], binding)
            if summary is not None:
                completed += 1
                successful += summary["status"] == "complete"
        if (root / "input_snapshot").exists():
            verify_manifest(root / "input_snapshot", "prepared_manifest.json")
        files = _tree_files(root);files.pop(".controller.lock", None)
        state = json_file(root / "state.json") if (root / "state.json").is_file() else {"status": "not_completed"}
        execution_complete = (completed == len(plan_data["units"])
                              and state.get("execution_status") == "complete"
                              and state.get("status") in ("complete", "completed_with_not_evaluable"))
        collection = {"schema_version": SCHEMA, "stage": manifest["stage"], "planned": len(plan_data["units"]), "terminal_units": completed,
                      "partial": not execution_complete, "run_status": state.get("status"), "execution_complete": execution_complete,
                      "successful": successful, "not_evaluable_units": completed - successful,
                      "not_completed": len(plan_data["units"]) - completed,
                      "all_units_evaluable": successful == len(plan_data["units"]),
                      "scientific_pass": False, "scientific_pass_assessed": False,
                      "files": files, "automatic_manuscript_promotion": False}
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(output.name + ".partial-" + uuid.uuid4().hex)
        try:
            with zipfile.ZipFile(temp, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for rel, expected_hash in files.items():
                    p = root / rel
                    if base.file_sha(p) != expected_hash:
                        raise InputError("Run changed during collection")
                    info = zipfile.ZipInfo("run/" + rel, STAMP);info.compress_type=zipfile.ZIP_DEFLATED
                    with p.open("rb") as stream, z.open(info, "w", force_zip64=True) as sink:
                        shutil.copyfileobj(stream, sink, length=2**20)
                z.writestr("collection_manifest.json", json.dumps(collection, indent=2, sort_keys=True, allow_nan=False) + "\n")
            # Hard-link publication refuses an existing destination atomically.
            os.link(temp, output);temp.unlink()
            with output.with_name(output.name + ".sha256").open("x", encoding="utf-8") as f:
                f.write(base.file_sha(output) + "  " + output.name + "\n")
        finally:
            if temp.exists():
                temp.unlink()
    return collection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare");p.add_argument("--spectral-prepared", type=Path, required=True);p.add_argument("--archive", type=Path, required=True);p.add_argument("--output", type=Path, required=True)
    for name in ("witness", "alternatives", "simulate"):
        p = commands.add_parser(name)
        if name != "simulate":p.add_argument("--prepared", type=Path)
        p.add_argument("--output", type=Path, required=True);p.add_argument("--execute-analysis", action="store_true");p.add_argument("--resume", action="store_true")
    p = commands.add_parser("collect");p.add_argument("--run", type=Path, required=True);p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":result = prepare(args.spectral_prepared, args.archive, args.output)
        elif args.command == "collect":result = collect(args.run, args.output)
        else:result = run(args.command, args.output, prepared=getattr(args, "prepared", None), execute_analysis=args.execute_analysis, resume=args.resume)
        print(json.dumps({k:v for k,v in result.items() if k not in ("files", "source_identity")}, sort_keys=True, allow_nan=False))
    except (InputError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print("EXTENSION_STOP:", str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
