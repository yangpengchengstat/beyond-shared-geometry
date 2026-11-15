#!/usr/bin/env python3
"""Opt-in, model-free finite-bank analysis. Preparation never computes a spectrum."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_NPZ_BYTES = 16 * 1024 * 1024
MAX_NPY_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 700_000
CELL_FILES = ("coordinates.npz", "pca.npz", "rows.json", "cell_complete.json")


class InputError(ValueError):
    """A failed input contract, never a scientific negative result."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path, value):
    # Every output directory is newly created; exclusive writes refuse replacement.
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def _json(raw):
    if len(raw) > MAX_JSON_BYTES:
        raise InputError("JSON exceeds declared size limit")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InputError("Duplicate JSON key")
            result[key] = value
        return result
    def bad_constant(value):
        raise InputError("Nonfinite JSON constant: " + value)
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise InputError("Nonfinite JSON number")
        return number
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant, parse_float=finite_float)
        if not isinstance(value, dict):
            raise InputError("Expected a JSON object")
        return value
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError("Invalid UTF-8 JSON") from exc


def safe_member(name):
    p = PurePosixPath(name)
    if not name or "\\" in name or name.startswith("/") or any(x in (".", "..") for x in name.split("/")) or ":" in p.parts[0]:
        raise InputError("Unsafe archive member name")
    return name


def zip_index(z, limit=MAX_ARCHIVE_MEMBERS):
    entries = z.infolist()
    if len(entries) > limit:
        raise InputError("Archive member count exceeds declared limit")
    result = {}
    for info in entries:
        safe_member(info.filename)
        if info.filename in result:
            raise InputError("Duplicate archive member: " + info.filename)
        if stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
            raise InputError("Symlink/encrypted archive member is not allowed")
        result[info.filename] = info
    return result


def read_member(z, index, name, maximum=MAX_JSON_BYTES):
    safe_member(name)
    info = index.get(name)
    if info is None or info.is_dir():
        raise InputError("Missing required member: " + name)
    if info.file_size < 0 or info.file_size > maximum:
        raise InputError("Oversized required member: " + name)
    raw = z.read(info)
    if len(raw) != info.file_size:
        raise InputError("Member size mismatch: " + name)
    return raw


def _npz_headers(raw):
    """Read only headers and bytes, never numeric arrays or pickle objects."""
    result = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as inner:
            index = zip_index(inner, 8)
            for name, info in index.items():
                if not name.endswith(".npy") or "/" in name:
                    raise InputError("Unexpected NPZ member")
                payload = read_member(inner, index, name, MAX_NPY_BYTES)
                f = io.BytesIO(payload)
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f, max_header_size=16384)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f, max_header_size=16384)
                else:
                    raise InputError("Unsupported NPY schema")
                if dtype.hasobject or dtype.kind != "f" or dtype.itemsize != 8:
                    raise InputError("NPZ must contain real float64 arrays without objects")
                count = 1
                for dimension in shape:
                    if not isinstance(dimension, int) or dimension < 0:
                        raise InputError("Invalid array shape")
                    count *= dimension
                    if count * dtype.itemsize > MAX_NPY_BYTES:
                        raise InputError("NPY declared allocation exceeds limit")
                if len(payload) - f.tell() != count * dtype.itemsize:
                    raise InputError("NPY payload/header size mismatch")
                result[name[:-4]] = {"shape": list(shape), "dtype": dtype.str, "fortran_order": bool(fortran), "npy_sha256": sha(payload)}
    except (zipfile.BadZipFile, EOFError, ValueError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError("Malformed NPZ/NPY") from exc
    return result


def source_lock(_skip_source_lock=False):
    path = HERE / "source_manifest.json"
    if _skip_source_lock:
        return {"test_fixture_only": True}
    if not path.is_file():
        raise InputError("Missing frozen analysis source_manifest.json")
    lock = _json(path.read_bytes())
    files = lock.get("files", {})
    if not files:
        raise InputError("Empty analysis source manifest")
    actual = set()
    for p in HERE.rglob("*"):
        rel = p.relative_to(HERE)
        if any(x in rel.parts for x in ("__pycache__", ".pytest_cache")) or p.suffix in (".pyc", ".pyo") or str(rel) == "source_manifest.json":
            continue
        if p.is_symlink():
            raise InputError("Symlink in analysis source")
        if p.is_file():
            actual.add(rel.as_posix())
    if actual != set(files):
        raise InputError("Analysis source manifest file coverage differs")
    for name, digest in files.items():
        safe_member(name)
        if not HASH_RE.fullmatch(str(digest)) or file_sha(HERE / name) != digest:
            raise InputError("Analysis source hash mismatch: " + name)
    return {"manifest_sha256": file_sha(path), "files": files}


def protocol():
    return _json((HERE / "protocol.json").read_bytes())


def model_key(model_id):
    return model_id.replace("/", "__")


def planned_cells(spec):
    return [{"cell": f'{model_key(m["id"])}__{domain}__{layer}', "model": m, "domain": domain, "layer": layer}
            for m in spec["models"] for domain in spec["domains"] for layer in m["layers"]]


def _selection(rows, domain, spec):
    if not isinstance(rows, list):
        raise InputError("Rows metadata is not a list")
    p = {"C7": 7, "D4": 4}[domain]
    # The frozen four-corner dataset is named D4_corner; D4 names the
    # analysis cell/group. Preserve original row and orbit identities.
    row_domain = spec.get("archive_row_domains", {}).get(domain, domain)
    seen, selected = set(), {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("domain") != row_domain:
            raise InputError("Row domain differs")
        keys = ("row_id", "split", "family_id", "paraphrase_id", "orbit_id", "state_id")
        if any(k not in row for k in keys):
            raise InputError("Row metadata incomplete")
        rid = row["row_id"]
        if not isinstance(rid, str) or not rid or rid in seen:
            raise InputError("Missing or duplicate row ID")
        seen.add(rid)
        if type(row["state_id"]) is not int or not 0 <= row["state_id"] < p:
            raise InputError("Invalid state ID")
        if row["split"] == spec["partition"]:
            key = (row["family_id"], row["paraphrase_id"], row["orbit_id"])
            if type(row["paraphrase_id"]) is not int:
                raise InputError("Invalid paraphrase ID")
            canonical_id = f'{row_domain}/{row["family_id"]}/p{row["paraphrase_id"]}/lexical_{row_domain}/s{row["state_id"]}'
            if rid != canonical_id:
                raise InputError("Test row identity is not canonical")
            states = selected.setdefault(key, {})
            if row["state_id"] in states:
                raise InputError("Repeated test state in a context block")
            states[row["state_id"]] = (i, rid)
    expected = [(family, paraphrase, "lexical_" + row_domain) for family in spec["test_families"] for paraphrase in spec["paraphrase_ids"]]
    if set(selected) != set(expected) or len(expected) != spec["context_blocks_per_cell"]:
        raise InputError("Missing/unexpected test context block; no subset is retained")
    indices, identities = [], []
    for key in expected:
        if set(selected[key]) != set(range(p)):
            raise InputError("Missing test state; no incomplete block is dropped")
        indices.append([selected[key][s][0] for s in range(p)])
        identities.append({"split": spec["partition"], "family_id": key[0], "paraphrase_id": key[1], "orbit_id": key[2],
                           "state_ids": list(range(p)), "row_ids": [selected[key][s][1] for s in range(p)]})
    return indices, identities


def _inspect_cell(z, index, cell, contract, spec):
    prefix = spec["archive_prefix"]
    directory = prefix + "/analysis/" + cell["cell"]
    raw = {name: read_member(z, index, directory + "/" + name, MAX_NPZ_BYTES if name.endswith(".npz") else MAX_JSON_BYTES) for name in CELL_FILES}
    hashes = {name: sha(value) for name, value in raw.items()}
    receipt = _json(raw["cell_complete.json"])
    binding = receipt.get("binding", {})
    if not isinstance(binding, dict) or not isinstance(receipt.get("artifacts"), dict):
        raise InputError("Malformed cell receipt binding/artifacts")
    for key, expected in (("contract_id", spec["study_contract_id"]), ("domain", cell["domain"]), ("layer", cell["layer"]),
                          ("numerical_source_manifest_sha256", spec["frozen_source_manifest_sha256"])):
        if binding.get(key) != expected:
            raise InputError("Cell receipt identity mismatch: " + key)
    for name in ("coordinates.npz", "pca.npz", "rows.json"):
        if receipt.get("artifacts", {}).get(name) != hashes[name]:
            raise InputError("Cell artifact hash mismatch: " + name)
    merged_name = prefix + "/provenance/" + model_key(cell["model"]["id"]) + "__" + cell["domain"] + "__merged.json"
    merged_raw = read_member(z, index, merged_name)
    merged = _json(merged_raw)
    if sha(merged_raw) != binding.get("cache_manifest_sha256"):
        raise InputError("Merged extraction manifest hash mismatch")
    m = cell["model"]
    identity = merged.get("model", {})
    if not isinstance(identity, dict):
        raise InputError("Malformed model identity")
    for field, value in (("model_id", m["id"]), ("model_revision", m["revision"]), ("tokenizer_revision", m["tokenizer_revision"])):
        if identity.get(field) != value:
            raise InputError("Model/tokenizer identity mismatch: " + field)
    if merged.get("readout") != spec["readout"] or merged.get("status") != "complete" or merged.get("layers") != m["layers"]:
        raise InputError("Readout/layer/extraction status mismatch")
    descriptor = merged.get("descriptor", {})
    if not isinstance(descriptor, dict):
        raise InputError("Malformed extraction descriptor")
    if descriptor.get("model") != identity or descriptor.get("readout") != spec["readout"] or descriptor.get("layers") != m["layers"]:
        raise InputError("Extraction descriptor disagrees with manifest")
    rows = _json(raw["rows.json"]).get("rows")
    indices, contexts = _selection(rows, cell["domain"], spec)
    total = len(rows)
    merged_arrays = merged.get("arrays", {})
    if not isinstance(merged_arrays, dict) or not isinstance(merged_arrays.get(str(cell["layer"])), dict):
        raise InputError("Malformed extraction array schema")
    if merged.get("n_rows") != total or merged_arrays[str(cell["layer"])].get("shape") != [total, m["hidden_dimension"]]:
        raise InputError("Extraction row/hidden-dimension schema mismatch")
    coords = _npz_headers(raw["coordinates.npz"])
    pca = _npz_headers(raw["pca.npz"])
    if set(coords) != {"z"} or coords["z"]["shape"] != [total, spec["rank"]]:
        raise InputError("Coordinate schema/row count/rank mismatch")
    if set(pca) != {"center", "basis"} or pca["center"]["shape"] != [m["hidden_dimension"]] or pca["basis"]["shape"] != [m["hidden_dimension"], spec["rank"]]:
        raise InputError("PCA schema/identity dimensions mismatch")
    result = {"status": "prepared_metadata_only", "cell": cell["cell"], "model_id": m["id"], "revision": m["revision"],
              "archive_row_domain": spec.get("archive_row_domains", {}).get(cell["domain"], cell["domain"]),
              "tokenizer_revision": m["tokenizer_revision"], "layer": cell["layer"], "domain": cell["domain"], "readout": spec["readout"],
              "binding": binding, "members": {directory + "/" + n: h for n, h in hashes.items()}, "merged_manifest": merged_name,
              "merged_manifest_sha256": sha(merged_raw), "coordinate_schema": coords, "pca_schema": pca, "row_indices": indices,
              "contexts": contexts, "partition": spec["partition"], "numerical_eligibility_checked": False}
    return result


def inspect_archive(archive, expected_sha256, spec, _skip_source_lock=False):
    if not HASH_RE.fullmatch(expected_sha256):
        raise InputError("Expected SHA-256 must be 64 lowercase hex characters")
    source = source_lock(_skip_source_lock)
    digest = file_sha(archive)
    if digest != expected_sha256:
        raise InputError("Archive SHA-256 mismatch")
    with zipfile.ZipFile(archive) as z:
        index = zip_index(z)
        contract_raw = read_member(z, index, spec["archive_prefix"] + "/study_contract.json")
        contract = _json(contract_raw)
        if sha(contract_raw) != spec["study_contract_sha256"] or sha(canonical(contract)) != spec["study_contract_id"]:
            raise InputError("Frozen study contract identity mismatch")
        frozen = read_member(z, index, spec["archive_prefix"] + "/provenance/frozen_source_manifest.json")
        if sha(frozen) != spec["frozen_source_manifest_sha256"] or contract.get("frozen_source_manifest_sha256") != sha(frozen):
            raise InputError("Frozen numerical source identity mismatch")
        if contract.get("driver_sha256") != spec["driver_sha256"] or contract.get("study_id") != spec["study_id"]:
            raise InputError("Frozen driver/study identity mismatch")
        actual_models = [{k: m[k] for k in ("id", "revision", "tokenizer_revision", "hidden_dimension", "layers")} for m in contract["models"]]
        if actual_models != spec["models"]:
            raise InputError("Planned model panel differs from contract")
        planned = planned_cells(spec)
        if len(planned) != spec["planned_cells"]:
            raise InputError("Protocol cell-count inconsistency")
        cells = []
        for item in planned:
            try:
                cells.append(_inspect_cell(z, index, item, contract, spec))
            except (InputError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
                cells.append({"cell": item["cell"], "domain": item["domain"], "model_id": item["model"]["id"], "layer": item["layer"],
                              "status": "not_evaluable_input", "reason": str(exc), "numerical_eligibility_checked": False})
    # Rehash after reading protects against ordinary concurrent archive replacement.
    if file_sha(archive) != digest:
        raise InputError("Archive changed during preparation")
    from gamma_r import group_spec
    return {"schema_version": 1, "stage": "metadata_preparation_only", "archive_sha256": digest, "protocol": spec,
            "protocol_sha256": sha(canonical(spec)), "source_lock": source, "groups": {d: group_spec(d) for d in spec["domains"]},
            "planned_cells": len(planned), "cells": cells, "coordinate_values_analyzed": False,
            "independent_confirmation": False, "population_certificate": False}


def _fresh(path):
    p = Path(path)
    if p.exists() or p.is_symlink():
        raise InputError("Output already exists; choose a fresh directory")
    p.mkdir(parents=True, exist_ok=False)
    return p


def prepare(archive, expected_sha256, output, *, _spec=None, _skip_source_lock=False):
    output, archive = Path(output), Path(archive).resolve()
    if output.exists() or output.is_symlink():
        raise InputError("Output already exists; choose a fresh directory")
    spec = protocol() if _spec is None else _spec
    manifest = inspect_archive(archive, expected_sha256, spec, _skip_source_lock)
    destination = _fresh(output)
    write_json(destination / "input_manifest.json", manifest)
    (destination / "input_manifest.json.sha256").write_text(file_sha(destination / "input_manifest.json") + "  input_manifest.json\n", encoding="utf-8")
    # Runtime path is private, separated from scientific identity and replaceable via --archive.
    write_json(destination / "local_input.json", {"archive_path": str(archive)})
    write_json(destination / "preparation_status.json", {"stage": "metadata_preparation_only", "planned": len(manifest["cells"]),
               "prepared": sum(c["status"] == "prepared_metadata_only" for c in manifest["cells"]), "coordinate_values_analyzed": False})
    return manifest


def analyze(prepared, output, *, execute_analysis=False, archive=None, _skip_source_lock=False, _analyzer=None):
    if not execute_analysis:
        raise InputError("Analysis requires explicit --execute-analysis; no coordinates were analyzed")
    prepared, output = Path(prepared), Path(output)
    if output.exists() or output.is_symlink():
        raise InputError("Output already exists; choose a fresh directory")
    manifest_path = prepared / "input_manifest.json"
    expected = (prepared / "input_manifest.json.sha256").read_text().split()[0]
    if not HASH_RE.fullmatch(expected) or file_sha(manifest_path) != expected:
        raise InputError("Prepared manifest checksum mismatch")
    saved = _json(manifest_path.read_bytes())
    spec = saved["protocol"]
    if not _skip_source_lock and spec != protocol():
        raise InputError("Prepared protocol differs from the frozen implementation")
    if archive is None:
        archive = _json((prepared / "local_input.json").read_bytes())["archive_path"]
    archive = Path(archive).resolve()
    current = inspect_archive(archive, saved["archive_sha256"], spec, _skip_source_lock)
    if current != saved:
        raise InputError("Prepared inputs or source identity changed; create a new preparation")
    destination = _fresh(output)
    # Seal input, protocol and software identity before the first numerical result.
    write_json(destination / "input_manifest.json", saved)
    write_json(destination / "protocol.json", spec)
    write_json(destination / "runtime.json", {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
               "source_lock": saved["source_lock"], "interpretation": "Retrospective finite-bank diagnostic; no new-context or causal certificate."})
    write_json(destination / "state.started.json", {"status": "running", "planned_cells": len(saved["cells"]), "input_sha256": expected})
    if _analyzer is None:
        from gamma_r import analyze_bank
        _analyzer = analyze_bank
    summaries = []
    with zipfile.ZipFile(archive) as z:
        index = zip_index(z)
        for item in saved["cells"]:
            folder = destination / "cells" / item["cell"]
            folder.mkdir(parents=True)
            arrays = {}
            if item["status"] != "prepared_metadata_only":
                summary = {"status": "not_evaluable_input", "reason": item["reason"]}
            else:
                try:
                    prefix = spec["archive_prefix"] + "/analysis/" + item["cell"]
                    # Verify current selected bytes again at their exact point of use.
                    coord_raw = read_member(z, index, prefix + "/coordinates.npz", MAX_NPZ_BYTES)
                    pca_raw = read_member(z, index, prefix + "/pca.npz", MAX_NPZ_BYTES)
                    for name, raw in (("coordinates.npz", coord_raw), ("pca.npz", pca_raw)):
                        if sha(raw) != item["members"][prefix + "/" + name]:
                            raise InputError("Selected input changed after preparation")
                        _npz_headers(raw)
                    with np.load(io.BytesIO(pca_raw), allow_pickle=False) as pca:
                        if not all(np.isfinite(pca[k]).all() for k in ("center", "basis")):
                            raise InputError("Nonfinite saved PCA parameters")
                    with np.load(io.BytesIO(coord_raw), allow_pickle=False) as coord:
                        Z = coord["z"][np.asarray(item["row_indices"], dtype=np.int64)]
                    summary, arrays = _analyzer(Z, item["domain"])
                except (InputError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
                    summary = {"status": "not_evaluable_input", "reason": str(exc)}
            summary = dict(summary, cell=item["cell"], domain=item["domain"], model_id=item["model_id"], layer=item["layer"],
                           input_status=item["status"], independent_confirmation=False, population_certificate=False)
            write_json(folder / "summary.json", summary)
            if arrays:
                with (folder / "diagnostics.npz").open("xb") as f:
                    np.savez_compressed(f, **arrays)
            summaries.append(summary)
            print("CELL", item["cell"], summary["status"], flush=True)
    if file_sha(archive) != saved["archive_sha256"]:
        write_json(destination / "state.failed.json", {"status": "input_changed_during_analysis", "results_eligible": False})
        raise InputError("Archive changed during analysis; results must not be interpreted")
    if source_lock(_skip_source_lock) != saved["source_lock"]:
        write_json(destination / "state.failed.json", {"status": "source_changed_during_analysis", "results_eligible": False})
        raise InputError("Analysis source changed during computation")
    fields = ("cell", "domain", "model_id", "layer", "status", "reason", "gamma", "C_r", "C_untyped", "delta_type",
              "delta_type_normalized", "L_lower", "L_upper", "eta_spec", "eta_LS")
    with (destination / "cell_summary.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = {k: summary.get(k, "") for k in fields}
            row.update({k: v for k, v in summary.get("metrics", {}).items() if k in fields})
            writer.writerow(row)
    type_columns = ("cell", "domain", "type", "real_dimension", "isotypic_dimension", "e", "b", "selected_copies", "selected_dimension", "multiplicity_index", "multiplicity_eigenvalue", "selected")
    with (destination / "type_spectra.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=type_columns)
        writer.writeheader()
        for summary in summaries:
            if summary["status"] != "complete":
                continue
            for t in summary.get("types", []):
                for j, eigenvalue in enumerate(t["multiplicity_eigenvalues"]):
                    row = {k: t[k] for k in ("real_dimension", "isotypic_dimension", "e", "b", "selected_copies", "selected_dimension")}
                    row.update(cell=summary["cell"], domain=summary["domain"], type=t["name"], multiplicity_index=j,
                               multiplicity_eigenvalue=eigenvalue, selected=j < t["selected_copies"])
                    writer.writerow(row)
    write_json(destination / "validation.json", {"planned": len(saved["cells"]), "reported": len(summaries),
               "complete": sum(s["status"] == "complete" for s in summaries), "all_cells_retained": True,
               "independent_confirmation": False, "population_certificate": False,
               "cell_checks": [{"cell": s["cell"], "status": s["status"], "checks": s.get("checks", {})} for s in summaries]})
    (destination / "interpretation.txt").write_text(
        "This is a retrospective structural analysis of the declared finite test-coordinate bank.\n"
        "The 48 selected cells are not independent replications. No confidence intervals, p-values,\n"
        "new-context coverage, native causal conclusion or automatic manuscript promotion is produced.\n"
        "Risk bounds concern the exact orthogonal class in the new empirical second-moment metric.\n"
        "Historical predictor scores are different endpoints. No witness action was constructed.\n"
        "Read all failures and raw residuals. Near-zero quantities are unresolved at the stated precision.\n", encoding="utf-8")
    outputs = {p.relative_to(destination).as_posix(): file_sha(p) for p in destination.rglob("*") if p.is_file()}
    write_json(destination / "output_manifest.json", {"files": outputs, "status": "complete"})
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare", help="Validate hashes and metadata; do not calculate spectra")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = subs.add_parser("analyze", help="Explicitly run the fixed existing-coordinate diagnostic")
    a.add_argument("--prepared", type=Path, required=True)
    a.add_argument("--archive", type=Path, help="Optional relocated archive with the identical checksum")
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--execute-analysis", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.archive, args.expected_sha256, args.output)
            print(json.dumps({"stage": result["stage"], "planned": len(result["cells"]), "prepared": sum(c["status"] == "prepared_metadata_only" for c in result["cells"]), "coordinate_values_analyzed": False}))
        else:
            summaries = analyze(args.prepared, args.output, execute_analysis=args.execute_analysis, archive=args.archive)
            print(json.dumps({"reported": len(summaries), "complete": sum(s["status"] == "complete" for s in summaries)}))
    except (InputError, OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        print("INPUT_STOP:", str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
