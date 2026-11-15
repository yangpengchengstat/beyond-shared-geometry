"""Export saved bytes and JSON projections; never import a numerical package.

Requires the two authenticated historical archives and the two existing diagnostic
archive directories. Outputs must be new. This is packaging, not an experiment.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile

ARCHIVE_HASHES = {
    "parent": "6a4c38237bf1368d4c77c97504f666d0d933bfebed9417d173fbf7441c0fecef",
    "followup": "ef87e9aaeb8f093ba0c2438590d6a30b5dffcd256993e2366bd600cca3cb5bbb",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def npz_headers(raw):
    answer = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for member in archive.infolist():
            if "/" in member.filename or not member.filename.endswith(".npy"):
                raise ValueError("Unexpected NPZ member")
            value = archive.read(member)
            stream = io.BytesIO(value)
            if stream.read(6) != b"\x93NUMPY":
                raise ValueError("Invalid NPY header")
            version = tuple(stream.read(2))
            width = 2 if version == (1, 0) else 4
            length = struct.unpack("<H" if width == 2 else "<I", stream.read(width))[0]
            if length > 16384:
                raise ValueError("Oversized NPY header")
            header = ast.literal_eval(stream.read(length).decode("utf-8" if version[0] == 3 else "latin1"))
            answer[member.filename[:-4]] = {
                "shape": list(header["shape"]), "dtype": header["descr"],
                "fortran_order": header["fortran_order"], "npy_sha256": digest(value),
            }
    return answer


def project_anchor(saved):
    """Copy existing fields used by the unchanged historical mathematical code."""
    keys = ("status", "domain", "n_contexts", "n_states", "r", "group_order", "metrics", "types")
    answer = {key: saved[key] for key in keys}
    answer["diagnostics"] = {key: saved["diagnostics"][key]
                             for key in ("sigma_min", "sigma_max", "sigma_min_max_ratio")}
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--within-archive", type=Path, required=True)
    parser.add_argument("--fixed-archive", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repository.resolve()
    data = repo / "data/retained_coordinates"
    study = repo / "studies/template_centering_v1"
    results = repo / "results/template_centering_v1"
    for path in (data / "manifest.json", data / "manifest_c7_d4.json"):
        if path.exists():
            raise FileExistsError(path)
    for alias in ARCHIVE_HASHES:
        if file_digest(getattr(args, alias)) != ARCHIVE_HASHES[alias]:
            raise ValueError("Historical archive identity differs: " + alias)
    objects = {}

    def write(relative, raw):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        if file_digest(path) != digest(raw):
            raise ValueError("Export identity differs")
        return {"path": relative, "sha256": digest(raw), "bytes": len(raw)}

    def store(raw, suffix):
        sha = digest(raw)
        relative = "data/retained_coordinates/objects/" + sha + suffix
        if relative not in objects:
            objects[relative] = write(relative, raw)
        return objects[relative]

    with zipfile.ZipFile(args.parent) as parent, zipfile.ZipFile(args.followup) as followup:
        parent_index = {entry.filename: entry for entry in parent.infolist()}
        input_name = next(name for name in followup.namelist()
                          if name.startswith("spectral-results-") and name.endswith("/input_manifest.json"))
        input_raw = followup.read(input_name)
        inputs = json.loads(input_raw)
        if inputs["archive_sha256"] != ARCHIVE_HASHES["parent"]:
            raise ValueError("Historical input archive binding differs")
        spectral_cells = {cell["cell"]: cell for cell in inputs["cells"]}
        if len(spectral_cells) != 48:
            raise ValueError("Expected 48 C7/D4 cells")
        coordinate_members = sorted(name for name in parent_index
                                    if name.endswith("/coordinates.npz")
                                    and name.startswith("parent-from-zero-v1/characterization/analysis/"))
        if len(coordinate_members) != 72:
            raise ValueError("Expected 72 retained-coordinate cells")
        cells = []
        for coordinate_name in coordinate_members:
            prefix = coordinate_name.rsplit("/", 1)[0]
            cell_id = prefix.rsplit("/", 1)[1]
            model_slug, domain, layer = cell_id.rsplit("__", 2)
            metadata = spectral_cells.get(cell_id)
            entry = {"cell": cell_id, "domain": domain, "layer": int(layer),
                     "source_archive": "parent", "objects": {}}
            for role, filename in (("coordinates", "coordinates.npz"), ("pca", "pca.npz"), ("rows", "rows.json")):
                member = prefix + "/" + filename
                if role == "rows" and member not in parent_index and domain == "S3":
                    member = "parent-from-zero-v1/characterization/analysis/shared/" + model_slug + "__S3/rows.json"
                raw = parent.read(member)
                sha = digest(raw)
                if metadata and sha != metadata["members"][member]:
                    raise ValueError("Spectral input member hash differs")
                record = {**store(raw, Path(filename).suffix), "source_member": member,
                          "source_member_sha256": sha, "byte_preserved": True, "role": role}
                if role in ("coordinates", "pca"):
                    record["arrays"] = npz_headers(raw)
                else:
                    parsed = json.loads(raw)
                    rows = parsed["rows"] if isinstance(parsed, dict) else parsed
                    record["row_count"] = len(rows)
                    record["split_counts"] = {split: sum(row["split"] == split for row in rows)
                                              for split in sorted({row["split"] for row in rows})}
                entry["objects"][role] = record
            if metadata:
                entry.update({key: metadata[key] for key in ("model_id", "revision", "tokenizer_revision", "readout")})
                entry["spectral_test_selection"] = {key: metadata[key] for key in ("row_indices", "contexts")}
                entry["original_binding"] = metadata["binding"]
            else:
                # S3 is not part of the C7/D4 spectral or centering analysis.
                model_id = model_slug.replace("__", "/", 1)
                model = next(m for m in inputs["protocol"]["models"] if m["id"] == model_id)
                entry.update(model_id=model_id, revision=model["revision"],
                             tokenizer_revision=model["tokenizer_revision"], readout="final_token",
                             model_metadata_source="same pinned panel in historical_protocol",
                             spectral_test_selection=None)
            cells.append(entry)
        spectral_prefix = input_name.rsplit("/", 1)[0]
        output_manifest = json.loads(followup.read(spectral_prefix + "/output_manifest.json"))
        plan_raw = followup.read("alternatives/plan.json")
        plan = json.loads(plan_raw)
        rank_name = next(name for name in followup.namelist()
                         if name.startswith("alternatives/reports/") and name.endswith("/descriptive_ranks.json"))
        ranks_raw = followup.read(rank_name)
        rank_data = json.loads(ranks_raw)["cells"]
        anchors = {"schema_version": 1, "archive_sha256": ARCHIVE_HASHES,
                   "historical_protocol": inputs["protocol"], "alt_plan": {"subgroups": plan["subgroups"]},
                   "original_ranks": rank_data, "semantic_baselines": {}, "baselines": {}, "source_bindings": []}
        for cell in spectral_cells.values():
            relative = "cells/" + cell["cell"] + "/summary.json"
            member = spectral_prefix + "/" + relative
            raw = followup.read(member)
            if digest(raw) != output_manifest["files"][relative]:
                raise ValueError("Semantic anchor member identity differs")
            saved = json.loads(raw)
            anchors["semantic_baselines"][cell["cell"]] = project_anchor(saved)
            anchors["source_bindings"].append({"role": "semantic_anchor", "cell": cell["cell"],
                                               "source_member": member, "source_member_sha256": digest(raw),
                                               "selected_subtree": "."})
        for unit in plan["units"]:
            receipt = json.loads(followup.read("alternatives/units/" + unit["unit"] + "/receipt.json"))
            member = "alternatives/" + receipt["attempt"] + "/summary.json"
            raw = followup.read(member)
            if digest(raw) != receipt["files"]["summary.json"]:
                raise ValueError("Image anchor member identity differs")
            saved = json.loads(raw)["test_bank_spectral"]
            anchors["baselines"][unit["unit"]] = project_anchor(saved)
            anchors["source_bindings"].append({"role": "image_anchor", "unit": unit["unit"],
                                               "source_member": member, "source_member_sha256": digest(raw),
                                               "selected_subtree": "test_bank_spectral"})
        if len(anchors["baselines"]) != 2952:
            raise ValueError("Expected 2952 image anchors")
        anchors_record = write("data/retained_coordinates/centering_anchors.json", encode(anchors))
        provenance = {"archive_sha256": ARCHIVE_HASHES, "input_manifest_source_member": input_name,
                      "input_manifest_source_sha256": digest(input_raw),
                      "alternative_plan_source_sha256": digest(plan_raw),
                      "rank_source_member": rank_name, "rank_source_sha256": digest(ranks_raw),
                      "centering_anchors": anchors_record,
                      "anchor_export": "Selected existing JSON fields; no new numerical analysis or float calculations."}
        manifest = {"schema_version": 1, "scope": "Original saved training-PCA coordinates, not raw activations.",
                    "cells": cells, "source_provenance": provenance}
        write("data/retained_coordinates/manifest.json", encode(manifest))
        cell_lookup = {cell["cell"]: cell for cell in cells}
        # Preserve the original 48-cell iteration order as well as row order.
        selected = {**manifest, "cells": [cell_lookup[cell["cell"]] for cell in inputs["cells"]]}
        if len(selected["cells"]) != 48:
            raise ValueError("Expected 48-cell anonymous subset")
        write("data/retained_coordinates/manifest_c7_d4.json", encode(selected))
        write("studies/template_centering_v1/historical_protocol.json", encode(inputs["protocol"]))

    derivatives = {}
    for form, source_dir, script, files in (
        ("rewhitened", args.within_archive, "diagnose.py", ("cell_results.csv", "image_results.csv", "summary.json")),
        ("three_way", args.fixed_archive, "diagnose_fixed_metric.py", ("three_way_cell_results.csv", "three_way_image_results.csv", "summary.json")),
    ):
        receipt_raw = (source_dir / "archive_record.json").read_bytes()
        receipt = json.loads(receipt_raw)
        for name in files:
            raw = (source_dir / name).read_bytes()
            if digest(raw) != receipt["archived_files_sha256"][name]:
                raise ValueError("Saved diagnostic identity differs: " + name)
            write("results/template_centering_v1/" + form + "/" + name, raw)
        write("results/template_centering_v1/" + form + "/archive_record.json", receipt_raw)
        original = (source_dir / script).read_bytes()
        if digest(original) != receipt["archived_files_sha256"][script]:
            raise ValueError("Original diagnostic source identity differs")
        write("studies/template_centering_v1/historical_sources/" + script, original)
        text = original.decode()
        replacements = [
            ("ROOT = Path.cwd()", "ROOT = Path(__file__).resolve().parents[2]"),
            ("OUT = Path(__file__).resolve().parent", "OUT = Path(__import__('os').environ['BSG_CENTERING_OUTPUT'])"),
            ('SNAPSHOT = ROOT / "dimension_rank_revision_20261105/offline_dimension_execution_v1/analysis/source_snapshot"',
             'SNAPSHOT = ROOT / "studies/spectral_dimension_sensitivity_v1"'),
            ("from archive_inputs import Study2Inputs", "from retained_inputs import Study2Inputs"),
            ('(SNAPSHOT / "historical_protocol.json")', '(Path(__file__).resolve().parent / "historical_protocol.json")'),
            ('(SNAPSHOT/"historical_protocol.json")', '(Path(__file__).resolve().parent/"historical_protocol.json")'),
        ]
        if form == "three_way":
            replacements.append(('ARCHIVED = ROOT / "work/study2-within-template-20261110/archive"',
                                 'ARCHIVED = ROOT / "results/template_centering_v1/rewhitened"'))
        applied = []
        for before, after in replacements:
            if before in text:
                if text.count(before) != 1:
                    raise ValueError("Unexpected source adaptation count")
                text = text.replace(before, after)
                applied.append({"before": before, "after": after})
        filename = "diagnose_rewhitened.py" if form == "rewhitened" else "diagnose_fixed_metric.py"
        record = write("studies/template_centering_v1/" + filename, text.encode())
        derivatives[filename] = {"original_source_sha256": digest(original),
                                 "publication_source_sha256": record["sha256"],
                                 "changes": applied, "mathematical_statements_changed": False,
                                 "original_source_distributed": True,
                                 "original_source_path": "historical_sources/" + script}
    write("studies/template_centering_v1/PUBLICATION_DERIVATIVES.json", encode(derivatives))
    allowed = {record["path"]: record["sha256"] for cell in selected["cells"]
               for record in cell["objects"].values()}
    allowed[anchors_record["path"]] = anchors_record["sha256"]
    subset_path = "data/retained_coordinates/manifest_c7_d4.json"
    allowed[subset_path] = file_digest(repo / subset_path)
    # Documentation is added by the package author after this data export.
    for relative in ("data/retained_coordinates/README.md",):
        if (repo / relative).is_file():
            allowed[relative] = file_digest(repo / relative)
    write("data/retained_coordinates/anonymous_files_c7_d4.json",
          encode({"files": sorted(allowed), "sha256": allowed,
                  "scope": "48 C7/D4 cells only; S3 objects and all-cell manifest excluded."}))
    print(json.dumps({"cells": len(cells), "c7_d4_cells": len(selected["cells"]),
                      "objects": len(objects), "object_bytes": sum(r["bytes"] for r in objects.values()),
                      "anonymous_object_bytes": sum((repo / p).stat().st_size for p in allowed),
                      "anchor_bytes": anchors_record["bytes"], "matrix_analysis_performed": False}))


if __name__ == "__main__":
    main()
