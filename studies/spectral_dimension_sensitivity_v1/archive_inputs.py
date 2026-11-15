"""Immutable Study 2 archive reader. Metadata preflight is standard-library only.

No numerical arrays are materialized by preflight(); load_test_bank() imports
NumPy lazily and is called only by the explicit future analysis command.
"""
from __future__ import annotations
import ast
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import struct
import zipfile

ARCHIVES = {
    "parent": ("incoming/rank_recovery_v1/from-zero-v1-rank-recovery.results.zip",
               "6a4c38237bf1368d4c77c97504f666d0d933bfebed9417d173fbf7441c0fecef"),
    "followup": ("incoming/tmlr_all_v2/tmlr-all-v2.20261030T203725Z-5489fd.results.zip",
                 "ef87e9aaeb8f093ba0c2438590d6a30b5dffcd256993e2366bd600cca3cb5bbb"),
}
MAX_MEMBER = 32 * 1024 * 1024


class InputError(ValueError):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def parse_json(raw):
    def pairs(items):
        answer = {}
        for key, value in items:
            if key in answer:
                raise InputError("duplicate JSON key")
            answer[key] = value
        return answer
    def invalid(value):
        raise InputError("nonfinite JSON constant: " + value)
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise InputError("nonfinite JSON number")
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=number)


def safe_name(name):
    p = PurePosixPath(name)
    if not name or name.startswith("/") or "\\" in name or any(x in (".", "..", "") for x in name.split("/")) or ":" in p.parts[0]:
        raise InputError("unsafe archive/source name: " + repr(name))
    return name


def npz_headers(raw):
    """NPY headers, sizes and hashes only; deliberately does not import NumPy."""
    output = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries = z.infolist()
        if len(entries) > 16 or len({e.filename for e in entries}) != len(entries):
            raise InputError("unexpected NPZ entry count/duplicates")
        for entry in entries:
            name = safe_name(entry.filename)
            if "/" in name or not name.endswith(".npy") or entry.file_size > MAX_MEMBER:
                raise InputError("unexpected NPZ member")
            data = z.read(entry)
            stream = io.BytesIO(data)
            if stream.read(6) != b"\x93NUMPY":
                raise InputError("not an NPY header")
            version = tuple(stream.read(2))
            if version == (1, 0):
                length = struct.unpack("<H", stream.read(2))[0]
            elif version in ((2, 0), (3, 0)):
                length = struct.unpack("<I", stream.read(4))[0]
            else:
                raise InputError("unsupported NPY version")
            if length > 16384:
                raise InputError("oversized NPY header")
            header = ast.literal_eval(stream.read(length).decode("utf-8" if version == (3, 0) else "latin1"))
            shape = header.get("shape")
            dtype = header.get("descr")
            if not isinstance(shape, tuple) or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in shape):
                raise InputError("bad NPY shape")
            if dtype != "<f8" or not isinstance(header.get("fortran_order"), bool):
                raise InputError("expected original little-endian float64 NPY")
            count = math.prod(shape)
            if count * 8 != len(data) - stream.tell():
                raise InputError("NPY size/header mismatch")
            output[name[:-4]] = {"shape": list(shape), "dtype": dtype,
                                "fortran_order": header["fortran_order"], "npy_sha256": sha(data)}
    return output


class Reader:
    def __init__(self, path, expected_hash):
        self.path = Path(path).resolve(strict=True)
        if not self.path.is_file():
            raise InputError("archive missing")
        self.sha256 = file_sha(self.path)
        if self.sha256 != expected_hash:
            raise InputError("archive SHA256 differs: " + str(self.path))
        self.z = zipfile.ZipFile(self.path)
        infos = self.z.infolist()
        if len(infos) > 1_000_000 or len({i.filename for i in infos}) != len(infos):
            self.z.close()
            raise InputError("archive duplicate/member-count contract")
        self.index = {i.filename: i for i in infos}
        self.used = {}

    def close(self):
        self.z.close()

    def read(self, name, expected=None):
        safe_name(name)
        info = self.index.get(name)
        if info is None or info.is_dir() or info.file_size > MAX_MEMBER:
            raise InputError("missing/oversized archive member: " + name)
        if info.flag_bits & 1 or ((info.external_attr >> 16) & 0o170000) == 0o120000:
            raise InputError("encrypted or symlink archive member")
        raw = self.z.read(info)
        digest = sha(raw)
        if expected is not None and digest != expected:
            raise InputError("member hash differs: " + name)
        self.used[name] = digest
        return raw

    def json(self, name, expected=None):
        return parse_json(self.read(name, expected))

    def unique_suffix(self, prefix, suffix):
        matches = [n for n in self.index if n.startswith(prefix) and n.endswith(suffix)]
        if len(matches) != 1:
            raise InputError("ambiguous archive member: " + prefix + "..." + suffix)
        return matches[0]


class Study2Inputs:
    """Binds the original 48 cells, complete test orbits and all old receipts."""
    def __init__(self, repo, historical_protocol):
        self.repo = Path(repo).resolve(strict=True)
        self.parent = Reader(self.repo / ARCHIVES["parent"][0], ARCHIVES["parent"][1])
        try:
            self.followup = Reader(self.repo / ARCHIVES["followup"][0], ARCHIVES["followup"][1])
        except Exception:
            self.parent.close()
            raise
        self.historical_protocol = historical_protocol
        self.manifest_name = self.followup.unique_suffix("spectral-results-", "/input_manifest.json")
        self.prefix = self.manifest_name.rsplit("/", 1)[0]
        self.manifest = self.followup.json(self.manifest_name)
        if self.manifest.get("archive_sha256") != self.parent.sha256 or self.manifest.get("protocol") != historical_protocol:
            raise InputError("historical protocol/archive binding mismatch")
        self.output_manifest = self.followup.json(self.prefix + "/output_manifest.json")
        self.alt_plan_raw = self.followup.read("alternatives/plan.json")
        self.alt_plan = parse_json(self.alt_plan_raw)
        self.alt_run_raw = self.followup.read("alternatives/run_manifest.json")
        self.alt_binding = {"plan_sha256": sha(self.alt_plan_raw), "run_manifest_sha256": sha(self.alt_run_raw)}
        rank_name = self.followup.unique_suffix("alternatives/reports/", "/descriptive_ranks.json")
        self.original_ranks = self.followup.json(rank_name)["cells"]
        self.cells = self.manifest.get("cells", [])
        self.by_cell = {c["cell"]: c for c in self.cells}
        self.baselines = {}
        self.semantic_baselines = {}
        self.cell_metadata = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.parent.close()
        self.followup.close()

    def preflight(self):
        self.baselines.clear()
        self.semantic_baselines.clear()
        self.cell_metadata.clear()
        p = self.historical_protocol
        expected = {(m["id"], layer, domain): m for m in p["models"] for layer in m["layers"] for domain in p["domains"]}
        if len(expected) != 48 or len(self.cells) != 48 or len(self.by_cell) != 48:
            raise InputError("48-cell coverage differs")
        actual = {(c["model_id"], c["layer"], c["domain"]) for c in self.cells}
        if actual != set(expected):
            raise InputError("model/layer/domain panel differs")
        if set(self.original_ranks) != set(self.by_cell) or any(v.get("status") != "complete_descriptive" for v in self.original_ranks.values()):
            raise InputError("original descriptive-rank anchor coverage differs")
        if set(self.alt_plan.get("subgroups", {})) != {"C7", "D4"}:
            raise InputError("alternative domains differ")
        for domain, count in [("C7", 120), ("D4", 3)]:
            groups = self.alt_plan["subgroups"][domain]
            if len(groups) != count or len({g["id"] for g in groups}) != count or sum(g["semantic"] for g in groups) != 1:
                raise InputError("alternative image coverage differs")
        expected_units = {(c["cell"], g["id"]) for c in self.cells for g in self.alt_plan["subgroups"][c["domain"]]}
        units = self.alt_plan.get("units", [])
        if len(units) != 2952 or {(u["cell"], u["subgroup_id"]) for u in units} != expected_units:
            raise InputError("alternative unit coverage differs")
        for c in self.cells:
            model = expected[c["model_id"], c["layer"], c["domain"]]
            if c["revision"] != model["revision"] or c["tokenizer_revision"] != model["tokenizer_revision"]:
                raise InputError("model/tokenizer revision differs")
            if c["partition"] != "test" or c["readout"] != "span_mean":
                raise InputError("readout/partition differs")
            base = "parent-from-zero-v1/characterization/analysis/" + c["cell"]
            names = {k: base + "/" + k for k in ("coordinates.npz", "pca.npz", "rows.json", "cell_complete.json")}
            if set(c["members"]) != set(names.values()):
                raise InputError("cell member coverage differs")
            raw = {key: self.parent.read(name, c["members"][name]) for key, name in names.items()}
            coordinate_schema = npz_headers(raw["coordinates.npz"])
            basis_schema = npz_headers(raw["pca.npz"])
            if coordinate_schema != c["coordinate_schema"] or basis_schema != c["pca_schema"]:
                raise InputError("coordinate/PCA header identity differs")
            if set(coordinate_schema) != {"z"} or coordinate_schema["z"]["shape"][1:] != [16]:
                raise InputError("original coordinates are not 16D")
            if basis_schema["basis"]["shape"] != [model["hidden_dimension"], 16] or basis_schema["center"]["shape"] != [model["hidden_dimension"]]:
                raise InputError("original PCA shape differs")
            row_records = parse_json(raw["rows.json"])["rows"]
            if len(row_records) != coordinate_schema["z"]["shape"][0]:
                raise InputError("row metadata length differs")
            selected = c["row_indices"]
            contexts = c["contexts"]
            n = 7 if c["domain"] == "C7" else 4
            if len(selected) != 128 or len(contexts) != 128:
                raise InputError("test context count differs")
            flattened = []
            for block_id, (indices, context) in enumerate(zip(selected, contexts)):
                family = "structural_test_%03d" % (block_id // 2)
                paraphrase = block_id % 2
                if len(indices) != n or context["state_ids"] != list(range(n)) or context["family_id"] != family or context["paraphrase_id"] != paraphrase or context["split"] != "test":
                    raise InputError("complete state/context block differs")
                if len(context["row_ids"]) != n:
                    raise InputError("state row IDs differ")
                for state, index in enumerate(indices):
                    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(row_records):
                        raise InputError("invalid source row index")
                    row = row_records[index]
                    expected_row = {"row_id": context["row_ids"][state], "state_id": state, "family_id": family,
                                    "paraphrase_id": paraphrase, "split": "test", "domain": c["archive_row_domain"], "orbit_id": context["orbit_id"]}
                    if any(row.get(k) != v for k, v in expected_row.items()):
                        raise InputError("parent row/context binding differs")
                    flattened.append(index)
            if len(set(flattened)) != 128 * n:
                raise InputError("reused source rows inside test bank")
            self.parent.read(c["merged_manifest"], c["merged_manifest_sha256"])
            summary_name = "cells/" + c["cell"] + "/summary.json"
            sem = self.followup.json(self.prefix + "/" + summary_name, self.output_manifest["files"][summary_name])
            if sem.get("status") != "complete" or sem.get("r") != 16 or sem.get("cell") != c["cell"]:
                raise InputError("original semantic anchor unavailable")
            self.semantic_baselines[c["cell"]] = sem
            self.cell_metadata.append({"cell": c["cell"], "model_id": c["model_id"], "domain": c["domain"], "layer": c["layer"],
                                       "source_dimension": 16, "test_shape": [128, n, 16], "coordinate_member": names["coordinates.npz"],
                                       "coordinate_sha256": c["members"][names["coordinates.npz"]], "pca_member": names["pca.npz"],
                                       "pca_sha256": c["members"][names["pca.npz"]], "source_rows_sha256": sha(canonical(selected)),
                                       "header_and_row_metadata_verified": True, "numerical_values_analyzed": False})
        for unit in units:
            cell = self.by_cell[unit["cell"]]
            group = next(g for g in self.alt_plan["subgroups"][cell["domain"]] if g["id"] == unit["subgroup_id"])
            if unit["unit"] != unit["cell"] + "--" + group["id"] or unit["domain"] != cell["domain"] or unit["semantic_registration"] != group["semantic"]:
                raise InputError("alternative unit metadata differs")
            rec = self.followup.json("alternatives/units/" + unit["unit"] + "/receipt.json")
            if rec.get("binding") != self.alt_binding or rec.get("unit") != unit["unit"] or rec.get("status") != "complete" or not rec.get("terminal"):
                raise InputError("committed alternative receipt differs")
            attempt = safe_name(rec["attempt"])
            if not attempt.startswith("attempts/" + unit["unit"] + "/"):
                raise InputError("alternative attempt binding differs")
            summary = self.followup.json("alternatives/" + attempt + "/summary.json", rec["files"]["summary.json"])
            if summary.get("unit") != unit["unit"] or summary.get("cell") != unit["cell"] or summary.get("subgroup_id") != group["id"] or summary.get("semantic_registration") != group["semantic"]:
                raise InputError("alternative summary identity differs")
            old = summary.get("test_bank_spectral", {})
            if old.get("status") != "complete" or old.get("r") != 16:
                raise InputError("original alternative spectral anchor unavailable")
            self.baselines[unit["unit"]] = old
        return {"status": "prepared_metadata_only", "coordinate_values_analyzed": False,
                "archive_sha256": {key: value[1] for key, value in ARCHIVES.items()}, "cells": self.cell_metadata,
                "cells_verified": len(self.cells), "committed_alternative_anchors_verified": len(self.baselines),
                "input_manifest_sha256": sha(canonical(self.manifest)), "alternative_plan_sha256": sha(self.alt_plan_raw),
                "used_member_sha256": {"parent": self.parent.used, "followup": self.followup.used},
                "higher_dimensions_available_from_this_input": False,
                "scope": "Header/identity verification, not rank/whitening/numerical outcome validation"}

    def load_test_bank(self, cell_id):
        """Future opt-in use only; no fitting or changed row order."""
        import numpy as np
        c = self.by_cell[cell_id]
        member = next(n for n in c["members"] if n.endswith("/coordinates.npz"))
        raw = self.parent.read(member, c["members"][member])
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            bank = arrays["z"][np.asarray(c["row_indices"], dtype=int)].copy()
        if bank.shape != (128, 7 if c["domain"] == "C7" else 4, 16):
            raise InputError("loaded test bank shape differs")
        return bank
