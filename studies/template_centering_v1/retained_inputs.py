"""Public retained-coordinate reader. Verification uses only the standard library.

NumPy is imported only by the explicitly requested load_test_bank method.
This reader authenticates distributed saved inputs, not original model forwards.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_file(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or "\\" in relative:
        raise ValueError("Unsafe publication input path")
    path = root.joinpath(*name.parts)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Missing or symlink publication input: " + relative)
    return path


class Study2Inputs:
    def __init__(self, repo, historical_protocol):
        self.repo = Path(repo).resolve(strict=True)
        self.manifest_path = self.repo / "data/retained_coordinates/manifest_c7_d4.json"
        self.manifest = json.loads(self.manifest_path.read_bytes())
        self.cells = self.manifest["cells"]
        self.by_cell = {cell["cell"]: cell for cell in self.cells}
        self.historical_protocol = historical_protocol
        self.baselines = {}
        self.semantic_baselines = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def preflight(self):
        allowed_path = self.repo / "data/retained_coordinates/anonymous_files_c7_d4.json"
        allowed = json.loads(allowed_path.read_bytes())
        for relative in allowed["files"]:
            if sha256(safe_file(self.repo, relative)) != allowed["sha256"][relative]:
                raise ValueError("Publication input checksum differs: " + relative)
        provenance = self.manifest["source_provenance"]
        binding = provenance["centering_anchors"]
        anchor_path = safe_file(self.repo, binding["path"])
        if sha256(anchor_path) != binding["sha256"]:
            raise ValueError("Historical anchor projection checksum differs")
        anchors = json.loads(anchor_path.read_bytes())
        if anchors["historical_protocol"] != self.historical_protocol:
            raise ValueError("Historical protocol differs")
        if len(self.cells) != 48 or len(self.by_cell) != 48:
            raise ValueError("Expected all 48 C7/D4 model/readout cells")
        if {domain: sum(cell["domain"] == domain for cell in self.cells)
                for domain in ("C7", "D4")} != {"C7": 24, "D4": 24}:
            raise ValueError("Fixed panel coverage differs")
        checked = set()
        for cell in self.cells:
            if cell["readout"] != "span_mean":
                raise ValueError("Unexpected spectral readout")
            for obj in cell["objects"].values():
                if obj["sha256"] != obj["source_member_sha256"] or not obj["byte_preserved"]:
                    raise ValueError("Coordinate object is not an original-byte export")
                path = safe_file(self.repo, obj["path"])
                if obj["path"] not in checked and sha256(path) != obj["sha256"]:
                    raise ValueError("Retained-coordinate object checksum differs")
                checked.add(obj["path"])
            parsed = json.loads(safe_file(self.repo, cell["objects"]["rows"]["path"]).read_bytes())
            rows = parsed["rows"] if isinstance(parsed, dict) else parsed
            selection = cell["spectral_test_selection"]
            n = 7 if cell["domain"] == "C7" else 4
            if len(selection["row_indices"]) != 128 or len(selection["contexts"]) != 128:
                raise ValueError("Expected 128 complete test blocks")
            selected = []
            for block, (indices, context) in enumerate(zip(selection["row_indices"], selection["contexts"])):
                if len(indices) != n or context["state_ids"] != list(range(n)):
                    raise ValueError("Incomplete or reordered state block")
                for state, index in enumerate(indices):
                    row = rows[index]
                    expected = {"row_id": context["row_ids"][state], "state_id": state,
                                "family_id": "structural_test_%03d" % (block // 2),
                                "paraphrase_id": block % 2, "split": "test"}
                    if any(row.get(key) != value for key, value in expected.items()):
                        raise ValueError("Saved row/state/test selection differs")
                    selected.append(index)
            if len(set(selected)) != 128 * n:
                raise ValueError("Reused source rows within a test bank")
        self.alt_plan = anchors["alt_plan"]
        self.original_ranks = anchors["original_ranks"]
        self.baselines = anchors["baselines"]
        self.semantic_baselines = anchors["semantic_baselines"]
        if len(self.baselines) != 2952 or len(self.semantic_baselines) != 48:
            raise ValueError("Historical anchor coverage differs")
        for domain, count in (("C7", 120), ("D4", 3)):
            if len(self.alt_plan["subgroups"][domain]) != count:
                raise ValueError("Image enumeration differs")
        return {"status": "publication_inputs_verified", "cells_verified": 48,
                "archive_sha256": anchors["archive_sha256"],
                "committed_alternative_anchors_verified": len(self.baselines),
                "coordinate_values_analyzed": False,
                "scope": "Saved byte hashes and row bindings, not model-forward or PCA refit verification."}

    def load_test_bank(self, cell_id):
        import numpy as np
        cell = self.by_cell[cell_id]
        record = cell["objects"]["coordinates"]
        raw = safe_file(self.repo, record["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Coordinate object changed after preflight")
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            bank = arrays["z"][np.asarray(cell["spectral_test_selection"]["row_indices"], dtype=int)].copy()
        if bank.shape != (128, 7 if cell["domain"] == "C7" else 4, 16):
            raise ValueError("Retained test-bank shape differs")
        return bank
