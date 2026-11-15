"""Offline import regressions; no model backend or incoming code is executed."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "characterization_import_under_test",
    ROOT / "studies/characterization_v1/tools/import_results.py")
IMPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IMPORTER
SPEC.loader.exec_module(IMPORTER)
CFG = {"test_context_families": 64, "test_entity_orbits": 16}


def _reject(call, message):
    try:
        call()
    except IMPORTER.ValidationError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("Invalid fixture was accepted: " + message)


@lru_cache(maxsize=1)
def _historical_rows():
    # Independently classify the full crossed population, then apply the frozen
    # development/heldout selection. Do not call the bank validator to build it.
    rows = []
    def append(context, fi, entity, oi, split):
        family = f"structural_{context}_{fi:03d}"
        orbit = f"{'pilot' if entity == 'pilot' else 'main'}_triple_{entity}_{oi:03d}"
        for para in range(2):
            for state in range(6):
                rows.append({"row_id": f"S3_role/{family}/p{para}/{orbit}/s{state}",
                             "family_id": family, "orbit_id": orbit,
                             "paraphrase_id": para, "state_id": state,
                             "split": split, "domain": "S3_role"})
    for context, nf in (("train", 48), ("validation", 16), ("test", 64)):
        for entity, no in (("train", 24), ("validation", 8), ("test", 16)):
            split = context if context == entity and context != "test" else "reserved"
            if context == entity == "test": split = "test_joint"
            if context == "test" and entity == "train": split = "test_context"
            if context == "train" and entity == "test": split = "test_entity"
            development = context != "test" and entity != "test"
            if not (development or split in ("test_joint", "test_context", "test_entity")):
                continue
            for fi in range(nf):
                for oi in range(no): append(context, fi, entity, oi, split)
    for fi in range(12):
        for oi in range(8): append("pilot", fi, "pilot", oi, "pilot")
    for fi in range(2):
        for oi in range(16): append("support", fi, "test", oi, "support")
    return tuple(rows)


def test_exact_historical_s3_bank_accepts_only_the_frozen_cross_blocks():
    rows = _historical_rows()
    assert len(rows) == 66048
    assert Counter(r["split"] for r in rows) == {
        "train": 13824, "validation": 1536, "reserved": 9216, "pilot": 1152,
        "test_entity": 9216, "test_context": 18432, "test_joint": 12288, "support": 384}
    IMPORTER.validate_s3_frozen_row_bank(rows, CFG)


def test_exact_bank_rejects_missing_extra_duplicate_and_unplanned_reserved_rows():
    rows = list(_historical_rows())
    reserved = next(i for i, r in enumerate(rows) if r["split"] == "reserved")
    variants = [rows[:reserved] + rows[reserved+1:], rows + [dict(rows[reserved])]]
    extra = dict(rows[reserved], row_id="S3_role/unplanned", family_id="structural_test_999")
    variants.append(rows + [extra])
    for variant in variants:
        _reject(lambda: IMPORTER.validate_s3_frozen_row_bank(variant, CFG), "metadata bank differs")


def test_exact_bank_rejects_relabelled_reserved_or_invalid_family_and_identity():
    rows = list(_historical_rows())
    i = next(i for i, r in enumerate(rows) if r["split"] == "reserved")
    for change in ({"split": "train"}, {"family_id": "structural_train_999"},
                   {"row_id": "different-id"}, {"orbit_id": "main_triple_validation_999"},
                   {"state_id": 6}, {"paraphrase_id": 2}, {"domain": "C7"}):
        variant = rows.copy(); variant[i] = dict(variant[i], **change)
        _reject(lambda: IMPORTER.validate_s3_frozen_row_bank(variant, CFG), "metadata bank differs")


def _zip(entries):
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, data in entries: archive.writestr(name, data)
    buffer.seek(0)
    return buffer


def test_zip_rejects_traversal_absolute_duplicate_special_and_colliding_members():
    for name in ("../escape.json", "/absolute.json", "C:/drive.json", "a//b.json", "a\\b.json"):
        with zipfile.ZipFile(_zip([(name, b"{}")])) as archive:
            try: IMPORTER.inspect_zip(archive)
            except IMPORTER.ValidationError: pass
            else: raise AssertionError("Unsafe path accepted: " + name)
    for entries, message in (([("a", b"x"), ("a", b"y")], "Duplicate ZIP"),
                             ([("a", b"x"), ("a/b", b"y")], "collision")):
        with zipfile.ZipFile(_zip(entries)) as archive:
            _reject(lambda: IMPORTER.inspect_zip(archive), message)
    link = zipfile.ZipInfo("link"); link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(_zip([(link, b"target")])) as archive:
        _reject(lambda: IMPORTER.inspect_zip(archive), "Symlink/special")


def test_zip_rejects_encryption_and_size_budgets():
    with zipfile.ZipFile(_zip([("a", b"1234")])) as archive:
        _reject(lambda: IMPORTER.inspect_zip(archive, max_total=3), "total-size")
        _reject(lambda: IMPORTER.inspect_zip(archive, max_file=3), "member-size")
        _reject(lambda: IMPORTER.inspect_zip(archive, max_files=0), "member-count")
        archive.infolist()[0].flag_bits |= 1
        _reject(lambda: IMPORTER.inspect_zip(archive), "Encrypted")


def test_sidecar_requires_exact_filename_hash_and_one_entry():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve(); archive = root / "data.zip"; sidecar = root / "data.zip.sha256"
        archive.write_bytes(b"transport fixture")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        sidecar.write_text(digest + "  data.zip\n")
        assert IMPORTER.check_sidecar(archive, sidecar) == digest
        for content, message in ((digest + "  other.zip\n", "filename differs"),
                                 ("0" * 64 + "  data.zip\n", "SHA256 mismatch"),
                                 (digest + "  data.zip\n" + digest + "  data.zip\n", "exactly one")):
            sidecar.write_text(content)
            _reject(lambda: IMPORTER.check_sidecar(archive, sidecar), message)


def test_path_boundary_rejects_outside_and_symlink_inputs():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve(); project = root / "project"; project.mkdir()
        outside = root / "outside"; outside.write_text("not an authorized input")
        _reject(lambda: IMPORTER.boundary(outside, project), "outside")
        link = project / "link"; link.symlink_to(outside)
        _reject(lambda: IMPORTER.boundary(link, project), "Symlinks")


def test_receipt_namespace_never_overwrites_existing_files():
    # Stub scientific validation only: this tests the independent atomic receipt
    # publication contract using a tiny synthetic payload, not model evidence.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve(); archive = root / "data.zip"; sidecar = root / "data.zip.sha256"
        payload = b'{"status":"fixture"}'
        archive.write_bytes(_zip([(IMPORTER.RESULT_PREFIX + "summary.json", payload)]).getvalue())
        sidecar.write_text(IMPORTER.sha_file(archive) + "  data.zip\n")
        source = root / "source"; source.mkdir()
        def report(*args, **kwargs):
            return {"panel_validation": "complete", "member_sha256": {"summary.json": IMPORTER.sha_bytes(payload)},
                    "run_recorded_status": "fixture", "counts": {}, "missing_or_incomplete": [], "warnings": [],
                    "paper_eligible": False, "automatic_manuscript_promotion": False}
        frozen = IMPORTER.Frozen({}, {}, "a" * 64, "b" * 64)
        with patch.object(IMPORTER, "STUDY_REL", Path("study")), \
             patch.object(IMPORTER, "load_frozen", return_value=frozen), \
             patch.object(IMPORTER, "validate_result", side_effect=report):
            result = IMPORTER.import_validated(archive, sidecar, source, receipt_name="one", project=root)
            target = root / "study/imports/one"
            before = {str(p.relative_to(target)): p.read_bytes() for p in target.rglob("*") if p.is_file()}
            assert result["imported"] is True and result["paper_eligible"] is False
            assert (target / "raw/summary.json").read_bytes() == payload
            _reject(lambda: IMPORTER.import_validated(archive, sidecar, source, receipt_name="one", project=root), "overwriting is forbidden")
            after = {str(p.relative_to(target)): p.read_bytes() for p in target.rglob("*") if p.is_file()}
            assert after == before
            (root / "study/imports/empty").mkdir()
            _reject(lambda: IMPORTER.import_validated(archive, sidecar, source, receipt_name="empty", project=root), "overwriting is forbidden")


def test_result_crc_corruption_is_rejected_before_contract_parsing():
    raw = bytearray(_zip([(IMPORTER.RESULT_PREFIX + "summary.json", b"crc-payload")]).getvalue())
    raw[raw.index(b"crc-payload")] ^= 1
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "corrupt.zip"; archive.write_bytes(raw)
        frozen = IMPORTER.Frozen({}, {}, "a" * 64, "b" * 64)
        _reject(lambda: IMPORTER.validate_result(archive, frozen), "CRC mismatch")


def test_untrusted_json_and_object_numpy_headers_cannot_execute_payloads():
    _reject(lambda: IMPORTER.parse_json(b'{"a":1,"a":2}'), "Duplicate JSON key")
    _reject(lambda: IMPORTER.parse_json(b'{"a":NaN}'), "Nonfinite JSON")
    import numpy as np
    buffer = io.BytesIO(); np.save(buffer, np.array([None], dtype=object))
    _reject(lambda: IMPORTER.npy_header(buffer.getvalue()), "Object/structured/unsupported")
