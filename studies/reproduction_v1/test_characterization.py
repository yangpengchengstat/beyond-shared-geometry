"""Model-free regression checks for the fresh characterization adapter."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fresh_characterization", HERE / "characterization.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
c.configure_environment()
c.numerical_imports()
import numpy as np


class FakeRuntime:
    """Never imports Torch or a model; deterministic row-pooled arrays."""
    calls = 0
    def identity(self):
        return {"model_id": "offline-fixture", "revision": "fixture-v1", "batch_size": 8}
    telemetry = []
    def features(self, records, layers, readout):
        for start in range(0, len(records), 8):
            batch = records[start:start+8]
            FakeRuntime.calls += 1
            values = np.array([[int(hashlib.sha256(r["row_id"].encode()).hexdigest()[:6], 16) / 2**24,
                                r["state_id"], r["paraphrase_id"]] for r in batch], dtype=np.float32)
            yield list(range(start, start+len(batch))), {layer: values + layer for layer in layers}
    def input_entity_features(self, records):
        return np.array([[r["state_id"], 1, 2] for r in records], dtype=np.float32)
    def close(self):
        pass


class CharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = c.config()
        cls.banks = c.generate_banks(cls.cfg)

    def test_frozen_dependency_unchanged(self):
        self.assertEqual(c.verify_frozen(), c.EXPECTED_FROZEN_MANIFEST)

    def test_exact_prompt_population_hashes(self):
        from bsg_next.data import dataset_identity
        for domain, bank in self.banks.items():
            with self.subTest(domain=domain):
                self.assertEqual(dataset_identity(bank["all"]), self.cfg["dataset_seals"][domain]["dataset_sha256"])
                self.assertEqual(tuple(len(bank[k]) for k in ("all", "development", "heldout")), c.DOMAIN_COUNTS[domain])
                self.assertEqual(len(bank["development"]) % 8, 0)

    def test_s3_reserved_and_support_roles(self):
        from collections import Counter
        dev = Counter(r["split"] for r in self.banks["S3"]["development"])
        self.assertEqual(dev, {"train": 13824, "validation": 1536, "reserved": 9216, "pilot": 1152})
        held = Counter(r["split"] for r in self.banks["S3"]["heldout"])
        self.assertEqual(dict(held), self.cfg["S3_new_row_breakdown"])
        self.assertTrue(all(r["split"] != "reserved" for r in self.banks["S3"]["heldout"]))

    def test_technical_limit_is_frozen(self):
        full = c.make_contract(self.cfg)
        limited = c.make_contract(self.cfg, 1)
        self.assertEqual(full["planned_cells"], 72)
        self.assertFalse(full["technical_test"])
        self.assertTrue(limited["technical_test"])
        self.assertFalse(limited["paper_eligible"])
        self.assertFalse(limited["independent_holdout"])
        for value in (0, 72, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                c.make_contract(self.cfg, value)

    def test_contract_change_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "fresh"
            with c.output_lock(out):
                c.bind_contract(out, c.make_contract(self.cfg, 1))
                with self.assertRaisesRegex(ValueError, "contract changed"):
                    c.bind_contract(out, c.make_contract(self.cfg, 2))

    def test_nonempty_unbound_output_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "old.json").write_text("{}")
            with c.output_lock(out), self.assertRaisesRegex(ValueError, "Nonempty"):
                c.bind_contract(out, c.make_contract(self.cfg, 1))

    def test_single_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with c.output_lock(out), self.assertRaisesRegex(ValueError, "Another"):
                with c.output_lock(out):
                    pass

    def test_prepare_never_imports_torch_or_hf(self):
        import sys
        before = set(sys.modules)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            contract = c.make_contract(self.cfg, 1)
            c.prepare(out, self.cfg, contract)
            for domain in c.DOMAIN_COUNTS:
                seal = c.read(out / "datasets" / domain / "manifest.json")
                self.assertFalse(seal["metadata"]["new_independent_holdout"])
            c.prepare(out, self.cfg, contract)
        added = set(sys.modules) - before
        self.assertFalse(any(name == "torch" or name.startswith("torch.") or name == "transformers" or name.startswith("transformers.") for name in added))

    def test_complete_receipt_verifies_data_and_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary) / "cell"
            binding = {"contract": "one", "input": "sha"}
            calls = []
            def builder(stage):
                calls.append(1)
                c.atomic(stage / "result.json", {"value": 1})
            self.assertTrue(c.transactional_directory(cell, "complete.json", binding, builder))
            self.assertFalse(c.transactional_directory(cell, "complete.json", binding, builder))
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(ValueError, "another"):
                c.transactional_directory(cell, "complete.json", {"input": "different"}, builder)
            c.atomic(cell / "result.json", {"value": 2})
            with self.assertRaisesRegex(ValueError, "changed"):
                c.transactional_directory(cell, "complete.json", binding, builder)

    def test_missing_receipt_never_reuses_existing_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary) / "cell"
            cell.mkdir()
            c.atomic(cell / "legacy.json", {"untrusted_incomplete": True})
            def builder(stage):
                c.atomic(stage / "legacy.json", {"new": True})
            c.transactional_directory(cell, "complete.json", {"contract": 1}, builder)
            self.assertEqual(c.read(cell / "legacy.json"), {"new": True})
            preserved = list(Path(temporary).glob(".cell.incomplete-*"))
            self.assertEqual(len(preserved), 1)
            self.assertEqual(c.read(preserved[0] / "legacy.json"), {"untrusted_incomplete": True})

    def test_failed_builder_never_publishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary) / "cell"
            def builder(stage):
                c.atomic(stage / "legacy.json", {"partial": True})
                raise RuntimeError("interrupted")
            with self.assertRaises(RuntimeError):
                c.transactional_directory(cell, "complete.json", {}, builder)
            self.assertFalse(cell.exists())
            self.assertFalse(any(Path(temporary).rglob("complete.json")))

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("../elsewhere", "/absolute"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    c.safe_file(root, name)

    def test_direct_cache_matches_separate_phase_batches_and_resumes(self):
        from bsg_next.extraction import extract_features, read_cache
        bank = self.banks["D4"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = FakeRuntime()
            FakeRuntime.calls = 0
            extract_features(runtime, bank["merged"], [1, 2], "span_mean", root / "merged")
            first_calls = FakeRuntime.calls
            extract_features(runtime, bank["merged"], [1, 2], "span_mean", root / "merged")
            self.assertEqual(first_calls, FakeRuntime.calls)
            for phase in ("development", "heldout"):
                extract_features(runtime, bank[phase], [1, 2], "span_mean", root / phase)
            merged = read_cache(root / "merged")[1]
            dev = read_cache(root / "development")[1]
            held = read_cache(root / "heldout")[1]
            for layer in (1, 2):
                np.testing.assert_array_equal(merged[layer], np.concatenate([dev[layer], held[layer]]))
            array = np.load(root / "merged" / "layer_1.npy", mmap_mode="r+")
            array[0, 0] += 1
            array.flush()
            with self.assertRaisesRegex(ValueError, "checksum"):
                extract_features(runtime, bank["merged"], [1, 2], "span_mean", root / "merged")

    def test_raw_disk_estimate_avoids_duplicate_caches(self):
        estimate = c.expected_raw_bytes(self.cfg, c.planned_cells(self.cfg), self.banks)
        self.assertEqual(estimate, 4 * 16384 * (3080 * 4 + 66048 * 7))
        self.assertLess(estimate / 2**30, 30)

    def test_shared_export_roundtrip_and_integrity(self):
        rows = [dict(row) for row in self.banks["S3"]["merged"][:48]]
        # Small complete fixture; deterministic role-input embeddings repeat by orbit/state.
        values = np.array([[r["state_id"], 1, 2] for r in rows], np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = c.export_shared(root, "fixture", rows, values, "cache-sha", "contract")
            with np.load(shared / "potential_unique.npz", allow_pickle=False) as arrays:
                np.testing.assert_array_equal(values, arrays["values"][arrays["inverse"]])
            c.export_shared(root, "fixture", rows, values, "cache-sha", "contract")
            with self.assertRaises(ValueError):
                c.export_shared(root, "fixture", rows, values, "different-cache", "contract")

    def test_collect_excludes_raw_and_incomplete_files(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            c.atomic(root / "analysis" / "valid" / "result.json", {"ok": True})
            c.atomic(root / "analysis" / ".valid.building-x" / "result.json", {"partial": True})
            c.atomic(root / "merged" / "cache" / "raw.json", {"raw": True})
            archive = c.collect(root)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.namelist(), ["run/analysis/valid/result.json"])
            self.assertTrue(archive.with_name(archive.name + ".sha256").exists())


if __name__ == "__main__":
    unittest.main()
