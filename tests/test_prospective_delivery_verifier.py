"""Synthetic publication-provenance tests; no scientific modules are imported."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prospective_delivery", ROOT / "scripts/verify_prospective_delivery.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PreservationChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        files = {}
        self.entries = {}
        for name in sorted(MODULE.ENGINEERING_PATHS):
            active = self.root / name
            active.parent.mkdir(parents=True, exist_ok=True)
            active.write_text("publication derivative")
            history = "release/history/frozen/" + name
            old = self.root / history
            old.parent.mkdir(parents=True, exist_ok=True)
            old.write_text("frozen execution source")
            files[name] = MODULE.digest(old)
            self.entries[name] = {"execution_source": history,
                                  "execution_sha256": files[name],
                                  "publication_sha256": MODULE.digest(active),
                                  "reason": "synthetic packaging fixture"}
        for name in ("src/bsg_next/model.py", "scripts/verify_prospective.py"):
            source = self.root / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("frozen numerical source")
            files[name] = MODULE.digest(source)
        self.manifest = self.root / "release/prospective_source_manifest.json"
        self.manifest.write_text(json.dumps({"version": "fixture", "files": files}))
        self.binding = {"schema_version": 1,
                        "frozen_manifest": "release/prospective_source_manifest.json",
                        "frozen_manifest_sha256": MODULE.digest(self.manifest),
                        "engineering_bindings": self.entries}
        self.save_binding()

    def tearDown(self):
        self.tmp.cleanup()

    def save_binding(self):
        (self.root / "release/engineering_source_bindings.json").write_text(json.dumps(self.binding))

    def test_explicit_execution_chain(self):
        before = self.manifest.read_bytes()
        result = MODULE.verify(self.root)
        self.assertEqual(result["active_frozen_matches"], 2)
        self.assertEqual(len(result["engineering_derivatives"]), 4)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_scientific_fallback_rejected(self):
        name = "src/bsg_next/model.py"
        self.entries[name] = dict(next(iter(self.entries.values())))
        self.save_binding()
        with self.assertRaisesRegex(ValueError, "forbidden"):
            MODULE.verify(self.root)

    def test_changed_scientific_source_rejected(self):
        (self.root / "src/bsg_next/model.py").write_text("changed numerical source")
        with self.assertRaisesRegex(ValueError, "Frozen scientific"):
            MODULE.verify(self.root)

    def test_unbound_engineering_change_rejected(self):
        (self.root / "scripts/package_release.py").write_text("another wrapper")
        with self.assertRaisesRegex(ValueError, "Unbound publication"):
            MODULE.verify(self.root)

    def test_missing_binding_rejected(self):
        del self.entries["pyproject.toml"]
        self.save_binding()
        with self.assertRaisesRegex(ValueError, "Frozen scientific/unbound"):
            MODULE.verify(self.root)

    def test_changed_historical_source_rejected(self):
        old = self.root / self.entries["pyproject.toml"]["execution_source"]
        old.write_text("wrong old bytes")
        with self.assertRaisesRegex(ValueError, "preserved execution"):
            MODULE.verify(self.root)

    def test_frozen_manifest_rebinding_rejected(self):
        self.manifest.write_text(self.manifest.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "different frozen manifest"):
            MODULE.verify(self.root)

    def test_parent_escape_rejected(self):
        self.entries["pyproject.toml"]["execution_source"] = "../outside"
        self.save_binding()
        with self.assertRaisesRegex(ValueError, "Nonrelative"):
            MODULE.verify(self.root)

    def test_absolute_escape_rejected(self):
        self.entries["pyproject.toml"]["execution_source"] = "/outside"
        self.save_binding()
        with self.assertRaisesRegex(ValueError, "Nonrelative"):
            MODULE.verify(self.root)

    def test_additional_scientific_file_rejected(self):
        (self.root / "src/bsg_next/new.py").write_text("new scientific source")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            MODULE.verify(self.root)


if __name__ == "__main__":
    unittest.main()
