"""Offline engineering checks for the anonymous saved-endpoint capsule."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("review_reproduce", ROOT / "reproduce.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.expected, cls.manifest = MODULE.load_inputs()

    def test_all_saved_rows_and_frozen_display_values(self):
        result, risks, margins, summaries = MODULE.derive(self.data)
        MODULE.compare_expected(result, self.expected)
        self.assertEqual((len(risks), len(margins), len(summaries)), (48, 48, 2))
        self.assertEqual(result["C7"]["standing_counts"], [7, 0, 17])
        self.assertEqual(result["D4"]["standing_counts"], [16, 0, 8])

    def test_hash_tamper_fails_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "capsule"
            shutil.copytree(ROOT, root)
            with (root / "data/spectral_cells.csv").open("ab") as handle:
                handle.write(b"\n")
            output = Path(tmp) / "output"
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                MODULE.reproduce(output, root)
            self.assertFalse(output.exists())

    def test_missing_cells_fail_even_after_hash_verification(self):
        for name in ("spectral_cells.csv", "witness_units.csv", "cell_risk_reference.csv", "registration_margins.csv"):
            with self.subTest(name=name):
                altered = copy.deepcopy(self.data)
                altered[name].pop()
                with self.assertRaisesRegex(ValueError, "expected 48 cells"):
                    MODULE.derive(altered)

    def test_duplicate_cells_fail_even_with_48_rows(self):
        for name in ("spectral_cells.csv", "witness_units.csv", "cell_risk_reference.csv", "registration_margins.csv"):
            with self.subTest(name=name):
                altered = copy.deepcopy(self.data)
                altered[name][1] = copy.deepcopy(altered[name][0])
                with self.assertRaisesRegex(ValueError, "duplicate cell"):
                    MODULE.derive(altered)

    def test_missing_and_duplicate_images_fail(self):
        missing = copy.deepcopy(self.data)
        missing["image_endpoints.csv"].pop()
        with self.assertRaisesRegex(ValueError, "2952"):
            MODULE.derive(missing)
        duplicate = copy.deepcopy(self.data)
        duplicate["image_endpoints.csv"][1] = copy.deepcopy(duplicate["image_endpoints.csv"][0])
        with self.assertRaisesRegex(ValueError, "Missing or duplicate image"):
            MODULE.derive(duplicate)

    def test_original_pairwise_tolerance_and_tied_rank(self):
        self.assertEqual(MODULE.tolerance(.9, .8), 1e-8 + 1e-7 * .9)
        self.assertEqual(MODULE.tolerance(.9, .8), MODULE.tolerance(.8, .9))
        rows = [dict(image_sha256=name, gamma=value, semantic_registration=str(semantic))
                for name, value, semantic in (("semantic", .9, True), ("near", .9 + 8e-8, False),
                                              ("higher", .9 + 16e-8, False))]
        result = MODULE.image_statistics(rows)
        self.assertEqual((result["rank_low"], result["rank_high"]), (2, 3))
        self.assertEqual(result["standing"], "behind")

    def test_semantic_image_is_excluded_from_alternative_margins(self):
        rows = [dict(image_sha256=name, gamma=value, semantic_registration=str(semantic))
                for name, value, semantic in (("a", .7, False), ("b", .9, True), ("c", .8, False))]
        result = MODULE.image_statistics(rows)
        self.assertAlmostEqual(result["d_top"], .1)
        self.assertAlmostEqual(result["d_median"], .15)
        self.assertEqual(result["standing"], "unique_leading")

    def test_top_two_share_denominator_uses_all_24_cells(self):
        result, risks, _, _ = MODULE.derive(self.data)
        summary = result["C7"]["lower_endpoint_summary"]
        self.assertEqual(summary["n"], 24)
        self.assertEqual(summary["top_two_share_percent"], 90.68064875787752)
        self.assertEqual(summary["median"], 0.00038929152944697926)
        self.assertLess(summary["top_two_sum"], summary["all_cells_sum"])
        self.assertEqual(len([r for r in risks if r["domain"] == "C7"]), 24)

    def test_output_directory_never_overwrites_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            marker = output / "marker.txt"
            marker.write_text("keep")
            with self.assertRaisesRegex(ValueError, "must be new"):
                MODULE.reproduce(output)
            self.assertEqual(marker.read_text(), "keep")
        with self.assertRaisesRegex(ValueError, "outside the evidence package"):
            MODULE.reproduce(ROOT / "unused-test-output")

    def test_standalone_cli_without_site_packages_and_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "capsule"
            shutil.copytree(ROOT, root)
            output = Path(tmp) / "tables"
            completed = subprocess.run([sys.executable, "-I", "-S", str(root / "reproduce.py"),
                                        "--output", str(output)], cwd=tmp, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["status"], "PASS")
            self.assertFalse(report["coordinate_identity_residual_check_repeated"])
            result = json.loads((output / "reproduced_values.json").read_text())
            MODULE.compare_expected(result, self.expected)
            for table in range(3, 7):
                self.assertEqual((output / f"table{table}.tex").read_text(), MODULE.render_table(table, self.expected))
            with (output / "cell_risk_recomputed.csv").open(newline="") as handle:
                risk_rows = list(csv.DictReader(handle))
            saved = {r["cell"]: r for r in self.data["cell_risk_reference.csv"]}
            for row in risk_rows:
                for key in ("gamma", "eta", "identity_risk", "premium_lower", "premium_upper"):
                    self.assertEqual(float(row[key]), float(saved[row["cell"]][key]))
            self.assertEqual(len(risk_rows), 48)
            MODULE.load_inputs(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
