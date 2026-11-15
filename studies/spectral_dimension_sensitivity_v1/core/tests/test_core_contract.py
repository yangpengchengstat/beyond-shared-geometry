"""Prepared software tests. NOT EXECUTED during no-experiment preparation.

Run only after explicit authorization to execute numerical software checks.
These synthetic arrays are software fixtures, not a scientific simulation grid.
"""
import unittest
import numpy as np

from core import spectral_dimension_core as core


class ContractTests(unittest.TestCase):
    def test_unmodified_vendor_hash(self):
        self.assertTrue(core.verify_kernel_source()["source_unmodified"])

    def test_no_upward_dimension_or_noninteger_selection(self):
        z = np.zeros((128, 7, 16))
        for dimension in (32, 64, 8.0, True):
            with self.assertRaises(ValueError):
                core._source_coordinates(z, "C7", dimension)

    def test_rank_failure_retained_without_repair(self):
        z = np.zeros((128, 7, 16), dtype=np.float64)
        before = z.copy()
        result, _ = core.analyze_semantic(z, "C7", 8)
        self.assertEqual(result["status"], "not_evaluable_rank")
        self.assertNotIn("metrics", result)
        self.assertFalse(result["jitter"])
        self.assertFalse(result["automatic_rank_reduction"])
        np.testing.assert_array_equal(z, before)

    def test_prefix_subspace_and_direct_frozen_kernel_agree(self):
        z = np.random.default_rng(701).normal(size=(128, 4, 16))
        before = z.copy()
        result, arrays = core.analyze_semantic(z, "D4", 8)
        reference, ref_arrays = core.spectral.analyze_bank(z[..., :8], "D4")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(reference["status"], "complete")
        for key, value in reference["metrics"].items():
            self.assertAlmostEqual(result["test_bank_spectral"]["metrics"][key], value, places=12)
        np.testing.assert_allclose(arrays["Sigma"], ref_arrays["Sigma"], rtol=0, atol=0)
        self.assertEqual(arrays["whitening"].shape, (8, 8))
        self.assertEqual(result["metrics"]["eta"], result["metrics"]["eta_LS"])
        self.assertEqual(result["metrics"]["p_minus"], max(0.0, result["metrics"]["L_lower"] - result["metrics"]["eta_LS"]))
        np.testing.assert_array_equal(z, before)

    def test_anchor_stops_on_changed_metric(self):
        z = np.random.default_rng(702).normal(size=(128, 4, 16))
        unit, _ = core.analyze_semantic(z, "D4", 16)
        self.assertEqual(unit["status"], "complete")
        import copy
        saved = copy.deepcopy(unit["test_bank_spectral"])
        self.assertTrue(core.validate_anchor(unit, saved)["passed"])
        saved["metrics"]["gamma"] += 0.01
        self.assertFalse(core.validate_anchor(unit, saved)["passed"])
        self.assertEqual(core.validate_anchor(unit, saved)["status"], "mismatch")

    def test_complete_image_count_and_incomplete_rank(self):
        for domain, expected in (("C7", 120), ("D4", 3)):
            groups = core.subgroup_specs(domain)
            self.assertEqual(len(groups), expected)
            self.assertEqual(sum(g["semantic"] for g in groups), 1)
            units = [{"domain": domain, "dimension": 8, "subgroup_id": g["id"],
                      "semantic_registration": g["semantic"], "status": "complete",
                      "metrics": {"gamma": 0.9}} for g in groups]
            complete = core.registration_summary(units, domain, 8)
            self.assertEqual(complete["rank_low"], 1)
            self.assertEqual(complete["rank_high"], expected)
            self.assertEqual(complete["classification"], "tied_leading")
            remove_id = next(g["id"] for g in groups if not g["semantic"])
            incomplete = core.registration_summary([u for u in units if u["subgroup_id"] != remove_id], domain, 8)
            self.assertEqual(incomplete["status"], "incomplete_descriptive")
            self.assertIsNone(incomplete["rank_low"])
            self.assertIsNone(incomplete["d_top"])
            self.assertIn(remove_id, incomplete["missing_image_ids"])


if __name__ == "__main__":
    unittest.main()
