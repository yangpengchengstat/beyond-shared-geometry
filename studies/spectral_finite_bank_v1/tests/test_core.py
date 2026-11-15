"""Small analytic software checks; no saved LLM arrays, model code, or calibration."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gamma_r as core


def c7_exact():
    rng = np.random.default_rng(37)
    r, n = 16, 20
    R = np.eye(r)
    for k in range(1, 4):
        theta = 2 * np.pi * k / 7
        R[2*k-2:2*k, 2*k-2:2*k] = [[np.cos(theta), np.sin(theta)],
                                                [-np.sin(theta), np.cos(theta)]]
    start = rng.normal(size=(n, r))
    return np.stack([start @ np.linalg.matrix_power(R, s) for s in range(7)], axis=1)


def d4_exact():
    rng = np.random.default_rng(73)
    context = rng.normal(size=(8, 4))
    return np.array([[np.kron(np.eye(4)[s], context[c]) for s in range(4)] for c in range(8)])


class CoreChecks(unittest.TestCase):
    def complete(self, Z, domain):
        summary, arrays = core.analyze_bank(Z, domain)
        self.assertEqual(summary["status"], "complete", summary.get("reason"))
        self.assertTrue(all(c["passed"] for c in summary["checks"].values()))
        json.dumps(summary, allow_nan=False)
        return summary, arrays

    def test_exact_C7_full_rank_and_complete_audit_arrays(self):
        s, a = self.complete(c7_exact(), "C7")
        self.assertAlmostEqual(s["metrics"]["gamma"], 1.0, places=10)
        self.assertAlmostEqual(s["metrics"]["eta_LS"], 0.0, places=10)
        self.assertEqual(a["K"].shape, (112, 112))
        self.assertEqual(len(a["sigma_eigenvalues"]), 16)
        self.assertAlmostEqual(sum(t["e"] for t in s["types"]), 1.0, places=10)
        self.assertAlmostEqual(sum(t["b"] for t in s["types"]), s["metrics"]["gamma"], places=10)
        self.assertFalse(s["witness_action_constructed"])
        self.assertFalse(s["provenance_verified_by_core"])

    def test_exact_D4_and_absent_types(self):
        s, a = self.complete(d4_exact(), "D4")
        self.assertAlmostEqual(s["metrics"]["gamma"], 1.0, places=10)
        self.assertEqual(a["K"].shape, (128, 128))
        absent = [t for t in s["types"] if t["expected_absent"]]
        self.assertEqual({t["name"] for t in absent}, {"chi_pm", "chi_mp"})
        for t in absent:
            self.assertAlmostEqual(t["e"], 0.0, places=10)

    def test_paper_rank_one_same_untyped_spectrum_different_typed_budget(self):
        spec = core._abelian_spec((2, 2, 7))
        elems = spec["elements"]
        A = np.array([((-1) ** a + (-1) ** b) / np.sqrt(2) for a, b, k in elems])
        B = np.array([np.sqrt(2) * np.cos(2 * np.pi * k / 7) for a, b, k in elems])
        sa, aa = core._analyze_spec(A.reshape(1, -1, 1), spec)
        sb, ab = core._analyze_spec(B.reshape(1, -1, 1), spec)
        self.assertEqual(sa["status"], "complete", sa.get("reason"))
        self.assertEqual(sb["status"], "complete", sb.get("reason"))
        np.testing.assert_allclose(aa["K_eigenvalues"], ab["K_eigenvalues"], atol=1e-12)
        self.assertAlmostEqual(sa["metrics"]["eta_spec"], 0.5, places=10)
        self.assertAlmostEqual(sb["metrics"]["eta_spec"], 0.5, places=10)
        self.assertAlmostEqual(sa["metrics"]["gamma"], 0.5, places=10)
        self.assertAlmostEqual(sb["metrics"]["gamma"], 0.0, places=10)
        self.assertAlmostEqual(sb["metrics"]["L_lower"], 2.0, places=10)

    def test_full_bank_LS_spectral_identity_for_inexact_data(self):
        Z = np.random.default_rng(41).normal(size=(12, 7, 6))
        s, _ = self.complete(Z, "C7")
        self.assertGreater(s["metrics"]["eta_spec"], 0.05)
        self.assertAlmostEqual(s["metrics"]["eta_LS"], s["metrics"]["eta_spec"], places=10)
        self.assertLessEqual(s["budget"]["used_dimension"], 6)

    def test_uncentered_second_moment_is_preserved(self):
        Z = np.ones((2, 7, 1)) * 3.0
        s, a = self.complete(Z, "C7")
        self.assertAlmostEqual(a["Sigma"][0, 0], 9.0)
        self.assertAlmostEqual(s["metrics"]["gamma"], 1.0)
        self.assertAlmostEqual(s["types"][0]["e"], 1.0)

    def test_rank_deficiency_has_no_risk_claim_and_does_not_mutate_input(self):
        Z = np.ones((2, 7, 2))
        before = Z.copy()
        s, a = core.analyze_bank(Z, "C7")
        self.assertEqual(s["status"], "not_evaluable_rank")
        self.assertNotIn("metrics", s)
        self.assertNotIn("K", a)
        self.assertEqual(len(s["diagnostics"]["sigma_eigenvalues"]), 2)
        np.testing.assert_array_equal(Z, before)

    def test_declared_rank_threshold_is_strict(self):
        # Complete orthogonal Fourier coordinates with second-moment ratio 1e-12.
        angle = np.arange(7) * 2 * np.pi / 7
        Z = np.stack([np.cos(angle), 1e-6*np.sin(angle)], axis=-1)[None]
        s, _ = core.analyze_bank(Z, "C7")
        self.assertEqual(s["status"], "not_evaluable_rank")
        self.assertNotIn("metrics", s)

    def test_subnormal_rank_failure_remains_strict_JSON_safe(self):
        angle = np.arange(7) * 2 * np.pi / 7
        Z = np.stack([np.cos(angle), 1e-160 * np.sin(angle)], axis=-1)[None]
        s, a = core.analyze_bank(Z, "C7")
        self.assertEqual(s["status"], "not_evaluable_rank")
        self.assertIsNone(s["diagnostics"]["sigma_condition_number"])
        self.assertTrue(s["diagnostics"]["sigma_condition_number_overflow"])
        self.assertNotIn("metrics", s)
        self.assertEqual(len(a["sigma_eigenvalues"]), 2)
        json.dumps(s, allow_nan=False)

    def test_zero_second_moment_reports_rank_failure_without_nan_summary(self):
        s, _ = core.analyze_bank(np.zeros((2, 7, 1)), "C7")
        self.assertEqual(s["status"], "not_evaluable_rank")
        json.dumps(s, allow_nan=False)

    def test_invalid_arrays_are_not_silently_coerced(self):
        for Z in (np.ones((0, 7, 1)), np.ones((2, 6, 1)), np.ones((7, 1)),
                  np.ones((2, 7, 1), complex), np.full((2, 7, 1), np.nan),
                  np.full((2, 7, 1), np.inf), np.full((2, 7, 1), "x"),
                  np.ones((2, 7, 1), dtype=object)):
            with self.subTest(dtype=Z.dtype, shape=Z.shape):
                s, _ = core.analyze_bank(Z, "C7")
                self.assertEqual(s["status"], "not_evaluable_input")
                self.assertNotIn("metrics", s)
                json.dumps(s, allow_nan=False)

    def test_ragged_array_is_an_input_failure(self):
        s, _ = core.analyze_bank([[1, 2], [3]], "C7")
        self.assertEqual(s["status"], "not_evaluable_input")
        self.assertNotIn("metrics", s)

    def test_pairing_failure_stops_before_budget(self):
        original = np.linalg.eigvalsh
        calls = 0
        def broken_pairing(matrix):
            nonlocal calls
            calls += 1
            values = original(matrix)
            # First call is K, second is trivial type, third is the first plane.
            if calls == 3:
                values[-1] += 0.01
            return values
        with patch.object(core.np.linalg, "eigvalsh", broken_pairing):
            s, _ = core.analyze_bank(c7_exact(), "C7")
        self.assertEqual(s["status"], "not_evaluable_numerics")
        self.assertIn("pairing", s["reason"])
        self.assertNotIn("metrics", s)
        self.assertNotIn("budget", s)

    def test_unknown_group_rejected(self):
        s, _ = core.analyze_bank(np.ones((2, 4, 1)), "custom")
        self.assertEqual(s["status"], "not_evaluable_input")
        self.assertNotIn("metrics", s)

    def test_incomplete_types_rejected_by_projector_completeness(self):
        spec = core.group_spec("C7")
        spec["types"] = spec["types"][:1]
        s, _ = core._analyze_spec(np.arange(14.0).reshape(2, 7, 1), spec)
        self.assertEqual(s["status"], "not_evaluable_numerics")
        self.assertEqual(s["reason"], "projector_completeness")
        self.assertNotIn("metrics", s)
        spec["types"] = []
        s, _ = core._analyze_spec(np.ones((2, 7, 1)), spec)
        self.assertEqual(s["status"], "not_evaluable_input")

    def test_bad_projector_is_rejected_without_eigenvalue_clipping(self):
        spec = core.group_spec("C7")
        spec["types"][1]["coefficient"] *= 2
        s, _ = core._analyze_spec(np.arange(14.0).reshape(2, 7, 1), spec)
        self.assertEqual(s["status"], "not_evaluable_numerics")
        self.assertIn("idempotence", s["reason"])
        self.assertNotIn("metrics", s)

    def test_wrong_group_action_rejected_before_whitening(self):
        spec = core.group_spec("D4")
        spec["actions"][1][0] = spec["actions"][1][1]
        s, a = core._analyze_spec(np.ones((2, 4, 1)), spec)
        self.assertEqual(s["status"], "not_evaluable_input")
        self.assertEqual(a, {})

    def test_bad_whitening_is_a_failure_not_just_a_report(self):
        original = np.linalg.eigh
        count = 0
        def bad_eigh(matrix):
            nonlocal count
            count += 1
            w, u = original(matrix)
            if count == 1:
                w = w * 2.0
            return w, u
        with patch.object(core.np.linalg, "eigh", bad_eigh):
            s, _ = core.analyze_bank(c7_exact(), "C7")
        self.assertEqual(s["status"], "not_evaluable_numerics")
        self.assertEqual(s["reason"], "whitening_identity")
        self.assertNotIn("metrics", s)

    def test_integer_budget_matches_small_exhaustive_allocations(self):
        rng = np.random.default_rng(29)
        for r in range(1, 7):
            for case in range(8):
                types = [(d, np.sort(rng.integers(0, 9, size=3).astype(float) / 8)[::-1])
                         for d in (1, 2, 2)]
                candidates = []
                for ks in itertools.product(range(4), repeat=3):
                    used = sum(d*k for (d, lam), k in zip(types, ks))
                    if used <= r:
                        candidates.append((sum(d*sum(lam[:k]) for (d, lam), k in zip(types, ks)), used, ks))
                expected = sorted(candidates, key=lambda x: (-x[0], x[1], x[2]))[0]
                actual = core.integer_budget(types, r)
                self.assertEqual(actual["value"], expected[0])
                self.assertEqual(actual["used_dimension"], expected[1])
                self.assertEqual(tuple(actual["selected_copies"]), expected[2])

    def test_zero_energy_ties_prefer_less_dimension_and_are_disclosed(self):
        result = core.integer_budget([(1, np.array([1., 0.])), (2, np.array([0.]))], 3)
        self.assertEqual(result["selected_copies"], [1, 0])
        self.assertEqual(result["unused_dimension"], 2)
        self.assertTrue(result["exact_allocation_tie"])
        self.assertTrue(result["near_allocation_tie"])

    def test_rotation_plane_is_not_split(self):
        result = core.integer_budget([(1, np.array([0.])), (2, np.array([0.5]))], 1)
        self.assertEqual(result["value"], 0.0)
        self.assertEqual(result["used_dimension"], 0)

    def test_small_negative_roundoff_is_retained_not_clipped(self):
        # DP is not a spectral validator: raw negative entries are simply less useful
        # than selecting nothing, and are never silently replaced with zero.
        lam = np.array([0.5, -1e-15])
        saved = lam.copy()
        result = core.integer_budget([(1, lam)], 2)
        self.assertEqual(result["selected_copies"], [1])
        np.testing.assert_array_equal(lam, saved)


if __name__ == "__main__":
    unittest.main()
