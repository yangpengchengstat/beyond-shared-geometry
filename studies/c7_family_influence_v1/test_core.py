"""Independent synthetic checks for paired-family deletion and C7 risks.

No retained LLM coordinates, archives, or result tables are read. Closed-form
fixtures and a complex Fourier oracle are independent of the frozen real-type
projector implementation. These are engineering checks, not new study outcomes.
"""
from __future__ import annotations

import copy
import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("family_influence_under_test", HERE / "core.py")
core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = core
_spec.loader.exec_module(core)


RISK_KEYS = (
    "eta", "L_orth", "beta", "gamma", "L_lower", "L_upper", "p_minus",
    "p_plus", "extra_law_lower", "extra_law_upper", "L_identity",
)


def metadata(families):
    return [
        {"family_id": f"family_{f:03d}", "paraphrase_id": v, "split": "test",
         "state_ids": list(range(7)),
         "row_ids": [f"family_{f:03d}/variant_{v}/state_{s}" for s in range(7)]}
        for f in range(families) for v in (0, 1)
    ]


def random_fixture(families=4, rank=3, seed=714):
    z = np.random.default_rng(seed).normal(size=(2 * families, 7, rank))
    # A nonzero offset makes an accidental extra-centering step detectable.
    z += np.linspace(0.3, 1.1, rank)
    return np.asarray(z, dtype=np.float64), metadata(families)


def independent_oracle(z):
    """Raw-coordinate residual risks plus a separate C7 Fourier budget.

    For Y_k = (1/7) FFT_state(X)_k, the complex frequency covariance is
    C_k = E_context[Y_k^* Y_k]. The trivial frequency has real cost one;
    k=1,2,3 each represents a conjugate pair with real cost two. Enumerating
    these four allocations replaces the implementation's real projectors/DP.
    """
    contexts, states, rank = z.shape
    assert states == 7
    flat = z.reshape(-1, rank)
    sigma = flat.T @ flat / len(flat)
    eig, basis = np.linalg.eigh((sigma + sigma.T) / 2)
    if not (eig[-1] > 0 and eig[0] > 1e-10 * eig[-1]):
        raise ValueError("Oracle requires the original full-rank condition")
    whitening = (basis * eig ** -0.5) @ basis.T
    x = z @ whitening
    xf = x.reshape(-1, rank)
    linear, orthogonal, identity = [], [], []
    for shift in range(7):
        target = x[:, (np.arange(7) + shift) % 7, :].reshape(-1, rank)
        cross = xf.T @ target / len(xf)
        left, _, right_t = np.linalg.svd(cross, full_matrices=False)
        q = left @ right_t
        linear.append(np.sum((xf @ cross - target) ** 2) / (len(xf) * rank))
        orthogonal.append(np.sum((xf @ q - target) ** 2) / (len(xf) * rank))
        identity.append(np.sum((xf - target) ** 2) / (len(xf) * rank))

    fourier = np.fft.fft(x, axis=1) / 7
    spectra = []
    for frequency in range(4):
        y = fourier[:, frequency, :]
        moment = y.conj().T @ y / contexts
        moment = (moment + moment.conj().T) / 2
        spectra.append(np.linalg.eigvalsh(moment)[::-1])
    costs = (1, 2, 2, 2)
    prefixes = [np.r_[0.0, np.cumsum(values)] for values in spectra]
    candidates = []
    for counts in itertools.product(*(range(rank // cost + 1) for cost in costs)):
        if sum(cost * count for cost, count in zip(costs, counts)) <= rank:
            candidates.append(sum(cost * prefix[count]
                                  for cost, prefix, count in zip(costs, prefixes, counts)))
    gamma = float(max(candidates) / rank)
    eta, orth = float(np.mean(linear)), float(np.mean(orthogonal))
    lower, upper = 2 * (1 - gamma), 2 * (1 - gamma * gamma)
    metrics = {
        "eta": eta, "L_orth": orth, "beta": orth - eta, "gamma": gamma,
        "L_lower": lower, "L_upper": upper,
        "p_minus": max(0.0, lower - eta), "p_plus": upper - eta,
        "extra_law_lower": max(0.0, lower - orth),
        "extra_law_upper": upper - orth, "L_identity": float(np.mean(identity)),
    }
    return metrics, {"sigma": sigma, "eigenvalues": eig, "whitening": whitening}


class FamilyInfluenceTests(unittest.TestCase):
    def assert_complete(self, result):
        self.assertEqual(result["status"], "complete", result.get("reason"))
        for block in (result, result["spectral"], result["orthogonal"]):
            self.assertTrue(all(check["passed"] for check in block["checks"].values()))
        self.assertFalse(result["model_forward"])
        self.assertFalse(result["PCA_refit"])
        self.assertFalse(result["predictor_training"])

    def assert_risks_close(self, actual, reference, atol=2e-10):
        for key in RISK_KEYS:
            with self.subTest(metric=key):
                np.testing.assert_allclose(actual[key], reference[key], rtol=2e-10, atol=atol)

    def test_scalar_rotation_has_known_positive_law_cost(self):
        z = np.broadcast_to(np.cos(2 * np.pi * np.arange(7) / 7)[None, :, None], (8, 7, 1)).copy()
        before = z.copy()
        result = core.analyze_bank(z)
        self.assert_complete(result)
        a7 = float(np.mean(np.abs(np.cos(2 * np.pi * np.arange(7) / 7))))
        expected = {
            "eta": 0.5, "gamma": 0.0, "L_lower": 2.0, "L_upper": 2.0,
            "L_orth": 2 - 2 * a7, "beta": 1.5 - 2 * a7,
            "p_minus": 1.5, "p_plus": 1.5,
            "extra_law_lower": 2 * a7, "extra_law_upper": 2 * a7,
            "L_identity": 2.0,
        }
        self.assert_risks_close(result["metrics"], expected)
        self.assertTrue(result["metrics"]["extra_law_lower_above_tolerance"])
        np.testing.assert_array_equal(z, before)

    def test_exact_sixteen_dimensional_action_survives_family_deletion(self):
        # Fourteen context directions + one complete C7 rotation plane. Each
        # context direction occurs in multiple distinct families, so dropping
        # both variants of any one family retains full rank.
        contexts = 32
        z = np.zeros((contexts, 7, 16), dtype=np.float64)
        for c in range(contexts):
            z[c, :, c % 14] = 1.0
        angles = 2 * np.pi * np.arange(7) / 7
        z[:, :, 14] = np.cos(angles)
        z[:, :, 15] = np.sin(angles)
        meta = metadata(contexts // 2)
        before = z.copy()
        for deleted in (None, "family_000", "family_015"):
            with self.subTest(deleted=deleted):
                bank = z if deleted is None else core.delete_family(z, meta, deleted)[0]
                result = core.analyze_bank(bank)
                self.assert_complete(result)
                expected = dict.fromkeys(RISK_KEYS, 0.0)
                expected.update(gamma=1.0, L_identity=0.25)
                self.assert_risks_close(result["metrics"], expected)
                self.assertFalse(result["metrics"]["extra_law_lower_above_tolerance"])
        np.testing.assert_array_equal(z, before)

    def test_fourier_and_residual_oracle_for_every_small_family_deletion(self):
        z, meta = random_fixture()
        for family in (None, *(f"family_{f:03d}" for f in range(4))):
            with self.subTest(deleted=family):
                indices = [i for i, item in enumerate(meta) if item["family_id"] != family]
                direct_bank = z[indices].copy()
                bank = z if family is None else core.delete_family(z, meta, family)[0]
                np.testing.assert_array_equal(bank, direct_bank)
                expected, audit = independent_oracle(direct_bank)
                result = core.analyze_bank(bank)
                self.assert_complete(result)
                self.assert_risks_close(result["metrics"], expected)
                eig = audit["eigenvalues"]
                np.testing.assert_allclose(result["diagnostics"]["sigma_min"], eig[0], rtol=1e-12)
                np.testing.assert_allclose(result["diagnostics"]["sigma_max"], eig[-1], rtol=1e-12)

    def test_random_sixteen_dimensional_fourier_oracle(self):
        z, meta = random_fixture(families=6, rank=16, seed=715)
        bank, _ = core.delete_family(z, meta, "family_003")
        expected, _ = independent_oracle(bank)
        result = core.analyze_bank(bank)
        self.assert_complete(result)
        self.assert_risks_close(result["metrics"], expected)

    def test_deletion_uses_family_identity_and_keeps_both_variants_and_all_states(self):
        z, meta = random_fixture()
        order = np.array([6, 1, 4, 3, 0, 7, 2, 5])
        bank, records = z[order].copy(), [copy.deepcopy(meta[i]) for i in order]
        bank_before, records_before = bank.copy(), copy.deepcopy(records)
        deleted, receipt = core.delete_family(bank, records, "family_001")
        expected_indices = [i for i, row in enumerate(records) if row["family_id"] != "family_001"]
        np.testing.assert_array_equal(deleted, bank[expected_indices])
        self.assertEqual(deleted.shape, (6, 7, 3))
        self.assertEqual(receipt["deleted_family"], "family_001")
        self.assertEqual(receipt["retained_context_indices"], expected_indices)
        self.assertEqual({r["paraphrase_id"] for r in receipt["removed_contexts"]}, {0, 1})
        self.assertTrue(all(r["family_id"] == "family_001" for r in receipt["removed_contexts"]))
        self.assertEqual(receipt["retained_families"], 3)
        self.assertFalse(np.shares_memory(deleted, bank))
        deleted[0, 0, 0] = 12345.0
        np.testing.assert_array_equal(bank, bank_before)
        self.assertEqual(records, records_before)

    def test_incomplete_duplicate_and_malformed_metadata_fail_closed(self):
        z, meta = random_fixture()
        with self.assertRaises(ValueError):
            core.delete_family(z[:-1], meta[:-1], "family_000")
        for mutation in ("duplicate_variant", "missing_state", "duplicate_row", "bad_split", "bad_variant"):
            bad = copy.deepcopy(meta)
            if mutation == "duplicate_variant": bad[1]["paraphrase_id"] = 0
            elif mutation == "missing_state": bad[0]["state_ids"] = list(range(6))
            elif mutation == "duplicate_row": bad[1]["row_ids"][0] = bad[0]["row_ids"][0]
            elif mutation == "bad_split": bad[0]["split"] = "train"
            else: bad[0]["paraphrase_id"] = 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                core.delete_family(z, bad, "family_000")
        with self.assertRaises(ValueError):
            core.delete_family(z, meta, "absent_family")
        with self.assertRaises(ValueError):
            core.delete_family(z[:2], meta[:2], "family_000")

    def test_nonzero_constant_is_not_recentered(self):
        z = np.full((4, 7, 1), 3.0, dtype=np.float64)
        result = core.analyze_bank(z)
        self.assert_complete(result)
        expected = dict.fromkeys(RISK_KEYS, 0.0)
        expected["gamma"] = 1.0
        self.assert_risks_close(result["metrics"], expected)
        self.assertAlmostEqual(result["diagnostics"]["sigma_min"], 9.0)
        centered = z - z.reshape(-1, 1).mean(axis=0)
        deficient = core.analyze_bank(centered)
        self.assertNotEqual(deficient["status"], "complete")
        self.assertEqual(deficient["spectral"]["status"], "not_evaluable_rank")

    def test_deletion_recomputes_whitening_on_remaining_bank(self):
        z, meta = random_fixture(seed=716)
        z[:2, :, 0] *= 30.0
        _, full_audit = independent_oracle(z)
        bank, _ = core.delete_family(z, meta, "family_000")
        expected, deleted_audit = independent_oracle(bank)
        old_white = full_audit["whitening"]
        incorrect_moment = old_white.T @ deleted_audit["sigma"] @ old_white
        self.assertGreater(np.linalg.norm(incorrect_moment - np.eye(3), "fro"), 0.1)
        result = core.analyze_bank(bank)
        self.assert_complete(result)
        self.assert_risks_close(result["metrics"], expected)
        self.assertTrue(result["fresh_bank_whitening"])

    def test_rank_failure_after_deletion_is_not_repaired(self):
        z, meta = random_fixture(families=2, rank=2)
        z[:2, :, 1] = 0.0
        self.assert_complete(core.analyze_bank(z))
        bank, receipt = core.delete_family(z, meta, "family_001")
        before = bank.copy()
        result = core.analyze_bank(bank)
        self.assertEqual(receipt["retained_families"], 1)
        self.assertNotEqual(result["status"], "complete")
        self.assertEqual(result["spectral"]["status"], "not_evaluable_rank")
        self.assertEqual(result["orthogonal"]["status"], "not_evaluable_rank")
        self.assertEqual(result["metrics"], {})
        self.assertEqual(len(result["diagnostics"]["sigma_eigenvalues"]), 2)
        self.assertFalse(result["orthogonal"]["jitter"])
        self.assertFalse(result["orthogonal"]["pseudoinverse"])
        self.assertFalse(result["orthogonal"]["automatic_rank_reduction"])
        np.testing.assert_array_equal(bank, before)

    def test_original_rank_threshold_is_not_relaxed(self):
        for ratio, should_pass in ((0.5e-10, False), (2e-10, True)):
            z = np.ones((4, 7, 2), dtype=np.float64)
            z[:, :, 1] = (np.array([1, -1, 1, -1]) * np.sqrt(ratio))[:, None]
            with self.subTest(ratio=ratio):
                result = core.analyze_bank(z)
                recorded = result["diagnostics"]["sigma_min_max_ratio"]
                self.assertEqual(recorded > 1e-10, should_pass)
                if should_pass: self.assert_complete(result)
                else:
                    self.assertNotEqual(result["status"], "complete")
                    self.assertEqual(result["metrics"], {})

    def test_nonfinite_and_incomplete_arrays_fail_closed(self):
        z, _ = random_fixture()
        for value in (np.nan, np.inf, -np.inf):
            bad = z.copy()
            bad[0, 0, 0] = value
            with self.subTest(value=value):
                result = core.analyze_bank(bad)
                self.assertNotEqual(result["status"], "complete")
                self.assertEqual(result["metrics"], {})
        self.assertNotEqual(core.analyze_bank(z[:, :6])["status"], "complete")

    def test_uniform_reordering_and_well_conditioned_coordinate_changes_preserve_risks(self):
        z, _ = random_fixture(seed=717)
        before = z.copy()
        base = core.analyze_bank(z)
        self.assert_complete(base)
        transform = np.array([[1.2, 0.2, 0.0], [0.0, 0.7, 0.1], [0.1, 0.0, 1.0]])
        variants = {
            "context_order": z[::-1],
            "cyclic_state_relabeling": z[:, (np.arange(7) + 3) % 7],
            "common_scale": 3.5 * z,
            "invertible_coordinates_then_fresh_whitening": z @ transform,
        }
        for name, bank in variants.items():
            with self.subTest(transformation=name):
                result = core.analyze_bank(bank)
                self.assert_complete(result)
                self.assert_risks_close(result["metrics"], base["metrics"])
        np.testing.assert_array_equal(z, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
