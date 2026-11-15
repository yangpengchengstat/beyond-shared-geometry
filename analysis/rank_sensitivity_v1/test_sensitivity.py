"""Synthetic engineering tests only: no retained LLM coordinate arrays loaded."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sensitivity as s


def fixture(domain="D4", families=4, rank=3, seed=92):
    names = [f"structural_test_{i:03d}" for i in range(families)]
    rd = {"C7": "C7", "D4": "D4_corner"}[domain]
    p = {"C7": 7, "D4": 4}[domain]
    contexts = [{"family_id": f, "paraphrase_id": v, "split": "test",
                 "orbit_id": "lexical_" + rd, "state_ids": list(range(p)),
                 "row_ids": [f"{rd}/{f}/p{v}/lexical_{rd}/s{x}" for x in range(p)]}
                for f in names for v in (0, 1)]
    Z = np.random.default_rng(seed).normal(size=(families * 2, p, rank))
    return Z, contexts, names


class FamilySensitivityTests(unittest.TestCase):
    def test_default_and_single_switch_cannot_execute(self):
        config = json.loads((s.HERE / "config.json").read_text())
        self.assertIs(config["full_run"], False)
        self.assertFalse(s.execution_authorized(config, False))
        self.assertFalse(s.execution_authorized(config, True))
        self.assertFalse(s.execution_authorized({"full_run": True}, False))
        self.assertFalse(s.execution_authorized({"full_run": "true"}, True))
        self.assertTrue(s.execution_authorized({"full_run": True}, True))

    def test_thread_limit_fallback_needs_no_optional_dependency(self):
        with patch.dict(sys.modules, {"threadpoolctl": None}):
            with s.cpu_thread_limit():
                self.assertEqual(s.os.environ["OPENBLAS_NUM_THREADS"], "1")
                self.assertEqual(s.os.environ["VECLIB_MAXIMUM_THREADS"], "1")

    def test_paired_deletion_preserves_all_states_and_same_ids_across_cells(self):
        for domain in ("C7", "D4"):
            Z, contexts, families = fixture(domain)
            for scale in (1, 2):
                # Deliberately shuffle contexts to prove deletion uses identities.
                order = np.random.default_rng(scale).permutation(len(contexts))
                bank, meta = (scale * Z)[order], [contexts[i] for i in order]
                kept, kept_meta = s.delete_family(bank, meta, domain, families, families[1])
                self.assertEqual(kept.shape, (6, Z.shape[1], 3))
                self.assertEqual({c["family_id"] for c in kept_meta}, set(families) - {families[1]})
                self.assertEqual([c["paraphrase_id"] for c in kept_meta], [0, 1] * 3)
                expected = (scale * Z)[[0, 1, 4, 5, 6, 7]]
                np.testing.assert_array_equal(kept, expected)

    def test_missing_variant_or_state_is_rejected_not_repaired(self):
        Z, contexts, families = fixture()
        with self.assertRaises(ValueError):
            s.family_moments(Z[:-1], contexts[:-1], "D4", families, rank=3)
        broken = copy.deepcopy(contexts)
        broken[0]["state_ids"].pop()
        with self.assertRaises(ValueError):
            s.family_moments(Z, broken, "D4", families, rank=3)
        broken = copy.deepcopy(contexts)
        broken[0]["row_ids"][0] = broken[0]["row_ids"][1]
        with self.assertRaises(ValueError):
            s.family_moments(Z, broken, "D4", families, rank=3)

    def test_all_image_sufficient_moments_match_direct_frozen_coordinates(self):
        # All 120 C7 and 3 D4 images; small synthetic dimensions only.
        for domain in ("C7", "D4"):
            Z, contexts, families = fixture(domain, rank=2)
            contributions = s.family_moments(Z, contexts, domain, families, rank=2)
            groups = s.extension.subgroup_specs(domain)
            structure = s.invariant_structure(groups[0]["spec"], rank=2)
            for deleted in (0, 2):
                direct, _ = s.delete_family(Z, contexts, domain, families, families[deleted])
                T = s.deleted_moment(contributions, deleted)
                np.testing.assert_allclose(T, np.einsum("nsa,ntb->stab", direct, direct) / len(direct), rtol=1e-13, atol=1e-14)
                white_summary, Twhite = s.whiten_moment(T)
                self.assertEqual(white_summary["status"], "complete")
                for group in groups:
                    reference, arrays = s.core._analyze_spec(direct, group["spec"])
                    optimized = s.gamma_from_moment(Twhite, group["spec"], structure)
                    self.assertEqual(reference["status"], "complete")
                    self.assertEqual(optimized["status"], "complete")
                    np.testing.assert_allclose(s.orbit_gram(Twhite, group["spec"]), arrays["K"], rtol=1e-12, atol=1e-13)
                    self.assertAlmostEqual(optimized["metrics"]["gamma"], reference["metrics"]["gamma"], places=12)
                    np.testing.assert_allclose(optimized["total_type_energies"], [t["e"] for t in reference["types"]], rtol=1e-12, atol=1e-13)

    def test_rank_16_fixed_dimension_direct_parity(self):
        # Actual q=16 is exercised on synthetic, complete state banks as well.
        for domain in ("C7", "D4"):
            Z, contexts, families = fixture(domain, families=6, rank=16)
            moment = s.deleted_moment(s.family_moments(Z, contexts, domain, families), 3)
            summary, white = s.whiten_moment(moment)
            self.assertEqual(summary["status"], "complete")
            direct, _ = s.delete_family(Z, contexts, domain, families, families[3])
            for group in s.extension.subgroup_specs(domain)[::60]:
                reference, arrays = s.core._analyze_spec(direct, group["spec"])
                optimized = s.gamma_from_moment(white, group["spec"])
                self.assertEqual(reference["status"], "complete")
                self.assertEqual(optimized["status"], "complete")
                np.testing.assert_allclose(s.orbit_gram(white, group["spec"]), arrays["K"], rtol=1e-11, atol=1e-12)
                self.assertAlmostEqual(optimized["metrics"]["gamma"], reference["metrics"]["gamma"], places=11)

    def test_rank_failed_deletion_is_retained_for_every_image(self):
        Z, contexts, families = fixture(families=2, rank=2)
        # First family contains only coordinate 0. Deleting the other is singular.
        Z[:2, :, 1] = 0
        results = list(s.execute_cell(Z, contexts, "D4", families, s.extension.subgroup_specs("D4"), rank=2))
        self.assertEqual(len(results), 2)
        failed = results[1]
        self.assertEqual(failed["deleted_family"], families[1])
        self.assertEqual(failed["whitening"]["status"], "not_evaluable_rank")
        self.assertEqual(failed["comparison"]["failed_spectral"], 3)
        self.assertIsNone(failed["comparison"]["d_top"])
        self.assertTrue(all(im["test_bank_spectral"]["status"] == "not_evaluable_rank" for im in failed["images"]))
        self.assertEqual(len(failed["whitening"]["diagnostics"]["sigma_eigenvalues"]), 2)

    def test_rewhitening_is_required_after_deletion(self):
        Z, contexts, families = fixture(rank=3)
        Z[:2, :, 0] *= 100
        F = s.family_moments(Z, contexts, "D4", families, rank=3)
        _, fullwhite = s.whiten_moment(F.mean(axis=0))
        _, deletedwhite = s.whiten_moment(s.deleted_moment(F, 0))
        self.assertGreater(np.linalg.norm(fullwhite - deletedwhite), .01)
        np.testing.assert_allclose(np.einsum("ssab->ab", deletedwhite) / 4, np.eye(3), atol=1e-10)

    def test_original_symmetric_tie_rule_and_semantic_exclusion(self):
        def rows(values):
            return [{"subgroup_id": str(i), "semantic_registration": i == 0,
                     "test_bank_spectral": {"status": "complete", "metrics": {"gamma": val}}}
                    for i, val in enumerate(values)]
        values = [.9, .9 - 5e-8, .7]
        result = s.margins_and_ranks(rows(values))
        self.assertGreater(result["d_top"], 0)
        self.assertEqual((result["rank_low"], result["rank_high"]), (1, 2))
        self.assertFalse(result["uniquely_leading"])
        self.assertAlmostEqual(result["d_median"], .9 - np.median(values[1:]))
        behind = s.margins_and_ranks(rows([.9, .91, .7]))
        self.assertEqual((behind["rank_low"], behind["rank_high"]), (2, 2))
        self.assertLess(behind["d_top"], 0)

    def test_image_identity_uses_permutations_not_display_id(self):
        groups = s.extension.subgroup_specs("D4")
        renamed = copy.deepcopy(groups[0])
        renamed["id"] = "not_the_original_number"
        self.assertEqual(s.image_identity(groups[0]), s.image_identity(renamed))
        self.assertNotEqual(s.image_identity(groups[0]), s.image_identity(groups[1]))

    def test_missing_input_stays_prepared_and_never_runs(self):
        config = json.loads((s.HERE / "config.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new"
            result = s.run(Path(temp) / "absent.zip", config, output, opt_in=False)
            self.assertEqual(result["status"], "PREPARED_NOT_RUN")
            self.assertEqual(result["real_family_deletions_executed"], 0)
            self.assertFalse((output / "deletions.jsonl").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
