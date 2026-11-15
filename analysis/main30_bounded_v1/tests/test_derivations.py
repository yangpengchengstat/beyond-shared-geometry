"""Engineering/arithmetic fixtures only; no new scientific experiment."""
import csv
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import derive


class BoundedDerivationTests(unittest.TestCase):
    def test_existing_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing"
            output.mkdir()
            marker = output / "validation_report.json"
            marker.write_text("preserved")
            with self.assertRaises(FileExistsError):
                derive.run(Path(tmp) / "not_read.zip", output)
            self.assertEqual(marker.read_text(), "preserved")

    def row(self, identity, value, semantic=False):
        return dict(image_sha256=identity,gamma=value,semantic_registration=semantic)

    def test_max_before_mean_synthetic_arithmetic(self):
        # Mixed signs intentionally test the arithmetic operator; this is not
        # asserted to be a realizable pair of exact spectral banks.
        differences=[-0.04,0.02]
        self.assertAlmostEqual(statistics.mean(max(0,d) for d in differences),0.01)
        self.assertEqual(max(0,statistics.mean(differences)),0)
        rs=[derive.risk_fields(.9,.24,.6,.5),derive.risk_fields(.9,.18,.6,.5)]
        self.assertAlmostEqual(statistics.mean(r['premium_lower'] for r in rs),.01)

    def test_identity_uses_total_not_selected_energy(self):
        r=derive.risk_fields(.8,.1,.6,.4)
        self.assertAlmostEqual(r['identity_risk'],.8)
        self.assertAlmostEqual(r['total_nontrivial_energy'],.4)
        self.assertAlmostEqual(r['selected_nontrivial_energy'],.4)
        self.assertNotAlmostEqual(r['identity_risk'],2*(1-r['b_triv']))

    def test_semantic_exclusion_gives_positive_margin(self):
        r=derive.registration_fields([self.row('H0',.9,True),self.row('H1',.8),self.row('H2',.7)])
        self.assertAlmostEqual(r['d_top'],.1)
        self.assertAlmostEqual(r['d_median'],.15)
        self.assertEqual((r['rank_low'],r['rank_high'],r['standing']),(1,1,'unique_leading'))

    def test_small_positive_margin_can_tie(self):
        r=derive.registration_fields([self.row('H0',.9,True),self.row('H1',.9-5e-8),self.row('H2',.7)])
        self.assertGreater(r['d_top'],0)
        self.assertEqual((r['rank_low'],r['rank_high'],r['standing']),(1,2,'tied_leading'))

    def test_original_tolerance_exact_equality(self):
        self.assertEqual(derive.frozen._tol(.9,.8),1e-8+1e-7*.9)
        self.assertEqual(derive.frozen._tol(.8,.9),derive.frozen._tol(.9,.8))
        r=derive.registration_fields([self.row('H0',.9,True),self.row('H1',.9)])
        self.assertEqual((r['rank_low'],r['rank_high']),(1,2))

    def test_single_best_alternative_change(self):
        first=derive.registration_fields([self.row('H0',.9,True),self.row('H1',.8),self.row('H2',.7)])
        second=derive.registration_fields([self.row('H0',.9,True),self.row('H1',.8),self.row('H2',.95)])
        self.assertEqual(first['best_alternative_images'],'H1')
        self.assertEqual(second['best_alternative_images'],'H2')
        self.assertAlmostEqual(second['d_top'],-.05)
        self.assertEqual(second['standing'],'behind')

    def test_nontransitive_pairwise_ties_use_semantic_reference(self):
        r=derive.registration_fields([self.row('H0',.9,True),self.row('H1',.9+8e-8),self.row('H2',.9+16e-8)])
        self.assertEqual((r['rank_low'],r['rank_high']),(2,3))

    def test_duplicate_semantic_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            derive.registration_fields([self.row('H0',.9,True),self.row('H0',.8)])

    def test_complete_actions_identity_fixed_points_and_direction(self):
        for domain in ['C7','D4']:
            spec=derive.frozen.spectral.group_spec(domain)
            a=np.asarray(spec['actions'])
            np.testing.assert_array_equal(a[spec['identity']],np.arange(spec['p']))
            for g in range(spec['m']):
                for h in range(spec['m']):
                    np.testing.assert_array_equal(a[g][a[h]],a[spec['multiplication'][g][h]])
        d4=derive.frozen.spectral.group_spec('D4')
        self.assertTrue(any(0<sum(x==i for i,x in enumerate(a))<4 for a in d4['actions']))

    def test_direct_identity_complete_grid_nonzero_context_means(self):
        rng=np.random.default_rng(41)
        for domain,p in [('C7',7),('D4',4)]:
            Z=rng.normal(size=(10,p,2))+np.array([2.0,-.4])
            r=derive.direct_identity(Z,derive.frozen.spectral.group_spec(domain))
            self.assertAlmostEqual(r['identity_risk_direct'],2*(1-r['e_triv_from_context_mean']),places=12)
            self.assertGreater(r['e_triv_from_context_mean'],0)

    def test_metric_conjugation_preserves_group_law_not_orthogonality(self):
        rho=np.array([[0.,1.],[1.,0.]])
        sqrt_test=np.diag([2.,1.])
        transformed=sqrt_test@rho@np.linalg.inv(sqrt_test)
        np.testing.assert_array_equal(transformed,np.array([[0.,2.],[.5,0.]]))
        np.testing.assert_array_equal(transformed@transformed,np.eye(2))
        np.testing.assert_array_equal(transformed.T@transformed,np.diag([.25,4.]))
        self.assertFalse(np.allclose(transformed.T@transformed,np.eye(2)))
        # The same original row-vector function has identical unwhitened outputs.
        z=np.array([[1.,3.],[-2.,4.]])
        np.testing.assert_array_equal((z@np.linalg.inv(sqrt_test))@transformed@sqrt_test,z@rho)

    def test_saved_panel_completeness_and_provenance(self):
        path=derive.OUTPUT/'validation_report.json'
        if not path.exists():self.skipTest('run derivation before output validation')
        r=json.loads(path.read_text())
        self.assertEqual(r['status'],'DONE')
        self.assertEqual(r['counts'],dict(spectral_cells=48,image_conditions=2952,type_rows=216,identity_direct_checks=48))
        self.assertTrue(r['source_files_unchanged'])
        risks=derive.read_csv(derive.OUTPUT/'cell_risk_reference.csv')
        images=derive.read_csv(derive.OUTPUT/'image_endpoints.csv')
        for row in risks:
            matching=[i for i in images if i['cell']==row['cell']]
            self.assertEqual(len(matching),120 if row['domain']=='C7' else 3)
            self.assertEqual({i['join_key'] for i in matching},{row['join_key']})
            self.assertEqual(len({i['image_sha256'] for i in matching}),len(matching))


if __name__=='__main__':
    unittest.main(verbosity=2)
