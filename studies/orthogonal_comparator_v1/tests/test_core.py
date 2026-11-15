"""PLANNED numerical software checks. Not executed during preparation."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import analyze_bank


class OrthogonalComparatorTests(unittest.TestCase):
    def c2(self, Z, **kwargs):
        return analyze_bank(Z, [[0, 1], [1, 0]], expected_action_count=2,
                            identity_index=0, **kwargs)

    def test_c2_analytic_nonzero_orthogonality_cost(self):
        # Sigma=2, swap correlation=sqrt(3)/2, identity correlation=1.
        a = np.sqrt(3.0)
        Z = np.array([[[a], [1.0]], [[1.0], [a]]])
        saved = Z.copy()
        result, arrays = self.c2(Z)
        self.assertEqual(result['status'], 'complete')
        m = result['metrics']; correlation = np.sqrt(3.0)/2.0
        self.assertAlmostEqual(m['eta'], 0.125, places=12)
        self.assertAlmostEqual(m['L_orth'], 1.0-correlation, places=12)
        self.assertAlmostEqual(m['beta_svd'], (1.0-correlation)**2/2.0, places=12)
        self.assertAlmostEqual(m['beta_eta_floor'], m['eta']**2/(1+np.sqrt(1-m['eta']))**2, places=12)
        self.assertTrue(np.array_equal(Z, saved))
        self.assertTrue(np.allclose(arrays['Q'].transpose(0,2,1) @ arrays['Q'], np.eye(1)))

    def test_c2_exact_sign_action(self):
        result, arrays = self.c2(np.array([[[-1.0], [1.0]]]))
        self.assertEqual(result['status'], 'complete')
        self.assertAlmostEqual(result['metrics']['eta'], 0.0, places=12)
        self.assertAlmostEqual(result['metrics']['L_orth'], 0.0, places=12)
        self.assertAlmostEqual(float(arrays['Q'][1,0,0]), -1.0, places=12)
        # Reflections are allowed: this is O(r), never determinant-adjusted SO(r).

    def test_c2_singular_cross_moment_is_allowed(self):
        a = np.sqrt(3.0)
        result, arrays = self.c2(np.array([[[a], [1.0]], [[a], [-1.0]]]))
        self.assertEqual(result['status'], 'complete')
        self.assertAlmostEqual(result['metrics']['eta'], 0.5, places=12)
        self.assertAlmostEqual(result['metrics']['L_orth'], 1.0, places=12)
        self.assertAlmostEqual(result['metrics']['beta_svd'], 0.5, places=12)
        self.assertFalse(result['per_action'][1]['Q_completion_unique_under_rank_rule'])
        self.assertTrue(np.isfinite(arrays['Q']).all())

    def test_c3_row_vector_svd_orientation(self):
        a=np.sqrt(3.0)/2.0
        Z=np.array([[[1.0,0.0],[-0.5,a],[-0.5,-a]]])
        result, arrays=analyze_bank(Z, [[0,1,2],[1,2,0],[2,0,1]],
                                   expected_action_count=3,identity_index=0)
        self.assertEqual(result['status'],'complete')
        self.assertTrue(np.allclose(arrays['Q'][1],[[-0.5,a],[-a,-0.5]],atol=1e-12))
        self.assertAlmostEqual(result['metrics']['L_orth'],0.0,places=12)

    def test_rank_deficient_second_moment_refused(self):
        Z=np.array([[[1.0,1.0],[-1.0,-1.0]],[[2.0,2.0],[-2.0,-2.0]]])
        result,_=self.c2(Z)
        self.assertEqual(result['status'],'not_evaluable_rank')
        self.assertNotIn('metrics',result)
        self.assertFalse(result['jitter'])
        self.assertFalse(result['pseudoinverse'])

    def test_nonuniform_context_state_and_action_weights_refused(self):
        Z=np.array([[[2.0],[1.0]],[[1.0],[2.0]]])
        for name in ('context_weights','state_weights','action_weights'):
            with self.subTest(name=name):
                result,_=self.c2(Z,**{name:[0.75,0.25]})
                self.assertEqual(result['status'],'not_evaluable_weight_contract')
                self.assertNotIn('metrics',result)

    def test_identity_and_fixed_point_masking_refused(self):
        Z=np.array([[[2.0],[1.0]]])
        for key in ('include_identity','include_fixed_points'):
            result,_=self.c2(Z,**{key:False})
            self.assertEqual(result['status'],'not_evaluable_weight_contract')
        result,_=analyze_bank(Z,[[1,0]],expected_action_count=2,identity_index=0)
        self.assertEqual(result['status'],'not_evaluable_action_contract')

    def test_nonfinite_input_refused(self):
        result,_=self.c2(np.array([[[float('nan')],[1.0]]]))
        self.assertEqual(result['status'],'not_evaluable_input')


if __name__=='__main__':unittest.main()
