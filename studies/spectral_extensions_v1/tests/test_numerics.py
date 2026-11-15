"""Small analytic software fixtures only; no real banks or research-grid run."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numerics as n


class NumericalExtensions(unittest.TestCase):
    def test_exact_C7_rotation_witness(self):
        angle=2*np.pi*np.arange(7)/7
        Z=np.stack([np.cos(angle),np.sin(angle)],axis=-1)[None]
        s,a=n.witness_comparison(Z,2*Z,"C7")
        self.assertEqual(s["status"],"complete",s.get("reason"))
        self.assertAlmostEqual(s["metrics"]["train_witness_risk"],0,places=10)
        self.assertAlmostEqual(s["metrics"]["test_witness_risk"],0,places=10)
        self.assertEqual(a["rho"].shape,(7,2,2))
        self.assertTrue(s["checks"]["witness_training_bound"]["passed"])

    def test_degenerate_isotypic_C7_requires_whole_copies(self):
        angle=2*np.pi*np.arange(7)/7
        orbit=np.stack([np.cos(angle),np.sin(angle)],axis=-1)
        # Equal-energy equivalent copies form a four-dimensional degenerate block.
        Z=np.array([np.c_[orbit,orbit],np.c_[orbit,-orbit]])
        s,a=n.witness_comparison(Z,Z,"C7")
        self.assertEqual(s["status"],"complete",s.get("reason"))
        self.assertAlmostEqual(s["metrics"]["train_witness_risk"],0,places=10)
        self.assertTrue(all(c["passed"] for c in s["checks"].values()))

    def test_partial_degenerate_cluster_selects_one_whole_copy(self):
        for domain,p in (("C7",7),("D4",4)):
            with self.subTest(domain=domain):
                angle=2*np.pi*np.arange(p)/p
                Z=np.zeros((2,p,3));Z[:,:,0]=1
                Z[0,:,1]=2*np.cos(angle);Z[1,:,2]=2*np.cos(angle)
                s,a=n.witness_comparison(Z,Z,domain)
                self.assertEqual(s["status"],"complete",s.get("reason"))
                self.assertAlmostEqual(s["metrics"]["train_gamma"],2/3)
                selected=[t for t in s["train_spectral"]["types"] if t["real_dimension"]==2 and t["selected_dimension"]]
                self.assertEqual(len(selected),1)
                self.assertEqual((selected[0]["selected_copies"],selected[0]["selected_dimension"]),(1,2))
                self.assertEqual(s["train_spectral"]["budget"]["used_dimension"],3)
                self.assertTrue(all(c["passed"] for c in s["checks"].values()))
                json.dumps(s,allow_nan=False)

    def test_exact_D4_permutation_witness(self):
        Z=np.eye(4)[None]
        s,a=n.witness_comparison(Z,3*Z,"D4")
        self.assertEqual(s["status"],"complete",s.get("reason"))
        self.assertAlmostEqual(s["metrics"]["test_witness_risk"],0,places=10)
        self.assertEqual(a["rho"].shape,(8,4,4))

    def test_D4_tied_E_copies_and_test_uses_train_whitening(self):
        Z=np.array([np.c_[np.eye(4),np.eye(4)],np.c_[np.eye(4),-np.eye(4)]])
        rng=np.random.default_rng(123)
        test=rng.normal(size=(3,4,8))
        s,a=n.witness_comparison(Z,test,"D4")
        self.assertEqual(s["status"],"complete",s.get("reason"))
        actions=n.spectral.group_spec("D4")["actions"]
        manual=n._risk(test@a["train_whitening"],a["rho"],actions)
        self.assertAlmostEqual(s["metrics"]["test_witness_risk"],manual,places=12)
        self.assertGreater(s["metrics"]["test_witness_risk"],s["metrics"]["train_L_upper"]+.1)

    def test_inexact_witness_training_bound_and_raw_map(self):
        rng=np.random.default_rng(131)
        train=rng.normal(size=(12,7,6)); test=rng.normal(size=(4,7,6))
        s,a=n.witness_comparison(train,test,"C7")
        self.assertEqual(s["status"],"complete",s.get("reason"))
        m=s["metrics"]
        self.assertLessEqual(m["train_witness_risk"],m["train_L_upper"]+1e-7)
        self.assertGreaterEqual(m["train_witness_risk"],m["train_L_lower"]-1e-7)
        np.testing.assert_allclose((test@a["raw_coordinate_operators"][1])@a["train_whitening"],
                                   (test@a["train_whitening"])@a["rho"][1],atol=1e-10)

    def test_subgroup_enumeration_unique_complete_counts(self):
        for domain,count in (("C7",120),("D4",3)):
            specs=n.subgroup_specs(domain)
            self.assertEqual(len(specs),count)
            self.assertEqual(sum(s["semantic"] for s in specs),1)
            self.assertEqual(len({str(s["image_permutations"]) for s in specs}),count)

    def test_alternative_target_action_is_used(self):
        rng=np.random.default_rng(4)
        train=rng.normal(size=(8,4,3)); test=rng.normal(size=(5,4,3))
        subs=n.subgroup_specs("D4")
        alt=next(s for s in subs if not s["semantic"])
        s,a=n.alternative_comparison(train,test,"D4",alt)
        direct,_=n.spectral._analyze_spec(test,alt["spec"])
        self.assertEqual(s["test_bank_spectral"]["status"],"complete")
        self.assertAlmostEqual(s["test_bank_spectral"]["metrics"]["gamma"],direct["metrics"]["gamma"])
        self.assertIsNone(s["p_value"])

    def test_alternative_overall_status_requires_both_endpoints(self):
        subgroup={"spec":{},"id":"fixture","semantic":False}
        for witness_status,test_status,expected in (
            ("complete","not_evaluable_rank","completed_with_not_evaluable"),
            ("not_evaluable_witness","complete","not_evaluable_witness"),
            ("complete","complete","complete")):
            with self.subTest(witness=witness_status,test=test_status):
                with mock.patch.object(n,"witness_comparison",return_value=({"status":witness_status},{})), \
                     mock.patch.object(n.spectral,"_analyze_spec",return_value=({"status":test_status},{})):
                    s,a=n.alternative_comparison(None,None,"C7",subgroup)
                self.assertEqual(s["status"],expected)
                self.assertEqual(s["witness_status"],witness_status)
                self.assertEqual(s["test_spectral_status"],test_status)

    def test_overflowing_test_risk_fails_without_nonfinite_metrics(self):
        angle=2*np.pi*np.arange(7)/7
        train=np.stack([np.cos(angle),np.sin(angle)],axis=-1)[None]
        s,a=n.witness_comparison(train,np.ones_like(train)*1e200,"C7")
        self.assertEqual(s["status"],"not_evaluable_witness")
        self.assertNotIn("metrics",s)
        json.dumps(s,allow_nan=False)

    def test_rank_failure_is_not_repaired(self):
        s,a=n.witness_comparison(np.ones((2,7,2)),np.ones((3,7,2)),"C7")
        self.assertEqual(s["status"],"not_evaluable_rank")
        self.assertNotIn("metrics",s)

    def test_simulation_M3_is_exact_shared_not_negative(self):
        train,test,meta=n.simulate_case("C7","M3",n_train=16,n_test=2,noise=0,seed=17)
        self.assertTrue(meta["shared_action_before_noise"])
        self.assertFalse(meta["M3_is_negative"])
        s,a=n.witness_comparison(train,test,"C7")
        self.assertEqual(s["status"],"complete",s.get("reason"))
        self.assertAlmostEqual(s["metrics"]["train_witness_risk"],0,places=9)
        self.assertAlmostEqual(s["metrics"]["test_witness_risk"],0,places=9)

    def test_simulation_generators_reproducible_and_M2_not_certified_null(self):
        for domain,mechanism in (("D4","M1"),("D4","M2"),("D4","M3"),("C7","M2")):
            a,b,meta=n.simulate_case(domain,mechanism,n_train=16,n_test=2,seed=18)
            a2,b2,_=n.simulate_case(domain,mechanism,n_train=16,n_test=2,seed=18)
            np.testing.assert_array_equal(a,a2);np.testing.assert_array_equal(b,b2)
            self.assertFalse(meta["mechanism_is_valid_negative_null"])
            self.assertEqual(meta["context_specific_conjugations"],mechanism=="M2")
            if mechanism=="M2":self.assertIsNone(meta["shared_action_before_noise"])
            else:self.assertTrue(meta["shared_action_before_noise"])
            self.assertEqual(a.shape[2],16)

    def test_descriptive_rank_ties_and_failures(self):
        rows=[{"subgroup_id":"a","semantic_registration":True,"test_bank_spectral":{"status":"complete","metrics":{"gamma":.5}}},
              {"subgroup_id":"b","semantic_registration":False,"test_bank_spectral":{"status":"complete","metrics":{"gamma":.5}}},
              {"subgroup_id":"c","semantic_registration":False,"test_bank_spectral":{"status":"not_evaluable_rank"}}]
        s=n.registration_ranks(rows)
        self.assertEqual((s["rank_low"],s["rank_high"]),(None,None))
        self.assertEqual((s["rank_among_evaluable_low"],s["rank_among_evaluable_high"]),(1,2))
        self.assertEqual((s["all_planned_possible_rank_low"],s["all_planned_possible_rank_high"]),(1,3))
        self.assertTrue(s["failed_subgroups_have_unknown_positions"])
        self.assertEqual(s["failed_spectral"],1)
        self.assertIsNone(s["p_value"])
        self.assertEqual(s["status"],"incomplete_descriptive")


if __name__=="__main__":unittest.main()
