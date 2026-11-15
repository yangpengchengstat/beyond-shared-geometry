"""Model-free tests for the separate, NumPy-only coordinate refitter."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location('independent_refit', HERE.parents[1]/'tools/refit_coordinates.py')
REFIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFIT)


def fixture(domain='C7'):
    """Disclosed synthetic lexical one-hots plus shared contextual coordinates."""
    elements, identity, generators, tables, words = REFIT.action_catalog(domain)
    states = len(tables[identity]); rng = np.random.default_rng(901)
    rows = []; vectors = []
    for split, count in [('train',8), ('validation',2), ('test',2)]:
        for family in range(count):
            for paraphrase in range(2):
                context = rng.normal(scale=.1, size=16-states)
                for state in range(states):
                    rows.append({'row_id':f'{split}-{family}-{paraphrase}-{state}',
                                 'family_id':f'{split}-{family}', 'split':split,
                                 'state_id':state, 'paraphrase_id':paraphrase,
                                 'orbit_id':'synthetic', 'domain':domain})
                    vectors.append(np.r_[np.eye(states)[state], context])
    return np.asarray(vectors), rows


class RefitterTests(unittest.TestCase):
    def test_literal_actions_order_and_alias_masks(self):
        _, identity, generators, tables, words = REFIT.action_catalog('D4_corner')
        r,s = generators
        def endpoint(word):
            result = np.arange(4)
            for g in word: result = np.asarray(tables[g])[result]
            return result
        np.testing.assert_array_equal(endpoint([r]*4), np.arange(4))
        np.testing.assert_array_equal(endpoint([s]*2), np.arange(4))
        self.assertEqual(np.count_nonzero(endpoint([s,r,s]) != endpoint([r,s,r])), 2)
        self.assertEqual(np.count_nonzero(endpoint([r,s]) != endpoint([s,r])), 4)
        _,_,_,s3_tables,_ = REFIT.action_catalog('S3_role')
        self.assertEqual(len(set(s3_tables.values())), 6)
        for table in s3_tables.values(): self.assertEqual(sorted(table), list(range(6)))

    def test_exact_direct_actions_all_domains(self):
        for domain in ('C7','D4_corner','S3_role'):
            z, rows = fixture(domain)
            result,_ = REFIT.direct_refit(z, rows, domain, [0.,.01,1.], 'test')
            self.assertLess(result['mean_losses']['primary_operator'], 1e-20)
            self.assertEqual(result['selection']['normalized_lambda'], 0.)

    def test_ridge_normalization_and_tie_rule(self):
        x=np.array([[1.,0.],[0.,2.]])
        result=REFIT.ridge(x,3*x,.4)
        # alpha = .4*(1+4)/2 = 1; solve diagonal normal equations.
        np.testing.assert_allclose(result,np.diag([1.5,2.4]),atol=1e-12)
        z,rows=fixture('C7');z*=1e-5
        # Absolute validation tolerance makes both eligible; smaller norm wins.
        _,_,_,tables,_=REFIT.action_catalog('C7')
        _,meta=REFIT.select_ridge(z,rows,REFIT.paired_targets(rows,tables),tuple(range(7)),[0.,1.])
        self.assertEqual(meta['normalized_lambda'],1.)

    def test_primitive_composition_and_cycle_targets(self):
        for domain in ('C7','D4_corner'):
            z, rows=fixture(domain)
            result=REFIT.generator_refit(z,rows,domain,[0.,.01])
            self.assertLess(result['mean_losses']['unseen_composite']['generator_ridge'],1e-20)
            self.assertTrue(all(v['mean_losses']['generator_ridge']<1e-20 for v in result['diagnostics'].values()))
            if domain=='D4_corner':
                self.assertEqual(result['order_tests']['relation_srs_vs_rsr']['n_pairs'],8)
                self.assertEqual(result['order_tests']['order_rs_vs_sr']['n_pairs'],16)
                self.assertLess(result['order_tests']['order_rs_vs_sr']['methods']['generator_ridge']['correct_minus_wrong'],0)

    def test_explicit_kernel_and_farthest_first(self):
        x=np.eye(7);target=x[np.roll(np.arange(7),-1)]
        field=REFIT.Field(x,{1:target});field.bandwidth=.125
        np.testing.assert_allclose(field.predict(x,1),target,atol=1e-10)
        # Equidistant candidates break ties by original index, full N is original order.
        np.testing.assert_array_equal(REFIT.farthest_first(np.eye(4),3),[0,1,2])
        np.testing.assert_array_equal(REFIT.farthest_first(np.eye(4),4),[0,1,2,3])
        single=REFIT.Field(np.zeros((1,2)),{1:np.ones((1,2))})
        np.testing.assert_array_equal(single.predict(np.zeros((2,2)),1),np.ones((2,2)))

    def test_test_values_do_not_select_hyperparameters_or_support(self):
        z,rows=fixture('D4_corner');changed=z.copy()
        test=np.array([r['split']=='test' for r in rows]);changed[test]+=1000
        before=REFIT.generator_refit(z,rows,'D4_corner',[0.,.01])
        after=REFIT.generator_refit(changed,rows,'D4_corner',[0.,.01])
        self.assertEqual(before['ridge_selection'],after['ridge_selection'])
        self.assertEqual(before['kernel_selection'],after['kernel_selection'])
        curves=REFIT.storage_refit(z,rows,'D4_corner',[0.,.01])
        for entry in curves:
            actions=8 if entry['track']=='all_element' else 2
            self.assertEqual(entry['counted_float64_coordinates'],entry['K']*16*(actions+1))
            self.assertEqual(entry['counted_predictor_array_bytes'],entry['counted_float64_coordinates']*8)
            self.assertEqual(entry['capacity_or_total_RAM_matched'],False)
            self.assertEqual(entry['selected_training_indices'],REFIT.farthest_first(z[np.array([r['split']=='train' for r in rows])],entry['K']).tolist())

    def test_strict_mismatch_and_cross_partition_rejection(self):
        checks=REFIT.Checks()
        with self.assertRaises(REFIT.AuditError):checks.equal({'mean':.1},{'mean':.2},'tamper')
        z,rows=fixture('C7');bad=[dict(r) for r in rows];bad[-1]['split']='train'
        _,_,_,tables,_=REFIT.action_catalog('C7')
        with self.assertRaises(REFIT.AuditError):REFIT.paired_targets(bad,tables)
        bad=[dict(r) for r in rows];bad[-1]['row_id']=bad[0]['row_id']
        with self.assertRaises(REFIT.AuditError):REFIT.paired_targets(bad,tables)

    def test_raw_schema_and_tampered_metric_rejection(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as directory:
            raw=Path(directory)/'raw'; cell=raw/'analysis'/'synthetic_C7'
            cell.mkdir(parents=True); (raw/'provenance').mkdir()
            config={'study_id':'icml2027-characterization-v1','structural':{'lambda_grid':[0.,.01]}}
            z,rows=fixture('C7')
            direct,_=REFIT.direct_refit(z,rows,'C7',[0.,.01],'test')
            expected={**direct,'group':'C7','test_split':'test'}
            generator=REFIT.generator_refit(z,rows,'C7',[0.,.01])
            fields={'generator':generator,'input_comparisons':{
                'storage_curves':REFIT.storage_refit(z,rows,'C7',[0.,.01]),
                'controls':{'primary_operator':{'mean_loss':direct['mean_losses']['primary_operator']}}}}
            for path,value in [(raw/'provenance/study_config.json',config),
                               (cell/'rows.json',{'rows':rows}),
                               (cell/'legacy.json',[expected]),(cell/'fields.json',fields)]:
                path.write_text(json.dumps(value))
            np.savez_compressed(cell/'coordinates.npz',z=z)
            report=REFIT.audit_raw(raw)
            self.assertFalse(report['paper_eligible']);self.assertEqual(report['cells_audited'],1)
            self.assertGreater(report['numeric_checks'],200)
            fields['generator']['mean_losses']['unseen_composite']['generator_ridge']+=.01
            (cell/'fields.json').write_text(json.dumps(fields))
            with self.assertRaises(REFIT.AuditError):REFIT.audit_raw(raw)
            output=Path(directory)/'occupied.json';output.write_text('preserve')
            with self.assertRaises(REFIT.AuditError):REFIT.main(['--raw',str(raw),'--output',str(output)])
            self.assertEqual(output.read_text(),'preserve')

    def test_nested_symlinks_rejected_before_content_access(self):
        # Link destinations are deliberately nonexistent; no outside content is
        # created or inspected. The mocked read/open prove rejection first.
        destination=Path('/__bsg_audit_nonexistent_outside_target__')
        config={'study_id':'icml2027-characterization-v1','structural':{'lambda_grid':[0.]}}
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as directory:
            root=Path(directory)
            for kind in ('config','analysis','legacy'):
                raw=root/kind;raw.mkdir();(raw/'provenance').mkdir()
                if kind=='config':
                    (raw/'provenance/study_config.json').symlink_to(destination)
                    with mock.patch.object(Path,'read_text',side_effect=AssertionError('content read')):
                        with self.assertRaises(REFIT.AuditError):REFIT.audit_raw(raw)
                elif kind=='analysis':
                    (raw/'analysis').symlink_to(destination,target_is_directory=True)
                    with mock.patch.object(Path,'iterdir',side_effect=AssertionError('outside listing')):
                        with self.assertRaises(REFIT.AuditError):REFIT.audit_raw(raw,config)
                else:
                    folder=raw/'analysis/cell';folder.mkdir(parents=True)
                    (folder/'legacy.json').symlink_to(destination)
                    with mock.patch.object(Path,'read_text',side_effect=AssertionError('content read')):
                        with self.assertRaises(REFIT.AuditError):REFIT.audit_raw(raw,config)
            linked=root/'linked.json';linked.symlink_to(destination)
            with mock.patch.object(Path,'open',side_effect=AssertionError('content read')):
                with self.assertRaises(REFIT.AuditError):REFIT.sha(linked)

    def test_import_receipt_rejects_changed_raw_bytes(self):
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as directory:
            receipt_dir=Path(directory);raw=receipt_dir/'raw';raw.mkdir()
            data=raw/'sample.json';data.write_text('{"synthetic":true}')
            receipt={'study_id':'icml2027-characterization-v1','archive_integrity':'passed',
                     'source_contract':'matches_frozen_v1','panel_validation':'complete','imported':True,
                     'import_status':'imported_complete_pending_scientific_audit','paper_eligible':False,
                     'member_sha256':{'sample.json':REFIT.sha(data)},'input_sha256':'synthetic-only'}
            (receipt_dir/'receipt.json').write_text(json.dumps(receipt))
            self.assertEqual(REFIT.verify_receipt(raw)['member_count'],1)
            data.write_text('{"synthetic":false}')
            with self.assertRaises(REFIT.AuditError):REFIT.verify_receipt(raw)

    def test_path_boundary_and_no_torch(self):
        with self.assertRaises(REFIT.AuditError):REFIT.bounded_path(REFIT.PROJECT_ROOT.parent/'outside_project_fixture.zip')
        with tempfile.TemporaryDirectory(dir=HERE.parents[1]) as directory:
            root=Path(directory); inside=root/'inside';inside.mkdir()
            self.assertEqual(REFIT.bounded_path(inside,root),inside)
            link=root/'linked';link.symlink_to(inside, target_is_directory=True)
            with self.assertRaises(REFIT.AuditError):REFIT.bounded_path(link,root)
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('transformers',sys.modules)


if __name__=='__main__':unittest.main()
