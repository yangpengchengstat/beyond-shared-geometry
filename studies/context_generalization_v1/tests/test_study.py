from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from bank import build_bank,validate_bank
from analysis import blocks,fit_predictors,evaluate,aggregate,action_tables
import run


class BankTests(unittest.TestCase):
    def test_bank_has_complete_unique_paired_states(self):
        rows=build_bank()
        self.assertEqual(validate_bank(rows)['rows'],1408)
        self.assertEqual(sum(r['domain']=='C7' for r in rows),896)
        lookup={(r['domain'],r['family_id'],r['state_id'],r['paraphrase_id']):r for r in rows}
        for r in rows:
            if r['paraphrase_id']:continue
            other=lookup[(r['domain'],r['family_id'],r['state_id'],1)]
            self.assertNotEqual(r['text'][:r['source_span'][0]],other['text'][:other['source_span'][0]])
    def test_invalid_spans_rejected(self):
        rows=build_bank();rows[0]['source_span']=[0,2]
        with self.assertRaises(ValueError):validate_bank(rows)
    def test_missing_or_duplicate_bank_rejected(self):
        rows=build_bank()
        with self.assertRaises(ValueError):validate_bank(rows[:-1])
        rows[-1]=dict(rows[0])
        with self.assertRaises(ValueError):validate_bank(rows)


class EstimationTests(unittest.TestCase):
    def arrays(self,domain='C7'):
        rng=np.random.default_rng(7);p=7 if domain=='C7' else 4
        return rng.normal(size=(8,2,p,3)),rng.normal(size=(4,2,p,3)),rng.normal(size=(5,2,p,3))
    def test_primary_selection_uses_same_context_validation_loss(self):
        train,val,_=self.arrays()
        params,meta=fit_predictors(train,val,'C7',[0,.001,.1,1])
        self.assertEqual(set(params),{'coordinate','coordinate_nativeval','difference'})
        primary=[r for r in meta['selection_ledger'] if 'arm' not in r]
        self.assertTrue(all(r['selection_objective']=='context_difference_validation_mse' for r in primary))
        for r in primary:
            losses=[x['context_mse'] for x in r['validation_candidates']]
            self.assertEqual(r['selected_grid_index'],int(np.argmin(losses)))
    def test_test_values_cannot_change_fits(self):
        train,val,test=self.arrays();grid=[0,.01]
        p,m=fit_predictors(train,val,'C7',grid)
        evaluate(p,m,test,'C7',[f'f{i}' for i in range(5)])
        q,n=fit_predictors(train,val,'C7',grid)
        for k in p:np.testing.assert_array_equal(p[k],q[k])
        self.assertEqual(m,n)
    def test_d4_equal_action_not_pooled_pair(self):
        train,val,test=self.arrays('D4');p,m=fit_predictors(train,val,'D4',[0,.01])
        got=evaluate(p,m,test,'D4',[f'f{i}' for i in range(5)])
        u=test[:,0]-test[:,1];energy=m['training_difference_energy']
        manual=[]
        for t in action_tables('D4'):
            mask=np.arange(4)!=t
            manual.append(((u[:,mask]-u[:,t[mask]])**2).sum(-1).mean(-1)/energy)
        np.testing.assert_allclose([r['identity'] for r in got['per_family']],np.mean(manual,axis=0))
    def test_zero_difference_energy_rejected(self):
        train,val,_=self.arrays();train[:,1]=train[:,0]
        with self.assertRaises(ValueError):fit_predictors(train,val,'C7',[0])
    def test_fit_nonfinite_rejected(self):
        train,val,_=self.arrays();val[0,0,0,0]=np.nan
        with self.assertRaises(ValueError):fit_predictors(train,val,'C7',[0])
    def test_evaluation_malformed_predictor_rejected(self):
        train,val,test=self.arrays();p,m=fit_predictors(train,val,'C7',[0]);p['coordinate']=p['coordinate'][:1]
        with self.assertRaises(ValueError):evaluate(p,m,test,'C7',[f'f{i}' for i in range(5)])
    def test_complete_orbit_selection_and_no_partition_mix(self):
        rows=[r for r in build_bank() if r['domain']=='D4'];z=np.arange(len(rows)*3).reshape(len(rows),3)
        value,families=blocks(z,rows,'D4','test')
        self.assertEqual(value.shape,(64,2,4,3));self.assertEqual(len(families),64)
        with self.assertRaises(ValueError):blocks(z[:-1],rows[:-1],'D4','test')


class SummaryTests(unittest.TestCase):
    def fixture(self):
        families=[f'f{i}' for i in range(4)]
        cells=[{'cell':f'{d}{j}','domain':d,'status':'complete','per_family':[
            {'family_id':f,'difference_minus_coordinate':float(i+j)} for i,f in enumerate(families)]}
            for d in ('C7','D4') for j in range(2)]
        return cells,families
    def test_family_bootstrap_preserves_panel_pairing(self):
        cells,f=self.fixture();s=aggregate(cells,f,expected=4,draws=20,seed=3)
        self.assertEqual(s['contrasts']['primary_equal_domain_panel']['mean_difference_minus_coordinate'],2)
        self.assertEqual(s,aggregate(cells,f,expected=4,draws=20,seed=3))
        self.assertFalse(s['population_confidence_certificate'])
    def test_missing_cell_does_not_form_smaller_primary_panel(self):
        cells,f=self.fixture();cells[-1]['status']='not_evaluable'
        s=aggregate(cells,f,expected=4)
        self.assertIsNone(s['primary_effect']);self.assertEqual(s['status'],'incomplete_panel')
    def test_misaligned_family_order_rejected(self):
        cells,f=self.fixture();cells[0]['per_family'].reverse()
        with self.assertRaises(ValueError):aggregate(cells,f,expected=4)


class WorkflowTests(unittest.TestCase):
    def test_execution_requires_explicit_flags_before_io_or_model_import(self):
        for func in (run.fit,run.extract,run.analyze):
            with self.assertRaises(ValueError):func('/does/not/exist')
    def test_review_hash_and_attestation_required(self):
        with tempfile.TemporaryDirectory() as td,patch.object(run,'_receipt',return_value=({'bank_sha256':'abc','source_manifest_sha256':'source'},[])):
            with self.assertRaises(ValueError):run.review(td,'abc')
            with self.assertRaises(ValueError):run.review(td,'other',True)
            result=run.review(td,'abc',True)
            self.assertTrue(result['approval_before_model_execution'])
            run._check_review(td,{'bank_sha256':'abc','source_manifest_sha256':'source'})
            with self.assertRaises(ValueError):run._check_review(td,{'bank_sha256':'bad','source_manifest_sha256':'source'})
    def test_artifact_mutation_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);f=root/'x';f.write_text('original')
            r=run._commit(root,root/'complete.json','identity',[f])
            run._verify_artifacts(root,r);f.write_text('changed')
            with self.assertRaises(ValueError):run._verify_artifacts(root,r)
    def test_output_existing_refused_before_archive_read(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):run.prepare('missing','0'*64,td)

    def test_mock_frozen_fit_extract_and_analyze_pipeline(self):
        import contextlib
        import io
        from bsg_next.hf import ModelRuntime as RealRuntime
        from bsg_next.data import validate_records
        bank=build_bank();validate_records(bank,require_complete_orbits=True)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);rng=np.random.default_rng(19);cells=[]
            for m in run.input_adapter.protocol()['models']:
                old=RealRuntime(m['id'],revision=m['revision'],batch_size=16,dtype='bfloat16').identity()
                for domain in ('C7','D4'):
                    p=7 if domain=='C7' else 4
                    for layer in m['layers']:
                        cell=f"{m['id'].replace('/','__')}__{domain}__{layer}";folder=root/'inputs'/cell
                        run.save_npz(folder/'development.npz',train=rng.normal(size=(48,2,p,16)),validation=rng.normal(size=(16,2,p,16)))
                        run.save_npz(folder/'pca.npz',center=np.zeros(16),basis=np.eye(16))
                        run.atomic(folder/'old_runtime.json',old)
                        cells.append({'cell':cell,'domain':domain,'layer':layer,'model_id':m['id'],'status':'prepared',
                                      'paths':{n:str((folder/n).relative_to(root)) for n in ('development.npz','pca.npz','old_runtime.json')}})
            record={'cells':cells,'protocol':run.spec(),'bank_sha256':'fixture-bank','source_manifest_sha256':'fixture-source'}
            run.atomic(root/'human_review.json',{'bank_sha256':'fixture-bank','source_manifest_sha256':'fixture-source'})
            def fake_runtime(model_id,revision,**kwargs):
                old=next(c for c in cells if c['model_id']==model_id)
                identity=run.read_json(root/old['paths']['old_runtime.json'])
                class Fake:
                    telemetry=[]
                    def identity(self):return identity
                    def close(self):pass
                    def features(self,records,layers,readout):
                        for start in range(0,len(records),32):
                            batch=records[start:start+32]
                            values=np.asarray([[r['state_id']*.1+r['paraphrase_id']*.03+len(r['text'])*.0001+k*.02 for k in range(16)] for r in batch],dtype=np.float32)
                            yield range(start,start+len(batch)),{layer:values+layer*.001 for layer in layers}
                return Fake()
            with patch.object(run,'_receipt',return_value=(record,bank)),contextlib.redirect_stdout(io.StringIO()):
                run.fit(root,True)
                fitted=run.file_sha(root/'fits_manifest.json')
                premature=run.analyze(root,True)
                self.assertEqual(premature['status'],'incomplete_panel')
                self.assertFalse((root/'analysis_complete.json').exists())
                attempts=list((root/'analysis_attempts').glob('*/receipt.json'))
                self.assertEqual(len(attempts),1)
                attempt_bytes=attempts[0].read_bytes()
                with patch('bsg_next.hf.ModelRuntime',side_effect=fake_runtime):
                    run.extract(root,True)
                    run.extract(root,True)  # cache hit and committed projections are verified
                self.assertEqual(fitted,run.file_sha(root/'fits_manifest.json'))
                result=run.analyze(root,True)
                self.assertEqual(result['status'],'complete')
                self.assertEqual(result['completed_cells'],48)
                self.assertEqual(result,run.analyze(root,True))
                self.assertEqual(len(list((root/'new_coordinates').glob('*/complete.json'))),48)
                self.assertEqual(attempts[0].read_bytes(),attempt_bytes)
                first=next((root/'new_coordinates').glob('*/coordinates.npz'))
                first.write_bytes(b'changed after analysis')
                with self.assertRaises(ValueError):run.analyze(root,True)

    def test_collect_is_compact_and_checksums_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td,patch.object(run,'_receipt',return_value=({},[])):
            root=Path(td)/'run';root.mkdir();(root/'state.json').write_text('{}')
            raw=root/'extraction';raw.mkdir();(raw/'huge.npy').write_bytes(b'not-in-export')
            run.atomic(root/'inputs'/'cell'/'old_runtime.json',{'dtype':'bfloat16','model_revision':'pinned'})
            run.atomic(raw/'model'/'manifest.json',{'model':{'dtype':'bfloat16'},'chunks':['excluded.npy']})
            (raw/'model'/'excluded.npy').write_bytes(b'raw-features-not-in-export')
            (root/'inputs'/'cell'/'development.npz').write_bytes(b'old-training-not-in-export')
            out=Path(td)/'out.zip';result=run.collect(root,out)
            self.assertFalse(result['raw_hidden_arrays_included'])
            import zipfile
            with zipfile.ZipFile(out) as z:
                self.assertEqual(set(z.namelist()),{'state.json','export_manifest.json',
                    'inputs/cell/old_runtime.json','extraction/model/manifest.json'})
                self.assertEqual(json.loads(z.read('inputs/cell/old_runtime.json'))['dtype'],'bfloat16')
                self.assertEqual(json.loads(z.read('extraction/model/manifest.json'))['model']['dtype'],'bfloat16')
                manifest=json.loads(z.read('export_manifest.json'))
                self.assertFalse(manifest['raw_feature_cache_complete'])
                self.assertFalse(manifest['artifact_validation']['scientific_pass'])
                self.assertEqual(manifest['artifact_validation']['analysis_receipt']['status'],'missing')
                for name,digest in manifest['files'].items():self.assertEqual(run.sha(z.read(name)),digest)
            self.assertEqual(out.with_suffix('.zip.sha256').read_text().split()[0],run.file_sha(out))
            with self.assertRaises(ValueError):run.collect(root,out)

    def test_collect_marks_corrupt_completed_receipt_invalid_but_preserves_it(self):
        import zipfile
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run';root.mkdir()
            record={'cells':[{'cell':'fixture'}]};fit_binding=run._binding(record)
            run.atomic(root/'human_review.json',{'fixture':'software test, not a human approval'})
            run.atomic(root/'fits'/'fixture'/'fit.json',{'fixture':True})
            fit=run._commit(root,root/'fits'/'fixture'/'complete.json',fit_binding,
                            [root/'fits'/'fixture'/'fit.json'],{'cell':'fixture'})
            frozen={'binding':fit_binding,'cells':[fit],'review_sha256':run.file_sha(root/'human_review.json')}
            run.atomic(root/'fits_manifest.json',frozen)
            binding=run.sha(run.canonical({'inputs':fit_binding,'fits':frozen}))
            run.atomic(root/'new_coordinates'/'fixture'/'rows.json',{'fixture':True})
            receipt=root/'new_coordinates'/'fixture'/'complete.json'
            run._commit(root,receipt,binding,[receipt.parent/'rows.json'])
            run.atomic(root/'summary.json',{'status':'complete'})
            run._commit(root,root/'analysis_complete.json',binding,[root/'summary.json'],{'status':'complete'})
            valid=run._export_validation(root,record)
            self.assertEqual(valid['analysis_receipt']['status'],'verified')
            self.assertEqual(valid['fits_manifest']['status'],'verified')
            damaged=run.read_json(receipt);damaged['binding']='changed';run.atomic(receipt,damaged)
            damaged_bytes=receipt.read_bytes()
            with patch.object(run,'_receipt',return_value=(record,[])):
                out=Path(td)/'damaged.zip';run.collect(root,out)
            with zipfile.ZipFile(out) as z:
                checks=json.loads(z.read('export_manifest.json'))['artifact_validation']
                self.assertEqual(checks['new_coordinate_receipts']['fixture']['status'],'invalid')
                self.assertEqual(checks['analysis_receipt']['status'],'invalid')
                self.assertEqual(checks['fits_manifest']['status'],'verified')
                self.assertFalse(checks['scientific_pass'])
                self.assertEqual(z.read('new_coordinates/fixture/complete.json'),damaged_bytes)

if __name__=='__main__':unittest.main()
