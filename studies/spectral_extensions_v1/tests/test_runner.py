"""Transaction and interface fixtures; never read real research archives."""
from __future__ import annotations
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('bsg_extension_runner_test', HERE/'runner.py')
runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
base = runner.base


def encoded(v):return json.dumps(v,sort_keys=True,indent=2).encode()+b'\n'

def npz(**kw):
    f=io.BytesIO();np.savez_compressed(f,**kw);return f.getvalue()


def fixture(root, *, missing_train=False):
    parent='fixture/characterization'
    model={'id':'fixture/model','revision':'a'*40,'tokenizer_revision':'b'*40,'hidden_dimension':6,'layers':[1]}
    frozen=encoded({'fixture.py':'c'*64})
    contract={'models':[model],'study_id':'adapter_fixture','driver_sha256':'d'*64,'frozen_source_manifest_sha256':base.sha(frozen)}
    raw_contract=encoded(contract)
    ps={'archive_prefix':parent,'study_id':'adapter_fixture','driver_sha256':'d'*64,'study_contract_sha256':base.sha(raw_contract),
        'study_contract_id':runner.digest(contract),'frozen_source_manifest_sha256':base.sha(frozen),'models':[model],
        'domains':['C7'],'planned_cells':1,'rank':4,'partition':'test','context_blocks_per_cell':4,
        'test_families':['structural_test_000','structural_test_001'],'paraphrase_ids':[0,1],'readout':'span_mean'}
    rows=[]
    for partition in ('train','test'):
        for i in range(2):
            for paraphrase in (0,1):
                for state in range(7):
                    family=f'structural_{partition}_{i:03d}'
                    rows.append({'domain':'C7','family_id':family,'paraphrase_id':paraphrase,'orbit_id':'lexical_C7',
                                 'split':partition,'state_id':state,'row_id':f'C7/{family}/p{paraphrase}/lexical_C7/s{state}'})
    if missing_train:rows.pop(0)
    z=np.random.default_rng(77).normal(size=(len(rows),4)).astype(np.float64)
    identity={'model_id':model['id'],'model_revision':model['revision'],'tokenizer_revision':model['tokenizer_revision']}
    merged={'model':identity,'readout':'span_mean','status':'complete','layers':[1],'n_rows':len(rows),
            'arrays':{'1':{'shape':[len(rows),6]}},'descriptor':{'model':identity,'readout':'span_mean','layers':[1]}}
    raw_merged=encoded(merged)
    directory=parent+'/analysis/fixture__model__C7__1'
    contents={'rows.json':encoded({'rows':rows}),'coordinates.npz':npz(z=z),
              'pca.npz':npz(center=np.zeros(6),basis=np.eye(6,4))}
    contents['cell_complete.json']=encoded({'artifacts':{k:base.sha(v) for k,v in contents.items()},
        'binding':{'contract_id':ps['study_contract_id'],'domain':'C7','layer':1,
                   'numerical_source_manifest_sha256':ps['frozen_source_manifest_sha256'],'cache_manifest_sha256':base.sha(raw_merged)}})
    archive=root/'fixture.zip'
    with zipfile.ZipFile(archive,'w') as out:
        out.writestr(parent+'/study_contract.json',raw_contract)
        out.writestr(parent+'/provenance/frozen_source_manifest.json',frozen)
        out.writestr(parent+'/provenance/fixture__model__C7__merged.json',raw_merged)
        for name,data in contents.items():out.writestr(directory+'/'+name,data)
    prepared=root/'parent-prepared'
    base.prepare(archive,base.file_sha(archive),prepared,_spec=ps,_skip_source_lock=True)
    ext=copy.deepcopy(runner.protocol())
    ext.update(planned_cells=1,domains=['C7'],rank=4,training_families=2,training_context_blocks=4,test_context_blocks=4)
    ext['alternatives']['planned_units']=120
    return archive,prepared,ext


def simulation_protocol(n=2):
    result=copy.deepcopy(runner.protocol())
    result['simulation'].update(domains=['C7'],mechanisms=['M1'],noise_levels=[0.0],seeds=list(range(10,10+n)),planned_units=n)
    return result


def fake_result(stage,unit):
    return {'status':'complete','metrics':{'train_gamma':0.5,'test_witness_risk':0.2}}, {'small_fixture':np.arange(3,dtype=float)}


class ExtensionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_atomic_record_failure_does_not_publish_or_replace_receipt(self):
        path=self.root/'receipt.json'
        with self.assertRaises(ValueError):
            runner.write_json(path,{'invalid':float('nan')})
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.glob('*.partial-*')),[])
        runner.write_json(path,{'status':'complete'})
        saved=path.read_bytes()
        with self.assertRaises(FileExistsError):
            runner.write_json(path,{'status':'different'})
        self.assertEqual(path.read_bytes(),saved)

    def test_prepare_reuses_verified_train_test_without_spectrum(self):
        archive,parent,ext=fixture(self.root)
        with mock.patch.object(runner.numerics,'witness_comparison',side_effect=AssertionError('no analysis in prepare')):
            manifest=runner.prepare(parent,archive,self.root/'prepared',_fixture=True,_protocol=ext)
        inputs=runner.json_file(self.root/'prepared/inputs.json')
        self.assertEqual(inputs['cells'][0]['status'],'prepared_inputs_only')
        self.assertFalse(inputs['coordinate_spectra_computed'])
        with np.load(self.root/'prepared/cells/fixture__model__C7__1/banks.npz',allow_pickle=False) as data:
            self.assertEqual(data['train'].shape,(4,7,4));self.assertEqual(data['test'].shape,(4,7,4))
        self.assertIn('inputs.json',manifest['files'])

    def test_missing_train_state_retained_as_failed_cell(self):
        archive,parent,ext=fixture(self.root,missing_train=True)
        runner.prepare(parent,archive,self.root/'prepared',_fixture=True,_protocol=ext)
        inputs=runner.json_file(self.root/'prepared/inputs.json')
        self.assertEqual(len(inputs['cells']),1)
        self.assertEqual(inputs['cells'][0]['status'],'not_evaluable_input')
        run=self.root/'run'
        with mock.patch.object(runner.numerics,'witness_comparison',side_effect=AssertionError('no invalid fit')):
            result=runner.run('witness',run,prepared=self.root/'prepared',execute_analysis=True,_fixture=True,_protocol=ext)
        self.assertEqual(result['reported_terminal'],1);self.assertEqual(result['successful'],0)
        self.assertEqual(result['not_evaluable_units'],1)
        state=runner.json_file(run/'state.json')
        self.assertEqual(state['status'],'completed_with_not_evaluable')
        self.assertEqual(state['execution_status'],'complete')
        collection=runner.collect(run,self.root/'not-evaluable-completed.zip')
        self.assertFalse(collection['partial']);self.assertTrue(collection['execution_complete'])
        self.assertEqual(collection['not_evaluable_units'],1)
        self.assertFalse(collection['scientific_pass'])
        self.assertFalse(collection['scientific_pass_assessed'])

    def test_complete_witness_fixture_real_core_and_relocatable_resume(self):
        archive,parent,ext=fixture(self.root)
        runner.prepare(parent,archive,self.root/'prepared',_fixture=True,_protocol=ext)
        result=runner.run('witness',self.root/'run',prepared=self.root/'prepared',execute_analysis=True,_fixture=True,_protocol=ext)
        self.assertEqual(result['reported_terminal'],1)
        self.assertEqual(result['successful'],1)
        moved=self.root/'relocated';(self.root/'run').rename(moved)
        # Its input snapshot is sufficient; external preparation need not remain.
        with mock.patch.object(runner.numerics,'witness_comparison',side_effect=AssertionError('no repeat')):
            again=runner.run('witness',moved,execute_analysis=True,resume=True,_fixture=True,_protocol=ext)
        self.assertEqual(again['reported_terminal'],1)
        self.assertEqual(runner.json_file(moved/'state.json')['execution_status'],'complete')
        self.assertFalse(runner.collect(moved,self.root/'complete.zip')['partial'])

    def test_explicit_execution_and_resume_required(self):
        ext=simulation_protocol(1)
        with self.assertRaisesRegex(runner.InputError,'execute-analysis'):
            runner.run('simulate',self.root/'run',_fixture=True,_protocol=ext)
        runner.run('simulate',self.root/'run',execute_analysis=True,_fixture=True,_protocol=ext,_executor=fake_result)
        with self.assertRaisesRegex(runner.InputError,'explicit --resume'):
            runner.run('simulate',self.root/'run',execute_analysis=True,_fixture=True,_protocol=ext)
        with self.assertRaisesRegex(runner.InputError,'missing run'):
            runner.run('simulate',self.root/'missing',execute_analysis=True,resume=True,_fixture=True,_protocol=ext)

    def test_partial_failure_resume_preserves_completed_unit_and_attempt(self):
        ext=simulation_protocol(2);calls=[]
        def first(stage,unit):
            calls.append(unit['unit'])
            if len(calls)==2:raise RuntimeError('fixture interruption')
            return fake_result(stage,unit)
        root=self.root/'run'
        with self.assertRaisesRegex(RuntimeError,'fixture interruption'):
            runner.run('simulate',root,execute_analysis=True,_fixture=True,_protocol=ext,_executor=first)
        old=list(root.glob('units/*/receipt.json'))[0];old_hash=base.file_sha(old)
        failed=list(root.glob('attempts/*/*/failure.json'));self.assertEqual(len(failed),1)
        resume_calls=[]
        def second(stage,unit):resume_calls.append(unit['unit']);return fake_result(stage,unit)
        result=runner.run('simulate',root,execute_analysis=True,resume=True,_fixture=True,_protocol=ext,_executor=second)
        self.assertEqual(len(resume_calls),1);self.assertEqual(result['reported_terminal'],2)
        self.assertEqual(base.file_sha(old),old_hash);self.assertTrue(failed[0].is_file())

    def test_corrupted_checkpoint_refused_and_not_replaced(self):
        ext=simulation_protocol(1);root=self.root/'run'
        runner.run('simulate',root,execute_analysis=True,_fixture=True,_protocol=ext,_executor=fake_result)
        summary=next(root.glob('attempts/*/*/summary.json'));summary.write_text('{}')
        with self.assertRaisesRegex(runner.InputError,'checksum mismatch'):
            runner.run('simulate',root,execute_analysis=True,resume=True,_fixture=True,_protocol=ext,_executor=fake_result)
        self.assertEqual(summary.read_text(),'{}')

    def test_changed_protocol_or_runtime_refused_on_resume(self):
        ext=simulation_protocol(1);root=self.root/'run'
        runner.run('simulate',root,execute_analysis=True,_fixture=True,_protocol=ext,_executor=fake_result)
        changed=copy.deepcopy(ext);changed['simulation']['noise_levels']=[0.05]
        with self.assertRaisesRegex(runner.InputError,'source/protocol/stage'):
            runner.run('simulate',root,execute_analysis=True,resume=True,_fixture=True,_protocol=changed)

    def test_prepare_overwrite_and_input_mutation_refused(self):
        archive,parent,ext=fixture(self.root)
        runner.prepare(parent,archive,self.root/'prepared',_fixture=True,_protocol=ext)
        with self.assertRaisesRegex(runner.InputError,'exists'):
            runner.prepare(parent,archive,self.root/'prepared',_fixture=True,_protocol=ext)
        bank=self.root/'prepared/cells/fixture__model__C7__1/banks.npz'
        with bank.open('ab') as f:f.write(b'bad')
        with self.assertRaisesRegex(runner.InputError,'checksum mismatch'):
            runner.run('witness',self.root/'run',prepared=self.root/'prepared',execute_analysis=True,_fixture=True,_protocol=ext)
        self.assertFalse((self.root/'run').exists())

    def test_collect_partial_run_and_refuse_overwrite_and_active_run(self):
        ext=simulation_protocol(2);root=self.root/'run'
        def interrupted(stage,unit):raise RuntimeError('fixture fail')
        with self.assertRaises(RuntimeError):
            runner.run('simulate',root,execute_analysis=True,_fixture=True,_protocol=ext,_executor=interrupted)
        with runner.controller(root):
            with self.assertRaisesRegex(runner.InputError,'active'):
                runner.collect(root,self.root/'active.zip')
        out=self.root/'partial.zip';result=runner.collect(root,out)
        self.assertTrue(result['partial']);self.assertEqual(result['terminal_units'],0)
        self.assertEqual(out.with_name(out.name+'.sha256').read_text().split()[0],base.file_sha(out))
        with zipfile.ZipFile(out) as z:
            self.assertIn('collection_manifest.json',z.namelist())
            self.assertTrue(any(n.endswith('/failure.json') for n in z.namelist()))
        with self.assertRaisesRegex(runner.InputError,'exists'):
            runner.collect(root,out)
        with self.assertRaisesRegex(runner.InputError,'outside'):
            runner.collect(root,root/'inside.zip')

    def test_complete_alternative_plan_no_silent_group_drop(self):
        archive,parent,ext=fixture(self.root)
        runner.prepare(parent,archive,self.root/'prepared',_fixture=True,_protocol=ext)
        seen=[]
        def one(stage,unit):
            seen.append(unit['subgroup_id'])
            return {'status':'not_evaluable_witness','test_bank_spectral':{'status':'not_evaluable_rank'}},{}
        result=runner.run('alternatives',self.root/'run',prepared=self.root/'prepared',execute_analysis=True,_fixture=True,_protocol=ext,_executor=one)
        self.assertEqual(result['reported_terminal'],120);self.assertEqual(len(set(seen)),120)
        report=json.loads(next((self.root/'run/reports').glob('*/descriptive_ranks.json')).read_text())
        self.assertEqual(report['cells']['fixture__model__C7__1']['planned'],120)
        self.assertFalse(report['p_values'])

    def test_source_snapshot_tampering_and_complete_count_failed_status(self):
        lock={"files":{},"base_files":{}}
        for folder,field in (("extension","extension_manifest_sha256"),("base","base_manifest_sha256")):
            directory=self.root/'snapshot-run/source_snapshot'/folder;directory.mkdir(parents=True)
            (directory/'source_manifest.json').write_text('{}')
            lock[field]=base.file_sha(directory/'source_manifest.json')
        runner._verify_source_snapshot(self.root/'snapshot-run',lock)
        (self.root/'snapshot-run/source_snapshot/base/source_manifest.json').write_text('changed')
        with self.assertRaisesRegex(runner.InputError,'snapshot'):
            runner._verify_source_snapshot(self.root/'snapshot-run',lock)
        ext=simulation_protocol(1);root=self.root/'run'
        runner.run('simulate',root,execute_analysis=True,_fixture=True,_protocol=ext,_executor=fake_result)
        runner.atomic_json(root/'state.json',{"status":"source_changed","results_eligible":False})
        collection=runner.collect(root,self.root/'flagged.zip')
        self.assertEqual(collection['terminal_units'],1)
        self.assertTrue(collection['partial']);self.assertFalse(collection['execution_complete'])

    def test_production_source_lock_required(self):
        with mock.patch.object(runner,'HERE',self.root),mock.patch.object(runner.base,'source_lock',return_value={'manifest_sha256':'0'*64,'files':{}}):
            with self.assertRaisesRegex(runner.InputError,'Missing extension'):
                runner.source_lock()


if __name__=='__main__':unittest.main()
