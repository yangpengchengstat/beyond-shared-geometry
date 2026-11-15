"""Small software fixtures only: no real coordinate archive or model is loaded."""
from __future__ import annotations
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))
import runner


def encode(value):
    return json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"


def npz(**arrays):
    f = io.BytesIO()
    np.savez_compressed(f, **arrays)
    return f.getvalue()


def fixture(root, mutation=None, *, missing=False, duplicate=False, unsafe=False):
    # A reduced, private adapter fixture, never accepted by the public CLI protocol.
    prefix = 'fixture/characterization'
    model = {'id': 'fixture/model', 'revision': 'a'*40, 'tokenizer_revision': 'b'*40, 'hidden_dimension': 6, 'layers': [1]}
    frozen = encode({'test_source.py': 'c'*64})
    contract = {'models': [model], 'study_id': 'software_fixture', 'driver_sha256': 'd'*64, 'frozen_source_manifest_sha256': runner.sha(frozen)}
    contract_raw = encode(contract)
    spec = {'archive_prefix': prefix, 'study_id': 'software_fixture', 'driver_sha256': 'd'*64,
            'study_contract_sha256': runner.sha(contract_raw), 'study_contract_id': runner.sha(runner.canonical(contract)),
            'frozen_source_manifest_sha256': runner.sha(frozen), 'models': [model], 'domains': ['C7'], 'planned_cells': 1,
            'rank': 4, 'partition': 'test', 'context_blocks_per_cell': 4,
            'test_families': ['structural_test_000', 'structural_test_001'], 'paraphrase_ids': [0, 1], 'readout': 'span_mean'}
    rows = [{'domain': 'C7', 'family_id': family, 'paraphrase_id': para, 'orbit_id': 'lexical_C7', 'split': 'test', 'state_id': s,
             'row_id': f'C7/{family}/p{para}/lexical_C7/s{s}'}
            for family in spec['test_families'] for para in spec['paraphrase_ids'] for s in range(7)]
    Z = np.random.default_rng(12).normal(size=(len(rows), 4)).astype(np.float64)
    center = np.zeros(6, dtype=np.float64)
    basis = np.eye(6, 4, dtype=np.float64)
    identity = {'model_id': model['id'], 'model_revision': model['revision'], 'tokenizer_revision': model['tokenizer_revision']}
    merged = {'model': identity, 'readout': 'span_mean', 'status': 'complete', 'layers': [1], 'n_rows': len(rows),
              'arrays': {'1': {'shape': [len(rows), 6]}}, 'descriptor': {'model': identity.copy(), 'readout': 'span_mean', 'layers': [1]}}
    state = {'rows': rows, 'z': Z, 'center': center, 'basis': basis, 'merged': merged}
    if mutation:
        mutation(state)
    merged_raw = encode(state['merged'])
    cell = 'fixture__model__C7__1'
    directory = prefix + '/analysis/' + cell
    payloads = {'coordinates.npz': npz(z=state['z']), 'pca.npz': npz(center=state['center'], basis=state['basis']), 'rows.json': encode({'rows': state['rows']})}
    receipt = {'schema_version': 1, 'artifacts': {k: runner.sha(v) for k,v in payloads.items()},
               'binding': {'contract_id': spec['study_contract_id'], 'domain': 'C7', 'layer': 1,
                           'numerical_source_manifest_sha256': spec['frozen_source_manifest_sha256'], 'cache_manifest_sha256': runner.sha(merged_raw)}}
    payloads['cell_complete.json'] = encode(receipt)
    members = {prefix+'/study_contract.json': contract_raw, prefix+'/provenance/frozen_source_manifest.json': frozen,
               prefix+'/provenance/fixture__model__C7__merged.json': merged_raw}
    members.update({directory+'/'+k:v for k,v in payloads.items()})
    if missing:
        members.pop(directory+'/coordinates.npz')
    path = root/'fixture.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name,data in members.items():
            z.writestr(name,data)
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                z.writestr(directory+'/rows.json', payloads['rows.json'])
        if unsafe:
            z.writestr('../escape.txt', b'bad')
    return path, runner.file_sha(path), spec


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def prepare_fixture(self, **kw):
        p,h,spec = fixture(self.root, **kw)
        output = self.root/'prepared'
        result = runner.prepare(p,h,output,_spec=spec,_skip_source_lock=True)
        return p,h,spec,output,result

    def test_prepare_is_metadata_only_and_preserves_input(self):
        p,h,spec,output,result = self.prepare_fixture()
        self.assertEqual(result['cells'][0]['status'], 'prepared_metadata_only')
        self.assertFalse(result['coordinate_values_analyzed'])
        self.assertEqual(runner.file_sha(p), h)
        self.assertEqual(len(result['cells'][0]['row_indices']), 4)
        self.assertEqual(set(x.name for x in output.iterdir()), {'input_manifest.json','input_manifest.json.sha256','local_input.json','preparation_status.json'})
        self.assertNotIn('metrics', result['cells'][0])

    def test_analysis_requires_explicit_opt_in(self):
        p,h,spec,prepared,result = self.prepare_fixture()
        with self.assertRaisesRegex(runner.InputError, 'explicit'):
            runner.analyze(prepared,self.root/'output',_skip_source_lock=True)
        self.assertFalse((self.root/'output').exists())

    def test_small_fixture_full_analysis_and_archive_relocation(self):
        p,h,spec,prepared,result = self.prepare_fixture()
        moved = self.root/'moved.zip'; p.rename(moved)
        summaries = runner.analyze(prepared,self.root/'output',execute_analysis=True,archive=moved,_skip_source_lock=True)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]['status'], 'complete')
        self.assertTrue((self.root/'output/cell_summary.csv').is_file())
        self.assertTrue((self.root/'output/cells/fixture__model__C7__1/diagnostics.npz').is_file())
        self.assertEqual(runner.file_sha(moved), h)

    def test_missing_cell_reported_not_dropped_and_no_core_call(self):
        p,h,spec,prepared,result = self.prepare_fixture(missing=True)
        self.assertEqual(result['cells'][0]['status'], 'not_evaluable_input')
        fake=mock.Mock(side_effect=AssertionError('Core must not run'))
        summaries=runner.analyze(prepared,self.root/'output',execute_analysis=True,_skip_source_lock=True,_analyzer=fake)
        fake.assert_not_called()
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]['status'], 'not_evaluable_input')

    def test_missing_state_with_matching_rehashed_receipt_rejected(self):
        def mutate(s):
            s['rows'].pop();s['z']=s['z'][:-1]
            s['merged']['n_rows']-=1;s['merged']['arrays']['1']['shape'][0]-=1
        p,h,spec,prepared,result = self.prepare_fixture(mutation=mutate)
        self.assertEqual(result['cells'][0]['status'], 'not_evaluable_input')
        self.assertIn('Missing test state',result['cells'][0]['reason'])

    def test_duplicate_row_rejected(self):
        p,h,spec,prepared,result = self.prepare_fixture(mutation=lambda s:s['rows'].__setitem__(-1,s['rows'][0].copy()))
        self.assertIn('duplicate row',result['cells'][0]['reason'])

    def test_pca_shape_identity_rejected(self):
        p,h,spec,prepared,result=self.prepare_fixture(mutation=lambda s:s.__setitem__('basis',s['basis'][:-1]))
        self.assertEqual(result['cells'][0]['status'],'not_evaluable_input')
        self.assertIn('PCA',result['cells'][0]['reason'])

    def test_model_revision_mismatch_rejected(self):
        def mutate(s):s['merged']['model']['model_revision']='f'*40
        p,h,spec,prepared,result=self.prepare_fixture(mutation=mutate)
        self.assertIn('Model/tokenizer',result['cells'][0]['reason'])

    def test_object_array_rejected_before_loading(self):
        p,h,spec,prepared,result=self.prepare_fixture(mutation=lambda s:s.__setitem__('z',s['z'].astype(object)))
        self.assertEqual(result['cells'][0]['status'],'not_evaluable_input')
        self.assertIn('without objects',result['cells'][0]['reason'])

    def test_unsafe_and_duplicate_zip_members_stop_global_preparation(self):
        for option in ('unsafe','duplicate'):
            with self.subTest(option=option):
                p,h,spec=fixture(self.root,**{option:True})
                with self.assertRaises(runner.InputError):
                    runner.prepare(p,h,self.root/'prepared',_spec=spec,_skip_source_lock=True)
                self.assertFalse((self.root/'prepared').exists())

    def test_archive_hash_mismatch_stops_before_output(self):
        p,h,spec=fixture(self.root)
        with self.assertRaisesRegex(runner.InputError,'Archive SHA'):
            runner.prepare(p,'0'*64,self.root/'prepared',_spec=spec,_skip_source_lock=True)
        self.assertFalse((self.root/'prepared').exists())

    def test_existing_preparation_or_analysis_is_never_overwritten(self):
        p,h,spec,prepared,result=self.prepare_fixture()
        with self.assertRaisesRegex(runner.InputError,'already exists'):
            runner.prepare(p,h,prepared,_spec=spec,_skip_source_lock=True)
        output=self.root/'output';output.mkdir();(output/'important').write_text('keep')
        with self.assertRaisesRegex(runner.InputError,'already exists'):
            runner.analyze(prepared,output,execute_analysis=True,_skip_source_lock=True)
        self.assertEqual((output/'important').read_text(),'keep')

    def test_prepared_manifest_tampering_rejected(self):
        p,h,spec,prepared,result=self.prepare_fixture()
        (prepared/'input_manifest.json').write_text('{}')
        with self.assertRaisesRegex(runner.InputError,'checksum'):
            runner.analyze(prepared,self.root/'output',execute_analysis=True,_skip_source_lock=True)

    def test_archive_modified_after_prepare_rejected(self):
        p,h,spec,prepared,result=self.prepare_fixture()
        with p.open('ab') as f:f.write(b'extra')
        with self.assertRaisesRegex(runner.InputError,'Archive SHA'):
            runner.analyze(prepared,self.root/'output',execute_analysis=True,_skip_source_lock=True)

    def test_missing_production_source_lock_rejected(self):
        with mock.patch.object(runner,'HERE',self.root):
            with self.assertRaisesRegex(runner.InputError,'Missing frozen'):
                runner.source_lock()

    def test_source_lock_detects_omissions_and_changes(self):
        (self.root/'runner.py').write_text('fixture source')
        lock={'schema_version':1,'files':{'runner.py':runner.file_sha(self.root/'runner.py')}}
        (self.root/'source_manifest.json').write_text(json.dumps(lock))
        with mock.patch.object(runner,'HERE',self.root):
            self.assertIn('manifest_sha256',runner.source_lock())
            (self.root/'extra.txt').write_text('unlocked')
            with self.assertRaisesRegex(runner.InputError,'coverage'):
                runner.source_lock()
            (self.root/'extra.txt').unlink();(self.root/'runner.py').write_text('changed')
            with self.assertRaisesRegex(runner.InputError,'hash mismatch'):
                runner.source_lock()

    def test_json_nonfinite_duplicate_keys_and_nonobject_rejected(self):
        for data in (b'[]', b'null', b'{"x":1e999}', b'{"x":NaN}', b'{"x":1,"x":2}'):
            with self.subTest(data=data):
                with self.assertRaises(runner.InputError):
                    runner._json(data)

    def test_malformed_nested_extraction_metadata_fails_cell(self):
        for field in ('model', 'descriptor', 'arrays'):
            with self.subTest(field=field):
                p,h,spec=fixture(self.root,mutation=lambda s:s['merged'].__setitem__(field, []))
                result=runner.prepare(p,h,self.root/('prepared_'+field),_spec=spec,_skip_source_lock=True)
                self.assertEqual(result['cells'][0]['status'],'not_evaluable_input')

    def test_nonfinite_pca_fails_cell_at_analysis(self):
        def mutate(s):s['center'][0]=np.nan
        p,h,spec,prepared,result=self.prepare_fixture(mutation=mutate)
        self.assertEqual(result['cells'][0]['status'],'prepared_metadata_only')
        summaries=runner.analyze(prepared,self.root/'output',execute_analysis=True,_skip_source_lock=True)
        self.assertEqual(summaries[0]['status'],'not_evaluable_input')
        self.assertIn('Nonfinite saved PCA',summaries[0]['reason'])


if __name__ == '__main__':
    unittest.main()
