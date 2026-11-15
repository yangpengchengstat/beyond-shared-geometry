"""Model-free hostile-archive and result-contract checks.

Fixtures below are explicitly synthetic transport fixtures, not experimental
results. All temporary files stay inside this test directory and are removed.
No model, Torch, weights, actual sealed outcomes or personal directories open.
"""
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import stat
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import import_results as mod


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        real = mod.load_frozen(mod.PROJECT / mod.SOURCE_REL)
        cfg = copy.deepcopy(real.config)
        cfg.update(test_context_families=1, test_entity_orbits=1,
                   all_test_rows_per_model={'C7': 7, 'D4': 4, 'S3': 900},
                   S3_new_row_breakdown={'test_joint': 12, 'test_context': 288,
                                         'test_entity': 576, 'support': 24})
        self.cfg = cfg
        self.frozen = mod.Frozen(cfg, real.manifest, real.manifest_sha256, real.zip_sha256)
        self.payload = {}
        self.put('study_contract.json', {'study': cfg, 'bundle_source_manifest_sha256': self.frozen.manifest_sha256,
                 'parent_input_hashes': cfg['parent_pins'], 'technical_test': False,
                 'old_study_continued': False, 'formal_inference_enabled': False})
        self.put('provenance/source_manifest.json', real.manifest)
        # Source manifest verification is byte-exact; preserve frozen formatting.
        self.payload['provenance/source_manifest.json'] = (mod.PROJECT / mod.SOURCE_REL / 'source_manifest.json').read_bytes()
        self.put('budget.json', {'limit_seconds': cfg['wall_budget_seconds_total'], 'spent_seconds': 1, 'running': False})
        self.put('provenance/dependency_build_versions.json', {'numpy': 'synthetic_fixture_only'})
        self.completed = []
        self.legacy = []
        self.fields = []

    def put(self, name, value):
        self.payload[name] = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n'

    @staticmethod
    def npy(shape, dtype='<f8', *, basis=False, fortran=False):
        header = repr({'descr': dtype, 'fortran_order': fortran, 'shape': tuple(shape)})
        pad = (- (10 + len(header) + 1)) % 64
        data = bytearray(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header) + pad + 1) + header.encode() + b' ' * pad + b'\n')
        offset = len(data)
        data += b'\0' * (int(dtype[-1]) * __import__('math').prod(shape))
        if basis:
            for i in range(16):
                index = i * shape[1] + i
                data[offset + index*8:offset + (index+1)*8] = struct.pack('<d', 1.)
        return bytes(data)

    @classmethod
    def npz(cls, arrays):
        f = io.BytesIO()
        with zipfile.ZipFile(f, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for name, raw in arrays.items():
                z.writestr(name + '.npy', raw)
        return f.getvalue()

    def s3_rows(self):
        # Build the historical crossed population independently of the validator.
        # Only the prospective test dimensions are reduced for this transport
        # fixture; the immutable development bank retains both reserved crosses.
        rows = []

        def append(context, fi, entity, oi, split):
            family = f'structural_{context}_{fi:03d}'
            orbit = f"{'pilot' if entity == 'pilot' else 'main'}_triple_{entity}_{oi:03d}"
            for paraphrase in (0, 1):
                for state in range(6):
                    rows.append({'row_id': f'S3_role/{family}/p{paraphrase}/{orbit}/s{state}',
                                 'family_id': family, 'orbit_id': orbit, 'split': split,
                                 'state_id': state, 'paraphrase_id': paraphrase, 'domain': 'S3_role'})

        for context, nf in (('train', 48), ('validation', 16),
                            ('test', self.cfg['test_context_families'])):
            for entity, no in (('train', 24), ('validation', 8),
                               ('test', self.cfg['test_entity_orbits'])):
                split = context if context == entity != 'test' else 'reserved'
                if context == entity == 'test':
                    split = 'test_joint'
                elif context == 'test' and entity == 'train':
                    split = 'test_context'
                elif context == 'train' and entity == 'test':
                    split = 'test_entity'
                if 'test' in (context, entity) and split == 'reserved':
                    continue
                for fi in range(nf):
                    for oi in range(no):
                        append(context, fi, entity, oi, split)
        for fi in range(12):
            for oi in range(8):
                append('pilot', fi, 'pilot', oi, 'pilot')
        for fi in range(2):
            for oi in range(self.cfg['test_entity_orbits']):
                append('support', fi, 'test', oi, 'support')
        return rows

    def one_condition(self, spec, domain, *, all_layers=False):
        key = mod.model_key(spec)
        state_count = {'C7': 7, 'D4': 4, 'S3': 6}[domain]
        group = {'C7': 'C7', 'D4': 'D4_corner', 'S3': 'S3_role'}[domain]
        splits = ['train', 'validation', 'pilot'] + (['test'] if domain != 'S3' else ['test_joint', 'test_context', 'test_entity', 'support'])
        rows = []
        for split in splits:
            for state in range(state_count):
                rows.append({'row_id': f'{domain}/{split}/{state}', 'family_id': 'family_' + split,
                             'split': split, 'state_id': state, 'paraphrase_id': 0,
                             'orbit_id': 'orbit_test' if split.startswith('test') else 'orbit_' + split, 'domain': group})
        if domain == 'S3':
            rows = self.s3_rows()
        development_count = sum(r['split'] in ('train', 'validation', 'pilot', 'reserved') for r in rows)
        identity = {'model_id': spec['id'], 'model_revision': spec['revision'], 'tokenizer_revision': spec['tokenizer_revision']}
        base = {'model': identity, 'layers': spec['layers'], 'status': 'complete', 'readout': 'final_token' if domain == 'S3' else 'span_mean',
                'extraction_code_sha256': '1'*64}
        for role, n in [('development', development_count), ('heldout', self.cfg['all_test_rows_per_model'][domain]), ('merged', len(rows))]:
            value = {**base, 'n_rows': n, 'arrays': {str(l): {'shape': [n, spec['hidden_dimension']], 'dtype': 'float32', 'sha256': '2'*64} for l in spec['layers']}}
            if domain == 'S3':
                value['potential_features'] = {'shape': [n, 3 * spec['hidden_dimension']], 'dtype': 'float32', 'sha256': '3'*64}
            name = f'provenance/{key}__{domain}__{role}.json'
            self.put(name, value)
            if role == 'development':
                self.cfg['expected_development_cache_manifest_hashes'][spec['id'] + '|' + domain] = mod.sha_bytes(self.payload[name])
        self.completed += [f'extract/{key}/{domain}', f'merge/{key}/{domain}']
        shared = 'analysis/shared/' + key + '__S3'
        if domain == 'S3':
            keys = sorted({(r['orbit_id'], r['state_id']) for r in rows})
            positions = {k: i for i, k in enumerate(keys)}
            inverse = [positions[(r['orbit_id'], r['state_id'])] for r in rows]
            first = {}
            for i, row in enumerate(rows):
                first.setdefault((row['orbit_id'], row['state_id']), i)
            first_ids = [first[k] for k in keys]
            def integers(values, dtype, code):
                raw = bytearray(self.npy((len(values),), dtype))
                offset = mod.npy_header(raw)['offset']
                raw[offset:] = struct.pack('<' + code * len(values), *values)
                return bytes(raw)
            self.put(shared + '/rows.json', {'rows': rows, 'potential_keys': [list(k) for k in keys]})
            self.payload[shared + '/potential_unique.npz'] = self.npz({'values': self.npy((len(keys), 3 * spec['hidden_dimension']), '<f4'),
                'inverse': integers(inverse, '<i4', 'i'), 'first_row_indices': integers(first_ids, '<i8', 'q')})
            self.put(shared + '/compact_export.json', {n: mod.sha_bytes(self.payload[shared + '/' + n]) for n in ('rows.json', 'potential_unique.npz')})
        for layer in spec['layers'] if all_layers else spec['layers'][:1]:
            cell = key + '__' + domain + '__' + str(layer)
            directory = 'analysis/' + cell
            values = []
            for split in ['test'] if domain != 'S3' else ['test_joint', 'test_context', 'test_entity']:
                r = {'group': group, 'test_split': split, 'layer': layer, 'model': identity, 'coordinate_dimension': 16,
                     'cache_manifest_sha256': mod.sha_bytes(self.payload[f'provenance/{key}__{domain}__merged.json']),
                     'training_scale': 1e-12, 'mean_losses': {'primary_operator': .2, 'local_displacement': .1},
                     'family_orbit_prediction': {row['family_id'] + '|' + row['orbit_id']: .1 for row in rows if row['split'] == split},
                     'family_orbit_context': {row['family_id'] + '|' + row['orbit_id']: .1 for row in rows if row['split'] == split}}
                values.append(r)
                self.legacy.append(r)
            self.put(directory + '/legacy.json', values)
            self.payload[directory + '/coordinates.npz'] = self.npz({'z': self.npy((len(rows), 16))})
            self.payload[directory + '/pca.npz'] = self.npz({'center': self.npy((spec['hidden_dimension'],)),
                'basis': self.npy((spec['hidden_dimension'], 16), basis=True)})
            required = {'legacy.json', 'coordinates.npz', 'pca.npz'}
            if domain != 'S3':
                field = {'model_id': spec['id'], 'group': group, 'layer': layer, 'paper_eligible': False,
                         'technical_test': False, 'formal_inference': False,
                         'generator': {'fit_action_count': 1 if domain == 'C7' else 2, 'intermediate_true_state_used': False,
                         'intermediate_projection_or_snapping': False,
                         'mean_losses': {kind: {'generator_ridge': .2, 'generator_input_only_field': .1} for kind in ('primitive', 'unseen_composite')}},
                         'input_comparisons': {'storage_curves': [{'track': 'all_element', 'full_training_support': True,
                                              'results': {'all_changed': {'mean_loss': .1}}}]}}
                self.put(directory + '/fields.json', field)
                self.put(directory + '/rows.json', {'rows': rows})
                required |= {'fields.json', 'rows.json'}
                self.fields.append(field)
            else:
                self.put(directory + '/coordinate_export.json', {'rows_and_potential_archive_relative_directory': shared,
                    'shared_input_hashes': json.loads(self.payload[shared + '/compact_export.json']),
                    'full_hidden_state_or_PCA_refit_from_raw_exported': False})
                required.add('coordinate_export.json')
            self.put(directory + '/cell_complete.json', {n: mod.sha_bytes(self.payload[directory + '/' + n]) for n in required})
            self.completed.append('analyze/' + cell)

    def finish(self, *, complete=False):
        # The contract is serialized after synthetic fixture pins have been set.
        contract = json.loads(self.payload['study_contract.json'])
        contract['study'] = self.cfg
        self.put('study_contract.json', contract)
        self.put('provenance/study_config.json', self.cfg)
        self.put('execution_state.json', {'completed': self.completed})
        status = mod.COMPLETE_STATUS if complete else 'budget_stopped'
        summary = {'status': status, 'error': None if complete else {'type': 'SyntheticFixture', 'message': 'deliberately incomplete transport fixture'},
                   'study_id': mod.STUDY_ID, 'legacy_split_records': len(self.legacy), 'field_cells': len(self.fields),
                   'paper_eligible': False, 'formal_inference_enabled': False, 'old_power_gate_passed': False,
                   'native_run': False, 'synthetic_calibration_run': False}
        if complete:
            group_keys = {r['group'] + '|' + r['test_split'] for r in self.legacy}
            summary['descriptive_legacy_means'] = {k: {'primary_operator': .2, 'local_displacement': .1} for k in group_keys}
            summary['descriptive_field_means'] = {g: {'cells': 24, 'generator_unseen': {'generator_ridge': .2, 'generator_input_only_field': .1},
                                                        'all_element_full_kernel': .1} for g in ('C7', 'D4_corner')}
            self.put('analysis/descriptive_uncertainty.json', {'seed': self.cfg['descriptive_uncertainty']['seed'],
                'replicates': self.cfg['descriptive_uncertainty']['replicates'], 'multiple_comparison_adjustment': False,
                'legacy_contrasts': {k: {} for k in group_keys}})
        self.put('summary.json', summary)
        self.put('collection_status.json', {'summary_status': status, 'artifact_collection_is_not_a_scientific_pass': True,
                                           'models_loaded_by_collection': False})
        return self.write()

    def write(self):
        path = self.root / 'synthetic.results.zip'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for name, raw in self.payload.items():
                z.writestr(mod.RESULT_PREFIX + name, raw)
        sidecar = path.with_suffix('.zip.sha256')
        sidecar.write_text(mod.sha_file(path) + '  ' + path.name + '\n')
        return path, sidecar


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.transport-fixture-', dir=Path(__file__).parent)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_pinned_expanded_source(self):
        f = mod.load_frozen(mod.PROJECT / mod.SOURCE_REL)
        self.assertEqual(f.manifest_sha256, mod.MANIFEST_SHA256)
        self.assertEqual(len(f.config['models']), 6)

    def test_no_outside_paths_or_symlinks(self):
        with self.assertRaises(mod.ValidationError):
            mod.boundary('/not-an-authorized-project/file', self.root)
        (self.root / 'link').symlink_to(self.root / 'other')
        with self.assertRaises(mod.ValidationError):
            mod.boundary(self.root / 'link' / 'file', self.root)

    def test_zip_security(self):
        for name in ('../escape', '/absolute', 'C:/drive', 'a/../b', 'a\\b', 'a//b'):
            with self.subTest(name=name):
                f = io.BytesIO()
                with zipfile.ZipFile(f, 'w') as z:
                    z.writestr(name, b'bad')
                with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z, self.assertRaises(mod.ValidationError):
                    mod.inspect_zip(z)
        for duplicate, symlink in ((True, False), (False, True)):
            f = io.BytesIO()
            with zipfile.ZipFile(f, 'w') as z:
                item = zipfile.ZipInfo('item')
                if symlink:
                    item.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(item, b'bad')
                if duplicate:
                    with __import__('warnings').catch_warnings():
                        __import__('warnings').simplefilter('ignore')
                        z.writestr('item', b'bad')
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z, self.assertRaises(mod.ValidationError):
                mod.inspect_zip(z)

    def test_zip_bomb_budget(self):
        f = io.BytesIO()
        with zipfile.ZipFile(f, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('zero', b'0' * 100000)
        with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z, self.assertRaises(mod.ValidationError):
            mod.inspect_zip(z, max_total=1000)

    def test_json_duplicate_and_nonfinite_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.assertRaises(mod.ValidationError):
                mod.parse_json(raw)

    def test_npy_object_truncation_and_fortran(self):
        good = Fixture.npy((3, 16), fortran=True)
        self.assertTrue(mod.npy_header(good)['fortran_order'])
        with self.assertRaises(mod.ValidationError):
            mod.npy_header(good[:-1])
        bad = good.replace(b"'<f8'", b"'|O8'")
        with self.assertRaises(mod.ValidationError):
            mod.npy_header(bad)

    def test_sidecar_corruption_and_filename(self):
        p = self.root / 'a.zip'
        p.write_bytes(b'fixture')
        sidecar = self.root / 'a.zip.sha256'
        for text in ('0'*64 + '  a.zip\n', mod.sha_file(p) + '  wrong.zip\n'):
            sidecar.write_text(text)
            with self.assertRaises(mod.ValidationError):
                mod.check_sidecar(p, sidecar)

    def test_incomplete_preserved_and_no_overwrite(self):
        f = Fixture(self.root)
        f.one_condition(f.cfg['models'][0], 'C7')
        archive, checksum = f.finish()
        report = mod.validate_result(archive, f.frozen)
        self.assertEqual(report['panel_validation'], 'incomplete_or_failed')
        self.assertEqual(report['counts']['completion_markers'], 1)
        self.assertFalse(report['paper_eligible'])
        with mock.patch.object(mod, 'load_frozen', return_value=f.frozen):
            with self.assertRaises(mod.ValidationError):
                mod.import_validated(archive, checksum, archive, project=self.root, receipt_name='test')
            report = mod.import_validated(archive, checksum, archive, project=self.root, receipt_name='test', allow_incomplete=True)
            receipt = self.root / mod.STUDY_REL / 'imports' / 'test'
            self.assertEqual(mod.sha_file(receipt / 'original.results.zip'), mod.sha_file(archive))
            self.assertTrue((receipt / 'raw' / 'analysis').is_dir())
            self.assertFalse(report['automatic_manuscript_promotion'])
            with self.assertRaises(mod.ValidationError):
                mod.import_validated(archive, checksum, archive, project=self.root, receipt_name='test', allow_incomplete=True)

    def test_per_cell_checksum_catches_tamper(self):
        f = Fixture(self.root)
        f.one_condition(f.cfg['models'][0], 'C7')
        archive, _ = f.finish()
        key = next(n for n in f.payload if n.endswith('/coordinates.npz'))
        f.payload[key] += b'tampered'
        archive, _ = f.write()
        with self.assertRaisesRegex(mod.ValidationError, 'checksum mismatch'):
            mod.validate_result(archive, f.frozen)

    def test_compact_s3_mapping_independent_of_numpy(self):
        f = Fixture(self.root)
        f.one_condition(f.cfg['models'][0], 'S3')
        archive, _ = f.finish()
        report = mod.validate_result(archive, f.frozen)
        self.assertEqual(report['counts']['legacy_split_records'], 3)
        name = next(n for n in f.payload if n.endswith('/potential_unique.npz'))
        with zipfile.ZipFile(io.BytesIO(f.payload[name])) as z:
            raw = {n[:-4]: z.read(n) for n in z.namelist()}
        inverse = bytearray(raw['inverse'])
        offset = mod.npy_header(inverse)['offset']
        inverse[offset:offset+4] = struct.pack('<i', 999)
        raw['inverse'] = bytes(inverse)
        f.payload[name] = f.npz(raw)
        shared = name.rsplit('/', 1)[0]
        f.put(shared + '/compact_export.json', {n: mod.sha_bytes(f.payload[shared + '/' + n]) for n in ('rows.json', 'potential_unique.npz')})
        archive, _ = f.write()
        with self.assertRaisesRegex(mod.ValidationError, 'mapping mismatch'):
            mod.validate_result(archive, f.frozen)

    def test_reject_split_leakage_and_missing_contrast_units(self):
        for domain in ('C7', 'S3'):
            with self.subTest(domain=domain):
                f = Fixture(self.root)
                f.one_condition(f.cfg['models'][0], domain)
                f.finish()
                name = next(n for n in f.payload if n.endswith('/rows.json'))
                obj = json.loads(f.payload[name])
                train = next(r for r in obj['rows'] if r['split'] == 'train')
                validation = next(r for r in obj['rows'] if r['split'] == 'validation')
                validation_unit = (validation['family_id'], validation['orbit_id'])
                # Move one entire validation orbit across the split boundary.
                # Moving every unit would instead trigger the duplicate-row guard.
                for row in obj['rows']:
                    if (row['family_id'], row['orbit_id']) == validation_unit and row['split'] == 'validation':
                        row['family_id'] = train['family_id']
                        row['orbit_id'] = train['orbit_id']
                f.put(name, obj)
                directory = name.rsplit('/', 1)[0]
                manifest_name = directory + ('/compact_export.json' if domain == 'S3' else '/cell_complete.json')
                manifest = json.loads(f.payload[manifest_name])
                manifest['rows.json'] = mod.sha_bytes(f.payload[name])
                f.put(manifest_name, manifest)
                archive, _ = f.write()
                with self.assertRaisesRegex(mod.ValidationError, 'leaks across splits'):
                    mod.validate_result(archive, f.frozen)
        f = Fixture(self.root)
        f.one_condition(f.cfg['models'][0], 'C7')
        f.finish()
        name = next(n for n in f.payload if n.endswith('/legacy.json'))
        values = json.loads(f.payload[name])
        values[0]['family_orbit_prediction'] = {'invented|invented': .1}
        f.put(name, values)
        marker_name = name.rsplit('/', 1)[0] + '/cell_complete.json'
        marker = json.loads(f.payload[marker_name])
        marker['legacy.json'] = mod.sha_bytes(f.payload[name])
        f.put(marker_name, marker)
        archive, _ = f.write()
        with self.assertRaisesRegex(mod.ValidationError, 'contrast coverage'):
            mod.validate_result(archive, f.frozen)

    def test_partial_export_remains_incomplete(self):
        f = Fixture(self.root)
        f.one_condition(f.cfg['models'][0], 'C7')
        f.finish()
        directory = next(n.rsplit('/', 1)[0] for n in f.payload if n.endswith('/legacy.json'))
        f.payload[directory + '/interrupted.json.partial'] = b'{incomplete'
        archive, _ = f.write()
        report = mod.validate_result(archive, f.frozen)
        self.assertTrue(any('uncommitted_partial_artifact' in v for v in report['missing_or_incomplete']))
        self.assertEqual(report['panel_validation'], 'incomplete_or_failed')

    @unittest.skipUnless(importlib.util.find_spec('numpy') is not None, 'Optional NumPy not installed')
    def test_optional_numpy_geometry_and_nonfinite_rejection(self):
        f = Fixture(self.root)
        f.one_condition(f.cfg['models'][0], 'C7')
        archive, _ = f.finish()
        report = mod.validate_result(archive, f.frozen, numerical=True)
        self.assertTrue(report['numerical_checks']['enabled'])
        name = next(n for n in f.payload if n.endswith('/coordinates.npz'))
        with zipfile.ZipFile(io.BytesIO(f.payload[name])) as z:
            array = bytearray(z.read('z.npy'))
        offset = mod.npy_header(array)['offset']
        array[offset:offset+8] = struct.pack('<d', float('nan'))
        f.payload[name] = f.npz({'z': bytes(array)})
        marker_name = name.rsplit('/', 1)[0] + '/cell_complete.json'
        marker = json.loads(f.payload[marker_name])
        marker['coordinates.npz'] = mod.sha_bytes(f.payload[name])
        f.put(marker_name, marker)
        archive, _ = f.write()
        with self.assertRaisesRegex(mod.ValidationError, 'Nonfinite NPZ'):
            mod.validate_result(archive, f.frozen, numerical=True)

    def test_exact_full_panel(self):
        f = Fixture(self.root)
        for spec in f.cfg['models']:
            for domain in ('C7', 'D4', 'S3'):
                f.one_condition(spec, domain, all_layers=True)
        archive, _ = f.finish(complete=True)
        report = mod.validate_result(archive, f.frozen)
        self.assertEqual(report['counts'], {'completion_markers': 72, 'legacy_split_records': 120, 'field_cells': 48})
        self.assertEqual(report['panel_validation'], 'complete')
        self.assertFalse(report['paper_eligible'])
        # Equal aggregate counts cannot conceal a duplicated/wrong condition.
        name = next(n for n in f.payload if n.endswith('/legacy.json'))
        values = json.loads(f.payload[name])
        values[0]['layer'] = 999
        f.put(name, values)
        directory = name.rsplit('/', 1)[0]
        marker = json.loads(f.payload[directory + '/cell_complete.json'])
        marker['legacy.json'] = mod.sha_bytes(f.payload[name])
        f.put(directory + '/cell_complete.json', marker)
        archive, _ = f.write()
        with self.assertRaisesRegex(mod.ValidationError, 'condition/cache identity'):
            mod.validate_result(archive, f.frozen)


if __name__ == '__main__':
    unittest.main()
