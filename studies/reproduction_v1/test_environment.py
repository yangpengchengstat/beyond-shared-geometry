"""Offline tests: never download weights, import torch or initialize CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).with_name('environment.py')
spec = importlib.util.spec_from_file_location('reproduction_environment', MODULE)
env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env)


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.spec = {'id': 'owner/model', 'revision': 'a' * 40, 'tokenizer_revision': 'a' * 40}

    def tearDown(self):
        self.temp.cleanup()

    def config(self, value):
        p = self.root / 'config.json'; p.write_text(json.dumps(value)); return p

    def snapshot(self, shard=False):
        p = self.root / 'models--owner--model' / 'snapshots' / ('a' * 40)
        p.mkdir(parents=True)
        (p / 'config.json').write_text(json.dumps({'hidden_size': 16, 'num_hidden_layers': 2}))
        (p / 'tokenizer.json').write_text('{}')
        if shard:
            (p / 'model-00001-of-00001.safetensors').write_bytes(b'fixture')
            (p / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'model-00001-of-00001.safetensors'}}))
        else:
            (p / 'model.safetensors').write_bytes(b'fixture')
        return p

    def report(self):
        names = ['torch', 'numpy', 'scipy', 'transformers', 'huggingface-hub', 'tokenizers']
        return {'install': [{'metadata': {'name': name, 'version': '1.0'},
                            'download_info': {'url': f'https://example.org/{name}-1.0-py3-none-any.whl',
                             'archive_info': {'hashes': {'sha256': 'a' * 64}}}} for name in names]}

    def write_report(self, value):
        p = self.root / 'report.json'; p.write_text(json.dumps(value)); return p

    def test_gather_nested_deduplicates_and_preserves_dtype(self):
        c = {'models': [dict(self.spec, dtype='float16')], 'else': [dict(self.spec, dtype='bfloat16')]}
        path = self.config(c)
        self.assertEqual(env.gather_models([path]), [self.spec])
        self.assertEqual(json.loads(path.read_text()), c)

    def test_reject_unpinned_or_mismatched_tokenizer(self):
        for item in (dict(self.spec, revision='main'), dict(self.spec, tokenizer_revision='b' * 40)):
            with self.subTest(item=item), self.assertRaises(ValueError):
                env.gather_models([self.config(item)])

    def test_root_only_weight_selection_excludes_credentials_duplicates_code(self):
        result = env.selected_snapshot_names(['config.json', 'tokenizer.json', 'model.safetensors',
                   'original/model.safetensors', 'pytorch_model.bin', 'token', 'modeling.py', '../secret'])
        self.assertEqual(result, ['config.json', 'model.safetensors', 'tokenizer.json'])

    def test_snapshot_inventory_hashes_model_and_tokenizer(self):
        p = self.snapshot()
        a = env.snapshot_inventory(p, self.spec)
        (p / 'tokenizer.json').write_text('{"changed":true}')
        b = env.snapshot_inventory(p, self.spec)
        self.assertNotEqual(a['files']['tokenizer.json']['sha256'], b['files']['tokenizer.json']['sha256'])
        self.assertEqual(a['hidden_size'], 16)
        self.assertNotIn(str(self.root), json.dumps(a))

    def test_snapshot_allows_internal_blob_links(self):
        p = self.snapshot(); weights = p / 'model.safetensors'; weights.unlink()
        blob = p.parent.parent / 'blobs' / '123'; blob.parent.mkdir(); blob.write_bytes(b'fixture')
        weights.symlink_to(blob)
        self.assertEqual(env.snapshot_inventory(p, self.spec)['files'][weights.name]['bytes'], 7)

    def test_snapshot_rejects_external_links(self):
        p = self.snapshot(); weights = p / 'model.safetensors'; weights.unlink()
        external = self.root / 'private'; external.write_bytes(b'private'); weights.symlink_to(external)
        with self.assertRaises(ValueError): env.snapshot_inventory(p, self.spec)

    def test_snapshot_rejects_revision_drift(self):
        p = self.snapshot()
        with self.assertRaises(ValueError): env.snapshot_inventory(p, dict(self.spec, revision='b' * 40))

    def test_snapshot_rejects_missing_extra_index_shards(self):
        p = self.snapshot(shard=True)
        env.snapshot_inventory(p, self.spec)
        (p / 'model-00002-of-00002.safetensors').write_bytes(b'extra')
        with self.assertRaises(ValueError): env.snapshot_inventory(p, self.spec)

    def test_expected_file_inventory_cannot_silently_shrink(self):
        p = self.snapshot()
        with self.assertRaises(ValueError): env.snapshot_inventory(p, self.spec, ['config.json', 'model.safetensors'])

    def test_full_dependency_report_becomes_hashed_direct_url_lock(self):
        p = self.write_report(self.report()); lock = self.root / 'requirements.lock'
        facts = env.pip_report_to_lock(p, lock)
        self.assertEqual(facts['packages'], 6)
        self.assertEqual(lock.read_text().count('--hash=sha256:'), 6)
        self.assertEqual(env.pip_report_to_lock(p, lock), facts)

    def test_partial_report_unhashed_credentials_and_sdist_rejected(self):
        changes = ['missing', 'hash', 'credentials', 'sdist', 'duplicate']
        for change in changes:
            report = self.report()
            if change == 'missing': report['install'].pop()
            if change == 'hash': report['install'][0]['download_info']['archive_info']['hashes'] = {}
            if change == 'credentials': report['install'][0]['download_info']['url'] = 'https://token@example.org/a.whl'
            if change == 'sdist': report['install'][0]['download_info']['url'] = 'https://example.org/a.tar.gz'
            if change == 'duplicate': report['install'].append(report['install'][0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                env.pip_report_to_lock(self.write_report(report), self.root / 'bad.lock')

    def test_existing_different_lock_not_overwritten(self):
        lock = self.root / 'lock'; lock.write_text('preserve')
        with self.assertRaises(ValueError): env.pip_report_to_lock(self.write_report(self.report()), lock)
        self.assertEqual(lock.read_text(), 'preserve')

    def test_source_only_inspection_does_not_import_torch_or_expose_env(self):
        config = self.config(self.spec); output = self.root / 'out'
        result = subprocess.run([sys.executable, str(MODULE), '--config', str(config), '--output', str(output)],
            env={**os.environ, 'HF_TOKEN': 'secret-test-value'}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        facts = json.loads((output / 'environment_inspection.json').read_text())
        self.assertFalse(facts['cuda_checked'])
        self.assertNotIn('secret-test-value', result.stdout + result.stderr + json.dumps(facts))
        self.assertNotIn('torch', sys.modules)

    def test_disk_guard_prevents_work_without_mutating_data(self):
        output = self.root / 'out'
        with self.assertRaises(ValueError): env.environment_snapshot(output, minimum_free_gib=1e12)
        self.assertEqual(json.loads((output / 'environment_inspection.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
