"""Input identity, excluded-data benchmark, checkpoint and collection tests."""
from contextlib import redirect_stdout, ExitStack
from concurrent.futures import Future
from types import SimpleNamespace
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as runner


def rows_and_coordinates(domain):
    n = {"C7": 7, "D4": 4, "S3": 6}[domain]
    canonical_domain = {"C7": "C7", "D4": "D4_corner", "S3": "S3_role"}[domain]
    rows = []
    for split in ("train", "validation", "pilot", "test", "reserved", "support"):
        for paraphrase in (0, 1):
            for state in range(n):
                rows.append(dict(row_id=f"{split}/{paraphrase}/{state}",
                                 family_id=split, orbit_id="one", state_id=state,
                                 paraphrase_id=paraphrase, domain=canonical_domain,
                                 split=split))
    z = np.random.default_rng(20261021).normal(size=(len(rows), 16)).astype(np.float64)
    z[[r['split'] in ('test', 'reserved', 'support') for r in rows]] += 1e4
    return rows, z


def add_cell(zf, model, domain, layer=6):
    rows, z = rows_and_coordinates(domain)
    cell = f"{model}__{domain}__{layer}"
    base = runner.PREFIX + "analysis/" + cell + "/"
    buf = io.BytesIO()
    np.savez_compressed(buf, z=z)
    zf.writestr(base + "coordinates.npz", buf.getvalue())
    if domain == 'S3':
        directory = f"analysis/shared/{model}__S3"
        zf.writestr(base + "coordinate_export.json", json.dumps({
            "rows_and_potential_archive_relative_directory": directory}))
        zf.writestr(runner.PREFIX + directory + "/rows.json", json.dumps({"rows": rows}))
    else:
        zf.writestr(base + "rows.json", json.dumps({"rows": rows}))
    return cell


def checkpoint_fixture(output):
    job = {"cell": "fixture__C7__6", "registration": "semantic", "component": "compact"}
    identifier = runner.job_id(job)
    identity = {"job_ids": [identifier], "study_id": "fixture"}
    identity_sha = hashlib.sha256(runner.canonical(identity)).hexdigest()
    data = {"job": job, "identity_sha256": identity_sha, "elapsed_seconds": 0.1,
            "result": {"finite_score": 0.125}, "paper_eligible": False}
    data['payload_sha256'] = hashlib.sha256(runner.canonical(data)).hexdigest()
    (output / 'checkpoints').mkdir()
    path = output / 'checkpoints' / (identifier + '.json')
    runner.atomic_json(path, data)
    runner.atomic_json(output / 'identity.json', identity)
    runner.atomic_json(output / 'state.json', {
        "status": "complete", "completed_jobs": 1, "planned_jobs": 1,
        "checkpoints": {identifier: {"sha256": runner.digest(path), "elapsed_seconds": .1}}})
    return job, identity_sha, path


class RunnerTests(unittest.TestCase):
    def test_unsafe_zip_paths_links_and_duplicate_members(self):
        cases = ('/absolute', '../outside', runner.PREFIX + '../outside', runner.PREFIX + 'a\\b', 'wrongroot/file')
        for member in cases:
            with self.subTest(member=member):
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w') as zf:
                    zf.writestr(member, 'x')
                with zipfile.ZipFile(buffer) as zf, self.assertRaises(ValueError):
                    runner.safe_members(zf)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            info = zipfile.ZipInfo(runner.PREFIX + 'link')
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, 'target')
        with zipfile.ZipFile(buffer) as zf, self.assertRaisesRegex(ValueError, 'Symlink'):
            runner.safe_members(zf)
        buffer = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(buffer, 'w') as zf:
                zf.writestr(runner.PREFIX + 'same', 'x')
                zf.writestr(runner.PREFIX + 'same', 'y')
        with zipfile.ZipFile(buffer) as zf, self.assertRaisesRegex(ValueError, 'Duplicate'):
            runner.safe_members(zf)

    def test_input_digest_and_exact_panel_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.zip'
            with zipfile.ZipFile(path, 'w') as zf:
                for domain in ('C7', 'D4', 'S3'):
                    add_cell(zf, 'fixture', domain)
            config = {'parent_archive_sha256': runner.digest(path)}
            with mock.patch.object(runner, 'EXPECTED_MODELS', {'fixture': (6,)}):
                report = runner.preflight(path, config)
                self.assertEqual(report['coordinate_cells'], 3)
                self.assertFalse(report['old_calibration_required'])
                self.assertFalse(report['model_weights_required'])
                self.assertFalse(report['raw_forward_replay_supported'])
            with self.assertRaisesRegex(ValueError, '72-cell'):
                runner.preflight(path, config)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                runner.preflight(path, {'parent_archive_sha256': '0' * 64})

    def test_pilot_fit_physically_cannot_see_test_support_reserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.zip'
            with zipfile.ZipFile(path, 'w') as zf:
                name = add_cell(zf, 'fixture', 'C7')
            job = {'cell': name, 'registration': 'semantic', 'component': 'compact'}
            def excluded_fit(cell, config, splits):
                self.assertEqual(splits, ['pilot'])
                self.assertEqual(set(row['split'] for row in cell.rows), {'train', 'validation', 'pilot'})
                self.assertLess(float(np.abs(cell.z).max()), 10)
                return {'observed_n': len(cell.rows)}
            with mock.patch('compact.evaluate_compact', side_effect=excluded_fit):
                result = runner.execute_job(path, job, {}, 'pilot')
            self.assertEqual(result['scientific_role'], 'excluded_development_benchmark')
            self.assertFalse(result['paper_eligible'])

    def test_jobs_cover_fixed_panel_and_pilot_is_fixed(self):
        cells = []
        for model, layers in runner.EXPECTED_MODELS.items():
            for literal, domain in (('C7', 'C7'), ('D4', 'D4_corner'), ('S3', 'S3_role')):
                for layer in layers:
                    cells.append({'name': f'{model}__{literal}__{layer}', 'domain': domain})
        config = {'registrations': ['semantic', 'consistent_swap01'],
                  'components': {'compact': ['C7', 'D4_corner'],
                                 'context': ['C7', 'D4_corner', 'S3_role']}}
        full = runner.jobs_for(cells, config, 'full')
        self.assertEqual(len(full), 240)
        self.assertEqual(len({runner.job_id(j) for j in full}), 240)
        pilot = runner.jobs_for(cells[::-1], config, 'pilot')
        self.assertEqual(len(pilot), 5)
        self.assertEqual({j['registration'] for j in pilot}, {'semantic'})
        self.assertTrue(all(j['cell'].startswith('Qwen__Qwen2.5-1.5B__') and j['cell'].endswith('__6') for j in pilot))

    def test_source_manifest_rejects_modified_or_escape_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'only.py'
            source.write_text('value = 1\n')
            manifest = root / 'source_manifest.json'
            runner.atomic_json(manifest, {'files': {'only.py': runner.digest(source)}})
            with mock.patch.object(runner, 'ROOT', root):
                self.assertEqual(runner.source_identity(), runner.digest(manifest))
                source.write_text('value = 2\n')
                with self.assertRaisesRegex(ValueError, 'Source changed'):
                    runner.source_identity()
                runner.atomic_json(manifest, {'files': {'../outside.py': 'ignored'}})
                with self.assertRaisesRegex(ValueError, 'Unsafe source'):
                    runner.source_identity()

    def test_checkpoint_corruption_and_identity_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            job, identity_sha, path = checkpoint_fixture(output)
            verified = runner.verify_checkpoint(path, job, identity_sha)
            self.assertEqual(verified['result']['finite_score'], .125)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                runner.verify_checkpoint(path, job, 'bad_identity')
            changed = json.loads(path.read_text())
            changed['result']['finite_score'] = .25
            runner.atomic_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'payload is corrupt'):
                runner.verify_checkpoint(path, job, identity_sha)

    def test_collection_verifies_all_planned_checkpoints_and_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'output'
            output.mkdir()
            job, identity_sha, path = checkpoint_fixture(output)
            with redirect_stdout(io.StringIO()):
                archive = runner.collect(output)
            self.assertTrue(archive.exists())
            self.assertTrue(archive.with_name(archive.name + '.sha256').read_text().startswith(runner.digest(archive)))
            with zipfile.ZipFile(archive) as zf:
                self.assertIsNone(zf.testzip())
                self.assertIn('output/checkpoints/' + path.name, zf.namelist())
            changed = json.loads(path.read_text())
            changed['result']['finite_score'] = 99
            runner.atomic_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'file changed'):
                runner.collect(output)

    def test_collection_rejects_partial_and_missing_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            checkpoint_fixture(output)
            state = json.loads((output / 'state.json').read_text())
            state['status'] = 'paused_budget'
            runner.atomic_json(output / 'state.json', state)
            with self.assertRaisesRegex(ValueError, 'requires complete'):
                runner.collect(output)
            state['status'] = 'complete'
            state['checkpoints'] = {}
            runner.atomic_json(output / 'state.json', state)
            with self.assertRaisesRegex(ValueError, 'all planned jobs'):
                runner.collect(output)

    def test_output_lock_excludes_second_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with runner.output_lock(output):
                with self.assertRaisesRegex(ValueError, 'Another process'):
                    with runner.output_lock(output):
                        pass
            with runner.output_lock(output):
                pass

    def test_nonfinite_result_cannot_be_serialized(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(bad=str(bad)), self.assertRaises(ValueError):
                runner.canonical({'score': bad})


class ImmediatePool:
    """Synchronous executor fixture preserving Future result/error semantics."""
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def submit(self, fn, *args):
        future = Future()
        try: future.set_result(fn(*args))
        except BaseException as exc: future.set_exception(exc)
        return future


class RunnerLifecycleTests(unittest.TestCase):
    def _setup(self, directory):
        root = directory / 'source'; root.mkdir()
        config = {'study_id': 'fixture', 'validation_comparability_factor': 1.25}
        runner.atomic_json(root / 'protocol.json', config)
        runner.atomic_json(root / 'source_manifest.json', {'files': {}})
        archive = directory / 'inputs.zip'; archive.write_bytes(b'synthetic runtime fixture only')
        info = {'archive_sha256': runner.digest(archive), 'cells': []}
        args = SimpleNamespace(output=str(directory / 'output'), archive=str(archive),
                               mode='full', workers=1, max_seconds=0)
        jobs = [{'cell': f'fixture{i}__C7__6', 'registration': 'semantic',
                 'component': 'context', 'domain': 'C7'} for i in range(2)]
        return root, config, info, args, jobs

    def test_rejects_preserved_source_and_old_output_before_creating_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            forbidden = [runner.ROOT / 'never_create_test_output',
                         base / 'icml2027-characterization-v1' / 'nested',
                         base / 'frozen_run' / 'nested', base / 'incoming' / 'nested']
            preserved = base / 'icml2027-bf16-f64-v2'; preserved.mkdir()
            alias = base / 'alias'; alias.symlink_to(preserved, target_is_directory=True)
            forbidden.append(alias / 'nested')
            for output in forbidden:
                with self.subTest(output=str(output)), self.assertRaises(ValueError):
                    runner.run(SimpleNamespace(output=str(output)), {}, 'source', {})
                self.assertFalse(output.exists())

    def test_budget_pause_preserves_checkpoint_and_resume_does_not_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, config, info, args, jobs = self._setup(Path(tmp))
            clock = [0.]
            called = []
            def execute(archive, job, config, mode):
                called.append(runner.job_id(job)); clock[0] += 2.
                return {'job': job, 'result': {'methods': []}, 'elapsed_seconds': 2.}
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(runner, 'ROOT', root))
                stack.enter_context(mock.patch.object(runner, 'jobs_for', return_value=jobs))
                stack.enter_context(mock.patch.object(runner, 'execute_job', side_effect=execute))
                stack.enter_context(mock.patch.object(runner, 'ProcessPoolExecutor', ImmediatePool))
                stack.enter_context(mock.patch.object(runner, 'source_identity', return_value='source'))
                stack.enter_context(mock.patch.object(runner.time, 'monotonic', side_effect=lambda: clock[0]))
                stack.enter_context(redirect_stdout(io.StringIO()))
                args.max_seconds = 1.
                self.assertEqual(runner.run(args, config, 'source', info), 75)
                state = json.loads((Path(args.output) / 'state.json').read_text())
                self.assertEqual(state['status'], 'paused_budget')
                self.assertEqual(state['completed_jobs'], 1)
                args.max_seconds = 0
                self.assertEqual(runner.run(args, config, 'source', info), 0)
                self.assertEqual(called, [runner.job_id(j) for j in jobs])
                self.assertEqual(runner.run(args, config, 'source', info), 0)
                self.assertEqual(called, [runner.job_id(j) for j in jobs])
            state = json.loads((Path(args.output) / 'state.json').read_text())
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['completed_jobs'], 2)

    def test_worker_failure_is_recorded_without_success_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, config, info, args, jobs = self._setup(Path(tmp))
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(runner, 'ROOT', root))
                stack.enter_context(mock.patch.object(runner, 'jobs_for', return_value=jobs[:1]))
                stack.enter_context(mock.patch.object(runner, 'execute_job', side_effect=RuntimeError('fixture worker error')))
                stack.enter_context(mock.patch.object(runner, 'ProcessPoolExecutor', ImmediatePool))
                stack.enter_context(redirect_stdout(io.StringIO()))
                with self.assertRaisesRegex(RuntimeError, 'fixture worker error'):
                    runner.run(args, config, 'source', info)
            state = json.loads((Path(args.output) / 'state.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertIn('fixture worker error', state['error'])
            self.assertEqual(state['completed_jobs'], 0)
            self.assertEqual(list((Path(args.output) / 'checkpoints').glob('*.json')), [])
            self.assertEqual(runner.digest(args.archive), info['archive_sha256'])


if __name__ == '__main__':
    unittest.main()
