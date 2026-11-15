"""Versioned engineering guards on synthetic fixtures; no model is loaded."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest

from bsg_next.io import atomic_json, read_json, file_sha256
from bsg_next.native import collect_features
from bsg_next.pipeline import StudyPipeline, extract_study, seal_study, validate_confirmation_freeze


class TinyFeatureRuntime:
    def __init__(self, model=None, config=None):
        self.model_id = (model or {}).get('id', 'fixture')
        self.revision = (model or {}).get('revision', 'a' * 40)
    def identity(self):
        return {'model_id': self.model_id, 'model_revision': self.revision, 'tokenizer_revision': self.revision}
    def features(self, records, layers, readout):
        values = np.array([[r.get('state_id', i), 1.] for i, r in enumerate(records)], dtype=np.float32)
        yield list(range(len(records))), {layer: values.copy() for layer in layers}
    def close(self): pass


@pytest.mark.parametrize('indices', [[0, 0, 1], [-1, 0], [0.9, 1.9], [False, True], [0, 2], [[0, 1]], []])
def test_native_rejects_malformed_indices_before_return(indices):
    class Invalid(TinyFeatureRuntime):
        def features(self, records, layers, readout):
            yield indices, {1: np.ones((len(indices), 2), np.float32)}
    with pytest.raises(ValueError, match='indices must'):
        collect_features(Invalid(), [{'id': 0}, {'id': 1}], [1])


def test_native_rejects_cross_batch_duplicates_and_shape_broadcasting():
    class Duplicate(TinyFeatureRuntime):
        def features(self, records, layers, readout):
            yield [0], {1: np.ones((1, 2))}
            yield [0, 1], {1: np.ones((2, 2))}
    with pytest.raises(ValueError, match='duplicate'):
        collect_features(Duplicate(), [{'id': 0}, {'id': 1}], [1])
    class WrongShape(TinyFeatureRuntime):
        def features(self, records, layers, readout):
            yield [0, 1], {1: np.ones((1, 2))}
    with pytest.raises(ValueError, match='batch shape'):
        collect_features(WrongShape(), [{'id': 0}, {'id': 1}], [1])
    actual = collect_features(TinyFeatureRuntime(), [{'id': 0}, {'id': 1}], [1])
    assert np.array_equal(actual[1], np.array([[0., 1.], [1., 1.]], np.float32))


@pytest.mark.parametrize('mutation', ['delete', 'modify', 'result', 'legacy'])
def test_resume_rejects_missing_changed_or_unversioned_outputs(tmp_path, mutation):
    pipeline = StudyPipeline({'study_id': 'synthetic'}, tmp_path / 'run')
    def writer(out):
        atomic_json(out / 'evidence.json', {'value': 1})
        return {'value': 1}
    pipeline.stage('fixture', writer)
    state_path = pipeline.state_dir / 'fixture.json'
    evidence = pipeline.output / 'outputs/fixture/evidence.json'
    if mutation == 'delete': evidence.unlink()
    elif mutation == 'modify': atomic_json(evidence, {'value': 2})
    else:
        state = read_json(state_path)
        if mutation == 'result': state['result']['value'] = 2
        else: state.pop('output_manifest')
        atomic_json(state_path, state)
    before = state_path.read_bytes()
    with pytest.raises(ValueError, match='missing or changed|lacks a versioned|differs from'):
        pipeline.stage('fixture', lambda out: pytest.fail('must not recompute'))
    assert state_path.read_bytes() == before


def test_known_stage_requires_its_output_and_native_preparation_survives_new_test_files(tmp_path):
    pipeline = StudyPipeline({'study_id': 'synthetic'}, tmp_path / 'run')
    with pytest.raises(ValueError, match='required output'):
        pipeline.stage('extract_development', lambda out: {'cache_dirs': []})
    root = pipeline.output / 'outputs/native_study'
    def prepare(out):
        atomic_json(root / 'selection_frozen.json', {'fixture': True})
        return {'selection': str(root / 'selection_frozen.json')}
    pipeline.stage('native_prepare', prepare)
    atomic_json(root / 'native_results.json', {'fixture_test': True})
    assert pipeline.stage('native_prepare', lambda out: pytest.fail('must reuse'))['selection']


def _confirmation_fixture(tmp_path):
    cfg = {'study_id': 'engineering_fixture', 'seed': 42, 'technical_test': True,
           'models': [{'id': 'fixture', 'revision': 'a' * 40, 'layers': [1]}],
           'data': {'groups': ['C7'], 'max_test_families': 64, 'max_test_orbits': 24},
           'structural': {}}
    pipeline = StudyPipeline(cfg, tmp_path / 'run')
    pipeline.stage('datasets', lambda out: seal_study(cfg, out))
    data = pipeline.output / 'outputs/datasets'
    dev = pipeline.output / 'outputs/extract_development'
    pipeline.stage('extract_development', lambda out: extract_study(cfg, out, data, runtime_factory=TinyFeatureRuntime))
    def artifact(out, filename, result): atomic_json(out / filename, result); return result
    pipeline.stage('structural_pilot', lambda out: artifact(out, 'summary.json', {'fixture_only': True}))
    pipeline.stage('pilot_context_eligibility', lambda out: artifact(out, 'eligibility.json', {'status': 'eligible_for_registered_power'}))
    pipeline.stage('structural_power', lambda out: artifact(out, 'power.json', {'selected_families': 32, 'fixture_only': True}))
    freeze = pipeline.output / 'confirmation_freeze.json'
    atomic_json(freeze, {'configuration': pipeline.descriptor['identity'], 'test_families': 32, 'test_orbits': 16,
                        'power_sha256': file_sha256(pipeline.output / 'outputs/structural_power/power.json'),
                        's3_power_sha256': None, 'maximum_dataset_seal_sha256': file_sha256(data / 'datasets.json')})
    return cfg, pipeline, data, dev, freeze


def test_confirmation_verifies_full_prerequisites_then_extracts_with_synthetic_runtime(tmp_path):
    cfg, pipeline, data, dev, freeze = _confirmation_fixture(tmp_path)
    assert validate_confirmation_freeze(cfg, freeze, data, dev)['test_families'] == 32
    calls = []
    def factory(*args): calls.append(True); return TinyFeatureRuntime(*args)
    result = extract_study(cfg, pipeline.output / 'outputs/extract_confirmation', data,
                           phase='confirmation', development_cache=dev, confirmation_freeze=freeze, runtime_factory=factory)
    assert len(calls) == 1 and len(result['cache_dirs']) == 1


@pytest.mark.parametrize('mutation', ['absent_freeze', 'absent_development', 'unbound', 'wrong_config', 'wrong_power', 'wrong_size', 'missing_stage', 'corrupt_cache'])
def test_confirmation_failures_happen_before_runtime_creation(tmp_path, mutation):
    cfg, pipeline, data, dev, freeze = _confirmation_fixture(tmp_path)
    if mutation == 'absent_freeze': freeze = None
    elif mutation == 'absent_development': dev = None
    elif mutation == 'unbound': atomic_json(freeze, {'test_families': 32, 'test_orbits': 16})
    elif mutation == 'wrong_config': cfg = deepcopy(cfg); cfg['seed'] += 1
    elif mutation == 'wrong_power': atomic_json(pipeline.output / 'outputs/structural_power/power.json', {'selected_families': 64})
    elif mutation == 'wrong_size': frozen = read_json(freeze); frozen['test_families'] = 48; atomic_json(freeze, frozen)
    elif mutation == 'missing_stage': (pipeline.state_dir / 'structural_pilot.json').unlink()
    elif mutation == 'corrupt_cache':
        path = next(dev.rglob('layer_1.npy')); path.write_bytes(path.read_bytes()[:-1])
    out = pipeline.output / 'outputs/new_confirmation'
    with pytest.raises(ValueError):
        extract_study(cfg, out, data, phase='confirmation', development_cache=dev, confirmation_freeze=freeze,
                      runtime_factory=lambda *args: pytest.fail('runtime must not be created'))
    assert not out.exists()


def test_cli_rejects_unbound_freeze_without_loading_models(tmp_path, monkeypatch, capsys):
    from bsg_next.cli import main
    run = tmp_path / 'unbound'; run.mkdir()
    freeze = run / 'confirmation_freeze.json'
    atomic_json(freeze, {'test_families': 32, 'test_orbits': 16})
    monkeypatch.setattr('bsg_next.hf.runtime_from_config', lambda *args: pytest.fail('must not load models'))
    code = main(['extract', '--phase', 'confirmation', '--execute-models', '--freeze', str(freeze),
                 '--development-cache', str(run / 'outputs/extract_development'),
                 '--datasets', str(run / 'outputs/datasets'), '--output', str(run / 'new')])
    assert code == 2 and 'original run identity' in capsys.readouterr().err
    assert not (run / 'new').exists()


def test_stage_manifest_rejects_receipt_and_symlink_tampering(tmp_path):
    pipeline = StudyPipeline({'study_id': 'fixture'}, tmp_path / 'run')
    def write(out): atomic_json(out / 'result.json', {'x': 1}); return {'x': 1}
    pipeline.stage('fixture', write)
    state = read_json(pipeline.state_dir / 'fixture.json')
    manifest = pipeline.output / state['output_manifest']['path']
    manifest.write_text(manifest.read_text() + ' ')
    with pytest.raises(ValueError, match='manifest is missing or changed'):
        pipeline.stage('fixture', lambda out: pytest.fail('must not recompute'))
    other = pipeline.output / 'outputs/symlink'; other.mkdir()
    (other / 'link').symlink_to(manifest)
    with pytest.raises(ValueError, match='symlink'):
        pipeline.stage('symlink', lambda out: {'x': 1})
