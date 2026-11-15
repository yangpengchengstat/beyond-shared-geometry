"""Scientific stage contracts with injected backends, without model downloads."""
import json
from pathlib import Path
import pytest
from bsg_next.pipeline import (StudyPipeline,ScientificGateError,seal_study,
                               extract_study,run_synthetic_workers)


def test_native_gate_preserves_failure_evidence(tmp_path,monkeypatch):
    import bsg_next.native as native
    def fail(config,output,mode):
        Path(output).mkdir(parents=True,exist_ok=True)
        (Path(output)/'raw_failure.json').write_text('{"status":"clean_competence_failed"}')
        return {'status':'clean_competence_failed'}
    monkeypatch.setattr(native,'run_native',fail)
    pipeline=StudyPipeline({'study_id':'test'},tmp_path/'run')
    with pytest.raises(ScientificGateError,match='clean_competence_failed'):
        pipeline.stage('native_prepare',lambda out:pipeline._native(out,'prepare'))
    assert (tmp_path/'run/outputs/native_study/raw_failure.json').exists()
    state=json.loads((tmp_path/'run/stage_state/native_prepare.json').read_text())
    assert state['status']=='gate_stopped'


def test_peer_anchors_use_their_pinned_original_model(tmp_path,monkeypatch):
    import bsg_next.peers as peers
    calls=[]
    monkeypatch.setattr(peers,'run_reproduction_anchor',lambda config,output:calls.append(config['peers']) or {'status':'tested'})
    source=tmp_path/'anchors.json'
    source.write_text(json.dumps({'anchors':[{'name':'original','anchor':'engels','spec':{'id':'original/model','revision':'a'*40},'layer':5}]}))
    config={'study_id':'test','models':[{'id':'study/model','revision':'b'*40}],
            'peers':{'original_model_config':str(source)}}
    result=StudyPipeline(config,tmp_path/'run')._peer_anchors(tmp_path/'anchors')
    assert calls[0]['model']['id']=='original/model' and calls[0]['layers']==[5]
    assert result['study_model_fallback'] is False


def test_maximum_orbit_pool_is_sealed_before_pilot(tmp_path,monkeypatch):
    import bsg_next.data as data
    seen=[]
    monkeypatch.setattr(data,'build_study_records',lambda domain,**kwargs:seen.append(kwargs) or [{'domain':'S3'}])
    monkeypatch.setattr(data,'build_support_records',lambda domain,**kwargs:[])
    def seal(rows,path,metadata):
        Path(path).mkdir(parents=True);(Path(path)/'manifest.json').write_text('{}')
    monkeypatch.setattr(data,'seal_dataset',seal)
    seal_study({'seed':1,'technical_test':True,'data':{'groups':['S3'],'max_test_orbits':24,'max_test_families':64}},tmp_path/'seal')
    assert seen[0]['max_test_orbits']==24 and seen[0]['max_test_families']==64


def test_extraction_forwards_both_power_selected_cluster_counts(tmp_path,monkeypatch):
    import bsg_next.data as data
    import bsg_next.extraction as extraction
    dataset=tmp_path/'dataset';domain=dataset/'S3';domain.mkdir(parents=True)
    (domain/'manifest.json').write_text('{}');(dataset/'datasets.json').write_text(json.dumps({'datasets':{'S3':str(domain)}}))
    observed=[]
    monkeypatch.setattr(data,'load_sealed_dataset',lambda path:[{'row_id':'one'}])
    monkeypatch.setattr(data,'select_phase_records',lambda rows,phase,f,o:observed.append((phase,f,o)) or rows)
    monkeypatch.setattr(extraction,'extract_features',lambda *args,**kwargs:None)
    class Runtime:
        def close(self):pass
    config={'models':[{'id':'fixture','layers':[1]}]}
    result=extract_study(config,tmp_path/'extract',dataset,phase='development',test_families=48,test_orbits=24,
                         runtime_factory=lambda *args:Runtime())
    assert observed==[('development',48,24)] and result['test_orbits']==24


def test_low_context_energy_is_a_preserved_scientific_gate(tmp_path):
    pipeline=StudyPipeline({'study_id':'test','data':{'groups':['C7','D4']}},tmp_path/'run')
    pilot={'family_contrasts':{'C7':{'eligible':False},'D4_corner':{'eligible':True}},
           'ineligible_cells':[{'group':'C7','context_energy_ratio':.01}]}
    with pytest.raises(ScientificGateError,match='insufficient context energy'):
        pipeline.stage('pilot_context_eligibility',lambda out:pipeline._pilot_eligibility(pilot,out))
    report=json.loads((tmp_path/'run/outputs/pilot_context_eligibility/eligibility.json').read_text())
    assert report['status']=='ineligible_context_energy' and report['not_a_software_failure']


@pytest.mark.parametrize('mode',['calibration','crossed','crossed_bootstrap'])
def test_cpu_two_worker_shards_match_serial_and_resume_without_refit(tmp_path,mode):
    import numpy as np
    from bsg_next.synthetic import run_synthetic
    config={'technical_test':True,'execution':{'cpu_workers':2},'synthetic':{
        'groups':['C7'],'regimes':['Y4'],'noise_ratios':[.1],'conditions':[1.],
        'train_families':3,'validation_families':2,'test_families':3,
        'lambda_grid':[0,.01],'replicates_per_cell':2,
        'crossed_families':[4],'crossed_orbits':[3],'crossed_variances':[[1,1,1]],
        'crossed_correlations':[0.],'crossed_heteroscedastic':[False],
        'crossed_replicates':2,'crossed_bootstrap_datasets':1,'bootstrap_replicates':9}}
    serial=tmp_path/'serial';parallel=tmp_path/'parallel'
    expected=run_synthetic(config,serial,mode=mode)
    actual=run_synthetic_workers(config,parallel,mode=mode)
    before={str(p.relative_to(parallel)):p.read_bytes() for p in (parallel/'seeds').rglob('*.json')}
    assert before
    for name,content in before.items():
        left=json.loads((serial/name).read_bytes());right=json.loads(content)
        assert left['seed']==right['seed'] and left['run_identity']==right['run_identity']
        if mode=='calibration':
            assert left['decisions']==right['decisions']
            for group in left['family_contrasts']:
                for endpoint in ('pred','ctx'):
                    assert np.allclose(left['family_contrasts'][group][endpoint],right['family_contrasts'][group][endpoint])
        else:
            assert left==right
    assert len(before)==len(list((serial/'seeds').rglob('*.json')))
    if mode=='crossed':
        assert actual['scenarios']==expected['scenarios']
    else:
        assert actual['completed_datasets']==expected['completed_datasets']
    run_synthetic_workers(config,parallel,mode=mode)
    assert before=={str(p.relative_to(parallel)):p.read_bytes() for p in (parallel/'seeds').rglob('*.json')}


def test_gate_stage_can_revalidate_completed_evidence(tmp_path):
    pipeline=StudyPipeline({'study_id':'test'},tmp_path/'run')
    calls=[]
    def initial_gate(out):
        from bsg_next.io import atomic_json
        calls.append('validated')
        result = {'pass': True}
        atomic_json(out/'calibration_gate.json', result)
        return result
    pipeline.stage('calibration_gate', initial_gate)
    def now_incomplete(out):
        calls.append('revalidated')
        raise ScientificGateError('calibration seed evidence is now incomplete')
    with pytest.raises(ScientificGateError,match='now incomplete'):
        pipeline.stage('calibration_gate',now_incomplete,resume=False)
    assert calls==['validated','revalidated']
