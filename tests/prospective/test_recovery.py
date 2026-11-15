"""Recovery regressions: early stopping, immutable seeds, bounded diagnosis."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bsg_next.config import load_config
from bsg_next.diagnostics import diagnose_context
from bsg_next.gates import _expected, _positive_results, CalibrationValidationError
from bsg_next.pipeline import StudyPipeline, ScientificGateError
from bsg_next.synthetic import run_synthetic

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('option',['--phase','--output','--config'])
def test_launcher_rejects_tail_overrides_before_runtime_or_billing_guards(tmp_path,option):
    import subprocess
    result=subprocess.run(['bash',str(ROOT/'runpod/next_run.sh'),'pilot',str(tmp_path),
                           '--execute-models',option,'cpu'],capture_output=True,text=True)
    assert result.returncode==2 and 'Do not override' in result.stderr


def test_positive_sentinel_keeps_original_calibration_seed_namespaces():
    cfg=load_config(ROOT/'configs/prospective/full.json')
    complete=_expected(cfg,'calibration');positive=_expected(cfg,'positive')
    assert len(complete)==14400 and len(positive)==800
    assert all(complete[key]==seed for key,seed in positive.items())
    assert {key[0] for key in positive}=={('Y4',g,n,1.) for g in ('C7','D4') for n in (0.,.1)}


def test_all_positive_failures_are_reported_without_lowering_threshold():
    records={}
    for group in ('C7','D4'):
        for noise in (0.,.1):
            successes=200 if noise==0 else 174 if group=='C7' else 165
            for rep in range(200):
                records[(('Y4',group,noise,1.),rep)]={'decisions':{'both_groups_both_endpoints_superior':rep<successes}}
    with pytest.raises(CalibrationValidationError) as error:_positive_results(records)
    assert '0.87000' in str(error.value) and '0.82500' in str(error.value)


def test_cpu_pipeline_stops_before_negative_and_offline_on_positive_failure(tmp_path):
    study=object.__new__(StudyPipeline);study.output=tmp_path;study.config={'seed':1}
    called=[]
    def stage(name,call,**kwargs):
        called.append(name)
        if name=='positive_sentinel_gate':raise ScientificGateError('test positive failure')
        return {}
    study.stage=stage
    with pytest.raises(ScientificGateError):study.run('cpu')
    assert called==['datasets','synthetic_positive','positive_sentinel_gate']


def test_gpu_pilot_cannot_start_extraction_without_complete_calibration(tmp_path):
    study=object.__new__(StudyPipeline);study.output=tmp_path;study.config={'seed':1}
    called=[]
    def stage(name,call,**kwargs):
        called.append(name)
        if name=='calibration_gate':raise ScientificGateError('missing calibration')
        return {}
    study.stage=stage
    with pytest.raises(ScientificGateError):study.run('pilot')
    assert called==['datasets','calibration_gate']


def test_cpu_does_not_repeat_retrospective_even_when_archive_was_supplied(tmp_path):
    study=StudyPipeline({'study_id':'test','seed':1,'technical_test':True},tmp_path/'run')
    called=[]
    study.stage=lambda name,call,**kwargs:called.append(name) or {}
    study.run('cpu',archive=tmp_path/'not-read.zip')
    assert 'synthetic_negative' in called and 'offline_full' not in called
    called.clear();study.run('retrospective',archive=tmp_path/'not-read.zip')
    assert called==['datasets','offline_pilot','offline_full']


def test_collection_publishes_only_completed_zip_and_does_not_overwrite_partial(tmp_path):
    import zipfile
    from bsg_next.reporting import collect_results
    run=tmp_path/'run';run.mkdir();(run/'study_config.json').write_text('{}')
    output=tmp_path/'collected.zip'
    collect_results(run,output)
    assert output.is_file() and output.with_suffix('.zip.sha256').is_file()
    assert not output.with_suffix('.zip.partial').exists()
    with zipfile.ZipFile(output) as archive:assert archive.testzip() is None
    interrupted=tmp_path/'other.zip';interrupted.with_suffix('.zip.partial').write_bytes(b'original partial')
    with pytest.raises(ValueError,match='unfinished'):collect_results(run,interrupted)
    assert interrupted.with_suffix('.zip.partial').read_bytes()==b'original partial'


def _fake_seed(config,cell,rep,seed):
    n=config['synthetic']['test_families']
    return {'cell':list(cell),'replicate':rep,'seed':seed,'decisions':{
        'both_groups_both_endpoints_superior':n>=48,
        'tests':{f'{g}.{ep}.superiority':{'reject':n>=48} for g in ('C7','D4') for ep in ('pred','ctx')}}}


def test_development_diagnosis_is_paired_and_never_promotable(tmp_path):
    output=tmp_path/'dev';cfg={'seed':1,'synthetic':{}}
    result=diagnose_context(cfg,output,replicates=2,max_seconds=10,simulator=_fake_seed)
    assert result['completed_fits']==6 and result['technical_test']
    assert not result['paper_eligible'] and not result['confirmation_authorized']
    seeds=[json.loads(p.read_text()) for p in (output/'seeds').glob('*.json')]
    assert len({r['seed'] for r in seeds})==2
    assert all(r['technical_test'] and not r['paper_eligible'] for r in seeds)
    with pytest.raises(ValueError):diagnose_context(cfg,output,simulator=_fake_seed)


def test_development_wall_budget_returns_partial_result(tmp_path):
    ticks=iter([0.,0.,2.,2.])
    result=diagnose_context({'seed':1},tmp_path/'dev',replicates=2,max_seconds=1,
                           simulator=_fake_seed,clock=lambda:next(ticks))
    assert result['completed_fits']==1 and result['status']=='budget_exhausted'
    assert not result['full_calibration_passed']


def test_revised_profile_keeps_margins_and_requires_real_structural_64_families():
    old=load_config(ROOT/'configs/prospective/full.json');new=load_config(ROOT/'configs/prospective/bf16_revision.json')
    assert new['statistics']==old['statistics']
    assert new['synthetic']['test_families']==new['data']['test_families']==64
    assert new['power']['test_families']==[64]
    assert new['runtime']['dtype']=='bfloat16'
    assert new['native']==old['native']
    with pytest.raises(ValueError,match='smaller than the calibrated'):
        load_config(ROOT/'configs/prospective/bf16_revision.json',['power.test_families=[32]'])


def test_optional_profiler_does_not_change_numeric_seed_results(tmp_path):
    cfg={'seed':1,'technical_test':True,'synthetic':{'groups':['C7'],'regimes':['Y4'],
         'timing_replicates':1,'train_families':3,'validation_families':2,'test_families':4,
         'lambda_grid':[0,.01]},'execution':{'trace_python_allocations':False}}
    plain=tmp_path/'plain';traced=tmp_path/'traced'
    a=run_synthetic(cfg,plain,mode='timing')
    cfg=deepcopy(cfg);cfg['execution']['trace_python_allocations']=True
    b=run_synthetic(cfg,traced,mode='timing')
    r=json.loads(next((plain/'seeds').glob('*.json')).read_text())
    s=json.loads(next((traced/'seeds').glob('*.json')).read_text())
    for item in (r,s):
        item.pop('seconds');item.pop('run_identity')
    assert r==s
    assert a['peak_traced_memory_bytes'] is None and b['peak_traced_memory_bytes']>0
