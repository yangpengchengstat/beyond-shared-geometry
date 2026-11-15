"""Fault-injection regression checks; artificial features only, no model/Torch load."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bsg_next import extensions, extraction, structural
from bsg_next.io import atomic_json, read_json, prepare_run
from bsg_next.runtime_records import load_record, save_record, recorded_session
from bsg_next.peers import run_factorial_stage
from test_native_peers import SemanticRuntime, mock_config


def interface_fixture(tmp_path, monkeypatch):
    cache=tmp_path/'cache';cache.mkdir()
    atomic_json(cache/'manifest.json',{'arrays':{'1':{}},'cache_key':'test-only'})
    rows=[{'domain':'C7','split':s} for s in ('train','validation','test')]
    monkeypatch.setattr(structural,'_collect_paths',lambda *a:[cache])
    monkeypatch.setattr(extraction,'read_cache',lambda p:(rows,{1:np.ones((3,4))},read_json(cache/'manifest.json')))
    calls=[]
    def train(h,rows,group,q,arm,weight,seed,**kwargs):
        calls.append((arm,weight,seed))
        return {'validation_selection_score':weight,'_parameters':{
            'encoder':np.zeros((4,q)),'decoder':np.zeros((q,4)),
            'head_weight':np.zeros((7,q)),'head_bias':np.zeros(7),
            'operators':np.zeros((7,q,q)),'center':np.zeros(4)}}
    monkeypatch.setattr(extensions,'_train_interface',train)
    config={'interface':{'q':2,'seeds':[101],'weights':[.01,.1]}}
    return config,calls,train


def test_interface_fresh_directory_verified_resume_and_corruption(tmp_path,monkeypatch):
    config,calls,_=interface_fixture(tmp_path,monkeypatch)
    output=tmp_path/'output'
    result=extensions.run_interface_training(config,output)
    assert len(calls)==6 and len(result['selected'])==4
    assert len(list((output/'fits').glob('*.npz')))==6
    assert read_json(output/'progress.json')['completed']==6
    before={p.name:p.read_bytes() for p in (output/'fits').iterdir()}
    assert extensions.run_interface_training(config,output)==result
    assert len(calls)==6
    assert read_json(output/'progress.json')['resumed']==6
    assert len(list((output/'sessions').glob('*.json')))==2
    assert before=={p.name:p.read_bytes() for p in (output/'fits').iterdir()}
    target=next((output/'fits').glob('*.npz'));target.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checkpoint identity mismatch'):
        extensions.run_interface_training(config,output)
    assert read_json(output/'progress.json')['status']=='interrupted_or_failed'


def test_interface_interruption_resumes_completed_fits(tmp_path,monkeypatch):
    config,calls,train=interface_fixture(tmp_path,monkeypatch)
    def interrupt(*a,**k):
        if len(calls)==2:raise KeyboardInterrupt('injected interruption')
        return train(*a,**k)
    monkeypatch.setattr(extensions,'_train_interface',interrupt)
    with pytest.raises(KeyboardInterrupt):extensions.run_interface_training(config,tmp_path/'out')
    assert len(list((tmp_path/'out'/'fits').glob('*.json')))==2
    monkeypatch.setattr(extensions,'_train_interface',train)
    extensions.run_interface_training(config,tmp_path/'out')
    assert len(calls)==6
    assert read_json(tmp_path/'out'/'progress.json')['resumed']==2


def test_interface_rejects_nonfinite_fit_before_checkpoint(tmp_path,monkeypatch):
    config,_,train=interface_fixture(tmp_path,monkeypatch)
    def invalid(*a,**k):
        value=train(*a,**k);value['_parameters']['encoder'][0,0]=np.nan;return value
    monkeypatch.setattr(extensions,'_train_interface',invalid)
    with pytest.raises(ValueError,match='invalid parameter'):
        extensions.run_interface_training(config,tmp_path/'out')
    assert not list((tmp_path/'out'/'fits').glob('*.json'))


def test_interface_rejects_old_or_foreign_output(tmp_path,monkeypatch):
    config,_,_=interface_fixture(tmp_path,monkeypatch)
    output=tmp_path/'old';prepare_run(output,config,'constructive_interface')
    before=(output/'run_identity.json').read_bytes()
    with pytest.raises(ValueError,match='different configuration'):
        extensions.run_interface_training(config,output)
    assert (output/'run_identity.json').read_bytes()==before
    assert not (output/'fits').exists()


def small_factorial():
    cfg=mock_config(domains=['C7'],families={'train':1,'validation':1,'test':1,'pilot':1})
    cfg['native']['models']=['mock-a'];return cfg


def test_factorial_atomic_resume_matches_uninterrupted(tmp_path):
    cfg=small_factorial();calls=[];instances=[]
    class CountingRuntime(SemanticRuntime):
        fail_at=None
        def score(self,*args,**kwargs):
            if self.fail_at is not None and len(calls)==self.fail_at:
                raise RuntimeError('injected interruption')
            calls.append(1);return super().score(*args,**kwargs)
    def factory(model,runtime):
        value=CountingRuntime(model,runtime);instances.append(value);return value
    CountingRuntime.fail_at=10
    out=tmp_path/'interrupted'
    with pytest.raises(RuntimeError,match='injected'):
        run_factorial_stage(cfg,out,runtime_factory=factory)
    checkpoints=list(out.glob('*/recipient_checkpoints/validation/*.json'))
    assert len(checkpoints)==1  # Eight scores committed; two partial scores discarded.
    assert all(v.closed for v in instances)
    first=checkpoints[0].read_bytes()
    CountingRuntime.fail_at=None
    result=run_factorial_stage(cfg,out,runtime_factory=factory)
    progress=read_json(out/'progress.json')
    assert progress['resumed']==1 and progress['completed']==progress['total']
    assert checkpoints[0].read_bytes()==first
    total=sum(v['evaluations'] for v in result['cells']['m0.C7']['splits'].values())
    assert len(calls)==total+2
    expected=run_factorial_stage(cfg,tmp_path/'clean',runtime_factory=factory)
    assert expected==result
    for split in ('validation','test'):
        assert (out/'m0.C7'/f'{split}_evaluations.jsonl').read_bytes()==(tmp_path/'clean'/'m0.C7'/f'{split}_evaluations.jsonl').read_bytes()
    old_calls=len(calls)
    assert run_factorial_stage(cfg,out,runtime_factory=factory)==result
    assert len(calls)==old_calls
    payload=json.loads(checkpoints[0].read_text());payload['payload']['rows'][0]['correct']=1.234
    atomic_json(checkpoints[0],payload)
    with pytest.raises(ValueError,match='checksum/schema'):
        run_factorial_stage(cfg,out,runtime_factory=factory)
    assert len(calls)==old_calls


def test_factorial_rejects_old_identity_before_loading_runtime(tmp_path):
    from bsg_next.native import normalized_config
    cfg=small_factorial();out=tmp_path/'legacy'
    prepare_run(out,normalized_config(cfg),'peer_factorial')
    def forbidden(*args):raise AssertionError('must not load model')
    with pytest.raises(ValueError,match='different configuration'):
        run_factorial_stage(cfg,out,runtime_factory=forbidden)
    with pytest.raises(PermissionError,match='explicit authorization'):
        run_factorial_stage(cfg,tmp_path/'no-permission')


def test_checkpoint_record_detects_mutation_and_session_lock(tmp_path):
    save_record(tmp_path/'checkpoint.json',{'result':42})
    assert load_record(tmp_path/'checkpoint.json')=={'result':42}
    with recorded_session(tmp_path,'fixture',1) as session:
        with pytest.raises(RuntimeError,match='Another process'):
            with recorded_session(tmp_path,'fixture',1):pass
        session.advance()
    assert read_json(tmp_path/'progress.json')['status']=='complete'


def test_incomplete_session_cannot_claim_completion(tmp_path):
    with pytest.raises(ValueError,match='all planned units'):
        with recorded_session(tmp_path,'fixture',2) as session:
            session.advance()
    assert read_json(tmp_path/'progress.json')['status']=='interrupted_or_failed'


def test_maintenance_launcher_defaults_to_offline_plan(monkeypatch,capsys,tmp_path):
    import importlib.util
    path=Path(__file__).resolve().parents[2]/'studies/maintenance_v2/run.py'
    spec=importlib.util.spec_from_file_location('maintenance_plan_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'verify_source',lambda:{'test_fixture':True})
    output=tmp_path/'never-created'
    assert module.main(['--task','interface','--output',str(output)])==0
    assert not output.exists()
    plan=json.loads(capsys.readouterr().out)
    assert plan['calibration_scheduled'] is False and plan['execution_requested'] is False
    with pytest.raises(SystemExit):
        module.main(['--task','interface','--output',str(output),'--execute'])
    assert not output.exists()
