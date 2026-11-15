"""Regression/integration checks; synthetic arrays only, never model evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'src'))
import common
import historical_worker as worker
spec=importlib.util.spec_from_file_location('rank_continuation',HERE/'continue.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
spec=importlib.util.spec_from_file_location('parent_replay_fixtures',ROOT/'studies/reproduction_v1/test_historical.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
h=common.historical


def make_abc_failure(tmp_path):
    from bsg.experiments import algebra_falsification as m
    from bsg.experiments.algebra_falsification_config import resolve_afs_config,afs_config_hash
    from bsg.experiments.canonical_units import fit_base_artifact
    from bsg.experiments.runtime_state import RunLedger,RuntimeTelemetry,experiment_key
    full=tmp_path/'full';fixtures.fixture_full(full)
    # A second synthetic FULL artifact permits real C7 and D4 postprocessing.
    second=full/'checkpoints/structural_base/d4.npz'
    np.savez_compressed(second,pca_mean=np.zeros(20),pca_components=np.eye(20)[:16],pca_explained=np.ones(16),pca_fit_rows=np.array(112),z=np.random.default_rng(44).normal(size=(160,16)),operators=np.stack([np.eye(16) for _ in range(8)]))
    full_ledger=h.read(full/'run_ledger.json')
    record=deepcopy(full_ledger['units']['structural_base:one']);record['fields']['group']='D4'
    record['outputs']=[{'path':str(second),'bytes':second.stat().st_size,'sha256':h.sha(second)}]
    full_ledger['units']['structural_base:d4']=record;h.atomic(full/'run_ledger.json',full_ledger)
    source=h.FreshCheckpointSource(full,h.source_binding(full))
    cfg=resolve_afs_config(ROOT/'configs/algebra_falsification.yaml','pre_full')
    cfg['selected_models']=[next(v for v in cfg['selected_models'] if v['key']=='llama_1b')]
    cfg.update(groups=['C7','D4'],depth_slots=[.4],split_seeds=[101],permutation_samples=3,
               scramble_seeds=cfg['scramble_seeds'][:1],bootstrap_samples=5)
    cfg['historical_source']=h.source_binding(full);cfg['replication_adapter']={'source_receipt_sha256':'original'}
    old=m.create_output_tree(tmp_path/'parent/historical/runs','afs_pre_full')
    ledger=RunLedger(old/'run_ledger.json','afs_pre_full',afs_config_hash(cfg));telemetry=RuntimeTelemetry(old/'runtime_profile.csv')
    h.atomic(old/'derived_fresh_binding_config.json',cfg)
    m._run_reused_modules(cfg,source,m.group_specs(),old,ledger,telemetry)
    group=m.group_specs()['C7'];split=m.template_split(group,cfg,101)
    hidden=np.pad(np.eye(7)[group.frame.state_id.to_numpy()],((0,0),(0,13)))
    fields={'model':cfg['selected_models'][0]['id'],'group':'C7','latent_dim':16,'split_seed':101}
    key=experiment_key('embedding_analysis',fields)
    with pytest.raises(ValueError,match='observed centered training rank 6'):
        m._run_dataframe_unit('embedding_analysis',fields,m._checkpoint(old,'embedding_analysis',key),ledger,telemetry,
                              lambda:fit_base_artifact(group,hidden,split,16,cfg))
    return old,cfg,source,ledger,hidden


@pytest.mark.parametrize('domain,q,nstates',[('C7',6,7),('D4',3,4)])
def test_exact_category_rank_guard_and_separately_named_matched_fit(domain,q,nstates):
    from bsg.experiments.afs_d import group_specs,resolve_config,template_split,embedding_rank_diagnostic_rows,assert_rank_diagnostics_pass
    from bsg.experiments.canonical_units import fit_base_artifact
    cfg=resolve_config(ROOT/'configs/afs_d.yaml');spec=group_specs()[domain];part=template_split(spec,cfg,101)
    hidden=np.pad(np.eye(nstates)[spec.frame.state_id.to_numpy()],((0,0),(0,20-nstates)))
    assert np.linalg.matrix_rank(hidden[part.train_mask]-hidden[part.train_mask].mean(axis=0))==q
    with pytest.raises(ValueError,match='requested latent_dim=16 exceeds observed centered training rank'):
        fit_base_artifact(spec,hidden,part,16,cfg)
    # The existing separately named rank-matched design remains executable.
    artifact=fit_base_artifact(spec,hidden,part,q,cfg)
    assert artifact['z'].shape==(len(spec.frame),q)
    assert cfg['group_latent_dimensions'][domain]==q


def test_real_abc_checkpoint_reuse_preserves_failed_D_and_all_parent_files(tmp_path,monkeypatch):
    from bsg.experiments import algebra_falsification as m
    from bsg.experiments.algebra_falsification_config import afs_config_hash,validate_primary_coverage
    from bsg.experiments.runtime_state import RunLedger,RuntimeTelemetry
    old,cfg,source,old_ledger,hidden=make_abc_failure(tmp_path)
    before=h.file_records(old)
    new=tmp_path/'new_afs';new.mkdir()
    changed=deepcopy(cfg);changed['replication_adapter']={'source_receipt_sha256':'recovery'}
    ledger=RunLedger(new/'run_ledger.json','afs_pre_full',afs_config_hash(changed))
    proof=worker.adopt_abc(old,new,ledger,changed)
    assert len(proof['units'])==6
    assert all(v['status']=='COMPLETE' for v in ledger.data['units'].values())
    def forbidden(*args,**kwargs):raise AssertionError('Already completed ABC was recomputed')
    for name in ('generator_composition_rows','permutation_null_rows','correspondence_scramble_rows'):
        monkeypatch.setattr(m,name,forbidden)
    m._run_reused_modules(changed,source,m.group_specs(),new,ledger,RuntimeTelemetry(new/'runtime_profile.csv'))
    for kind in worker.ABC_MODULES:validate_primary_coverage(m._collect(ledger,kind),changed)
    h.verify_ledger(new)
    worker.adopt_abc(old,new,ledger,changed)
    assert before==h.file_records(old)
    assert any(v['status']=='FAILED' for v in h.read(old/'run_ledger.json')['units'].values())


def test_reuse_rejects_changed_scientific_config_and_corrupted_checkpoint(tmp_path):
    from bsg.experiments.runtime_state import RunLedger
    old,cfg,*_=make_abc_failure(tmp_path)
    new=tmp_path/'new';new.mkdir();ledger=RunLedger(new/'run_ledger.json','afs_pre_full','fixture')
    changed=deepcopy(cfg);changed['ridge_alpha']*=2
    with pytest.raises(ValueError,match='scientific configuration'):worker.adopt_abc(old,new,ledger,changed)
    path=next((old/'checkpoints/generator_composition').glob('*.csv'));path.write_text('corrupt')
    with pytest.raises(ValueError,match='checksum'):worker.adopt_abc(old,new,ledger,cfg)


def parent_fixture(tmp_path):
    parent=tmp_path/'parent';old=parent/'historical'
    old.mkdir(parents=True)
    h.atomic(parent/'identity.json',{'source_sha256':common.source_identity(),'plan':common.runner.plan()})
    h.atomic(parent/'state.json',{'status':'technical_failure','stages':{'characterization':{'status':'complete'},'compact_context':{'status':'complete'},'historical':{'status':'technical_failure'}}})
    h.atomic(old/'source_receipt.json',{'study_id':'bsg-historical-design-reproduction-v1','source':h.source_snapshot(ROOT),
        'environment':h.environment(),'reproduction_not_new_confirmation':True})
    for stage in ('legacy_pre_full','legacy_full'):
        folder=old/'runs'/stage;fixtures.fixture_full(folder)
        manifest=h.read(folder/'manifest.json');manifest['tier']='pre_full' if stage=='legacy_pre_full' else 'full'
        manifest['replication']['source_receipt_sha256']=h.sha(old/'source_receipt.json');h.atomic(folder/'manifest.json',manifest)
        h.atomic(old/'stage_state'/(stage+'.json'),{'stage':stage,'status':'complete','source_receipt_sha256':h.sha(old/'source_receipt.json'),'outputs':h.file_records(folder)})
    failed=old/'runs/afs_pre_full';failed.mkdir(parents=True)
    message='requested latent_dim=16 exceeds observed centered training rank 6'
    h.atomic(old/'stage_state/afs_pre_full.json',{'stage':'afs_pre_full','status':'failed','error':{'type':'ValueError','message':message}})
    h.atomic(failed/'run_ledger.json',{'units':{'embedding_analysis:fixture':{'status':'FAILED','message':message,'fields':{'latent_dim':16}}}})
    h.atomic(parent/'characterization/analysis/result.json',{'fixture':True})
    return parent


def test_parent_import_verifies_full_and_keeps_original_identity(tmp_path):
    parent=parent_fixture(tmp_path);before=h.file_records(parent)
    proof=common.validate_parent(parent)
    assert proof['original_q16_D']=='INFEASIBLE_RANK_DEFICIENT_PRESERVED'
    recovery=worker.Recovery(parent,tmp_path/'recovery/historical')
    assert recovery.run_dir('legacy_full')==parent/'historical/runs/legacy_full'
    assert recovery.run_stage('legacy_full')['imported_unchanged']
    assert recovery.run_stage('legacy_pre_full')['imported_unchanged']
    assert before==h.file_records(parent)
    assert recovery.receipt_sha!=h.sha(parent/'historical/source_receipt.json')
    with pytest.raises(ValueError,match='never rerun'):recovery.legacy('full')


def test_parent_import_refuses_other_failure_or_bad_full(tmp_path):
    parent=parent_fixture(tmp_path)
    failed=parent/'historical/stage_state/afs_pre_full.json';value=h.read(failed);value['error']['message']='CUDA out of memory';h.atomic(failed,value)
    with pytest.raises(ValueError,match='restricted'):common.validate_parent(parent)
    value['error']['message']='requested latent_dim=16 exceeds observed centered training rank 6';h.atomic(failed,value)
    p=parent/'historical/runs/legacy_full/checkpoints/structural_base/cell.npz';p.write_bytes(b'bad')
    with pytest.raises(ValueError,match='changed'):common.validate_parent(parent)


def test_parent_and_new_output_cannot_overlap(tmp_path):
    for out in (tmp_path,tmp_path/'child'):
        with pytest.raises(ValueError,match='nonoverlapping'):controller.compatible_output(tmp_path,out)


def test_cached_extensions_point_to_real_parent_cache(tmp_path):
    parent=tmp_path/'parent';out=tmp_path/'new'
    for task in ('structural_transport','interface'):
        command=controller.command(task,out,parent)
        assert command[command.index('--cache-root')+1]==str(parent/'characterization/cache')
    assert 'historical_worker.py' in controller.command('historical',out,parent)[1]


def test_combined_export_preserves_failure_and_excludes_raw_and_tokens(tmp_path):
    parent=tmp_path/'parent';out=tmp_path/'new';out.mkdir()
    h.atomic(parent/'historical/stage_state/afs_pre_full.json',{'status':'failed','error':'rank6'})
    h.atomic(parent/'characterization/analysis/metric.json',{'fixture':True})
    h.atomic(out/'identity.json',{'fixture':True})
    h.atomic(out/'parent_evidence.json',{'files':controller.parent_inventory(parent)})
    for root in (parent,out):
        p=root/'historical/activation_cache/secret.npy';p.parent.mkdir(parents=True);p.write_bytes(b'EXCLUDED_RAW')
    archive=controller.collect(parent,out)
    with zipfile.ZipFile(archive) as z:
        assert any('parent-parent/historical/stage_state/afs_pre_full.json'==name for name in z.namelist())
        assert all(b'EXCLUDED_RAW' not in z.read(name) for name in z.namelist())
        manifest=json.loads(z.read('new/export_manifest.json'))
        for name,digest in manifest['files'].items():
            import hashlib
            assert hashlib.sha256(z.read(name)).hexdigest()==digest
    assert h.read(parent/'historical/stage_state/afs_pre_full.json')['status']=='failed'


def test_real_subprocess_continuation_imports_mainline_and_continues_independent_branches(tmp_path,monkeypatch):
    parent=parent_fixture(tmp_path);out=tmp_path/'new'
    h.atomic(out/'environment/hardware_preflight.json',{'status':'hardware_passed'})
    h.atomic(out/'environment/model_inventory/seal.json',{'fixture':True})
    h.atomic(out/'environment/model_inventory.json',{'models':[{'seal_file':'model_inventory/seal.json','seal_sha256':h.sha(out/'environment/model_inventory/seal.json')}]})
    fake=tmp_path/'worker.py';fake.write_text('import sys\nprint(sys.argv[1],flush=True)\nsys.exit(3 if sys.argv[1]=="pilot" else 0)\n')
    seen=[]
    def command(stage,*args):seen.append(stage);return [sys.executable,str(fake),stage]
    monkeypatch.setattr(controller,'command',command)
    before=h.file_records(parent)
    assert controller.execute(parent,out)==0
    state=h.read(out/'state.json')
    assert 'characterization' not in seen and 'compact_context' not in seen
    assert state['stages']['confirmation']['status']=='blocked_by_scientific_dependency'
    assert state['stages']['interface']['status']=='complete'
    assert state['original_q16_D']=='INFEASIBLE_RANK_DEFICIENT_PRESERVED'
    assert before==h.file_records(parent,exclude=('.orchestrator.lock','historical/historical.lock'))


def test_actual_recovery_abc_postprocess_keeps_D_infeasible_and_authorizes_by_coverage(tmp_path,monkeypatch):
    from bsg.experiments import algebra_falsification as m,algebra_falsification_config as config_module
    old,cfg,source,*_=make_abc_failure(tmp_path)
    recovery=worker.Recovery.__new__(worker.Recovery)
    recovery.project=ROOT;recovery.parent=old.parents[2];recovery.output=tmp_path/'recovery/historical'
    recovery.receipt_sha='new-technical-receipt'
    recovery.replication=lambda:{'source_receipt_sha256':'new-technical-receipt'}
    recovery.run_dir=lambda stage:source.root if stage=='legacy_full' else recovery.output/'runs'/stage
    monkeypatch.setattr(config_module,'resolve_afs_config',lambda *args:deepcopy(cfg))
    def forbidden(*args,**kwargs):raise AssertionError('Old infeasible D was invoked')
    monkeypatch.setattr(m,'_run_embedding_modules',forbidden)
    before=h.file_records(old)
    result=recovery.afs('pre_full')
    root=recovery.run_dir('afs_pre_full')
    assert result['scientific_gate']['D']=='NOT_EVALUABLE_RANK_DEFICIENT'
    assert h.read(root/'manifest.json')['execution_modules']==['A','B','C']
    authority=worker.validate_abc_pilot(root,cfg,source_receipt_sha256=recovery.receipt_sha)
    assert authority['authorized_for_afs_full'] and not authority['scientific_pass']
    assert authority['scientific_gate']['overall']=='NOT_EVALUABLE'
    # A second worker attempt reuses the same checkpoints with a JSON round-trip config.
    recovery.afs('pre_full')
    assert before==h.file_records(old)
    ledger=h.read(root/'run_ledger.json')
    first=next(iter(ledger['units']));ledger['units'][first]['status']='FAILED'
    h.atomic(root/'run_ledger.json',ledger)
    with pytest.raises(ValueError,match='Incomplete'):worker.validate_abc_pilot(root,cfg,source_receipt_sha256=recovery.receipt_sha)
