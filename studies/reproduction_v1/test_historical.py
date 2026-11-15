"""Fresh provenance adapter regressions, without model loading."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('historical_replay_tests',HERE/'historical.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

def fixture_full(root):
    from bsg.experiments.main_config import CANONICAL_REVISIONS
    root.mkdir(parents=True);checkpoint=root/'checkpoints/structural_base/cell.npz';checkpoint.parent.mkdir(parents=True)
    rng=np.random.default_rng(42)
    np.savez_compressed(checkpoint,pca_mean=np.zeros(20),pca_components=np.eye(20)[:16],pca_explained=np.ones(16),pca_fit_rows=np.array(196),z=rng.normal(size=(280,16)),operators=np.stack([np.eye(16) for _ in range(7)]))
    fields={'model_key':'llama_1b','model':'meta-llama/Llama-3.2-1B','group':'C7','depth_slot':0.4,'layer':6,'split_seed':101,'latent_dim':16}
    record={'status':'COMPLETE','fields':fields,'outputs':[{'path':str(checkpoint),'bytes':checkpoint.stat().st_size,'sha256':h.sha(checkpoint)}]}
    h.atomic(root/'run_ledger.json',{'units':{'structural_base:terminal':{'status':'COMPLETE','fields':{**fields,'depth_slot':'terminal'},'outputs':[]},'structural_base:one':record}})
    h.atomic(root/'manifest.json',{'run_id':'fresh_full','experiment':'beyond_shared_geometry_main','tier':'full','status':'SUCCESS','git_commit':'UNKNOWN','git_dirty':True,'config_hash':'fixture','replication':{'source_receipt_sha256':'fresh-source'},'models':[{'id':m,'requested_revision':r} for m,r in CANONICAL_REVISIONS.items()],'primary_latent_dimension':16,'template_split':{'train_templates':28,'validation_templates':6,'test_templates':6}})
    return checkpoint

def test_fresh_full_accepts_true_new_identity_and_skips_terminal(tmp_path):
    from bsg.experiments.afs_d import group_specs
    root=tmp_path/'full';checkpoint=fixture_full(root);binding=h.source_binding(root)
    source=h.FreshCheckpointSource(root,binding)
    m={'key':'llama_1b','id':'meta-llama/Llama-3.2-1B'}
    array,ref=source.load_artifact('structural_base',group_specs()['C7'],m,0.4,6,101)
    assert array['z'].shape==(280,16)
    assert binding['run_id']=='fresh_full' and binding['git_dirty'] is True
    assert ref['sha256']==h.sha(checkpoint)
    source.load_cell(group_specs()['C7'],model_key=m['key'],model=m['id'],depth_slot=.4,layer=6,split_seed=101,latent_dim=16)
    checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
    with pytest.raises(ValueError):h.source_binding(root)

def test_rebinding_cannot_lie_about_source(tmp_path):
    root=tmp_path/'full';fixture_full(root);binding=h.source_binding(root);binding['run_id']='main_full_003'
    with pytest.raises(ValueError,match='binding changed'):h.FreshCheckpointSource(root,binding)

def test_pre_full_gate_preserved(tmp_path):
    root=tmp_path/'pre';fixture_full(root)
    value=h.read(root/'manifest.json');value['tier']='pre_full';h.atomic(root/'manifest.json',value)
    h.atomic(root/'go_no_go.json',{'overall':'NO_GO'})
    (root/'FULL_COST_PROJECTION.md').write_text('fixture cost report')
    with pytest.raises(h.ScientificStop):h.validate_pre_full_for_full(root)
    h.atomic(root/'go_no_go.json',{'overall':'CONDITIONAL_GO'})
    assert h.validate_pre_full_for_full(root)['gate']=='CONDITIONAL_GO'

def test_complete_ledger_required(tmp_path):
    root=tmp_path/'full';fixture_full(root);ledger=h.read(root/'run_ledger.json')
    ledger['units']['structural_base:one']['status']='RUNNING';h.atomic(root/'run_ledger.json',ledger)
    with pytest.raises(ValueError,match='Incomplete'):h.source_binding(root)

def test_numerical_configs_resolve_unmodified_and_required_apis_exist():
    from bsg.experiments import main_experiment as main,algebra_falsification as afs,afs_d
    from bsg.experiments.main_config import resolve_main_config
    from bsg.experiments.algebra_falsification_config import resolve_afs_config
    assert resolve_main_config(ROOT/'configs/main_experiment.yaml','full')['dtype']=='float16'
    for tier in ('pre_full','full'):
        cfg=resolve_afs_config(ROOT/'configs/algebra_falsification.yaml',tier)
        assert cfg['primary_latent_dimension']==16
    cfg=afs_d.resolve_config(ROOT/'configs/afs_d.yaml')
    assert cfg['group_latent_dimensions']=={'C7':6,'D4':3}
    for module,names in ((main,['run','create_output_tree','_validate_calibration']), (afs,['_build_manifest','_run_reused_modules','_run_embedding_modules','_collect','_permutation_summaries','create_output_tree']), (afs_d,['run','create_output_tree','_build_manifest'])):
        for name in names:assert callable(getattr(module,name))

def test_snapshot_has_new_adapter_and_original_numerics():
    snap=h.source_snapshot(ROOT)
    assert snap['numerical_config_changes'] is False
    assert 'configs/afs_d.yaml' in snap['numerical_source']
    assert snap['adapter_sha256']==h.sha(HERE/'historical.py')

def test_real_frozen_abc_numerics_consume_new_full_binding(tmp_path):
    from bsg.experiments import algebra_falsification as m
    from bsg.experiments.algebra_falsification_config import resolve_afs_config,afs_config_hash,validate_primary_coverage
    from bsg.experiments.runtime_state import RunLedger,RuntimeTelemetry
    full=tmp_path/'full';fixture_full(full)
    source=h.FreshCheckpointSource(full,h.source_binding(full))
    cfg=resolve_afs_config(ROOT/'configs/algebra_falsification.yaml','pre_full')
    cfg['selected_models']=[next(x for x in cfg['selected_models'] if x['key']=='llama_1b')]
    cfg.update(groups=['C7'],depth_slots=[.4],split_seeds=[101],permutation_samples=3,scramble_seeds=cfg['scramble_seeds'][:1])
    out=m.create_output_tree(tmp_path,'technical_abc')
    ledger=RunLedger(out/'run_ledger.json','technical_abc',afs_config_hash(cfg));telemetry=RuntimeTelemetry(out/'runtime_profile.csv')
    m._run_reused_modules(cfg,source,m.group_specs(),out,ledger,telemetry)
    assert len(ledger.data['units'])==3
    for kind in ('generator_composition','algebra_permutation_null','correspondence_scramble'):
        frame=m._collect(ledger,kind);assert not frame.empty;validate_primary_coverage(frame,cfg)
    h.verify_ledger(out)
    before=h.sha(out/'run_ledger.json')
    m._run_reused_modules(cfg,source,m.group_specs(),out,ledger,telemetry)
    assert h.sha(out/'run_ledger.json')==before
    source.assert_unchanged()

def test_retrospective_adapter_replays_new_checkpoint_arrays(tmp_path):
    from bsg.experiments.afs_d import group_specs
    from bsg_next.offline import historical_replay,verify_historical_replay,retrospective_rows
    from bsg.experiments.canonical_units import evaluate_operators
    import pandas as pd
    root=tmp_path/'full';fixture_full(root);source=h.FreshCheckpointSource(root,h.source_binding(root))
    module=h.load_additive(ROOT);model=next(m for m in module.MODELS if m['key']=='llama_1b');spec=group_specs()['C7']
    artifact,ref=source.load_artifact('structural_base',spec,model,.4,6,101)
    part=module.template_split(spec,101)
    fields={'model':model['id'],'group':'C7','layer':6,'latent_dim':16,'split_seed':101}
    with np.load(root/ref['path'],allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    cell={'fields':fields,'split':{'seed':101,'train_ids':list(part.train_ids),'validation_ids':list(part.validation_ids),'test_ids':list(part.test_ids)},'base':arrays}
    reference=evaluate_operators(spec,artifact,part,'independent_ridge',optimization_seed=None,random_baseline_seed=None)
    for row in reference:row.update(fields)
    replay=historical_replay(cell)
    assert verify_historical_replay(replay,pd.DataFrame(reference))['status']=='matched_source_CSV'
    rows=retrospective_rows(fields,cell['split'])
    assert len(rows)==280 and len({r['family_id'] for r in rows if r['split']=='train'})==14
