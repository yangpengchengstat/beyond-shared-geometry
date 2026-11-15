#!/usr/bin/env python3
"""Independently scheduled original/prospective recipes under a new replay identity."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, importlib.metadata, sys, time
from pathlib import Path
from receipt import ROOT,sha,read,write,identity,inventory,verify_inventory,lock
sys.path.insert(0,str(ROOT/'src'))
from bsg_next.config import load_config
from bsg_next.io import source_identity as numeric_source_identity,jsonable
from bsg_next.pipeline import StudyPipeline,ScientificGateError

TASKS=('original_calibration','revised_calibration','pilot','confirmation','native_prepare','native_confirmation','peer_anchors','peer_factorial','word_extract','word_analyze','structural_transport','product','interface')

def configuration(original=False):
    filename='full.json' if original else 'bf16_revision.json'
    cfg=load_config(ROOT/'configs/prospective'/filename)
    cfg['study_id']='icml2027-reproduction-v1-'+('original-recipe' if original else 'revised-recipe')
    cfg['replication']={'role':'replay_of_previously_inspected_design','independent_confirmation':False,
        'parent_config_sha256':sha(ROOT/'configs/prospective'/filename),
        'numerics':'current preserved bsg_next 0.3.2; historical numerical bugs are not emulated'}
    cfg['runtime']['local_files_only']=True
    cfg['runtime_source_sha256']=numeric_source_identity(ROOT/'src/bsg_next')
    cfg['protocol_dependency_versions']={p:importlib.metadata.version(p).split('+',1)[0] for p in ('numpy','scipy','torch','transformers','tokenizers','huggingface-hub')}
    cfg['protocol_asset_sha256']={}
    for key,value in [('peer_anchors',cfg['peers']['original_model_config']),('word_bank',cfg['crossmodel']['word_file']),('historical_cell_index',cfg['offline']['index'])]:
        cfg['protocol_asset_sha256'][key]=sha(ROOT/value)
    return cfg

def run_task(task,out,cache_root=None):
    if task not in TASKS:raise ValueError('Unknown task')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=configuration(task=='original_calibration')
    # Freeze model execution permission before any package identity is generated.
    cfg['execution']['allowed_model_execution']=True
    cfg_sha=identity(cfg)
    receipt=out/'receipts'/(task+'.json')
    if receipt.exists():
        previous=read(receipt)
        if previous['config_identity']!=cfg_sha:raise ValueError('Prospective replay identity changed')
        verify_inventory(out,previous['files'])
        print('RESUME_VERIFIED '+task,flush=True)
        return 3 if previous['status']=='scientific_gate_stopped' else 0
    root=out/('original_recipe' if task=='original_calibration' else 'revised_recipe')
    tick=time.monotonic();status='complete';message=None
    touched=[]
    try:
        if task in ('original_calibration','revised_calibration','pilot','confirmation'):
            pipeline=StudyPipeline(cfg,root)
            phase={'original_calibration':'cpu','revised_calibration':'cpu','pilot':'pilot','confirmation':'confirm'}[task]
            before={p.stem:sha(p) for p in pipeline.state_dir.glob('*.json')}
            try:result=pipeline.run(phase)
            finally:
                # Bind actual stage outputs, not timestamps or a later global state file.
                for state in pipeline.state_dir.glob('*.json'):
                    if before.get(state.stem)!=sha(state):
                        entry=read(state)
                        if entry['status']=='complete':
                            for path in (root/'outputs'/state.stem).rglob('*'):
                                if path.is_file() and not path.name.startswith('.'):touched.append(path)
            # On re-entering after an interrupted session include all phase dependencies.
            for state in pipeline.state_dir.glob('*.json'):
                if read(state)['status']=='complete':
                    for p in (root/'outputs'/state.stem).rglob('*'):
                        if p.is_file() and not p.name.startswith('.'):touched.append(p)
        elif task in ('native_prepare','native_confirmation'):
            from bsg_next.native import run_native
            target=out/'native_independent'
            result=run_native(cfg,target,mode='prepare' if task=='native_prepare' else 'confirm')
            if result.get('status') in ('clean_competence_failed','power_gate_not_met_no_test_outcomes'):
                status='scientific_gate_stopped';message=result['status']
            touched.extend(p for p in target.rglob('*') if p.is_file() and not p.name.startswith('.'))
        elif task=='peer_anchors':
            # This method loops every registered original-model anchor.
            target=out/task;result=StudyPipeline(cfg,out/'anchor_controller')._peer_anchors(target)
            touched.extend(p for p in target.rglob('*') if p.is_file() and not p.name.startswith('.'))
        else:
            target=out/task
            if task=='peer_factorial':
                from bsg_next.peers import run_factorial_stage
                result=run_factorial_stage(cfg,target)
            elif task=='word_extract':
                from bsg_next.wordfields import collect_word_fields
                result=collect_word_fields(cfg,target)
            elif task=='word_analyze':
                from bsg_next.wordfields import analyze_word_fields
                result=analyze_word_fields(cfg,target,out/'word_extract')
            elif task=='structural_transport':
                from bsg_next.wordfields import analyze_structural_transport
                if not cache_root:raise ValueError('Fresh characterization cache root required')
                result=analyze_structural_transport(cfg,target,cache_root=cache_root)
            elif task=='product':
                from bsg_next.extensions import run_product
                result=run_product(cfg,target)
            elif task=='interface':
                from bsg_next.extensions import run_interface_training
                if not cache_root:raise ValueError('Fresh characterization cache root required')
                result=run_interface_training(cfg,target,cache_root=cache_root)
            touched.extend(p for p in target.rglob('*') if p.is_file() and not p.name.startswith('.'))
    except ScientificGateError as exc:
        status='scientific_gate_stopped';message=str(exc);result={'gate_reason':message}
    if status=='scientific_gate_stopped' and root.exists():
        # Include failure receipts and raw scientific outputs for diagnosis.
        touched.extend(p for p in (root/'stage_state').glob('*.json'))
        for p in (root/'outputs').rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl') and not p.name.startswith('.'):touched.append(p)
    # Shared recipe stage-state changes on subsequent phases; those states are
    # retained in the final bundle but excluded from phase receipt identities.
    files={str(p.relative_to(out)):sha(p) for p in set(touched) if p.is_file() and 'stage_state' not in p.parts and not p.name.endswith('.partial')}
    value={'task':task,'status':status,'config_identity':cfg_sha,'result':jsonable(result),'message':message,
        'wall_seconds':time.monotonic()-tick,'files':files,'independent_confirmation':False,'paper_eligible':False}
    write(receipt,value)
    print('TASK_RESULT '+task+' '+status,flush=True)
    return 3 if status=='scientific_gate_stopped' else 0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-root',type=Path);p.add_argument('--execute-models',action='store_true');a=p.parse_args()
    if a.task not in ('original_calibration','revised_calibration','word_analyze','structural_transport') and not a.execute_models:p.error('--execute-models required')
    with lock(a.output/'.controller.lock'):return run_task(a.task,a.output,a.cache_root)
if __name__=='__main__':raise SystemExit(main())
