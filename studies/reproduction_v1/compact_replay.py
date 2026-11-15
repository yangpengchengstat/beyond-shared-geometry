#!/usr/bin/env python3
"""Replay frozen compact/context estimators on fresh, verified q16 coordinates."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from receipt import ROOT,sha,read,write,identity,lock,verify_inventory
COMPACT=ROOT/'studies/compact_context_v1'
sys.path.insert(0,str(COMPACT))
from common import make_cell
from compact import evaluate_compact
from context import evaluate_context
from compare import make_comparison

def dependency_identity():
    manifest=read(COMPACT/'source_manifest.json')
    for name,expected in manifest['files'].items():
        if sha(COMPACT/name)!=expected:raise ValueError('Compact dependency changed: '+name)
    return sha(COMPACT/'source_manifest.json')

def read_cell(directory,name,registration):
    folder=directory/'analysis'/name
    rows_path=folder/'rows.json'
    if not rows_path.exists():
        reference=Path(read(folder/'coordinate_export.json')['rows_and_potential_archive_relative_directory'])
        if reference.is_absolute() or '..' in reference.parts:raise ValueError('Unsafe metadata path')
        rows_path=directory/reference/'rows.json'
    rows=read(rows_path)['rows']
    with np.load(folder/'coordinates.npz',allow_pickle=False) as archive:z=archive['z']
    if z.dtype!=np.float64 or z.shape!=(len(rows),16) or not np.isfinite(z).all():raise ValueError('Invalid q16 coordinates')
    return make_cell(z,rows,rows[0]['domain'],registration)

def run(parent,output):
    parent=Path(parent);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config=read(COMPACT/'protocol.json');dep=dependency_identity()
    names=sorted(p.parent.name for p in (parent/'analysis').glob('*/coordinates.npz'))
    # A fresh numerical run has its own input hashes; never spoof the old archive hash.
    spec=read(ROOT/'studies/characterization_v1/frozen_run/study_config.json')
    expected=sorted(m['id'].replace('/','__')+'__'+g+'__'+str(layer) for m in spec['models'] for g in ('C7','D4','S3') for layer in m['layers'])
    if names!=expected:raise ValueError('Fresh characterization must contain all 72 registered cells')
    inputs={}
    for name in names:
        folder=parent/'analysis'/name
        for filename in ('coordinates.npz','rows.json','coordinate_export.json','pca.npz','cell_complete.json'):
            if (folder/filename).exists():inputs[str((folder/filename).relative_to(parent))]=sha(folder/filename)
        if not (folder/'cell_complete.json').is_file():raise ValueError('Uncommitted characterization cell '+name)
        if not (folder/'rows.json').exists():
            shared=read(folder/'coordinate_export.json')['rows_and_potential_archive_relative_directory']+'/rows.json'
            inputs[shared]=sha(parent/shared)
    contract = read(parent/'study_contract.json')
    contract_sha = identity(contract)
    for name in names:
        folder = parent/'analysis'/name
        receipt = read(folder/'cell_complete.json')
        if receipt['binding']['contract_id'] != contract_sha or not receipt.get('artifacts'):
            raise ValueError('Characterization cell belongs to another contract')
        verify_inventory(folder, receipt['artifacts'])
        if not (folder/'rows.json').exists():
            shared = parent/read(folder/'coordinate_export.json')['rows_and_potential_archive_relative_directory']
            shared_receipt = read(shared/'shared_complete.json')
            if shared_receipt['binding']['contract_id'] != contract_sha:
                raise ValueError('Shared metadata contract differs')
            verify_inventory(shared, shared_receipt['artifacts'])
    # Bind a new input identity, without changing the original compact protocol.
    run_identity={'role':'descriptive_replication_of_inspected_rows','dependency_sha256':dep,
        'input_files':inputs,'numpy':np.__version__,'config_sha256':sha(COMPACT/'protocol.json')}
    run_sha=identity(run_identity);jobs=[]
    for name in names:
        cell=read_cell(parent,name,'semantic')
        for registration in config['registrations']:
            for component,domains in config['components'].items():
                if cell.domain in domains:jobs.append({'cell':name,'domain':cell.domain,'registration':registration,'component':component})
    if len(jobs)!=240:raise ValueError('Expected 240 complete-panel comparisons')
    with lock(output/'.lock'):
        if (output/'identity.json').exists():
            if read(output/'identity.json')!=run_identity:raise ValueError('Replay source/input identity changed')
        elif any(p.name!='.lock' for p in output.iterdir()):raise ValueError('Unknown nonempty compact output')
        else:write(output/'identity.json',run_identity)
        (output/'checkpoints').mkdir(exist_ok=True)
        completed={};start=time.monotonic()
        for job in jobs:
            key=job['cell']+'--'+job['registration']+'--'+job['component'];path=output/'checkpoints'/(key+'.json')
            if path.exists():
                result=read(path);payload=result.pop('payload_sha256',None)
                if payload!=identity(result) or result['identity_sha256']!=run_sha or result['job']!=job:raise ValueError('Corrupt compact checkpoint')
            else:
                print('START_COMPARISON '+key,flush=True);tick=time.monotonic()
                cell=read_cell(parent,job['cell'],job['registration'])
                result={'job':job,'result':(evaluate_compact if job['component']=='compact' else evaluate_context)(cell,config,list(cell.test_splits)),
                    'identity_sha256':run_sha,'evaluation_splits':list(cell.test_splits),'elapsed_seconds':time.monotonic()-tick,
                    'scientific_role':'descriptive_replication_of_inspected_rows','paper_eligible':False,'confirmation_authorized':False}
                result['payload_sha256']=identity(result);write(path,result)
            completed[key]={'sha256':sha(path),'elapsed_seconds':result['elapsed_seconds']}
            write(output/'state.json',{'status':'running','completed_jobs':len(completed),'planned_jobs':len(jobs),'checkpoints':completed})
            print(f'COMPARISONS {len(completed)}/{len(jobs)}',flush=True)
        verify_inventory(parent,inputs)
        write(output/'validation_comparability.json',make_comparison(output,config['validation_comparability_factor']))
        write(output/'state.json',{'status':'complete','completed_jobs':240,'planned_jobs':240,'checkpoints':completed,'wall_seconds':time.monotonic()-start,
            'paper_eligible':False,'confirmation_authorized':False,'parent_identity_sha256':run_sha})
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();raise SystemExit(run(args.parent,args.output))
