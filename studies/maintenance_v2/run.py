#!/usr/bin/env python3
"""Opt-in selective maintenance-v2 runner; default is an offline plan/check."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def verify_source():
    spec=importlib.util.spec_from_file_location('maintenance_verify',ROOT/'scripts/verify_prospective.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.verify(ROOT)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('interface','peer_factorial'),required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/prospective/bf16_revision.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--cache-root',type=Path)
    parser.add_argument('--execute',action='store_true',help='Actually run the selected task; never implied by checking')
    parser.add_argument('--execute-models',action='store_true',help='Explicit model/GPU execution authorization')
    args=parser.parse_args(argv)
    verification=verify_source()
    plan={'release':'maintenance-v2','task':args.task,'source_verification':verification,
          'config':str(args.config),'output':str(args.output) if args.output else None,
          'execution_requested':args.execute,'calibration_scheduled':False,'whole_study_scheduled':False,
          'old_outputs_compatible':False,'old_source_capsules_unchanged':True,
          'scientific_gate_waiver':False,'paper_eligible':False}
    print(json.dumps(plan,indent=2),flush=True)
    if not args.execute:return 0
    if not args.execute_models:parser.error('--execute-models is required for an actual run')
    if args.output is None:parser.error('--output is required for an actual run')
    if args.task=='interface' and args.cache_root is None:
        parser.error('--cache-root is required: this task never extracts missing activations')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    from bsg_next.config import load_config
    from bsg_next.io import file_sha256,source_identity,atomic_json
    cfg=load_config(args.config)
    cfg['maintenance_release']={'version':2,'parent_config_sha256':file_sha256(args.config),
                                'runtime_source_sha256':source_identity(ROOT/'src/bsg_next'),
                                'replay_role':'previously_inspected_design','independent_confirmation':False}
    cfg.setdefault('runtime',{})['local_files_only']=True
    cfg.setdefault('execution',{})['allowed_model_execution']=True
    if args.task=='interface':
        from bsg_next.extensions import run_interface_training
        result=run_interface_training(cfg,args.output,cache_root=args.cache_root)
    else:
        from bsg_next.peers import run_factorial_stage
        result=run_factorial_stage(cfg,args.output,cache_root=args.cache_root)
    atomic_json(args.output/'maintenance_completion.json',{'task':args.task,'status':'complete',
        'paper_eligible':False,'independent_confirmation':False,'scientific_gate_waiver':False})
    print('MAINTENANCE_TASK_COMPLETE; no next task or calibration is scheduled.',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
