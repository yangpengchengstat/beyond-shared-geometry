#!/usr/bin/env python3
"""Verify immutable characterization source and separately versioned offline tools."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
PIN='5832afe9f3d31b4962ab30ba6c717f151bcb99a14b23ae25f146108a3396f739'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def safe(root,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe manifest member: '+name)
    target=root/p
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Manifest member escapes its root: '+name)
    return target

def workflow_files(root):
    study=root/'studies/characterization_v1'
    files=[p for p in study.rglob('*') if p.is_file()
           and not set(p.relative_to(study).parts)&{'frozen_run','imports','incoming','local_results','__pycache__','.pytest_cache'}
           and p.suffix in {'.py','.md','.json','.txt','.sh'}]
    files += [root/name for name in ['scripts/verify_characterization.py','scripts/build_characterization_status.py',
                                   'scripts/build_characterization_assets.py','release/characterization_review.json',
                                   'docs/CHARACTERIZATION_WORKFLOW.md','release/current_study_status.json']]
    return sorted(files)

def verify(root=ROOT):
    study=root/'studies/characterization_v1'
    frozen=study/'frozen_run'
    manifest=frozen/'source_manifest.json'
    if digest(manifest)!=PIN:raise ValueError('Frozen source manifest changed')
    hashes=json.loads(manifest.read_text())
    for name,expected in hashes.items():
        path=safe(frozen,name)
        if not path.is_file() or digest(path)!=expected:
            raise ValueError('Frozen characterization source mismatch: '+name)
    actual={str(p.relative_to(frozen)) for p in frozen.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if actual!=set(hashes)|{'source_manifest.json'}:
        raise ValueError('Unexpected or missing frozen source members')
    lock=json.loads((root/'release/characterization_source_manifest.json').read_text())
    if set(lock['workflow_files'])!={str(p.relative_to(root)) for p in workflow_files(root)}:
        raise ValueError('Unexpected or missing offline workflow files')
    for name,expected in lock['workflow_files'].items():
        path=safe(root,name)
        if not path.is_file() or digest(path)!=expected:
            raise ValueError('Characterization workflow source changed: '+name)
    status=json.loads((root/'release/current_study_status.json').read_text())
    if status['study_id']!='icml2027-characterization-v1' or status['paper_eligible'] or status['automatic_promotion']:
        raise ValueError('Current status must retain independent review boundary')
    return {'frozen_study_files':len(hashes),'workflow_files':len(lock['workflow_files']),
       'source_manifest_sha256':PIN,'model_execution':False,'results_promoted':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    print(json.dumps(verify(p.parse_args().root),indent=2))
