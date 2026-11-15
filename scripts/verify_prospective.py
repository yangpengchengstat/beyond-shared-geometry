#!/usr/bin/env python3
"""Verify the versioned new runtime without weakening legacy source locks."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]

def prospective_files(root):
    selected=[]
    for directory in ('src/bsg_next','configs/prospective','tests/prospective'):
        selected.extend(p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.yaml','.yml'))
    selected.extend(p for p in (root/'runpod').glob('next_*') if p.is_file())
    selected.extend(p for p in (root/'studies/maintenance_v2').rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.txt','.json','.sh'))
    selected.extend(root/name for name in ('scripts/verify_prospective.py','scripts/package_runpod_next.py',
        'scripts/check_release.py','scripts/package_release.py','tests/run_prospective_checks.py',
        'pyproject.toml','.github/workflows/ci.yml') if (root/name).is_file())
    return sorted(selected)

def verify(root=ROOT):
    manifest=json.loads((root/'release/prospective_source_manifest.json').read_text())
    current={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in prospective_files(root)}
    if current!=manifest['files']:
        changed=sorted(k for k in set(current)|set(manifest['files']) if current.get(k)!=manifest['files'].get(k))
        raise ValueError('Prospective source mismatch: '+', '.join(changed))
    return {'prospective_files':len(current),'version':manifest['version'],'model_execution_during_verification':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    try:print(json.dumps(verify(args.root),indent=2))
    except (ValueError,FileNotFoundError) as error:sys.exit(str(error))
