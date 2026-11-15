#!/usr/bin/env python3
"""Reconstruct a preserved source tree in a NEW directory, without running it."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]

def restore(output):
    output=Path(output)
    if output.exists():raise ValueError('Destination must not exist; nothing is overwritten')
    index=json.loads((ROOT/'release/frozen_source_restore.json').read_text())
    checked=[]
    for name,entry in index['files'].items():
        for value in (name,entry['stored_path']):
            if Path(value).is_absolute() or '..' in Path(value).parts or '\\' in value:
                raise ValueError('Unsafe restore path')
        p=ROOT/entry['stored_path']
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Preserved source mismatch: '+name)
        checked.append((name,p))
    output.mkdir(parents=True,exist_ok=False)
    for name,p in checked:
        q=output/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    return {'restored_files':len(checked),'original_source_manifest_sha256':index['original_source_manifest_sha256'],
            'experiments_executed':False,'original_known_failures_fixed':False,
            'scope':'Exact recorded numerical source and launcher inputs; old known failure behavior is deliberately preserved.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(restore(p.parse_args().output),indent=2))
