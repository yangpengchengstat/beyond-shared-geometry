#!/usr/bin/env python3
"""Verify portable frozen-evidence and generated-asset checksums without model access."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def contained_path(relative):
    path=ROOT/relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f'Nonportable or escaping registry path: {relative}')
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry-only',action='store_true',help='Check included evidence only, before rebuilding generated assets.')
    args=parser.parse_args()
    registry=json.loads((ROOT/'results/registry.json').read_text())
    failures=[]
    for item in registry['entries']:
        path=contained_path(item['path'])
        if not path.is_file():
            failures.append(f'Missing evidence: {item["path"]}');continue
        if sha256(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:
            failures.append(f'Evidence checksum/size mismatch: {item["path"]}')
        if item['byte_original'] and item['sha256']!=item['source_sha256']:
            failures.append(f'Byte-original/source mismatch: {item["path"]}')
    asset_count=0
    if not args.registry_only:
        manifest=ROOT/'release/artifact_manifest.json'
        if not manifest.is_file():
            failures.append('Missing release/artifact_manifest.json; rebuild paper assets.')
        else:
            items=json.loads(manifest.read_text())['artifacts']
            names=[i['path'] for i in items]
            if len(names)!=len(set(names)):failures.append('Duplicate release artifact paths.')
            for item in items:
                path=contained_path(item['path']);asset_count+=1
                if not path.is_file() or sha256(path)!=item['sha256']:
                    failures.append(f'Release artifact checksum mismatch: {item["path"]}')
    if failures:
        print('\n'.join(failures))
        raise SystemExit(1)
    print(f'PASS: {len(registry["entries"])} included evidence files and {asset_count} release artifact checks.')
    print('External archives are identified by historical hashes only; this check does not claim that omitted raw data are publicly available or locally present.')

if __name__=='__main__':main()
