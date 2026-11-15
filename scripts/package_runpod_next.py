#!/usr/bin/env python3
"""Prepare a private RunPod upload bundle; never provision a pod or download a model."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

from check_release import check_release,source_files

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def package(root,output,archive=None):
    root=root.resolve();output=output.resolve()
    check_release(root)
    if output.exists():raise ValueError('Choose a new output: a RunPod bundle already exists')
    if archive:
        archive=archive.resolve()
        if archive==output:raise ValueError('Input and output archive cannot be the same')
        expected=json.loads((root/'configs/prospective/full.json').read_text())['offline']['expected_archive_sha256']
        if digest(archive)!=expected:raise ValueError('Historical archive SHA-256 mismatch')
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                path=Path(info.filename)
                if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe historical archive member')
    files=source_files(root)
    manifest={'version':'0.3.1','scope':'private RunPod upload; do not publish this bundle',
        'source_files':{str(p.relative_to(root)):digest(p) for p in files},
        'historical_archive':{'path':'inputs/full.zip','sha256':digest(archive),'bytes':archive.stat().st_size} if archive else None,
        'model_weights_included':False,'research_models_executed':False,
        'start':'beyond-shared-geometry/docs/RECOVERY_RUNPOD.md'}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.partial')
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as result:
            for path in files:result.write(path,'beyond-shared-geometry/'+str(path.relative_to(root)))
            if archive:result.write(archive,'beyond-shared-geometry/inputs/full.zip',compress_type=zipfile.ZIP_STORED)
            result.writestr('RUNPOD_BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        temporary.replace(output)
    finally:
        if temporary.exists():temporary.unlink()
    checksum=digest(output);output.with_suffix(output.suffix+'.sha256').write_text(checksum+'  '+output.name+'\n')
    return {'archive':output.name,'sha256':checksum,'bytes':output.stat().st_size,'historical_input_included':bool(archive),'private':True}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path);args=p.parse_args()
    print(json.dumps(package(args.root,args.output,args.archive),indent=2))
