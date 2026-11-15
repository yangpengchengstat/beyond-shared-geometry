#!/usr/bin/env python3
"""Freeze and package the portable from-zero source, never experiment results."""
from pathlib import Path
import argparse, hashlib, json, sys, zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from check_release import source_files,check_release,scan_publication_files

def build(destination):
    destination=Path(destination)
    check_release(ROOT)
    manifest=HERE/'source_manifest.json'
    candidates=[p for p in source_files(ROOT) if p!=manifest]
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates
           if p.suffix in ('.py','.sh','.json','.yaml','.yml','.txt','.in','.toml') and
           p.parts[len(ROOT.parts)] in ('src','configs','studies','scripts','release','docs','runpod')}
    manifest.write_text(json.dumps({'schema_version':1,'role':'new_reproduction_source_identity','files':files},indent=2,sort_keys=True)+'\n')
    candidates=source_files(ROOT);scan_publication_files(ROOT,candidates)
    destination.mkdir(parents=True,exist_ok=True)
    path=destination/'bsg-from-zero-v1.zip';temporary=path.with_name(path.name+'.partial')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in candidates:
            item=zipfile.ZipInfo('beyond-shared-geometry/'+str(p.relative_to(ROOT)),date_time=(2026,10,21,0,0,0))
            item.create_system=3;item.external_attr=(0o100755 if p.suffix=='.sh' else 0o100644)<<16
            archive.writestr(item,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip():raise ValueError('Package CRC failure')
    temporary.replace(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name+'.sha256').write_text(digest+'  '+path.name+'\n')
    report={'archive':path.name,'sha256':digest,'bytes':path.stat().st_size,'files':len(candidates),'source_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'GPU_validation_performed':False,'scientific_results_inferred':False,'new_experiment_result_count':0}
    (destination/'package_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'dist/2026-10-21-from-zero-v1');a=p.parse_args();print(json.dumps(build(a.output),indent=2))
