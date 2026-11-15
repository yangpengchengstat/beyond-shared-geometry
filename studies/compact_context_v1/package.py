#!/usr/bin/env python3
"""Build a source-only portable bundle; no private inputs or experiment launch."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parent
FILES=['AGENTS.md','LICENSE','README.md','__init__.py','common.py','compact.py','context.py','compare.py','run.py','package.py','protocol.json','requirements.txt','run_cached.sh']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);args=p.parse_args()
    files=FILES+[str(f.relative_to(ROOT)) for f in sorted((ROOT/'tests').glob('test_*.py'))]
    if any(not (ROOT/f).is_file() for f in files): raise ValueError('Incomplete source package')
    manifest={'study_id':'icml2027-compact-context-exploratory-v1','files':{f:sha(ROOT/f) for f in files}}
    (ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    dest=out/'bsg-compact-context-v1.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(files+['source_manifest.json']):
            info=zipfile.ZipInfo('bsg-compact-context-v1/'+name,date_time=(2026,10,21,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,(ROOT/name).read_bytes())
    with zipfile.ZipFile(dest) as z:
        if z.testzip(): raise ValueError('Package CRC verification failed')
    checksum=sha(dest)
    dest.with_name(dest.name+'.sha256').write_text(checksum+'  '+dest.name+'\n')
    print(json.dumps({'archive':str(dest),'sha256':checksum,'members':len(files)+1,'bytes':dest.stat().st_size},indent=2))
if __name__=='__main__':main()
